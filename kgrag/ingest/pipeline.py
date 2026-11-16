"""Phase 1 + 2 ingestion: chunk -> estimate -> extract (cached) -> resolve -> Neo4j + pgvector.

    python -m ingest --path data/corpus [--estimate-only] [--check-idempotent]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

from kgrag.config import ROOT, Settings, get_settings
from kgrag.embeddings import get_embedder
from kgrag.ingest.chunker import chunk_document, discover, load_document
from kgrag.ingest.extract import Extractor
from kgrag.ingest.resolve import EntityResolver, duplicate_canonicals
from kgrag.llm import BudgetExceededError, LLMClient
from kgrag.log import get_logger
from kgrag.models import Chunk, ExtractionResult
from kgrag.ontology import EntityType
from kgrag.stores.factory import make_stores, verify_chunk_invariant

log = get_logger(__name__)


def run(corpus: Path, settings: Settings, estimate_only: bool = False, check_idempotent: bool = False,
        llm: LLMClient | None = None) -> dict[str, object]:
    t_start = time.perf_counter()
    paths = discover(corpus)
    if not paths:
        raise SystemExit(f"no .txt documents under {corpus}")
    docs = [load_document(p) for p in paths]
    chunks_by_doc = {d.document_id: chunk_document(d, settings.chunk_max_chars, settings.chunk_overlap_chars)
                     for d in docs}
    all_chunks: list[Chunk] = [c for cs in chunks_by_doc.values() for c in cs]

    llm = llm or LLMClient(settings)
    extractor = Extractor(llm, settings)

    # ---------------------------------------------------------------- cost guardrail
    estimates = [extractor.estimate(d, chunks_by_doc[d.document_id]) for d in docs]
    total_est = sum(e.est_usd for e in estimates)
    print(f"\n{'document':<28}{'chunks':>8}{'est. USD':>12}  cached")
    for e in estimates:
        print(f"{e.document_id:<28}{e.chunks:>8}{e.est_usd:>12.2f}  {'yes' if e.cached else 'no'}")
    print(f"{'TOTAL':<28}{len(all_chunks):>8}{total_est:>12.2f}   model={settings.extraction_model}")
    print(f"ledger spend so far ${llm.spent():.2f} / cap ${settings.budget_usd:.2f}\n")
    if estimate_only:
        return {"estimate_usd": total_est, "chunks": len(all_chunks)}
    try:
        llm.check_budget(total_est, "extraction (full corpus)")
    except BudgetExceededError as err:
        raise SystemExit(f"ABORT: {err}") from err

    # ---------------------------------------------------------------- extraction
    spend_before = llm.spent("extraction")
    t0 = time.perf_counter()
    results: dict[str, ExtractionResult] = {}
    for d in docs:
        results.update(extractor.extract_document(d, chunks_by_doc[d.document_id]))
    t_extract = time.perf_counter() - t0
    extraction_usd = llm.spent("extraction") - spend_before

    # ---------------------------------------------------------------- entity resolution
    embedder = get_embedder()
    resolver = EntityResolver(embedder, settings.er_threshold, ROOT / "data" / "aliases.json",
                              settings.logs_dir / "er_merges.jsonl")
    nodes, edges, er = resolver.resolve(results)
    dups = duplicate_canonicals(nodes)
    if dups:
        raise SystemExit(f"entity resolution produced duplicate canonicals: {dups[:5]}")

    # ---------------------------------------------------------------- embeddings (same chunk ids)
    t0 = time.perf_counter()
    vectors = embedder.embed([f"{c.company} | {c.section_path}\n{c.text}" for c in all_chunks])
    t_embed = time.perf_counter() - t0
    chunk_entities: dict[str, list[str]] = {}
    for n in nodes:
        for cid in n.chunk_ids:
            chunk_entities.setdefault(cid, []).append(n.node_id)

    # ---------------------------------------------------------------- writes (MERGE / upsert only)
    graph, vstore = make_stores(settings)
    try:
        graph.setup()
        vstore.setup()
        t0 = time.perf_counter()
        graph.write(nodes, edges, all_chunks)
        t_graph = time.perf_counter() - t0
        vstore.upsert(all_chunks, vectors, chunk_entities)
        counts = graph.counts()
        if check_idempotent:
            graph.write(nodes, edges, all_chunks)
            vstore.upsert(all_chunks, vectors, chunk_entities)
            again = graph.counts()
            if again != counts:
                raise SystemExit(f"NOT idempotent: {counts} -> {again}")
            print(f"idempotency check passed: {counts}")
        invariant = verify_chunk_invariant(graph, vstore)
    finally:
        graph.close()
        vstore.close()

    # ---------------------------------------------------------------- reporting
    spot = _spot_check(nodes, settings)
    report: dict[str, object] = {
        "documents": [d.document_id for d in docs],
        "chunks": len(all_chunks),
        "graph_counts": counts,
        "invariant": invariant,
        "entity_resolution": {**asdict(er), "merge_rate": round(er.merge_rate, 4)},
        "extraction_failed_chunks": len(extractor.failed),
        "chunks_without_entities": sum(1 for r in results.values() if not r.entities),
        "cost_usd": {"extraction": round(extraction_usd, 4), "embedding": 0.0,
                     "note": "embeddings are local (fastembed); extraction cost is this run's ledger delta"},
        "seconds": {"extract": round(t_extract, 1), "embed": round(t_embed, 1), "graph_write": round(t_graph, 1),
                    "total": round(time.perf_counter() - t_start, 1)},
        "spot_check_file": str(spot),
        "offline": settings.offline,
    }
    out = settings.results_dir / "ingest_report.json"
    # keep the cost of the first (uncached) run: a cached re-run reports $0 extraction
    if out.exists():
        prev = json.loads(out.read_text(encoding="utf-8"))
        if report["cost_usd"]["extraction"] == 0 and prev.get("cost_usd", {}).get("extraction"):  # type: ignore[index]
            report["cost_usd"]["first_run_extraction"] = prev["cost_usd"].get(  # type: ignore[index]
                "first_run_extraction", prev["cost_usd"]["extraction"])
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def _spot_check(nodes: list, settings: Settings) -> Path:  # type: ignore[type-arg]
    """Write 20 random Company nodes (seeded) with aliases for the manual duplicate spot-check."""
    companies = [n for n in nodes if n.entity_type == EntityType.COMPANY]
    rng = random.Random(42)
    sample = rng.sample(companies, min(20, len(companies)))
    path = settings.results_dir / "er_spotcheck.md"
    lines = ["# Entity-resolution spot check (20 random Company nodes, seed=42)", "",
             f"threshold = {settings.er_threshold}", "",
             "| node_id | canonical | aliases | mentions | duplicate? (fill in) |", "|---|---|---|---|---|"]
    for n in sorted(sample, key=lambda n: n.node_id):
        lines.append(f"| `{n.node_id}` | {n.canonical_name} | {', '.join(n.aliases[:6])} | {n.mention_count} | |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m ingest", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", type=Path, required=True, help="corpus directory (or one .txt file)")
    p.add_argument("--estimate-only", action="store_true", help="print the cost estimate and exit")
    p.add_argument("--check-idempotent", action="store_true", help="write twice and assert stable counts")
    a = p.parse_args(argv)
    run(a.path, get_settings(), a.estimate_only, a.check_idempotent)


if __name__ == "__main__":
    main(sys.argv[1:])
