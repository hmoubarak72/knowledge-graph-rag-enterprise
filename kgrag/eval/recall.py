"""Phase 2 gate: recall@k on a labeled set, across HNSW configurations.

    python -m kgrag.eval.recall [--set data/eval/recall_set.jsonl]

Labels are content rules, not chunk IDs (chunk IDs only exist after chunking): a chunk is relevant
to a query if it belongs to the labeled company and (optionally) its section starts with
`section_prefix` and its text contains every phrase in `all_of` (case-insensitive).
recall@k = share of queries with >= 1 relevant chunk in the top k. ANN recall (HNSW vs exact
search overlap) is reported separately so index tuning and embedding quality aren't conflated.
Gate: recall@k >= 0.85 at the configured k, else the report says FAIL and why.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from kgrag.config import get_settings
from kgrag.embeddings import get_embedder
from kgrag.models import Chunk
from kgrag.stores.factory import make_stores
from kgrag.stores.vector import PgVectorStore

GATE = 0.85
INDEX_CONFIGS = [(16, 64), (16, 128), (32, 200)]
EF_SEARCH = [20, 40, 80, 160]
KS = [5, 8, 10]


def load_set(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def relevant_ids(q: dict[str, Any], chunks: list[Chunk]) -> set[str]:
    comp = q.get("company", "").lower()
    out = set()
    for c in chunks:
        if comp and comp not in c.company.lower() and not c.document_id.lower().startswith(comp):
            continue
        if q.get("section_prefix") and not c.section_path.lower().startswith(q["section_prefix"].lower()):
            continue
        t = c.text.lower()
        if all(p.lower() in t for p in q.get("all_of", [])):
            out.add(c.chunk_id)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=Path, default=get_settings().eval_dir / "recall_set.jsonl")
    a = ap.parse_args()
    s = get_settings()
    graph, store = make_stores(s)
    emb = get_embedder()
    chunks = store.all_chunks()
    qs = load_set(a.set)
    labeled: list[tuple[dict[str, Any], set[str]]] = []
    unresolved: list[tuple[dict[str, Any], set[str]]] = []
    for q in qs:
        rel = relevant_ids(q, chunks)
        (labeled if rel else unresolved).append((q, rel))
    print(f"{len(qs)} queries, {len(labeled)} with >=1 relevant chunk, {len(unresolved)} unresolved labels")
    qvecs = [emb.embed_query(q["query"]) for q, _ in labeled]
    kmax = max(KS)

    def score(results: list[list[str]]) -> dict[str, float]:
        out = {}
        for k in KS:
            hits = [bool(rel & set(r[:k])) for (_q, rel), r in zip(labeled, results, strict=True)]
            out[f"recall@{k}"] = sum(hits) / max(1, len(labeled))
        return out

    rows: list[dict[str, Any]] = []
    exact = [[p.chunk_id for p in store.search(v, kmax, exact=True)] for v in qvecs]
    rows.append({"config": "exact (no index)", **score(exact), "ann_overlap@10": 1.0, "p50_ms": None})

    if isinstance(store, PgVectorStore):
        for m, efc in INDEX_CONFIGS:
            t0 = time.perf_counter()
            store.build_index(m, efc)
            build_s = time.perf_counter() - t0
            for ef in EF_SEARCH:
                lat, res = [], []
                for v in qvecs:
                    t = time.perf_counter()
                    res.append([p.chunk_id for p in store.search(v, kmax, ef_search=ef)])
                    lat.append((time.perf_counter() - t) * 1000)
                overlap = statistics.mean(len(set(r[:10]) & set(e[:10])) / max(1, len(e[:10]))
                                          for r, e in zip(res, exact, strict=True))
                rows.append({"config": f"m={m} ef_construction={efc} ef_search={ef}", **score(res),
                             "ann_overlap@10": round(overlap, 3), "p50_ms": round(statistics.median(lat), 2),
                             "index_build_s": round(build_s, 1)})
        store.build_index(s.hnsw_m, s.hnsw_ef_construction)  # restore configured index

    k = s.vector_top_k
    key = f"recall@{k}" if f"recall@{k}" in rows[0] else "recall@8"
    best = max(rows, key=lambda r: (r[key], -(r["p50_ms"] or 0)))
    passed = best[key] >= GATE
    report = {"n_queries": len(qs), "n_labeled": len(labeled), "unresolved": [q["id"] for q, _ in unresolved],
              "gate": GATE, "k": k, "passed": passed, "best": best, "rows": rows}
    (s.results_dir / "recall.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{'config':<44}" + "".join(f"{f'recall@{x}':>11}" for x in KS) + f"{'ANN@10':>9}{'p50 ms':>9}")
    for r in rows:
        print(f"{r['config']:<44}" + "".join(f"{r[f'recall@{x}']:>11.3f}" for x in KS)
              + f"{r['ann_overlap@10']:>9}" + f"{str(r['p50_ms']):>9}")
    print(f"\nGATE {key} >= {GATE}: {'PASS' if passed else 'FAIL'} (best: {best['config']} = {best[key]:.3f})")
    if not passed:
        print("Do not build on this retrieval: inspect misses in data/results/recall.json (chunk size, "
              "embedding model, query prefix) before Phase 3.")
    graph.close()
    store.close()


if __name__ == "__main__":
    main()
