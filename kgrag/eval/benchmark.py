"""Phase 5: hybrid vs vector-only on the same labeled questions.

    python -m kgrag.eval.benchmark [--limit N] [--write-readme]

Reports accuracy by category (single_hop, two_hop, three_hop, aggregation, out_of_scope/refusal),
latency p50/p95 and cost per query for both systems, plus the one-time ingestion cost.
Grading:
  * out_of_scope: correct iff the system refused.
  * in-scope: an LLM judge compares the answer with the gold answer (refusal = wrong); a
    deterministic keyword check is reported alongside as a sanity check on the judge.
Bootstrap 95% CIs are included because n per category is small. Questions with
`verified: false` are counted and flagged in the report.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from kgrag.config import ROOT, get_settings
from kgrag.embeddings import get_embedder
from kgrag.llm import LLMClient
from kgrag.rag import HybridRAG, VectorOnlyRAG
from kgrag.stores.factory import make_stores

CATEGORIES = ["single_hop", "two_hop", "three_hop", "aggregation", "out_of_scope"]
LABELS = {"single_hop": "Single-hop", "two_hop": "2-hop", "three_hop": "3-hop", "aggregation": "Aggregation",
          "out_of_scope": "Refusal (out-of-scope)"}


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    correct: bool
    reason: str


JUDGE_SYSTEM = """You grade answers to questions about SEC 10-K filings against a gold answer.
correct=true only if the answer states the gold answer's key facts (all listed items for list
questions; minor extra items are fine if not contradicting). Wording may differ. A refusal or an
answer that contradicts the gold answer is incorrect. Be strict and brief."""


def keyword_ok(answer: str, keywords: list[str]) -> bool:
    a = answer.lower()
    return all(any(alt.strip().lower() in a for alt in kw.split("|")) for kw in keywords)


def judge(llm: LLMClient, q: dict[str, Any], answer: str, refused: bool) -> tuple[bool, str]:
    if q["category"] == "out_of_scope":
        return refused, "refused" if refused else "answered an out-of-scope question"
    if refused:
        return False, "refused an in-scope question"
    plain = re.sub(r" \[[0-9, ]+\]", "", answer)  # strip citation markers before judging
    v, _ = llm.structured(
        model=get_settings().synthesis_model, system=JUDGE_SYSTEM,
        user=f"Question: {q['question']}\nGold answer: {q['gold_answer']}\nSystem answer: {plain}",
        output=Verdict, purpose="judge", max_tokens=600, expected_output_tokens=80, effort="low",
    )
    return v.correct, v.reason


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) - 1)))))
    return xs[i]


def bootstrap_ci(hits: list[bool], n: int = 2000, seed: int = 7) -> tuple[float, float]:
    if not hits:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(hits, k=len(hits))) / len(hits) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"by_category": {}, "systems": {}}
    for sysname in ("vector_only", "hybrid"):
        lat = [r[sysname]["latency_ms"] for r in rows]
        cost = [r[sysname]["cost_usd"] for r in rows]
        out["systems"][sysname] = {
            "accuracy": sum(r[sysname]["correct"] for r in rows) / max(1, len(rows)),
            "keyword_accuracy": sum(r[sysname]["keyword_ok"] for r in rows) / max(1, len(rows)),
            "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95),
            "cost_per_query_usd": statistics.mean(cost) if cost else 0.0,
        }
    for cat in CATEGORIES:
        rs = [r for r in rows if r["category"] == cat]
        if not rs:
            continue
        d: dict[str, Any] = {"n": len(rs), "unverified": sum(1 for r in rs if not r["verified"])}
        for sysname in ("vector_only", "hybrid"):
            hits = [bool(r[sysname]["correct"]) for r in rs]
            d[sysname] = {"accuracy": sum(hits) / len(hits), "ci95": bootstrap_ci(hits),
                          "keyword_accuracy": sum(r[sysname]["keyword_ok"] for r in rs) / len(rs)}
        out["by_category"][cat] = d
    hy = {r["id"]: r for r in rows}
    out["graph_hurt"] = [i for i, r in hy.items() if r["vector_only"]["correct"] and not r["hybrid"]["correct"]]
    out["graph_helped"] = [i for i, r in hy.items() if r["hybrid"]["correct"] and not r["vector_only"]["correct"]]
    return out


def markdown(summary: dict[str, Any], ingest: dict[str, Any], n_docs: int, offline: bool) -> str:
    L = []
    if offline:
        L.append("> **OFFLINE DRY RUN** — fake LLM + hash embeddings. These numbers are NOT results.\n")
    L.append("| Question type | n | Vector-only | Hybrid (graph + vector) | Δ |")
    L.append("|---|---:|---:|---:|---:|")
    for cat in CATEGORIES:
        d = summary["by_category"].get(cat)
        if not d:
            continue
        v, h = d["vector_only"], d["hybrid"]
        L.append(f"| {LABELS[cat]} | {d['n']} | {v['accuracy']:.0%} ({v['ci95'][0]:.0%}–{v['ci95'][1]:.0%}) | "
                 f"{h['accuracy']:.0%} ({h['ci95'][0]:.0%}–{h['ci95'][1]:.0%}) | "
                 f"{(h['accuracy'] - v['accuracy']) * 100:+.0f} pts |")
    sv, sh = summary["systems"]["vector_only"], summary["systems"]["hybrid"]
    L.append(f"| **All** | {sum(d['n'] for d in summary['by_category'].values())} | **{sv['accuracy']:.0%}** | "
             f"**{sh['accuracy']:.0%}** | {(sh['accuracy'] - sv['accuracy']) * 100:+.0f} pts |")
    L.append(f"| Latency p50 / p95 | | {sv['p50_ms']:.0f} / {sv['p95_ms']:.0f} ms | "
             f"{sh['p50_ms']:.0f} / {sh['p95_ms']:.0f} ms | |")
    L.append(f"| Cost per query | | ${sv['cost_per_query_usd']:.4f} | ${sh['cost_per_query_usd']:.4f} | |")
    c = ingest.get("cost_usd", {})
    ext = c.get("first_run_extraction") or c.get("extraction", 0.0)
    L.append("")
    L.append(f"One-time ingestion ({n_docs} filings, {ingest.get('chunks', '?')} chunks): "
             f"extraction ${ext:.2f} · embeddings local ($0) · graph "
             f"{ingest.get('graph_counts', {}).get('nodes', '?')} nodes / "
             f"{ingest.get('graph_counts', {}).get('edges', '?')} edges, "
             f"{ingest.get('seconds', {}).get('total', '?')} s wall-clock.")
    L.append(f"Accuracy = LLM judge vs. gold answer (95% bootstrap CI). Keyword-check accuracy: "
             f"vector {sv['keyword_accuracy']:.0%}, hybrid {sh['keyword_accuracy']:.0%}. "
             f"Graph hurt on {len(summary['graph_hurt'])} question(s), helped on {len(summary['graph_helped'])}.")
    unv = sum(d["unverified"] for d in summary["by_category"].values())
    if unv:
        L.append(f"⚠ {unv} gold answers are not yet verified against the filings.")
    return "\n".join(L)


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=get_settings().eval_dir / "benchmark_questions.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--write-readme", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    qs = [json.loads(x) for x in a.questions.read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.limit:
        qs = qs[: a.limit]
    llm = LLMClient(s)
    graph, vectors = make_stores(s)
    emb = get_embedder()
    hybrid = HybridRAG(s, llm, emb, vectors, graph)
    baseline = VectorOnlyRAG(s, llm, emb, vectors)

    rows: list[dict[str, Any]] = []
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_jsonl = s.results_dir / f"benchmark_{ts}.jsonl"
    for i, q in enumerate(qs, 1):
        row: dict[str, Any] = {"id": q["id"], "category": q["category"], "question": q["question"],
                               "verified": q.get("verified", False)}
        for sysobj in (baseline, hybrid):
            llm.read_cache = False  # real latency + cost
            resp, tr = sysobj.ask(q["question"])
            llm.read_cache = True
            ok, why = judge(llm, q, resp.answer, resp.refused)
            row[sysobj.name] = {"answer": resp.answer, "refused": resp.refused, "route": resp.route_used,
                                "citations": [c.chunk_id for c in resp.citations], "correct": ok, "judge": why,
                                "keyword_ok": (resp.refused if q["category"] == "out_of_scope"
                                               else keyword_ok(resp.answer, q.get("gold_keywords", []))),
                                "latency_ms": tr.latency_ms, "cost_usd": tr.cost_usd,
                                "query_type": tr.query_type, "graph_fallback": tr.graph_fallback}
        rows.append(row)
        with out_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i}/{len(qs)}] {q['id']:<8} {q['category']:<13} vector={'✓' if row['vector_only']['correct'] else '✗'} "
              f"hybrid={'✓' if row['hybrid']['correct'] else '✗'} route={row['hybrid']['route']}")

    summary = summarize(rows)
    ingest_path = s.results_dir / "ingest_report.json"
    ingest = json.loads(ingest_path.read_text(encoding="utf-8")) if ingest_path.exists() else {}
    md = markdown(summary, ingest, len(ingest.get("documents", [])), s.offline)
    (s.results_dir / f"benchmark_{ts}.json").write_text(json.dumps({"summary": summary, "offline": s.offline},
                                                                   indent=2), encoding="utf-8")
    (s.results_dir / "benchmark_latest.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)
    if a.write_readme:
        if s.offline:
            print("\nrefusing to write offline numbers into README.md")
        else:
            readme = ROOT / "README.md"
            text = readme.read_text(encoding="utf-8")
            text = re.sub(r"<!-- BENCHMARK:START -->.*<!-- BENCHMARK:END -->",
                          f"<!-- BENCHMARK:START -->\n{md}\n<!-- BENCHMARK:END -->", text, flags=re.S)
            readme.write_text(text, encoding="utf-8")
            print("README.md benchmark table updated")
    graph.close()
    vectors.close()


if __name__ == "__main__":
    main()
