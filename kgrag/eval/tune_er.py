"""Tune the entity-resolution similarity threshold on labeled name pairs.

    python -m kgrag.eval.tune_er [--pairs data/eval/er_pairs.jsonl]

Each line: {"a": "...", "b": "...", "type": "Company", "same": true}. Uses the exact same
`should_merge` decision as the resolver, sweeps thresholds, and reports precision / recall / F1.
False merges are worse than missed merges (a merged node silently corrupts every path through
it), so the recommendation is the highest-F1 threshold among those with precision == 1.0 when
one exists, otherwise the highest F0.5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kgrag.config import get_settings
from kgrag.embeddings import get_embedder
from kgrag.ingest.resolve import normalize, should_merge
from kgrag.ontology import EntityType


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=get_settings().eval_dir / "er_pairs.jsonl")
    a = ap.parse_args()
    pairs = [json.loads(x) for x in a.pairs.read_text(encoding="utf-8").splitlines() if x.strip()]
    emb = get_embedder()
    va = emb.embed([p["a"] for p in pairs])
    vb = emb.embed([p["b"] for p in pairs])
    sims = np.sum(va * vb, axis=1)
    rows = []
    for t in np.round(np.arange(0.80, 0.991, 0.01), 2):
        tp = fp = fn = tn = 0
        for p, sim in zip(pairs, sims, strict=True):
            et = EntityType(p["type"])
            pred = should_merge(normalize(p["a"], et), p["a"], normalize(p["b"], et), p["b"], float(sim), et, float(t))
            pred_b = pred is not None
            tp += pred_b and p["same"]
            fp += pred_b and not p["same"]
            fn += (not pred_b) and p["same"]
            tn += (not pred_b) and not p["same"]
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f05 = 1.25 * prec * rec / (0.25 * prec + rec) if prec + rec else 0.0
        rows.append({"threshold": float(t), "precision": prec, "recall": rec, "f1": f1, "f0.5": f05,
                     "false_merges": fp, "missed_merges": fn})
    perfect = [r for r in rows if r["precision"] == 1.0]
    # ties -> the higher (safer) threshold
    best = (max(perfect, key=lambda r: (round(r["f1"], 6), r["threshold"])) if perfect
            else max(rows, key=lambda r: r["f0.5"]))
    print(f"{'thr':>5}{'P':>7}{'R':>7}{'F1':>7}{'FP':>5}{'FN':>5}")
    for r in rows:
        mark = "  <- recommended" if r is best else ""
        print(f"{r['threshold']:>5.2f}{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}"
              f"{r['false_merges']:>5}{r['missed_merges']:>5}{mark}")
    s = get_settings()
    (s.results_dir / "er_tuning.json").write_text(
        json.dumps({"pairs": len(pairs), "embed_model": s.embed_model, "offline": s.offline,
                    "recommended": best, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nset KGRAG_ER_THRESHOLD={best['threshold']:.2f} in .env (current {s.er_threshold})")


if __name__ == "__main__":
    main()
