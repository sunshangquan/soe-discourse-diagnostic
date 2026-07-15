"""Baseline 1: Dictionary scorer.

For each segment, count seed-word hits per dimension and normalize.

Output: results/scores_dict.csv with columns
  seg_id, doc_id, <dim>_dict_score for dim in DIM_KEYS
"""
from __future__ import annotations
import csv
import warnings
from pathlib import Path

import jieba

from dimensions import DIMENSIONS, DIM_KEYS, SEED_WORDS_FLAT
from utils import DATA_DIR, RESULTS_DIR

warnings.filterwarnings("ignore")


def score_segment(text: str) -> dict[str, float]:
    """Dictionary hit ratio per dimension (seed word matches / total tokens)."""
    tokens = list(jieba.cut(text))
    total = max(1, sum(1 for t in tokens if t.strip()))
    out = {}
    for dim_key, words in SEED_WORDS_FLAT.items():
        words_set = set(words)
        hits = sum(1 for t in tokens if t in words_set)
        out[f"{dim_key}_dict_score"] = round(hits / total, 6)
    return out


def main():
    segs = list(csv.DictReader((DATA_DIR / "segments.csv").open(encoding="utf-8")))
    rows = []
    for s in segs:
        scores = score_segment(s["text"])
        rows.append({"seg_id": s["seg_id"], "doc_id": s["doc_id"], **scores})

    out = RESULTS_DIR / "scores_dict.csv"
    fns = ["seg_id", "doc_id"] + [f"{k}_dict_score" for k in DIM_KEYS]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} with {len(rows)} segments")

    # Quick sanity: top-scored segment per dim
    print("\nTop 1 segment per dimension (by dict score):")
    for dk in DIM_KEYS:
        col = f"{dk}_dict_score"
        top = max(rows, key=lambda r: float(r[col]))
        seg_text = next(s for s in segs if s["seg_id"] == top["seg_id"])["text"][:50]
        print(f"  [{dk}] score={top[col]} | {seg_text}...")


if __name__ == "__main__":
    main()
