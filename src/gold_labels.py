"""Load the released three-annotator majority L1 labels for evaluation."""

from __future__ import annotations

import csv

from utils import RESULTS_DIR


def load_majority_l1() -> list[dict[str, str]]:
    path = RESULTS_DIR / "three_annotator_l1_merged.csv"
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or any(not row["seg_id"] or not row["majority_l1"] for row in rows):
        raise ValueError(f"Missing majority L1 labels or segment IDs in {path}")
    if len({row["seg_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate segment IDs in {path}")
    return [
        {
            "doc_id": row["doc_id"],
            "seg_id": row["seg_id"],
            "l1": row["majority_l1"],
            "l2": "",
            "match_score": "1.0",
        }
        for row in rows
    ]
