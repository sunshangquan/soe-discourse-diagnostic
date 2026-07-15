"""Load corpus → segments parquet/csv with metadata."""
from __future__ import annotations
import os
import csv
from pathlib import Path

from utils import CORPUS_DIR, DATA_DIR, read_text, split_paragraphs, parse_filename, normalize_company


def build_segments():
    files = sorted([f for f in os.listdir(CORPUS_DIR) if f.endswith(".txt")])
    rows = []
    for fn in files:
        meta = parse_filename(fn)
        meta["company_norm"] = normalize_company(meta["company"])
        text = read_text(CORPUS_DIR / fn)
        paras = split_paragraphs(text)
        for i, p in enumerate(paras):
            rows.append({
                "doc_id": fn,
                "company": meta["company"],
                "company_norm": meta["company_norm"],
                "speaker": meta["speaker"],
                "industry": meta["industry"],
                "wave": meta["wave"],
                "seg_id": f"{fn}::{i:03d}",
                "seg_idx": i,
                "char_count": len(p),
                "text": p,
            })

    out = DATA_DIR / "segments.csv"
    fieldnames = list(rows[0].keys())
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} with {len(rows)} segments from {len(files)} docs")
    return rows


if __name__ == "__main__":
    build_segments()
