"""Baseline 3: BGE-zh segment-vs-dimension cosine similarity.

For each segment, encode it; encode the NL description of each dimension; cosine.

Model: BAAI/bge-small-zh-v1.5 (~100MB, fast on CPU)

Output: results/scores_bge.csv
"""
from __future__ import annotations
import csv
import warnings
from pathlib import Path

import numpy as np

from dimensions import DIMENSIONS, DIM_KEYS
from utils import DATA_DIR, RESULTS_DIR

warnings.filterwarnings("ignore")

MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed. Install via:")
        print("  pip install sentence-transformers torch")
        return

    print(f"Loading {MODEL_NAME} (will auto-download if missing, ~100MB)...")
    model = SentenceTransformer(MODEL_NAME)

    # Encode dimension descriptions
    dim_texts = [d["nl_description"] for d in DIMENSIONS]
    dim_emb = model.encode(dim_texts, normalize_embeddings=True, show_progress_bar=False)
    print(f"Encoded {len(DIMENSIONS)} dimension descriptions, dim={dim_emb.shape[1]}")

    # Encode segments
    segs = list(csv.DictReader((DATA_DIR / "segments.csv").open(encoding="utf-8")))
    seg_texts = [s["text"] for s in segs]
    print(f"Encoding {len(seg_texts)} segments...")
    seg_emb = model.encode(seg_texts, normalize_embeddings=True, batch_size=64,
                            show_progress_bar=True)

    # Cosine = dot product on L2-normalized vectors
    sims = seg_emb @ dim_emb.T  # (N_seg, 5)

    rows = []
    for i, s in enumerate(segs):
        row = {"seg_id": s["seg_id"], "doc_id": s["doc_id"]}
        for j, dim_key in enumerate(DIM_KEYS):
            row[f"{dim_key}_bge_score"] = round(float(sims[i, j]), 6)
        rows.append(row)

    out = RESULTS_DIR / "scores_bge.csv"
    fns = ["seg_id", "doc_id"] + [f"{k}_bge_score" for k in DIM_KEYS]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out} with {len(rows)} segments")

    # Sanity: top-3 segs per dim
    print("\nTop 3 segments per dimension (by BGE cosine):")
    for j, dk in enumerate(DIM_KEYS):
        idx = np.argsort(-sims[:, j])[:3]
        print(f"  [{dk}]")
        for k in idx:
            print(f"    {sims[k, j]:.3f} | {seg_texts[k][:60]}")


if __name__ == "__main__":
    main()
