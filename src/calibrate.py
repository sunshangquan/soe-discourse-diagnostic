"""Slogan-calibrated stance extraction.

Two sources of slogan signal:
  1. LLM's per-segment slogan_density (from score_llm.py)
  2. Hand-crafted slogan n-gram lexicon (from phase0/shared_ngrams_top200.tsv)

Calibrated score per (segment, dim):

  final_score = raw_score
                × (1 − λ_llm   · slogan_density_llm)
                × (1 − λ_ngram · slogan_density_ngram)
                × confidence_substantive

The reported defaults are λ_llm=1.0 and λ_ngram=2.0. The grid is an
exploratory sensitivity analysis, not selection on the pilot gold labels.

Output:
  results/scores_calibrated.csv  (final scores)
  results/slogan_density_ngram.csv  (per-segment ngram density, for inspection)
"""
from __future__ import annotations
import csv
import argparse
from pathlib import Path
from collections import defaultdict

from dimensions import DIM_KEYS
from utils import DATA_DIR, RESULTS_DIR, PHASE0_DIR


def load_slogan_lexicon() -> list[tuple[str, int]]:
    """Load shared n-grams from phase0, optionally extended.

    Returns list of (phrase, weight) where weight = log(doc_freq + 1).
    """
    p = PHASE0_DIR / "shared_ngrams_top200.tsv"
    out = []
    if not p.exists():
        print(f"WARN: {p} not found; using empty lexicon")
        return out
    import math
    with p.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            n = int(r["n"])
            df = int(r["doc_freq"])
            ng = r["ngram"]
            if n >= 5 and df >= 8:  # only use 5+ gram phrases that recur ≥10% of corpus
                out.append((ng, math.log(df + 1)))
    return out


def compute_ngram_slogan_density(text: str, lexicon: list[tuple[str, int]]) -> float:
    """Fraction of text characters covered by slogan phrases (weighted)."""
    if not lexicon or not text:
        return 0.0
    n_chars = len(text)
    total_weighted_hits = 0.0
    for phrase, weight in lexicon:
        # use simple substring presence (fast); allow multiple matches per phrase
        count = text.count(phrase)
        if count > 0:
            total_weighted_hits += weight * count * len(phrase)
    # normalize by text length; cap at 1.0
    density = min(1.0, total_weighted_hits / max(1, n_chars * 3))
    return density


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-llm", type=float, default=1.0,
                    help="Weight for LLM-reported slogan_density")
    ap.add_argument("--lambda-ngram", type=float, default=2.0,
                    help="Weight for n-gram lexicon slogan density")
    ap.add_argument("--no-confidence", action="store_true",
                    help="Skip the confidence_substantive multiplier")
    args = ap.parse_args()

    segs = {s["seg_id"]: s for s in csv.DictReader(
        (DATA_DIR / "segments.csv").open(encoding="utf-8"))}
    llm = {r["seg_id"]: r for r in csv.DictReader(
        (RESULTS_DIR / "scores_llm.csv").open(encoding="utf-8"))}
    print(f"Loaded {len(segs)} segments, {len(llm)} LLM scores")

    lexicon = load_slogan_lexicon()
    print(f"Slogan lexicon: {len(lexicon)} phrases")

    # n-gram density per segment
    ngram_density = {}
    for sid, s in segs.items():
        ngram_density[sid] = compute_ngram_slogan_density(s["text"], lexicon)

    # Inspection output
    ng_out = RESULTS_DIR / "slogan_density_ngram.csv"
    with ng_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seg_id", "ngram_slogan_density"])
        for sid, v in ngram_density.items():
            w.writerow([sid, round(v, 6)])
    print(f"Wrote {ng_out}")

    # Calibrate
    rows = []
    for sid, s in segs.items():
        if sid not in llm:
            continue
        l = llm[sid]
        sd_llm = float(l["slogan_density"])
        sd_ng = ngram_density.get(sid, 0.0)
        conf_sub = float(l["confidence_substantive"]) if not args.no_confidence else 1.0

        factor_llm = max(0.0, 1.0 - args.lambda_llm * sd_llm)
        factor_ng = max(0.0, 1.0 - args.lambda_ngram * sd_ng)
        mult = factor_llm * factor_ng * conf_sub

        row = {"seg_id": sid, "doc_id": s["doc_id"],
               "slogan_density_llm": round(sd_llm, 4),
               "slogan_density_ngram": round(sd_ng, 4),
               "confidence_substantive": round(conf_sub, 4),
               "calibration_mult": round(mult, 4)}
        for dk in DIM_KEYS:
            raw = float(l[f"raw_{dk}"])
            row[f"{dk}_cal_score"] = round(raw * mult, 6)
        rows.append(row)

    out = RESULTS_DIR / "scores_calibrated.csv"
    fns = ["seg_id", "doc_id",
           "slogan_density_llm", "slogan_density_ngram",
           "confidence_substantive", "calibration_mult"] \
          + [f"{k}_cal_score" for k in DIM_KEYS]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} with {len(rows)} calibrated rows")

    # Quick stats
    import statistics
    mults = [r["calibration_mult"] for r in rows]
    print(f"\ncalibration_mult: mean={statistics.mean(mults):.3f}, median={statistics.median(mults):.3f}")
    print(f"  → fraction of LLM raw signal preserved on average: {statistics.mean(mults):.1%}")


if __name__ == "__main__":
    main()
