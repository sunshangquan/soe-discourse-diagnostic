"""Cross-LLM agreement between Qwen3.5:9b (main) and a second model.

For segments that both models scored, compute:
  - L1 Cohen's kappa
  - L2 Cohen's kappa on substantive
  - Per-dimension Pearson correlation of stance_scores
  - slogan_density correlation

Output: results/cross_llm_agreement.json
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from dimensions import DIM_KEYS
from utils import DATA_DIR, RESULTS_DIR


def cohen_kappa(a: list, b: list) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = {c: a.count(c) / n for c in cats}
    pb = {c: b.count(c) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    if pe == 1.0:
        return 0.0
    return (po - pe) / (1 - pe)


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--second-cache", default="llm_subset_cache_qwen35_27b.jsonl")
    ap.add_argument("--second-label", default="qwen3.5:27b")
    args = ap.parse_args()

    # Load 9b
    p9b = DATA_DIR / "llm_cache.jsonl"
    nine = {}
    with p9b.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                nine[obj["seg_id"]] = obj
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"9b: {len(nine)} entries")

    # Load second model
    p2 = DATA_DIR / args.second_cache
    second = {}
    if p2.exists():
        with p2.open(encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    second[obj["seg_id"]] = obj
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"{args.second_label}: {len(second)} entries")

    # Shared
    shared = sorted(set(nine.keys()) & set(second.keys()))
    print(f"Shared: {len(shared)} segments")
    if len(shared) < 10:
        print("Too few shared segments for κ.")
        return

    a_l1 = [nine[s]["l1"] for s in shared]
    b_l1 = [second[s]["l1"] for s in shared]
    kappa_l1 = cohen_kappa(a_l1, b_l1)
    print(f"\nL1 Cohen's κ = {kappa_l1:.3f}")
    print(f"L1 raw agreement = {sum(1 for x,y in zip(a_l1,b_l1) if x==y)/len(shared):.3f}")
    from collections import Counter
    print(f"  9b L1 dist: {dict(Counter(a_l1))}")
    print(f"  {args.second_label} L1 dist: {dict(Counter(b_l1))}")

    sub_idx = [i for i, (x, y) in enumerate(zip(a_l1, b_l1))
               if x == "substantive" and y == "substantive"]
    if sub_idx:
        a_l2 = [nine[shared[i]].get("l2", "") for i in sub_idx]
        b_l2 = [second[shared[i]].get("l2", "") for i in sub_idx]
        kappa_l2 = cohen_kappa(a_l2, b_l2)
        print(f"\nL2 Cohen's κ (on {len(sub_idx)} mutually-substantive) = {kappa_l2:.3f}")
    else:
        kappa_l2 = 0.0
        print("\nNo mutually-substantive segments for L2 κ.")

    # Per-dim stance correlation
    per_dim_corr = {}
    print("\nPer-dimension stance_score Pearson r (raw 0-1):")
    for dk in DIM_KEYS:
        xs = [nine[s].get(f"raw_{dk}", 0) for s in shared]
        ys = [second[s].get(f"raw_{dk}", 0) for s in shared]
        r = pearson(xs, ys)
        per_dim_corr[dk] = r
        print(f"  {dk:30s} r = {r:+.3f}")

    sd_a = [nine[s].get("slogan_density", 0) for s in shared]
    sd_b = [second[s].get("slogan_density", 0) for s in shared]
    sd_r = pearson(sd_a, sd_b)
    print(f"\nslogan_density Pearson r = {sd_r:+.3f}")

    cs_a = [nine[s].get("confidence_substantive", 0) for s in shared]
    cs_b = [second[s].get("confidence_substantive", 0) for s in shared]
    cs_r = pearson(cs_a, cs_b)
    print(f"confidence_substantive Pearson r = {cs_r:+.3f}")

    out = {
        "n_shared": len(shared),
        "second_model": args.second_label,
        "kappa_l1": round(kappa_l1, 3),
        "kappa_l2": round(kappa_l2, 3),
        "per_dim_pearson_r": {k: round(v, 3) for k, v in per_dim_corr.items()},
        "slogan_density_pearson_r": round(sd_r, 3),
        "confidence_substantive_pearson_r": round(cs_r, 3),
        "L1_distribution_9b": dict(Counter(a_l1)),
        "L1_distribution_second": dict(Counter(b_l1)),
    }
    out_path = RESULTS_DIR / "cross_llm_agreement.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
