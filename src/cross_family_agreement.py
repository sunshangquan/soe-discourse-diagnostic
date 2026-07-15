"""Phase 2-4 — Cross-family LLM agreement (Qwen3.5:9b vs DeepSeek-r1:8b).

The within-family check (vs Qwen3.5:27b) does not control for shared
pretraining or architecture.  DeepSeek-r1 is a different family from a
different vendor and tokenizer; an agreement check here addresses the
pretraining-contamination concern.

Output: results/cross_family_agreement.json
"""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path

from dimensions import DIM_KEYS
from utils import DATA_DIR, RESULTS_DIR


def cohen_kappa_3way(pairs):
    """Cohen's kappa for nominal labels.

    pairs: list of (a, b) tuples; both a, b are categorical labels.
    """
    if not pairs:
        return 0.0
    cats = sorted(set([a for a, _ in pairs] + [b for _, b in pairs]))
    n = len(pairs)
    obs_agree = sum(1 for a, b in pairs if a == b) / n
    # marginals
    p_a = {c: sum(1 for a, _ in pairs if a == c) / n for c in cats}
    p_b = {c: sum(1 for _, b in pairs if b == c) / n for c in cats}
    exp_agree = sum(p_a[c] * p_b[c] for c in cats)
    if exp_agree == 1.0:
        return 1.0
    return (obs_agree - exp_agree) / (1 - exp_agree)


def pearson(xs, ys):
    if not xs or not ys or len(xs) != len(ys):
        return 0.0
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    deny = (sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / (denx * deny) if denx > 0 and deny > 0 else 0.0


def load_cache(suffix):
    p = DATA_DIR / f"llm_subset_cache_{suffix}.jsonl"
    out = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                out[obj["seg_id"]] = obj
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def main():
    # 9b scores from main run
    main_csv = RESULTS_DIR / "scores_llm.csv"
    qwen9b = {r["seg_id"]: r for r in csv.DictReader(main_csv.open(encoding="utf-8"))}
    print(f"9b: {len(qwen9b)} segments")

    ds = load_cache("deepseek_r1_8b")
    print(f"DeepSeek-r1:8b: {len(ds)} segments")

    # Intersection
    shared = sorted(set(qwen9b.keys()) & set(ds.keys()))
    print(f"shared: {len(shared)}")

    # L1 pairs
    l1_pairs = [(qwen9b[sid]["l1"], ds[sid]["l1"]) for sid in shared]
    n = len(l1_pairs)
    same = sum(1 for a, b in l1_pairs if a == b)
    kappa_l1 = cohen_kappa_3way(l1_pairs)
    print(f"\nL1 agreement: {same}/{n} = {100*same/n:.1f}% raw; κ = {kappa_l1:.3f}")

    # L1 distribution comparison
    dist_9b = Counter(p[0] for p in l1_pairs)
    dist_ds = Counter(p[1] for p in l1_pairs)
    print(f"  9b L1 dist: {dict(dist_9b)}")
    print(f"  DeepSeek L1 dist: {dict(dist_ds)}")

    # L2 agreement (among shared substantive)
    sub_pairs = [(qwen9b[sid]["l2"], ds[sid]["l2"]) for sid in shared
                 if qwen9b[sid]["l1"] == "substantive" and ds[sid]["l1"] == "substantive"
                 and qwen9b[sid]["l2"] and ds[sid]["l2"]]
    print(f"\nMutual substantive segments with both L2: {len(sub_pairs)}")
    if sub_pairs:
        kappa_l2 = cohen_kappa_3way(sub_pairs)
        print(f"L2 kappa: {kappa_l2:.3f}")
    else:
        kappa_l2 = None

    # Per-dimension Pearson on raw stance scores
    print("\nPer-dimension Pearson r on raw_dim scores:")
    per_dim = {}
    for dk in DIM_KEYS:
        xs = [float(qwen9b[sid][f"raw_{dk}"]) for sid in shared]
        ys = [float(ds[sid][f"raw_{dk}"]) for sid in shared]
        r = pearson(xs, ys)
        per_dim[dk] = r
        print(f"  {dk:30s}: r = {r:.3f}")
    mean_r = sum(per_dim.values()) / len(per_dim)
    print(f"  mean: {mean_r:.3f}")

    # slogan_density, confidence_substantive correlations
    sd_xs = [float(qwen9b[sid]["slogan_density"]) for sid in shared]
    sd_ys = [float(ds[sid]["slogan_density"]) for sid in shared]
    r_sd = pearson(sd_xs, sd_ys)
    cs_xs = [float(qwen9b[sid]["confidence_substantive"]) for sid in shared]
    cs_ys = [float(ds[sid]["confidence_substantive"]) for sid in shared]
    r_cs = pearson(cs_xs, cs_ys)
    print(f"\nslogan_density r = {r_sd:.3f}")
    print(f"confidence_substantive r = {r_cs:.3f}")

    out = {
        "n_shared": n,
        "second_model": "deepseek-r1:8b",
        "kappa_l1": round(kappa_l1, 3),
        "raw_agreement_l1": round(same / n, 3),
        "kappa_l2": round(kappa_l2, 3) if kappa_l2 is not None else None,
        "n_mutually_substantive_with_l2": len(sub_pairs),
        "per_dim_pearson_r": {k: round(v, 3) for k, v in per_dim.items()},
        "per_dim_mean_r": round(mean_r, 3),
        "slogan_density_pearson_r": round(r_sd, 3),
        "confidence_substantive_pearson_r": round(r_cs, 3),
        "L1_distribution_9b": dict(dist_9b),
        "L1_distribution_deepseek": dict(dist_ds),
    }
    with (RESULTS_DIR / "cross_family_agreement.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {RESULTS_DIR / 'cross_family_agreement.json'}")


if __name__ == "__main__":
    main()
