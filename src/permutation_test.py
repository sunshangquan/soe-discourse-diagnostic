"""Phase 2 — Permutation test replacing bootstrap CI for paired contrast.

Under H0 (leader identity does not affect doc-vector distance), the 29
pair labels {LC, SL} can be re-shuffled freely.  For each shuffle we
recompute the test statistic (mean LC distance - mean SL distance) and
get the proportion of shuffles where |stat_shuffled| >= |stat_observed|.

This yields an exact-ish p-value that does not assume normality and is
robust to the small n_SL = 5.

We also report Hedges' g (small-sample-corrected) alongside Cohen's d.

Output: results/permutation_test.csv
"""
from __future__ import annotations
import csv
import math
import random
from itertools import combinations
from pathlib import Path

from dimensions import DIM_KEYS
from utils import DATA_DIR, RESULTS_DIR
from stats_addon import load_all, find_pairs, doc_vectors, cosine_distance, cohen_d

random.seed(42)
N_PERM = 200000  # > C(29,5)=118755 so exact enumeration applies


def hedges_g(d_lc, d_sl):
    """Cohen's d with Hedges' small-sample bias correction."""
    if not d_lc or not d_sl:
        return 0.0
    n1, n2 = len(d_lc), len(d_sl)
    d = cohen_d(d_lc, d_sl)
    # Hedges correction factor J = 1 - 3/(4*(n1+n2)-9)
    df = n1 + n2 - 2
    if df < 1:
        return d
    J = 1 - 3 / (4 * df - 1)
    return d * J


def glass_delta(d_lc, d_sl):
    """Glass's Delta uses control-group (SL) SD only."""
    if not d_lc or not d_sl:
        return 0.0
    m_lc = sum(d_lc) / len(d_lc)
    m_sl = sum(d_sl) / len(d_sl)
    if len(d_sl) < 2:
        return 0.0
    var_sl = sum((x - m_sl) ** 2 for x in d_sl) / (len(d_sl) - 1)
    sd_sl = math.sqrt(var_sl)
    return (m_lc - m_sl) / sd_sl if sd_sl > 0 else 0.0


def permutation_pvalue(d_lc, d_sl, n_perm=N_PERM):
    """Two-sided AND one-sided permutation p-values."""
    if not d_lc or not d_sl:
        return 1.0, 1.0, "n/a", 0
    all_vals = list(d_lc) + list(d_sl)
    n_lc = len(d_lc)
    obs_stat = sum(d_lc) / n_lc - sum(d_sl) / len(d_sl)

    total_n = len(all_vals)
    if math.comb(total_n, n_lc) <= n_perm:
        cnt2 = cnt1 = total = 0
        for idx_lc in combinations(range(total_n), n_lc):
            idx_lc_set = set(idx_lc)
            new_lc = [all_vals[i] for i in idx_lc]
            new_sl = [all_vals[i] for i in range(total_n) if i not in idx_lc_set]
            stat = sum(new_lc) / len(new_lc) - sum(new_sl) / len(new_sl)
            if abs(stat) >= abs(obs_stat) - 1e-12:
                cnt2 += 1
            if stat >= obs_stat - 1e-12:
                cnt1 += 1
            total += 1
        return cnt2 / total, cnt1 / total, "exact", total
    else:
        cnt2 = cnt1 = 0
        for _ in range(n_perm):
            shuffled = all_vals[:]
            random.shuffle(shuffled)
            new_lc = shuffled[:n_lc]
            new_sl = shuffled[n_lc:]
            stat = sum(new_lc) / len(new_lc) - sum(new_sl) / len(new_sl)
            if abs(stat) >= abs(obs_stat) - 1e-12:
                cnt2 += 1
            if stat >= obs_stat - 1e-12:
                cnt1 += 1
        return cnt2 / n_perm, cnt1 / n_perm, "monte_carlo", n_perm


def main():
    seg_meta, methods, llm_l1, gold = load_all()
    lc, sl = find_pairs(seg_meta)
    print(f"Pairs: LC={len(lc)}, SL={len(sl)}")
    # Total possible permutations: C(29, 24) = C(29, 5) = 118755
    total_perms = math.comb(29, 5)
    print(f"Total permutations C(29,5) = {total_perms}; running exact enumeration")

    rows = []
    print(f"\n{'method':12s} {'D_lc':>7s} {'D_sl':>7s} {'diff':>8s} {'cohen d':>8s} {'hedges g':>9s} {'glass Δ':>8s} {'p_2-sided':>10s} {'p_1-sided':>10s}")
    for name, sc in methods.items():
        dv = doc_vectors(sc, seg_meta)
        d_lc = [cosine_distance(dv[a], dv[b]) for a, b in lc if a in dv and b in dv]
        d_sl = [cosine_distance(dv[a], dv[b]) for a, b in sl if a in dv and b in dv]
        if not d_lc or not d_sl:
            continue
        m_lc = sum(d_lc) / len(d_lc)
        m_sl = sum(d_sl) / len(d_sl)
        d = cohen_d(d_lc, d_sl)
        g = hedges_g(d_lc, d_sl)
        gd = glass_delta(d_lc, d_sl)
        p2, p1, ptype, ptotal = permutation_pvalue(d_lc, d_sl)
        rows.append({
            "method": name,
            "mean_D_lc": round(m_lc, 4),
            "mean_D_sl": round(m_sl, 4),
            "diff": round(m_lc - m_sl, 4),
            "cohen_d": round(d, 3),
            "hedges_g": round(g, 3),
            "glass_delta": round(gd, 3),
            "perm_p_2sided": round(p2, 4),
            "perm_p_1sided": round(p1, 4),
            "perm_type": ptype,
            "n_perm": ptotal,
        })
        print(f"{name:12s} {m_lc:7.4f} {m_sl:7.4f} {m_lc - m_sl:+8.4f} {d:8.3f} {g:9.3f} {gd:8.3f} {p2:10.4f} {p1:10.4f}")

    out = RESULTS_DIR / "permutation_test.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
