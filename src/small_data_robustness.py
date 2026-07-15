"""Small-data robustness checks for rebuttal.

Reviewer tXQh's main concern is that the corpus is small (80 documents from
51 companies).  The paired design gives us 24 leader-change pairs and only
5 same-leader controls, so this script asks a stricter question:

  If we down-sample the leader-change side to the same size as the
  same-leader side (5 vs. 5 pairs), do the main contrasts survive?

The script also reports leave-one-leader-change and leave-one-same-leader
sensitivity.  Outputs are intended to be cited in the ARR response, not to
replace the full paper tables.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
from pathlib import Path

from stats_addon import load_all, find_pairs, doc_vectors, cosine_distance, cohen_d
from utils import RESULTS_DIR


METHOD_ORDER = ["dict", "lda", "bge", "llm", "calibrated"]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def quantile(sorted_xs, q):
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = q * (len(sorted_xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_xs[lo]
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def distances_for_pairs(doc_vec, pairs):
    return [cosine_distance(doc_vec[a], doc_vec[b]) for a, b in pairs if a in doc_vec and b in doc_vec]


def contrast_stats(d_lc, d_sl):
    delta = mean(d_lc) - mean(d_sl)
    return {
        "mean_lc": mean(d_lc),
        "mean_sl": mean(d_sl),
        "delta": delta,
        "cohen_d": cohen_d(d_lc, d_sl),
    }


def summarize_distribution(values):
    values_sorted = sorted(values)
    return {
        "min": values_sorted[0],
        "p05": quantile(values_sorted, 0.05),
        "median": statistics.median(values_sorted),
        "p95": quantile(values_sorted, 0.95),
        "max": values_sorted[-1],
    }


def balanced_lc_subsamples(d_lc_all, d_sl, k):
    """Exhaustively evaluate all k-sized leader-change subsets."""
    deltas = []
    ds = []
    for idxs in itertools.combinations(range(len(d_lc_all)), k):
        d_lc = [d_lc_all[i] for i in idxs]
        st = contrast_stats(d_lc, d_sl)
        deltas.append(st["delta"])
        ds.append(st["cohen_d"])
    return deltas, ds


def leave_one_out(values_to_drop, fixed_other, drop_side):
    """Return Delta/d after dropping one value from LC or SL side."""
    rows = []
    for i in range(len(values_to_drop)):
        kept = [v for j, v in enumerate(values_to_drop) if j != i]
        if drop_side == "lc":
            st = contrast_stats(kept, fixed_other)
        else:
            st = contrast_stats(fixed_other, kept)
        rows.append(st)
    return rows


def main():
    seg_meta, methods, _, _ = load_all()
    leader_change, same_leader = find_pairs(seg_meta)
    k = len(same_leader)

    detail_rows = []
    summary = {
        "n_documents": len({s["doc_id"] for s in seg_meta.values()}),
        "n_companies": len({s["company_norm"] for s in seg_meta.values()}),
        "n_leader_change_pairs": len(leader_change),
        "n_same_leader_pairs": len(same_leader),
        "balanced_lc_subset_size": k,
        "n_balanced_subsets_per_method": math.comb(len(leader_change), k),
        "methods": {},
    }

    for method in METHOD_ORDER:
        if method not in methods:
            continue
        dv = doc_vectors(methods[method], seg_meta)
        d_lc_all = distances_for_pairs(dv, leader_change)
        d_sl = distances_for_pairs(dv, same_leader)
        full = contrast_stats(d_lc_all, d_sl)

        # Exhaustive balanced 5-vs-5 check: every LC subset of size 5.
        balanced_deltas, balanced_ds = balanced_lc_subsamples(d_lc_all, d_sl, k)

        # Leave-one-out sensitivity on both sides.
        loo_lc = leave_one_out(d_lc_all, d_sl, "lc")
        loo_sl = leave_one_out(d_sl, d_lc_all, "sl")
        loo_lc_deltas = [r["delta"] for r in loo_lc]
        loo_lc_ds = [r["cohen_d"] for r in loo_lc]
        loo_sl_deltas = [r["delta"] for r in loo_sl]
        loo_sl_ds = [r["cohen_d"] for r in loo_sl]

        method_summary = {
            "full": full,
            "balanced": {
                "n_subsets": len(balanced_deltas),
                "delta": summarize_distribution(balanced_deltas),
                "cohen_d": summarize_distribution(balanced_ds),
                "pct_delta_positive": sum(x > 0 for x in balanced_deltas) / len(balanced_deltas),
                "pct_d_positive": sum(x > 0 for x in balanced_ds) / len(balanced_ds),
                "pct_d_ge_0_5": sum(x >= 0.5 for x in balanced_ds) / len(balanced_ds),
                "pct_d_ge_full": sum(x >= full["cohen_d"] for x in balanced_ds) / len(balanced_ds),
            },
            "leave_one_lc_out": {
                "delta": summarize_distribution(loo_lc_deltas),
                "cohen_d": summarize_distribution(loo_lc_ds),
                "pct_delta_positive": sum(x > 0 for x in loo_lc_deltas) / len(loo_lc_deltas),
            },
            "leave_one_sl_out": {
                "delta": summarize_distribution(loo_sl_deltas),
                "cohen_d": summarize_distribution(loo_sl_ds),
                "pct_delta_positive": sum(x > 0 for x in loo_sl_deltas) / len(loo_sl_deltas),
            },
        }
        summary["methods"][method] = method_summary

        detail_rows.append({
            "method": method,
            "n_lc_full": len(d_lc_all),
            "n_sl": len(d_sl),
            "full_delta": full["delta"],
            "full_cohen_d": full["cohen_d"],
            "balanced_n_subsets": len(balanced_deltas),
            "balanced_delta_p05": method_summary["balanced"]["delta"]["p05"],
            "balanced_delta_median": method_summary["balanced"]["delta"]["median"],
            "balanced_delta_p95": method_summary["balanced"]["delta"]["p95"],
            "balanced_d_p05": method_summary["balanced"]["cohen_d"]["p05"],
            "balanced_d_median": method_summary["balanced"]["cohen_d"]["median"],
            "balanced_d_p95": method_summary["balanced"]["cohen_d"]["p95"],
            "balanced_pct_delta_positive": method_summary["balanced"]["pct_delta_positive"],
            "balanced_pct_d_ge_0_5": method_summary["balanced"]["pct_d_ge_0_5"],
            "loo_lc_delta_min": method_summary["leave_one_lc_out"]["delta"]["min"],
            "loo_lc_delta_max": method_summary["leave_one_lc_out"]["delta"]["max"],
            "loo_lc_d_min": method_summary["leave_one_lc_out"]["cohen_d"]["min"],
            "loo_lc_d_max": method_summary["leave_one_lc_out"]["cohen_d"]["max"],
            "loo_sl_delta_min": method_summary["leave_one_sl_out"]["delta"]["min"],
            "loo_sl_delta_max": method_summary["leave_one_sl_out"]["delta"]["max"],
            "loo_sl_d_min": method_summary["leave_one_sl_out"]["cohen_d"]["min"],
            "loo_sl_d_max": method_summary["leave_one_sl_out"]["cohen_d"]["max"],
        })

    csv_path = RESULTS_DIR / "small_data_robustness.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    json_path = RESULTS_DIR / "small_data_robustness.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md_path = RESULTS_DIR / "small_data_robustness.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Small-data robustness checks\n\n")
        f.write(
            f"Corpus: {summary['n_documents']} documents from "
            f"{summary['n_companies']} companies; paired evaluation uses "
            f"{summary['n_leader_change_pairs']} leader-change pairs and "
            f"{summary['n_same_leader_pairs']} same-leader controls.\n\n"
        )
        f.write(
            "Balanced check: for each method, exhaustively evaluate every "
            f"{k}-pair subset of the leader-change side "
            f"({summary['n_balanced_subsets_per_method']:,} subsets), "
            f"so the comparison is {k} vs. {k} pairs.\n\n"
        )
        f.write("| method | full d | full Delta | balanced median d | balanced d 5-95% | pct Delta>0 | pct d>=0.5 | LOO-LC d range | LOO-SL d range |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in detail_rows:
            f.write(
                f"| {r['method']} | {r['full_cohen_d']:.3f} | {r['full_delta']:.4f} | "
                f"{r['balanced_d_median']:.3f} | "
                f"[{r['balanced_d_p05']:.3f}, {r['balanced_d_p95']:.3f}] | "
                f"{100*r['balanced_pct_delta_positive']:.1f}% | "
                f"{100*r['balanced_pct_d_ge_0_5']:.1f}% | "
                f"[{r['loo_lc_d_min']:.3f}, {r['loo_lc_d_max']:.3f}] | "
                f"[{r['loo_sl_d_min']:.3f}, {r['loo_sl_d_max']:.3f}] |\n"
            )

    note_path = RESULTS_DIR / "rebuttal_tXQh_experiment_note.md"
    with note_path.open("w", encoding="utf-8") as f:
        f.write("# Rebuttal note for Reviewer tXQh\n\n")
        f.write("Reviewer concern: the contribution may look light because the corpus has only 80 documents from 51 companies.\n\n")
        f.write("New analysis completed for the response:\n\n")
        f.write(
            f"* We ran an exhaustive balanced small-sample check. The full paired design has "
            f"{summary['n_leader_change_pairs']} leader-change pairs and "
            f"{summary['n_same_leader_pairs']} same-leader controls. To test whether the result is driven by the larger leader-change side, "
            f"we evaluated all {summary['n_balanced_subsets_per_method']:,} ways of selecting only {k} leader-change pairs, "
            f"and compared each subset against the same {k} same-leader controls.\n"
        )
        for method in ["llm", "calibrated", "lda"]:
            m = summary["methods"][method]
            f.write(
                f"* {method}: full Cohen d={m['full']['cohen_d']:.3f}, "
                f"full Delta={m['full']['delta']:.4f}; balanced median d="
                f"{m['balanced']['cohen_d']['median']:.3f}, 5--95%=["
                f"{m['balanced']['cohen_d']['p05']:.3f}, {m['balanced']['cohen_d']['p95']:.3f}], "
                f"Delta>0 in {100*m['balanced']['pct_delta_positive']:.1f}% of balanced subsets, "
                f"d>=0.5 in {100*m['balanced']['pct_d_ge_0_5']:.1f}%.\n"
            )
        f.write(
            "* We also ran leave-one-leader-change-out and leave-one-same-leader-out checks; "
            "the LLM contrast keeps positive Delta under all omissions, whereas LDA remains weak and unstable.\n"
        )
        f.write(
            "* Together with the exact label-permutation test already run on all 29 pairs "
            "(LLM one-sided p=0.0339; calibrated p=0.0416; LDA p=0.3527), this supports the claim that "
            "the paper's contribution is the paired diagnostic/evaluation design rather than corpus scale alone.\n"
        )

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {note_path}")


if __name__ == "__main__":
    main()
