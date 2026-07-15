"""Phase 2 — Balance check between leader-change (24) and same-leader (5) pair groups.

For each pair group, report marginal distributions over:
  - industry tag (from filename)
  - speaker role
  - wave-gap proxy (we don't have explicit dates per doc, but A=wave-A, B=wave-B,
    and 'B' is the 2nd interview wave; gaps for SL vs LC are similar by design)
  - speakers identified as regulator (国资委) vs operator (firm-level)

Output: results/balance_check.csv + console summary
"""
from __future__ import annotations
import csv
from collections import defaultdict, Counter
from pathlib import Path

from utils import DATA_DIR, RESULTS_DIR
from stats_addon import find_pairs


def main():
    segs = list(csv.DictReader((DATA_DIR / "segments.csv").open(encoding="utf-8")))
    seg_meta = {s["seg_id"]: s for s in segs}

    leader_change, same_leader = find_pairs(seg_meta)
    print(f"Pairs: LC={len(leader_change)}, SL={len(same_leader)}")

    # Build doc → meta lookup
    doc_meta = {}
    for s in segs:
        if s["doc_id"] not in doc_meta:
            doc_meta[s["doc_id"]] = {
                "company_norm": s["company_norm"],
                "company": s["company"],
                "speaker": s["speaker"],
                "industry": s["industry"],
                "wave": s["wave"],
                "is_regulator": "国资委" in s["company"] or "国资委" in s["company_norm"],
            }

    def group_stats(pairs, name):
        rows = []
        for a, b in pairs:
            rows.append({
                "doc_a": a,
                "doc_b": b,
                "company": doc_meta[a]["company_norm"],
                "speaker_a": doc_meta[a]["speaker"],
                "speaker_b": doc_meta[b]["speaker"],
                "industry_a": doc_meta[a]["industry"],
                "industry_b": doc_meta[b]["industry"],
                "is_regulator_pair": doc_meta[a]["is_regulator"] or doc_meta[b]["is_regulator"],
                "wave_a": doc_meta[a]["wave"],
                "wave_b": doc_meta[b]["wave"],
            })

        print(f"\n== {name} (n={len(rows)}) ==")
        # Industry distribution
        all_industries = []
        for r in rows:
            for ind in (r["industry_a"], r["industry_b"]):
                all_industries.append(ind or "(unknown)")
        ind_counter = Counter(all_industries)
        print(f"  industry distribution (top 8): {dict(ind_counter.most_common(8))}")
        # Regulator pairs
        regulator_count = sum(1 for r in rows if r["is_regulator_pair"])
        print(f"  pairs containing regulator (国资委): {regulator_count}/{len(rows)} ({100*regulator_count/max(1,len(rows)):.0f}%)")
        # Wave types (all should be A-B = different waves)
        wave_pairs = Counter((min(r["wave_a"], r["wave_b"]), max(r["wave_a"], r["wave_b"])) for r in rows)
        print(f"  wave types: {dict(wave_pairs)}")
        return rows

    lc_rows = group_stats(leader_change, "Leader-change (LC)")
    sl_rows = group_stats(same_leader, "Same-leader (SL)")

    # Write detailed CSV
    out = RESULTS_DIR / "balance_check.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "doc_a", "doc_b", "company",
                                          "speaker_a", "speaker_b", "industry_a", "industry_b",
                                          "is_regulator_pair", "wave_a", "wave_b"])
        w.writeheader()
        for r in lc_rows:
            w.writerow({"group": "LC", **r})
        for r in sl_rows:
            w.writerow({"group": "SL", **r})
    print(f"\nWrote {out}")

    # Summary verdict
    print("\n=== Verdict ===")
    lc_reg = sum(1 for r in lc_rows if r["is_regulator_pair"])
    sl_reg = sum(1 for r in sl_rows if r["is_regulator_pair"])
    print(f"  Regulator share: LC {lc_reg}/{len(lc_rows)} vs SL {sl_reg}/{len(sl_rows)}")
    if sl_reg > 0:
        print("  ⚠️ SL group includes regulator pair — this could be a confound (regulators speak in formulaic register).")
    # Speaker-rotation pattern: SL by definition has same speaker; LC has different. That's the design intent.
    lc_inds = set(i for r in lc_rows for i in [r["industry_a"], r["industry_b"]] if i)
    sl_inds = set(i for r in sl_rows for i in [r["industry_a"], r["industry_b"]] if i)
    print(f"  Industries present: LC has {len(lc_inds)} distinct; SL has {len(sl_inds)} distinct")
    print(f"  LC-only industries: {lc_inds - sl_inds}")
    print(f"  SL-only industries: {sl_inds - lc_inds}")


if __name__ == "__main__":
    main()
