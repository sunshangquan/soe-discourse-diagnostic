"""L2 sub-type analysis.

We collected LLM L2 labels (firm_action/policy_history/system_aggregate)
during the main scoring run.  This script evaluates:

  1. L2 distribution across speakers (does it differ between operators,
     regulators, and capital-management leaders?)
  2. L2 agreement with the hand-coded gold L2 labels (on the matched
     subset of the pilot)
  3. Per-L2 paired-contrast: does the leader-change vs same-leader
     contrast differ by L2 type?

Output: results/l2_analysis.json
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

from utils import DATA_DIR, RESULTS_DIR


def load_llm_l2():
    """LLM L2 labels per seg_id (from cache, which preserves L2)."""
    out = {}
    p = DATA_DIR / "llm_cache.jsonl"
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                out[obj["seg_id"]] = obj.get("l2", "")
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def main():
    segs = list(csv.DictReader((DATA_DIR / "segments.csv").open(encoding="utf-8")))
    seg_meta = {s["seg_id"]: s for s in segs}

    llm = {r["seg_id"]: r for r in csv.DictReader(
        (RESULTS_DIR / "scores_llm.csv").open(encoding="utf-8"))}
    l2 = load_llm_l2()
    print(f"LLM L2 labels: {len(l2)} non-empty entries")

    # 1. Distribution of L2 (only for substantive segments)
    sub_l2 = [l2.get(sid, "") for sid, r in llm.items() if r["l1"] == "substantive"]
    from collections import Counter
    l2_dist = Counter(sub_l2)
    print(f"\nL2 distribution (substantive only, n={len(sub_l2)}):")
    for k, v in l2_dist.most_common():
        print(f"  {k or '(empty)':20s} {v:4d}  ({100*v/len(sub_l2):.0f}%)")

    # 2. Per-speaker L2 mix (focus on the 5 hand-coded docs to compare to coding)
    doc_focus = [
        "国务院国资委-郝鹏.txt",
        "中国建材集团-宋志平-建材2.txt",
        "中国航天科技集团有限公司-吴燕生-航天2.txt",
        "中国节能环保集团有限公司-余红辉.txt",
        "中国国新控股有限责任公司-周渝波-资本2.txt",
    ]
    print(f"\nPer-doc L2 distribution (5 hand-coded docs, substantive segs only):")
    for did in doc_focus:
        rs = [llm.get(s["seg_id"]) for s in segs if s["doc_id"] == did]
        rs = [r for r in rs if r and r["l1"] == "substantive"]
        if not rs:
            continue
        l2s = [l2.get(r["seg_id"], "") for r in rs]
        c = Counter(l2s)
        total = len(rs)
        print(f"  {did[:40]:40s} n={total} "
              + " ".join(f"{k or '?'}={v}" for k, v in c.most_common()))

    # 3. Paired contrast on substantive-firm_action vs substantive-policy_history
    # i.e., do leader-change pairs differ more on firm_action content than on policy_history?
    # Aggregate per-doc the fraction of substantive segs that are each L2
    by_doc = defaultdict(list)
    for s in segs:
        by_doc[s["doc_id"]].append(s["seg_id"])

    def doc_l2_fractions(did):
        sids = by_doc[did]
        sub = [sid for sid in sids if llm.get(sid, {}).get("l1") == "substantive"]
        if not sub:
            return {t: 0.0 for t in ("firm_action", "policy_history", "system_aggregate")}
        out = {t: sum(1 for sid in sub if l2.get(sid, "") == t) / len(sub)
               for t in ("firm_action", "policy_history", "system_aggregate")}
        return out

    # Pairs
    from stats_addon import find_pairs
    lc, sl = find_pairs(seg_meta)
    print(f"\nPer-L2 paired contrast (mean abs frac diff):")
    print(f"  {'L2 type':22s} {'lc':>7s} {'sl':>7s} {'diff':>7s}")
    paired_stats = {}
    for t in ("firm_action", "policy_history", "system_aggregate"):
        d_lc = []
        d_sl = []
        for a, b in lc:
            va = doc_l2_fractions(a)[t]
            vb = doc_l2_fractions(b)[t]
            d_lc.append(abs(va - vb))
        for a, b in sl:
            va = doc_l2_fractions(a)[t]
            vb = doc_l2_fractions(b)[t]
            d_sl.append(abs(va - vb))
        m_lc = sum(d_lc) / len(d_lc) if d_lc else 0
        m_sl = sum(d_sl) / len(d_sl) if d_sl else 0
        paired_stats[t] = {"mean_lc": m_lc, "mean_sl": m_sl, "diff": m_lc - m_sl}
        print(f"  {t:22s} {m_lc:>7.3f} {m_sl:>7.3f} {m_lc-m_sl:>+7.3f}")

    out = {
        "l1_eq_substantive_l2_dist": dict(l2_dist),
        "per_doc_l2_focus": {did: dict(Counter(l2.get(s["seg_id"], "")
                                  for s in segs if s["doc_id"] == did
                                  and llm.get(s["seg_id"], {}).get("l1") == "substantive"))
                             for did in doc_focus},
        "paired_contrast_l2": paired_stats,
    }
    with (RESULTS_DIR / "l2_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {RESULTS_DIR / 'l2_analysis.json'}")


if __name__ == "__main__":
    main()
