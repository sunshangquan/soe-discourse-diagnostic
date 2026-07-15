"""Summarise paraphrase robustness in a way that maps cleanly to the
paper's claim: a stance-aware method's score should COLLAPSE on
slogan-style rewrites of substantive content; a surface method's
score should be preserved.

For each method, compute on the 50 pairs:
  - mean of max-dim score on orig
  - mean of max-dim score on rewrite
  - mean retention (rewrite/orig per pair, averaged)
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

from utils import DATA_DIR, RESULTS_DIR
from dimensions import DIM_KEYS


def main():
    pairs = []
    with (DATA_DIR / "paraphrase_pairs.jsonl").open(encoding="utf-8") as f:
        for line in f:
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(pairs)} pairs")

    methods = ["dict", "raw", "calibrated"]
    rows = []
    for m in methods:
        orig_max = []
        rew_max = []
        retentions = []
        for p in pairs:
            v_o = [p["orig"][m][dk] for dk in DIM_KEYS]
            v_r = [p["rewritten"][m][dk] for dk in DIM_KEYS]
            o_max = max(v_o)
            r_max = max(v_r)
            orig_max.append(o_max)
            rew_max.append(r_max)
            if o_max > 1e-6:
                retentions.append(r_max / o_max)
        mean_orig = sum(orig_max) / len(orig_max)
        mean_rew = sum(rew_max) / len(rew_max)
        mean_ret = sum(retentions) / len(retentions) if retentions else 0.0
        rows.append({
            "method": m,
            "mean_max_orig": round(mean_orig, 4),
            "mean_max_rew": round(mean_rew, 4),
            "absolute_drop": round(mean_orig - mean_rew, 4),
            "mean_retention": round(mean_ret, 3),
        })
        print(f"  {m:12s}  orig_max={mean_orig:.4f}  rew_max={mean_rew:.4f}  drop={mean_orig-mean_rew:+.4f}  retention={mean_ret:.3f}")

    # Also: L1 stability (how often does LLM call rewrite slogan?)
    flipped = sum(1 for p in pairs
                  if p["orig"]["l1"] == "substantive" and p["rewritten"]["l1"] == "slogan")
    stay = sum(1 for p in pairs
               if p["orig"]["l1"] == "substantive" and p["rewritten"]["l1"] == "substantive")
    print(f"\nLLM L1 after rewrite: substantive→slogan = {flipped}/{len(pairs)} = {100*flipped/len(pairs):.0f}%")
    print(f"                       substantive→substantive = {stay}/{len(pairs)}")

    # Mean LLM-reported slogan_density on orig vs rewrite
    sd_o = [p["orig"]["slogan_density_llm"] for p in pairs]
    sd_r = [p["rewritten"]["slogan_density_llm"] for p in pairs]
    print(f"\nLLM slogan_density: orig mean = {sum(sd_o)/len(sd_o):.3f}, rewritten mean = {sum(sd_r)/len(sd_r):.3f}")

    cs_o = [p["orig"]["confidence_substantive"] for p in pairs]
    cs_r = [p["rewritten"]["confidence_substantive"] for p in pairs]
    print(f"LLM confidence_substantive: orig mean = {sum(cs_o)/len(cs_o):.3f}, rewritten mean = {sum(cs_r)/len(cs_r):.3f}")

    # Save
    out = {
        "n_pairs": len(pairs),
        "llm_l1_flip_substantive_to_slogan": flipped,
        "llm_slogan_density_orig_mean": round(sum(sd_o)/len(sd_o), 3),
        "llm_slogan_density_rewrite_mean": round(sum(sd_r)/len(sd_r), 3),
        "llm_confsub_orig_mean": round(sum(cs_o)/len(cs_o), 3),
        "llm_confsub_rewrite_mean": round(sum(cs_r)/len(cs_r), 3),
        "method_rows": rows,
    }
    out_path = RESULTS_DIR / "paraphrase_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
