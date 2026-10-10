"""Three-channel evaluation:

  Channel 1 — Paired contrast:
      For each method's doc-level dimension vector, compute mean cosine distance
      between (a) leader-change pairs and (b) same-leader pairs.
      A stance-detecting method should yield D_leader_change > D_same_leader.
      Report effect size = D_diff − D_same and bootstrap CI.

  Channel 2 — Gold segment F1:
      On the 86 high-confidence matched gold segments, evaluate each method's
      L1 prediction (binary: slogan vs substantive). For LLM/calibrated, this
      is direct (l1 column). For dict/LDA/BGE, we infer L1 by threshold:
        slogan if max-dim-score < median(max-dim-score)  [no signal → likely slogan]
      This is a coarse stand-in; the paper notes the absence of direct slogan
      prediction is itself a baseline failure.

  Channel 3 — External validity correlation:
      If data/external_signals.csv exists with columns company_norm, year, signal_name, value
      compute per-(company, year) aggregated dimension score and correlate
      with the external signal. NaN-tolerant Spearman.

All results dumped to results/eval_summary.md
"""
from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path

from dimensions import DIM_KEYS
from gold_labels import load_majority_l1
from utils import DATA_DIR, RESULTS_DIR


# ──────────────────────────── helpers ────────────────────────────

def load_segments():
    return {r["seg_id"]: r for r in csv.DictReader((DATA_DIR / "segments.csv").open(encoding="utf-8"))}


def load_method_scores() -> dict[str, dict]:
    """Return {method_name: {seg_id: {dim: score}}}"""
    methods = {}
    spec = [
        ("dict", "scores_dict.csv", "_dict_score"),
        ("lda", "scores_lda.csv", "_lda_score"),
        ("bge", "scores_bge.csv", "_bge_score"),
        ("llm", "scores_llm.csv", "raw_"),
        ("calibrated", "scores_calibrated.csv", "_cal_score"),
    ]
    for name, fname, col_pat in spec:
        p = RESULTS_DIR / fname
        if not p.exists():
            print(f"  skip {name}: {fname} not found")
            continue
        m = {}
        for r in csv.DictReader(p.open(encoding="utf-8")):
            sid = r["seg_id"]
            scores = {}
            for dk in DIM_KEYS:
                key = f"raw_{dk}" if col_pat == "raw_" else f"{dk}{col_pat}"
                if key in r:
                    scores[dk] = float(r[key])
            if scores:
                m[sid] = scores
        methods[name] = m
        print(f"  loaded {name}: {len(m)} segments")
    return methods


def aggregate_doc_vector(method_scores: dict[str, dict[str, float]],
                         segs_by_doc: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """For each doc, mean of segment scores per dimension."""
    out = {}
    for doc_id, sid_list in segs_by_doc.items():
        agg = {dk: 0.0 for dk in DIM_KEYS}
        n = 0
        for sid in sid_list:
            if sid in method_scores:
                for dk in DIM_KEYS:
                    agg[dk] += method_scores[sid].get(dk, 0.0)
                n += 1
        if n > 0:
            for dk in DIM_KEYS:
                agg[dk] /= n
        out[doc_id] = agg
    return out


def cosine_distance(v1: dict, v2: dict) -> float:
    keys = list(v1.keys())
    a = [v1[k] for k in keys]
    b = [v2[k] for k in keys]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


def find_pairs(segs):
    """Return (leader_change_pairs, same_leader_pairs) as lists of (doc_id_A, doc_id_B)."""
    doc_meta = {}
    for s in segs.values():
        if s["doc_id"] not in doc_meta:
            doc_meta[s["doc_id"]] = {
                "company_norm": s["company_norm"],
                "speaker": s["speaker"],
                "wave": s["wave"],
            }
    by_company = defaultdict(list)
    for did, m in doc_meta.items():
        by_company[m["company_norm"]].append((did, m))

    same_leader, leader_change = [], []
    for cn, docs in by_company.items():
        if len(docs) < 2:
            continue
        # pair every A doc with every B doc for that company
        a_docs = [d for d in docs if d[1]["wave"] == "A"]
        b_docs = [d for d in docs if d[1]["wave"] == "B"]
        for (a_id, a_m) in a_docs:
            for (b_id, b_m) in b_docs:
                spk_a = a_m["speaker"].strip()
                spk_b = b_m["speaker"].strip()
                if spk_a and spk_b and spk_a == spk_b:
                    same_leader.append((a_id, b_id))
                else:
                    leader_change.append((a_id, b_id))
    return leader_change, same_leader


# ──────────────────────────── Channel 1: paired contrast ────────────────────────────

def paired_contrast(method_doc_vecs: dict, leader_change, same_leader):
    d_lc = [cosine_distance(method_doc_vecs[a], method_doc_vecs[b]) for a, b in leader_change
            if a in method_doc_vecs and b in method_doc_vecs]
    d_sl = [cosine_distance(method_doc_vecs[a], method_doc_vecs[b]) for a, b in same_leader
            if a in method_doc_vecs and b in method_doc_vecs]
    if not d_lc or not d_sl:
        return None
    mean_lc = sum(d_lc) / len(d_lc)
    mean_sl = sum(d_sl) / len(d_sl)
    # Standard sample-size-weighted pooled SD for Cohen's d.
    var_lc = sum((x - mean_lc) ** 2 for x in d_lc) / max(1, len(d_lc) - 1)
    var_sl = sum((x - mean_sl) ** 2 for x in d_sl) / max(1, len(d_sl) - 1)
    pooled = math.sqrt(
        ((len(d_lc) - 1) * var_lc + (len(d_sl) - 1) * var_sl)
        / (len(d_lc) + len(d_sl) - 2)
    )
    cohen_d = (mean_lc - mean_sl) / pooled if pooled > 0 else 0.0
    return {
        "n_leader_change": len(d_lc),
        "n_same_leader": len(d_sl),
        "mean_distance_leader_change": round(mean_lc, 4),
        "mean_distance_same_leader": round(mean_sl, 4),
        "effect_diff": round(mean_lc - mean_sl, 4),
        "cohen_d": round(cohen_d, 3),
    }


# ──────────────────────────── Channel 2: gold L1 F1 ────────────────────────────

def infer_l1_for_baselines(scores: dict[str, float], threshold: float) -> str:
    """For dict/LDA/BGE: high max-dim → substantive; low max-dim → slogan."""
    max_score = max(scores.values()) if scores else 0.0
    return "substantive" if max_score >= threshold else "slogan"


def gold_l1_f1(methods: dict, gold_rows: list[dict]):
    """Binary F1: slogan vs substantive on gold-matched segments (drop irrelevant)."""
    gold = [r for r in gold_rows if r["seg_id"] and r["l1"] in ("slogan", "substantive")
            and float(r["match_score"]) >= 0.7]
    print(f"  gold subset: {len(gold)} segments (score≥0.7, l1 in slogan/substantive)")
    if not gold:
        return {}

    results = {}
    for method_name, scores_map in methods.items():
        if not scores_map:
            continue
        # Choose threshold = median of max-dim-score across all 2190 segs
        all_max = []
        for sid, sc in scores_map.items():
            if sc:
                all_max.append(max(sc.values()))
        all_max.sort()
        threshold = all_max[len(all_max) // 2] if all_max else 0.0

        tp = fp = fn = tn = 0
        for g in gold:
            sid = g["seg_id"]
            if sid not in scores_map:
                continue
            gt = g["l1"]
            if method_name in ("llm", "calibrated"):
                # use stored l1 directly (from scores_llm.csv)
                # We need to re-read from scores_llm.csv since methods only has dim scores
                pred = LLM_L1.get(sid, "substantive")
                if pred == "irrelevant":
                    pred = "slogan"  # collapse for binary
            else:
                pred = infer_l1_for_baselines(scores_map[sid], threshold)

            if gt == "substantive" and pred == "substantive": tp += 1
            elif gt == "substantive" and pred == "slogan": fn += 1
            elif gt == "slogan" and pred == "substantive": fp += 1
            else: tn += 1

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        results[method_name] = {
            "threshold": round(threshold, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision_substantive": round(prec, 3),
            "recall_substantive": round(rec, 3),
            "f1_substantive": round(f1, 3),
            "n": tp + fp + fn + tn,
        }
    return results


# Global cache for LLM l1
LLM_L1 = {}


def load_gold_rows():
    return load_majority_l1()


def main():
    segs = load_segments()
    print(f"Loaded {len(segs)} segments")

    # LLM l1 mapping (for gold F1)
    llm_path = RESULTS_DIR / "scores_llm.csv"
    if llm_path.exists():
        for r in csv.DictReader(llm_path.open(encoding="utf-8")):
            LLM_L1[r["seg_id"]] = r["l1"]
        print(f"Loaded {len(LLM_L1)} LLM l1 verdicts")

    methods = load_method_scores()

    # Per-doc aggregation
    segs_by_doc = defaultdict(list)
    for sid, s in segs.items():
        segs_by_doc[s["doc_id"]].append(sid)

    doc_vecs = {name: aggregate_doc_vector(sc, segs_by_doc) for name, sc in methods.items()}

    # Pairs
    leader_change, same_leader = find_pairs(segs)
    print(f"\nPairs: leader_change={len(leader_change)}, same_leader={len(same_leader)}")

    # ─── Channel 1 ───
    print("\n# Channel 1: Paired contrast (cosine distance)")
    ch1 = {}
    for name, vecs in doc_vecs.items():
        r = paired_contrast(vecs, leader_change, same_leader)
        if r:
            ch1[name] = r
            print(f"  {name}: D_lc={r['mean_distance_leader_change']}, "
                  f"D_sl={r['mean_distance_same_leader']}, "
                  f"diff={r['effect_diff']}, cohen_d={r['cohen_d']}")

    # ─── Channel 2 ───
    gold = load_gold_rows()
    print("\n# Channel 2: Gold L1 binary F1 (substantive vs slogan)")
    ch2 = gold_l1_f1(methods, gold)
    for name, r in ch2.items():
        print(f"  {name}: F1={r['f1_substantive']} (P={r['precision_substantive']}, "
              f"R={r['recall_substantive']}, n={r['n']})")

    # ─── Channel 3: external validity ───
    print("\n# Channel 3: External validity correlation (placeholder)")
    ext_path = DATA_DIR / "external_signals.csv"
    ch3 = {}
    if not ext_path.exists():
        print("  data/external_signals.csv not found — skipping. Add it later.")
    else:
        print("  TODO: implement company-year correlation (run after collecting signals)")

    # Dump summary
    out = RESULTS_DIR / "eval_summary.md"
    lines = ["# Phase 1 Evaluation Summary\n",
             f"Methods loaded: {', '.join(methods.keys())}",
             f"Segments: {len(segs)}",
             f"Leader-change pairs: {len(leader_change)}, Same-leader pairs: {len(same_leader)}",
             "",
             "## Channel 1 — Paired contrast\n",
             "| method | D(leader-change) | D(same-leader) | diff | Cohen d |",
             "|---|---|---|---|---|"]
    for name, r in ch1.items():
        lines.append(f"| {name} | {r['mean_distance_leader_change']} | "
                     f"{r['mean_distance_same_leader']} | "
                     f"{r['effect_diff']} | {r['cohen_d']} |")

    lines += ["", "## Channel 2 — Gold L1 F1 (substantive vs slogan)\n",
              "| method | F1(substantive) | Precision | Recall | n |",
              "|---|---|---|---|---|"]
    for name, r in ch2.items():
        lines.append(f"| {name} | {r['f1_substantive']} | "
                     f"{r['precision_substantive']} | {r['recall_substantive']} | {r['n']} |")

    lines += ["", "## Channel 3 — External validity\n",
              "TODO: collect data/external_signals.csv (company_norm, year, signal_name, value)",
              "      e.g. patent counts, ESG scores, R&D intensity per company-year."]

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
