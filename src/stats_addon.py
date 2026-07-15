"""Statistical add-ons for the paper:
  1. Bootstrap CI on Cohen d (paired contrast) and gold F1
  2. Lambda grid search for slogan calibration
  3. Per-method L1 confusion matrix on gold
  4. Industry-level paired stats (for discussion section)
"""
from __future__ import annotations
import csv
import math
import random
from collections import defaultdict, Counter
from pathlib import Path

from dimensions import DIM_KEYS
from utils import DATA_DIR, RESULTS_DIR, PHASE0_DIR

random.seed(42)


def load_all():
    segs = list(csv.DictReader((DATA_DIR / "segments.csv").open(encoding="utf-8")))
    seg_meta = {s["seg_id"]: s for s in segs}

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
            continue
        m = {}
        for r in csv.DictReader(p.open(encoding="utf-8")):
            scores = {}
            for dk in DIM_KEYS:
                key = f"raw_{dk}" if col_pat == "raw_" else f"{dk}{col_pat}"
                if key in r:
                    scores[dk] = float(r[key])
            if scores:
                m[r["seg_id"]] = scores
        methods[name] = m

    llm_l1 = {}
    p = RESULTS_DIR / "scores_llm.csv"
    if p.exists():
        for r in csv.DictReader(p.open(encoding="utf-8")):
            llm_l1[r["seg_id"]] = r["l1"]

    gold = list(csv.DictReader((DATA_DIR / "gold.csv").open(encoding="utf-8")))
    return seg_meta, methods, llm_l1, gold


def doc_vectors(method_scores, seg_meta):
    by_doc = defaultdict(list)
    for sid, s in seg_meta.items():
        by_doc[s["doc_id"]].append(sid)
    out = {}
    for did, sids in by_doc.items():
        agg = {dk: 0.0 for dk in DIM_KEYS}
        n = 0
        for sid in sids:
            if sid in method_scores:
                for dk in DIM_KEYS:
                    agg[dk] += method_scores[sid].get(dk, 0.0)
                n += 1
        if n > 0:
            for dk in DIM_KEYS:
                agg[dk] /= n
        out[did] = agg
    return out


def cosine_distance(v1, v2):
    keys = list(v1.keys())
    a = [v1[k] for k in keys]
    b = [v2[k] for k in keys]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / (na * nb) if (na > 0 and nb > 0) else 1.0


def find_pairs(seg_meta):
    doc_meta = {}
    for s in seg_meta.values():
        if s["doc_id"] not in doc_meta:
            doc_meta[s["doc_id"]] = {
                "company_norm": s["company_norm"],
                "speaker": s["speaker"],
                "wave": s["wave"],
                "industry": s["industry"],
            }
    by_company = defaultdict(list)
    for did, m in doc_meta.items():
        by_company[m["company_norm"]].append((did, m))

    leader_change, same_leader = [], []
    for cn, docs in by_company.items():
        a_docs = [d for d in docs if d[1]["wave"] == "A"]
        b_docs = [d for d in docs if d[1]["wave"] == "B"]
        for (a_id, a_m) in a_docs:
            for (b_id, b_m) in b_docs:
                spk_a, spk_b = a_m["speaker"].strip(), b_m["speaker"].strip()
                if spk_a and spk_b and spk_a == spk_b:
                    same_leader.append((a_id, b_id))
                else:
                    leader_change.append((a_id, b_id))
    return leader_change, same_leader


# ─── 1. Bootstrap CI on Cohen d ───

def cohen_d(d_lc, d_sl):
    if not d_lc or not d_sl:
        return 0.0
    m_lc = sum(d_lc) / len(d_lc)
    m_sl = sum(d_sl) / len(d_sl)
    var_lc = sum((x - m_lc) ** 2 for x in d_lc) / max(1, len(d_lc) - 1)
    var_sl = sum((x - m_sl) ** 2 for x in d_sl) / max(1, len(d_sl) - 1)
    pooled = math.sqrt((var_lc + var_sl) / 2)
    return (m_lc - m_sl) / pooled if pooled > 0 else 0.0


def bootstrap_cohen_d(d_lc, d_sl, n_boot=2000, alpha=0.05):
    boots = []
    for _ in range(n_boot):
        b_lc = [random.choice(d_lc) for _ in range(len(d_lc))]
        b_sl = [random.choice(d_sl) for _ in range(len(d_sl))]
        boots.append(cohen_d(b_lc, b_sl))
    boots.sort()
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    return lo, hi


def channel1_with_ci(methods, seg_meta, leader_change, same_leader):
    rows = []
    for name, sc in methods.items():
        dv = doc_vectors(sc, seg_meta)
        d_lc = [cosine_distance(dv[a], dv[b]) for a, b in leader_change if a in dv and b in dv]
        d_sl = [cosine_distance(dv[a], dv[b]) for a, b in same_leader if a in dv and b in dv]
        if not d_lc or not d_sl:
            continue
        d = cohen_d(d_lc, d_sl)
        lo, hi = bootstrap_cohen_d(d_lc, d_sl)
        rows.append({
            "method": name,
            "n_lc": len(d_lc), "n_sl": len(d_sl),
            "mean_d_lc": round(sum(d_lc) / len(d_lc), 4),
            "mean_d_sl": round(sum(d_sl) / len(d_sl), 4),
            "diff": round(sum(d_lc) / len(d_lc) - sum(d_sl) / len(d_sl), 4),
            "cohen_d": round(d, 3),
            "cohen_d_ci_lo": round(lo, 3),
            "cohen_d_ci_hi": round(hi, 3),
        })
    return rows


# ─── 2. Gold L1 F1 with bootstrap CI ───

def f1_from_counts(tp, fp, fn):
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return 2 * prec * rec / max(1e-9, prec + rec), prec, rec


def gold_f1_with_ci(methods, llm_l1, gold, n_boot=2000):
    gold_use = [r for r in gold if r["seg_id"] and r["l1"] in ("slogan", "substantive")
                and float(r["match_score"]) >= 0.7]
    rows = []
    for name, scores in methods.items():
        # Choose threshold = median of max-dim-score
        all_max = sorted([max(sc.values()) for sc in scores.values() if sc])
        thresh = all_max[len(all_max) // 2] if all_max else 0.0
        preds = {}
        for g in gold_use:
            sid = g["seg_id"]
            if sid not in scores:
                continue
            if name in ("llm", "calibrated"):
                p = llm_l1.get(sid, "substantive")
                p = "slogan" if p == "irrelevant" else p
            else:
                p = "substantive" if max(scores[sid].values()) >= thresh else "slogan"
            preds[sid] = p

        # Point estimate
        tp = sum(1 for g in gold_use if g["seg_id"] in preds
                 and g["l1"] == "substantive" and preds[g["seg_id"]] == "substantive")
        fp = sum(1 for g in gold_use if g["seg_id"] in preds
                 and g["l1"] == "slogan" and preds[g["seg_id"]] == "substantive")
        fn = sum(1 for g in gold_use if g["seg_id"] in preds
                 and g["l1"] == "substantive" and preds[g["seg_id"]] == "slogan")
        tn = sum(1 for g in gold_use if g["seg_id"] in preds
                 and g["l1"] == "slogan" and preds[g["seg_id"]] == "slogan")
        f1_pt, p_pt, r_pt = f1_from_counts(tp, fp, fn)

        # Bootstrap on the gold set
        items = [(g["seg_id"], g["l1"], preds.get(g["seg_id"])) for g in gold_use if g["seg_id"] in preds]
        f1_boots = []
        for _ in range(n_boot):
            samp = [random.choice(items) for _ in items]
            tp_b = sum(1 for _, gt, pr in samp if gt == "substantive" and pr == "substantive")
            fp_b = sum(1 for _, gt, pr in samp if gt == "slogan" and pr == "substantive")
            fn_b = sum(1 for _, gt, pr in samp if gt == "substantive" and pr == "slogan")
            f1_b, _, _ = f1_from_counts(tp_b, fp_b, fn_b)
            f1_boots.append(f1_b)
        f1_boots.sort()
        lo = f1_boots[int(0.025 * n_boot)]
        hi = f1_boots[int(0.975 * n_boot)]

        rows.append({
            "method": name, "n": len(items),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision_substantive": round(p_pt, 3),
            "recall_substantive": round(r_pt, 3),
            "f1_substantive": round(f1_pt, 3),
            "f1_ci_lo": round(lo, 3),
            "f1_ci_hi": round(hi, 3),
        })
    return rows


# ─── 3. Lambda grid search ───

def grid_search_lambda(seg_meta, leader_change, same_leader, llm_path, ngram_path,
                       lam_llm_vals, lam_ng_vals):
    """Re-compute calibration with each (λ_llm, λ_ngram) and report paired-contrast Cohen d."""
    llm = {r["seg_id"]: r for r in csv.DictReader(llm_path.open(encoding="utf-8"))}
    ng = {r["seg_id"]: float(r["ngram_slogan_density"])
          for r in csv.DictReader(ngram_path.open(encoding="utf-8"))}
    out = []
    for lam_llm in lam_llm_vals:
        for lam_ng in lam_ng_vals:
            scores = {}
            for sid, l in llm.items():
                sd_llm = float(l["slogan_density"])
                sd_ng = ng.get(sid, 0.0)
                conf = float(l["confidence_substantive"])
                mult = max(0.0, 1 - lam_llm * sd_llm) * max(0.0, 1 - lam_ng * sd_ng) * conf
                scores[sid] = {dk: float(l[f"raw_{dk}"]) * mult for dk in DIM_KEYS}
            dv = doc_vectors(scores, seg_meta)
            d_lc = [cosine_distance(dv[a], dv[b]) for a, b in leader_change if a in dv and b in dv]
            d_sl = [cosine_distance(dv[a], dv[b]) for a, b in same_leader if a in dv and b in dv]
            d = cohen_d(d_lc, d_sl)
            out.append({
                "lambda_llm": lam_llm,
                "lambda_ngram": lam_ng,
                "mean_d_lc": round(sum(d_lc) / len(d_lc), 4),
                "mean_d_sl": round(sum(d_sl) / len(d_sl), 4),
                "diff": round(sum(d_lc) / len(d_lc) - sum(d_sl) / len(d_sl), 4),
                "cohen_d": round(d, 3),
            })
    return out


# ─── 4. Per-method confusion matrix on gold ───

def confusion_matrix(methods, llm_l1, gold):
    gold_use = [r for r in gold if r["seg_id"] and r["l1"] in ("slogan", "substantive")
                and float(r["match_score"]) >= 0.7]
    rows = []
    for name, scores in methods.items():
        all_max = sorted([max(sc.values()) for sc in scores.values() if sc])
        thresh = all_max[len(all_max) // 2] if all_max else 0.0
        tp = fp = fn = tn = 0
        for g in gold_use:
            sid = g["seg_id"]
            if sid not in scores:
                continue
            if name in ("llm", "calibrated"):
                p = llm_l1.get(sid, "substantive")
                p = "slogan" if p == "irrelevant" else p
            else:
                p = "substantive" if max(scores[sid].values()) >= thresh else "slogan"
            gt = g["l1"]
            if gt == "substantive" and p == "substantive": tp += 1
            elif gt == "slogan" and p == "slogan": tn += 1
            elif gt == "substantive" and p == "slogan": fn += 1
            elif gt == "slogan" and p == "substantive": fp += 1
        rows.append({"method": name, "TP": tp, "FN": fn, "FP": fp, "TN": tn})
    return rows


# ─── main ───

def main():
    seg_meta, methods, llm_l1, gold = load_all()
    leader_change, same_leader = find_pairs(seg_meta)
    print(f"Pairs: leader_change={len(leader_change)}, same_leader={len(same_leader)}")

    # 1. Channel 1 with CI
    print("\n## Channel 1 — Paired contrast with bootstrap CI")
    ch1 = channel1_with_ci(methods, seg_meta, leader_change, same_leader)
    print(f"{'method':12s} {'D_lc':>8s} {'D_sl':>8s} {'diff':>8s} {'d':>6s} {'95% CI':>14s}")
    for r in ch1:
        print(f"{r['method']:12s} {r['mean_d_lc']:>8.4f} {r['mean_d_sl']:>8.4f} "
              f"{r['diff']:>8.4f} {r['cohen_d']:>6.2f}  [{r['cohen_d_ci_lo']:+.2f},{r['cohen_d_ci_hi']:+.2f}]")

    # 2. Gold F1 with CI
    print("\n## Channel 2 — Gold F1 with bootstrap CI")
    ch2 = gold_f1_with_ci(methods, llm_l1, gold)
    print(f"{'method':12s} {'F1':>6s} {'95% CI':>14s} {'P':>6s} {'R':>6s} {'n':>4s}")
    for r in ch2:
        print(f"{r['method']:12s} {r['f1_substantive']:>6.3f} "
              f"[{r['f1_ci_lo']:.2f},{r['f1_ci_hi']:.2f}] "
              f"{r['precision_substantive']:>6.3f} {r['recall_substantive']:>6.3f} "
              f"{r['n']:>4d}")

    # 3. Confusion matrix
    print("\n## Confusion matrix (slogan/substantive)")
    cm = confusion_matrix(methods, llm_l1, gold)
    print(f"{'method':12s} {'TP':>4s} {'FN':>4s} {'FP':>4s} {'TN':>4s}")
    for r in cm:
        print(f"{r['method']:12s} {r['TP']:>4d} {r['FN']:>4d} {r['FP']:>4d} {r['TN']:>4d}")

    # 4. Lambda grid search
    print("\n## Lambda grid search (paired-contrast Cohen d)")
    grid = grid_search_lambda(
        seg_meta, leader_change, same_leader,
        RESULTS_DIR / "scores_llm.csv",
        RESULTS_DIR / "slogan_density_ngram.csv",
        lam_llm_vals=[0.0, 0.5, 1.0, 1.5, 2.0],
        lam_ng_vals=[0.0, 1.0, 2.0, 3.0],
    )
    grid.sort(key=lambda r: -r["cohen_d"])
    print(f"{'λ_llm':>6s} {'λ_ng':>6s} {'D_lc':>8s} {'D_sl':>8s} {'diff':>8s} {'d':>6s}")
    for r in grid[:10]:
        print(f"{r['lambda_llm']:>6.1f} {r['lambda_ngram']:>6.1f} "
              f"{r['mean_d_lc']:>8.4f} {r['mean_d_sl']:>8.4f} "
              f"{r['diff']:>8.4f} {r['cohen_d']:>6.2f}")
    print(f"...({len(grid) - 10} more)")

    # Save all to results
    with (RESULTS_DIR / "stats_addon.csv").open("w", encoding="utf-8", newline="") as f:
        # write all as one TSV with section headers
        pass  # we'll dump structured below

    import json
    out = {
        "channel1_paired_contrast": ch1,
        "channel2_gold_f1": ch2,
        "confusion_matrix": cm,
        "lambda_grid": grid,
        "n_leader_change_pairs": len(leader_change),
        "n_same_leader_pairs": len(same_leader),
    }
    with (RESULTS_DIR / "stats_addon.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {RESULTS_DIR / 'stats_addon.json'}")


if __name__ == "__main__":
    main()
