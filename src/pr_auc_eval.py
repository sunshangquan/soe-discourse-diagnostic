"""Phase 2 — PR-AUC and ROC-AUC for the gold L1 evaluation.

The main paper uses a median-threshold rule for Dict/LDA/BGE to predict
SUBSTANTIVE vs SLOGAN, which fixes the predicted positive rate at 50%.
A threshold-free alternative is PR-AUC and ROC-AUC computed on the
maximum-dimension score across the 83 high-confidence gold paragraphs.

For LLM and Calibrated we don't have a continuous score on the same
axis (they output L1 directly), so we use:
  - LLM: 1 - slogan_density as the substantive score (continuous proxy)
  - Calibrated: max-dim of calibrated dimension scores

Output: results/pr_auc.csv
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

from dimensions import DIM_KEYS
from utils import DATA_DIR, RESULTS_DIR
from stats_addon import load_all


def rank_auc(scores, labels):
    """Compute ROC-AUC via Mann-Whitney U statistic.

    scores: list of floats
    labels: list of 0/1 (1 = positive class = substantive)
    """
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def pr_auc(scores, labels):
    """Compute average precision (PR-AUC)."""
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = 0
    fp = 0
    ap = 0.0
    pos_total = sum(labels)
    if pos_total == 0:
        return 0.0
    last_recall = 0.0
    for score, lbl in pairs:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / pos_total
        ap += precision * (recall - last_recall)
        last_recall = recall
    return ap


def main():
    seg_meta, methods, llm_l1, gold = load_all()
    # Keep only high-confidence matched gold with binary L1
    gold_use = [r for r in gold
                if r["seg_id"] and r["l1"] in ("slogan", "substantive")
                and float(r["match_score"]) >= 0.7]
    print(f"Gold subset: {len(gold_use)} paragraphs ({sum(1 for r in gold_use if r['l1']=='substantive')} sub)")

    # Compute LLM-derived continuous score: prefer 1 - slogan_density_llm
    llm_csv = RESULTS_DIR / "scores_llm.csv"
    llm_scores = {r["seg_id"]: r for r in csv.DictReader(llm_csv.open(encoding="utf-8"))}

    rows = []
    for method, sc in methods.items():
        scores = []
        labels = []
        for g in gold_use:
            sid = g["seg_id"]
            if sid not in sc:
                continue
            # Score: max-dim-score for all methods (continuous proxy for substantive)
            max_dim = max(sc[sid].values()) if sc[sid] else 0.0
            scores.append(max_dim)
            labels.append(1 if g["l1"] == "substantive" else 0)
        if not scores:
            continue
        roc = rank_auc(scores, labels)
        pr = pr_auc(scores, labels)
        rows.append({"method": method, "n": len(scores),
                     "ROC_AUC": round(roc, 3), "PR_AUC": round(pr, 3)})

    # Also evaluate LLM substantive score = 1 - slogan_density
    scores = []
    labels = []
    for g in gold_use:
        sid = g["seg_id"]
        if sid not in llm_scores:
            continue
        slogan_density = float(llm_scores[sid]["slogan_density"])
        confidence_sub = float(llm_scores[sid]["confidence_substantive"])
        # Use confidence_substantive directly — it's exactly P(substantive)
        scores.append(confidence_sub)
        labels.append(1 if g["l1"] == "substantive" else 0)
    if scores:
        roc = rank_auc(scores, labels)
        pr = pr_auc(scores, labels)
        rows.append({"method": "llm_csub", "n": len(scores),
                     "ROC_AUC": round(roc, 3), "PR_AUC": round(pr, 3)})

    out = RESULTS_DIR / "pr_auc.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'method':14s} {'n':>4s} {'ROC-AUC':>9s} {'PR-AUC':>9s}")
    for r in rows:
        print(f"{r['method']:14s} {r['n']:4d} {r['ROC_AUC']:9.3f} {r['PR_AUC']:9.3f}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
