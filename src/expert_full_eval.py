"""Evaluate saved document scores against each expert's full-corpus labels."""
import csv
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score

from stats_addon import cosine_distance, doc_vectors, find_pairs, load_all

ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = ("innovation", "competition_cooperation", "organization_market",
              "social_responsibility", "national_mission")
LABEL_DIMENSIONS = ("innovation", "competition_cooperation", "organisation_market",
                    "social_responsibility", "national_mission")


def main():
    csv.field_size_limit(10_000_000)
    segments, methods, _, _ = load_all()
    if len(segments) != 2190 or set(methods) != {"dict", "lda", "bge", "llm", "calibrated"}:
        raise ValueError("Unexpected segment or method count")
    vectors = {}
    for method, scores in methods.items():
        if set(scores) != set(segments) or any(set(v) != set(DIMENSIONS) for v in scores.values()):
            raise ValueError(f"Incomplete score file: {method}")
        vectors[method] = doc_vectors(scores, segments)
        if len(vectors[method]) != 80:
            raise ValueError(f"Unexpected document count: {method}")
    lc, sl = find_pairs(segments)
    raw = vectors["llm"]
    mean = lambda pairs: sum(cosine_distance(raw[a], raw[b]) for a, b in pairs) / len(pairs)
    delta = mean(lc) - mean(sl)
    if len(lc) != 24 or len(sl) != 5 or abs(delta - .104704) > .000001:
        raise ValueError(f"Reference aggregation mismatch: {delta}")
    print(f"Reference raw LLM delta: {delta:.6f}")

    labels = defaultdict(dict)
    with (ROOT / "data/expert_labels.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        group = (row["expert"], row["dimension"])
        doc = row["doc_id"]
        if doc in labels[group]:
            raise ValueError(f"Duplicate label: {group}, {doc}")
        labels[group][doc] = row["judgement"].strip()
    if len(rows) != 800 or len(labels) != 10:
        raise ValueError("Expected 800 expert judgments in 10 groups")
    for expert in ("1", "2"):
        for dim, label_dim in zip(DIMENSIONS, LABEL_DIMENSIONS):
            group = labels[(expert, label_dim)]
            docs = sorted(vectors["llm"])
            if set(group) != set(docs) or not set(group.values()) <= {"明确立场", "仅涉及", "未涉及"}:
                raise ValueError(f"Missing or uncertain labels: {expert}, {label_dim}")
            y = [int(group[doc] == "明确立场") for doc in docs]
            if len(set(y)) != 2:
                raise ValueError(f"Single class: {expert}, {label_dim}")
            for method, vec in vectors.items():
                scores = [vec[doc][dim] for doc in docs]
                print(f"expert={expert} dimension={dim} method={method} n={len(y)} "
                      f"positives={sum(y)} AP={average_precision_score(y, scores):.4f} "
                      f"ROC_AUC={roc_auc_score(y, scores):.4f}")


if __name__ == "__main__":
    main()
