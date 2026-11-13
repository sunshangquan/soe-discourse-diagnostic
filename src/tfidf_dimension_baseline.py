"""Compare TF-IDF to the BGE anchors on all scored segments and expert labels."""
import csv
from pathlib import Path

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score

from dimensions import DIMENSIONS, DIM_KEYS
from stats_addon import doc_vectors, load_all

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/expert_labels.csv"
OUTPUT = ROOT / "results/tfidf_dimension_baseline_recomputed.csv"
LABEL_KEYS = ("innovation", "competition_cooperation", "organisation_market",
              "social_responsibility", "national_mission")


def main():
    if OUTPUT.exists():
        raise FileExistsError("Existing result preserved; do not rerun without checking it")
    meta, methods, _, _ = load_all()
    if len(meta) != 2190 or len({row["doc_id"] for row in meta.values()}) != 80:
        raise ValueError("Unexpected corpus coverage")
    if set(methods["bge"]) != set(meta):
        raise ValueError("BGE coverage differs")
    ids = list(meta)
    anchors = [d["nl_description"] for d in DIMENSIONS]
    texts = [meta[sid]["text"] for sid in ids]
    vectorizer = TfidfVectorizer(tokenizer=jieba.lcut, token_pattern=None,
                                 lowercase=False, norm="l2")
    matrix = vectorizer.fit_transform(texts + anchors)
    similarities = (matrix[:len(ids)] @ matrix[len(ids):].T).toarray()
    scores = {sid: {dim: float(similarities[i, j]) for j, dim in enumerate(DIM_KEYS)}
              for i, sid in enumerate(ids)}
    tfidf_docs = doc_vectors(scores, meta)
    bge_docs = doc_vectors(methods["bge"], meta)
    with LABELS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 800:
        raise ValueError("Unexpected expert label count")
    results = []
    for expert in ("1", "2"):
        for key, label_key in zip(DIM_KEYS, LABEL_KEYS):
            selected = [r for r in rows if r["expert"] == expert and r["dimension"] == label_key]
            labels = {r["doc_id"]: r["judgement"] for r in selected}
            if len(selected) != 80 or set(labels) != set(tfidf_docs):
                raise ValueError("Expert documents do not match full scored corpus")
            if not set(labels.values()) <= {"明确立场", "仅涉及", "未涉及"}:
                raise ValueError("Unexpected or uncertain label; do not filter")
            docs = sorted(labels)
            y = [int(labels[doc] == "明确立场") for doc in docs]
            for name, vectors in (("tfidf_anchor", tfidf_docs), ("bge_reference", bge_docs)):
                x = [vectors[doc][key] for doc in docs]
                results.append({"expert": expert, "dimension": key, "method": name,
                                "n": len(y), "positive": sum(y),
                                "ap": f"{average_precision_score(y, x):.6f}",
                                "roc_auc": f"{roc_auc_score(y, x):.6f}"})
    with OUTPUT.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} results for 80 documents / 2190 segments: {OUTPUT}")


if __name__ == "__main__":
    main()
