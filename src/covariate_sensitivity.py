"""Sensitivity of the paired screen to the SASAC regulator control pair.

Industry matching is not reported because three of five SL pairs have no
industry tag and the two tagged industries leave too few comparable pairs.
"""
from __future__ import annotations

import csv

from evaluate import aggregate_doc_vector, cosine_distance, find_pairs, load_method_scores, load_segments
from incremental_validity import pair_stats
from rebuttal_controls import build_doc_texts, cosine_distance_dense, tfidf_doc_vectors
from utils import RESULTS_DIR


def exclude_pairs(pairs: list[tuple[str, str]], excluded_docs: set[str]) -> list[tuple[str, str]]:
    return [pair for pair in pairs if not any(doc in excluded_docs for doc in pair)]


def distances_sparse(doc_vectors, pairs):
    return [cosine_distance(doc_vectors[a], doc_vectors[b]) for a, b in pairs]


def distances_dense(doc_vectors, pairs):
    return [cosine_distance_dense(doc_vectors[a], doc_vectors[b]) for a, b in pairs]


def main() -> None:
    segs = load_segments()
    methods = load_method_scores()
    leader_change, same_leader = find_pairs(segs)
    segs_by_doc: dict[str, list[str]] = {}
    regulator_docs = set()
    for seg_id, row in segs.items():
        segs_by_doc.setdefault(row["doc_id"], []).append(seg_id)
        if "国资委" in row.get("company", "") or "国资委" in row.get("company_norm", ""):
            regulator_docs.add(row["doc_id"])

    filtered_sl = exclude_pairs(same_leader, regulator_docs)
    method_vectors = {
        "llm": (aggregate_doc_vector(methods["llm"], segs_by_doc), distances_sparse),
        "lda": (aggregate_doc_vector(methods["lda"], segs_by_doc), distances_sparse),
        "tfidf": (tfidf_doc_vectors(build_doc_texts(segs)), distances_dense),
    }

    rows = []
    for method, (doc_vectors, distance_fn) in method_vectors.items():
        for analysis, sl_pairs in [("full", same_leader), ("exclude_sasac", filtered_sl)]:
            stat = pair_stats(
                distance_fn(doc_vectors, leader_change),
                distance_fn(doc_vectors, sl_pairs),
            )
            rows.append({
                "method": method,
                "analysis": analysis,
                "n_lc": len(leader_change),
                "n_sl": len(sl_pairs),
                **{key: f"{value:.6f}" if isinstance(value, float) else value for key, value in stat.items()},
            })

    out_csv = RESULTS_DIR / "covariate_sensitivity.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    out_md = RESULTS_DIR / "covariate_sensitivity.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Covariate sensitivity\n\n")
        f.write("The only regulator pair (SASAC) is removed from the SL group. ")
        f.write("Industry matching is not estimated because 3/5 SL pairs lack an industry tag.\n\n")
        f.write("| method | analysis | n LC | n SL | Delta | Cohen d | exact p1 |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['method']} | {row['analysis']} | {row['n_lc']} | {row['n_sl']} | "
                f"{row['Delta']} | {row['Cohen_d']} | {row['perm_p_1sided']} |\n"
            )

    print(out_csv)
    print(out_md)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
