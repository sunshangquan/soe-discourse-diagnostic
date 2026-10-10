"""Incremental-validity check for pair-level LLM distances.

Question: after controlling for generic lexical distance (TF-IDF) and
style distance, do LLM document-vector distances still separate
leader-change (LC) from same-leader (SL) pairs?

This is a small diagnostic, not a confirmatory model: the sample has
only 29 pairs.  We fit OLS on all pairs without the LC/SL label, then
compare residuals by pair label with the same exact permutation protocol
used elsewhere in the paper.

Output:
  results/incremental_validity.csv
  results/incremental_validity.md
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from evaluate import aggregate_doc_vector, cosine_distance, find_pairs, load_method_scores, load_segments
from rebuttal_controls import (
    build_doc_texts,
    cosine_distance_dense,
    exact_permutation_pvalue,
    tfidf_doc_vectors,
)
from stats_addon import cohen_d
from style_baseline import cosine_dist, doc_style_features, z_normalize
from utils import CORPUS_DIR, RESULTS_DIR, read_text


def ols_coefficients(x_rows: list[list[float]], y: list[float]) -> list[float]:
    """Fit y ~ 1 + X using a small Gauss-Jordan OLS solver."""
    n = len(y)
    if n != len(x_rows) or not y:
        raise ValueError("x_rows and y must be non-empty and have equal length")
    x1 = [[1.0] + row for row in x_rows]
    p = len(x1[0])
    xtx = [[sum(x1[k][i] * x1[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    xty = [sum(x1[k][i] * y[k] for k in range(n)) for i in range(p)]
    aug = [xtx[i] + [xty[i]] for i in range(p)]
    for i in range(p):
        pivot = max(range(i, p), key=lambda r: abs(aug[r][i]))
        aug[i], aug[pivot] = aug[pivot], aug[i]
        if abs(aug[i][i]) < 1e-12:
            raise RuntimeError("Singular OLS design matrix")
        div = aug[i][i]
        aug[i] = [v / div for v in aug[i]]
        for r in range(p):
            if r == i:
                continue
            factor = aug[r][i]
            aug[r] = [aug[r][c] - factor * aug[i][c] for c in range(p + 1)]
    return [aug[i][p] for i in range(p)]


def predict_row(beta: list[float], x_row: list[float]) -> float:
    return beta[0] + sum(coef * value for coef, value in zip(beta[1:], x_row))


def ols_residuals(x_rows: list[list[float]], y: list[float]) -> list[float]:
    beta = ols_coefficients(x_rows, y)
    return [target - predict_row(beta, row) for row, target in zip(x_rows, y)]


def cross_fitted_residuals(x_rows: list[list[float]], y: list[float]) -> list[float]:
    """Leave-one-pair-out residuals, so each prediction is out of sample."""
    if len(y) <= len(x_rows[0]) + 1:
        raise ValueError("not enough observations for leave-one-out OLS")
    residuals = []
    for held_out in range(len(y)):
        train_x = [row for i, row in enumerate(x_rows) if i != held_out]
        train_y = [value for i, value in enumerate(y) if i != held_out]
        beta = ols_coefficients(train_x, train_y)
        residuals.append(y[held_out] - predict_row(beta, x_rows[held_out]))
    return residuals


def pearson_r(a: list[float], b: list[float]) -> float:
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denominator = math.sqrt(
        sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)
    )
    return numerator / denominator if denominator else 0.0


def standardize(values: list[float]) -> list[float]:
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    sd = math.sqrt(variance)
    return [(value - mean_value) / sd for value in values] if sd else [0.0] * len(values)


def regression_diagnostics(
    x_rows: list[list[float]], y: list[float], predictor_names: list[str]
) -> dict[str, float]:
    """Return in-sample fit, correlations, and standardized coefficients."""
    beta = ols_coefficients(x_rows, y)
    predictions = [predict_row(beta, row) for row in x_rows]
    mean_y = sum(y) / len(y)
    total_ss = sum((value - mean_y) ** 2 for value in y)
    residual_ss = sum((target - pred) ** 2 for target, pred in zip(y, predictions))
    result = {"r_squared": 1.0 - residual_ss / total_ss if total_ss else 0.0}

    columns = [[row[j] for row in x_rows] for j in range(len(predictor_names))]
    standardized_x = [standardize(column) for column in columns]
    standardized_y = standardize(y)
    standardized_rows = [
        [standardized_x[j][i] for j in range(len(predictor_names))]
        for i in range(len(y))
    ]
    standardized_beta = ols_coefficients(standardized_rows, standardized_y)
    for j, name in enumerate(predictor_names):
        result[f"pearson_r_{name}_y"] = pearson_r(columns[j], y)
        result[f"standardized_beta_{name}"] = standardized_beta[j + 1]
    return result


def load_or_compute_style_features(
    docs: list[str], corpus_dir: Path, cache_path: Path
) -> dict[str, dict[str, float]]:
    """Use raw speeches when available; otherwise load the release cache."""
    if all((corpus_dir / doc_id).exists() for doc_id in docs):
        features = {
            doc_id: doc_style_features(read_text(corpus_dir / doc_id))
            for doc_id in docs
        }
        feature_names = list(next(iter(features.values())))
        with cache_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["doc_id", *feature_names])
            writer.writeheader()
            for doc_id in docs:
                writer.writerow({"doc_id": doc_id, **features[doc_id]})
        return features

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Neither raw speeches nor cached style features are available: {cache_path}"
        )
    features = {}
    with cache_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            features[row["doc_id"]] = {
                key: float(value) for key, value in row.items() if key != "doc_id"
            }
    missing = [doc_id for doc_id in docs if doc_id not in features]
    if missing:
        raise ValueError(f"Style-feature cache is missing {len(missing)} documents")
    return {doc_id: features[doc_id] for doc_id in docs}


def pair_stats(lc_values: list[float], sl_values: list[float]) -> dict[str, float | int]:
    mean_lc = sum(lc_values) / len(lc_values)
    mean_sl = sum(sl_values) / len(sl_values)
    p2, p1, n_perm = exact_permutation_pvalue(lc_values, sl_values)
    return {
        "D_lc": mean_lc,
        "D_sl": mean_sl,
        "Delta": mean_lc - mean_sl,
        "Cohen_d": cohen_d(lc_values, sl_values),
        "perm_p_2sided": p2,
        "perm_p_1sided": p1,
        "n_perm": n_perm,
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    segs = load_segments()
    methods = load_method_scores()
    leader_change, same_leader = find_pairs(segs)
    all_pairs = [("LC", a, b) for a, b in leader_change] + [("SL", a, b) for a, b in same_leader]

    segs_by_doc: dict[str, list[str]] = {}
    for sid, row in segs.items():
        segs_by_doc.setdefault(row["doc_id"], []).append(sid)

    llm_doc = aggregate_doc_vector(methods["llm"], segs_by_doc)
    tfidf_doc = tfidf_doc_vectors(build_doc_texts(segs))

    docs = sorted({row["doc_id"] for row in segs.values()})
    style_raw = load_or_compute_style_features(
        docs, CORPUS_DIR, RESULTS_DIR / "style_doc_features.csv"
    )
    style_doc = z_normalize(style_raw)

    labels: list[str] = []
    llm_dist: list[float] = []
    tfidf_dist: list[float] = []
    style_dist: list[float] = []
    for label, a, b in all_pairs:
        if a not in llm_doc or b not in llm_doc or a not in tfidf_doc or b not in tfidf_doc:
            continue
        labels.append(label)
        llm_dist.append(cosine_distance(llm_doc[a], llm_doc[b]))
        tfidf_dist.append(cosine_distance_dense(tfidf_doc[a], tfidf_doc[b]))
        style_dist.append(cosine_dist(style_doc[a], style_doc[b]))

    tfidf_rows = [[v] for v in tfidf_dist]
    style_rows = [[v] for v in style_dist]
    combined_rows = [[a, b] for a, b in zip(tfidf_dist, style_dist)]
    specs = [
        ("raw_llm_pair_distance", llm_dist),
        ("llm_residualized_on_tfidf", ols_residuals(tfidf_rows, llm_dist)),
        ("llm_residualized_on_style", ols_residuals(style_rows, llm_dist)),
        (
            "llm_residualized_on_tfidf_and_style",
            ols_residuals(combined_rows, llm_dist),
        ),
        ("llm_lopo_residualized_on_tfidf", cross_fitted_residuals(tfidf_rows, llm_dist)),
        ("llm_lopo_residualized_on_style", cross_fitted_residuals(style_rows, llm_dist)),
        (
            "llm_lopo_residualized_on_tfidf_and_style",
            cross_fitted_residuals(combined_rows, llm_dist),
        ),
    ]

    diagnostic_specs = [
        ("tfidf", tfidf_rows, ["tfidf"]),
        ("style", style_rows, ["style"]),
        ("tfidf_and_style", combined_rows, ["tfidf", "style"]),
    ]

    out_rows = []
    for name, values in specs:
        lc_values = [v for label, v in zip(labels, values) if label == "LC"]
        sl_values = [v for label, v in zip(labels, values) if label == "SL"]
        stat = pair_stats(lc_values, sl_values)
        out_rows.append({
            "method": name,
            "n_lc": len(lc_values),
            "n_sl": len(sl_values),
            **{k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in stat.items()},
        })

    out_csv = RESULTS_DIR / "incremental_validity.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    diagnostics_rows = []
    for name, rows, predictor_names in diagnostic_specs:
        diagnostics_rows.append({
            "model": name,
            **{
                key: f"{value:.6f}"
                for key, value in regression_diagnostics(rows, llm_dist, predictor_names).items()
            },
        })
    diagnostic_keys = sorted({key for row in diagnostics_rows for key in row})
    diagnostics_csv = RESULTS_DIR / "incremental_validity_diagnostics.csv"
    with diagnostics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=diagnostic_keys)
        writer.writeheader()
        writer.writerows(diagnostics_rows)

    out_md = RESULTS_DIR / "incremental_validity.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Incremental-validity diagnostic\n\n")
        f.write("Pair-level LLM cosine distance residualized on TF-IDF and/or pure style distance.\n\n")
        f.write("| Method | Delta | Cohen d | exact p1 |\n")
        f.write("|---|---:|---:|---:|\n")
        for r in out_rows:
            f.write(f"| {r['method']} | {r['Delta']} | {r['Cohen_d']} | {r['perm_p_1sided']} |\n")

        f.write("\n## Regression diagnostics\n\n")
        for row in diagnostics_rows:
            f.write(f"- {row}\n")

    print(out_csv)
    print(diagnostics_csv)
    print(out_md)
    for r in out_rows:
        print(r)


if __name__ == "__main__":
    main()
