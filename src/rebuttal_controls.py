"""Reviewer-requested controls for the ARR rebuttal.

Controls implemented:
  1. TF-IDF document vectors: generic lexical document/speaker signal.
  2. Sanitized TF-IDF variants: remove names/years/numbers/slogan terms.
  3. Doc2Vec document vectors across multiple random seeds.
  4. DICT restricted to LLM-substantive segments under two aggregation
     semantics: conditional profile vs. document-level mass.
  5. LLM cache parse-failure accounting.

Outputs:
  results/rebuttal_controls.csv
  results/rebuttal_controls.md
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import jieba
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer

from dimensions import DIM_KEYS
from evaluate import (
    aggregate_doc_vector,
    cosine_distance,
    find_pairs,
    load_segments,
)
from utils import DATA_DIR, RESULTS_DIR


SLOGAN_TERMS = [
    "习近平", "党中央", "国务院", "中央", "总书记", "新时代", "中国特色",
    "中国特色社会主义", "社会主义", "马克思", "党的领导", "党的建设",
    "两个一百年", "中华民族伟大复兴", "中国梦", "不忘初心", "牢记使命",
    "四个意识", "四个自信", "两个维护", "五位一体", "四个全面",
    "一带一路", "人类命运共同体", "绿水青山", "金山银山", "高质量发展",
    "新发展理念", "稳中求进", "国有企业", "国有经济", "中央企业",
    "做强做优做大", "主责主业", "改革发展", "政治责任", "社会责任",
]


def cosine_distance_dense(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


def paired_contrast_dense(doc_vecs: dict[str, list[float]], leader_change, same_leader):
    d_lc = [
        cosine_distance_dense(doc_vecs[a], doc_vecs[b])
        for a, b in leader_change
        if a in doc_vecs and b in doc_vecs
    ]
    d_sl = [
        cosine_distance_dense(doc_vecs[a], doc_vecs[b])
        for a, b in same_leader
        if a in doc_vecs and b in doc_vecs
    ]
    if not d_lc or not d_sl:
        return None
    mean_lc = sum(d_lc) / len(d_lc)
    mean_sl = sum(d_sl) / len(d_sl)
    var_lc = sum((x - mean_lc) ** 2 for x in d_lc) / max(1, len(d_lc) - 1)
    var_sl = sum((x - mean_sl) ** 2 for x in d_sl) / max(1, len(d_sl) - 1)
    pooled = math.sqrt((var_lc + var_sl) / 2)
    return {
        "n_leader_change": len(d_lc),
        "n_same_leader": len(d_sl),
        "mean_distance_leader_change": mean_lc,
        "mean_distance_same_leader": mean_sl,
        "effect_diff": mean_lc - mean_sl,
        "cohen_d": (mean_lc - mean_sl) / pooled if pooled > 0 else 0.0,
        "_d_lc": d_lc,
        "_d_sl": d_sl,
    }


def paired_contrast_sparse(doc_vecs: dict[str, dict[str, float]], leader_change, same_leader):
    d_lc = [
        cosine_distance(doc_vecs[a], doc_vecs[b])
        for a, b in leader_change
        if a in doc_vecs and b in doc_vecs
    ]
    d_sl = [
        cosine_distance(doc_vecs[a], doc_vecs[b])
        for a, b in same_leader
        if a in doc_vecs and b in doc_vecs
    ]
    if not d_lc or not d_sl:
        return None
    mean_lc = sum(d_lc) / len(d_lc)
    mean_sl = sum(d_sl) / len(d_sl)
    var_lc = sum((x - mean_lc) ** 2 for x in d_lc) / max(1, len(d_lc) - 1)
    var_sl = sum((x - mean_sl) ** 2 for x in d_sl) / max(1, len(d_sl) - 1)
    pooled = math.sqrt((var_lc + var_sl) / 2)
    return {
        "n_leader_change": len(d_lc),
        "n_same_leader": len(d_sl),
        "mean_distance_leader_change": mean_lc,
        "mean_distance_same_leader": mean_sl,
        "effect_diff": mean_lc - mean_sl,
        "cohen_d": (mean_lc - mean_sl) / pooled if pooled > 0 else 0.0,
        "_d_lc": d_lc,
        "_d_sl": d_sl,
    }


def exact_permutation_pvalue(d_lc: list[float], d_sl: list[float]) -> tuple[float, float, int]:
    """Exact two-sided and one-sided p-values over all LC/SL label assignments."""
    all_vals = list(d_lc) + list(d_sl)
    n_lc = len(d_lc)
    obs = sum(d_lc) / len(d_lc) - sum(d_sl) / len(d_sl)
    cnt2 = cnt1 = total = 0
    for idx_lc in combinations(range(len(all_vals)), n_lc):
        idx_lc_set = set(idx_lc)
        new_lc = [all_vals[i] for i in idx_lc]
        new_sl = [all_vals[i] for i in range(len(all_vals)) if i not in idx_lc_set]
        stat = sum(new_lc) / len(new_lc) - sum(new_sl) / len(new_sl)
        if abs(stat) >= abs(obs) - 1e-12:
            cnt2 += 1
        if stat >= obs - 1e-12:
            cnt1 += 1
        total += 1
    return cnt2 / total, cnt1 / total, total


def tokenise(text: str) -> list[str]:
    return [t.strip() for t in jieba.cut(text) if t.strip()]


def build_doc_texts(segs: dict[str, dict]) -> dict[str, str]:
    by_doc = defaultdict(list)
    for s in segs.values():
        by_doc[s["doc_id"]].append((int(s["seg_idx"]), s["text"]))
    return {
        doc_id: "\n".join(text for _, text in sorted(parts))
        for doc_id, parts in by_doc.items()
    }


def tfidf_doc_vectors(doc_texts: dict[str, str]) -> dict[str, list[float]]:
    doc_ids = list(doc_texts)
    corpus = [doc_texts[d] for d in doc_ids]
    vectorizer = TfidfVectorizer(
        tokenizer=tokenise,
        token_pattern=None,
        lowercase=False,
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        norm="l2",
    )
    mat = vectorizer.fit_transform(corpus)
    return {doc_id: mat[i].toarray()[0].tolist() for i, doc_id in enumerate(doc_ids)}


def doc2vec_doc_vectors(doc_texts: dict[str, str], seed: int = 13) -> dict[str, list[float]]:
    docs = [
        TaggedDocument(words=tokenise(text), tags=[doc_id])
        for doc_id, text in doc_texts.items()
    ]
    model = Doc2Vec(
        vector_size=100,
        window=8,
        min_count=1,
        workers=1,
        epochs=80,
        seed=seed,
        dm=1,
        negative=5,
    )
    model.build_vocab(docs)
    model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)
    return {doc_id: model.dv[doc_id].tolist() for doc_id in doc_texts}


def load_dict_scores_substantive_only(segs: dict[str, dict], *, zero_non_substantive: bool = False) -> dict[str, dict[str, float]]:
    llm_l1 = {}
    for r in csv.DictReader((RESULTS_DIR / "scores_llm.csv").open(encoding="utf-8")):
        llm_l1[r["seg_id"]] = r["l1"]

    dict_scores = {}
    for r in csv.DictReader((RESULTS_DIR / "scores_dict.csv").open(encoding="utf-8")):
        sid = r["seg_id"]
        is_sub = llm_l1.get(sid) == "substantive"
        if not is_sub and not zero_non_substantive:
            continue
        if is_sub:
            dict_scores[sid] = {
                dk: float(r[f"{dk}_dict_score"])
                for dk in DIM_KEYS
            }
        else:
            dict_scores[sid] = {dk: 0.0 for dk in DIM_KEYS}
    return dict_scores


def make_redaction_terms(segs: dict[str, dict]) -> list[str]:
    terms = set(SLOGAN_TERMS)
    suffixes = [
        "有限公司", "有限责任公司", "集团有限公司", "集团", "公司",
        "控股", "股份", "中国", "国家",
    ]
    for s in segs.values():
        for key in ("company", "company_norm", "speaker"):
            val = s.get(key, "").strip()
            if not val:
                continue
            terms.add(val)
            simplified = val
            for suf in suffixes:
                simplified = simplified.replace(suf, "")
            if len(simplified) >= 2:
                terms.add(simplified)
    return sorted(terms, key=len, reverse=True)


def sanitize_text(text: str, terms: list[str], *, remove_numbers: bool, remove_slogans: bool) -> str:
    out = text
    # Always remove explicit metadata-like company/person terms for sanitized variants.
    metadata_terms = [t for t in terms if t not in SLOGAN_TERMS]
    for term in metadata_terms:
        out = out.replace(term, " ")
    out = re.sub(r"[12]\d{3}\s*年?", " ", out)
    if remove_numbers:
        out = re.sub(r"\d+(?:[.,:：]\d+)?%?", " ", out)
        out = re.sub(r"[A-Za-z]+[\w.-]*", " ", out)
    if remove_slogans:
        for term in SLOGAN_TERMS:
            out = out.replace(term, " ")
    out = re.sub(r"\s+", " ", out)
    return out


def sanitized_doc_texts(doc_texts: dict[str, str], segs: dict[str, dict], *, remove_numbers: bool, remove_slogans: bool) -> dict[str, str]:
    terms = make_redaction_terms(segs)
    return {
        doc_id: sanitize_text(text, terms, remove_numbers=remove_numbers, remove_slogans=remove_slogans)
        for doc_id, text in doc_texts.items()
    }


def summarize_doc2vec_seed_rows(rows: list[dict]) -> dict:
    ds = [float(r["Cohen_d"]) for r in rows]
    deltas = [float(r["Delta"]) for r in rows]
    p1s = [float(r["perm_p_1sided"]) for r in rows]
    return {
        "method": "doc2vec_doc_vector_seed_summary",
        "n_leader_change": rows[0]["n_leader_change"],
        "n_same_leader": rows[0]["n_same_leader"],
        "D_lc": "",
        "D_sl": "",
        "Delta": f"{statistics.mean(deltas):.6f}",
        "Cohen_d": f"{statistics.mean(ds):.3f}",
        "perm_p_2sided": "",
        "perm_p_1sided": f"{statistics.mean(p1s):.4f}",
        "n_perm": rows[0]["n_perm"],
        "note": (
            f"Doc2Vec over {len(rows)} seeds; Delta mean/range="
            f"{statistics.mean(deltas):.3f}/[{min(deltas):.3f},{max(deltas):.3f}], "
            f"d mean/range={statistics.mean(ds):.3f}/[{min(ds):.3f},{max(ds):.3f}], "
            f"p1 mean/range={statistics.mean(p1s):.4f}/[{min(p1s):.4f},{max(p1s):.4f}]."
        ),
    }


def llm_cache_parse_stats() -> dict:
    path = DATA_DIR / "llm_cache.jsonl"
    total = 0
    valid_outer = 0
    valid_inner = 0
    labels = Counter()
    if not path.exists():
        return {"total": 0, "valid_outer": 0, "valid_inner": 0, "parse_failures": None}
    with path.open(encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                row = json.loads(line)
                valid_outer += 1
            except json.JSONDecodeError:
                continue
            labels[row.get("l1", "missing")] += 1
            raw = row.get("_raw", "")
            try:
                json.loads(raw)
                valid_inner += 1
            except Exception:
                pass
    return {
        "total": total,
        "valid_outer": valid_outer,
        "valid_inner": valid_inner,
        "outer_parse_failures": total - valid_outer,
        "inner_raw_parse_failures": valid_outer - valid_inner,
        "l1_distribution": dict(labels),
    }


def rounded_result(method: str, result: dict, note: str) -> dict:
    p2, p1, n_perm = exact_permutation_pvalue(result["_d_lc"], result["_d_sl"])
    return {
        "method": method,
        "n_leader_change": result["n_leader_change"],
        "n_same_leader": result["n_same_leader"],
        "D_lc": f"{result['mean_distance_leader_change']:.6f}",
        "D_sl": f"{result['mean_distance_same_leader']:.6f}",
        "Delta": f"{result['effect_diff']:.6f}",
        "Cohen_d": f"{result['cohen_d']:.3f}",
        "perm_p_2sided": f"{p2:.4f}",
        "perm_p_1sided": f"{p1:.4f}",
        "n_perm": str(n_perm),
        "note": note,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    segs = load_segments()
    leader_change, same_leader = find_pairs(segs)
    doc_texts = build_doc_texts(segs)

    rows = []

    tfidf_vecs = tfidf_doc_vectors(doc_texts)
    rows.append(rounded_result(
        "tfidf_doc_vector",
        paired_contrast_dense(tfidf_vecs, leader_change, same_leader),
        "Generic lexical document vector; bounds non-LLM speaker/topic signal.",
    ))

    tfidf_name_year_vecs = tfidf_doc_vectors(
        sanitized_doc_texts(doc_texts, segs, remove_numbers=False, remove_slogans=False)
    )
    rows.append(rounded_result(
        "tfidf_sanitized_names_years",
        paired_contrast_dense(tfidf_name_year_vecs, leader_change, same_leader),
        "TF-IDF after removing company/person names and years.",
    ))

    tfidf_no_num_vecs = tfidf_doc_vectors(
        sanitized_doc_texts(doc_texts, segs, remove_numbers=True, remove_slogans=False)
    )
    rows.append(rounded_result(
        "tfidf_sanitized_names_years_numbers",
        paired_contrast_dense(tfidf_no_num_vecs, leader_change, same_leader),
        "TF-IDF after removing names, years, Arabic numerals, and alphanumeric tokens.",
    ))

    tfidf_no_slogan_vecs = tfidf_doc_vectors(
        sanitized_doc_texts(doc_texts, segs, remove_numbers=True, remove_slogans=True)
    )
    rows.append(rounded_result(
        "tfidf_sanitized_plus_slogan_terms",
        paired_contrast_dense(tfidf_no_slogan_vecs, leader_change, same_leader),
        "TF-IDF after also removing a conservative political-slogan term list.",
    ))

    doc2vec_seed_rows = []
    for seed in [1, 7, 13, 21, 42]:
        d2v_vecs = doc2vec_doc_vectors(doc_texts, seed=seed)
        row = rounded_result(
            f"doc2vec_doc_vector_seed_{seed}",
            paired_contrast_dense(d2v_vecs, leader_change, same_leader),
            f"Generic trained document embedding; seed={seed}, vector_size=100, epochs=80.",
        )
        rows.append(row)
        doc2vec_seed_rows.append(row)
    rows.append(summarize_doc2vec_seed_rows(doc2vec_seed_rows))

    segs_by_doc = defaultdict(list)
    for sid, s in segs.items():
        segs_by_doc[s["doc_id"]].append(sid)

    dict_sub_scores = load_dict_scores_substantive_only(segs, zero_non_substantive=False)
    dict_sub_doc_vecs = aggregate_doc_vector(dict_sub_scores, segs_by_doc)
    rows.append(rounded_result(
        "dict_on_llm_substantive_conditional",
        paired_contrast_sparse(dict_sub_doc_vecs, leader_change, same_leader),
        "Conditional profile: old dictionary averaged only over LLM-substantive segments.",
    ))

    dict_sub_mass_scores = load_dict_scores_substantive_only(segs, zero_non_substantive=True)
    dict_sub_mass_doc_vecs = aggregate_doc_vector(dict_sub_mass_scores, segs_by_doc)
    rows.append(rounded_result(
        "dict_on_llm_substantive_mass",
        paired_contrast_sparse(dict_sub_mass_doc_vecs, leader_change, same_leader),
        "Document-level mass: non-substantive segments set to zero before document averaging.",
    ))

    out_csv = RESULTS_DIR / "rebuttal_controls.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    parse_stats = llm_cache_parse_stats()
    out_md = RESULTS_DIR / "rebuttal_controls.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Rebuttal controls\n\n")
        f.write(f"Pairs: leader-change={len(leader_change)}, same-leader={len(same_leader)}.\n\n")
        f.write("| Method | D_lc | D_sl | Delta | Cohen d | exact p1 | Note |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write(
                f"| {r['method']} | {r['D_lc']} | {r['D_sl']} | "
                f"{r['Delta']} | {r['Cohen_d']} | {r['perm_p_1sided']} | {r['note']} |\n"
            )
        f.write("\n## LLM cache / JSON parse accounting\n\n")
        f.write("```json\n")
        f.write(json.dumps(parse_stats, ensure_ascii=False, indent=2))
        f.write("\n```\n")

    print(out_md)
    print(out_csv)
    print(json.dumps({"rows": rows, "parse_stats": parse_stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
