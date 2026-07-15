"""Phase 2 — Style/idiolect baseline.

Reviewer concern (R2/DA): the leader-change paired-contrast signal
may come from leader IDIOLECT (sentence length, function-word
distribution, pronoun rate) rather than stance.  If a pure
stylometric feature vector already passes the paired-contrast test
with comparable Cohen d, the diagnostic confounds stance with style.

We compute 8 stylometric features per document:
  1. mean character count per paragraph
  2. mean sentence count per paragraph
  3. mean characters per sentence
  4. first-person ratio (我/我们/我方 over total chars)
  5. ratio of formal-modal hedges (可以/应该/必须/需要)
  6. ratio of certainty markers (一定/必然/无疑)
  7. ratio of policy-cliche bigrams (sampled set)
  8. unique 2-gram ratio (a crude lexical-diversity measure)

Then compute the paired contrast (Cohen d + permutation p) using
just the doc-level style vector.

If style-baseline d >> 0 with p < 0.10, the leader-change signal is
NOT specifically about stance; the diagnostic is confounded.
If style-baseline d ≈ 0, the leader-change paired-contrast result for
our methods is style-controlled.

Output: results/style_baseline.csv
"""
from __future__ import annotations
import csv
import math
import re
from pathlib import Path

from utils import CORPUS_DIR, DATA_DIR, RESULTS_DIR, read_text
from stats_addon import find_pairs
from permutation_test import cohen_d, hedges_g, permutation_pvalue


FIRST_PERSON = re.compile(r"(我们|我国|我方|我|本集团|本公司)")
HEDGES = re.compile(r"(可以|应该|必须|需要|有望|有可能|或许|大概)")
CERTAINTY = re.compile(r"(一定|必然|无疑|毫无疑问|肯定|坚定不移|毫不动摇)")
CLICHE_2GRAM = [
    "高质量", "深化改革", "新时代", "新发展", "走中国",
    "习近平", "总书记", "党中央", "人民群众",
    "重大战略", "重要部署",
]


def doc_style_features(text: str) -> dict[str, float]:
    """8 stylometric features per document."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    n_para = max(1, len(paras))
    n_char = sum(len(p) for p in paras)
    if n_char == 0:
        return {f: 0.0 for f in ["mean_para_chars", "mean_para_sents", "mean_sent_chars",
                                  "first_person_ratio", "hedge_ratio", "certainty_ratio",
                                  "cliche_2gram_ratio", "unique_2gram_ratio"]}
    sentences = re.split(r"[。！？]", text)
    sentences = [s for s in sentences if s.strip()]
    n_sent = max(1, len(sentences))

    first_count = len(FIRST_PERSON.findall(text))
    hedge_count = len(HEDGES.findall(text))
    certainty_count = len(CERTAINTY.findall(text))
    cliche_count = sum(text.count(c) for c in CLICHE_2GRAM)
    # 2-gram unique ratio
    chars = re.sub(r"\s+", "", text)
    bigrams = [chars[i:i+2] for i in range(len(chars)-1)]
    unique_2g = len(set(bigrams)) / max(1, len(bigrams))

    return {
        "mean_para_chars": n_char / n_para,
        "mean_para_sents": n_sent / n_para,
        "mean_sent_chars": n_char / n_sent,
        "first_person_ratio": first_count * 100 / n_char,
        "hedge_ratio": hedge_count * 100 / n_char,
        "certainty_ratio": certainty_count * 100 / n_char,
        "cliche_2gram_ratio": cliche_count * 100 / n_char,
        "unique_2gram_ratio": unique_2g,
    }


def cosine_dist(v1: dict, v2: dict) -> float:
    """Cosine distance with z-score normalisation across all docs (done outside)."""
    keys = list(v1.keys())
    a = [v1[k] for k in keys]
    b = [v2[k] for k in keys]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / (na * nb) if (na > 0 and nb > 0) else 1.0


def z_normalize(features_by_doc):
    """Z-score each feature across docs."""
    keys = list(next(iter(features_by_doc.values())).keys())
    out = {did: {} for did in features_by_doc}
    for k in keys:
        vals = [features_by_doc[did][k] for did in features_by_doc]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)
        sd = math.sqrt(var) if var > 0 else 1.0
        for did in features_by_doc:
            out[did][k] = (features_by_doc[did][k] - mu) / sd
    return out


def main():
    # Build doc vectors
    segs = list(csv.DictReader((DATA_DIR / "segments.csv").open(encoding="utf-8")))
    seg_meta = {s["seg_id"]: s for s in segs}
    docs = sorted(set(s["doc_id"] for s in segs))
    features = {}
    for did in docs:
        text = read_text(CORPUS_DIR / did)
        features[did] = doc_style_features(text)
    features = z_normalize(features)
    print(f"{len(features)} docs, {len(next(iter(features.values())))} features")

    lc, sl = find_pairs(seg_meta)
    print(f"Pairs: LC={len(lc)}, SL={len(sl)}")

    d_lc = [cosine_dist(features[a], features[b]) for a, b in lc if a in features and b in features]
    d_sl = [cosine_dist(features[a], features[b]) for a, b in sl if a in features and b in features]
    m_lc = sum(d_lc) / len(d_lc)
    m_sl = sum(d_sl) / len(d_sl)
    d = cohen_d(d_lc, d_sl)
    g = hedges_g(d_lc, d_sl)
    p2, p1, ptype, ptotal = permutation_pvalue(d_lc, d_sl)

    print(f"\n== Pure style baseline (8 features) ==")
    print(f"  D_lc = {m_lc:.4f}, D_sl = {m_sl:.4f}, diff = {m_lc-m_sl:+.4f}")
    print(f"  Cohen d = {d:.3f}, Hedges g = {g:.3f}")
    print(f"  Permutation: 2-sided p = {p2:.4f}, 1-sided p = {p1:.4f}  ({ptype})")

    if p1 < 0.05:
        print(f"  ⚠️  Style alone passes paired-contrast at α=0.05 (1-sided)")
        print(f"  → Leader-change signal IS confounded with idiolect")
    else:
        print(f"  ✓ Style alone does NOT pass paired-contrast (1-sided p > 0.05)")
        print(f"  → Style is NOT sufficient to explain the leader-change signal")

    out = RESULTS_DIR / "style_baseline.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mean_D_lc", "mean_D_sl", "diff", "cohen_d", "hedges_g", "perm_p_2sided", "perm_p_1sided"])
        w.writerow([m_lc, m_sl, m_lc - m_sl, d, g, p2, p1])
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
