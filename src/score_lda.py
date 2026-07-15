"""Baseline 2: LDA scorer.

Train LDA(K=5) on the 80-doc corpus (each doc as one document), then for each segment:
  1. Project to topic distribution
  2. Map topics → 5 dimensions via seed-word weight argmax
  3. Output per-dimension topic-prob as the dimension score

Output: results/scores_lda.csv
"""
from __future__ import annotations
import csv
import re
import warnings
from collections import defaultdict
from pathlib import Path

import jieba
from gensim import corpora, models

from dimensions import DIM_KEYS, SEED_WORDS_FLAT
from utils import CORPUS_DIR, DATA_DIR, RESULTS_DIR, ROOT, read_text

warnings.filterwarnings("ignore")

K = 5  # number of LDA topics


def load_stopwords() -> set[str]:
    # Try an optional project stopword file if present, else use a minimal Chinese stop set.
    old = ROOT / "stopword" / "stopword.txt"
    if old.exists():
        return set(l.strip() for l in old.read_text(encoding="utf-8").splitlines() if l.strip())
    return set("的 了 是 在 和 有 与 也 都 就 我们 他们 你们 我 你 他 她 它 这 那 个 们 把 被 让 使 这个 那个 这些 那些 一个 一种 一些 还 又 而 但 但是 而且 然后 所以 因为 因此 不 没有 没 不是 不能 不会 可以 应该 必须 需要 比较 非常 更加 这样 那样 怎么 什么 怎样 为什么 上 下 中 里 外 前 后 左 右 大 小 多 少 高 低 好 坏 新 老 早 晚 来 去 进 出 起 走 跑 上来 下去 起来 出去 进来 出来".split())


PUNCT_RE = re.compile(r"^[\W_]+$")


def tokenize(text: str, stopwords: set[str]) -> list[str]:
    return [t for t in jieba.cut(text)
            if t.strip() and not PUNCT_RE.match(t)
            and t not in stopwords and len(t) > 1]


def train_lda(doc_tokens: list[list[str]], K: int = 5):
    dictionary = corpora.Dictionary(doc_tokens)
    dictionary.filter_extremes(no_below=3, no_above=0.6)
    corpus = [dictionary.doc2bow(d) for d in doc_tokens]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = [tfidf[c] for c in corpus]
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=K,
                          random_state=42, passes=10, iterations=200)
    return dictionary, tfidf, lda


def map_topics_to_dimensions(lda, dictionary) -> dict[str, int]:
    """For each dimension, pick the topic with the highest aggregate weight on its seed words."""
    mapping = {}
    used_topics = set()
    # Greedy assignment: rank (dim, topic) by score, pick highest first
    scores = []
    for dim_key, seeds in SEED_WORDS_FLAT.items():
        seed_ids = [dictionary.token2id[w] for w in seeds if w in dictionary.token2id]
        for t in range(lda.num_topics):
            topic_terms = dict(lda.get_topic_terms(t, topn=200))
            agg = sum(topic_terms.get(sid, 0.0) for sid in seed_ids)
            scores.append((agg, dim_key, t))
    scores.sort(reverse=True)
    for agg, dim_key, t in scores:
        if dim_key in mapping or t in used_topics:
            continue
        mapping[dim_key] = t
        used_topics.add(t)
        if len(mapping) == lda.num_topics:
            break
    # Fallback: if some dim has no topic, give the leftover
    leftover_topics = [t for t in range(lda.num_topics) if t not in used_topics]
    for dim_key in DIM_KEYS:
        if dim_key not in mapping and leftover_topics:
            mapping[dim_key] = leftover_topics.pop()
    return mapping


def main():
    stopwords = load_stopwords()
    print(f"Loaded {len(stopwords)} stopwords")

    # Train LDA on docs (one doc per file)
    doc_ids = sorted([f for f in CORPUS_DIR.iterdir() if f.suffix == ".txt"])
    doc_tokens = [tokenize(read_text(p), stopwords) for p in doc_ids]
    print(f"Tokenized {len(doc_tokens)} docs; training LDA(K={K})...")

    dictionary, tfidf, lda = train_lda(doc_tokens, K=K)

    print("\nTopics (top-10 words):")
    for t in range(K):
        words = [w for w, _ in lda.show_topic(t, topn=10)]
        print(f"  T{t}: {' '.join(words)}")

    mapping = map_topics_to_dimensions(lda, dictionary)
    print(f"\nTopic→dimension mapping: {mapping}")
    reverse = {t: d for d, t in mapping.items()}

    # Score each segment
    segs = list(csv.DictReader((DATA_DIR / "segments.csv").open(encoding="utf-8")))
    rows = []
    for s in segs:
        toks = tokenize(s["text"], stopwords)
        bow = dictionary.doc2bow(toks)
        bow_tfidf = tfidf[bow]
        topic_dist = dict(lda[bow_tfidf])  # {topic_id: prob}
        row = {"seg_id": s["seg_id"], "doc_id": s["doc_id"]}
        for dim_key in DIM_KEYS:
            t = mapping[dim_key]
            row[f"{dim_key}_lda_score"] = round(topic_dist.get(t, 0.0), 6)
        rows.append(row)

    out = RESULTS_DIR / "scores_lda.csv"
    fns = ["seg_id", "doc_id"] + [f"{k}_lda_score" for k in DIM_KEYS]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out} with {len(rows)} segments")


if __name__ == "__main__":
    main()
