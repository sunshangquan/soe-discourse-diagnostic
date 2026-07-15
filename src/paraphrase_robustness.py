"""Paraphrase-robustness experiment.

For 50 substantive segments (sampled from those the LLM labelled
substantive with high confidence_substantive), prompt the same LLM to
*rewrite each in canonical SOE-slogan style* while preserving the
substantive proposition.  Then re-score each rewritten paragraph
under all five methods and report:

  - L1-stability of LLM (does it still call it substantive?)
  - dimension-score Spearman before/after (per method)
  - paired contrast on the rewritten corpus

Output:
  - data/paraphrase_pairs.jsonl  (orig + rewritten + scores)
  - results/paraphrase_robustness.csv
"""
from __future__ import annotations
import csv
import json
import random
import re
from pathlib import Path

import jieba

from dimensions import DIM_KEYS, SEED_WORDS_FLAT
from utils import DATA_DIR, RESULTS_DIR, PROMPTS_DIR
from score_llm import call_ollama, parse_json_response, normalize_response, build_prompt

random.seed(123)

REWRITE_PROMPT = """请把下面这段中国国企领导讲话改写成"政治表态/口号"风格：保留其表达的核心实质内容（同样的产业、行动、维度），但用抽象的政治套语、四字格、领导人提法、宏大叙述包装起来，不要保留具体的数字、具名项目或时间。改写后段落字数尽量接近原文。

只输出改写后的中文段落，不要任何解释或前缀。

原段落：
{seg_text}
"""


def call_rewrite(text: str, model: str = "qwen3.5:9b") -> str:
    """Use the LLM to rewrite a substantive paragraph in slogan style."""
    import urllib.request
    import json as _json
    body = _json.dumps({
        "model": model,
        "prompt": REWRITE_PROMPT.format(seg_text=text),
        "stream": False,
        "think": False,
        "options": {"temperature": 0.4, "num_predict": 600},
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        obj = _json.loads(resp.read().decode("utf-8"))
    return obj.get("response", "").strip()


def dict_score(text: str) -> dict[str, float]:
    toks = list(jieba.cut(text))
    total = max(1, sum(1 for t in toks if t.strip()))
    out = {}
    for dk, words in SEED_WORDS_FLAT.items():
        ws = set(words)
        hits = sum(1 for t in toks if t in ws)
        out[dk] = hits / total
    return out


def load_ngram_lexicon():
    p = DATA_DIR / "shared_ngrams_top200.tsv"
    if not p.exists():
        return []
    out = []
    import math
    with p.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            n = int(r["n"])
            df = int(r["doc_freq"])
            ng = r["ngram"]
            if n >= 5 and df >= 8:
                out.append((ng, math.log(df + 1)))
    return out


def ngram_density(text: str, lexicon) -> float:
    if not lexicon or not text:
        return 0.0
    total = 0.0
    for phrase, w in lexicon:
        cnt = text.count(phrase)
        if cnt > 0:
            total += w * cnt * len(phrase)
    return min(1.0, total / max(1, len(text) * 3))


def main():
    segs = list(csv.DictReader((DATA_DIR / "segments.csv").open(encoding="utf-8")))
    llm = {r["seg_id"]: r for r in csv.DictReader(
        (RESULTS_DIR / "scores_llm.csv").open(encoding="utf-8"))}

    # Sample 50 substantive segments with high confidence
    pool = [s for s in segs
            if llm.get(s["seg_id"], {}).get("l1") == "substantive"
            and float(llm.get(s["seg_id"], {}).get("confidence_substantive", 0)) >= 0.6
            and 100 <= int(s["char_count"]) <= 500]
    print(f"Pool: {len(pool)} candidate segments")
    sample = random.sample(pool, min(50, len(pool)))

    out_path = DATA_DIR / "paraphrase_pairs.jsonl"
    pairs = []
    cached = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    p = json.loads(line)
                    cached.add(p["seg_id"])
                    pairs.append(p)
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"Cached: {len(cached)} prior pairs")

    lexicon = load_ngram_lexicon()

    for i, s in enumerate(sample):
        sid = s["seg_id"]
        if sid in cached:
            continue
        orig = s["text"]
        print(f"[{i+1}/{len(sample)}] rewriting {sid[:35]}...", flush=True)
        try:
            rewritten = call_rewrite(orig, "qwen3.5:9b")
        except Exception as e:
            print(f"  rewrite-fail: {e}")
            continue
        if len(rewritten) < 30:
            print(f"  too-short rewrite ({len(rewritten)}); skipping")
            continue

        # Score both orig and rewritten under LLM
        try:
            r_orig = parse_json_response(call_ollama(build_prompt(orig), "qwen3.5:9b"))
            r_rew = parse_json_response(call_ollama(build_prompt(rewritten), "qwen3.5:9b"))
        except Exception as e:
            print(f"  score-fail: {e}")
            continue
        if not r_orig or not r_rew:
            print(f"  score-parse-fail")
            continue
        n_orig = normalize_response(r_orig)
        n_rew = normalize_response(r_rew)

        # Dict / ngram scoring for both
        d_orig = dict_score(orig)
        d_rew = dict_score(rewritten)
        ng_orig = ngram_density(orig, lexicon)
        ng_rew = ngram_density(rewritten, lexicon)

        # Compute calibrated for both
        def calibrated(n, sd_ng, lam_llm=1.0, lam_ng=2.0):
            sd_llm = n["slogan_density"]
            c = n["confidence_substantive"]
            m = c * max(0, 1 - lam_llm * sd_llm) * max(0, 1 - lam_ng * sd_ng)
            return {dk: n[f"raw_{dk}"] * m for dk in DIM_KEYS}

        entry = {
            "seg_id": sid,
            "doc_id": s["doc_id"],
            "orig_text": orig,
            "rewritten_text": rewritten,
            "orig": {
                "l1": n_orig["l1"],
                "confidence_substantive": n_orig["confidence_substantive"],
                "slogan_density_llm": n_orig["slogan_density"],
                "slogan_density_ngram": ng_orig,
                "raw": {dk: n_orig[f"raw_{dk}"] for dk in DIM_KEYS},
                "dict": d_orig,
                "calibrated": calibrated(n_orig, ng_orig),
            },
            "rewritten": {
                "l1": n_rew["l1"],
                "confidence_substantive": n_rew["confidence_substantive"],
                "slogan_density_llm": n_rew["slogan_density"],
                "slogan_density_ngram": ng_rew,
                "raw": {dk: n_rew[f"raw_{dk}"] for dk in DIM_KEYS},
                "dict": d_rew,
                "calibrated": calibrated(n_rew, ng_rew),
            },
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        pairs.append(entry)

    print(f"\nCollected {len(pairs)} pairs total")
    if len(pairs) < 5:
        print("not enough for analysis; rerun later")
        return

    # Analysis
    print("\n=== LLM L1 stability ===")
    same = sum(1 for p in pairs if p["orig"]["l1"] == p["rewritten"]["l1"])
    flipped = sum(1 for p in pairs
                  if p["orig"]["l1"] == "substantive" and p["rewritten"]["l1"] == "slogan")
    print(f"  total: {len(pairs)}, l1 unchanged: {same} ({100*same/len(pairs):.0f}%)")
    print(f"  substantive → slogan after rewrite: {flipped}")

    # Spearman per method
    print("\n=== Score correlation orig vs rewritten (per method, mean over 5 dims) ===")
    from statistics import mean

    def correl(xs, ys):
        # Simple Pearson (Spearman is overkill on 50 pts)
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        denx = sum((x - mx) ** 2 for x in xs) ** 0.5
        deny = sum((y - my) ** 2 for y in ys) ** 0.5
        return num / (denx * deny) if denx > 0 and deny > 0 else 0.0

    method_keys = ["dict", "raw", "calibrated"]
    method_corrs = {}
    for m in method_keys:
        per_dim_corr = []
        for dk in DIM_KEYS:
            xs = [p["orig"][m][dk] for p in pairs]
            ys = [p["rewritten"][m][dk] for p in pairs]
            per_dim_corr.append(correl(xs, ys))
        method_corrs[m] = mean(per_dim_corr)
        print(f"  {m:12s}: mean per-dim Pearson r = {mean(per_dim_corr):.3f}")

    # Magnitude shift: mean L1-norm of (orig - rewritten) per method
    print("\n=== Score shift magnitude (mean L1 distance, 5-dim vectors) ===")
    for m in method_keys:
        shifts = []
        for p in pairs:
            v_o = [p["orig"][m][dk] for dk in DIM_KEYS]
            v_r = [p["rewritten"][m][dk] for dk in DIM_KEYS]
            shifts.append(sum(abs(a - b) for a, b in zip(v_o, v_r)))
        print(f"  {m:12s}: mean L1 = {mean(shifts):.4f}")

    # Save summary CSV
    with (RESULTS_DIR / "paraphrase_robustness.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "dict", "raw_llm", "calibrated"])
        w.writerow(["pearson_r_orig_rewritten",
                    method_corrs["dict"], method_corrs["raw"], method_corrs["calibrated"]])
        # L1 shift
        for m in method_keys:
            pass
        w.writerow(["l1_stability_fraction", "", same/len(pairs), ""])
        w.writerow(["substantive_to_slogan_flip", "", flipped, ""])
    print(f"\nWrote {RESULTS_DIR / 'paraphrase_robustness.csv'}")


if __name__ == "__main__":
    main()
