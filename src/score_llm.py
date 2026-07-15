"""LLM zero-shot stance extraction.

Single-segment prompting; outputs:
  - l1 (slogan/substantive/irrelevant)
  - l2 (firm_action/policy_history/system_aggregate or "")
  - confidence_substantive ∈ [0,1]
  - stance_scores[dim] ∈ [0,1]  (raw, uncalibrated)
  - slogan_density ∈ [0,1]

Run modes:
  --dry-run         # print first 3 prompts, no API calls
  --limit N         # only score first N segments (smoke test)
  --provider claude|openai
  --model MODEL_ID
  --resume          # skip segments already in cache

Cache: data/llm_cache.jsonl (one JSON line per seg_id)
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from dimensions import DIMENSIONS, DIM_KEYS
from utils import DATA_DIR, RESULTS_DIR, PROMPTS_DIR

PROMPT_PATH = PROMPTS_DIR / "stance_extraction.txt"
CACHE_PATH = DATA_DIR / "llm_cache.jsonl"


_DIM_SHORT = {
    "innovation":              "innovation = 研发/专利/技术突破/新产品具体动作",
    "competition_cooperation": "competition_cooperation = 合资/并购/海外项目具体数字",
    "organization_market":     "organization_market = 混改/股权/激励/上市/治理具体动作",
    "social_responsibility":   "social_responsibility = 环保/减排/扶贫/安全具体项目数字",
    "national_mission":        "national_mission = 国家重大工程/关键技术攻关具体落地",
}


def build_prompt(seg_text: str) -> str:
    tmpl = PROMPT_PATH.read_text(encoding="utf-8")
    dim_desc = "\n".join(f"- {_DIM_SHORT[d['key']]}" for d in DIMENSIONS)
    return tmpl.replace("{dim_descriptions}", dim_desc).replace("{segment_text}", seg_text)


def parse_json_response(text: str) -> dict | None:
    """Robustly extract a JSON object from LLM output."""
    text = text.strip()
    # Strip code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.strip().startswith("```"))
    # Find the first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def normalize_response(resp: dict) -> dict:
    """Ensure all expected fields exist with sane defaults."""
    out = {
        "l1": resp.get("l1", "irrelevant"),
        "l2": resp.get("l2", ""),
        "confidence_substantive": float(resp.get("confidence_substantive", 0.0)),
        "slogan_density": float(resp.get("slogan_density", 0.0)),
    }
    stance = resp.get("stance_scores", {})
    for dk in DIM_KEYS:
        out[f"raw_{dk}"] = float(stance.get(dk, 0.0))
    return out


def load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    cache = {}
    with CACHE_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cache[obj["seg_id"]] = obj
            except json.JSONDecodeError:
                continue
    return cache


def append_cache(entry: dict):
    with CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ─── Provider wrappers ────────────────────────────────────────────────────────

def call_claude(prompt: str, model: str = "claude-opus-4-7") -> str:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def call_openai(prompt: str, model: str = "gpt-5") -> str:
    from openai import OpenAI
    client = OpenAI()
    msg = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    return msg.choices[0].message.content


def call_ollama(prompt: str, model: str = "qwen3.5:9b", num_predict: int = 320) -> str:
    """Local Ollama via its native /api/generate endpoint.

    Use native API (not OpenAI-compatible /v1) because qwen3.5 thinking models
    emit `thinking` and `response` as separate fields; the OpenAI shim drops
    the response. Native API returns the answer cleanly.

    num_predict=320 fits a typical JSON output for this task (~250 chars + buffer).
    """
    import urllib.request
    import json as _json
    body = _json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,  # explicitly disable thinking to save tokens & latency
        "options": {"temperature": 0.0, "num_predict": num_predict, "top_p": 1.0},
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        obj = _json.loads(resp.read().decode("utf-8"))
    return obj.get("response", "")


CALLERS = {"claude": call_claude, "openai": call_openai, "ollama": call_ollama}


# ─── Mock mode: deterministic plausible scores for pipeline testing ──────────

def run_mock(segs):
    """Emit mock scores correlated with surface text patterns, so the downstream
    pipeline (calibration + evaluation) is exercisable without an API key.

    Heuristic mock:
      - slogan_density ↑ if doc contains Xi-quote markers ("习近平" / "重要讲话" / 四字格)
      - confidence_substantive ↑ if segment contains numbers / 年份 / 具体动词
      - l1 = "slogan" if slogan_density>0.5; "substantive" if confidence_sub>0.5; else "irrelevant"
      - stance_scores[dim] ← jaccard between segment tokens and dim seed words
    """
    import re
    import hashlib
    import jieba
    from dimensions import SEED_WORDS_FLAT

    rows = []
    for s in segs:
        text = s["text"]
        toks = set(t for t in jieba.cut(text) if t.strip())
        # Surface signals
        has_xi = sum(1 for w in ["习近平", "总书记", "重要讲话", "讲话指出"] if w in text)
        has_4char = len(re.findall(r"[一-鿿]{4}[、，。]", text))
        has_num = len(re.findall(r"\d+(?:\.\d+)?[%亿万千百]?", text))
        has_year = len(re.findall(r"(19|20)\d{2}年", text))

        slogan_density = min(1.0, 0.15 * has_xi + 0.02 * has_4char)
        confidence_substantive = min(1.0, 0.18 * has_num + 0.10 * has_year)

        if slogan_density > 0.5 and confidence_substantive < 0.3:
            l1 = "slogan"
        elif confidence_substantive > 0.5:
            l1 = "substantive"
        elif slogan_density > 0.3:
            l1 = "slogan"
        else:
            l1 = "irrelevant" if len(text) < 80 else "substantive"
        l2 = "firm_action" if l1 == "substantive" else ""

        stance_scores = {}
        for dim, seeds in SEED_WORDS_FLAT.items():
            inter = len(toks & set(seeds))
            stance_scores[dim] = min(1.0, inter / 5.0 + 0.1 * (has_num > 0 and dim == "innovation"))

        entry = {
            "seg_id": s["seg_id"],
            "l1": l1, "l2": l2,
            "confidence_substantive": round(confidence_substantive, 3),
            "slogan_density": round(slogan_density, 3),
            **{f"raw_{k}": round(v, 3) for k, v in stance_scores.items()},
            "_mock": True,
        }
        rows.append(entry)

    out = RESULTS_DIR / "scores_llm.csv"
    fns = ["seg_id", "doc_id", "l1", "l2", "confidence_substantive", "slogan_density"] \
          + [f"raw_{k}" for k in DIM_KEYS]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        seg_map = {s["seg_id"]: s for s in segs}
        for r in rows:
            sid = r["seg_id"]
            w.writerow({"seg_id": sid, "doc_id": seg_map[sid]["doc_id"],
                        "l1": r["l1"], "l2": r["l2"],
                        "confidence_substantive": r["confidence_substantive"],
                        "slogan_density": r["slogan_density"],
                        **{f"raw_{k}": r[f"raw_{k}"] for k in DIM_KEYS}})
    print(f"\n[MOCK] Wrote {out} with {len(rows)} mock scores")
    print(f"[MOCK] L1 distribution: { {l: sum(1 for r in rows if r['l1']==l) for l in ['slogan','substantive','irrelevant']} }")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="claude", choices=list(CALLERS))
    ap.add_argument("--model", default="claude-opus-4-7")
    ap.add_argument("--limit", type=int, default=None,
                    help="Score only the first N segments")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print first 3 prompts, no API calls")
    ap.add_argument("--mock", action="store_true",
                    help="Generate plausible mock scores instead of calling API "
                         "(deterministic per seg_id; for pipeline testing)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip segments already cached")
    ap.add_argument("--retry", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="Sleep between API calls (set 0 for ollama with concurrency)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Parallel requests (ollama: set 2-4; rate-limited APIs: keep 1)")
    args = ap.parse_args()

    segs = list(csv.DictReader((DATA_DIR / "segments.csv").open(encoding="utf-8")))
    if args.limit:
        segs = segs[: args.limit]
    print(f"Loaded {len(segs)} segments")

    if args.dry_run:
        for s in segs[:3]:
            print("=" * 80)
            print(f"seg_id={s['seg_id']}")
            print(build_prompt(s["text"]))
        print("\n[dry-run] No API calls made.")
        return

    if args.mock:
        run_mock(segs)
        return

    if args.provider == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. export it or use --dry-run / --mock.")
        sys.exit(1)
    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. export it or use --dry-run.")
        sys.exit(1)
    if args.provider == "ollama":
        # Sanity check: ensure ollama daemon is reachable
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3).read()
        except Exception as e:
            print(f"ERROR: cannot reach Ollama at localhost:11434 ({e}). "
                  f"Start with `ollama serve` (or check `ollama list`).")
            sys.exit(1)

    cache = load_cache() if args.resume else {}
    print(f"Cache: {len(cache)} prior entries (resume={args.resume})")

    caller = CALLERS[args.provider]
    results = {}
    n_called = 0
    n_failed = 0
    cache_lock = None  # only used in threaded path

    def process_one(s):
        sid = s["seg_id"]
        if sid in cache:
            return sid, cache[sid], None
        prompt = build_prompt(s["text"])
        resp_text = None
        last_err = None
        for attempt in range(args.retry):
            try:
                resp_text = caller(prompt, model=args.model)
                break
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        if resp_text is None:
            return sid, None, f"call failed: {last_err}"
        parsed = parse_json_response(resp_text)
        if parsed is None:
            return sid, None, f"parse fail: {resp_text[:120]}"
        norm = normalize_response(parsed)
        entry = {"seg_id": sid, **norm, "_raw": resp_text[:500]}
        return sid, entry, None

    if args.concurrency <= 1:
        for s in segs:
            sid, entry, err = process_one(s)
            if err:
                print(f"  [fail] {sid}: {err}")
                n_failed += 1
                continue
            results[sid] = entry
            if entry.get("_raw") is not None and sid not in cache:
                append_cache(entry)
                n_called += 1
                if n_called % 25 == 0:
                    print(f"  ... {n_called} calls done, {n_failed} failed")
            time.sleep(args.sleep)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        cache_lock = threading.Lock()
        print(f"  Running with concurrency={args.concurrency}")
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {ex.submit(process_one, s): s["seg_id"] for s in segs}
            for fut in as_completed(futures):
                sid, entry, err = fut.result()
                if err:
                    print(f"  [fail] {sid}: {err}")
                    n_failed += 1
                    continue
                results[sid] = entry
                if sid not in cache:
                    with cache_lock:
                        append_cache(entry)
                    n_called += 1
                    if n_called % 25 == 0:
                        print(f"  ... {n_called} calls done, {n_failed} failed")

    # Emit final CSV
    out = RESULTS_DIR / "scores_llm.csv"
    fns = ["seg_id", "doc_id", "l1", "l2", "confidence_substantive", "slogan_density"] \
          + [f"raw_{k}" for k in DIM_KEYS]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        for s in segs:
            r = results.get(s["seg_id"])
            if not r:
                continue
            w.writerow({
                "seg_id": s["seg_id"],
                "doc_id": s["doc_id"],
                "l1": r["l1"],
                "l2": r["l2"],
                "confidence_substantive": r["confidence_substantive"],
                "slogan_density": r["slogan_density"],
                **{f"raw_{k}": r[f"raw_{k}"] for k in DIM_KEYS},
            })
    print(f"\nWrote {out}: {len(results)} segments scored; {n_failed} failures")


if __name__ == "__main__":
    main()
