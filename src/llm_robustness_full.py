"""Full LLM robustness runs for rebuttal.

Runs Qwen3.5:9b on all 2,190 segments without overwriting the main cache.
Supported checks:

  repeated_zh  : same Chinese prompt as the main experiment
  english_full : English instruction prompt, same JSON schema, Chinese input text

Each run writes:
  data/llm_cache_<mode>.jsonl
  data/llm_failures_<mode>.jsonl
  results/scores_llm_<mode>.csv
  results/llm_robustness_<mode>.json
  results/llm_robustness_<mode>.md

The report compares the new run against the original Chinese-prompt cache.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

from dimensions import DIMENSIONS, DIM_KEYS
from score_llm import build_prompt, parse_json_response, normalize_response
from stats_addon import find_pairs, doc_vectors, cosine_distance, cohen_d
from utils import DATA_DIR, RESULTS_DIR


MAIN_CACHE = DATA_DIR / "llm_cache.jsonl"


_DIM_SHORT_EN = {
    "innovation": "innovation: concrete R&D, patents, technical breakthroughs, new products, or capability-building actions",
    "competition_cooperation": "competition_cooperation: concrete joint ventures, M&A, overseas projects, alliances, competition, or cooperation with numbers/objects",
    "organization_market": "organization_market: concrete ownership reform, equity, incentives, listing, governance, marketization, or internal organization changes",
    "social_responsibility": "social_responsibility: concrete environmental, safety, poverty-alleviation, public-service, or ESG projects/outcomes",
    "national_mission": "national_mission: concrete national major projects, key technology breakthroughs, supply security, or strategic task implementation",
}


def build_english_prompt(seg_text: str) -> str:
    dim_desc = "\n".join(f"- {_DIM_SHORT_EN[d['key']]}" for d in DIMENSIONS)
    return f"""Analyze the following segment from a Chinese state-owned enterprise leadership speech. Output JSON only.

Five dimensions:
{dim_desc}

Decide l1:
- slogan: abstract slogan / political statement / leader quotation / formulaic expression / no specific subject-verb-object-number-tradeoff. Heuristic: if the paragraph could be moved verbatim into almost any SOE speech and still read naturally, label slogan.
- substantive: concrete verb + object, resource allocation number, observable outcome, comparison or tradeoff, or time/place/object specificity. Use substantive when at least two such cues are present.
- irrelevant: greeting, transition, or unrelated to the five dimensions.

l2, only when l1 = substantive:
- firm_action: firm-specific action, decision, reform, investment, capability, or outcome
- policy_history: policy or institutional history
- system_aggregate: system-level or industry-level aggregate statistics

Output strict JSON:
{{
  "l1": "slogan|substantive|irrelevant",
  "l2": "",
  "confidence_substantive": 0.0,
  "stance_scores": {{
    "innovation": 0.0,
    "competition_cooperation": 0.0,
    "organization_market": 0.0,
    "social_responsibility": 0.0,
    "national_mission": 0.0
  }},
  "slogan_density": 0.0
}}

confidence_substantive is P(segment is substantive), from 0 to 1.
stance_scores[dim] is the stance strength for that dimension conditional on the segment being substantive, from 0 to 1. Keyword presence alone should not imply a high score; concrete actions, numbers, outcomes, and tradeoffs matter.
slogan_density is the estimated fraction of the segment occupied by formulaic political language, from 0 to 1.

Segment:
{seg_text}
"""


def build_prompt_for_mode(mode: str, text: str) -> str:
    if mode == "repeated_zh":
        return build_prompt(text)
    if mode == "english_full":
        return build_english_prompt(text)
    raise ValueError(f"unknown mode: {mode}")


def call_ollama(prompt: str, model: str, num_predict: int) -> str:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": num_predict, "top_p": 1.0},
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=180) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    return obj.get("response", "")


def read_jsonl_by_seg(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("seg_id")
                if sid:
                    out[sid] = obj
            except json.JSONDecodeError:
                continue
    return out


def append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def cohen_kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    pa = {c: a.count(c) / len(a) for c in cats}
    pb = {c: b.count(c) / len(b) for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def exact_permutation_pvalue(d_lc: list[float], d_sl: list[float]) -> tuple[float, float, int]:
    all_vals = list(d_lc) + list(d_sl)
    n_lc = len(d_lc)
    obs = sum(d_lc) / len(d_lc) - sum(d_sl) / len(d_sl)
    cnt2 = cnt1 = total = 0
    for idx_lc in combinations(range(len(all_vals)), n_lc):
        idx = set(idx_lc)
        new_lc = [all_vals[i] for i in idx_lc]
        new_sl = [all_vals[i] for i in range(len(all_vals)) if i not in idx]
        stat = sum(new_lc) / len(new_lc) - sum(new_sl) / len(new_sl)
        if abs(stat) >= abs(obs) - 1e-12:
            cnt2 += 1
        if stat >= obs - 1e-12:
            cnt1 += 1
        total += 1
    return cnt2 / total, cnt1 / total, total


def write_scores_csv(mode: str, segs: list[dict], cache: dict[str, dict]) -> Path:
    seg_lookup = {s["seg_id"]: s for s in segs}
    out = RESULTS_DIR / f"scores_llm_{mode}.csv"
    fns = ["seg_id", "doc_id", "l1", "l2", "confidence_substantive", "slogan_density"] + [f"raw_{k}" for k in DIM_KEYS]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        for sid in sorted(cache):
            if sid not in seg_lookup:
                continue
            r = cache[sid]
            w.writerow({
                "seg_id": sid,
                "doc_id": seg_lookup[sid]["doc_id"],
                "l1": r.get("l1", ""),
                "l2": r.get("l2", ""),
                "confidence_substantive": r.get("confidence_substantive", 0.0),
                "slogan_density": r.get("slogan_density", 0.0),
                **{f"raw_{k}": r.get(f"raw_{k}", 0.0) for k in DIM_KEYS},
            })
    return out


def load_scores_dict(cache: dict[str, dict]) -> dict[str, dict[str, float]]:
    return {
        sid: {dk: float(r.get(f"raw_{dk}", 0.0)) for dk in DIM_KEYS}
        for sid, r in cache.items()
    }


def paired_contrast_from_cache(cache: dict[str, dict], seg_meta: dict[str, dict]) -> dict:
    lc, sl = find_pairs(seg_meta)
    dv = doc_vectors(load_scores_dict(cache), seg_meta)
    d_lc = [cosine_distance(dv[a], dv[b]) for a, b in lc if a in dv and b in dv]
    d_sl = [cosine_distance(dv[a], dv[b]) for a, b in sl if a in dv and b in dv]
    if not d_lc or not d_sl:
        return {
            "n_leader_change": len(d_lc),
            "n_same_leader": len(d_sl),
            "D_lc": None,
            "D_sl": None,
            "Delta": None,
            "cohen_d": None,
            "exact_p_2sided": None,
            "exact_p_1sided": None,
            "n_perm": 0,
        }
    delta = sum(d_lc) / len(d_lc) - sum(d_sl) / len(d_sl)
    p2, p1, n_perm = exact_permutation_pvalue(d_lc, d_sl)
    return {
        "n_leader_change": len(d_lc),
        "n_same_leader": len(d_sl),
        "D_lc": round(sum(d_lc) / len(d_lc), 6),
        "D_sl": round(sum(d_sl) / len(d_sl), 6),
        "Delta": round(delta, 6),
        "cohen_d": round(cohen_d(d_lc, d_sl), 3),
        "exact_p_2sided": round(p2, 4),
        "exact_p_1sided": round(p1, 4),
        "n_perm": n_perm,
    }


def compare_runs(mode: str, segs: list[dict], main: dict[str, dict], new: dict[str, dict], failures: dict[str, dict]):
    shared = sorted(set(main) & set(new))
    a_l1 = [main[s].get("l1", "") for s in shared]
    b_l1 = [new[s].get("l1", "") for s in shared]
    l1_agreement = sum(1 for x, y in zip(a_l1, b_l1) if x == y) / len(shared) if shared else 0.0
    l1_kappa = cohen_kappa(a_l1, b_l1)

    mutually_sub = [s for s in shared if main[s].get("l1") == "substantive" and new[s].get("l1") == "substantive"]
    l2_kappa = cohen_kappa(
        [main[s].get("l2", "") for s in mutually_sub],
        [new[s].get("l2", "") for s in mutually_sub],
    ) if mutually_sub else 0.0

    per_dim_r = {}
    for dk in DIM_KEYS:
        per_dim_r[dk] = pearson(
            [float(main[s].get(f"raw_{dk}", 0.0)) for s in shared],
            [float(new[s].get(f"raw_{dk}", 0.0)) for s in shared],
        )
    conf_r = pearson(
        [float(main[s].get("confidence_substantive", 0.0)) for s in shared],
        [float(new[s].get("confidence_substantive", 0.0)) for s in shared],
    )
    slogan_r = pearson(
        [float(main[s].get("slogan_density", 0.0)) for s in shared],
        [float(new[s].get("slogan_density", 0.0)) for s in shared],
    )

    seg_meta = {s["seg_id"]: s for s in segs}
    paired = paired_contrast_from_cache(new, seg_meta)
    label_dist = dict(Counter(b_l1))

    report = {
        "mode": mode,
        "n_segments": len(segs),
        "n_valid": len(new),
        "n_shared_with_main": len(shared),
        "parse_failures": len(failures),
        "missing_or_failed": len(segs) - len(new),
        "l1_exact_agreement": round(l1_agreement, 4),
        "l1_cohen_kappa": round(l1_kappa, 3),
        "l2_cohen_kappa_mutually_substantive": round(l2_kappa, 3),
        "n_mutually_substantive": len(mutually_sub),
        "confidence_substantive_pearson_r": round(conf_r, 3),
        "slogan_density_pearson_r": round(slogan_r, 3),
        "per_dim_pearson_r": {k: round(v, 3) for k, v in per_dim_r.items()},
        "mean_per_dim_pearson_r": round(sum(per_dim_r.values()) / len(per_dim_r), 3),
        "l1_distribution": label_dist,
        "paired_contrast": paired,
    }
    return report


def write_report(mode: str, report: dict) -> tuple[Path, Path]:
    out_json = RESULTS_DIR / f"llm_robustness_{mode}.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md = RESULTS_DIR / f"llm_robustness_{mode}.md"
    lines = [
        f"# LLM robustness: {mode}",
        "",
        f"JSON parse failures: {report['parse_failures']} / {report['n_segments']}",
        f"L1 exact agreement: {report['l1_exact_agreement']:.3f}",
        f"L1 Cohen kappa: {report['l1_cohen_kappa']:.3f}",
        f"L2 Cohen kappa on mutually substantive segments: {report['l2_cohen_kappa_mutually_substantive']:.3f} (n={report['n_mutually_substantive']})",
        f"confidence_substantive Pearson r: {report['confidence_substantive_pearson_r']:.3f}",
        f"slogan_density Pearson r: {report['slogan_density_pearson_r']:.3f}",
        "",
        "Per-dimension stance-score Pearson r:",
    ]
    for dk, r in report["per_dim_pearson_r"].items():
        lines.append(f"- {dk}: {r:.3f}")
    lines.extend([
        f"- mean per-dimension r: {report['mean_per_dim_pearson_r']:.3f}",
        "",
        "LC-vs-SL paired contrast:",
        f"- D_lc: {report['paired_contrast']['D_lc']}",
        f"- D_sl: {report['paired_contrast']['D_sl']}",
        f"- Delta: {report['paired_contrast']['Delta']}",
        f"- Cohen d: {report['paired_contrast']['cohen_d']}",
        f"- exact one-sided p: {report['paired_contrast']['exact_p_1sided']}",
        "",
        "L1 distribution:",
        "```json",
        json.dumps(report["l1_distribution"], ensure_ascii=False, indent=2),
        "```",
    ])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_json, out_md


def run_mode(args):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        opener.open("http://127.0.0.1:11434/api/tags", timeout=10).read()
    except Exception as e:
        raise SystemExit(f"Cannot reach Ollama at 127.0.0.1:11434: {e}")

    segs = list(csv.DictReader((DATA_DIR / "segments.csv").open(encoding="utf-8")))
    if args.limit:
        segs = segs[: args.limit]

    cache_path = DATA_DIR / f"llm_cache_{args.mode}.jsonl"
    failure_path = DATA_DIR / f"llm_failures_{args.mode}.jsonl"
    cache = read_jsonl_by_seg(cache_path)
    failures = read_jsonl_by_seg(failure_path)
    main_cache = read_jsonl_by_seg(MAIN_CACHE)
    print(f"mode={args.mode}; segments={len(segs)}; cache={len(cache)}; failures={len(failures)}")

    seg_lookup = {s["seg_id"]: s for s in segs}
    pending = [s for s in segs if s["seg_id"] not in cache]
    if not args.retry_failures:
        pending = [s for s in pending if s["seg_id"] not in failures]
    print(f"pending={len(pending)}")

    def process_one(s: dict):
        sid = s["seg_id"]
        prompt = build_prompt_for_mode(args.mode, s["text"])
        raw = ""
        last_err = ""
        for attempt in range(args.retry):
            try:
                raw = call_ollama(prompt, args.model, args.num_predict)
                parsed = parse_json_response(raw)
                if parsed is None:
                    last_err = f"parse fail: {raw[:200]}"
                    time.sleep(min(8, 2 ** attempt))
                    continue
                norm = normalize_response(parsed)
                return sid, {"seg_id": sid, "model": args.model, "mode": args.mode, **norm, "_raw": raw[:600]}, None
            except Exception as e:
                last_err = str(e)
                time.sleep(min(8, 2 ** attempt))
        return sid, None, {"seg_id": sid, "mode": args.mode, "error": last_err, "_raw": raw[:600]}

    n_done = 0
    started = time.time()
    if pending:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(process_one, s): s["seg_id"] for s in pending}
            for fut in as_completed(futs):
                sid, entry, err = fut.result()
                if entry:
                    append_jsonl(cache_path, entry)
                    cache[sid] = entry
                else:
                    append_jsonl(failure_path, err)
                    failures[sid] = err
                n_done += 1
                if n_done % args.progress_every == 0 or n_done == len(pending):
                    elapsed = time.time() - started
                    rate = n_done / max(elapsed, 1)
                    remain = (len(pending) - n_done) / max(rate, 0.001) / 60
                    print(f"... {n_done}/{len(pending)} processed; valid={len(cache)} failures={len(failures)} ETA={remain:.1f}min")

    write_scores_csv(args.mode, segs, cache)
    report = compare_runs(args.mode, segs, main_cache, cache, failures)
    out_json, out_md = write_report(args.mode, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(out_json)
    print(out_md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["repeated_zh", "english_full"], required=True)
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--num-predict", type=int, default=320)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--retry", type=int, default=3)
    ap.add_argument("--retry-failures", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--progress-every", type=int, default=25)
    args = ap.parse_args()
    run_mode(args)


if __name__ == "__main__":
    main()
