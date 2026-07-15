"""Parse phase0/coding_{1..5}.md into a structured gold CSV.

Strategy:
  1. Extract per-paragraph entries (P_i, L1, L2, preview) from each coding_*.md
  2. For each entry, find matching auto-split segment in segments.csv by either:
     - first-30-chars Jaccard overlap > threshold, or
     - prefix match of normalized characters
  3. Drop entries that fail to match (rare; reported as warnings)

Output: data/gold.csv with columns:
  doc_id, seg_id (matched), gold_p_idx, l1, l2, preview, matched_text_preview, match_score
"""
from __future__ import annotations
import re
import csv
from pathlib import Path
from collections import defaultdict

from utils import DATA_DIR, PHASE0_DIR

# Map coding_{i}.md → doc_id (parsed from the H1 line of each file)
CODING_FILES = [PHASE0_DIR / f"coding_{i}.md" for i in range(1, 6)]

PARA_HEADER_RE = re.compile(r"^###\s+P(\d+)\s*\[(.+?)\]\s*$")
QUOTE_RE = re.compile(r"^>\s*[\"\"](.+?)[\"\"]?$")
H1_RE = re.compile(r"^#\s+Coding\s+\d+:\s*(.+\.txt)\s*$")


def parse_coding_md(path: Path) -> tuple[str, list[dict]]:
    """Return (doc_id, [entries])."""
    doc_id = None
    entries = []
    cur = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if doc_id is None:
            m = H1_RE.match(line)
            if m:
                doc_id = m.group(1).strip()
                continue
        m = PARA_HEADER_RE.match(line)
        if m:
            if cur is not None:
                entries.append(cur)
            p_idx = int(m.group(1))
            tag = m.group(2).strip()
            parts = [t.strip() for t in tag.split("|")]
            l1 = parts[0]
            l2 = parts[1] if len(parts) > 1 else ""
            cur = {"p_idx": p_idx, "l1": l1, "l2": l2, "preview": ""}
            continue
        m = QUOTE_RE.match(line)
        if m and cur is not None and not cur["preview"]:
            cur["preview"] = m.group(1).strip()
    if cur is not None:
        entries.append(cur)
    return doc_id, entries


def normalize_for_match(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", s)


def jaccard_char(a: str, b: str, n: int = 4) -> float:
    if len(a) < n or len(b) < n:
        return 0.0
    ga = set(a[i : i + n] for i in range(len(a) - n + 1))
    gb = set(b[i : i + n] for i in range(len(b) - n + 1))
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def clean_preview_prefix(preview: str) -> str:
    """Take only the part of preview before the first '...' or '…' marker."""
    for sep in ("...", "…", "。。。"):
        idx = preview.find(sep)
        if idx != -1:
            preview = preview[:idx]
    return preview.strip()


def match_entries_to_segments(entries: list[dict], segments_by_doc: list[dict]) -> list[dict]:
    """Match each gold entry to an auto-split segment by strict prefix.

    Strategy:
      1) Take preview prefix before first '...' → normalize
      2) Find segment whose normalized text STARTS with that prefix (length ≥ 8)
      3) If no prefix match, fall back to 4-gram jaccard over first 60 chars
      4) Each segment can be claimed by at most one gold entry
    """
    # Pre-normalize all segments
    seg_norms = []
    for seg in segments_by_doc:
        seg_norms.append({**seg, "_norm": normalize_for_match(seg["text"])})

    matched = []
    used_seg_ids = set()
    for entry in entries:
        cleaned = clean_preview_prefix(entry["preview"])
        preview_norm = normalize_for_match(cleaned)
        if not preview_norm or len(preview_norm) < 6:
            entry["seg_id"] = ""
            entry["matched_text_preview"] = ""
            entry["match_score"] = 0.0
            matched.append(entry)
            continue

        # Use up to 25 chars of cleaned prefix for strict matching
        prefix = preview_norm[:25]
        best = None
        best_score = -1.0
        match_type = "none"

        # Primary: strict prefix
        for seg in seg_norms:
            if seg["seg_id"] in used_seg_ids:
                continue
            if seg["_norm"].startswith(prefix[:8]):
                # score by how much of prefix matches
                k = 0
                while k < len(prefix) and k < len(seg["_norm"]) and seg["_norm"][k] == prefix[k]:
                    k += 1
                score = k / max(8, len(prefix))
                if score > best_score:
                    best_score = score
                    best = seg
                    match_type = "prefix"

        # Fallback: substring (preview may target a middle paragraph)
        if best is None:
            for seg in seg_norms:
                if seg["seg_id"] in used_seg_ids:
                    continue
                if prefix[:10] in seg["_norm"]:
                    score = 0.7
                    if score > best_score:
                        best_score = score
                        best = seg
                        match_type = "substring"

        # Final fallback: jaccard over first 60 chars of normalized seg
        if best is None:
            for seg in seg_norms:
                if seg["seg_id"] in used_seg_ids:
                    continue
                score = jaccard_char(preview_norm[:60], seg["_norm"][:60])
                if score > best_score:
                    best_score = score
                    best = seg
                    match_type = "jaccard"

        if best is not None and best_score >= 0.3:
            used_seg_ids.add(best["seg_id"])
            entry["seg_id"] = best["seg_id"]
            entry["matched_text_preview"] = best["text"][:40]
            entry["match_score"] = round(best_score, 3)
            entry["match_type"] = match_type
        else:
            entry["seg_id"] = ""
            entry["matched_text_preview"] = ""
            entry["match_score"] = 0.0
            entry["match_type"] = "fail"
        matched.append(entry)
    return matched


def main():
    # Load segments.csv grouped by doc_id
    segs = list(csv.DictReader((DATA_DIR / "segments.csv").open(encoding="utf-8")))
    by_doc = defaultdict(list)
    for s in segs:
        by_doc[s["doc_id"]].append(s)

    out_rows = []
    for cf in CODING_FILES:
        doc_id, entries = parse_coding_md(cf)
        print(f"  {cf.name} → doc_id={doc_id}, entries={len(entries)}")
        if doc_id not in by_doc:
            print(f"  WARNING: doc_id {doc_id!r} not found in segments.csv; skipping")
            continue
        matched = match_entries_to_segments(entries, by_doc[doc_id])
        for e in matched:
            out_rows.append({
                "doc_id": doc_id,
                "gold_p_idx": e["p_idx"],
                "seg_id": e["seg_id"],
                "l1": e["l1"],
                "l2": e["l2"],
                "preview": e["preview"][:60],
                "matched_text_preview": e["matched_text_preview"],
                "match_score": e["match_score"],
                "match_type": e.get("match_type", ""),
            })

    out = DATA_DIR / "gold.csv"
    fns = ["doc_id", "gold_p_idx", "seg_id", "l1", "l2", "preview", "matched_text_preview", "match_score", "match_type"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        w.writerows(out_rows)
    n = len(out_rows)
    n_matched = sum(1 for r in out_rows if r["seg_id"])
    n_hi = sum(1 for r in out_rows if r["match_score"] and float(r["match_score"]) >= 0.7)
    print(f"\nWrote {out}: {n} gold entries, matched_to_seg={n_matched}, high_conf(≥0.7)={n_hi}")
    from collections import Counter
    l1c = Counter(r["l1"] for r in out_rows)
    mt = Counter(r["match_type"] for r in out_rows)
    print(f"L1 distribution: {dict(l1c)}")
    print(f"Match types: {dict(mt)}")


if __name__ == "__main__":
    main()
