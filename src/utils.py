"""Shared utilities: paths, encoding, paragraph splitting."""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "raw_corpus"
PHASE0_DIR = ROOT / "phase0"
PHASE1_DIR = ROOT
DATA_DIR = PHASE1_DIR / "data"
RESULTS_DIR = PHASE1_DIR / "results"
PROMPTS_DIR = PHASE1_DIR / "prompts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    for enc in ("utf-8", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


# Discourse markers used to force-split overly long paragraphs.
SECOND_LEVEL_SPLIT_RE = re.compile(
    r"(?<=[。！？])\s*(?=(?:第[一二三四五六七八九十]+[、，是]|首先|其次|再次|最后|此外|另外|同时|然后))"
)


def split_paragraphs(text: str, max_chars: int = 600) -> list[str]:
    """Split text into paragraphs by blank lines, then force-split overly long ones."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    intermediate = []
    for b in blocks:
        b = re.sub(r"\s+", " ", b)
        if len(b) <= max_chars:
            intermediate.append(b)
            continue
        parts = SECOND_LEVEL_SPLIT_RE.split(b)
        intermediate.extend(p.strip() for p in parts if p.strip())
    # Final pass: any segment still > max_chars gets hard-wrapped at sentence boundaries.
    out = []
    for s in intermediate:
        if len(s) <= max_chars:
            out.append(s)
            continue
        sents = re.split(r"(?<=[。！？])", s)
        buf = ""
        for sent in sents:
            if len(buf) + len(sent) > max_chars and buf:
                out.append(buf.strip())
                buf = sent
            else:
                buf += sent
        if buf.strip():
            out.append(buf.strip())
    return [p for p in out if len(p) >= 10]


def parse_filename(name: str) -> dict:
    """Mirror of phase0/01_inventory.py logic."""
    wave = "B" if name.endswith("2.txt") else "A"
    stem = name[:-4]
    if wave == "B":
        stem = stem[:-1]
    parts = stem.split("-")
    return {
        "file": name,
        "company": parts[0] if parts else "",
        "speaker": parts[1] if len(parts) > 1 else "",
        "industry": parts[2] if len(parts) > 2 else "",
        "wave": wave,
    }


def normalize_company(c: str) -> str:
    c = c.replace("有限责任公司", "").replace("有限公司", "")
    c = c.replace("（集团）", "").replace("(集团)", "")
    c = c.replace("集团", "").replace(" ", "").strip()
    return c
