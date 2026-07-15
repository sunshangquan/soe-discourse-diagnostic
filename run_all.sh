#!/usr/bin/env bash
# Phase 1 end-to-end pipeline.
#
# Usage:
#   bash run_all.sh             # full local-LLM pipeline when Ollama is available
#   bash run_all.sh --mock      # mock LLM for offline testing
#
# Each step skips work that already exists. Delete the relevant file in
# data/ or results/ to force re-run of a step.

set -euo pipefail

cd "$(dirname "$0")"
HERE="$(pwd)"
SRC="$HERE/src"

MOCK_MODE=""
for arg in "$@"; do
    case "$arg" in
        --mock) MOCK_MODE="--mock" ;;
        *) echo "Unknown arg: $arg" ; exit 2 ;;
    esac
done

echo "=== [1/6] Build segments.csv ==="
[[ -f data/segments.csv ]] || python3 "$SRC/corpus.py"

echo "=== [2/6] Parse gold annotations ==="
[[ -f data/gold.csv ]] || python3 "$SRC/gold_parser.py"

echo "=== [3/6] Dictionary baseline ==="
[[ -f results/scores_dict.csv ]] || python3 "$SRC/score_dict.py"

echo "=== [4/6] LDA baseline ==="
[[ -f results/scores_lda.csv ]] || python3 "$SRC/score_lda.py"

echo "=== [5/6] BGE-zh baseline ==="
if ! [[ -f results/scores_bge.csv ]]; then
    if python3 -c "import sentence_transformers" 2>/dev/null; then
        python3 "$SRC/score_bge.py"
    else
        echo "  skip BGE (sentence-transformers not installed)"
    fi
fi

echo "=== [6/6] LLM stance extraction ==="
if ! [[ -f results/scores_llm.csv ]]; then
    if [[ -n "$MOCK_MODE" ]]; then
        python3 "$SRC/score_llm.py" --mock
    elif command -v ollama &>/dev/null && curl -s http://localhost:11434/api/tags &>/dev/null; then
        # Prefer free local Ollama
        MODEL="qwen3.5:9b"
        echo "  Using local Ollama with $MODEL (~6h on M4 Pro for 2190 segs)"
        echo "  Tip: prefer 'caffeinate -i bash run_all.sh' to prevent sleep"
        python3 "$SRC/score_llm.py" --provider ollama --model "$MODEL" --resume
    else
        echo "  No LLM backend available — falling back to --mock."
        echo "  Install/start Ollama for the local real-LLM path."
        python3 "$SRC/score_llm.py" --mock
    fi
fi

echo "=== Slogan calibration ==="
python3 "$SRC/calibrate.py"

echo "=== Evaluation ==="
python3 "$SRC/evaluate.py"

echo ""
echo "Done. Inspect:"
echo "  results/eval_summary.md       — main results table"
echo "  results/scores_*.csv          — per-method per-segment scores"
echo "  data/llm_cache.jsonl          — raw LLM responses (if real run)"
