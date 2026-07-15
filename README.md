# SOE Discourse Diagnostic

This repository contains the code, prompts, derived data, model outputs, and evaluation summaries for a diagnostic study of entrepreneurial-discourse extraction in Chinese state-owned enterprise (SOE) leadership speeches.

The central question is whether common discourse-measurement instruments recover leader-level substantive stance or instead capture recurrent slogans, lexical variation, speaker style, and institutional discourse templates.

## Contents

- `src/`: scoring, calibration, evaluation, robustness, and reviewer-requested control scripts.
- `data/segments.csv`: 2,190 speech segments with document metadata.
- `data/gold.csv`: pilot gold labels for slogan/substantive/irrelevant evaluation.
- `data/llm_cache*.jsonl`: raw model outputs used by the reported LLM runs.
- `results/scores_*.csv`: per-segment scores for dictionary, LDA, sentence-encoder, LLM, calibrated LLM, and robustness runs.
- `results/eval_summary.md`: main evaluation summary.
- `results/rebuttal_controls.md`: TF-IDF, sanitized TF-IDF, Doc2Vec, and DICT-on-LLM-substantive controls.
- `results/llm_robustness_*.md`: deterministic repeated-run and English-prompt robustness checks.
- `results/three_annotator_l1_report.md`: multi-annotator agreement summary for the pilot label set.
- `prompts/stance_extraction.txt`: Chinese prompt template for segment-level extraction.

## Quick Start

The repository includes the derived segment file and reported model outputs, so the main evaluations can be reproduced without re-running the LLM.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 src/calibrate.py
python3 src/evaluate.py
```

If local Ollama is available, the full LLM scoring can be re-run with:

```bash
python3 src/score_llm.py --provider ollama --model qwen3.5:9b --resume
```

Optional robustness/control scripts:

```bash
# Slower: trains TF-IDF/Doc2Vec controls and rewrites results/rebuttal_controls.*
python3 src/rebuttal_controls.py

# Requires local Ollama if cache entries are missing; reported outputs are already in results/
python3 src/llm_robustness_full.py --mode repeated_zh
python3 src/llm_robustness_full.py --mode english_full
```

For a lightweight offline smoke test:

```bash
bash run_all.sh --mock
```

## Key Reported Checks

Full deterministic repeated run:

- JSON parse failures: `0 / 2190`
- L1 agreement: `1.000`
- L1 Cohen's kappa: `1.000`
- Mean per-dimension stance-score Pearson correlation: `1.000`
- LC-vs-SL contrast: `Delta=0.104704`, `d=1.088`, exact one-sided `p=0.0339`

Full English-prompt run:

- JSON parse failures: `0 / 2190`
- L1 agreement with Chinese-prompt run: `0.928`
- L1 Cohen's kappa: `0.844`
- Mean per-dimension stance-score Pearson correlation: `0.841`
- LC-vs-SL contrast: `Delta=0.067602`, `d=0.923`, exact one-sided `p=0.0635`

Control results:

- Raw TF-IDF document vectors: `Delta=0.235`, `d=1.186`, `p=0.0039`
- Sanitized TF-IDF remains strong after removing company/person names, years, numbers, and a conservative slogan-term list.
- DICT-on-LLM-substantive: `Delta=0.141`, `d=0.857`, `p=0.0042`

These controls support the paper's diagnostic claim: passing the leader-change contrast is not sufficient to validate stance recovery, because generic lexical/document variation can also pass the diagnostic.

## Notes on Raw Sources

The repository ships derived segment-level data and model outputs used for evaluation. Public source speeches were released by SASAC or SOE websites as part of the SOE Open Lecture initiative. Source URLs and release documentation should be checked against the final paper or accompanying artifact notes.

If raw `.txt` speeches are available locally, place them under:

```text
raw_corpus/
```

Then run:

```bash
python3 src/corpus.py
```

## License

Code is released under the MIT License. Data are provided for research reproducibility; users are responsible for respecting the terms of the original public source pages.
