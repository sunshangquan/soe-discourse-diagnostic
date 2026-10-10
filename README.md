# SOE Discourse Diagnostic

This repository contains the code, prompts, derived data, model outputs, and evaluation summaries for a diagnostic study of entrepreneurial-discourse extraction in Chinese state-owned enterprise (SOE) leadership speeches.

The central question is whether common discourse-measurement instruments recover leader-level substantive stance or instead capture recurrent slogans, lexical variation, speaker style, and institutional discourse templates.

## Contents

- `src/`: scoring, calibration, evaluation, robustness, and reviewer-requested control scripts.
- `data/segments.csv`: 2,190 speech segments with document metadata.
- `data/gold.csv`: original single-annotator pilot labels, retained for provenance; not the default evaluation labels.
- `results/three_annotator_l1_merged.csv`: three-annotator L1 labels and majority decisions used by default evaluation.
- `phase0/shared_ngrams_top200.tsv`: exact 53-entry mining input required to regenerate n-gram densities.
- `data/llm_cache*.jsonl`: raw model outputs used by the reported LLM runs.
- `results/scores_*.csv`: per-segment scores for dictionary, LDA, sentence-encoder, LLM, calibrated LLM, and robustness runs.
- `results/eval_summary.md`: main evaluation summary.
- `results/rebuttal_controls.md`: TF-IDF, sanitized TF-IDF, Doc2Vec, and DICT-on-LLM-substantive controls.
- `results/incremental_validity.md`: in-sample and leave-one-pair-out Stage-2 residualisation checks.
- `results/style_doc_features.csv`: cached document-level style features used when raw speeches are unavailable.
- `results/covariate_sensitivity.md`: sensitivity to excluding the SASAC regulator pair.
- `results/llm_robustness_*.md`: deterministic repeated-run and English-prompt robustness checks.
- `results/three_annotator_l1_report.md`: multi-annotator agreement summary for the pilot label set.
- `src/dimensions.py`: the exact implemented dictionary seeds (10 terms for each of five dimensions).
- `prompts/stance_extraction.txt`: Chinese prompt template for segment-level extraction.

## Quick Start

The repository includes the derived segment file and reported model outputs, so
the offline calibration and main evaluations can be reproduced with Python 3's
standard library, without re-running the LLM or installing model dependencies.

```bash
python3 src/calibrate.py
python3 src/evaluate.py
python3 src/pr_auc_eval.py
python3 src/stats_addon.py
python3 src/permutation_test.py
python3 -m unittest discover -s tests -v
```

The evaluation scripts use the 86 aligned three-annotator majority L1 labels
(83 binary slogan/substantive items); they do not use `data/gold.csv` by default.
The offline test makes a fresh temporary copy and verifies regenerated
calibration, pilot evaluation, bootstrap summary, and exact permutation outputs
byte-for-byte against the saved files. No model inference is required for this check.

If local Ollama is available, the full LLM scoring can be re-run with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/score_llm.py --provider ollama --model qwen3.5:9b --resume
```

Optional robustness/control scripts:

```bash
# Slower: trains TF-IDF/Doc2Vec controls and rewrites results/rebuttal_controls.*
python3 src/rebuttal_controls.py
python3 src/incremental_validity.py
python3 src/covariate_sensitivity.py

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
- LC-vs-SL contrast: `Delta=0.104704`, `d=0.871`, exact one-sided `p=0.0339`

Full English-prompt run:

- JSON parse failures: `0 / 2190`
- L1 agreement with Chinese-prompt run: `0.928`
- L1 Cohen's kappa: `0.844`
- Mean per-dimension stance-score Pearson correlation: `0.841`
- LC-vs-SL contrast: `Delta=0.067602`, `d=0.761`, exact one-sided `p=0.0635`

Control results:

- Raw TF-IDF document vectors: `Delta=0.235`, `d=1.790`, `p=0.0039`
- Sanitized TF-IDF remains strong after removing company/person names, years, numbers, and a conservative slogan-term list.
- DICT-on-LLM-substantive: `Delta=0.141`, `d=0.657`, `p=0.0042`
- LOPO LLM residualised on TF-IDF: `Delta=-0.028`, `d=-0.258`, `p=0.704`
- LOPO LLM residualised on TF-IDF and style: `Delta=-0.035`, `d=-0.308`, `p=0.737`
- Excluding the SASAC SL pair: LLM `d=0.787`, `p=0.075`; TF-IDF `d=1.680`, `p=0.0127`

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
