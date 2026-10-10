# Anonymous Artifact Manifest

This artifact supports the experiments reported in *Slogans or Stance?*.
It is prepared for anonymous peer review and contains no author names,
email addresses, repository history, or non-anonymous repository links.

## Included

- `data/segments.csv`: 2,190 derived speech segments and document metadata.
- `src/dimensions.py`: the implemented dictionary with 10 Chinese seed terms
  for each of five dimensions (50 terms total).
- `phase0/shared_ngrams_top200.tsv`: the exact 53-entry slogan n-gram input.
- `results/three_annotator_l1_merged.csv`: 86 aligned three-annotator items,
  including the majority L1 labels used by the evaluation (83 binary items).
- `data/llm_cache*.jsonl`: cached raw outputs for the reported LLM runs.
- `results/scores_*.csv`: 2,190 segment-level scores for each reported main
  method and the repeated/English-prompt runs.
- `src/`, `tests/`, and `run_all.sh`: scoring, calibration, evaluation,
  control, robustness, and integrity-check code.
- `results/`: the saved tables and diagnostics referenced by the paper.

## Reproduction scope

The main calibration, pilot evaluation, bootstrap summaries, and exact
permutation tests can be regenerated offline from the included derived data
and cached model outputs. Run:

```bash
bash run_all.sh
```

The integrity test also creates a fresh temporary copy and compares core
regenerated outputs byte-for-byte with the saved files.

Re-running neural inference requires the optional dependencies and local model
tags documented in `README.md`. The artifact does not include local model
weights or an exact historical Ollama digest, which was not logged.

## Source-text boundary

The original raw `.txt` webpages are not redistributed in this anonymous
artifact. The included `segments.csv` is the derived research corpus used by
the released evaluation. Public speeches originated from SASAC or SOE public
websites; source documentation will accompany the final archival release.

