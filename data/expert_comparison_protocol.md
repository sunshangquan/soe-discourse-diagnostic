# Expert comparison protocol

Frozen before examining model-versus-expert scores.

- Use all 80 documents and their existing 2,190 scored segments; retain the
  original corpus and the two individual expert labels. The released labels
  already contain document IDs mapped from the original task IDs. Check for
  duplicate or missing IDs.
- For each of the five existing dimensions, average each method's existing
  segment scores within the document, following the paper's aggregation.
  Compare DICT, LDA, BGE, raw LLM, and CALIBRATED on the same documents.
- For each expert separately, `明确立场` is positive; `仅涉及` and `未涉及`
  are negative for the specific question of explicit, attributable stance.
  `无法判断` is not a negative label: if present, report its count and pause
  to determine treatment before calculating metrics. Do not silently drop it.
- Report standard average precision (AP) and ROC-AUC, plus positive prevalence
  and sample counts for each dimension and expert. AP's no-information reference
  is positive prevalence. These are exploratory results for this newly annotated
  corpus, not an official benchmark or a model-selection test.
- Do not tune thresholds, select favourable dimensions, merge annotators into a
  fabricated consensus, or claim that these labels validate behaviour outside
  the speeches. Check the aggregation against an existing published number
  before relying on a new implementation. Preserve both original files.
