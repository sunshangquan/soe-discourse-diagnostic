# Evidence pack for Reviewer cHmu

## Reviewer concern map

| Concern | Evidence now available | Rebuttal stance |
|---|---|---|
| Small corpus | small-data balanced subsampling; exact permutation | Accept limitation; show paired diagnostic is not driven by one/few pairs |
| Single annotator | three-annotator IAA | Resolved for pilot labels |
| Paraphrase robustness self-referential | cross-model agreement + downgraded paraphrase claim | Treat paraphrase as stress test, not independent validation |
| Requested controls | sanitized TF-IDF, multi-seed Doc2Vec, DICT-on-substantive variants | Add to Table 3/appendix; they downgrade the LLM claim |
| Calibration formula | ablation/grid | Confidence multiplier drives lift; slogan-density penalties are not the source |
| Stability/parse failures | cross-model subset; parse stats | Report what is tested; mark unquantized/English-prompt as future work |

## Requested controls

| Control | Delta | Cohen d | exact p1 | Interpretation |
|---|---:|---:|---:|---|
| tfidf_doc_vector | 0.235438 | 1.790 | 0.0039 | Generic lexical document vector; bounds non-LLM speaker/topic signal. |
| tfidf_sanitized_names_years | 0.237981 | 1.807 | 0.0039 | TF-IDF after removing company/person names and years. |
| tfidf_sanitized_names_years_numbers | 0.239165 | 1.808 | 0.0038 | TF-IDF after removing names, years, Arabic numerals, and alphanumeric tokens. |
| tfidf_sanitized_plus_slogan_terms | 0.238901 | 1.803 | 0.0039 | TF-IDF after also removing a conservative political-slogan term list. |
| doc2vec_doc_vector_seed_summary | 0.162965 | 0.983 | 0.0207 | Doc2Vec over 5 seeds; Delta mean/range=0.163/[0.156,0.175], d mean/range=0.983/[0.921,1.053], p1 mean/range=0.0207/[0.0127,0.0271]. |
| dict_on_llm_substantive_conditional | 0.140769 | 0.657 | 0.0042 | Conditional profile: old dictionary averaged only over LLM-substantive segments. |
| dict_on_llm_substantive_mass | 0.140769 | 0.657 | 0.0042 | Document-level mass: non-substantive segments set to zero before document averaging. |

Interpretation: these controls strengthen the reviewer's caution. Raw TF-IDF is stronger than the LLM paired contrast, and the contrast remains after removing firm/person names, years, numbers, and a conservative political-slogan term list. Therefore the LC-vs-SL paired contrast alone cannot validate stance recovery; it is a diagnostic that reveals construct-validity risk from generic lexical/speaker/time/style signal.

## Calibration and Equation 1

| Variant | Delta | Cohen d |
|---|---:|---:|
| raw | 0.1047 | 0.871 |
| x conf_sub only | 0.1331 | 0.669 |
| x (1-sd_llm) only | 0.1013 | 0.838 |
| x (1-2*sd_ng) only | 0.1045 | 0.874 |
| x conf_sub x (1-sd_llm) | 0.1300 | 0.654 |
| x (1-sd_llm) x (1-2*sd_ng) | 0.1011 | 0.840 |
| all (default) | 0.1299 | 0.654 |

Lambda grid: standard pooled-SD d ranges from 0.625 to 0.670; Delta ranges from 0.1243 to 0.1331. The flat landscape shows that the full formula should not be presented as an optimized estimator.

## Robustness and stability

* Balanced 5-vs-5 sensitivity check (overlapping subsets, not independent replications): raw LLM median d=1.040, Delta>0 in 98.7%; calibrated median d=0.931, Delta>0 in 98.2%.
* Style residualization: raw LLM d=0.408, p=0.213; calibrated d=0.371, p=0.226. This supports reframing the abstract around a diagnostic/negative result rather than a positive stance-recovery claim.
* Cross-model subset: qwen3.5:27b vs 9b on 85 shared segments gives L1 kappa=0.746, mean per-dimension Pearson r=0.882, confidence_substantive r=0.89.
* Cross-family subset: deepseek-r1:8b vs 9b on 89 shared segments gives L1 kappa=0.695, mean per-dimension r=0.764, confidence_substantive r=0.844.
* Paraphrase stress test: 50 self-generated rewrites; L1 stability 0.78; because rewrites are model-generated, this should be described only as a stress test.
* JSON parse accounting: 0 outer parse failures and 0 inner raw parse failures out of 2190 segments.

## Annotation reliability

Three-annotator summary from `three_annotator_l1_report.md`:

* Items: 86
* All-three exact agreement: 0.791 (68/86)
* At least two annotators agree: 1.000 (86/86)
* Fleiss' kappa: 0.719
* Krippendorff's alpha, nominal: 0.719
* | A1-A2 | 0.797 |
* | A1-A3 | 0.728 |
* | A2-A3 | 0.631 |

## Untested items to state honestly

* We have not run an unquantized-weight version.
* We have not run a full English-prompt version over all 2,190 segments.
* Calibration parameters were not selected on a disjoint held-out set; therefore the calibrated variant should be treated as an ablation/diagnostic transformation, with raw LLM as the cleaner main result.
