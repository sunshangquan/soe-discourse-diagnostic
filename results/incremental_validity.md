# Incremental-validity diagnostic

Pair-level LLM cosine distance residualized on TF-IDF and/or pure style distance.

| Method | Delta | Cohen d | exact p1 |
|---|---:|---:|---:|
| raw_llm_pair_distance | 0.104704 | 0.870882 | 0.033902 |
| llm_residualized_on_tfidf | -0.009106 | -0.090624 | 0.582207 |
| llm_residualized_on_style | 0.086995 | 0.731198 | 0.070119 |
| llm_residualized_on_tfidf_and_style | -0.013973 | -0.140582 | 0.618913 |
| llm_lopo_residualized_on_tfidf | -0.028410 | -0.257918 | 0.704391 |
| llm_lopo_residualized_on_style | 0.093032 | 0.720848 | 0.073277 |
| llm_lopo_residualized_on_tfidf_and_style | -0.035205 | -0.308314 | 0.736820 |

## Regression diagnostics

- {'model': 'tfidf', 'r_squared': '0.373524', 'pearson_r_tfidf_y': '0.611166', 'standardized_beta_tfidf': '0.611166'}
- {'model': 'style', 'r_squared': '0.050810', 'pearson_r_style_y': '0.225410', 'standardized_beta_style': '0.225410'}
- {'model': 'tfidf_and_style', 'r_squared': '0.385833', 'pearson_r_tfidf_y': '0.611166', 'standardized_beta_tfidf': '0.589623', 'pearson_r_style_y': '0.225410', 'standardized_beta_style': '0.113019'}
