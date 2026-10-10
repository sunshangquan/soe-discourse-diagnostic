# Rebuttal controls

Pairs: leader-change=24, same-leader=5.

| Method | D_lc | D_sl | Delta | Cohen d | exact p1 | Note |
|---|---:|---:|---:|---:|---:|---|
| tfidf_doc_vector | 0.694669 | 0.459231 | 0.235438 | 1.790 | 0.0039 | Generic lexical document vector; bounds non-LLM speaker/topic signal. |
| tfidf_sanitized_names_years | 0.699802 | 0.461821 | 0.237981 | 1.807 | 0.0039 | TF-IDF after removing company/person names and years. |
| tfidf_sanitized_names_years_numbers | 0.697022 | 0.457857 | 0.239165 | 1.808 | 0.0038 | TF-IDF after removing names, years, Arabic numerals, and alphanumeric tokens. |
| tfidf_sanitized_plus_slogan_terms | 0.698827 | 0.459926 | 0.238901 | 1.803 | 0.0039 | TF-IDF after also removing a conservative political-slogan term list. |
| doc2vec_doc_vector_seed_1 | 0.337912 | 0.175939 | 0.161973 | 1.008 | 0.0190 | Generic trained document embedding; seed=1, vector_size=100, epochs=80. |
| doc2vec_doc_vector_seed_7 | 0.339805 | 0.183047 | 0.156758 | 0.921 | 0.0271 | Generic trained document embedding; seed=7, vector_size=100, epochs=80. |
| doc2vec_doc_vector_seed_13 | 0.344879 | 0.179968 | 0.164911 | 0.996 | 0.0197 | Generic trained document embedding; seed=13, vector_size=100, epochs=80. |
| doc2vec_doc_vector_seed_21 | 0.343321 | 0.168594 | 0.174727 | 1.053 | 0.0127 | Generic trained document embedding; seed=21, vector_size=100, epochs=80. |
| doc2vec_doc_vector_seed_42 | 0.340025 | 0.183571 | 0.156454 | 0.938 | 0.0248 | Generic trained document embedding; seed=42, vector_size=100, epochs=80. |
| doc2vec_doc_vector_seed_summary |  |  | 0.162965 | 0.983 | 0.0207 | Doc2Vec over 5 seeds; Delta mean/range=0.163/[0.156,0.175], d mean/range=0.983/[0.921,1.053], p1 mean/range=0.0207/[0.0127,0.0271]. |
| dict_on_llm_substantive_conditional | 0.155836 | 0.015067 | 0.140769 | 0.657 | 0.0042 | Conditional profile: old dictionary averaged only over LLM-substantive segments. |
| dict_on_llm_substantive_mass | 0.155836 | 0.015067 | 0.140769 | 0.657 | 0.0042 | Document-level mass: non-substantive segments set to zero before document averaging. |

## LLM cache / JSON parse accounting

```json
{
  "total": 2190,
  "valid_outer": 2190,
  "valid_inner": 2190,
  "outer_parse_failures": 0,
  "inner_raw_parse_failures": 0,
  "l1_distribution": {
    "slogan": 760,
    "substantive": 1419,
    "irrelevant": 11
  }
}
```
