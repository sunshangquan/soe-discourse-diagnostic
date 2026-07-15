# Phase 1 Evaluation Summary

Methods loaded: dict, lda, bge, llm, calibrated
Segments: 2190
Leader-change pairs: 24, Same-leader pairs: 5

## Channel 1 — Paired contrast

| method | D(leader-change) | D(same-leader) | diff | Cohen d |
|---|---|---|---|---|
| dict | 0.1471 | 0.0487 | 0.0983 | 0.807 |
| lda | 0.2128 | 0.1478 | 0.065 | 0.2 |
| bge | 0.0012 | 0.0005 | 0.0007 | 0.654 |
| llm | 0.1488 | 0.0441 | 0.1047 | 1.088 |
| calibrated | 0.179 | 0.049 | 0.1299 | 0.835 |

## Channel 2 — Gold L1 F1 (substantive vs slogan)

| method | F1(substantive) | Precision | Recall | n |
|---|---|---|---|---|
| dict | 0.489 | 0.479 | 0.5 | 83 |
| lda | 0.558 | 0.5 | 0.63 | 83 |
| bge | 0.547 | 0.483 | 0.63 | 83 |
| llm | 0.781 | 0.695 | 0.891 | 83 |
| calibrated | 0.781 | 0.695 | 0.891 | 83 |

## Channel 3 — External validity

TODO: collect data/external_signals.csv (company_norm, year, signal_name, value)
      e.g. patent counts, ESG scores, R&D intensity per company-year.