# Phase 1 Evaluation Summary

Methods loaded: dict, lda, bge, llm, calibrated
Segments: 2190
Leader-change pairs: 24, Same-leader pairs: 5

## Channel 1 — Paired contrast

| method | D(leader-change) | D(same-leader) | diff | Cohen d |
|---|---|---|---|---|
| dict | 0.1471 | 0.0487 | 0.0983 | 0.636 |
| lda | 0.2128 | 0.1478 | 0.065 | 0.201 |
| bge | 0.0012 | 0.0005 | 0.0007 | 0.536 |
| llm | 0.1488 | 0.0441 | 0.1047 | 0.871 |
| calibrated | 0.179 | 0.049 | 0.1299 | 0.654 |

## Channel 2 — Gold L1 F1 (substantive vs slogan)

| method | F1(substantive) | Precision | Recall | n |
|---|---|---|---|---|
| dict | 0.495 | 0.5 | 0.49 | 83 |
| lda | 0.598 | 0.552 | 0.653 | 83 |
| bge | 0.587 | 0.533 | 0.653 | 83 |
| llm | 0.796 | 0.729 | 0.878 | 83 |
| calibrated | 0.796 | 0.729 | 0.878 | 83 |

## Channel 3 — External validity

TODO: collect data/external_signals.csv (company_norm, year, signal_name, value)
      e.g. patent counts, ESG scores, R&D intensity per company-year.