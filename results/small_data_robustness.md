# Small-data robustness checks

Corpus: 80 documents from 51 companies; paired evaluation uses 24 leader-change pairs and 5 same-leader controls.

Balanced check: for each method, exhaustively evaluate every 5-pair subset of the leader-change side (42,504 subsets), so the comparison is 5 vs. 5 pairs.

| method | full d | full Delta | balanced median d | balanced d 5-95% | pct Delta>0 | pct d>=0.5 | LOO-LC d range | LOO-SL d range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dict | 0.636 | 0.0983 | 0.768 | [0.025, 1.524] | 95.5% | 77.2% | [0.586, 0.674] | [0.565, 0.744] |
| lda | 0.201 | 0.0650 | 0.153 | [-0.625, 0.799] | 67.0% | 16.7% | [0.121, 0.227] | [0.086, 0.697] |
| bge | 0.536 | 0.0007 | 0.634 | [-0.107, 1.200] | 91.8% | 66.3% | [0.479, 0.571] | [0.451, 0.722] |
| llm | 0.871 | 0.1047 | 1.040 | [0.359, 2.042] | 98.7% | 90.9% | [0.827, 0.933] | [0.774, 0.971] |
| calibrated | 0.654 | 0.1299 | 0.931 | [0.290, 1.762] | 98.2% | 88.6% | [0.613, 0.800] | [0.586, 0.722] |
