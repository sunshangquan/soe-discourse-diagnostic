# Small-data robustness checks

Corpus: 80 documents from 51 companies; paired evaluation uses 24 leader-change pairs and 5 same-leader controls.

Balanced check: for each method, exhaustively evaluate every 5-pair subset of the leader-change side (42,504 subsets), so the comparison is 5 vs. 5 pairs.

| method | full d | full Delta | balanced median d | balanced d 5-95% | pct Delta>0 | pct d>=0.5 | LOO-LC d range | LOO-SL d range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dict | 0.807 | 0.0983 | 0.768 | [0.025, 1.524] | 95.5% | 77.2% | [0.738, 0.852] | [0.728, 0.983] |
| lda | 0.200 | 0.0650 | 0.153 | [-0.625, 0.799] | 67.0% | 16.7% | [0.117, 0.227] | [0.082, 0.927] |
| bge | 0.654 | 0.0007 | 0.634 | [-0.107, 1.200] | 91.8% | 66.3% | [0.577, 0.696] | [0.554, 0.949] |
| llm | 1.088 | 0.1047 | 1.040 | [0.359, 2.042] | 98.7% | 90.9% | [1.029, 1.162] | [0.985, 1.252] |
| calibrated | 0.835 | 0.1299 | 0.931 | [0.290, 1.762] | 98.2% | 88.6% | [0.779, 0.978] | [0.761, 0.944] |
