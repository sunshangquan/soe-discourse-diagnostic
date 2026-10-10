# Covariate sensitivity

The only regulator pair (SASAC) is removed from the SL group. Industry matching is not estimated because 3/5 SL pairs lack an industry tag.

| method | analysis | n LC | n SL | Delta | Cohen d | exact p1 |
|---|---|---:|---:|---:|---:|---:|
| llm | full | 24 | 5 | 0.104704 | 0.870882 | 0.033902 |
| llm | exclude_sasac | 24 | 4 | 0.096233 | 0.786909 | 0.074969 |
| lda | full | 24 | 5 | 0.065030 | 0.200971 | 0.352701 |
| lda | exclude_sasac | 24 | 4 | 0.028145 | 0.085767 | 0.452503 |
| tfidf | full | 24 | 5 | 0.235438 | 1.790188 | 0.003916 |
| tfidf | exclude_sasac | 24 | 4 | 0.224609 | 1.680148 | 0.012698 |
