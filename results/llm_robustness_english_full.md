# LLM robustness: english_full

JSON parse failures: 0 / 2190
L1 exact agreement: 0.928
L1 Cohen kappa: 0.844
L2 Cohen kappa on mutually substantive segments: 0.244 (n=1339)
confidence_substantive Pearson r: 0.860
slogan_density Pearson r: 0.858

Per-dimension stance-score Pearson r:
- innovation: 0.918
- competition_cooperation: 0.824
- organization_market: 0.842
- social_responsibility: 0.855
- national_mission: 0.765
- mean per-dimension r: 0.841

LC-vs-SL paired contrast:
- D_lc: 0.098336
- D_sl: 0.030733
- Delta: 0.067602
- Cohen d: 0.923
- exact one-sided p: 0.0635

L1 distribution:
```json
{
  "slogan": 778,
  "substantive": 1409,
  "irrelevant": 3
}
```
