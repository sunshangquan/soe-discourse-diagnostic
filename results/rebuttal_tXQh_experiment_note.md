# Rebuttal note for Reviewer tXQh

Reviewer concern: the contribution may look light because the corpus has only 80 documents from 51 companies.

New analysis completed for the response:

* We ran an exhaustive balanced small-sample check. The full paired design has 24 leader-change pairs and 5 same-leader controls. To test whether the result is driven by the larger leader-change side, we evaluated all 42,504 ways of selecting only 5 leader-change pairs, and compared each subset against the same 5 same-leader controls.
* llm: full Cohen d=0.871, full Delta=0.1047; balanced median d=1.040, 5--95%=[0.359, 2.042], Delta>0 in 98.7% of balanced subsets, d>=0.5 in 90.9%.
* calibrated: full Cohen d=0.654, full Delta=0.1299; balanced median d=0.931, 5--95%=[0.290, 1.762], Delta>0 in 98.2% of balanced subsets, d>=0.5 in 88.6%.
* lda: full Cohen d=0.201, full Delta=0.0650; balanced median d=0.153, 5--95%=[-0.625, 0.799], Delta>0 in 67.0% of balanced subsets, d>=0.5 in 16.7%.
* We also ran leave-one-leader-change-out and leave-one-same-leader-out checks; the LLM contrast keeps positive Delta under all omissions, whereas LDA remains weak and unstable.
* Together with the exact label-permutation test already run on all 29 pairs (LLM one-sided p=0.0339; calibrated p=0.0416; LDA p=0.3527), this supports the claim that the paper's contribution is the paired diagnostic/evaluation design rather than corpus scale alone.
