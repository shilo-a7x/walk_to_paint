# Attention direction / SHAP seed stability (2026-10-15)

Script: experiments/edge_identity_tokens/attn_shap_multiseed.py (production measured with the WSDM
scripts' own functions, unchanged; seed 42 reproduces the WSDM text exactly: b-alpha fwd .443 / bwd .411).
Data: attention_per_seed.csv, shap_per_seed.csv (120 model x seed x dataset results).
"k/10" = number of seeds where the within-seed cluster-robust 95% CI is significant in that direction.

## Attention, layer 0: forward vs backward (fwd sig / bwd sig out of 10 seeds)
| dataset | production | H3 |
|---|---|---|
| b-alpha | 5 / 3 | 6 / 1 |
| b-otc | 2 / 7 | 7 / 3 |
| epinions | 7 / 3 | 6 / 4 |
| slashdot | 10 / 0 | 10 / 0 |
| wiki-elec | 6 / 4 | 10 / 0 |
| wiki-rfa | 4 / 6 | 9 / 1 |
WSDM text (seed 42 only) claims b-alpha+slashdot forward, the other 4 backward. Across seeds only
slashdot is stable in production; wiki-rfa (WSDM: 0.189 vs 0.557) splits 4/6.

## Attention: vertex vs edge tokens (vertex sig / edge sig)
| dataset | production | H3 |
|---|---|---|
| b-alpha | 8 / 2 | 9 / 1 |
| b-otc | 6 / 3 | 10 / 0 |
| epinions | 5 / 5 | 7 / 3 |
| slashdot | 0 / 10 | 6 / 4 |
| wiki-elec | 9 / 1 | 10 / 0 |
| wiki-rfa | 6 / 4 | 10 / 0 |
WSDM text claims b-alpha and slashdot edge-dominant; b-alpha is vertex-dominant in 8/10 seeds.

## SHAP (remove = whole edge token; production: sign token, H3: identity+sign)
hop1 > hop2 (mean of fwd,bwd): 10/10 seeds on all 6 datasets, both models. ROBUST.
hop-1 fwd > bwd, seeds out of 10:
| dataset | production | H3 |
|---|---|---|
| b-alpha | 6 | 8 |
| b-otc | 5 | 7 |
| epinions | 1 | 4 |
| slashdot | 10 | 10 |
| wiki-elec | 8 | 10 |
| wiki-rfa | 8 | 10 |
H3 SHAP magnitudes 1.5-4x production's (it removes identity too).
H3 sign-only removal (identity kept, sign hidden): out-of-distribution for H3 (that state never
occurs in training); larger values, decay 7/10 on b-alpha -- treat as secondary.

## Takeaways
- Within-seed significance (cluster-robust over edges) ignores seed-to-seed variance; the WSDM
  per-dataset attention claims rest on one checkpoint and mostly do not replicate across seeds.
- Robust across seeds and both models: SHAP distance decay (6/6, 10/10); forward dominance on
  slashdot, wiki-elec, wiki-rfa (SHAP in both models, attention in H3).
