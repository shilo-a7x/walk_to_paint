# Lead 4c — atomic-direction entropy regression

Per-edge `correct` regressed on the **6 atomic directional sign-entropies** at once:

```
logit(correct) ~ src_out + src_in + tgt_out + tgt_in + twohop_in + twohop_out  [+ dataset FE when pooled]
```

Each β is a partial coefficient (effect of that direction holding the others fixed). Two-way cluster-robust SE on (u,v). `*` = BH-FDR p<0.05 within the atomic family. No direction is privileged a priori — which ones matter is read off the fitted βs below. Entropy is in bits (0 = uniform signs, 1 = 50/50).

**Common-support caveat:** an edge is dropped if any of the 6 entropies is undefined (node lacks in- or out-edges), so the atomic-model n is ≤ the per-combo n.


## TL;DR — the bottom line

- **The walk model's advantage is direction-specific, not blanket robustness.** GNNs are hurt *more* than the walk only on **tgt_in** (contested target reputation): walk β≈-1.94 vs GNN β≈-3.02 (gap -1.08).
- On **src_out** (an inconsistent *rater* as source) — the largest entropy effect of all — the **walk is hurt more**: walk β≈-2.71 vs GNN β≈-2.03 (gap +0.68).
- **tgt_out** and **src_in** are near-null; **path/2-hop** entropy barely moves any model. So the effect is a specific src/tgt **asymmetry**, not 'entropy hurts GNNs'.
- **Compact 2-number companion** (count-pooled, §1b): node-entropy β walk≈-3.32 / GNN≈-4.28; path-entropy β walk≈+0.22 / GNN≈+0.51. NOTE the node number pools src_out (walk-worse) with tgt_in (GNN-worse), so it *cancels* the asymmetry — read it together with the atomic forest, never alone.

## 1. Headline — pooled (shared slope, dataset fixed effects)

| term | model | β | 95% CI (robust) | p (FDR) | OR | n |
|---|---|---:|---|---:|---:|---:|
| src_out | walk_full | -2.692* | [-2.778, -2.606] | 0.00e+00 | 0.068 | 125212 |
| src_out | walk_localattn4 | -2.726* | [-2.812, -2.640] | 0.00e+00 | 0.066 | 125212 |
| src_out | GINEConv | -1.339* | [-1.450, -1.229] | 0.00e+00 | 0.262 | 125212 |
| src_out | SiGAT | -2.721* | [-2.814, -2.628] | 0.00e+00 | 0.066 | 125212 |
| src_in | walk_full | +0.006 | [-0.091, +0.103] | 9.10e-01 | 1.006 | 125212 |
| src_in | walk_localattn4 | -0.008 | [-0.104, +0.087] | 9.02e-01 | 0.992 | 125212 |
| src_in | GINEConv | -0.651* | [-0.777, -0.525] | 0.00e+00 | 0.521 | 125212 |
| src_in | SiGAT | -0.278* | [-0.382, -0.174] | 3.36e-07 | 0.757 | 125212 |
| tgt_out | walk_full | -0.111* | [-0.182, -0.040] | 3.03e-03 | 0.895 | 125212 |
| tgt_out | walk_localattn4 | -0.123* | [-0.196, -0.050] | 1.49e-03 | 0.884 | 125212 |
| tgt_out | GINEConv | +0.032 | [-0.036, +0.099] | 4.09e-01 | 1.032 | 125212 |
| tgt_out | SiGAT | -0.018 | [-0.088, +0.052] | 6.73e-01 | 0.982 | 125212 |
| tgt_in | walk_full | -1.928* | [-2.030, -1.825] | 0.00e+00 | 0.145 | 125212 |
| tgt_in | walk_localattn4 | -1.955* | [-2.058, -1.853] | 0.00e+00 | 0.142 | 125212 |
| tgt_in | GINEConv | -3.519* | [-3.658, -3.380] | 0.00e+00 | 0.030 | 125212 |
| tgt_in | SiGAT | -2.530* | [-2.639, -2.421] | 0.00e+00 | 0.080 | 125212 |
| twohop_in | walk_full | -0.402* | [-0.559, -0.245] | 9.74e-07 | 0.669 | 125212 |
| twohop_in | walk_localattn4 | -0.459* | [-0.620, -0.299] | 4.61e-08 | 0.632 | 125212 |
| twohop_in | GINEConv | -0.746* | [-0.935, -0.558] | 1.97e-14 | 0.474 | 125212 |
| twohop_in | SiGAT | -0.359* | [-0.534, -0.184] | 9.66e-05 | 0.698 | 125212 |
| twohop_out | walk_full | -0.120* | [-0.220, -0.020] | 2.38e-02 | 0.887 | 125212 |
| twohop_out | walk_localattn4 | -0.144* | [-0.245, -0.044] | 6.64e-03 | 0.866 | 125212 |
| twohop_out | GINEConv | +0.174* | [+0.071, +0.276] | 1.49e-03 | 1.190 | 125212 |
| twohop_out | SiGAT | +0.085 | [-0.023, +0.193] | 1.46e-01 | 1.089 | 125212 |

## 1b. Compact — one node-β + one path-β (count-pooled)

`b_node` = entropy of sign-counts pooled across BOTH endpoints' incident edges; `b_path` = 2-hop consistency entropy pooled over both anchors. NOT a variant average. Companion to §1 — pooling cancels the src_out/tgt_in asymmetry, so read with the forest.

| term | model | β | 95% CI (robust) | p (FDR) | OR | n |
|---|---|---:|---|---:|---:|---:|
| b_node | walk_full | -3.320* | [-3.408, -3.231] | 0.00e+00 | 0.036 | 167917 |
| b_node | walk_localattn4 | -3.322* | [-3.412, -3.233] | 0.00e+00 | 0.036 | 167917 |
| b_node | GINEConv | -4.123* | [-4.264, -3.982] | 0.00e+00 | 0.016 | 167917 |
| b_node | SiGAT | -4.436* | [-4.543, -4.330] | 0.00e+00 | 0.012 | 167917 |
| b_path | walk_full | +0.229* | [+0.118, +0.339] | 5.94e-05 | 1.257 | 167917 |
| b_path | walk_localattn4 | +0.209* | [+0.097, +0.321] | 2.47e-04 | 1.232 | 167917 |
| b_path | GINEConv | +0.271* | [+0.140, +0.403] | 5.94e-05 | 1.312 | 167917 |
| b_path | SiGAT | +0.758* | [+0.640, +0.876] | 0.00e+00 | 2.133 | 167917 |

## 1c. Per-dataset intercepts (atomic + compact models)

Each dataset gets its own intercept in the pooled fit (the entropy slopes in §1/§1b are still SHARED across datasets). `baseline_accuracy` = sigmoid(intercept) = predicted accuracy when every entropy term is 0 (fully homogeneous neighborhood).


**Atomic model:**

| model | dataset | intercept | 95% CI | baseline accuracy | n |
|---|---|---:|---|---:|---:|
| walk_full | bitcoin-alpha | +4.641 | [+4.383, +4.899] | 0.990 | 125212 |
| walk_full | bitcoin-otc | +4.907 | [+4.695, +5.120] | 0.993 | 125212 |
| walk_full | epinions | +5.206 | [+5.085, +5.328] | 0.995 | 125212 |
| walk_full | wiki-elec | +5.096 | [+4.933, +5.258] | 0.994 | 125212 |
| walk_full | wiki-rfa | +4.927 | [+4.775, +5.078] | 0.993 | 125212 |
| walk_full | slashdot090221 | +4.647 | [+4.514, +4.779] | 0.990 | 125212 |
| walk_localattn4 | bitcoin-alpha | +4.763 | [+4.540, +4.985] | 0.992 | 125212 |
| walk_localattn4 | bitcoin-otc | +4.983 | [+4.775, +5.191] | 0.993 | 125212 |
| walk_localattn4 | epinions | +5.288 | [+5.165, +5.411] | 0.995 | 125212 |
| walk_localattn4 | wiki-elec | +5.218 | [+5.045, +5.390] | 0.995 | 125212 |
| walk_localattn4 | wiki-rfa | +5.171 | [+5.018, +5.324] | 0.994 | 125212 |
| walk_localattn4 | slashdot090221 | +4.782 | [+4.646, +4.917] | 0.992 | 125212 |
| GINEConv | bitcoin-alpha | +4.974 | [+4.706, +5.242] | 0.993 | 125212 |
| GINEConv | bitcoin-otc | +5.116 | [+4.909, +5.324] | 0.994 | 125212 |
| GINEConv | epinions | +5.115 | [+4.982, +5.248] | 0.994 | 125212 |
| GINEConv | wiki-elec | +5.383 | [+5.195, +5.570] | 0.995 | 125212 |
| GINEConv | wiki-rfa | +5.674 | [+5.501, +5.847] | 0.997 | 125212 |
| GINEConv | slashdot090221 | +5.218 | [+5.069, +5.366] | 0.995 | 125212 |
| SiGAT | bitcoin-alpha | +4.801 | [+4.589, +5.012] | 0.992 | 125212 |
| SiGAT | bitcoin-otc | +4.648 | [+4.449, +4.848] | 0.991 | 125212 |
| SiGAT | epinions | +5.287 | [+5.152, +5.421] | 0.995 | 125212 |
| SiGAT | wiki-elec | +5.513 | [+5.333, +5.693] | 0.996 | 125212 |
| SiGAT | wiki-rfa | +5.565 | [+5.400, +5.729] | 0.996 | 125212 |
| SiGAT | slashdot090221 | +5.248 | [+5.100, +5.397] | 0.995 | 125212 |

**Compact model:**

| model | dataset | intercept | 95% CI | baseline accuracy | n |
|---|---|---:|---|---:|---:|
| walk_full | bitcoin-alpha | +4.024 | [+3.768, +4.279] | 0.982 | 167917 |
| walk_full | bitcoin-otc | +4.149 | [+3.938, +4.360] | 0.984 | 167917 |
| walk_full | epinions | +4.279 | [+4.188, +4.370] | 0.986 | 167917 |
| walk_full | wiki-elec | +3.849 | [+3.731, +3.967] | 0.979 | 167917 |
| walk_full | wiki-rfa | +3.630 | [+3.521, +3.739] | 0.974 | 167917 |
| walk_full | slashdot090221 | +3.529 | [+3.439, +3.620] | 0.972 | 167917 |
| walk_localattn4 | bitcoin-alpha | +4.097 | [+3.872, +4.322] | 0.984 | 167917 |
| walk_localattn4 | bitcoin-otc | +4.178 | [+3.960, +4.395] | 0.985 | 167917 |
| walk_localattn4 | epinions | +4.279 | [+4.188, +4.370] | 0.986 | 167917 |
| walk_localattn4 | wiki-elec | +3.827 | [+3.708, +3.946] | 0.979 | 167917 |
| walk_localattn4 | wiki-rfa | +3.725 | [+3.618, +3.832] | 0.976 | 167917 |
| walk_localattn4 | slashdot090221 | +3.573 | [+3.481, +3.664] | 0.973 | 167917 |
| GINEConv | bitcoin-alpha | +4.401 | [+4.153, +4.650] | 0.988 | 167917 |
| GINEConv | bitcoin-otc | +4.542 | [+4.332, +4.752] | 0.989 | 167917 |
| GINEConv | epinions | +4.611 | [+4.497, +4.725] | 0.990 | 167917 |
| GINEConv | wiki-elec | +4.415 | [+4.275, +4.554] | 0.988 | 167917 |
| GINEConv | wiki-rfa | +4.465 | [+4.340, +4.590] | 0.989 | 167917 |
| GINEConv | slashdot090221 | +3.974 | [+3.862, +4.087] | 0.982 | 167917 |
| SiGAT | bitcoin-alpha | +4.257 | [+4.046, +4.469] | 0.986 | 167917 |
| SiGAT | bitcoin-otc | +4.139 | [+3.943, +4.335] | 0.984 | 167917 |
| SiGAT | epinions | +4.590 | [+4.484, +4.695] | 0.990 | 167917 |
| SiGAT | wiki-elec | +4.396 | [+4.268, +4.525] | 0.988 | 167917 |
| SiGAT | wiki-rfa | +4.325 | [+4.206, +4.444] | 0.987 | 167917 |
| SiGAT | slashdot090221 | +4.161 | [+4.054, +4.268] | 0.985 | 167917 |

## 1d. Same models, entropy inputs literally z-scored before fitting

Each entropy column standardized to mean 0 / sd 1 BEFORE the regression is run (`spec` suffix `_zscored`) -- a real separate fit with its own CI/p, not beta×SD of the raw fit (the two agree to ~1e-3, confirming no bug, but these are the genuine z-scored numbers). **This does not and cannot change the walk-vs-GNN comparison for a given term**: all 4 models share the same entropy columns, so the same SD is applied to all of them -- it only matters for comparing different terms to each other within one model.


**Atomic model (z-scored):**

| term | model | β (z-scored) | 95% CI | p (FDR) | n |
|---|---|---:|---|---:|---:|
| src_out | walk_full | -1.002* | [-1.034, -0.970] | 0.00e+00 | 125212 |
| src_out | walk_localattn4 | -1.015* | [-1.047, -0.983] | 0.00e+00 | 125212 |
| src_out | GINEConv | -0.499* | [-0.540, -0.457] | 0.00e+00 | 125212 |
| src_out | SiGAT | -1.013* | [-1.047, -0.978] | 0.00e+00 | 125212 |
| src_in | walk_full | +0.002 | [-0.031, +0.035] | 9.10e-01 | 125212 |
| src_in | walk_localattn4 | -0.003 | [-0.035, +0.029] | 9.02e-01 | 125212 |
| src_in | GINEConv | -0.220* | [-0.262, -0.177] | 0.00e+00 | 125212 |
| src_in | SiGAT | -0.094* | [-0.129, -0.059] | 3.36e-07 | 125212 |
| tgt_out | walk_full | -0.043* | [-0.070, -0.015] | 3.03e-03 | 125212 |
| tgt_out | walk_localattn4 | -0.047* | [-0.075, -0.019] | 1.49e-03 | 125212 |
| tgt_out | GINEConv | +0.012 | [-0.014, +0.038] | 4.09e-01 | 125212 |
| tgt_out | SiGAT | -0.007 | [-0.034, +0.020] | 6.73e-01 | 125212 |
| tgt_in | walk_full | -0.629* | [-0.662, -0.596] | 0.00e+00 | 125212 |
| tgt_in | walk_localattn4 | -0.638* | [-0.672, -0.605] | 0.00e+00 | 125212 |
| tgt_in | GINEConv | -1.148* | [-1.194, -1.103] | 0.00e+00 | 125212 |
| tgt_in | SiGAT | -0.826* | [-0.861, -0.790] | 0.00e+00 | 125212 |
| twohop_in | walk_full | -0.108* | [-0.150, -0.066] | 9.74e-07 | 125212 |
| twohop_in | walk_localattn4 | -0.123* | [-0.166, -0.080] | 4.61e-08 | 125212 |
| twohop_in | GINEConv | -0.200* | [-0.251, -0.150] | 1.97e-14 | 125212 |
| twohop_in | SiGAT | -0.096* | [-0.143, -0.049] | 9.66e-05 | 125212 |
| twohop_out | walk_full | -0.037* | [-0.068, -0.006] | 2.38e-02 | 125212 |
| twohop_out | walk_localattn4 | -0.045* | [-0.076, -0.013] | 6.64e-03 | 125212 |
| twohop_out | GINEConv | +0.054* | [+0.022, +0.086] | 1.49e-03 | 125212 |
| twohop_out | SiGAT | +0.026 | [-0.007, +0.060] | 1.46e-01 | 125212 |

**Compact model (z-scored):**

| term | model | β (z-scored) | 95% CI | p (FDR) | n |
|---|---|---:|---|---:|---:|
| b_node | walk_full | -1.034* | [-1.062, -1.007] | 0.00e+00 | 167917 |
| b_node | walk_localattn4 | -1.035* | [-1.063, -1.007] | 0.00e+00 | 167917 |
| b_node | GINEConv | -1.284* | [-1.328, -1.240] | 0.00e+00 | 167917 |
| b_node | SiGAT | -1.382* | [-1.415, -1.349] | 0.00e+00 | 167917 |
| b_path | walk_full | +0.055* | [+0.028, +0.081] | 5.94e-05 | 167917 |
| b_path | walk_localattn4 | +0.050* | [+0.023, +0.077] | 2.47e-04 | 167917 |
| b_path | GINEConv | +0.065* | [+0.033, +0.096] | 5.94e-05 | 167917 |
| b_path | SiGAT | +0.181* | [+0.153, +0.209] | 0.00e+00 | 167917 |

## 2. What the data support (read off section 1, not assumed)

- **src_out**: walk β≈-2.71, GNN β≈-2.03 (GNN−walk gap +0.68; walk more-negative than GNN). FDR-sig models: walk_full, walk_localattn4, GINEConv, SiGAT.
- **src_in**: walk β≈-0.00, GNN β≈-0.46 (GNN−walk gap -0.46; GNN more-negative than walk). FDR-sig models: GINEConv, SiGAT.
- **tgt_out**: walk β≈-0.12, GNN β≈+0.01 (GNN−walk gap +0.12; walk more-negative than GNN). FDR-sig models: walk_full, walk_localattn4.
- **tgt_in**: walk β≈-1.94, GNN β≈-3.02 (GNN−walk gap -1.08; GNN more-negative than walk). FDR-sig models: walk_full, walk_localattn4, GINEConv, SiGAT.
- **twohop_in**: walk β≈-0.43, GNN β≈-0.55 (GNN−walk gap -0.12; GNN more-negative than walk). FDR-sig models: walk_full, walk_localattn4, GINEConv, SiGAT.
- **twohop_out**: walk β≈-0.13, GNN β≈+0.13 (GNN−walk gap +0.26; walk more-negative than GNN). FDR-sig models: walk_full, walk_localattn4, GINEConv.

The src/tgt asymmetry (if any) is whatever the above shows — compare the src_* rows to the tgt_* rows directly.


## 3. Cross-dataset consistency (per-dataset atomic fits)

Fraction of the 6 datasets where mean GNN β is more negative than mean walk β, with a sign-test p.

| term | GNN-more-negative share | n datasets | sign-test p |
|---|---:|---:|---:|
| src_out | 17% | 6 | 0.2188 |
| src_in | 83% | 6 | 0.2188 |
| tgt_out | 33% | 6 | 0.6875 |
| tgt_in | 100% | 6 | 0.0312 |
| twohop_in | 33% | 6 | 0.6875 |
| twohop_out | 0% | 6 | 0.0312 |

## 4. Pooling — shared vs interacted slope (plain language)

- **Shared slope** (what section 1 reports): ONE β per term for all datasets; the datasets are allowed only to shift the intercept (dataset fixed effects). Answers *"is the effect the same everywhere?"*

- **Interacted slope**: each dataset gets its OWN β per term (equivalently, the per-dataset fits). Answers *"or does the effect vary by dataset?"* The `atomic_pooling_caterpillar.png` figure overlays the two: dots = per-dataset (interacted) βs, red line = the single shared-slope β.


## 5. Model-free cross-check — Spearman ρ & MI (pooled)

| term | model | Spearman ρ | p | MI (bits) | n |
|---|---|---:|---:|---:|---:|
| src_out | walk_full | -0.270* | 0.00e+00 | 0.0573 | 173072 |
| src_out | walk_localattn4 | -0.270* | 0.00e+00 | 0.0572 | 173072 |
| src_out | GINEConv | -0.235* | 0.00e+00 | 0.0384 | 173072 |
| src_out | SiGAT | -0.293* | 0.00e+00 | 0.0665 | 173072 |
| src_in | walk_full | -0.134* | 0.00e+00 | 0.0176 | 151554 |
| src_in | walk_localattn4 | -0.136* | 0.00e+00 | 0.0179 | 151554 |
| src_in | GINEConv | -0.191* | 0.00e+00 | 0.0314 | 151554 |
| src_in | SiGAT | -0.143* | 0.00e+00 | 0.0183 | 151554 |
| tgt_out | walk_full | -0.067* | 1.12e-144 | 0.0039 | 143893 |
| tgt_out | walk_localattn4 | -0.067* | 2.28e-141 | 0.0037 | 143893 |
| tgt_out | GINEConv | -0.041* | 3.46e-55 | 0.0016 | 143893 |
| tgt_out | SiGAT | -0.047* | 2.13e-71 | 0.0020 | 143893 |
| tgt_in | walk_full | -0.248* | 0.00e+00 | 0.0551 | 173072 |
| tgt_in | walk_localattn4 | -0.248* | 0.00e+00 | 0.0541 | 173072 |
| tgt_in | GINEConv | -0.344* | 0.00e+00 | 0.1025 | 173072 |
| tgt_in | SiGAT | -0.259* | 0.00e+00 | 0.0590 | 173072 |
| twohop_in | walk_full | -0.155* | 0.00e+00 | 0.0194 | 151030 |
| twohop_in | walk_localattn4 | -0.156* | 0.00e+00 | 0.0193 | 151030 |
| twohop_in | GINEConv | -0.203* | 0.00e+00 | 0.0312 | 151030 |
| twohop_in | SiGAT | -0.149* | 0.00e+00 | 0.0170 | 151030 |
| twohop_out | walk_full | -0.040* | 1.96e-52 | 0.0023 | 142099 |
| twohop_out | walk_localattn4 | -0.042* | 6.52e-56 | 0.0024 | 142099 |
| twohop_out | GINEConv | -0.013* | 9.73e-07 | 0.0010 | 142099 |
| twohop_out | SiGAT | -0.026* | 2.87e-23 | 0.0010 | 142099 |

ρ signs should match the atomic-β signs — confirmation the effect is not a logistic functional-form artifact.


## 6. Appendix — the legacy 12-combo sweep

The 4 node-variants × 3 two-hop variants are overlapping pairings of the 6 atoms (see `atomic_explainer.png`). `appendix_combo_ranking.png` shows the b_tgt GNN−walk gap across all 12 combos, grouped by whether the target term uses v's IN-edges; combos sharing `tgt_in` cluster together, i.e. the atomic model already explains the combo sweep. The full per-combo numbers remain in `fit_results.csv` (`spec == 'marginal3'`).

