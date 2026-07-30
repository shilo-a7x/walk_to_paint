# Lead 4c — regression coefficients (z-scored inputs)

Logistic regression: `correct ~ entropy terms + dataset intercepts`, fit separately per model
(walk_full, walk_localattn4, GINEConv, SiGAT). All entropy inputs were standardized
(mean 0, SD 1) **before** fitting — so each `beta` is "change in log-odds of a correct
prediction per 1-SD increase in that entropy term." Two-way cluster-robust SEs (clustered on
both edge endpoints u and v). p-values are Benjamini-Hochberg FDR-corrected across all terms.

Every number below is read directly off that one regression fit — nothing here is derived or
rescaled after the fact.

## Atomic model — 6 entropy directions, one joint regression

| Term | Model | beta | 95% CI | p (FDR) | sig | odds ratio | n |
|---|---|---|---|---|---|---|---|
| src_out | walk_full | -1.002 | [-1.034, -0.970] | 0.0e+00 | yes | 0.367 | 125212 |
| src_in | walk_full | 0.002 | [-0.031, 0.035] | 9.1e-01 | no | 1.002 | 125212 |
| tgt_out | walk_full | -0.043 | [-0.070, -0.015] | 3.0e-03 | yes | 0.958 | 125212 |
| tgt_in | walk_full | -0.629 | [-0.662, -0.596] | 0.0e+00 | yes | 0.533 | 125212 |
| twohop_in | walk_full | -0.108 | [-0.150, -0.066] | 9.7e-07 | yes | 0.898 | 125212 |
| twohop_out | walk_full | -0.037 | [-0.068, -0.006] | 2.4e-02 | yes | 0.964 | 125212 |
| src_out | walk_localattn4 | -1.015 | [-1.047, -0.983] | 0.0e+00 | yes | 0.363 | 125212 |
| src_in | walk_localattn4 | -0.003 | [-0.035, 0.029] | 9.0e-01 | no | 0.997 | 125212 |
| tgt_out | walk_localattn4 | -0.047 | [-0.075, -0.019] | 1.5e-03 | yes | 0.954 | 125212 |
| tgt_in | walk_localattn4 | -0.638 | [-0.672, -0.605] | 0.0e+00 | yes | 0.528 | 125212 |
| twohop_in | walk_localattn4 | -0.123 | [-0.166, -0.080] | 4.6e-08 | yes | 0.884 | 125212 |
| twohop_out | walk_localattn4 | -0.045 | [-0.076, -0.013] | 6.6e-03 | yes | 0.956 | 125212 |
| src_out | GINEConv | -0.499 | [-0.540, -0.457] | 0.0e+00 | yes | 0.607 | 125212 |
| src_in | GINEConv | -0.220 | [-0.262, -0.177] | 0.0e+00 | yes | 0.803 | 125212 |
| tgt_out | GINEConv | 0.012 | [-0.014, 0.038] | 4.1e-01 | no | 1.012 | 125212 |
| tgt_in | GINEConv | -1.148 | [-1.194, -1.103] | 0.0e+00 | yes | 0.317 | 125212 |
| twohop_in | GINEConv | -0.200 | [-0.251, -0.150] | 2.0e-14 | yes | 0.819 | 125212 |
| twohop_out | GINEConv | 0.054 | [0.022, 0.086] | 1.5e-03 | yes | 1.055 | 125212 |
| src_out | SiGAT | -1.013 | [-1.047, -0.978] | 0.0e+00 | yes | 0.363 | 125212 |
| src_in | SiGAT | -0.094 | [-0.129, -0.059] | 3.4e-07 | yes | 0.910 | 125212 |
| tgt_out | SiGAT | -0.007 | [-0.034, 0.020] | 6.7e-01 | no | 0.993 | 125212 |
| tgt_in | SiGAT | -0.826 | [-0.861, -0.790] | 0.0e+00 | yes | 0.438 | 125212 |
| twohop_in | SiGAT | -0.096 | [-0.143, -0.049] | 9.7e-05 | yes | 0.908 | 125212 |
| twohop_out | SiGAT | 0.026 | [-0.007, 0.060] | 1.5e-01 | no | 1.027 | 125212 |

**Reading it:** `tgt_in` (target node's in-edge sign mix) hurts GINEConv (-1.148) and SiGAT
(-0.826) far more than either walk variant (-0.629/-0.638). `src_out` (source node's out-edge
sign mix) hurts both walk variants (-1.002/-1.015) and SiGAT (-1.013) about equally, but hurts
GINEConv noticeably less (-0.499). `tgt_out`/`src_in` are near zero for the walk models but
mildly significant for the GNNs. The two-hop terms are small for everyone.

### Per-dataset intercepts (atomic model)

| Model | Dataset | Intercept (beta) | 95% CI | Odds ratio | Baseline accuracy | n |
|---|---|---|---|---|---|---|
| walk_full | bitcoin-alpha | 2.544 | [2.296, 2.792] | 12.730 | 0.893 | 125212 |
| walk_full | bitcoin-otc | 2.810 | [2.611, 3.009] | 16.611 | 0.893 | 125212 |
| walk_full | epinions | 3.109 | [3.055, 3.162] | 22.398 | 0.893 | 125212 |
| walk_full | wiki-elec | 2.998 | [2.892, 3.105] | 20.051 | 0.893 | 125212 |
| walk_full | wiki-rfa | 2.829 | [2.744, 2.915] | 16.934 | 0.893 | 125212 |
| walk_full | slashdot090221 | 2.549 | [2.496, 2.602] | 12.797 | 0.893 | 125212 |
| walk_localattn4 | bitcoin-alpha | 2.581 | [2.370, 2.791] | 13.206 | 0.894 | 125212 |
| walk_localattn4 | bitcoin-otc | 2.801 | [2.609, 2.993] | 16.464 | 0.894 | 125212 |
| walk_localattn4 | epinions | 3.106 | [3.051, 3.161] | 22.326 | 0.894 | 125212 |
| walk_localattn4 | wiki-elec | 3.036 | [2.919, 3.153] | 20.818 | 0.894 | 125212 |
| walk_localattn4 | wiki-rfa | 2.989 | [2.906, 3.072] | 19.866 | 0.894 | 125212 |
| walk_localattn4 | slashdot090221 | 2.600 | [2.546, 2.653] | 13.459 | 0.894 | 125212 |
| GINEConv | bitcoin-alpha | 2.549 | [2.292, 2.805] | 12.790 | 0.879 | 125212 |
| GINEConv | bitcoin-otc | 2.691 | [2.501, 2.881] | 14.746 | 0.879 | 125212 |
| GINEConv | epinions | 2.690 | [2.611, 2.768] | 14.725 | 0.879 | 125212 |
| GINEConv | wiki-elec | 2.957 | [2.825, 3.089] | 19.242 | 0.879 | 125212 |
| GINEConv | wiki-rfa | 3.248 | [3.146, 3.350] | 25.746 | 0.879 | 125212 |
| GINEConv | slashdot090221 | 2.792 | [2.726, 2.858] | 16.320 | 0.879 | 125212 |
| SiGAT | bitcoin-alpha | 2.558 | [2.363, 2.753] | 12.912 | 0.899 | 125212 |
| SiGAT | bitcoin-otc | 2.406 | [2.225, 2.587] | 11.089 | 0.899 | 125212 |
| SiGAT | epinions | 3.044 | [2.984, 3.105] | 20.990 | 0.899 | 125212 |
| SiGAT | wiki-elec | 3.271 | [3.150, 3.392] | 26.326 | 0.899 | 125212 |
| SiGAT | wiki-rfa | 3.322 | [3.235, 3.409] | 27.717 | 0.899 | 125212 |
| SiGAT | slashdot090221 | 3.006 | [2.942, 3.070] | 20.202 | 0.899 | 125212 |

## Compact model — 2 terms (count-pooled node entropy, count-pooled path entropy), one joint regression

| Term | Model | beta | 95% CI | p (FDR) | sig | odds ratio | n |
|---|---|---|---|---|---|---|---|
| node_entropy | walk_full | -1.034 | [-1.062, -1.007] | 0.0e+00 | yes | 0.356 | 167917 |
| path_entropy | walk_full | 0.055 | [0.028, 0.081] | 5.9e-05 | yes | 1.056 | 167917 |
| node_entropy | walk_localattn4 | -1.035 | [-1.063, -1.007] | 0.0e+00 | yes | 0.355 | 167917 |
| path_entropy | walk_localattn4 | 0.050 | [0.023, 0.077] | 2.5e-04 | yes | 1.051 | 167917 |
| node_entropy | GINEConv | -1.284 | [-1.328, -1.240] | 0.0e+00 | yes | 0.277 | 167917 |
| path_entropy | GINEConv | 0.065 | [0.033, 0.096] | 5.9e-05 | yes | 1.067 | 167917 |
| node_entropy | SiGAT | -1.382 | [-1.415, -1.349] | 0.0e+00 | yes | 0.251 | 167917 |
| path_entropy | SiGAT | 0.181 | [0.153, 0.209] | 0.0e+00 | yes | 1.198 | 167917 |

**Caveat:** `node_entropy`/`path_entropy` pool sign-counts across both endpoints before
computing entropy, so this collapses the src/tgt asymmetry seen in the atomic model above —
useful as a single compact number per model, but the atomic table is what shows *which*
direction actually drives the difference.

### Per-dataset intercepts (compact model)

| Model | Dataset | Intercept (beta) | 95% CI | Odds ratio | Baseline accuracy | n |
|---|---|---|---|---|---|---|
| walk_full | bitcoin-alpha | 2.484 | [2.235, 2.733] | 11.990 | 0.875 | 167917 |
| walk_full | bitcoin-otc | 2.609 | [2.408, 2.811] | 13.588 | 0.875 | 167917 |
| walk_full | epinions | 2.739 | [2.691, 2.788] | 15.475 | 0.875 | 167917 |
| walk_full | wiki-elec | 2.310 | [2.225, 2.394] | 10.071 | 0.875 | 167917 |
| walk_full | wiki-rfa | 2.091 | [2.023, 2.158] | 8.090 | 0.875 | 167917 |
| walk_full | slashdot090221 | 1.989 | [1.952, 2.027] | 7.312 | 0.875 | 167917 |
| walk_localattn4 | bitcoin-alpha | 2.542 | [2.322, 2.761] | 12.701 | 0.877 | 167917 |
| walk_localattn4 | bitcoin-otc | 2.622 | [2.415, 2.829] | 13.765 | 0.877 | 167917 |
| walk_localattn4 | epinions | 2.724 | [2.674, 2.773] | 15.234 | 0.877 | 167917 |
| walk_localattn4 | wiki-elec | 2.272 | [2.186, 2.358] | 9.697 | 0.877 | 167917 |
| walk_localattn4 | wiki-rfa | 2.170 | [2.106, 2.234] | 8.757 | 0.877 | 167917 |
| walk_localattn4 | slashdot090221 | 2.017 | [1.980, 2.054] | 7.517 | 0.877 | 167917 |
| GINEConv | bitcoin-alpha | 2.479 | [2.239, 2.720] | 11.934 | 0.866 | 167917 |
| GINEConv | bitcoin-otc | 2.620 | [2.422, 2.817] | 13.732 | 0.866 | 167917 |
| GINEConv | epinions | 2.689 | [2.618, 2.761] | 14.723 | 0.866 | 167917 |
| GINEConv | wiki-elec | 2.493 | [2.395, 2.591] | 12.097 | 0.866 | 167917 |
| GINEConv | wiki-rfa | 2.543 | [2.472, 2.615] | 12.720 | 0.866 | 167917 |
| GINEConv | slashdot090221 | 2.053 | [2.002, 2.103] | 7.787 | 0.866 | 167917 |
| SiGAT | bitcoin-alpha | 2.530 | [2.326, 2.733] | 12.550 | 0.884 | 167917 |
| SiGAT | bitcoin-otc | 2.411 | [2.224, 2.598] | 11.148 | 0.884 | 167917 |
| SiGAT | epinions | 2.862 | [2.811, 2.913] | 17.496 | 0.884 | 167917 |
| SiGAT | wiki-elec | 2.669 | [2.584, 2.753] | 14.422 | 0.884 | 167917 |
| SiGAT | wiki-rfa | 2.597 | [2.533, 2.661] | 13.427 | 0.884 | 167917 |
| SiGAT | slashdot090221 | 2.433 | [2.391, 2.476] | 11.398 | 0.884 | 167917 |

## Degree-augmented atomic model — 6 entropy terms + 2 degree terms, one joint regression

Same atomic model as above, plus `log(out-degree of u)` and `log(in-degree of v)` as two more
z-scored covariates, refit together. Kept as a fully separate fit from the plain atomic model
above (verified byte-identical before/after adding these terms).

| Term | Model | beta | 95% CI | p (FDR) | sig | odds ratio | n |
|---|---|---|---|---|---|---|---|
| src_out | walk_full | -1.063 | [-1.094, -1.032] | 0.0e+00 | yes | 0.345 | 125212 |
| src_out | walk_localattn4 | -1.072 | [-1.103, -1.041] | 0.0e+00 | yes | 0.342 | 125212 |
| src_out | GINEConv | -0.499 | [-0.542, -0.456] | 0.0e+00 | yes | 0.607 | 125212 |
| src_out | SiGAT | -1.075 | [-1.109, -1.040] | 0.0e+00 | yes | 0.341 | 125212 |
| src_in | walk_full | -0.033 | [-0.065, -0.002] | 4.2e-02 | yes | 0.967 | 125212 |
| src_in | walk_localattn4 | -0.039 | [-0.069, -0.008] | 1.6e-02 | yes | 0.962 | 125212 |
| src_in | GINEConv | -0.216 | [-0.258, -0.175] | 0.0e+00 | yes | 0.805 | 125212 |
| src_in | SiGAT | -0.123 | [-0.156, -0.090] | 4.3e-13 | yes | 0.884 | 125212 |
| tgt_out | walk_full | -0.059 | [-0.086, -0.031] | 4.3e-05 | yes | 0.943 | 125212 |
| tgt_out | walk_localattn4 | -0.059 | [-0.088, -0.031] | 6.9e-05 | yes | 0.942 | 125212 |
| tgt_out | GINEConv | 0.001 | [-0.026, 0.028] | 9.6e-01 | no | 1.001 | 125212 |
| tgt_out | SiGAT | -0.025 | [-0.052, 0.002] | 8.4e-02 | no | 0.975 | 125212 |
| tgt_in | walk_full | -0.633 | [-0.666, -0.601] | 0.0e+00 | yes | 0.531 | 125212 |
| tgt_in | walk_localattn4 | -0.642 | [-0.675, -0.610] | 0.0e+00 | yes | 0.526 | 125212 |
| tgt_in | GINEConv | -1.129 | [-1.174, -1.085] | 0.0e+00 | yes | 0.323 | 125212 |
| tgt_in | SiGAT | -0.822 | [-0.857, -0.787] | 0.0e+00 | yes | 0.440 | 125212 |
| twohop_in | walk_full | -0.114 | [-0.154, -0.075] | 1.8e-08 | yes | 0.892 | 125212 |
| twohop_in | walk_localattn4 | -0.130 | [-0.170, -0.089] | 5.2e-10 | yes | 0.878 | 125212 |
| twohop_in | GINEConv | -0.196 | [-0.245, -0.146] | 3.3e-14 | yes | 0.822 | 125212 |
| twohop_in | SiGAT | -0.105 | [-0.149, -0.060] | 5.9e-06 | yes | 0.900 | 125212 |
| twohop_out | walk_full | -0.064 | [-0.095, -0.032] | 9.5e-05 | yes | 0.938 | 125212 |
| twohop_out | walk_localattn4 | -0.062 | [-0.094, -0.031] | 1.3e-04 | yes | 0.940 | 125212 |
| twohop_out | GINEConv | 0.026 | [-0.006, 0.059] | 1.2e-01 | no | 1.027 | 125212 |
| twohop_out | SiGAT | -0.008 | [-0.041, 0.026] | 6.7e-01 | no | 0.992 | 125212 |
| log_outdeg_u | walk_full | 0.383 | [0.347, 0.418] | 0.0e+00 | yes | 1.466 | 125212 |
| log_outdeg_u | walk_localattn4 | 0.374 | [0.337, 0.411] | 0.0e+00 | yes | 1.454 | 125212 |
| log_outdeg_u | GINEConv | -0.017 | [-0.065, 0.031] | 5.3e-01 | no | 0.983 | 125212 |
| log_outdeg_u | SiGAT | 0.360 | [0.320, 0.400] | 0.0e+00 | yes | 1.433 | 125212 |
| log_indeg_v | walk_full | 0.144 | [0.110, 0.178] | 0.0e+00 | yes | 1.155 | 125212 |
| log_indeg_v | walk_localattn4 | 0.100 | [0.066, 0.134] | 1.4e-08 | yes | 1.105 | 125212 |
| log_indeg_v | GINEConv | 0.126 | [0.087, 0.166] | 4.1e-10 | yes | 1.135 | 125212 |
| log_indeg_v | SiGAT | 0.171 | [0.138, 0.204] | 0.0e+00 | yes | 1.187 | 125212 |

**Reading it:** the 6 entropy coefficients barely move once degree is controlled for (e.g.
`src_out` walk_full -1.002 → -1.063) — the entropy asymmetry is not a degree/hubness artifact.
The new terms: `log_outdeg_u` is large and significant for walk_full/walk_localattn4/SiGAT
(0.36–0.38) but ~0 and non-significant for GINEConv (-0.017, p=0.53) — GINEConv's node embedding
for u never sees u's own outgoing edges (message passing aggregates in-neighbors only), so u's
out-degree is structurally invisible to it. `log_indeg_v` is positive and significant for every
model (0.10–0.17) — more of v's in-neighborhood helps everyone, including GNNs whose
representation of v is literally built from that neighborhood.

### Per-dataset intercepts (degree-augmented atomic model)

| Model | Dataset | Intercept (beta) | 95% CI | Odds ratio | Baseline accuracy | n |
|---|---|---|---|---|---|---|
| walk_full | bitcoin-alpha | 2.827 | [2.555, 3.099] | 16.898 | 0.893 | 125212 |
| walk_full | bitcoin-otc | 3.098 | [2.891, 3.305] | 22.156 | 0.893 | 125212 |
| walk_full | epinions | 3.105 | [3.050, 3.160] | 22.304 | 0.893 | 125212 |
| walk_full | wiki-elec | 2.981 | [2.872, 3.091] | 19.712 | 0.893 | 125212 |
| walk_full | wiki-rfa | 2.779 | [2.692, 2.866] | 16.099 | 0.893 | 125212 |
| walk_full | slashdot090221 | 2.691 | [2.634, 2.747] | 14.742 | 0.893 | 125212 |
| walk_localattn4 | bitcoin-alpha | 2.833 | [2.601, 3.065] | 16.994 | 0.894 | 125212 |
| walk_localattn4 | bitcoin-otc | 3.055 | [2.855, 3.254] | 21.217 | 0.894 | 125212 |
| walk_localattn4 | epinions | 3.099 | [3.041, 3.157] | 22.168 | 0.894 | 125212 |
| walk_localattn4 | wiki-elec | 3.017 | [2.896, 3.138] | 20.422 | 0.894 | 125212 |
| walk_localattn4 | wiki-rfa | 2.944 | [2.859, 3.029] | 18.994 | 0.894 | 125212 |
| walk_localattn4 | slashdot090221 | 2.730 | [2.673, 2.787] | 15.330 | 0.894 | 125212 |
| GINEConv | bitcoin-alpha | 2.605 | [2.342, 2.867] | 13.526 | 0.879 | 125212 |
| GINEConv | bitcoin-otc | 2.754 | [2.559, 2.949] | 15.703 | 0.879 | 125212 |
| GINEConv | epinions | 2.684 | [2.611, 2.756] | 14.639 | 0.879 | 125212 |
| GINEConv | wiki-elec | 2.939 | [2.808, 3.069] | 18.889 | 0.879 | 125212 |
| GINEConv | wiki-rfa | 3.213 | [3.113, 3.313] | 24.858 | 0.879 | 125212 |
| GINEConv | slashdot090221 | 2.782 | [2.714, 2.851] | 16.156 | 0.879 | 125212 |
| SiGAT | bitcoin-alpha | 2.843 | [2.637, 3.049] | 17.165 | 0.899 | 125212 |
| SiGAT | bitcoin-otc | 2.692 | [2.515, 2.868] | 14.755 | 0.899 | 125212 |
| SiGAT | epinions | 3.039 | [2.973, 3.106] | 20.888 | 0.899 | 125212 |
| SiGAT | wiki-elec | 3.253 | [3.132, 3.374] | 25.877 | 0.899 | 125212 |
| SiGAT | wiki-rfa | 3.273 | [3.185, 3.361] | 26.402 | 0.899 | 125212 |
| SiGAT | slashdot090221 | 3.138 | [3.072, 3.204] | 23.064 | 0.899 | 125212 |
