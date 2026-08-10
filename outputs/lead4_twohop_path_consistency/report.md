# Lead 4b: 2-Hop Sign-Path-Consistency Entropy vs. AUC (canonical, shared edges)

## Methodology

Part of Lead 4 (see `lead4_entropy_heterogeneity.py` / its report for the
node sign-entropy version). This variant gives EACH shared test edge a
heterogeneity score from 2-hop sign-path consistency, so the AUC-vs-
heterogeneity plot is a 1D binned bar chart (not a 2D heatmap).

**Same-edge ground truth.** Predictions are read from
`predictions_raw_canonical.pkl` and restricted to the **shared edge set**
(intersection of `(u, v)` across models = walk-covered edges) via
`lead4_entropy_heterogeneity.load_shared_predictions`. So all models are
bucketed over the *identical* edges -- and the per-bucket sample count `n` is
the **same for every model**, shown once under each x-axis tick.

**2-hop path-consistency entropy.** A path is *consistent* if both its edge
signs match. `p` = consistent fraction, turned into binary Shannon entropy
`H(p) = -p*log2(p) - (1-p)*log2(1-p)`. `H=0` -> second-hop sign fully
predictable from the first (locally balanced), `H=1` -> maximally
unpredictable. Computed over **all edges** (train+val+test) from the
canonical edge list -- a diagnostic grouping, no leakage.

**Path-direction variants**, for edge `(u, v)`:

| variant | traversal | consistency | anchored at |
|---|---|---|---|
| `out` | forward, OUT-edges both hops (`v->m->k`) | `sign(v->m) == sign(m->k)` | target `v` |
| `in` | backward, IN-edges both hops (`s->t->u`) | `sign(s->t) == sign(t->u)` | source `u` |
| `inout` | pool `out`'s tally (from `v`) and `in`'s tally (into `u`) into ONE count for this edge, then take entropy | both | edge `(u,v)` |

`out`/`in` are really per-node lookups (by `v` / `u` respectively); `inout`
is necessarily per-edge since it mixes both endpoints' counts before taking
entropy (not an average of two entropies). Edges with zero applicable 2-hop
paths for a variant are dropped.

Bars grow from a **0.5 baseline** (height = AUC - 0.5); the y-axis starts at
0.5 unless a bucket dips below chance. Walk models are blue, GNNs orange.
`n=.. (<min_n)` / `n=.. (1 class)` buckets have no defined AUC (no bar). Raw
numbers are dumped to `raw_data.txt`; per-edge `(ent, y, p)` cached in
`computed_data.pkl` for re-binning via `--mode plot`.

## Results

### bitcoin-alpha

**out paths**

![bitcoin-alpha out b2](bitcoin-alpha/bitcoin-alpha_out_fixed_b2.png)

![bitcoin-alpha out b4](bitcoin-alpha/bitcoin-alpha_out_fixed_b4.png)

![bitcoin-alpha out b8](bitcoin-alpha/bitcoin-alpha_out_fixed_b8.png)

![bitcoin-alpha out b16](bitcoin-alpha/bitcoin-alpha_out_fixed_b16.png)

![bitcoin-alpha out b32](bitcoin-alpha/bitcoin-alpha_out_fixed_b32.png)

**variant = `out` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.8956 (n=1450) | 0.9100 (n=888) | -0.0144 |
| walk_localattn4 | 0.9080 (n=1450) | 0.9293 (n=888) | -0.0214 |
| GINEConv | 0.8225 (n=1450) | 0.8617 (n=888) | -0.0392 |
| SiGAT | 0.8812 (n=1450) | 0.8140 (n=888) | +0.0672 |

**in paths**

![bitcoin-alpha in b2](bitcoin-alpha/bitcoin-alpha_in_fixed_b2.png)

![bitcoin-alpha in b4](bitcoin-alpha/bitcoin-alpha_in_fixed_b4.png)

![bitcoin-alpha in b8](bitcoin-alpha/bitcoin-alpha_in_fixed_b8.png)

![bitcoin-alpha in b16](bitcoin-alpha/bitcoin-alpha_in_fixed_b16.png)

![bitcoin-alpha in b32](bitcoin-alpha/bitcoin-alpha_in_fixed_b32.png)

**variant = `in` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9251 (n=1892) | 0.9128 (n=523) | +0.0123 |
| walk_localattn4 | 0.9333 (n=1892) | 0.9342 (n=523) | -0.0009 |
| GINEConv | 0.8629 (n=1892) | 0.8434 (n=523) | +0.0195 |
| SiGAT | 0.8931 (n=1892) | 0.8366 (n=523) | +0.0564 |

**inout paths**

![bitcoin-alpha inout b2](bitcoin-alpha/bitcoin-alpha_inout_fixed_b2.png)

![bitcoin-alpha inout b4](bitcoin-alpha/bitcoin-alpha_inout_fixed_b4.png)

![bitcoin-alpha inout b8](bitcoin-alpha/bitcoin-alpha_inout_fixed_b8.png)

![bitcoin-alpha inout b16](bitcoin-alpha/bitcoin-alpha_inout_fixed_b16.png)

![bitcoin-alpha inout b32](bitcoin-alpha/bitcoin-alpha_inout_fixed_b32.png)

**variant = `inout` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9186 (n=1590) | 0.9244 (n=829) | -0.0058 |
| walk_localattn4 | 0.9269 (n=1590) | 0.9415 (n=829) | -0.0147 |
| GINEConv | 0.8525 (n=1590) | 0.8617 (n=829) | -0.0092 |
| SiGAT | 0.9159 (n=1590) | 0.8136 (n=829) | +0.1023 |

### bitcoin-otc

**out paths**

![bitcoin-otc out b2](bitcoin-otc/bitcoin-otc_out_fixed_b2.png)

![bitcoin-otc out b4](bitcoin-otc/bitcoin-otc_out_fixed_b4.png)

![bitcoin-otc out b8](bitcoin-otc/bitcoin-otc_out_fixed_b8.png)

![bitcoin-otc out b16](bitcoin-otc/bitcoin-otc_out_fixed_b16.png)

![bitcoin-otc out b32](bitcoin-otc/bitcoin-otc_out_fixed_b32.png)

**variant = `out` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9342 (n=1645) | 0.9352 (n=1722) | -0.0010 |
| walk_localattn4 | 0.9305 (n=1645) | 0.9349 (n=1722) | -0.0043 |
| GINEConv | 0.8452 (n=1645) | 0.9018 (n=1722) | -0.0566 |
| SiGAT | 0.8672 (n=1645) | 0.8570 (n=1722) | +0.0102 |

**in paths**

![bitcoin-otc in b2](bitcoin-otc/bitcoin-otc_in_fixed_b2.png)

![bitcoin-otc in b4](bitcoin-otc/bitcoin-otc_in_fixed_b4.png)

![bitcoin-otc in b8](bitcoin-otc/bitcoin-otc_in_fixed_b8.png)

![bitcoin-otc in b16](bitcoin-otc/bitcoin-otc_in_fixed_b16.png)

![bitcoin-otc in b32](bitcoin-otc/bitcoin-otc_in_fixed_b32.png)

**variant = `in` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9374 (n=2723) | 0.9374 (n=835) | +0.0000 |
| walk_localattn4 | 0.9361 (n=2723) | 0.9386 (n=835) | -0.0025 |
| GINEConv | 0.9019 (n=2723) | 0.8275 (n=835) | +0.0744 |
| SiGAT | 0.8984 (n=2723) | 0.7996 (n=835) | +0.0988 |

**inout paths**

![bitcoin-otc inout b2](bitcoin-otc/bitcoin-otc_inout_fixed_b2.png)

![bitcoin-otc inout b4](bitcoin-otc/bitcoin-otc_inout_fixed_b4.png)

![bitcoin-otc inout b8](bitcoin-otc/bitcoin-otc_inout_fixed_b8.png)

![bitcoin-otc inout b16](bitcoin-otc/bitcoin-otc_inout_fixed_b16.png)

![bitcoin-otc inout b32](bitcoin-otc/bitcoin-otc_inout_fixed_b32.png)

**variant = `inout` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9310 (n=1966) | 0.9456 (n=1594) | -0.0146 |
| walk_localattn4 | 0.9278 (n=1966) | 0.9464 (n=1594) | -0.0186 |
| GINEConv | 0.8798 (n=1966) | 0.8812 (n=1594) | -0.0014 |
| SiGAT | 0.8937 (n=1966) | 0.8469 (n=1594) | +0.0468 |

### epinions

**out paths**

![epinions out b2](epinions/epinions_out_fixed_b2.png)

![epinions out b4](epinions/epinions_out_fixed_b4.png)

![epinions out b8](epinions/epinions_out_fixed_b8.png)

![epinions out b16](epinions/epinions_out_fixed_b16.png)

![epinions out b32](epinions/epinions_out_fixed_b32.png)

**variant = `out` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9746 (n=14682) | 0.9415 (n=61315) | +0.0332 |
| walk_localattn4 | 0.9785 (n=14682) | 0.9415 (n=61315) | +0.0370 |
| GINEConv | 0.9130 (n=14682) | 0.8087 (n=61315) | +0.1043 |
| SiGAT | 0.9342 (n=14682) | 0.8906 (n=61315) | +0.0436 |

**in paths**

![epinions in b2](epinions/epinions_in_fixed_b2.png)

![epinions in b4](epinions/epinions_in_fixed_b4.png)

![epinions in b8](epinions/epinions_in_fixed_b8.png)

![epinions in b16](epinions/epinions_in_fixed_b16.png)

![epinions in b32](epinions/epinions_in_fixed_b32.png)

**variant = `in` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9744 (n=36371) | 0.9659 (n=40571) | +0.0084 |
| walk_localattn4 | 0.9735 (n=36371) | 0.9647 (n=40571) | +0.0088 |
| GINEConv | 0.9123 (n=36371) | 0.8395 (n=40571) | +0.0728 |
| SiGAT | 0.9294 (n=36371) | 0.9269 (n=40571) | +0.0024 |

**inout paths**

![epinions inout b2](epinions/epinions_inout_fixed_b2.png)

![epinions inout b4](epinions/epinions_inout_fixed_b4.png)

![epinions inout b8](epinions/epinions_inout_fixed_b8.png)

![epinions inout b16](epinions/epinions_inout_fixed_b16.png)

![epinions inout b32](epinions/epinions_inout_fixed_b32.png)

**variant = `inout` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9795 (n=16511) | 0.9522 (n=66170) | +0.0274 |
| walk_localattn4 | 0.9825 (n=16511) | 0.9519 (n=66170) | +0.0306 |
| GINEConv | 0.9449 (n=16511) | 0.8376 (n=66170) | +0.1074 |
| SiGAT | 0.9436 (n=16511) | 0.9096 (n=66170) | +0.0340 |

### wiki-elec

**out paths**

![wiki-elec out b2](wiki-elec/wiki-elec_out_fixed_b2.png)

![wiki-elec out b4](wiki-elec/wiki-elec_out_fixed_b4.png)

![wiki-elec out b8](wiki-elec/wiki-elec_out_fixed_b8.png)

![wiki-elec out b16](wiki-elec/wiki-elec_out_fixed_b16.png)

![wiki-elec out b32](wiki-elec/wiki-elec_out_fixed_b32.png)

**variant = `out` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9471 (n=405) | 0.8966 (n=6755) | +0.0506 |
| walk_localattn4 | 0.9486 (n=405) | 0.8944 (n=6755) | +0.0542 |
| GINEConv | 0.9051 (n=405) | 0.8592 (n=6755) | +0.0460 |
| SiGAT | 0.9218 (n=405) | 0.8817 (n=6755) | +0.0401 |

**in paths**

![wiki-elec in b2](wiki-elec/wiki-elec_in_fixed_b2.png)

![wiki-elec in b4](wiki-elec/wiki-elec_in_fixed_b4.png)

![wiki-elec in b8](wiki-elec/wiki-elec_in_fixed_b8.png)

![wiki-elec in b16](wiki-elec/wiki-elec_in_fixed_b16.png)

![wiki-elec in b32](wiki-elec/wiki-elec_in_fixed_b32.png)

**variant = `in` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9394 (n=755) | 0.9157 (n=5017) | +0.0237 |
| walk_localattn4 | 0.9384 (n=755) | 0.9165 (n=5017) | +0.0218 |
| GINEConv | 0.9151 (n=755) | 0.8828 (n=5017) | +0.0323 |
| SiGAT | 0.9440 (n=755) | 0.8979 (n=5017) | +0.0461 |

**inout paths**

![wiki-elec inout b2](wiki-elec/wiki-elec_inout_fixed_b2.png)

![wiki-elec inout b4](wiki-elec/wiki-elec_inout_fixed_b4.png)

![wiki-elec inout b8](wiki-elec/wiki-elec_inout_fixed_b8.png)

![wiki-elec inout b16](wiki-elec/wiki-elec_inout_fixed_b16.png)

![wiki-elec inout b32](wiki-elec/wiki-elec_inout_fixed_b32.png)

**variant = `inout` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9506 (n=526) | 0.9056 (n=8397) | +0.0449 |
| walk_localattn4 | 0.9553 (n=526) | 0.9053 (n=8397) | +0.0499 |
| GINEConv | 0.9207 (n=526) | 0.8730 (n=8397) | +0.0477 |
| SiGAT | 0.9479 (n=526) | 0.8909 (n=8397) | +0.0569 |

### wiki-rfa

**out paths**

![wiki-rfa out b2](wiki-rfa/wiki-rfa_out_fixed_b2.png)

![wiki-rfa out b4](wiki-rfa/wiki-rfa_out_fixed_b4.png)

![wiki-rfa out b8](wiki-rfa/wiki-rfa_out_fixed_b8.png)

![wiki-rfa out b16](wiki-rfa/wiki-rfa_out_fixed_b16.png)

![wiki-rfa out b32](wiki-rfa/wiki-rfa_out_fixed_b32.png)

**variant = `out` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9267 (n=685) | 0.8808 (n=14863) | +0.0459 |
| walk_localattn4 | 0.9110 (n=685) | 0.8815 (n=14863) | +0.0295 |
| GINEConv | 0.9040 (n=685) | 0.8523 (n=14863) | +0.0517 |
| SiGAT | 0.9204 (n=685) | 0.8681 (n=14863) | +0.0523 |

**in paths**

![wiki-rfa in b2](wiki-rfa/wiki-rfa_in_fixed_b2.png)

![wiki-rfa in b4](wiki-rfa/wiki-rfa_in_fixed_b4.png)

![wiki-rfa in b8](wiki-rfa/wiki-rfa_in_fixed_b8.png)

![wiki-rfa in b16](wiki-rfa/wiki-rfa_in_fixed_b16.png)

![wiki-rfa in b32](wiki-rfa/wiki-rfa_in_fixed_b32.png)

**variant = `in` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9244 (n=834) | 0.9088 (n=10326) | +0.0156 |
| walk_localattn4 | 0.9236 (n=834) | 0.9064 (n=10326) | +0.0172 |
| GINEConv | 0.8966 (n=834) | 0.8898 (n=10326) | +0.0067 |
| SiGAT | 0.9132 (n=834) | 0.8958 (n=10326) | +0.0174 |

**inout paths**

![wiki-rfa inout b2](wiki-rfa/wiki-rfa_inout_fixed_b2.png)

![wiki-rfa inout b4](wiki-rfa/wiki-rfa_inout_fixed_b4.png)

![wiki-rfa inout b8](wiki-rfa/wiki-rfa_inout_fixed_b8.png)

![wiki-rfa inout b16](wiki-rfa/wiki-rfa_inout_fixed_b16.png)

![wiki-rfa inout b32](wiki-rfa/wiki-rfa_inout_fixed_b32.png)

**variant = `inout` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9295 (n=483) | 0.8926 (n=16486) | +0.0368 |
| walk_localattn4 | 0.9189 (n=483) | 0.8920 (n=16486) | +0.0269 |
| GINEConv | 0.9138 (n=483) | 0.8673 (n=16486) | +0.0465 |
| SiGAT | 0.9264 (n=483) | 0.8813 (n=16486) | +0.0451 |

### slashdot090221

**out paths**

![slashdot090221 out b2](slashdot090221/slashdot090221_out_fixed_b2.png)

![slashdot090221 out b4](slashdot090221/slashdot090221_out_fixed_b4.png)

![slashdot090221 out b8](slashdot090221/slashdot090221_out_fixed_b8.png)

![slashdot090221 out b16](slashdot090221/slashdot090221_out_fixed_b16.png)

![slashdot090221 out b32](slashdot090221/slashdot090221_out_fixed_b32.png)

**variant = `out` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9205 (n=11222) | 0.8974 (n=26467) | +0.0231 |
| walk_localattn4 | 0.9177 (n=11222) | 0.8934 (n=26467) | +0.0243 |
| GINEConv | 0.8178 (n=11222) | 0.7762 (n=26467) | +0.0416 |
| SiGAT | 0.8848 (n=11222) | 0.8436 (n=26467) | +0.0412 |

**in paths**

![slashdot090221 in b2](slashdot090221/slashdot090221_in_fixed_b2.png)

![slashdot090221 in b4](slashdot090221/slashdot090221_in_fixed_b4.png)

![slashdot090221 in b8](slashdot090221/slashdot090221_in_fixed_b8.png)

![slashdot090221 in b16](slashdot090221/slashdot090221_in_fixed_b16.png)

![slashdot090221 in b32](slashdot090221/slashdot090221_in_fixed_b32.png)

**variant = `in` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9115 (n=8792) | 0.8961 (n=42391) | +0.0155 |
| walk_localattn4 | 0.9061 (n=8792) | 0.8929 (n=42391) | +0.0132 |
| GINEConv | 0.8367 (n=8792) | 0.7627 (n=42391) | +0.0740 |
| SiGAT | 0.8634 (n=8792) | 0.8515 (n=42391) | +0.0119 |

**inout paths**

![slashdot090221 inout b2](slashdot090221/slashdot090221_inout_fixed_b2.png)

![slashdot090221 inout b4](slashdot090221/slashdot090221_inout_fixed_b4.png)

![slashdot090221 inout b8](slashdot090221/slashdot090221_inout_fixed_b8.png)

![slashdot090221 inout b16](slashdot090221/slashdot090221_inout_fixed_b16.png)

![slashdot090221 inout b32](slashdot090221/slashdot090221_inout_fixed_b32.png)

**variant = `inout` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9070 (n=8036) | 0.8954 (n=45329) | +0.0116 |
| walk_localattn4 | 0.9040 (n=8036) | 0.8924 (n=45329) | +0.0116 |
| GINEConv | 0.8324 (n=8036) | 0.7664 (n=45329) | +0.0660 |
| SiGAT | 0.8636 (n=8036) | 0.8507 (n=45329) | +0.0129 |

### Combined (all datasets)

![ALL_DATASETS out b2](combined/ALL_DATASETS_out_fixed_b2.png)

![ALL_DATASETS out b4](combined/ALL_DATASETS_out_fixed_b4.png)

![ALL_DATASETS out b8](combined/ALL_DATASETS_out_fixed_b8.png)

![ALL_DATASETS out b16](combined/ALL_DATASETS_out_fixed_b16.png)

![ALL_DATASETS out b32](combined/ALL_DATASETS_out_fixed_b32.png)

![ALL_DATASETS in b2](combined/ALL_DATASETS_in_fixed_b2.png)

![ALL_DATASETS in b4](combined/ALL_DATASETS_in_fixed_b4.png)

![ALL_DATASETS in b8](combined/ALL_DATASETS_in_fixed_b8.png)

![ALL_DATASETS in b16](combined/ALL_DATASETS_in_fixed_b16.png)

![ALL_DATASETS in b32](combined/ALL_DATASETS_in_fixed_b32.png)

![ALL_DATASETS inout b2](combined/ALL_DATASETS_inout_fixed_b2.png)

![ALL_DATASETS inout b4](combined/ALL_DATASETS_inout_fixed_b4.png)

![ALL_DATASETS inout b8](combined/ALL_DATASETS_inout_fixed_b8.png)

![ALL_DATASETS inout b16](combined/ALL_DATASETS_inout_fixed_b16.png)

![ALL_DATASETS inout b32](combined/ALL_DATASETS_inout_fixed_b32.png)
