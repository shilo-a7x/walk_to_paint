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
| walk_full | 0.8897 (n=1450) | 0.8918 (n=888) | -0.0021 |
| walk_localattn4 | 0.9155 (n=1450) | 0.9224 (n=888) | -0.0069 |
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
| walk_full | 0.9081 (n=1892) | 0.9188 (n=523) | -0.0107 |
| walk_localattn4 | 0.9347 (n=1892) | 0.9354 (n=523) | -0.0007 |
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
| walk_full | 0.9032 (n=1590) | 0.9195 (n=829) | -0.0163 |
| walk_localattn4 | 0.9324 (n=1590) | 0.9374 (n=829) | -0.0051 |
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
| walk_full | 0.9352 (n=1645) | 0.9411 (n=1722) | -0.0059 |
| walk_localattn4 | 0.9299 (n=1645) | 0.9264 (n=1722) | +0.0035 |
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
| walk_full | 0.9412 (n=2723) | 0.9360 (n=835) | +0.0053 |
| walk_localattn4 | 0.9315 (n=2723) | 0.9255 (n=835) | +0.0060 |
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
| walk_full | 0.9331 (n=1966) | 0.9464 (n=1594) | -0.0134 |
| walk_localattn4 | 0.9191 (n=1966) | 0.9403 (n=1594) | -0.0211 |
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
| walk_full | 0.9619 (n=12603) | 0.9112 (n=54048) | +0.0507 |
| walk_localattn4 | 0.9718 (n=12603) | 0.9252 (n=54048) | +0.0466 |
| GINEConv | 0.9158 (n=12603) | 0.8118 (n=54048) | +0.1040 |
| SiGAT | 0.9319 (n=12603) | 0.8874 (n=54048) | +0.0445 |

**in paths**

![epinions in b2](epinions/epinions_in_fixed_b2.png)

![epinions in b4](epinions/epinions_in_fixed_b4.png)

![epinions in b8](epinions/epinions_in_fixed_b8.png)

![epinions in b16](epinions/epinions_in_fixed_b16.png)

![epinions in b32](epinions/epinions_in_fixed_b32.png)

**variant = `in` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9603 (n=31959) | 0.9467 (n=35592) | +0.0137 |
| walk_localattn4 | 0.9675 (n=31959) | 0.9573 (n=35592) | +0.0102 |
| GINEConv | 0.9124 (n=31959) | 0.8445 (n=35592) | +0.0679 |
| SiGAT | 0.9280 (n=31959) | 0.9228 (n=35592) | +0.0052 |

**inout paths**

![epinions inout b2](epinions/epinions_inout_fixed_b2.png)

![epinions inout b4](epinions/epinions_inout_fixed_b4.png)

![epinions inout b8](epinions/epinions_inout_fixed_b8.png)

![epinions inout b16](epinions/epinions_inout_fixed_b16.png)

![epinions inout b32](epinions/epinions_inout_fixed_b32.png)

**variant = `inout` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9687 (n=14171) | 0.9265 (n=58330) | +0.0423 |
| walk_localattn4 | 0.9770 (n=14171) | 0.9392 (n=58330) | +0.0378 |
| GINEConv | 0.9457 (n=14171) | 0.8419 (n=58330) | +0.1037 |
| SiGAT | 0.9410 (n=14171) | 0.9070 (n=58330) | +0.0340 |

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
| walk_full | 0.9002 (n=359) | 0.8835 (n=5756) | +0.0167 |
| walk_localattn4 | 0.9235 (n=359) | 0.8821 (n=5756) | +0.0415 |
| GINEConv | 0.9040 (n=359) | 0.8583 (n=5756) | +0.0457 |
| SiGAT | 0.9185 (n=359) | 0.8774 (n=5756) | +0.0411 |

**in paths**

![wiki-elec in b2](wiki-elec/wiki-elec_in_fixed_b2.png)

![wiki-elec in b4](wiki-elec/wiki-elec_in_fixed_b4.png)

![wiki-elec in b8](wiki-elec/wiki-elec_in_fixed_b8.png)

![wiki-elec in b16](wiki-elec/wiki-elec_in_fixed_b16.png)

![wiki-elec in b32](wiki-elec/wiki-elec_in_fixed_b32.png)

**variant = `in` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9457 (n=702) | 0.9125 (n=4906) | +0.0332 |
| walk_localattn4 | 0.9373 (n=702) | 0.9113 (n=4906) | +0.0260 |
| GINEConv | 0.9153 (n=702) | 0.8843 (n=4906) | +0.0309 |
| SiGAT | 0.9430 (n=702) | 0.8982 (n=4906) | +0.0448 |

**inout paths**

![wiki-elec inout b2](wiki-elec/wiki-elec_inout_fixed_b2.png)

![wiki-elec inout b4](wiki-elec/wiki-elec_inout_fixed_b4.png)

![wiki-elec inout b8](wiki-elec/wiki-elec_inout_fixed_b8.png)

![wiki-elec inout b16](wiki-elec/wiki-elec_inout_fixed_b16.png)

![wiki-elec inout b32](wiki-elec/wiki-elec_inout_fixed_b32.png)

**variant = `inout` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9297 (n=460) | 0.8968 (n=7367) | +0.0329 |
| walk_localattn4 | 0.9416 (n=460) | 0.8960 (n=7367) | +0.0455 |
| GINEConv | 0.9180 (n=460) | 0.8737 (n=7367) | +0.0443 |
| SiGAT | 0.9459 (n=460) | 0.8891 (n=7367) | +0.0567 |

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
| walk_full | 0.9108 (n=597) | 0.8673 (n=12794) | +0.0435 |
| walk_localattn4 | 0.9151 (n=597) | 0.8762 (n=12794) | +0.0389 |
| GINEConv | 0.9041 (n=597) | 0.8540 (n=12794) | +0.0500 |
| SiGAT | 0.9153 (n=597) | 0.8647 (n=12794) | +0.0506 |

**in paths**

![wiki-rfa in b2](wiki-rfa/wiki-rfa_in_fixed_b2.png)

![wiki-rfa in b4](wiki-rfa/wiki-rfa_in_fixed_b4.png)

![wiki-rfa in b8](wiki-rfa/wiki-rfa_in_fixed_b8.png)

![wiki-rfa in b16](wiki-rfa/wiki-rfa_in_fixed_b16.png)

![wiki-rfa in b32](wiki-rfa/wiki-rfa_in_fixed_b32.png)

**variant = `in` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9105 (n=660) | 0.9018 (n=10073) | +0.0087 |
| walk_localattn4 | 0.9062 (n=660) | 0.9085 (n=10073) | -0.0023 |
| GINEConv | 0.8938 (n=660) | 0.8898 (n=10073) | +0.0040 |
| SiGAT | 0.9066 (n=660) | 0.8954 (n=10073) | +0.0112 |

**inout paths**

![wiki-rfa inout b2](wiki-rfa/wiki-rfa_inout_fixed_b2.png)

![wiki-rfa inout b4](wiki-rfa/wiki-rfa_inout_fixed_b4.png)

![wiki-rfa inout b8](wiki-rfa/wiki-rfa_inout_fixed_b8.png)

![wiki-rfa inout b16](wiki-rfa/wiki-rfa_inout_fixed_b16.png)

![wiki-rfa inout b32](wiki-rfa/wiki-rfa_inout_fixed_b32.png)

**variant = `inout` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.8978 (n=369) | 0.8823 (n=14398) | +0.0155 |
| walk_localattn4 | 0.9047 (n=369) | 0.8895 (n=14398) | +0.0152 |
| GINEConv | 0.9135 (n=369) | 0.8708 (n=14398) | +0.0427 |
| SiGAT | 0.9185 (n=369) | 0.8807 (n=14398) | +0.0378 |

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
| walk_full | 0.9139 (n=11030) | 0.8886 (n=26044) | +0.0252 |
| walk_localattn4 | 0.9157 (n=11030) | 0.8894 (n=26044) | +0.0263 |
| GINEConv | 0.8184 (n=11030) | 0.7764 (n=26044) | +0.0420 |
| SiGAT | 0.8829 (n=11030) | 0.8417 (n=26044) | +0.0413 |

**in paths**

![slashdot090221 in b2](slashdot090221/slashdot090221_in_fixed_b2.png)

![slashdot090221 in b4](slashdot090221/slashdot090221_in_fixed_b4.png)

![slashdot090221 in b8](slashdot090221/slashdot090221_in_fixed_b8.png)

![slashdot090221 in b16](slashdot090221/slashdot090221_in_fixed_b16.png)

![slashdot090221 in b32](slashdot090221/slashdot090221_in_fixed_b32.png)

**variant = `in` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9083 (n=8647) | 0.8889 (n=41606) | +0.0194 |
| walk_localattn4 | 0.9098 (n=8647) | 0.8896 (n=41606) | +0.0202 |
| GINEConv | 0.8378 (n=8647) | 0.7633 (n=41606) | +0.0745 |
| SiGAT | 0.8632 (n=8647) | 0.8497 (n=41606) | +0.0135 |

**inout paths**

![slashdot090221 inout b2](slashdot090221/slashdot090221_inout_fixed_b2.png)

![slashdot090221 inout b4](slashdot090221/slashdot090221_inout_fixed_b4.png)

![slashdot090221 inout b8](slashdot090221/slashdot090221_inout_fixed_b8.png)

![slashdot090221 inout b16](slashdot090221/slashdot090221_inout_fixed_b16.png)

![slashdot090221 inout b32](slashdot090221/slashdot090221_inout_fixed_b32.png)

**variant = `inout` paths**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9046 (n=7903) | 0.8885 (n=44513) | +0.0162 |
| walk_localattn4 | 0.9066 (n=7903) | 0.8891 (n=44513) | +0.0175 |
| GINEConv | 0.8333 (n=7903) | 0.7670 (n=44513) | +0.0663 |
| SiGAT | 0.8615 (n=7903) | 0.8489 (n=44513) | +0.0126 |

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
