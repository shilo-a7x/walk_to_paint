# Lead 4: Node Sign-Entropy Heterogeneity vs. AUC (canonical, shared edges)

## Methodology

For each shared test edge `(u, v)`, bin source-node sign-entropy x
target-node sign-entropy into a 2D grid and compute AUC within each cell,
for 4 models: `walk_full` (E14_HARDNODE_L10, full attention),
`walk_localattn4` (E14_HARDNODE_L10_LOCALATTN4, +-2 hop banded attention),
`GINEConv`, `SiGAT`.

**Same-edge ground truth.** Per-edge predictions are read from
`predictions_raw_canonical.pkl` (built by `baselines/postprocess_canonical.py`
after the canonical-split baseline reruns); every model lives in the same raw
`(u, v)` id space. We restrict all models to the **shared edge set** (the
intersection of `(u, v)`, which equals the walk-covered test edges -- walk is
a strict subset of the GNN test sets). So every cell compares the four models
on the *identical* edges with the *identical* ground truth, and the per-cell
sample count `n` is the **same for all four models**. (This is the fix for the
old version, which bucketed each model over its own ~independent test sample;
see CANONICAL_RERUN_FINDINGS.md.)

**Entropy** is binary Shannon entropy in bits over a node's sign labels,
`H(p) = -p*log2(p) - (1-p)*log2(1-p)`, `p` = fraction of positive-sign edges in
the relevant direction. `H=0` -> homogeneous, `H=1` -> maximally heterogeneous.
Computed over **all edges** of the dataset (train+val+test) from the canonical
edge list -- a diagnostic grouping of already-trained models' predictions, not a
training-time feature, so no leakage. Nodes with zero edges in the relevant
direction are dropped.

**Entropy variants** -- source/target entropy from out-edges, in-edges, or
in+out (combined) signs of that node:

| variant | source | target |
|---|---|---|
| `out_out` | H(out-signs of u) | H(out-signs of v) |
| `in_in` | H(in-signs of u) | H(in-signs of v) |
| `out_in` | H(out-signs of u) | H(in-signs of v) |
| `inout_inout` | H(all incident signs of u) | H(all incident signs of v) |

**No retraining / recomputation here**: predictions are read straight from the
canonical pkl (walk-model func_logit_power aggregation, GINEConv saved
predictions, and the SiGAT logistic read-out all happened upstream). This
script only intersects edges, computes entropy, bins, and plots.

Gray cells (`n=.. (<min_n)` / `n=.. (1 class)` / empty) have no defined AUC.
Raw per-cell numbers are also dumped to `raw_data.txt`. Per-edge records
`(src_ent, tgt_ent, y, p)` are cached in `computed_data.pkl` for re-binning via
`--mode plot`.

## Results

### bitcoin-alpha

![bitcoin-alpha out_out b2](bitcoin-alpha/bitcoin-alpha_out_out_fixed_b2.png)

![bitcoin-alpha out_out b4](bitcoin-alpha/bitcoin-alpha_out_out_fixed_b4.png)

![bitcoin-alpha out_out b8](bitcoin-alpha/bitcoin-alpha_out_out_fixed_b8.png)

![bitcoin-alpha out_out b16](bitcoin-alpha/bitcoin-alpha_out_out_fixed_b16.png)

![bitcoin-alpha out_out b32](bitcoin-alpha/bitcoin-alpha_out_out_fixed_b32.png)

**variant = `out_out`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9091 (n=1616) | 0.7875 (n=109) | +0.1215 |
| walk_localattn4 | 0.9328 (n=1616) | 0.8718 (n=109) | +0.0610 |
| GINEConv | 0.8725 (n=1616) | 0.8037 (n=109) | +0.0689 |
| SiGAT | 0.9232 (n=1616) | 0.7370 (n=109) | +0.1862 |

![bitcoin-alpha in_in b2](bitcoin-alpha/bitcoin-alpha_in_in_fixed_b2.png)

![bitcoin-alpha in_in b4](bitcoin-alpha/bitcoin-alpha_in_in_fixed_b4.png)

![bitcoin-alpha in_in b8](bitcoin-alpha/bitcoin-alpha_in_in_fixed_b8.png)

![bitcoin-alpha in_in b16](bitcoin-alpha/bitcoin-alpha_in_in_fixed_b16.png)

![bitcoin-alpha in_in b32](bitcoin-alpha/bitcoin-alpha_in_in_fixed_b32.png)

**variant = `in_in`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9386 (n=1919) | 0.7361 (n=51) | +0.2025 |
| walk_localattn4 | 0.9650 (n=1919) | 0.7531 (n=51) | +0.2119 |
| GINEConv | 0.8415 (n=1919) | 0.6975 (n=51) | +0.1440 |
| SiGAT | 0.9024 (n=1919) | 0.5463 (n=51) | +0.3561 |

![bitcoin-alpha out_in b2](bitcoin-alpha/bitcoin-alpha_out_in_fixed_b2.png)

![bitcoin-alpha out_in b4](bitcoin-alpha/bitcoin-alpha_out_in_fixed_b4.png)

![bitcoin-alpha out_in b8](bitcoin-alpha/bitcoin-alpha_out_in_fixed_b8.png)

![bitcoin-alpha out_in b16](bitcoin-alpha/bitcoin-alpha_out_in_fixed_b16.png)

![bitcoin-alpha out_in b32](bitcoin-alpha/bitcoin-alpha_out_in_fixed_b32.png)

**variant = `out_in`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9562 (n=1762) | 0.6825 (n=108) | +0.2737 |
| walk_localattn4 | 0.9732 (n=1762) | 0.7511 (n=108) | +0.2221 |
| GINEConv | 0.8731 (n=1762) | 0.6338 (n=108) | +0.2393 |
| SiGAT | 0.9280 (n=1762) | 0.6190 (n=108) | +0.3090 |

![bitcoin-alpha inout_inout b2](bitcoin-alpha/bitcoin-alpha_inout_inout_fixed_b2.png)

![bitcoin-alpha inout_inout b4](bitcoin-alpha/bitcoin-alpha_inout_inout_fixed_b4.png)

![bitcoin-alpha inout_inout b8](bitcoin-alpha/bitcoin-alpha_inout_inout_fixed_b8.png)

![bitcoin-alpha inout_inout b16](bitcoin-alpha/bitcoin-alpha_inout_inout_fixed_b16.png)

![bitcoin-alpha inout_inout b32](bitcoin-alpha/bitcoin-alpha_inout_inout_fixed_b32.png)

**variant = `inout_inout`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.8610 (n=1698) | 0.8057 (n=137) | +0.0554 |
| walk_localattn4 | 0.8572 (n=1698) | 0.8402 (n=137) | +0.0170 |
| GINEConv | 0.7740 (n=1698) | 0.7267 (n=137) | +0.0473 |
| SiGAT | 0.8695 (n=1698) | 0.7209 (n=137) | +0.1485 |

### bitcoin-otc

![bitcoin-otc out_out b2](bitcoin-otc/bitcoin-otc_out_out_fixed_b2.png)

![bitcoin-otc out_out b4](bitcoin-otc/bitcoin-otc_out_out_fixed_b4.png)

![bitcoin-otc out_out b8](bitcoin-otc/bitcoin-otc_out_out_fixed_b8.png)

![bitcoin-otc out_out b16](bitcoin-otc/bitcoin-otc_out_out_fixed_b16.png)

![bitcoin-otc out_out b32](bitcoin-otc/bitcoin-otc_out_out_fixed_b32.png)

**variant = `out_out`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.8773 (n=2192) | 0.9186 (n=210) | -0.0413 |
| walk_localattn4 | 0.8636 (n=2192) | 0.9149 (n=210) | -0.0512 |
| GINEConv | 0.8706 (n=2192) | 0.8416 (n=210) | +0.0289 |
| SiGAT | 0.8266 (n=2192) | 0.6832 (n=210) | +0.1433 |

![bitcoin-otc in_in b2](bitcoin-otc/bitcoin-otc_in_in_fixed_b2.png)

![bitcoin-otc in_in b4](bitcoin-otc/bitcoin-otc_in_in_fixed_b4.png)

![bitcoin-otc in_in b8](bitcoin-otc/bitcoin-otc_in_in_fixed_b8.png)

![bitcoin-otc in_in b16](bitcoin-otc/bitcoin-otc_in_in_fixed_b16.png)

![bitcoin-otc in_in b32](bitcoin-otc/bitcoin-otc_in_in_fixed_b32.png)

**variant = `in_in`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9397 (n=2415) | 0.8478 (n=219) | +0.0920 |
| walk_localattn4 | 0.9211 (n=2415) | 0.8536 (n=219) | +0.0675 |
| GINEConv | 0.8678 (n=2415) | 0.6044 (n=219) | +0.2634 |
| SiGAT | 0.8713 (n=2415) | 0.4707 (n=219) | +0.4006 |

![bitcoin-otc out_in b2](bitcoin-otc/bitcoin-otc_out_in_fixed_b2.png)

![bitcoin-otc out_in b4](bitcoin-otc/bitcoin-otc_out_in_fixed_b4.png)

![bitcoin-otc out_in b8](bitcoin-otc/bitcoin-otc_out_in_fixed_b8.png)

![bitcoin-otc out_in b16](bitcoin-otc/bitcoin-otc_out_in_fixed_b16.png)

![bitcoin-otc out_in b32](bitcoin-otc/bitcoin-otc_out_in_fixed_b32.png)

**variant = `out_in`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9180 (n=2311) | 0.7776 (n=270) | +0.1405 |
| walk_localattn4 | 0.9055 (n=2311) | 0.7811 (n=270) | +0.1245 |
| GINEConv | 0.9114 (n=2311) | 0.6511 (n=270) | +0.2604 |
| SiGAT | 0.8388 (n=2311) | 0.5406 (n=270) | +0.2982 |

![bitcoin-otc inout_inout b2](bitcoin-otc/bitcoin-otc_inout_inout_fixed_b2.png)

![bitcoin-otc inout_inout b4](bitcoin-otc/bitcoin-otc_inout_inout_fixed_b4.png)

![bitcoin-otc inout_inout b8](bitcoin-otc/bitcoin-otc_inout_inout_fixed_b8.png)

![bitcoin-otc inout_inout b16](bitcoin-otc/bitcoin-otc_inout_inout_fixed_b16.png)

![bitcoin-otc inout_inout b32](bitcoin-otc/bitcoin-otc_inout_inout_fixed_b32.png)

**variant = `inout_inout`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.8406 (n=2193) | 0.8589 (n=346) | -0.0183 |
| walk_localattn4 | 0.8274 (n=2193) | 0.8462 (n=346) | -0.0188 |
| GINEConv | 0.8330 (n=2193) | 0.6990 (n=346) | +0.1340 |
| SiGAT | 0.8048 (n=2193) | 0.5636 (n=346) | +0.2412 |

### epinions

![epinions out_out b2](epinions/epinions_out_out_fixed_b2.png)

![epinions out_out b4](epinions/epinions_out_out_fixed_b4.png)

![epinions out_out b8](epinions/epinions_out_out_fixed_b8.png)

![epinions out_out b16](epinions/epinions_out_out_fixed_b16.png)

![epinions out_out b32](epinions/epinions_out_out_fixed_b32.png)

**variant = `out_out`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9047 (n=31062) | 0.8582 (n=8782) | +0.0465 |
| walk_localattn4 | 0.9290 (n=31062) | 0.8695 (n=8782) | +0.0595 |
| GINEConv | 0.8445 (n=31062) | 0.8145 (n=8782) | +0.0300 |
| SiGAT | 0.9226 (n=31062) | 0.7588 (n=8782) | +0.1638 |

![epinions in_in b2](epinions/epinions_in_in_fixed_b2.png)

![epinions in_in b4](epinions/epinions_in_in_fixed_b4.png)

![epinions in_in b8](epinions/epinions_in_in_fixed_b8.png)

![epinions in_in b16](epinions/epinions_in_in_fixed_b16.png)

![epinions in_in b32](epinions/epinions_in_in_fixed_b32.png)

**variant = `in_in`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9653 (n=41986) | 0.9036 (n=4603) | +0.0617 |
| walk_localattn4 | 0.9729 (n=41986) | 0.9145 (n=4603) | +0.0584 |
| GINEConv | 0.9058 (n=41986) | 0.7048 (n=4603) | +0.2011 |
| SiGAT | 0.9418 (n=41986) | 0.8629 (n=4603) | +0.0789 |

![epinions out_in b2](epinions/epinions_out_in_fixed_b2.png)

![epinions out_in b4](epinions/epinions_out_in_fixed_b4.png)

![epinions out_in b8](epinions/epinions_out_in_fixed_b8.png)

![epinions out_in b16](epinions/epinions_out_in_fixed_b16.png)

![epinions out_in b32](epinions/epinions_out_in_fixed_b32.png)

**variant = `out_in`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9253 (n=42608) | 0.8010 (n=6003) | +0.1243 |
| walk_localattn4 | 0.9430 (n=42608) | 0.8120 (n=6003) | +0.1311 |
| GINEConv | 0.8444 (n=42608) | 0.7465 (n=6003) | +0.0979 |
| SiGAT | 0.9261 (n=42608) | 0.7605 (n=6003) | +0.1656 |

![epinions inout_inout b2](epinions/epinions_inout_inout_fixed_b2.png)

![epinions inout_inout b4](epinions/epinions_inout_inout_fixed_b4.png)

![epinions inout_inout b8](epinions/epinions_inout_inout_fixed_b8.png)

![epinions inout_inout b16](epinions/epinions_inout_inout_fixed_b16.png)

![epinions inout_inout b32](epinions/epinions_inout_inout_fixed_b32.png)

**variant = `inout_inout`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.8826 (n=34938) | 0.8801 (n=11259) | +0.0025 |
| walk_localattn4 | 0.9120 (n=34938) | 0.8971 (n=11259) | +0.0149 |
| GINEConv | 0.8236 (n=34938) | 0.8223 (n=11259) | +0.0012 |
| SiGAT | 0.8911 (n=34938) | 0.8339 (n=11259) | +0.0572 |

### wiki-elec

![wiki-elec out_out b2](wiki-elec/wiki-elec_out_out_fixed_b2.png)

![wiki-elec out_out b4](wiki-elec/wiki-elec_out_out_fixed_b4.png)

![wiki-elec out_out b8](wiki-elec/wiki-elec_out_out_fixed_b8.png)

![wiki-elec out_out b16](wiki-elec/wiki-elec_out_out_fixed_b16.png)

![wiki-elec out_out b32](wiki-elec/wiki-elec_out_out_fixed_b32.png)

**variant = `out_out`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9396 (n=930) | 0.8280 (n=2381) | +0.1116 |
| walk_localattn4 | 0.9270 (n=930) | 0.8325 (n=2381) | +0.0945 |
| GINEConv | 0.8407 (n=930) | 0.8524 (n=2381) | -0.0117 |
| SiGAT | 0.8939 (n=930) | 0.8401 (n=2381) | +0.0538 |

![wiki-elec in_in b2](wiki-elec/wiki-elec_in_in_fixed_b2.png)

![wiki-elec in_in b4](wiki-elec/wiki-elec_in_in_fixed_b4.png)

![wiki-elec in_in b8](wiki-elec/wiki-elec_in_in_fixed_b8.png)

![wiki-elec in_in b16](wiki-elec/wiki-elec_in_in_fixed_b16.png)

![wiki-elec in_in b32](wiki-elec/wiki-elec_in_in_fixed_b32.png)

**variant = `in_in`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9690 (n=2012) | 0.8032 (n=1079) | +0.1658 |
| walk_localattn4 | 0.9653 (n=2012) | 0.8042 (n=1079) | +0.1610 |
| GINEConv | 0.9318 (n=2012) | 0.7036 (n=1079) | +0.2282 |
| SiGAT | 0.9440 (n=2012) | 0.7736 (n=1079) | +0.1704 |

![wiki-elec out_in b2](wiki-elec/wiki-elec_out_in_fixed_b2.png)

![wiki-elec out_in b4](wiki-elec/wiki-elec_out_in_fixed_b4.png)

![wiki-elec out_in b8](wiki-elec/wiki-elec_out_in_fixed_b8.png)

![wiki-elec out_in b16](wiki-elec/wiki-elec_out_in_fixed_b16.png)

![wiki-elec out_in b32](wiki-elec/wiki-elec_out_in_fixed_b32.png)

**variant = `out_in`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9599 (n=1908) | 0.7226 (n=2808) | +0.2373 |
| walk_localattn4 | 0.9427 (n=1908) | 0.7268 (n=2808) | +0.2159 |
| GINEConv | 0.8819 (n=1908) | 0.7186 (n=2808) | +0.1633 |
| SiGAT | 0.9431 (n=1908) | 0.7195 (n=2808) | +0.2236 |

![wiki-elec inout_inout b2](wiki-elec/wiki-elec_inout_inout_fixed_b2.png)

![wiki-elec inout_inout b4](wiki-elec/wiki-elec_inout_inout_fixed_b4.png)

![wiki-elec inout_inout b8](wiki-elec/wiki-elec_inout_inout_fixed_b8.png)

![wiki-elec inout_inout b16](wiki-elec/wiki-elec_inout_inout_fixed_b16.png)

![wiki-elec inout_inout b32](wiki-elec/wiki-elec_inout_inout_fixed_b32.png)

**variant = `inout_inout`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9606 (n=1659) | 0.7830 (n=3362) | +0.1776 |
| walk_localattn4 | 0.9545 (n=1659) | 0.7878 (n=3362) | +0.1667 |
| GINEConv | 0.9060 (n=1659) | 0.7763 (n=3362) | +0.1297 |
| SiGAT | 0.9322 (n=1659) | 0.7776 (n=3362) | +0.1546 |

### wiki-rfa

![wiki-rfa out_out b2](wiki-rfa/wiki-rfa_out_out_fixed_b2.png)

![wiki-rfa out_out b4](wiki-rfa/wiki-rfa_out_out_fixed_b4.png)

![wiki-rfa out_out b8](wiki-rfa/wiki-rfa_out_out_fixed_b8.png)

![wiki-rfa out_out b16](wiki-rfa/wiki-rfa_out_out_fixed_b16.png)

![wiki-rfa out_out b32](wiki-rfa/wiki-rfa_out_out_fixed_b32.png)

**variant = `out_out`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.8996 (n=1923) | 0.8247 (n=5409) | +0.0749 |
| walk_localattn4 | 0.9122 (n=1923) | 0.8298 (n=5409) | +0.0824 |
| GINEConv | 0.8893 (n=1923) | 0.8427 (n=5409) | +0.0466 |
| SiGAT | 0.9068 (n=1923) | 0.8180 (n=5409) | +0.0888 |

![wiki-rfa in_in b2](wiki-rfa/wiki-rfa_in_in_fixed_b2.png)

![wiki-rfa in_in b4](wiki-rfa/wiki-rfa_in_in_fixed_b4.png)

![wiki-rfa in_in b8](wiki-rfa/wiki-rfa_in_in_fixed_b8.png)

![wiki-rfa in_in b16](wiki-rfa/wiki-rfa_in_in_fixed_b16.png)

![wiki-rfa in_in b32](wiki-rfa/wiki-rfa_in_in_fixed_b32.png)

**variant = `in_in`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9613 (n=3380) | 0.7891 (n=2360) | +0.1722 |
| walk_localattn4 | 0.9701 (n=3380) | 0.7938 (n=2360) | +0.1763 |
| GINEConv | 0.9347 (n=3380) | 0.7413 (n=2360) | +0.1933 |
| SiGAT | 0.9625 (n=3380) | 0.7710 (n=2360) | +0.1915 |

![wiki-rfa out_in b2](wiki-rfa/wiki-rfa_out_in_fixed_b2.png)

![wiki-rfa out_in b4](wiki-rfa/wiki-rfa_out_in_fixed_b4.png)

![wiki-rfa out_in b8](wiki-rfa/wiki-rfa_out_in_fixed_b8.png)

![wiki-rfa out_in b16](wiki-rfa/wiki-rfa_out_in_fixed_b16.png)

![wiki-rfa out_in b32](wiki-rfa/wiki-rfa_out_in_fixed_b32.png)

**variant = `out_in`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9678 (n=3063) | 0.7288 (n=5187) | +0.2390 |
| walk_localattn4 | 0.9607 (n=3063) | 0.7322 (n=5187) | +0.2285 |
| GINEConv | 0.8962 (n=3063) | 0.7354 (n=5187) | +0.1608 |
| SiGAT | 0.9509 (n=3063) | 0.7206 (n=5187) | +0.2303 |

![wiki-rfa inout_inout b2](wiki-rfa/wiki-rfa_inout_inout_fixed_b2.png)

![wiki-rfa inout_inout b4](wiki-rfa/wiki-rfa_inout_inout_fixed_b4.png)

![wiki-rfa inout_inout b8](wiki-rfa/wiki-rfa_inout_inout_fixed_b8.png)

![wiki-rfa inout_inout b16](wiki-rfa/wiki-rfa_inout_inout_fixed_b16.png)

![wiki-rfa inout_inout b32](wiki-rfa/wiki-rfa_inout_inout_fixed_b32.png)

**variant = `inout_inout`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9706 (n=2519) | 0.7903 (n=6265) | +0.1803 |
| walk_localattn4 | 0.9750 (n=2519) | 0.7951 (n=6265) | +0.1799 |
| GINEConv | 0.9150 (n=2519) | 0.7883 (n=6265) | +0.1266 |
| SiGAT | 0.9724 (n=2519) | 0.7737 (n=6265) | +0.1987 |

### slashdot090221

![slashdot090221 out_out b2](slashdot090221/slashdot090221_out_out_fixed_b2.png)

![slashdot090221 out_out b4](slashdot090221/slashdot090221_out_out_fixed_b4.png)

![slashdot090221 out_out b8](slashdot090221/slashdot090221_out_out_fixed_b8.png)

![slashdot090221 out_out b16](slashdot090221/slashdot090221_out_out_fixed_b16.png)

![slashdot090221 out_out b32](slashdot090221/slashdot090221_out_out_fixed_b32.png)

**variant = `out_out`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9522 (n=14329) | 0.7370 (n=6766) | +0.2153 |
| walk_localattn4 | 0.9550 (n=14329) | 0.7418 (n=6766) | +0.2131 |
| GINEConv | 0.8398 (n=14329) | 0.7034 (n=6766) | +0.1364 |
| SiGAT | 0.9180 (n=14329) | 0.6751 (n=6766) | +0.2429 |

![slashdot090221 in_in b2](slashdot090221/slashdot090221_in_in_fixed_b2.png)

![slashdot090221 in_in b4](slashdot090221/slashdot090221_in_in_fixed_b4.png)

![slashdot090221 in_in b8](slashdot090221/slashdot090221_in_in_fixed_b8.png)

![slashdot090221 in_in b16](slashdot090221/slashdot090221_in_in_fixed_b16.png)

![slashdot090221 in_in b32](slashdot090221/slashdot090221_in_in_fixed_b32.png)

**variant = `in_in`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9348 (n=12279) | 0.8303 (n=17016) | +0.1045 |
| walk_localattn4 | 0.9350 (n=12279) | 0.8337 (n=17016) | +0.1013 |
| GINEConv | 0.8392 (n=12279) | 0.6052 (n=17016) | +0.2340 |
| SiGAT | 0.8913 (n=12279) | 0.7824 (n=17016) | +0.1089 |

![slashdot090221 out_in b2](slashdot090221/slashdot090221_out_in_fixed_b2.png)

![slashdot090221 out_in b4](slashdot090221/slashdot090221_out_in_fixed_b4.png)

![slashdot090221 out_in b8](slashdot090221/slashdot090221_out_in_fixed_b8.png)

![slashdot090221 out_in b16](slashdot090221/slashdot090221_out_in_fixed_b16.png)

![slashdot090221 out_in b32](slashdot090221/slashdot090221_out_in_fixed_b32.png)

**variant = `out_in`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9636 (n=15421) | 0.6899 (n=15607) | +0.2737 |
| walk_localattn4 | 0.9612 (n=15421) | 0.6952 (n=15607) | +0.2660 |
| GINEConv | 0.8566 (n=15421) | 0.6210 (n=15607) | +0.2355 |
| SiGAT | 0.9262 (n=15421) | 0.6629 (n=15607) | +0.2632 |

![slashdot090221 inout_inout b2](slashdot090221/slashdot090221_inout_inout_fixed_b2.png)

![slashdot090221 inout_inout b4](slashdot090221/slashdot090221_inout_inout_fixed_b4.png)

![slashdot090221 inout_inout b8](slashdot090221/slashdot090221_inout_inout_fixed_b8.png)

![slashdot090221 inout_inout b16](slashdot090221/slashdot090221_inout_inout_fixed_b16.png)

![slashdot090221 inout_inout b32](slashdot090221/slashdot090221_inout_inout_fixed_b32.png)

**variant = `inout_inout`**

| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |
|---|---|---|---|
| walk_full | 0.9497 (n=13483) | 0.7577 (n=18485) | +0.1921 |
| walk_localattn4 | 0.9546 (n=13483) | 0.7600 (n=18485) | +0.1947 |
| GINEConv | 0.8700 (n=13483) | 0.6102 (n=18485) | +0.2598 |
| SiGAT | 0.9093 (n=13483) | 0.7011 (n=18485) | +0.2082 |

### Combined (all datasets)

![ALL_DATASETS out_out b2](combined/ALL_DATASETS_out_out_fixed_b2.png)

![ALL_DATASETS out_out b4](combined/ALL_DATASETS_out_out_fixed_b4.png)

![ALL_DATASETS out_out b8](combined/ALL_DATASETS_out_out_fixed_b8.png)

![ALL_DATASETS out_out b16](combined/ALL_DATASETS_out_out_fixed_b16.png)

![ALL_DATASETS out_out b32](combined/ALL_DATASETS_out_out_fixed_b32.png)

![ALL_DATASETS in_in b2](combined/ALL_DATASETS_in_in_fixed_b2.png)

![ALL_DATASETS in_in b4](combined/ALL_DATASETS_in_in_fixed_b4.png)

![ALL_DATASETS in_in b8](combined/ALL_DATASETS_in_in_fixed_b8.png)

![ALL_DATASETS in_in b16](combined/ALL_DATASETS_in_in_fixed_b16.png)

![ALL_DATASETS in_in b32](combined/ALL_DATASETS_in_in_fixed_b32.png)

![ALL_DATASETS out_in b2](combined/ALL_DATASETS_out_in_fixed_b2.png)

![ALL_DATASETS out_in b4](combined/ALL_DATASETS_out_in_fixed_b4.png)

![ALL_DATASETS out_in b8](combined/ALL_DATASETS_out_in_fixed_b8.png)

![ALL_DATASETS out_in b16](combined/ALL_DATASETS_out_in_fixed_b16.png)

![ALL_DATASETS out_in b32](combined/ALL_DATASETS_out_in_fixed_b32.png)

![ALL_DATASETS inout_inout b2](combined/ALL_DATASETS_inout_inout_fixed_b2.png)

![ALL_DATASETS inout_inout b4](combined/ALL_DATASETS_inout_inout_fixed_b4.png)

![ALL_DATASETS inout_inout b8](combined/ALL_DATASETS_inout_inout_fixed_b8.png)

![ALL_DATASETS inout_inout b16](combined/ALL_DATASETS_inout_inout_fixed_b16.png)

![ALL_DATASETS inout_inout b32](combined/ALL_DATASETS_inout_inout_fixed_b32.png)
