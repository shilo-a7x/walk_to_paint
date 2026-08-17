# Per-dataset node-entropy regression — 6 independent coefficient sets

This package fits **one independent logistic regression per dataset** (not pooled across
datasets) relating node sign-entropy to prediction accuracy, for each of 4 models. Every
(dataset, model) pair gets its own intercept and its own 4 slopes — **6 separate sets of
coefficients per model** (24 total fits), rather than one shared slope across all datasets.
It is self-contained — the data, the fitting code, and this description are everything
needed to inspect, verify, or extend the analysis without the parent research repository.

## 1. Background — what problem the models solve

The task is **edge sign prediction in directed signed graphs**: each edge `u -> v` carries a
label (e.g. "trust"/"distrust", "+"/"-"), and the model predicts the sign of held-out edges
from the graph structure and the signs of observed edges. Six public directed signed-graph
datasets are used:

| Dataset | Nodes | Edges | Domain |
|---|---|---|---|
| bitcoin-alpha | 3,783 | 24,186 | trust ratings between traders |
| bitcoin-otc | 5,901 | 35,592 | trust ratings between traders |
| epinions | ~131,000 | 841,372 | trust/distrust between reviewers |
| wiki-elec | ~7,100 | 103,689 | admin-election support/oppose votes |
| wiki-rfa | ~11,400 | 184,546 | admin-election support/oppose votes |
| slashdot090221 | ~82,000 | 549,202 | friend/foe relationships |

All six use the same canonical split: 80% train / 10% validation / 10% test, fixed seed,
identical across every model compared below (so all models are scored on the *same* held-out
edges — an "apples-to-apples" comparison, not each model's own independently-drawn test set).

## 2. The four models compared

| Model | Type | What it does |
|---|---|---|
| **walk_full** | Random-walk Transformer, full attention | Samples random walks through the graph anchored at the target edge, tokenizes each walk as an alternating node/edge sequence, and trains a Transformer encoder to predict a masked edge's sign from the rest of the walk. Full self-attention: every token can attend to every other token in the walk. |
| **walk_localattn4** | Same walk-Transformer, restricted attention | Identical to walk_full except attention is masked to a ±2-hop window around each position instead of attending across the whole walk. |
| **GINEConv** | Message-passing GNN | A standard edge-feature-aware graph neural network. Builds one embedding per node by aggregating (summing) messages from its neighbors, then reads the edge label off the pair of endpoint embeddings. |
| **SiGAT** | Graph-attention signed GNN | A signed-graph-specific attention-based GNN (multiple parallel attention channels distinguishing edge sign/direction), also read out via endpoint embeddings. |

All four are evaluated on the identical shared (intersection) test-edge set per dataset,
using each model's own trained predictions. **No models are retrained or modified in this
package** — this is a pure post-hoc statistical analysis of already-computed predictions.
GINEConv's predictions come directly from its own trained edge classifier; SiGAT's come from
its trained node embeddings plus a separate logistic read-out fit on the train split (SiGAT
doesn't produce an edge score directly). The walk models are the current production
checkpoints (edge-anchored coverage-guaranteed random-walk sampler, per-dataset walk budget;
`walk_localattn4` restricts attention to a ±2-hop window, `walk_full` doesn't).

## 3. What "correct" means, and what the regression does

For every test edge `(u, v)` and every model, we have a predicted probability `p` that the
edge is positive, and the true label `y`. Define:

```
correct(u, v) = 1  if (p >= 0.5) == y,  else 0
```

i.e. a per-edge, per-model binary indicator of whether that model's hard prediction was
right. **The regression asks: how does `correct` vary with the local sign-entropy around
that edge's endpoints?**

### 3.1 Entropy features (the coefficients)

For a node `n`, define the **binary Shannon entropy of its sign distribution** in a given
direction:

```
p = fraction of n's edges (in that direction) that are positive
H(n) = -p*log2(p) - (1-p)*log2(1-p)          (H = 0 if p in {0, 1})
```

`H = 0` means the node's edges in that direction are all one sign (perfectly predictable /
homogeneous); `H = 1` means a 50/50 split (maximally unpredictable / heterogeneous). This
package uses the **4 node-level directional entropy terms** (no 2-hop/path terms):

| Term | Definition | Reads as |
|---|---|---|
| `src_out` | H of **u's outgoing** edge signs | how consistently does u rate others? |
| `src_in` | H of **u's incoming** edge signs | how consistently is u itself rated? |
| `tgt_out` | H of **v's outgoing** edge signs | how consistently does v rate others? |
| `tgt_in` | H of **v's incoming** edge signs | how contested is v's reputation already? |

All four are computed from the full graph (all edges, not just train) — this is a read-only
diagnostic of already-trained models' predictions, so there is no leakage risk (nothing is
fed back into training).

### 3.2 The model: 4 terms + bias, fit separately per dataset

```
logit(P(correct)) = const + b1*src_out + b2*src_in + b3*tgt_out + b4*tgt_in
```

Fit **independently for each of the 6 datasets and each of the 4 models — 24 regressions
total, no data pooled across datasets.** Each fit produces its own intercept (`const`) and
its own 4 slopes. This is different from (and a companion to) a *pooled* fit, which would
stack all 6 datasets together and force one shared slope across all of them (with only the
intercept allowed to vary by dataset) — here, every coefficient, including the slopes, is
free to differ from dataset to dataset. That's the right setup if you want to see how much
the relationship itself varies by graph, rather than assuming one universal slope.

### 3.3 Z-scored variant

Every fit is run twice: once on the raw entropy values (bits), and once with every entropy
column standardized (mean 0, standard deviation 1) **before** fitting for that specific
dataset — an actual re-fit on standardized inputs, not a post-hoc rescaling of the raw beta.
Z-scored coefficients are directly comparable in magnitude across terms and across
models/datasets (a raw beta depends on that term's natural scale, which differs by dataset).

### 3.4 Standard errors and significance

Because many edges share an endpoint (the same node `u` or `v` appears in many test edges),
edges are not independent draws — naive standard errors understate uncertainty. Every fit
uses **two-way cluster-robust standard errors** (clustered on both the source node and the
target node, Cameron-Gelbach-Miller 2011 formula: `V = V_u + V_v - V_(u,v)`), and p-values
within each (dataset, model) fit's 4 slope terms are corrected for multiple comparisons via
**Benjamini-Hochberg FDR** (column `p_fdr`) — the intercept is not included in that
correction, matching the convention used elsewhere in this analysis.

## 4. The 6 sets of coefficients, per model (raw scale)

Each table below is one model's 6 independent per-dataset fits.

### walk_full

| Dataset | const | src_out | src_in | tgt_out | tgt_in |
|---|---|---|---|---|---|
| bitcoin-alpha | 4.659 | -3.429 | -0.188 | -0.511 | -3.458 |
| bitcoin-otc | 4.239 | -2.017 | -1.202 | -0.589 | -2.473 |
| epinions | 5.196 | -2.602 | -0.757 | -0.356 | -2.240 |
| slashdot090221 | 4.173 | -3.077 | -0.087 | -0.166 | -1.257 |
| wiki-elec | 5.600 | -2.171 | -0.237 | -0.369 | -3.636 |
| wiki-rfa | 5.567 | -1.866 | -0.155 | -0.259 | -3.953 |

### walk_localattn4

| Dataset | const | src_out | src_in | tgt_out | tgt_in |
|---|---|---|---|---|---|
| bitcoin-alpha | 5.208 | -3.548 | 0.366 | -0.093 | -3.718 |
| bitcoin-otc | 4.577 | -1.967 | -0.395 | 0.506 | -3.185 |
| epinions | 5.326 | -2.625 | -0.653 | -0.219 | -2.319 |
| slashdot090221 | 4.354 | -3.040 | 0.024 | -0.094 | -1.546 |
| wiki-elec | 5.588 | -2.227 | -0.222 | -0.397 | -3.507 |
| wiki-rfa | 5.740 | -2.192 | 0.010 | -0.206 | -3.991 |

### GINEConv

| Dataset | const | src_out | src_in | tgt_out | tgt_in |
|---|---|---|---|---|---|
| bitcoin-alpha | 6.236 | -3.426 | -0.674 | 0.218 | -4.549 |
| bitcoin-otc | 5.471 | -1.672 | -1.089 | 0.217 | -4.090 |
| epinions | 5.120 | -1.808 | -1.391 | 0.229 | -3.318 |
| slashdot090221 | 4.512 | -0.839 | -1.025 | 0.019 | -3.297 |
| wiki-elec | 4.951 | -0.672 | 0.095 | 0.137 | -4.612 |
| wiki-rfa | 6.347 | -1.994 | 0.070 | 0.021 | -4.775 |

### SiGAT

| Dataset | const | src_out | src_in | tgt_out | tgt_in |
|---|---|---|---|---|---|
| bitcoin-alpha | 6.246 | -3.681 | -1.272 | 0.227 | -4.327 |
| bitcoin-otc | 5.596 | -2.156 | -1.866 | 0.347 | -4.239 |
| epinions | 5.202 | -2.677 | -0.861 | 0.063 | -2.354 |
| slashdot090221 | 4.701 | -2.885 | -0.224 | -0.081 | -1.956 |
| wiki-elec | 5.548 | -2.369 | 0.119 | -0.002 | -3.491 |
| wiki-rfa | 5.953 | -2.154 | 0.048 | -0.088 | -4.062 |

**Reading it:** `src_out` and `tgt_in` are consistently the two large, negative,
statistically-significant terms across nearly every dataset and model (`p_fdr` in
`results/node4_per_dataset_long.csv` confirms this — see there for the full stats, not just
the point estimates above). `src_in`/`tgt_out` are close to zero and inconsistent in sign
across datasets for most models — the same "two terms carry the story" pattern found in the
pooled analysis holds up dataset-by-dataset, not just on average. The **intercept is
large and positive everywhere** (5-8 in bits/log-odds units) because most test edges sit in
low-entropy (predictable) neighborhoods — the intercept is the model's baseline log-odds of
being correct when all four entropy terms are 0 (perfectly homogeneous neighborhood on all
four sides), and every model gets most such edges right by default; the negative slopes are
what pull accuracy down as any of the entropy terms rises.

## 5. File manifest

```
README.md
data/
  predictions_raw_canonical_e27.pkl    raw per-edge predictions, all 4 models x 6 datasets
  joined_table.pkl                     predictions + all node entropy features, precomputed
results/
  node4_per_dataset_long.csv           full stats (beta, se_robust, p, p_fdr, n, ...), raw scale
  node4_per_dataset_long_zscored.csv   same, z-scored inputs
  node4_per_dataset_wide.csv           compact: 24 rows x [dataset, model, const, src_out, src_in, tgt_out, tgt_in]
  node4_per_dataset_wide_zscored.csv   same, z-scored inputs -- the tables in section 4 above are the raw wide file
scripts/
  regression_lib.py                    standalone fitting code (no parent-repo dependencies)
  reproduce.py                         driver: rebuilds every results/*.csv from data/joined_table.pkl
```

### 5.1 `data/predictions_raw_canonical_e27.pkl`

`dict[dataset_name][model_name] -> {"u": [...], "v": [...], "y": [...], "p": [...]}` — raw
per-edge test predictions, already restricted so all 4 models within a dataset share the
identical edge set (the intersection of all 4 models' test edges). `u`/`v` are raw node ids
(same id space across all 4 models within a dataset), `y` is the true binary label, `p` is
the model's predicted P(positive).

### 5.2 `data/joined_table.pkl`

`dict[dataset_name][model_name] -> {column_name: numpy array}`, one row per shared test edge.
`regression_lib.node4_feature_columns` reads `ent_out_u`, `ent_in_u`, `ent_out_v`, `ent_in_v`
(and `correct`, `u`, `v`) directly off this table — no recomputation from the raw graph is
needed to reproduce any fit in this package. **Not included** (out of scope for a compact
package): the code and raw graph files needed to *build* `joined_table.pkl` from scratch —
that requires the parent repository's full data pipeline and the original dataset files.
Everything needed to rerun or extend the regression itself is fully contained here.

### 5.3 `results/node4_per_dataset_long*.csv`

One row per (dataset, model, term). Key columns:

| Column | Meaning |
|---|---|
| `dataset` | one of the 6 dataset names (never `POOLED` in this package) |
| `model` | walk_full / walk_localattn4 / GINEConv / SiGAT |
| `term` | `const`, `src_out`, `src_in`, `tgt_out`, or `tgt_in` |
| `beta` | the fitted coefficient (log-odds per unit, or per 1-SD if `_zscored`) |
| `se_robust` | two-way cluster-robust standard error |
| `p`, `p_fdr` | raw and FDR-corrected p-value (FDR applied within each fit's 4 slope terms) |
| `odds_ratio` | `exp(beta)` |
| `n` | number of edges that fit used (after dropping any row with a missing feature) |

95% CI = `beta ± 1.96 * se_robust`.

## 6. Reproducing / extending

```bash
cd scripts/
pip install numpy pandas scipy statsmodels     # only dependencies needed
python reproduce.py --joined-table ../data/joined_table.pkl --out-dir ../results_reproduced
```

Regenerates all four `results/*.csv` files from `joined_table.pkl` — verified to match the
shipped files to floating-point precision. To fit a different term set or a different
dataset subset, call `run_node4_fits`/`run_node4_fits_zscored` in `regression_lib.py`
directly with a different `datasets` list, or use `run_atomic_fits`/`run_srctgt_fits` (also
included in `regression_lib.py`) for the 6-term or 2-term variants of the same idea.
