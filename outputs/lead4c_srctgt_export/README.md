# Source-out / target-in entropy asymmetry — standalone regression package

This package answers one question: **does a node's local sign-entropy hurt prediction
accuracy differently depending on its *role* in the edge (source vs. target, outgoing vs.
incoming), and does that hurt differently for a walk-based Transformer than for
message-passing GNN baselines?** It is self-contained — the data, the fitting code, and
this description are everything needed to inspect, verify, or extend the analysis without
the parent research repository.

## 1. Background — what problem the models solve

The task is **edge sign prediction in directed signed graphs**: each edge `u -> v` carries a
label (e.g. "trust"/"distrust", "+"/"-"), and the model predicts the sign of held-out edges
from the graph structure and the signs of observed edges. Six public directed signed-graph
datasets are used throughout:

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
| **walk_localattn4** | Same walk-Transformer, restricted attention | Identical to walk_full except attention is masked to a ±2-hop window around each position (a "local attention" variant) instead of attending across the whole walk. |
| **GINEConv** | Message-passing GNN | A standard edge-feature-aware graph neural network. Builds one embedding per node by aggregating (summing) messages from its neighbors, then reads the edge label off the pair of endpoint embeddings. |
| **SiGAT** | Graph-attention signed GNN | A signed-graph-specific attention-based GNN (multiple parallel attention channels distinguishing edge sign/direction), also read out via endpoint embeddings. |

`walk_full`/`walk_localattn4` are two attention variants of the same underlying architecture;
`GINEConv`/`SiGAT` are two different, independently-trained GNN baselines. All four are
evaluated on the identical shared (intersection) test-edge set per dataset, using each
model's own trained predictions — **no models are retrained or modified in this package**;
this is a pure post-hoc statistical analysis of already-computed predictions.

**Provenance of the predictions used here:** the walk models are the current production
checkpoints (edge-anchored coverage-guaranteed random-walk sampler, per-dataset walk budget
tuned by a prior sweep; `walk_localattn4` uses a ±2-hop attention window, `walk_full` uses
unrestricted attention, otherwise identical training recipe). GINEConv/SiGAT are the
standard baseline checkpoints, trained independently and never modified for this analysis.

## 3. What "correct" means, and what the regression does

For every test edge `(u, v)` and every model, we have a predicted probability `p` that the
edge is positive, and the true label `y`. Define:

```
correct(u, v) = 1  if (p >= 0.5) == y,  else 0
```

i.e. a per-edge, per-model binary indicator of whether that model's hard prediction was
right. **The regression asks: how does `correct` vary with the local sign-entropy around
that edge's endpoints?**

### 3.1 Entropy features (the "coefficients")

For a node `n`, define the **binary Shannon entropy of its sign distribution** in a given
direction:

```
p = fraction of n's edges (in that direction) that are positive
H(n) = -p*log2(p) - (1-p)*log2(1-p)          (H = 0 if p in {0, 1})
```

`H = 0` means the node's edges in that direction are all one sign (perfectly predictable /
homogeneous); `H = 1` means a 50/50 split (maximally unpredictable / heterogeneous). For an
edge `(u, v)` we compute this in four directions, giving **six atomic entropy terms** (two
are 2-hop path-consistency variants, not simple node entropy):

| Term | Definition | Reads as |
|---|---|---|
| `src_out` | H of **u's outgoing** edge signs | how consistently does u rate others? |
| `src_in` | H of **u's incoming** edge signs | how consistently is u itself rated? |
| `tgt_out` | H of **v's outgoing** edge signs | how consistently does v rate others? |
| `tgt_in` | H of **v's incoming** edge signs | how contested is v's reputation already? |
| `twohop_in` | H of 2-hop path-sign-consistency ending at u (`s -> t -> u`) | is the path leading into u predictable? |
| `twohop_out` | H of 2-hop path-sign-consistency starting at v (`v -> m -> k`) | is the path leading out of v predictable? |

All six are computed from the full graph (all edges, not just train), since this is a
read-only diagnostic of already-trained models' predictions — there is no leakage risk here
(nothing is fed back into training).

### 3.2 Three model families fit in this package

**Full 6-term ("atomic") model** — all six terms entered together in one logistic regression,
so each gets a partial coefficient controlling for the other five:

```
logit(P(correct)) = b0 + b1*src_out + b2*src_in + b3*tgt_out + b4*tgt_in
                        + b5*twohop_in + b6*twohop_out   [+ per-dataset intercepts, pooled fits]
```

**4-term node-only ("node4") model** — drops the two 2-hop/path terms entirely, keeping all
four node-level directional entropies:

```
logit(P(correct)) = b0 + b1*src_out + b2*src_in + b3*tgt_out + b4*tgt_in   [+ per-dataset intercepts, pooled fits]
```

**Minimal 2-term ("srctgt2") model** — drops the two node terms that also came out
~null/negligible (`src_in`, `tgt_out`), keeping only the two that carry the story:

```
logit(P(correct)) = b0 + b1*src_out + b2*tgt_in   [+ per-dataset intercepts, pooled fits]
```

All three are fit **separately for each of the 4 models** (so you get 4 independent sets of
coefficients, one per model, not a joint model with a model term) — the interesting result is
*comparing* the fitted `src_out`/`tgt_in` coefficients across the 4 models, not a single joint
fit.

### 3.3 Per-dataset vs. pooled fits

Each model family is fit two ways:
- **Per-dataset**: one regression per (dataset, model) pair — 6 datasets x 4 models = 24 fits.
- **Pooled**: all 6 datasets' edges stacked together in one regression per model, with a
  **full set of dataset dummy variables** (no dropped reference category, no global
  intercept) so each dummy's coefficient IS that dataset's own intercept directly. The
  `src_out`/`tgt_in` slope is *shared* across all pooled datasets — one number describing the
  average effect across all of them.

### 3.4 Z-scored variant

Every fit is run twice: once on the raw entropy values (bits), and once with every entropy
column standardized (mean 0, standard deviation 1) **before** fitting — an actual
re-fit on standardized inputs, not a post-hoc rescaling of the raw beta. Z-scored
coefficients are directly comparable in magnitude across terms and across models (a raw beta
depends on that term's natural scale, which differs by dataset and by direction).

### 3.5 Standard errors and significance

Because many edges share an endpoint (the same node `u` or `v` appears in many test edges),
edges are not independent draws — naive standard errors understate uncertainty. Every fit
uses **two-way cluster-robust standard errors** (clustered on both the source node and the
target node, Cameron-Gelbach-Miller 2011 formula: `V = V_u + V_v - V_(u,v)`), and p-values
across each family of simultaneous terms are corrected for multiple comparisons via
**Benjamini-Hochberg FDR** (column `p_fdr`).

## 4. What's new in this package vs. the parent project's prior work

This package intentionally does **not** carry forward the parent project's full research
history (prior hypotheses tested and ruled out, earlier model checkpoints, etc.) — just this
self-contained analysis, run fresh, on the **current production models**:

1. **Full 6-term model, all 6 datasets** (per-dataset + pooled) — re-run on the current
   production walk checkpoints (previously this had only been run on an older, much
   walk-budget-heavier training configuration; confirms the result is not an artifact of that
   older setup).
2. **4-term node-only model (`src_out`+`src_in`+`tgt_out`+`tgt_in`, no 2-hop terms), all 6
   datasets** — new; the middle ground between the full and minimal models, dropping only the
   two path/2-hop terms.
3. **Minimal 2-term model (`src_out` + `tgt_in` only), all 6 datasets** — new; tests whether
   the story survives with only the two terms that matter, dropping the other four entirely.
4. **Full 6-term model, wiki-elec + wiki-rfa only** — new; these two datasets behave
   differently from the other four on a related (but distinct) raw-data property, so this
   checks whether the model-sensitivity asymmetry itself also looks different when restricted
   to just these two.
5. **4-term node-only model, wiki-elec + wiki-rfa only** — new; same restriction, node-only model.
6. **Minimal 2-term model, wiki-elec + wiki-rfa only** — new; same restriction, minimal model.

## 5. Headline results (z-scored, pooled)

**Reading the sign:** more negative = a 1-SD increase in that entropy term hurts that model's
accuracy more. Compare the four models' coefficients on the *same* term to see which model is
more sensitive to it.

### 5.1 All 6 datasets pooled

| Term | walk_full | walk_localattn4 | GINEConv | SiGAT |
|---|---|---|---|---|
| **Full 6-term model** | | | | |
| src_out | -1.013 | -1.021 | -0.499 | -1.013 |
| tgt_in | -0.688 | -0.748 | **-1.148** | -0.826 |
| **4-term node-only model** | | | | |
| src_out | -1.012 | -1.018 | -0.494 | -1.009 |
| tgt_in | -0.692 | -0.751 | **-1.155** | -0.823 |
| **Minimal 2-term model** | | | | |
| src_out | -0.770 | -0.764 | -0.546 | -0.904 |
| tgt_in | -0.617 | -0.653 | **-1.057** | -0.735 |

### 5.2 wiki-elec + wiki-rfa only, pooled

| Term | walk_full | walk_localattn4 | GINEConv | SiGAT |
|---|---|---|---|---|
| **Full 6-term model** | | | | |
| src_out | -0.561 | -0.631 | -0.440 | -0.636 |
| tgt_in | -1.389 | -1.382 | **-1.692** | -1.392 |
| **4-term node-only model** | | | | |
| src_out | -0.561 | -0.631 | -0.439 | -0.637 |
| tgt_in | -1.382 | -1.376 | **-1.686** | -1.390 |
| **Minimal 2-term model** | | | | |
| src_out | -0.457 | -0.527 | -0.342 | -0.557 |
| tgt_in | -1.188 | -1.210 | **-1.596** | -1.261 |

**Reading across the three tables:** in every version (full/node4/minimal, all-6/wiki-only),
the walk models are consistently more negative than GINEConv on `src_out`, and GINEConv is
consistently the most negative on `tgt_in`. The full and node-only (4-term) models are nearly
identical to each other — the two dropped 2-hop terms barely move `src_out`/`tgt_in` at all,
confirming those two terms carry essentially the whole story on their own. The gap sizes shift
somewhat between the all-6 and wiki-only scopes (narrower on `src_out`, wider on `tgt_in` for
wiki) and between the node-level and minimal specifications, but **the ranking itself does not
flip** in any subset or model specification — this is a stable, model-family-level asymmetry,
not an artifact of one particular term set or one particular dataset mix.

## 6. File manifest

```
README.md                              this file
data/
  predictions_raw_canonical_e27.pkl    raw per-edge predictions, all 4 models x 6 datasets
  joined_table.pkl                     predictions + all 6 entropy features, precomputed
results/
  fit_results_all6.csv                 all-6-dataset fits (both model families, raw+z-scored)
  fit_results_wiki_only.csv            wiki-elec+wiki-rfa-only fits (both families, raw+z-scored)
scripts/
  regression_lib.py                    standalone fitting code (no parent-repo dependencies)
  reproduce.py                         driver: rebuilds both results/*.csv from data/joined_table.pkl
```

### 6.1 `data/predictions_raw_canonical_e27.pkl`

`dict[dataset_name][model_name] -> {"u": [...], "v": [...], "y": [...], "p": [...]}` — raw
per-edge test predictions, already restricted so all 4 models within a dataset share the
identical edge set (the intersection of all 4 models' test edges). `u`/`v` are raw node ids
(same id space across all 4 models within a dataset), `y` is the true binary label, `p` is
the model's predicted P(positive).

### 6.2 `data/joined_table.pkl`

`dict[dataset_name][model_name] -> {column_name: numpy array}`, one row per shared test edge,
same length/order as the corresponding `u`/`v` arrays above. Key columns:

| Column | Meaning |
|---|---|
| `u`, `v`, `y`, `p` | same as the raw predictions above |
| `correct` | `int((p >= 0.5) == y)` — the regression's response variable |
| `ent_out_u`, `ent_in_u`, `ent_inout_u` | u's out/in/both-direction sign entropy |
| `ent_out_v`, `ent_in_v`, `ent_inout_v` | v's out/in/both-direction sign entropy |
| `th_out_total_v`, `th_out_consistent_v` | raw counts behind v's 2-hop-out path entropy |
| `th_in_total_u`, `th_in_consistent_u` | raw counts behind u's 2-hop-in path entropy |
| `deg_out_u`, `deg_in_v` | raw out-degree(u), in-degree(v) (not used by the fits in this package) |

`regression_lib.atomic_feature_columns`/`srctgt_feature_columns` read directly off these
columns — no recomputation from the raw graph is needed to reproduce any fit in this package.
**Not included** (out of scope for a compact package): the code and raw graph files needed to
*build* `joined_table.pkl` from scratch — that requires the parent repository's full data
pipeline and the original dataset files. Everything needed to rerun or extend the regression
itself, however, is fully contained here.

### 6.3 `results/fit_results_*.csv`

One row per (dataset-or-POOLED, model, term, spec). Key columns:

| Column | Meaning |
|---|---|
| `spec` | `atomic` / `atomic_zscored` (6-term) / `node4` / `node4_zscored` (4-term, no 2-hop) / `srctgt2` / `srctgt2_zscored` (2-term) |
| `dataset` | dataset name, or `POOLED` |
| `model` | walk_full / walk_localattn4 / GINEConv / SiGAT |
| `term` | the coefficient name (`src_out`, `tgt_in`, ..., or `ds_<dataset>` for a pooled fit's per-dataset intercept) |
| `beta` | the fitted coefficient (log-odds per unit, or per 1-SD if `_zscored`) |
| `se_robust` | two-way cluster-robust standard error |
| `p`, `p_fdr` | raw and FDR-corrected p-value |
| `odds_ratio` | `exp(beta)` |
| `n` | number of edges the fit actually used (after dropping any row with a missing feature) |
| `pooled`, `pooled_spec` | whether this is a pooled fit, and its type (`shared_slope`) |

95% CI = `beta ± 1.96 * se_robust`.

## 7. Reproducing / extending

```bash
cd scripts/
pip install numpy pandas scipy statsmodels     # only dependencies needed
python reproduce.py --joined-table ../data/joined_table.pkl --out-dir ../results_reproduced
```

This regenerates both `fit_results_all6.csv` and `fit_results_wiki_only.csv` from
`joined_table.pkl` — verified to match the shipped `results/` files to floating-point
precision (max coefficient difference ~1e-16, i.e. exact). To test a different dataset
subset, a different term set, or a different z-scoring choice, edit `reproduce.py` or call
the functions in `regression_lib.py` directly (`run_atomic_fits`, `run_atomic_fits_zscored`,
`run_srctgt_fits`, `run_srctgt_fits_zscored` — each takes `(joined_table_dict, list_of_dataset_names)`).
