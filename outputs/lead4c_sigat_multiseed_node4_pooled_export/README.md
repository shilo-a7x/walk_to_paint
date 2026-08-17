# SiGAT node-entropy regression — pooled across all 6 datasets, 10-seed robustness check

This package fits **one shared regression across all 6 datasets at once** (not a separate
fit per dataset), refit independently on **10 different random seeds** (42–51), and reports
each coefficient as mean ± SD across those 10 seeds. It's the pooled companion to a
per-dataset version of the same idea — see section 4.4 for how the two relate. Self-contained
— data, code, plot, and this description are everything needed to inspect or extend the
result.

## 1. What SiGAT is, and what "correct" means

**SiGAT** is a signed-graph-specific graph-attention neural network. It builds one embedding
per node via multiple attention channels that distinguish edge sign/direction, then an
edge's predicted label comes from a logistic classifier fit on top of the pair of endpoint
embeddings (SiGAT doesn't output an edge score directly). For every test edge `(u, v)`, with
predicted probability `p` and true label `y`:

```
correct(u, v) = 1  if (p >= 0.5) == y,  else 0
```

**No model is retrained in this package** — this reuses SiGAT's already-trained node
embeddings (one independently-trained set per seed) and asks a purely statistical question
about the relationship between graph structure and prediction accuracy.

## 2. Datasets

| Dataset | Nodes | Edges | Domain |
|---|---|---|---|
| bitcoin-alpha | 3,783 | 24,186 | trust ratings between traders |
| bitcoin-otc | 5,901 | 35,592 | trust ratings between traders |
| wiki-elec | ~7,100 | 103,689 | admin-election support/oppose votes |
| wiki-rfa | ~11,400 | 184,546 | admin-election support/oppose votes |
| epinions | ~131,000 | 841,372 | trust/distrust between reviewers |
| slashdot090221 | ~82,000 | 549,202 | friend/foe relationships |

Every seed's pooled fit uses **all 6 datasets' test edges at once**, ~173,072 edges total
per seed.

## 3. The model, the entropy terms, and the pooling setup

### 3.1 Entropy features

For a node `n`, the **binary Shannon entropy of its sign distribution** in a given
direction: `p` = fraction of `n`'s edges (in that direction) that are positive,
`H = -p·log2(p) - (1-p)·log2(1-p)` (0 if `p ∈ {0,1}`). `H=0` = perfectly predictable,
`H=1` = maximally unpredictable (50/50). Four node-level directional terms:

| Term | Definition | Reads as |
|---|---|---|
| `src_out` | H of **u's outgoing** edge signs | how consistently does u rate others? |
| `src_in` | H of **u's incoming** edge signs | how consistently is u itself rated? |
| `tgt_out` | H of **v's outgoing** edge signs | how consistently does v rate others? |
| `tgt_in` | H of **v's incoming** edge signs | how contested is v's reputation already? |

Computed from the full graph (all edges) — the same fixed value regardless of which seed's
train/test split is used, since it's a property of the graph itself, not of any one split.

### 3.2 The pooled model

```
logit(P(correct)) = b1*src_out + b2*src_in + b3*tgt_out + b4*tgt_in
                    + [ds_bitcoin-alpha] + [ds_bitcoin-otc] + [ds_epinions]
                    + [ds_wiki-elec] + [ds_wiki-rfa] + [ds_slashdot090221]
```

Edges from all 6 datasets are stacked into one regression, with a **full set of dataset
dummy variables** (one per dataset, no dropped reference category, no separate global
intercept) — each dummy's own coefficient IS that dataset's own intercept directly. The 4
entropy slopes (`src_out`/`src_in`/`tgt_out`/`tgt_in`) are **shared** across all 6 datasets —
one number per term describing the average effect across the whole pooled edge set, rather
than letting each dataset have its own slope. Node ids are offset per dataset before
stacking so a raw id collision across two different datasets' node-numbering can't
accidentally merge two unrelated nodes into one cluster for the standard-error calculation.

This is fit **independently per seed** — 10 pooled regressions total, each on that seed's
own 6 test splits (~173,000 edges). Two-way cluster-robust standard errors (clustered on
source and target node); p-values BH-FDR corrected within each fit's 4 slope terms (the 6
dataset intercepts are not part of that correction — they're not the object of the
significance question here). Every fit is run twice — raw scale and z-scored (entropy
columns standardized on the pooled data before fitting) — z-scored is what the headline
numbers below use.

### 3.3 Why 10 seeds

Only the *predictions* change across seeds (a fresh SiGAT model trained on a different
80/10/10 split each time, independently per dataset) — the entropy features are
seed-independent. A single seed's pooled fit is one noisy draw; refitting on 10
independently-trained models lets you see how stable the pooled slope actually is, following
the same "fit each split independently, then mean ± SD" convention used elsewhere in the
parent project for combining multiple SiGAT/PEWTER splits.

## 4. Results

### 4.1 The plot

`results/sigat_multiseed_node4_pooled_forest.png` — two panels. **Left**: the 4 shared
slopes, one dot per term, mean z-scored coefficient across the 10 seeds with ±1 SD whiskers
— this is the headline. **Filled marker** = significant (p_fdr<0.05) in at least 8 of the 10
seeds; **open marker** = fewer. **Right**: the 6 per-dataset intercepts (mean ± SD across
seeds) — shown for context/completeness, not the main result; it just says how much SiGAT's
baseline (zero-entropy) accuracy differs by dataset.

### 4.2 The numbers (z-scored, mean ± SD across 10 seeds, n≈173,072 edges pooled)

| Term | mean β | SD across seeds | significant in |
|---|---|---|---|
| `src_out` | **−1.003** | 0.025 | 10/10 seeds |
| `tgt_in` | **−0.819** | 0.019 | 10/10 seeds |
| `src_in` | −0.140 | 0.013 | 10/10 seeds |
| `tgt_out` | +0.013 | 0.012 | 2/10 seeds |

Per-dataset intercepts (context): bitcoin-alpha 2.538 ± 0.078, bitcoin-otc 2.544 ± 0.066,
epinions 2.975 ± 0.051, slashdot090221 2.956 ± 0.065, wiki-elec 3.215 ± 0.042, wiki-rfa
3.246 ± 0.055.

Full per-seed detail (all 200 individual fitted coefficients): `data/per_seed_pooled_fits.csv`.

### 4.3 Reading it

**`src_out` and `tgt_in` are large, negative, and significant in all 10 seeds — by a wide
margin the two terms that matter, exactly matching the per-dataset analysis's headline.**
`src_in` is small but real and tightly estimated (SD only 0.013 on a huge pooled sample) —
significant in all 10 seeds despite a modest effect size. `tgt_out` stays indistinguishable
from zero (mean 0.013, essentially the width of its own SD), significant in only 2/10 seeds
— noise, same conclusion as everywhere else this has been checked.

### 4.4 Why this pooled number looks different from simply "averaging" the per-dataset numbers — and why that's expected, not a bug

If you look at the companion per-dataset package (`lead4c_sigat_multiseed_node4_export`),
`src_in` came out as a **clean null specifically on wiki-elec/wiki-rfa** (mean ≈ 0, 0/10
seeds significant) while being real and significant on the other 4 datasets. Here, pooled,
`src_in` comes out small-but-significant (−0.140) — that's **not a contradiction**. A pooled
fit with a *shared* slope doesn't average the 6 per-dataset slopes evenly; it's dominated by
whichever datasets contribute the most edges. epinions and slashdot090221 alone contribute
~125,000 of the ~173,000 pooled edges (more than 70%), and both show a real, robust `src_in`
effect per-dataset — so the pooled number mostly reflects those two large datasets, with the
wiki-elec/wiki-rfa null diluted rather than cancelling it out. **This is exactly the kind of
thing a per-dataset fit is needed to catch and a pooled fit can hide** — if you only had this
package, you'd conclude `src_in` matters a little bit everywhere; the per-dataset package
shows that's not true, it matters a lot on 4 datasets and not at all on 2, and the pooled
number is just an edge-count-weighted blend of that. Read the two packages together, not as
alternatives.

## 5. File manifest

```
README.md
data/
  per_seed_pooled_fits.csv       200 rows: 10 seeds x 2 scales x (4 shared slopes + 6 dataset intercepts)
results/
  aggregated_summary.csv         20 rows: mean/SD/significance-count per (term, scale)
  sigat_multiseed_node4_pooled_forest.png   the plot described in 4.1
scripts/
  aggregate_and_plot.py          standalone: rebuilds aggregated_summary.csv + the plot from per_seed_pooled_fits.csv
```

### 5.1 `data/per_seed_pooled_fits.csv`

One row per (seed, term, scale). Columns: `seed` (42–51), `scale` (`raw`/`zscored`), `term`
(`src_out`/`src_in`/`tgt_out`/`tgt_in`, or `ds_<dataset>` for that seed's per-dataset
intercept), `beta`, `se_naive`, `se_robust` (two-way cluster-robust), `p`, `p_fdr` (BH-FDR
within that seed's 4 slope terms only — not applied to the intercepts), `odds_ratio`,
`n` (edges used, ~173,072 for every row), `base_rate`, `pseudo_r2`.

**How this file was built (not reproducible from this package alone — needs the parent
repository):** for each of the 10 seeds, SiGAT's node embeddings for that seed
(`baselines/SGA/results_our_splits_canonical/<ds>/SiGAT/seed<N>/best_epoch_artifacts.pkl` in
the parent repo, one file per dataset per seed) are used to fit a fresh `LogisticRegression`
per dataset on that seed's own train-split edges, then predict on that seed's own test-split
edges. All 6 datasets' test-split predictions for that seed are then stacked (with the
dataset-dummy encoding described in 3.2) and joined with the seed-independent entropy
features. **Out of scope for this compact package**: the SiGAT embedding files themselves
(large, one per seed per dataset) and the graph-loading code needed to reproduce this
extraction step from scratch. What's included (`per_seed_pooled_fits.csv`) is that
extraction's *output* — sufficient to reproduce every number and the plot in this package
without needing any of that.

### 5.2 `results/aggregated_summary.csv`

One row per (term, scale). Columns: `is_dataset_intercept` (True for the 6 `ds_*` terms,
False for the 4 shared slopes), `mean_beta`, `std_beta` (both across the 10 seeds),
`min_beta`, `max_beta`, `n_seeds`, `n_seeds_significant` (slopes only), `frac_seeds_significant`.

## 6. Reproducing / extending

```bash
cd scripts/
pip install numpy pandas matplotlib   # only dependencies needed
python aggregate_and_plot.py --per-seed ../data/per_seed_pooled_fits.csv --out-dir ../results_reproduced
```

Regenerates `aggregated_summary.csv` and the plot from `per_seed_pooled_fits.csv` — verified
to match the shipped files to floating-point precision. To test a different "robust"
threshold (currently ≥8/10 seeds) or different plot styling, edit the constants at the top
of `aggregate_and_plot.py` — the aggregation and plotting logic are both in that one file,
no other dependencies.
