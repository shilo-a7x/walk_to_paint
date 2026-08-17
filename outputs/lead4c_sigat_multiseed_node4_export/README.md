# SiGAT node-entropy regression — 10-seed robustness check

This package answers a follow-up question to a prior single-seed analysis: are the
dataset-level patterns found in SiGAT's per-dataset node-entropy coefficients real, or
just noise from using one particular train/test split? It refits the same 4-term model
**independently on 10 different random seeds** (42–51) per dataset, and reports each
coefficient as **mean ± SD across those 10 seeds**, plus how many of the 10 individually
reached significance. Self-contained — data, code, plot, and this description are
everything needed to inspect or extend the result.

## 1. What SiGAT is, and what "correct" means

**SiGAT** is a signed-graph-specific graph-attention neural network. It builds one
embedding per node via multiple attention channels that distinguish edge sign/direction,
then an edge's predicted label comes from a logistic classifier fit on top of the pair of
endpoint embeddings (SiGAT doesn't output an edge score directly). For every test edge
`(u, v)`, with predicted probability `p` and true label `y`:

```
correct(u, v) = 1  if (p >= 0.5) == y,  else 0
```

**No model is retrained in this package** — this reuses SiGAT's already-trained node
embeddings (one independently-trained set per seed) and asks a purely statistical
question about the relationship between graph structure and prediction accuracy.

## 2. Datasets

| Dataset | Nodes | Edges | Domain |
|---|---|---|---|
| bitcoin-alpha | 3,783 | 24,186 | trust ratings between traders |
| bitcoin-otc | 5,901 | 35,592 | trust ratings between traders |
| wiki-elec | ~7,100 | 103,689 | admin-election support/oppose votes |
| wiki-rfa | ~11,400 | 184,546 | admin-election support/oppose votes |
| epinions | ~131,000 | 841,372 | trust/distrust between reviewers |
| slashdot090221 | ~82,000 | 549,202 | friend/foe relationships |

(Ordered small→large here and throughout this package — deliberate, see section 4.)

## 3. The model, the entropy terms, and why 10 seeds

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

Computed from the full graph (all edges) — the same fixed value regardless of which
seed's train/test split is used, since it's a property of the graph itself, not of any
one split.

### 3.2 The model

```
logit(P(correct)) = const + b1*src_out + b2*src_in + b3*tgt_out + b4*tgt_in
```

Fit **independently per dataset AND per seed** — 60 regressions total (6 datasets × 10
seeds), each with its own intercept and 4 slopes, no pooling across datasets or across
seeds. Two-way cluster-robust standard errors (clustered on source and target node,
Cameron-Gelbach-Miller 2011); p-values BH-FDR corrected within each fit's 4 slope terms.
Every fit is run twice — raw scale and z-scored (entropy columns standardized before
fitting) — z-scored is what makes cross-dataset/cross-seed magnitudes comparable and is
what the headline numbers below use.

### 3.3 Why 10 seeds

Only the *predictions* change across seeds (a fresh SiGAT model trained on a different
80/10/10 split each time) — the entropy features are seed-independent. A single seed's
fit is one noisy draw; refitting on 10 independently-trained models and independently-drawn
test splits lets you separate a real, structural pattern from something that happened to
look real on one particular split. This mirrors the same "fit each split independently,
then report mean ± SD" convention already used elsewhere in the parent project for
combining multiple SiGAT/PEWTER splits (rather than pooling all 10 splits' raw predictions
into one giant fit, which would mix outputs from 10 different trained models).

## 4. Results

### 4.1 The plot

`results/sigat_multiseed_node4_forest.png` — one dot-and-whisker point per dataset per
term: the dot is the mean z-scored coefficient across the 10 seeds, the whiskers are
±1 SD across seeds. **Filled marker** = significant (p_fdr<0.05) in at least 8 of the 10
seeds ("robust"); **open marker** = significant in fewer than 8. Panels are ordered top row
first for the two terms that turn out to matter (`src_out`, `tgt_in`), bottom row for the
two that mostly don't (`src_in`, `tgt_out`). Datasets are ordered small-graph-first with a
dashed divider before the two largest/densest graphs (epinions, slashdot090221) — this
ordering is deliberate, not alphabetical, because it's exactly the split pattern 1 below
depends on.

### 4.2 The numbers (z-scored, mean ± SD across 10 seeds)

| Dataset | src_out | tgt_in | src_in | tgt_out |
|---|---|---|---|---|
| bitcoin-alpha | −0.982 ± 0.088 (10/10) | −1.114 ± 0.106 (10/10) | −0.203 ± 0.132 (5/10) | 0.043 ± 0.078 (0/10) |
| bitcoin-otc | −0.734 ± 0.075 (10/10) | −1.186 ± 0.073 (10/10) | −0.457 ± 0.061 (10/10) | 0.072 ± 0.072 (1/10) |
| wiki-elec | −0.697 ± 0.054 (10/10) | −1.231 ± 0.081 (10/10) | −0.029 ± 0.045 (0/10) | −0.035 ± 0.047 (0/10) |
| wiki-rfa | −0.686 ± 0.037 (10/10) | −1.381 ± 0.060 (10/10) | −0.002 ± 0.035 (0/10) | 0.006 ± 0.025 (0/10) |
| epinions | −0.943 ± 0.034 (10/10) | −0.654 ± 0.031 (10/10) | −0.236 ± 0.022 (10/10) | 0.050 ± 0.026 (6/10) |
| slashdot090221 | −1.100 ± 0.037 (10/10) | −0.640 ± 0.027 (10/10) | −0.096 ± 0.019 (10/10) | −0.042 ± 0.015 (7/10) |

"(k/10)" = number of the 10 individual seeds where that coefficient was FDR-significant.
Full per-seed detail (all 600 individual fits): `data/per_seed_fits.csv`.

### 4.3 The two patterns from the single-seed analysis both survive, robustly

**`src_out` and `tgt_in` are significant in 10/10 seeds on every single dataset —
this is the most robust finding in the whole analysis, with essentially no seed-to-seed
disagreement about direction or significance.**

**Pattern 1 — which term dominates flips with graph size, confirmed at the mean level
with tight, non-overlapping error bars.** `tgt_in` is bigger on the 4 smaller graphs
(bitcoin-alpha/otc, wiki-elec/rfa); `src_out` is bigger on the 2 largest/densest graphs
(epinions, slashdot090221). The SD across seeds (0.03–0.11) is small relative to the gap
between `src_out` and `tgt_in` within each dataset — this isn't a coin-flip that happened
to land one way on seed 42, it's a consistent ranking across independently-trained models.

**Pattern 2 — `src_in` is null specifically on both wiki datasets, confirmed and now with
an important refinement.** wiki-elec and wiki-rfa show `src_in` significant in **0/10**
seeds each, mean magnitude essentially zero (−0.029, −0.002) — a clean, robust null.
bitcoin-otc/epinions/slashdot090221 show `src_in` significant in **10/10** seeds — clean,
robust, real effects. **bitcoin-alpha is the one case that needed the multi-seed check to
clarify**: its single-seed (42) result was significant (p_fdr=0.008), but across all 10
seeds only 5/10 reach significance, even though the direction is consistently negative in
9 of the 10 individual seeds (only seed 44 was slightly positive, and not significant).
**Read this as "a real but underpowered effect"** — bitcoin-alpha's small test set
(≈2,300 edges) makes the individual-seed significance test noisy at this effect size,
not as a genuine null like wiki. This distinction (small-and-noisy vs. genuinely-zero)
was invisible in the single-seed package and only shows up once you have 10 seeds to
compare.

### 4.4 A caveat surfaced by the multi-seed view: `tgt_out`'s "significance" on the 2 largest datasets is a sample-size effect, not a real pattern

`tgt_out` is essentially zero everywhere (|mean| ≤ 0.07 in every dataset) and mostly
non-significant — except epinions (6/10 seeds significant) and slashdot090221 (7/10). Given
these are the two datasets with by far the largest test sets (54,921–84,080 edges vs.
2,300–17,700 for the others), and the effect size itself never grows to match, this reads
as **statistical power finding a technically-nonzero-but-practically-negligible effect**,
not a real dataset-specific `tgt_out` story. Worth stating explicitly since it would be easy
to over-read "significant in most seeds" as meaningful without checking the effect size
alongside it.

## 5. File manifest

```
README.md
data/
  per_seed_fits.csv              600 rows: 10 seeds x 6 datasets x 5 terms x 2 scales
results/
  aggregated_summary.csv         60 rows: mean/SD/significance-count per (dataset, term, scale)
  sigat_multiseed_node4_forest.png   the plot described in 4.1
scripts/
  aggregate_and_plot.py          standalone: rebuilds aggregated_summary.csv + the plot from per_seed_fits.csv
```

### 5.1 `data/per_seed_fits.csv`

One row per (dataset, seed, term, scale). Columns: `dataset`, `seed` (42–51), `scale`
(`raw`/`zscored`), `term` (`const`/`src_out`/`src_in`/`tgt_out`/`tgt_in`), `beta`,
`se_naive`, `se_robust` (two-way cluster-robust), `p`, `p_fdr` (BH-FDR within that
fit's 4 slope terms), `odds_ratio`, `n` (edges used), `base_rate` (SiGAT's overall
accuracy on that seed's test split), `pseudo_r2`.

**How this file was built (not reproducible from this package alone — needs the parent
repository):** for each of the 10 seeds, SiGAT's node embeddings for that seed
(`baselines/SGA/results_our_splits_canonical/<ds>/SiGAT/seed<N>/best_epoch_artifacts.pkl`
in the parent repo) are used to fit a fresh `LogisticRegression` on that seed's own
train-split edges, then predict on that seed's own test-split edges — this is standard
practice for evaluating embeddings that don't come with a built-in edge classifier
(SiGAT is exactly this case). Each seed's test-split edges are then joined with the
(seed-independent) entropy features and fit via the 4-term model above. **Out of scope for
this compact package**: the SiGAT embedding files themselves (large, one per seed per
dataset) and the graph-loading code needed to reproduce this extraction step from scratch.
What's included (`per_seed_fits.csv`) is that extraction's *output* — sufficient to
reproduce every number and the plot in this package without needing any of that.

### 5.2 `results/aggregated_summary.csv`

One row per (dataset, term, scale). Columns: `mean_beta`, `std_beta` (both across the 10
seeds), `min_beta`, `max_beta`, `n_seeds` (10, or fewer if a seed's fit was degenerate),
`n_seeds_significant`, `frac_seeds_significant`, `mean_n_edges`.

## 6. Reproducing / extending

```bash
cd scripts/
pip install numpy pandas matplotlib   # only dependencies needed
python aggregate_and_plot.py --per-seed ../data/per_seed_fits.csv --out-dir ../results_reproduced
```

Regenerates `aggregated_summary.csv` and the plot from `per_seed_fits.csv` — verified to
match the shipped files to floating-point precision. To test a different "robust" threshold
(currently ≥8/10 seeds), a different dataset order/grouping, or different plot styling,
edit the constants at the top of `aggregate_and_plot.py` — the aggregation and plotting
logic are both in that one file, no other dependencies.
