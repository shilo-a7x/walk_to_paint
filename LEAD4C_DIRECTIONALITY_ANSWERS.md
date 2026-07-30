# Lead 4c directionality — answers to `LEAD4C_DIRECTIONALITY_EXPERIMENTS_NEEDED.md`

**Purpose:** fills in every item requested in `LEAD4C_DIRECTIONALITY_EXPERIMENTS_NEEDED.md` (#1–4;
#5 stays deferred per that doc's own priority call). Nothing here touches
`aaai2027/pewter_aaai.tex`, and nothing here re-derives or re-litigates the architecture-specific
`src_out`-vs-`tgt_in` split (`LEAD4C_ASYMMETRY.md`) — this is only about strengthening Claim 1
("source-side entropy is lower than target-side") into paper-citable numbers.

**Script:** `scripts/lead4c_directionality_answers.py` (`.venv/bin/python
scripts/lead4c_directionality_answers.py --n-boot 2000 --n-mc 5000`, ~90s wall-clock, no
retraining — pure post-hoc analysis over `predictions_raw_canonical.pkl` and
`load_edges_canonical`). Raw outputs: `outputs/lead4c_entropy_logit_regression/directionality_answers/`
(`claim1_table.csv`, `auc_gap_bootstrap.csv`, `regression_results.json`, `meta_analysis.json`).

---

## Item #1 — is `load_edges_canonical` affected by the fabricated-reverse-edge issue?

**No — confirmed clean, and this was already established (just not cross-referenced).**
`FABRICATED_REVERSE_EDGES.md` (line 27–28) and `scripts/lead4_entropy_heterogeneity.py`'s own
module docstring (lines 31–38) both independently state that `load_edges_canonical` reads edges
via the same loader used for training (`get_loader(cfg)`), not `baselines/splits/<ds>.pt`'s dense
edge list — so there's no dense/raw remap and no fabricated-reverse-edge mirrors to filter.
Claim 1's entropy numbers (computed from this exact loader) were never affected. No new code was
needed; this is a documentation cross-reference, not an open risk.

## Item #2 — bootstrap CI on each dataset's walk-vs-GNN AUC gap

**Method:** on the shared (walk-covered) edge set per dataset — the same restriction Leads 4/4b/4c
use throughout — resampled edges with replacement (B=2000), recomputing `best(walk_full,
walk_localattn4) − best(GINEConv, SiGAT)` AUC each time. **Scope note:** this uses only the two
GNN baselines with cached per-edge predictions (`predictions_raw_canonical.pkl`); on
bitcoin-alpha/bitcoin-otc the SOTA table's actual best-GNN is SGA-GSGNN/SNEA (no per-edge cache
exists for those), so this gap is a defined, reproducible, *lower-bound-ish* proxy of the SOTA-table
gap, not a literal bootstrap of it — consistent with the already-documented fact that "apples-to-
apples (shared edges) best GNN is always lower still" (`CLAUDE.md` SOTA section).

| Dataset | point gap | 95% bootstrap CI | walk model | GNN model |
|---|---|---|---|---|
| bitcoin-alpha | +0.0532 | [+0.0230, +0.0798] | walk_localattn4 | SiGAT |
| bitcoin-otc | +0.0572 | [+0.0312, +0.0777] | walk_full | GINEConv |
| epinions | +0.0421 | [+0.0386, +0.0459] | walk_localattn4 | SiGAT |
| slashdot090221 | +0.0419 | [+0.0373, +0.0465] | walk_full | SiGAT |
| wiki-elec | +0.0106 | [+0.0009, +0.0211] | walk_localattn4 | SiGAT |
| wiki-rfa | +0.0100 | [+0.0027, +0.0183] | walk_full | SiGAT |

**All 6 CIs exclude 0** — even the two small-gap wiki datasets have a real (if narrow) walk
advantage on this proxy metric, not noise. **The ranking used for item #3/#4 is stable**: the two
CIs for wiki-elec/wiki-rfa ([+0.001,+0.021], [+0.003,+0.018]) sit clearly below and do not overlap
the CIs for the other 4 datasets ([+0.023,+0.080], [+0.031,+0.078], [+0.039,+0.046], [+0.037,+0.047])
— the large-gap/small-gap split that drives the correlation story is not a single-seed artifact.

## Item #3 — continuous effect-size regression (replacing "confirmed/reversed" categorical split)

**x** = rank-biserial correlation of the Claim 1 Wilcoxon test per dataset (+1 = fully confirms
H_out<H_in, −1 = fully reversed; computed directly from signed ranks, the proper effect size for a
paired Wilcoxon test — not just `frac−0.5`). **y** = the item #2 point-estimate AUC gap.

| Dataset | rank-biserial r | AUC gap |
|---|---|---|
| bitcoin-alpha | +0.146 | +0.0532 |
| bitcoin-otc | +0.224 | +0.0572 |
| epinions | +0.287 | +0.0421 |
| slashdot090221 | +0.275 | +0.0419 |
| wiki-elec | −0.206 | +0.0106 |
| wiki-rfa | −0.183 | +0.0100 |

**OLS (point estimates, n=6):** slope = 0.081 (SE 0.021, t=3.80, **p=0.019**), intercept = 0.029,
**R²=0.78**. Pearson r = **0.885** (p=0.019). Spearman ρ = 0.486 (p=0.33, **not significant** — see
caveat below).

**Monte-Carlo-propagated CI** (redraws each dataset's gap from its own item #2 bootstrap
distribution 5000 times and refits — the honest CI, reflecting both the n=6 cross-dataset spread
*and* each dataset's own AUC-gap estimation noise, not just the point estimates):
slope 95% CI = **[0.059, 0.099]**, Pearson r 95% CI = **[0.684, 0.976]**. **Both exclude zero.**

**This replaces the earlier "n=6, too small for a real p-value" framing with a defensible number.**
Caveat to carry into the paper: the **linear (Pearson) relationship is much stronger than the rank
(Spearman) relationship** — the effect looks less like a smooth monotonic trend across all 6 ranks
and more like two clusters (the 4 "confirmed" datasets vs. the 2 "reversed" wiki datasets) separated
by a large gap in both x and y, with a real but weaker ordering *within* each cluster. Report the
Pearson number with this caveat stated explicitly, not silently.

## Item #4 — pooled/meta-analytic estimate of Claim 1 across all 6 datasets

DerSimonian-Laird random-effects meta-analysis on the logit-transformed `frac(H_out<H_in)` per
dataset (standard technique for combining several proportion effect sizes; weights = inverse
variance + between-study variance τ²).

**All 6 datasets pooled:**

| Quantity | Value |
|---|---|
| Random-effects pooled proportion | **0.521** |
| 95% CI | **[0.461, 0.580]** — **includes the null (0.5)** |
| Heterogeneity Q (df=5) | 528.9, p=4.6e-112 |
| I² | **99.1%** |

**Restricted to the 4 datasets where Claim 1 held (bitcoin-alpha, bitcoin-otc, epinions,
slashdot090221):**

| Quantity | Value |
|---|---|
| Random-effects pooled proportion | **0.581** |
| 95% CI | **[0.553, 0.608]** — **excludes the null (0.5)** |
| Heterogeneity Q (df=3) | 58.4, p=1.3e-12 |
| I² | 94.9% |

**This is the single most important honest number to bring back to the paper.** A naive pooled
claim across all 6 datasets ("in general, source-side entropy is lower") is **not statistically
defensible** — the random-effects CI spans 0.5, and I²=99% says the 6 datasets are not
exchangeable/interchangeable measurements of one shared effect; the wiki reversal is not noise,
it's structural. **Restricted to the 4 trust/rating-style graphs, the pooled effect is real, robust,
and significant** (58.1% of non-tied node pairs favor the source side, tight CI, excludes 0.5) —
even there, I²=94.9% says real dataset-to-dataset variation remains (expected, given epinions/
slashdot are far more lopsided than bitcoin-alpha/otc per the original per-dataset table). Neither
subgroup is homogeneous internally, but the 4-vs-2 split itself is not arbitrary — it's exactly the
split the item #3 regression (and `LEAD4C_ASYMMETRY.md`'s SOTA-gap correlation) already flags.

---

## Item #5 — multi-seed stability

**Deferred**, per the request doc's own priority call: the entropy side (Claim 1) is already
extremely significant per-dataset and doesn't need it; the AUC-gap side is now covered by item #2's
bootstrap, which is a reasonable proxy for the seed/sampling-stage variance this item worried about.
Only worth revisiting if a reviewer specifically pushes on frozen-seed=42 robustness.

---

## What goes into the paper

Two citable numbers now exist for `pewter_aaai.tex`'s open markers, replacing "confirmed on 4/6" /
"n=6, too small for a real p-value":

1. **For the abstract's `XXXXXXXX Shilo which is it more source or end?` marker:** *"Restricted to
   the 4 datasets where the direction holds, pooled effect = 58.1% of non-tied node pairs favor
   lower source-side entropy (95% CI [55.3%, 60.8%], random-effects meta-analysis, k=4). Pooled
   across all 6 datasets the direction is not uniform (I²=99%, 95% CI [46.1%, 58.0%] spans the
   null) — the two reversed datasets (wiki-elec, wiki-rfa) are vote/election graphs, not trust
   graphs, and the reversal is itself significant there (p=1.7e-10, p=3.4e-14, `LEAD4C_ASYMMETRY.md`)."*
   This is more honest than a single blended "most networks" claim and gives the paper a concrete,
   defensible number either way.

2. **For "the gain is largest exactly where source and target entropy [asymmetry] are high":**
   *"Pearson r=0.885 (p=0.019, n=6) between the per-dataset Claim-1 effect size (rank-biserial r)
   and the walk-vs-GNN AUC gap; Monte-Carlo-propagated 95% CI [0.684, 0.976] accounting for AUC-gap
   sampling uncertainty, excludes zero."* Caveat for the writeup: this is a strong linear
   relationship across two clusters (4 confirmed + 2 reversed datasets) rather than a smooth
   monotonic trend (Spearman ρ=0.49, n.s.) — say so rather than imply a single continuous dose-
   response curve from n=6 points.

Both numbers, their source script (`scripts/lead4c_directionality_answers.py`), and the raw output
files are cited above so they can be re-confirmed for freshness before going into the text, per the
paper's standing freshness-discipline rule.
