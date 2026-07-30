# Lead 4c — handoff note (logistic regression formulation + where everything lives)

Written 2026-06-29 to checkpoint a messy multi-turn session before handing off
to a fresh one. This file is the thing to read first; everything else below
is a pointer, not a duplicate.

## What this is

A follow-up to Lead 4 / 4b (`RESEARCH_LEADS_SUMMARY.md`, `CANONICAL_RERUN_FINDINGS.md`
§2.1). Lead 4 binned sign-entropy into buckets and compared per-bucket AUC.
Lead 4c instead fits one continuous logistic regression per
(dataset-or-pooled, model, node-entropy variant, two-hop variant):

```
logit(P(correct_i)) = b0 + b_src*H_src(u_i) + b_tgt*H_tgt(v_i) + b_2hop*H_2hop(edge_i)  [+ dataset FE for pooled]
```

- `correct_i` — whether model's predicted sign matches ground truth for edge i = (u→v).
- `H_src(u)`, `H_tgt(v)` — binary Shannon entropy of the sign distribution around node
  u / v. 4 variants (which edge direction feeds each side): `out_out`, `in_in`,
  `out_in`, `inout_inout` (first half of name = src direction, second = tgt direction).
  These are Lead 4's existing 4 entropy definitions, kept as TWO separate regression
  terms instead of collapsed into one "node entropy" feature — collapsing would average
  away the u/v asymmetry that is the actual finding (see below).
- `H_2hop(edge)` — Lead 4b's two-hop path-consistency entropy. 3 variants: `out`
  (forward from target v), `in` (backward into source u), `inout` (pooled counts from
  both before taking entropy).
- Fit with `statsmodels.api.Logit`, two-way cluster-robust SEs clustered on u and v
  (Cameron-Gelbach-Miller 2011: `V = V_u + V_v - V_{u,v}`) — edges sharing an endpoint
  aren't independent draws, naive SEs understated uncertainty 1.1–2.2x empirically.
- Pooled fits (all 6 datasets at once) use one-hot dataset fixed effects on the
  intercept; two specs exist — `shared_slope` (one b per term across all datasets) and
  `interacted` (b allowed to vary per dataset via b×dummy terms) — to test slope
  homogeneity. The compact final figure uses `shared_slope`.
- BH-FDR correction applied across all fits (336+ fits total) before calling anything
  significant.

Same ground truth as Lead 4/4b: `predictions_raw_canonical.pkl`, restricted to the
shared (E15 full-coverage) edge set via `load_shared_predictions` — same edges, same 4
models (walk_full, walk_localattn4, GINEConv, SiGAT) — see `lead4-retracted-canonical-split`
memory / `outputs/lead4_entropy_heterogeneity/E15_FULLCOVERAGE_RERUN_NOTE.md`.

## Headline finding — UPDATED 2026-06-29 (atomic decomposition supersedes the v1 framing)

**Consolidated, current writeup: [`LEAD4_ENTROPY_REPORT.md`](LEAD4_ENTROPY_REPORT.md)
(parts A/B/C in one file). Equations: [`LEAD4C_EQUATIONS.md`](LEAD4C_EQUATIONS.md).**

**v1 framing (now corrected):** "it's specifically the target node's IN-edge entropy;
`b_src` (source) is weak/non-significant everywhere." The "source is weak" half was an
**over-generalization from the `in_in` headline combo** (whose source term uses *in*-edges,
which genuinely is ≈0).

**Corrected finding (atomic model — all 6 directional entropies entered at once,
`spec="atomic"`):** the effect is a **source/target directional asymmetry**, not a clean
"GNNs worse with entropy":

- **`tgt_in`** (others' signs *into* v = contested target reputation): GNNs hurt MORE than
  the walk — pooled β walk ≈ −1.9 vs GNN ≈ −3.0, gap negative on 100 % of datasets
  (sign-test p = 0.031). *(This is the part the v1 "target in-edge" story got right.)*
- **`src_out`** (u's outgoing signs = inconsistent rater): the **largest** entropy effect
  of all, and here the **WALK is hurt more** (β walk ≈ −2.7 vs GINEConv ≈ −1.3). This
  shows up in the atomic model AND in the old marginal3 `out_*`/`inout` combos — so "source
  is weak" was wrong.
- `tgt_out`, `src_in` ≈ null; `twohop_*` (2-hop/path) entropy barely matters (and
  `twohop_out` runs slightly positive for GNNs).

A **count-pooled composite** (one node-β + one path-β, `spec="composite"`) is provided as a
compact companion, but pooling `src_out` (walk-worse) with `tgt_in` (GNN-worse) **cancels
the asymmetry** — read it with the atomic forest, never alone.

Outputs: `outputs/lead4c_entropy_logit_regression/` (zip `lead4c_atomic_outputs.zip`):
`atomic_forest.png` (one-glance, annotated), `atomic_heatmap.png`,
`atomic_binned_accuracy.png`, `composite_forest.png`, `atomic_pooling_caterpillar.png`,
`appendix_combo_ranking.png`, `correlation_heatmap.png`, `report.md` (TL;DR + tables).
Cross-refs also updated: `RESEARCH_LEADS_SUMMARY.md`, `CANONICAL_RERUN_FINDINGS.md`,
`CLAUDE.md`, memory `lead4c-target-entropy-regression`.

## Where the raw results actually are

Script (reproducible, 3 stages — compute/fit/plot, see its docstring for exact CLI):
`scripts/lead4c_entropy_logit_regression.py`

All outputs live in `outputs/lead4c_entropy_logit_regression/` (gitignored, not
committed — regenerate via the script if missing):

| File | What it is |
|---|---|
| `joined_table.pkl` | per-edge feature table (u,v,y,p,correct + all entropy/2-hop columns), built once per dataset |
| `fit_results.pkl` / `fit_results.csv` | **the full numeric sweep** — every (dataset\|pooled spec, model, node-variant, twohop-variant, term) row: beta, beta_std, se_naive, se_robust, z, p, p_fdr, odds_ratio, pseudo_r2, base_rate, n. 2352 rows. This is the ground truth table — everything else is a view onto it. |
| `professional_summary.csv` / `.png` | **the compact "final" deliverable** — pooled, shared-slope spec, twohop fixed at `inout`, `b_tgt` only, 4 node-variants × 4 models, with beta + 95% CI + BH-FDR significance stars + pseudo-R². This is what satisfied the last "professional but compact" request. |
| `advisor_summary.png` | earlier plain-language 2x2 figure: live (`in_in`/`inout`) vs null (`out_out`/`inout`) bar comparison + full-sweep sign-test scorecard panel |
| `win_matrix_b_src.png` / `_b_tgt.png` / `_b_2hop.png` | per-term win-matrix (model x dataset x variant, color = sign/significance) — most exhaustive view |
| `forest_in_in_inout.png` / `forest_out_in_out.png` | forest plots (beta + CI per model) for two representative variant pairings |
| `pdp_*.png` (6 files) | partial-dependence-style plots for two representative variant pairings × 3 terms |
| `report.md` | auto-generated full report, regenerates from `fit_results.csv` |
| `SIMPLE_*`, `COMPACT_*`, `PROFESSIONAL_compact.*` | **scratch only** — built via inline bash during the session, NOT wired into the script. Superseded by `professional_summary.*`. Safe to delete; not reproducible from the script as-is. |

## Open / not yet decided (for the next session)

1. **Not committed.** `scripts/lead4c_entropy_logit_regression.py` and the doc updates
   (`RESEARCH_LEADS_SUMMARY.md`, `CANONICAL_RERUN_FINDINGS.md`, `CLAUDE.md`, memory
   files) are written but not yet `git add`/committed — confirm with user before
   committing.
2. **Scratch files cleanup.** The `SIMPLE_*`/`COMPACT_*`/`PROFESSIONAL_compact.*` files
   in the output dir are leftover scratch, not reproducible from the script. Decide:
   delete, or formalize any of them as real script functions.
3. **Not yet disentangled:** v's in-edge entropy could still correlate with in-degree
   (hub-ness proxy) — an explicit degree-control regression was deliberately skipped
   this round to keep the first pass to the literal entropy spec (see memory
   `lead4c-target-entropy-regression`).
4. Whether `professional_summary.png` is the FINAL agreed figure, or needs another
   iteration — last user instruction was "professional but compact, don't drop p
   values," which this figure satisfies, but it was not explicitly confirmed before
   the session got compacted.
