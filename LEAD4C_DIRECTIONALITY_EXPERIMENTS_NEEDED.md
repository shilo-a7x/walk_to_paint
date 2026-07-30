# Lead 4c follow-up — experiments needed to firm up the directionality claim for PEWTER

**Purpose of this doc:** a request list for the Lead 4c research thread, written up because the
paper (`aaai2027/pewter_aaai.tex`, AAAI-27 submission "PEWTER") needs a more statistically solid
version of a claim that currently rests on thin evidence in one place. This doc does not run
anything itself — it specifies what's needed and why, for whoever picks it up in the Lead 4c
session. Context lives in `LEAD4C_ASYMMETRY.md`, `LEAD4C_EQUATIONS.md`, `lead4_coefficients.md`,
and `CLAUDE.md`'s "Lead 4/4b/4c" section.

## The claim the paper currently makes

The draft's abstract and intro (professor's own framing, kept as the paper's headline claim for
now — see `\S`Introduction, "Assymetric vertex entropy") say: edge-label entropy measured from an
edge's **source** side is lower (more informative) than from its **target** side, so information
about the label flows *with* the edge orientation, and this is why Pewter (which reads context in
the walk's forward/edge-anchored direction) beats vertex-centric GNNs. The abstract has a literal
open marker — `XXXXXXXX Shilo which is it more source or end ? XXXXXXXXX` — that this thread
should help answer with numbers, not just intuition.

**Framing decision (2026-07-15):** for now, keep this as a one-sided claim ("GNNs are the ones
hurt by this, source-side is where the information is") matching the professor's original draft,
not the more nuanced architecture-specific directional split Lead 4c's atomic regression actually
found (walk model hurt more by `src_out`, GNNs hurt more by `tgt_in` — see
`LEAD4C_ASYMMETRY.md` "Key finding: model families fail on opposite entropy directions"). That
reframing is real and already measured, but is being held back for a later revision of the
paper's framing (per the standing reconciliation-memo plan) — **don't re-derive or re-litigate
it here.** This doc is only about strengthening the simpler, already-adopted claim below.

## What already exists (Claim 1, from `LEAD4C_ASYMMETRY.md`)

Per-node paired test, `H_out(n)` (entropy of n's outgoing edge signs) vs `H_in(n)` (entropy of
n's incoming edge signs), one-sided Wilcoxon signed-rank test on non-tied pairs, computed
directly from `load_edges_canonical` (no walk sampling involved — a property of the raw graph):

| Dataset | frac(H_out<H_in) among non-tied | p (one-sided) | Walk-vs-GNN AUC gap (pp, from `CLAUDE.md` SOTA table) |
|---|---|---|---|
| bitcoin-alpha | 0.516 | 5.1e-04 | 3.11 |
| bitcoin-otc | 0.551 | 2.7e-11 | 4.55 |
| epinions | 0.623 | 3.7e-244 | 4.22 |
| slashdot090221 | 0.602 | 2.6e-218 | 4.25 |
| wiki-elec | 0.404 (reversed) | 1.00 | 1.08 |
| wiki-rfa | 0.424 (reversed) | 1.00 | 1.01 |

Each per-dataset row is individually rock-solid (huge n, extreme p-values). **The weak link is
the cross-dataset story**, which is currently only a Spearman rank correlation on 6 points
(ρ≈0.71–0.77) between "how strongly Claim 1 holds" and "how big the walk-vs-GNN AUC gap is" —
`LEAD4C_ASYMMETRY.md` itself flags this as "n=6 — too small for a real p-value." This is exactly
the number the paper wants to lean on for "the gain is largest exactly where source and endpoint
entropy [asymmetry] are high" — it needs to be either strengthened or its uncertainty made
explicit and honest in the writeup.

**Also unverified:** whether Claim 1's numbers change once the fabricated-reverse-edge mirrors
are excluded (`FABRICATED_REVERSE_EDGES.md` documents 14–48% fabricated edges in
`baselines/splits/<ds>.pt`'s edge lists specifically). Confirm whether `load_edges_canonical`
(the loader Claim 1 was computed from) already excludes these or not before trusting the numbers
above as clean — if it doesn't, this is a prerequisite check before anything else below.

## Experiments requested (ranked)

| # | Task | Cost | Why it matters | Notes |
|---|---|---|---|---|
| 1 | **Confirm `load_edges_canonical` is not affected by the fabricated-reverse-edge issue** (or re-derive Claim 1's table using `build_real_dense_edge_set()` if it is) | ~15 min, no retraining | Prerequisite — if Claim 1's entropy numbers are computed on a graph with 14–48% fabricated mirror edges, everything downstream is suspect | See `FABRICATED_REVERSE_EDGES.md` |
| 2 | **Bootstrap CI on each dataset's walk-vs-GNN AUC gap** (resample test edges with replacement from the already-saved predictions — `predictions_raw_canonical.pkl` per `CLAUDE.md`'s file list — recompute AUC gap per resample) | ~30–60 min, no retraining, reuses existing predictions | The y-axis of the n=6 correlation is currently a **single-seed point estimate** (no multi-seed infra exists anywhere in this project yet, per `plan-stats-rigor.md`) — a resampling CI is the cheapest way to know if the ranking of the 6 gaps is even stable, without waiting on real multi-seed retraining | Doesn't need GPU training — pure post-hoc resampling of saved predictions |
| 3 | **Make the x-axis continuous, not categorical.** Replace "confirmed/reversed" with the actual effect size per dataset — e.g. `frac(H_out<H_in) - 0.5` (or the Wilcoxon rank-biserial correlation) — and refit gap ~ effect_size as a real regression on 6 points with a CI on the slope, instead of eyeballing a 4-vs-2 split | ~15 min, no retraining | A continuous regression is a more honest and more citable number than a binary "4/6 confirmed" split, and directly produces the number the abstract wants ("gain is largest where asymmetry is largest") | Still n=6 — report the CI width honestly, don't oversell precision |
| 4 | **Pooled/meta-analytic estimate of Claim 1 across all 6 datasets**, treating each dataset's `frac(H_out<H_in)` and its sampling variance as one effect-size estimate, combined via a random-effects meta-analysis (report pooled effect, 95% CI, and a heterogeneity statistic like Cochran's Q / I²) | ~30 min, no retraining | This is the number that actually answers "source or end" for the abstract in one aggregate, honest sentence — including the wiki-elec/wiki-rfa reversal as *quantified heterogeneity* rather than silently dropped exceptions | Standard technique for combining several dataset-level effect sizes; don't need to invent methodology, just apply it (e.g. `statsmodels`' meta-analysis tools or a manual inverse-variance-weighted pooled estimate) |
| 5 | **Multi-seed stability check on the AUC-gap side only** (entropy side is already extremely significant per-dataset, doesn't need this) — at minimum, if full multi-seed retraining is out of scope, extend #2's bootstrap to also resample by walk-subset (not just test edges) to approximate seed variance from the sampling stage | larger — needs scoping, possibly real retraining on 1-2 datasets as a spot check | Confirms the ranking of AUC gaps used in #3/#4 isn't an artifact of the single frozen seed=42 split | Lowest priority — only worth it if #2's bootstrap CIs turn out to be wide enough to threaten the ranking |

## Explicitly out of scope for this request

- Re-litigating or expanding the architecture-specific `src_out`-vs-`tgt_in` directional-split
  finding (the atomic regression in `lead4_coefficients.md`) — that's a separate, already-settled
  result, just not the one currently being written into the paper.
- Any new walk-sampling or attention-direction experiments (Lead 5/6, the `attention_analysis.py`
  BFS-distance bug) — unrelated to this specific claim.
- Retraining the production model or the hardness miner — none of the above needs it except the
  optional spot-check in #5.

## What comes back to the paper

Once #1–#4 (and #5 if needed) are done, the numbers that matter for `pewter_aaai.tex` are: (a) a
single pooled effect size + CI for "source-side entropy is lower than target-side" (replaces the
current "confirmed on 4/6" framing in the abstract's open marker), and (b) a defensible
correlation/regression statistic (with CI, not just a Spearman ρ on 6 points) for "the AUC gain
is largest where this asymmetry is largest." Bring both back with their source script/output path
so they can be cited precisely and their freshness re-confirmed before going into the text, per
the paper's standing freshness-discipline rule.
