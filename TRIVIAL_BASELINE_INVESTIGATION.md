# Trivial vertex-reputation baseline — investigation (2026-08-23)

## Why this exists

While auditing the `mask_edge_tokens` ablation (see `PAPER_CLOSEOUT_LOG.md`'s 2026-08-23
entries and `aaai2027/WSDM_format_revised.tex`'s `abl:maskedge` paragraph), the corrected
numbers showed edge-sign context has **no statistically significant effect on test AUC on any
of the six datasets** once evaluated correctly. That raised an obvious, uncomfortable
question: if the model barely uses edge-sign context, and vertex identity alone accounts for
nearly all of the signal (`abl:masknode`'s 13-33pp collapse when vertex tokens are masked),
how much of Pewter's AUC — and its 3-5pp lead over every GNN/SGNN baseline — is really just
"knowing which vertices these are," something a much simpler model could plausibly capture
too?

This doc is the falsifiable probe built to answer that, plus a full stress test to rule out
the result being a bug or a data-plumbing artifact before trusting it.

## The trivial model

Two features per edge $(u, v)$, computed only from the train+mask pool (never val/test):

- `out_rate[u]` — the fraction of $u$'s train-pool **out-edges** that are positive (how
  consistently $u$ rates others positively).
- `in_rate[v]` — the fraction of $v$'s train-pool **in-edges** that are positive (how
  positively $v$ tends to be rated by others).
- Cold start (vertex never seen in train pool): falls back to the train-pool's global
  positive rate.

Fit a plain `sklearn.linear_model.LogisticRegression` on these two features, using the
**val** split (same split PEWTER's own model selection uses), then score on **test**. No
graph structure beyond these two scalar rates, no embeddings, no message passing, no walks.

Code: [`scripts/trivial_baseline_probe.py`](scripts/trivial_baseline_probe.py) — reads the
walk model's own canonical `dataset_cache["splits"]` (the authoritative split, not the
GNN-baseline splits which have a separate, documented provenance issue — see
`FABRICATED_REVERSE_EDGES.md`) and PEWTER's saved `test_predictions.pkl`, read-only, no
training or checkpoint touched. Run with `.venv/bin/python scripts/trivial_baseline_probe.py`.

## Why this is not a new idea — literature check

This is, in substance, not a novel construction — it is close to two established results in
the signed-network literature, on these exact same benchmark datasets:

- **Leskovec, Huttenlocher & Kleinberg, "Predicting Positive and Negative Links in Online
  Social Networks"** (WWW 2010) and its companion **"Signed Networks in Social Media"** (CHI
  2010) — the paper that introduced Epinions/Slashdot/Wikipedia-vote as signed-network sign-
  prediction benchmarks in the first place. Their **"degree features"** are exactly this: the
  source's out-degree split by sign and the target's in-degree split by sign, fed into a
  logistic regression, evaluated as one of their core feature classes (alongside triad/status-
  theory features). They report that degree features alone already predict sign with high
  accuracy on all of these networks.
- **Kumar, Spezzano, Subrahmanian & Faloutsos, "Edge Weight Prediction in Weighted Signed
  Networks"** (ICDM 2016) — the **Fairness–Goodness** model. "Goodness" of $v$ (how
  liked/trusted $v$ is) and "fairness" of $u$ (how reliably $u$ rates others) are defined via a
  mutually recursive fixed point over the whole graph, rather than a one-shot rate — a more
  refined version of the same idea — and are shown to have strong predictive power on Bitcoin,
  Epinions, and Wikipedia-derived signed networks (the same dataset family used here).

So the honest answer to "why hasn't anyone used this" is: **they have** — this is a simplified,
one-shot version of an established baseline family in this exact literature, not something
overlooked. What's specific to this investigation is comparing it directly against a modern
walk-Transformer on the project's own canonical splits, which doesn't currently appear in
Table 1 (that table's baselines are all GNN/SGNN architectures — GCN, GAT, SGCN, SiGAT, GSGNN,
SNEA, CopulaLSP, node2vec — none of which include a plain vertex-reputation baseline like this
or the Fairness-Goodness model).

## Stress test — is this a bug or an artifact?

Before trusting the numbers, three checks (all in the same script):

1. **Split disjointness**: train_pool/val/test triple overlap is exactly 0 on all 6 datasets.
2. **Shuffle control**: permute train-pool signs (keep graph structure fixed), refit, re-score.
   If the pipeline had a leak or artifact, this should still show high AUC; if the pipeline is
   clean, it should collapse to ~0.50. **Result: 0.477–0.517 on all 6 datasets** — collapses to
   noise every time, as expected from a clean pipeline.
3. **Paired bootstrap** (2000 resamples) of (full − trivial) AUC on the *exact same* shared
   test edges for both models (not just point estimates) — this is what makes the comparison
   below defensible rather than an eyeball difference.

## Results (single split, seed 42 — same split PEWTER's own `E32_PY314_LOCALATTN4` checkpoints use)

| dataset | trivial AUC | full model AUC | delta (full − trivial) | 95% CI | P(full ≤ trivial) | shuffle control |
|---|---|---|---|---|---|---|
| Wiki-elec | 0.9098 | 0.9075 | −0.0023 | [−0.0060, +0.0015] | 0.886 | 0.493 |
| Bitcoin-alpha | 0.9085 | 0.9185 | **+0.0100** | [+0.0006, +0.0198] | 0.020 | 0.490 |
| Bitcoin-otc | 0.9179 | 0.9360 | **+0.0181** | [+0.0085, +0.0289] | 0.000 | 0.477 |
| Epinions | 0.9431 | 0.9527 | **+0.0096** | [+0.0080, +0.0111] | 0.000 | 0.476 |
| Wiki-RfA | 0.9065 | 0.8975 | **−0.0090** | [−0.0121, −0.0059] | 1.000 | 0.517 |
| Slashdot | 0.9020 | 0.8980 | **−0.0040** | [−0.0059, −0.0022] | 1.000 | 0.500 |

(P(full ≤ trivial) is the fraction of the 2000 paired bootstrap resamples where the trivial
model's AUC was at least as high as the full model's — read it as a one-sided significance
check in the direction the sign of the delta suggests.)

## What this actually shows

**The trivial 2-parameter baseline is genuinely strong everywhere (AUC 0.90–0.94), confirmed
real by the shuffle control, and this is not surprising given the literature above — the
surprising part is the comparison to the full model, and it does not tell one clean story:**

- **Bitcoin-otc and Epinions**: full model beats trivial by a real, statistically significant
  margin (+1.8pp, +1.0pp, both $p<0.05$ with tight CIs excluding 0). Bitcoin-alpha similarly,
  though more marginal (+1.0pp, $p=0.02$, CI barely excludes 0).
- **Wiki-elec**: statistically indistinguishable (CI includes 0, $p=0.89$) — the sophisticated
  model does not measurably beat two vertex-reputation numbers here.
- **Wiki-RfA and Slashdot**: the full model is **significantly worse** than the trivial
  baseline (−0.9pp, −0.4pp, both $p=1.000$ in bootstrap, CIs excluding 0 in the negative
  direction). On these two datasets, a 2-parameter logistic regression measurably outperforms
  the entire walk-Transformer pipeline.

**An earlier, narrower version of this investigation (2 datasets: wiki-elec, bitcoin-alpha)
suggested a clean story — "big walk-vs-GNN AUC gap correlates with full model clearly beating
trivial." The full 6-dataset picture does not support that story as stated**: bitcoin-alpha is
one of the largest walk-vs-GNN gap datasets (per `CLAUDE.md`'s SOTA table) and only marginally
beats trivial; slashdot is also a large-gap dataset yet trivial *beats* the full model there.
So whatever explains the gap-vs-trivial-baseline pattern, it isn't simply "dataset has a big
walk-vs-GNN gap."

## What this does *not* mean

- **It does not undermine Table 1's headline claim.** Pewter still beats every GNN/SGNN
  baseline (SiGAT, SGCN, GAT, GCN, GSGNN, SNEA, CopulaLSP, node2vec) on all 6 datasets on the
  10-seed canonical splits — that comparison is unaffected, since none of those baselines is
  this trivial reputation model. This investigation adds a *new* comparison point that simply
  isn't in the current baseline suite, not a correction to an existing one.
- **It is consistent with, not contradictory to, the project's existing Lead 4c finding**
  (`CLAUDE.md`'s "Lead 4/4b/4c" section): message-passing GNNs structurally cannot see a
  source vertex's own outgoing behavior (`src_out`), since they only aggregate over a node's
  in-neighbors — this is exactly the architectural reason a 2-parameter model that explicitly
  uses `out_rate[u]` can beat GNN baselines. Pewter's embedding-table + attention approach can
  in principle see this too (and does, per `abl:masknode`'s huge drop when vertex tokens are
  removed) — the open question is why it doesn't reliably beat a model that uses only this
  feature on 2 of 6 datasets.
- **It is not (yet) evidence of a bug.** The shuffle control and split-disjointness checks came
  back clean on all 6 datasets; the paired bootstrap uses the model's own real saved
  predictions on the model's own real split.

## Scope limitation — read before citing these numbers anywhere

This is a **single split (seed 42)** comparison, not PEWTER's own 10-seed canonical
methodology (`scripts/run_multiseed_pewter.py`, the standard every Table 1 number in the paper
follows). The full-model AUCs above come from the `E32_PY314_LOCALATTN4` seed-42 checkpoints
specifically and are close to but not identical to Table 1's 10-seed means (e.g. Wiki-RfA:
0.8975 here vs. 0.8914 ± 0.0046 mean±std in Table 1 — within one std, consistent with ordinary
seed-to-seed variance). **Before this goes anywhere near the paper**, the trivial baseline
itself should be refit across the same 10 seeds (42-51) the same way, so the comparison is
apples-to-apples with the rest of the paper's methodology — not yet done.

## Open questions / suggested next steps

1. **Extend to 10 seeds** (per the scope limitation above) — is the wiki-rfa/slashdot
   full-loses-to-trivial pattern stable across seeds, or is it seed-42-specific noise on top of
   an otherwise-close comparison?
2. **Why does the sophisticated model sometimes lose to 2 numbers?** Candidate explanations,
   none yet tested: (a) PEWTER's training-time regularization (dynamic resplit, 20% node-token
   replacement — both load-bearing defaults per `CLAUDE.md`) may specifically dilute the sharp
   reputation signal these two datasets lean on hardest, trading it for robustness that doesn't
   pay off against a baseline this simple; (b) wiki-rfa/slashdot may have different train-pool
   size/degree-distribution properties that make the one-shot rate estimate unusually reliable
   there relative to what a walk-sampled model can extract; (c) something specific to these two
   datasets' graph structure — worth checking against `aaai2027/DATASET_STATS.md`'s existing
   per-dataset statistics before proposing new ones.
3. **Should this go in the paper at all, and if so where?** This directly overlaps with the
   professor's own still-unresolved Abstract line about vertex-vs-edge information "alternative
   theories" and the stubbed-out Section 6.1 paragraph (D) (see the closeout plan's Group C-bis)
   — that thread already flags an open, undecided question about why some datasets' predictions
   lean more on edges vs. vertices. This trivial-baseline result is likely relevant evidence for
   that discussion, but integrating it is a separate decision from this investigation, not made
   here.
