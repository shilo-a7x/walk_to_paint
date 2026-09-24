# EID overfitting/gap investigation — 2026-09-16

Session goal: real 10-seed noablation numbers looked worse than hoped on wiki-elec/wiki-rfa
(EID trails production by -2.06pp / -3.29pp respectively), while epinions is the only dataset
where EID beats production (+0.13pp). This doc records the investigation into why, before
committing more GPU time to a fix.

## 1. Real 10-seed noablation results (post both bug fixes -- trustworthy)

Source: `experiments/edge_identity_tokens/thesis_figure_data/eid_result1_table.csv`
(`eid_result1_table.py`).

| dataset | EID (10-seed) | production (local attn, 10-seed) | delta |
|---|---|---|---|
| bitcoin-alpha | 0.9036 +/- 0.0120 | 0.9134 +/- 0.0173 | -0.98pp |
| bitcoin-otc | 0.9269 +/- 0.0062 | 0.9317 +/- 0.0065 | -0.48pp |
| epinions | 0.9549 +/- 0.0011 | 0.9536 +/- 0.0015 | **+0.13pp (EID wins)** |
| wiki-elec | 0.8817 +/- 0.0053 | 0.9023 +/- 0.0031 | -2.06pp |
| wiki-rfa | 0.8585 +/- 0.0335 | 0.8914 +/- 0.0046 | -3.29pp |
| slashdot090221 | 0.8890 +/- 0.0031 | 0.8968 +/- 0.0016 | -0.78pp |

## 2. Ablation seed-42 backfills are apples-to-apples -- confirmed, not assumed

Checked file mtimes directly: every seed-42 ablation checkpoint used for backfill
(`EID_GAP_*`, `EID_ABL2_*`, `EID_PHASE0_SIGNONLY_PILOT_*`) was built *before* Phase 1's
multiseed campaign ever started (seeds 43-51 didn't exist yet at that point). Two
independent reasons this makes them safe to reuse:
- **Cache-seed bug** (the walk-cache seed-collision bug fixed earlier this workstream) never
  had anything to collide with -- seed 42 was always the first/only seed touching each
  `(dataset, num_walks)` cache path at build time.
- **Epoch-hardcode bug**: the old buggy hardcode forced epochs=50 for every dataset; the new
  corrected behavior also lands on 50 for every dataset in EID context (epinions/slashdot090221
  via the explicit `EID_EPOCH_OVERRIDE`, the other 4 via config default). No discrepancy.

## 3. Seed 50 investigation -- NOT a general "bad seed," wiki-rfa-specific overfitting

Per-dataset seed-50 check (all 6 datasets, noablation): bitcoin-alpha/bitcoin-otc/epinions/
slashdot090221's seed 50 is unremarkable -- within normal cross-seed spread every time.

**wiki-rfa's seed 50 is uniquely broken, and it's overfitting, not an optimization failure.**
Pulled real per-epoch curves via TensorBoard EventAccumulator:
- train_auc_epoch climbs cleanly to 0.92+ over 50 epochs (healthy, normal training curve, no
  NaN, no loss spike -- train_loss decreases smoothly 0.65->0.34).
- val_auc_epoch gets stuck at 0.76 the entire run (best epoch 46, val=0.7602).
- **Train/val gap = 0.160** at seed 50, vs. 0.073 at seed 43 (wiki-rfa's own "good" seed).

The model learns the training edges just fine; it simply fails to generalize for this
seed's particular split/init. This is a real, reproducible failure mode, not contamination
or measurement noise (checked post both bug fixes).

## 4. Train/val overfitting gap, all 6 datasets (seed43 noablation, gap at best val epoch)

| dataset | train/val gap | EID vs. production delta |
|---|---|---|
| slashdot090221 | -0.025 (still underfitting) | -0.78pp |
| epinions | **0.020** (smallest) | **+0.13pp (EID wins)** |
| bitcoin-otc | 0.048 | -0.48pp |
| wiki-elec | 0.056 | -2.06pp |
| bitcoin-alpha | 0.063 | -0.98pp |
| **wiki-rfa** | **0.073** (largest, even at a "good" seed) | -3.29pp |

Overfitting-gap ranking tracks the EID-vs-production gap almost perfectly. wiki-rfa's
catastrophic seed 50 (gap=0.160) is this same failure mode taken to an extreme.

## 5. Architecture-capacity pattern (from `logs/eid_gap_closer/state.json`/`state_v2.json`)

| dataset | \|E\| | edge_embed_rank | node_embed_dim |
|---|---|---|---|
| epinions | 840,799 (largest) | **11 (smallest)** | **0 (no dedicated node channel)** |
| bitcoin-alpha | 24,186 | 16 | 96 |
| bitcoin-otc | 35,592 | 20 | 64 |
| wiki-elec | 103,689 | 20 | 64 |
| wiki-rfa | 177,211 | **26 (2nd-largest)** | 64 |
| slashdot090221 | 549,202 | 31 (largest) | 32 |

wiki-elec/wiki-rfa carry large identity-embedding capacity despite having the fewest edges
of any dataset except bitcoin-alpha/bitcoin-otc -- and unlike those two, their EID deficit
vs. production is much larger, suggesting the extra capacity isn't earning its keep.

## 6. Real attention-weight evidence (not hyperparameter inference) -- closes the loop

Script: `experiments/edge_identity_tokens/inspect_attention_node_vs_edge.py` (read-only,
monkey-patches `LocalAttentionEncoderLayer._sa_block` to capture real softmax attention
weights with no approximation). 300 masked-target-edge instances per dataset, real
checkpoints (epinions/wiki-elec s43, wiki-rfa s42).

Attention mass from a masked target-edge query, by source-token type. **Re-run directly
(not just via the delegated agent) to pull the per-layer breakdown** -- user correctly
flagged that layer 0 (before any depth-mixing) is the more causally-interpretable read than
pooled/last-layer, since at layer 0 the attended-to keys are still raw embeddings, not
representations already mixed across several rounds of self-attention:

| dataset | layer 0: node | layer 0: edge-context | layer 0: self |
|---|---|---|---|
| **epinions** | **22.3%** | 41.6% | 36.1% |
| wiki-elec | 41.0% | 42.6% | 16.4% |
| **wiki-rfa** | **49.0%** | 43.3% | 7.7% |

Clean monotonic gradient right from the first layer: epinions attends least to node identity
and most to its own revealed identity + edge context; wiki-rfa attends most to node identity
from the very start. This is a cleaner, more directly interpretable confirmation than the
depth-mixed pooled/last-layer numbers (kept below for reference).

Pooled (all layers) / last-layer, for reference:

| dataset | node (pooled / last layer) | edge-context (pooled / last) | self/identity (pooled / last) |
|---|---|---|---|
| epinions | 52% / 69% | 31% / **26%** | 17% / 5% |
| wiki-elec | 70% / 93% | 21% / **1%** | 9% / 6% |
| wiki-rfa | 76% / 96% | 21% / **1%** | 3% / 3% |

By the final layer, wiki-elec/wiki-rfa have further collapsed toward node identity (93-96%),
while epinions still retains a real quarter of its attention on edge context. Together: EID's
identity-embedding capacity is being used less, from the very first layer onward, on exactly
the two datasets (wiki-elec/wiki-rfa) where it also carries the most identity-embedding
capacity and the largest train/val overfitting gap -- consistent with that capacity being
spare overfitting surface rather than useful signal.

Rerun with: `.venv/bin/python experiments/edge_identity_tokens/inspect_attention_node_vs_edge.py --device <N> --n-instances 300 --datasets epinions,wiki-elec,wiki-rfa`

**Caveat** (flagged correctly by the agent that ran this): attention weight is a
mechanism-consistent signal, not strict proof of causal reliance. A `model.mask_node_tokens`
ablation on EID would close that loop; not done here.

## 7. Token-replacement regularization mechanism (answers "why UNK vs real token")

Confirmed via code read (`src/model/lit_model.py`, `eid_src/model/lit_model.py`):
- `*_replace_prob`: per-token independent probability a node/edge-identity token gets
  touched at all.
- `*_replace_unk_ratio`: conditional on being touched, probability of blanking to `<UNK>`
  vs. probability `(1 - ratio)` of substituting a different real token drawn from elsewhere
  in the batch.
- **No BERT-80/10/10 citation or deeper documented rationale anywhere** (MECHANISM.md just
  says "forces the model to not over-rely on any one vertex's specific identity embedding").
  Self-motivated regularizer, Optuna-tuned freely, no cited technique behind the UNK/real
  split specifically.
- `edge_embed_weight_decay` confirmed real (not a no-op): separate AdamW param group
  applying L2 only to the identity embedding table (`eid_src/model/lit_model.py`
  `configure_optimizers()`).
- EID's edge-identity replacement only touches `input_ids` (identity), never `sign_ids` --
  deliberate, per the docstring: pushes reliance toward sign+structure over memorized
  per-edge identity.

## 8. Cheap rank-reduction diagnostic (edge_embed_rank swept on wiki-elec/wiki-rfa)

Script: `experiments/edge_identity_tokens/run_rank_diagnostic.py`. Single-seed=42, winning
architecture otherwise unchanged, `edge_embed_rank` in {0 (identity fully disabled), 4, 8}
vs. the current winners (20/26).

| dataset | rank | test AUC | vs. current winner |
|---|---|---|---|
| wiki-elec | 20 (current winner) | 0.8888 | -- |
| wiki-elec | 8 | 0.8777 | -1.11pp |
| wiki-elec | 4 | 0.8874 | -0.14pp |
| wiki-elec | 0 (full-width unfactorized identity -- NOT disabled, see correction) | **0.8883** | **-0.05pp (essentially unchanged)** |
| wiki-rfa | 26 (current winner) | 0.8863 | -- |
| wiki-rfa | 8 | 0.8690 | -1.73pp |
| wiki-rfa | 4 | **0.5035 (collapsed to ~random)** | catastrophic |
| wiki-rfa | 0 (full-width unfactorized identity -- NOT disabled, see correction) | 0.8639 | -2.24pp |

(wiki-elec rank=0 needed a manual retry -- the original 3-way-simultaneous sweep raced to
build a never-before-built seed42 EID cache and one process lost the race, truncated read,
not a real bug. Retried standalone once the cache existed; result above is real.)

> **CORRECTION (2026-09-24, caught by the user, verified in code): `edge_embed_rank=0` does
> NOT disable identity.** In `eid_src/model/model.py`, rank<=0 keeps the unified `self.embed`
> table built by the parent class, and `prepare_eid_data.py` sets `cfg.model.vocab_size` to
> the EID cache's expanded vocab (nodes + one row per edge). So rank=0 = every edge gets its
> own FULL-WIDTH (embedding_dim) identity row, unfactorized -- the *most* identity capacity,
> not none. **No existing EID setting turns identity off or reproduces production.** The
> rank=0 rows in the table above therefore compare "full-width vs. low-rank identity," and the
> interpretation below ("identity is nearly inert on wiki-elec", "disabling identity costs
> wiki-rfa 2.2pp") is WRONG and is retracted. Kept only for provenance.

**~~Two genuinely different stories per dataset, not one:~~ (RETRACTED, see correction above)**

- **wiki-elec: identity is nearly inert.** rank=0 (0.8883) is within noise of rank=20
  (0.8888) -- disabling EID's whole identity mechanism costs essentially nothing. Combined
  with the attention finding (wiki-elec's node-attention share is already high even at layer
  0, 41%), this says wiki-elec's -2.06pp gap to production is **NOT primarily an
  identity-capacity/overfitting problem** -- the identity channel barely does anything either
  way. Whatever's causing wiki-elec's gap likely sits elsewhere (architecture-search fit,
  LocalAttn4 config choice, or something structural about this dataset independent of EID's
  identity mechanism specifically).
- **wiki-rfa: identity matters, and the architecture is fragile.** Every rank reduction hurts
  monotonically-ish (26 -> 8 -> 0: 0.8863 -> 0.8690 -> 0.8639), so identity is contributing
  real signal here, unlike wiki-elec -- shrinking capacity is not a free lunch. But rank=4
  collapsing to ~random (0.5035, single seed=42) is a second, independent instability
  data point on top of seed 50's collapse -- wiki-rfa's chosen architecture (rank=26,
  hidden_dim=128, lr=0.0033) looks fragile across both the seed axis and the rank axis, not
  just prone to one specific bad accident.

## 9. Phase 1 fully complete (2026-09-16 18:45) -- final significance numbers

Verified genuinely complete against disk (not just the log): 162/162 seed-sweep jobs +
19 seed-42 backfills/fresh-trains = 181/181, 0 processes running, 0 GPU memory in use, 0
failures across the whole campaign.

**EID vs. production** (`eid_significance.py --mode vs_production`, one-sided paired
Wilcoxon, 10 seeds):

| dataset | eid mean | production mean | wins | p-value |
|---|---|---|---|---|
| epinions | 0.9549 | 0.9536 | 8/10 | **0.0225 (significant, EID wins)** |
| bitcoin-otc | 0.9269 | 0.9318 | 2/10 | 0.981 |
| bitcoin-alpha | 0.9036 | 0.9134 | 1/10 | 0.997 |
| slashdot090221 | 0.8890 | 0.8968 | 0/10 | 1.0 |
| wiki-elec | 0.8817 | 0.9023 | 0/10 | 1.0 |
| wiki-rfa | 0.8585 | 0.8914 | 0/10 | 1.0 |

Significant on 1/6 datasets (epinions only). wiki-elec/wiki-rfa lose every single seed
pairing (0/10).

**Ablation: mask_context_edges** (context edges lose BOTH identity+sign) vs. noablation:

| dataset | noablation | mask_context_edges | diff | p-value |
|---|---|---|---|---|
| wiki-rfa | 0.8585 | **0.7324** | **+0.0456 (biggest drop)** | 0.00098 |
| wiki-elec | 0.8817 | 0.8464 | +0.0358 | 0.00098 |
| bitcoin-alpha | 0.9036 | 0.8693 | +0.0330 | 0.00098 |
| bitcoin-otc | 0.9269 | 0.8970 | +0.0297 | 0.00098 |
| slashdot090221 | 0.8890 | 0.8787 | +0.0108 | 0.00098 |
| epinions | 0.9549 | 0.9544 | +0.0005 | 0.043 |

Significant on 6/6 -- context-edge information (identity+sign together) is genuinely used
everywhere, wiki-rfa hit hardest of all.

**Ablation: mask_context_sign_only** (context edges keep identity, lose only sign) vs.
noablation:

| dataset | noablation | sign_only | diff | p-value |
|---|---|---|---|---|
| epinions | 0.9549 | 0.9544 | +0.0005 | 0.0039 |
| bitcoin-otc | 0.9269 | 0.9233 | +0.0028 | 0.042 |
| slashdot090221 | 0.8890 | 0.8862 | +0.0034 | 0.020 |
| bitcoin-alpha | 0.9036 | 0.9030 | -0.0007 | 0.36 (n.s.) |
| wiki-elec | 0.8817 | 0.8810 | +0.0003 | 0.50 (n.s.) |
| wiki-rfa | 0.8585 | 0.8681 | -0.0018 | 0.78 (n.s., slightly favors the ablation) |

Only 3/6 significant, effect sizes tiny throughout -- losing sign ALONE (identity still
visible) barely matters, in sharp contrast to losing both together. **This means context
edges' identity signal, not their sign, is carrying most of what `mask_context_edges`
takes away** -- consistent with the attention-weight finding that node/edge identity
dominates attention mass, and with wiki-rfa specifically depending on identity being
present (rank reduction and identity-removal both hurt it) while barely caring whether
that identity comes with a visible sign or not.

## 10. wiki-rfa regularization diagnostic -- launched 2026-09-16 18:48 (complete)

Motivated by section 8's finding that wiki-rfa's architecture is fragile on two axes
(seed 50's overfitting collapse, rank=4's collapse to ~random) while identity is
genuinely load-bearing there (section 9) -- so shrinking capacity isn't free, but the
architecture may need stabilizing instead. `experiments/edge_identity_tokens/
run_wikirfa_regularization_diagnostic.py`, 4 jobs, wiki-rfa only, seeds {42 (reference),
50 (known-bad)}, two variants:
  - `WD10X`: edge_embed_weight_decay x10 (direct L2 shrinkage, no rank change)
  - `REPLACE065`: edge_replace_prob boosted 0.41->0.65 (more train-time identity
    corruption)

**Results (2026-09-16 19:01, complete):**

| variant | seed42 test AUC | seed50 test AUC | vs baseline (seed50=0.7602) |
|---|---|---|---|
| baseline (winning config) | 0.8863 | 0.7602 | -- |
| WD10X (10x identity weight decay) | 0.8833 | **0.5136** | **worse -- collapsed further toward random** |
| REPLACE065 (edge_replace_prob 0.41->0.65) | 0.8891 | 0.7389 | roughly neutral, slightly worse |

**Both failed. WD10X made seed 50 meaningfully worse, not better.** This is itself a
useful negative result: more L2 pressure on the identity table didn't stabilize
training, it hurt -- arguing against a pure "undertrained regularizer" story and toward
an **optimization-instability** story instead (an unlucky large step early in training
landing weights in a bad, poorly-generalizing basin -- consistent with the perfectly
smooth train_loss/train_auc curve seen in section 3, since this failure mode doesn't
require divergence, just an unlucky trajectory). Supporting fact: wiki-rfa's winning
architecture has the **highest learning rate of any of the 6 datasets** (0.0033 vs.
0.0022/0.0016/0.0016/0.00008/0.00039 for alpha/otc/elec/epinions/slashdot) -- a classic
lever for exactly this kind of seed-dependent instability.

## 11. wiki-rfa LR diagnostic -- launched 2026-09-16 19:02 (complete)

Follow-up to section 10's negative regularization result. `run_wikirfa_lr_diagnostic.py`,
4 jobs, same seeds {42, 50}, two LR variants: `LRHALF` (0.00330->0.001649), `LRQUARTER`
(0.00330->0.000824).

**Results (2026-09-16 19:15, complete) -- this is the strongest lead so far:**

| variant | seed42 | seed50 | seed42-seed50 gap |
|---|---|---|---|
| baseline (lr=0.00330) | 0.8863 | 0.7602 | 12.6pp |
| **LRHALF (lr=0.001649)** | 0.8762 | **0.8734** | **0.28pp** |
| LRQUARTER (lr=0.000824) | 0.8584 | 0.8580 | 0.04pp |

Halving the LR nearly eliminates the seed-42-vs-seed-50 gap (12.6pp -> 0.28pp) at a small
cost to seed 42's own peak (-1.01pp). Quartering stabilizes even further (0.04pp gap) but
costs more peak performance (-2.79pp on seed 42). This is a clean, monotonic
speed/stability tradeoff -- strong support for the "optimization instability from a too-
large LR" theory from section 10, and the best lead yet since baseline's actual 10-seed
mean (0.8585) is being dragged down hard by exactly this kind of seed-dependent collapse.

## 12. wiki-rfa LRHALF full 10-seed completion -- launched 2026-09-16 19:16 (complete)

Section 11's LRHALF result is promising enough to check properly, not just eyeball from 2
seeds. `run_wikirfa_lrhalf_multiseed.py` trains the remaining 8 seeds (43-49, 51) under
LRHALF to get a complete, directly-comparable 10-seed mean against baseline's
0.8585 +/- 0.0335. Once complete, compute via the same `eid_seed_auc`-style lookup used by
`eid_result1_table.py` (checkpoints are named `EID_WIKIRFALR_LRHALF_s<seed>`, not the
standard `EID_MULTISEED_*` pattern, so eid_result1_table.py itself won't pick these up
automatically -- pull test AUCs directly from `logs/eid_gap_closer/EID_WIKIRFALR_LRHALF_s*.posthoc.log`).
**Results (2026-09-16 19:42, complete) -- CONFIRMED, real win, not a 2-seed fluke:**

All 10 seeds, `training.lr` halved (0.0033 -> 0.001649), everything else unchanged:

| seed | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 |
|---|---|---|---|---|---|---|---|---|---|---|
| test AUC | 0.8762 | 0.8624 | 0.8702 | 0.8630 | 0.8399 | 0.8630 | 0.8613 | 0.8653 | 0.8734 | 0.8602 |

| config | 10-seed mean | 10-seed std |
|---|---|---|
| baseline (lr=0.0033) | 0.8585 | 0.0335 |
| **LRHALF (lr=0.001649)** | **0.8635** | **0.0094** |

**+0.50pp on the mean, and std shrank 3.6x** (worst seed is now 0.8399, not 0.7604 -- the
catastrophic collapse is gone). Still trails production (0.8914) by -2.79pp (vs baseline's
-3.29pp) -- a real improvement, not a full fix, but the best lever found this session.

**This is a genuine, adoptable candidate fix for wiki-rfa's gap-closer config** -- but
adopting it means updating `logs/eid_gap_closer/state*.json`'s winning entry for wiki-rfa
and **re-deriving every wiki-rfa row already computed in Phase 1** (noablation 10-seed
mean, both ablations, all significance tests) since they all used the old lr=0.0033
config. **Needs explicit sign-off before doing that** -- not applied automatically.

## 13. wiki-rfa LR75 refinement -- launched 2026-09-16 19:44 (complete)

Testing an intermediate point (lr x0.75 = 0.002473) on 4 representative seeds (42 good, 45/46
moderate, 50 originally catastrophic) to see if there's a better stability/peak tradeoff
between baseline (x1.0, mean 0.8585) and LRHALF (x0.5, mean 0.8635) before making an
adoption recommendation. `run_wikirfa_lr75_diagnostic.py`.

**Results (2026-09-16 19:56, complete) -- LR75 is NOT a better middle ground:**

| seed | baseline (x1.0) | LR75 (x0.75) | LRHALF (x0.5) |
|---|---|---|---|
| 42 | 0.8863 | 0.8672 | **0.8762** |
| 45 | 0.8764 | **0.8710** | 0.8630 |
| 46 | 0.8488 | 0.8181 (worse than baseline) | **0.8399** |
| 50 | 0.7602 | 0.8714 | **0.8734** |
| mean (these 4) | 0.8429 | 0.8569 | **0.8631** |

LR75 doesn't cleanly interpolate between baseline and LRHALF -- it's non-monotonic (seed 46
actually drops *below* baseline at LR75, worse than either endpoint), and its 4-seed mean
(0.8569) trails LRHALF's (0.8631) on the identical seed subset. **LRHALF (lr x0.5) is
confirmed as the best point found among {x1.0, x0.75, x0.5, x0.25}** -- no further LR
refinement attempted, this is a clean stopping point for a cheap-diagnostic investigation
(finer-grained search would cross into real hyperparameter-tuning territory).

## Summary / recommendation

**wiki-rfa**: adopt lr x0.5 (0.001649) as a documented, evidence-backed improvement to the
gap-closer's winning config -- confirmed on a full 10-seed sweep (mean 0.8585->0.8635,
std 0.0335->0.0094, catastrophic seed eliminated), tested against 3 alternative levers
(rank reduction, 2 regularization variants, an intermediate LR point) that either did
nothing or actively hurt. Still trails production by -2.79pp, so this is a genuine partial
fix, not a full close of the gap. **Needs your sign-off before actually adopting** --
doing so means updating `logs/eid_gap_closer/state*.json`'s wiki-rfa winning entry and
re-deriving every wiki-rfa number already computed in Phase 1 (noablation mean,
both ablations, all significance tests), since all of that used the old lr=0.0033.

**wiki-elec**: unresolved. (An earlier version of this line claimed rank=0 "disabled"
identity and ruled identity out as the driver -- wrong, rank=0 is full-width identity; see the
section 8 correction.) Its -2.06pp gap to production has no supported explanation yet.

## 14. Root-cause hypothesis (2026-09-24): EID's configs drifted from production's

Compared each dataset's EID gap-closer winner against production's own validated
`configs/<ds>.yaml`. EID's configs came from an independent single-seed Optuna search and
diverge from production in ways **production's own sweeps had already shown to be worse**:

| dataset | EID winner vs. production config | what production already knew |
|---|---|---|
| wiki-elec | same arch/lr, but **8x budget** (829,512) vs. 1.5x (155,534); node_replace 0.49 vs 0.2 | E26: wiki-elec over-saturates past 1.5x -- production at 8x = **0.8937** vs. 0.9036 at 1.5x |
| wiki-rfa | **lr 0.0033 (2x production's 0.0016)**, hidden 128/4 layers vs 64/5, 3x budget vs 1.5x | LRHALF (section 12) = returning to production's LR -- the single best fix found |
| slashdot090221 | **lr 3x production's**, hidden 384/5 layers vs 256/4, **50 epochs vs production's 75** | production raised to 75 because val was still climbing at 50 |
| bitcoin-alpha | close to production (same arch), 8x budget vs 5x, lr 0.0022 vs 0.0016 | -- |
| bitcoin-otc | same arch/lr as production, 8x budget vs 5x | -- |
| epinions | much larger model (hidden 256, emb 384 vs 32/32), lr 8e-5 vs 0.0029, 1.5x vs 1x | EID wins here -- the one place the divergent config paid off |

> **CORRECTION (2026-09-24): the budget part of this section is wrong and is retracted** (user
> objection, confirmed by EID's own sweep in `state_v2.json`). EID has per-edge parameters that
> only train on that edge's own appearances, so more walks help it, unlike production. EID's
> wiki-elec test AUC by budget: 1x 0.845, 1.5x 0.869, 3x 0.871, 5x 0.886, 8x **0.889**.
> Production's is the opposite shape: it peaks at 1.5x (0.904) and falls to 0.894 at 8x.
> Moving EID to production's 1.5x would cost it ~2pp, not recover it. The "rank=0 = identity
> off" claim used here was also wrong (section 8 correction). What still stands from this
> section: wiki-rfa's LR (2x production's; halving it helped over 10 seeds), slashdot's
> LR/epochs, and the methodology point that EID's non-identity settings differ from production's.
>
> Notable from the sweep: **at the same 1.5x budget, EID trails production by ~3.5pp on
> wiki-elec (0.869 vs 0.904)**, and it only partly closes that with 5x more walks. Identity
> rows are data-hungry, and even well-fed they don't reach production's best on the wikis.

## 15. Revised hypothesis (2026-09-24): the target's own identity is a train-only shortcut

By design (MECHANISM.md section 3), a target edge keeps its identity token visible and only
its sign is hidden. Production shows `<MASK>` there instead. Under dynamic masking every TRAIN
edge is the prediction target in some epochs. So its identity row gets trained, directly, to
predict its own sign, which is a per-edge label lookup. TEST edges are never targets during
training, so their rows never learn their own sign. **The target token means something different
at train time (it carries the label) than at test time (it doesn't).** Predictions of what we'd
see if this is the driver:
- a large train/val gap where EID trails most -- **observed** (section 4);
- the `edge_replace` regularizer being needed at all -- its own docstring names this exact failure
  ("I've memorized that edge #17360 specifically tends to be positive");
- hiding context SIGNS barely mattering once identity is visible, because train edges' identity
  rows already encode their signs -- **observed** (`mask_context_sign_only`, section 9);
- identity needing many walks per edge to pay off -- **observed** (EID's budget curve, section 14).

Not yet tested. Two controls would settle it, and neither exists in the code today:
1. **Identity-off mode**: every edge position maps to one shared edge row + the sign embedding,
   target hidden. Mathematically the same parameterization as production's two sign tokens, so
   it should reproduce production's numbers. It's the missing baseline for everything above, and
   the special case that lets EID's code replace production's.
2. **Context-identity-only mode**: the target renders hidden (like production), context edges
   keep their identity. This removes the train/test mismatch at the target and keeps what the
   `mask_context_*` ablations measure.

**Prior evidence found 2026-09-24 (single seed, bitcoin-alpha, 2026-09-08, identical config
otherwise: rank 20, edge_replace 0.15, reveal on, 300k walks):** target identity visible
(`EID_REVEAL_HOLDOUT_300000`) = 0.9175 test AUC; target identity hidden
(`EID_ABL_MASKTARGET_300000`, `model.mask_target_identity=true`, already implemented) =
**0.9288**, +1.1pp and above production's 0.9134 mean. Never followed up. Control 1
(`model.eid_identity_off`) was added 2026-09-24 (`lit_model.py` + `eid_posthoc.py`); both
controls are running via `run_identity_controls.py` (results ->
`logs/eid_gap_closer/identity_controls_results.csv`).

## 16. Regularization inventory (audit 2026-09-24)

**Already searched/tested** (Optuna space in `optuna_eid.py:184-199`; stage-1 search was only
24 trials total, single seed): dropout (0.1-0.6), node replacement R (prob + UNK ratio), edge-
identity replacement (prob 0-0.5, UNK ratio; UNK ratio 1.0 = identity dropout), L2 on the
identity table (`edge_embed_weight_decay`, 1e-4-1.0 log), identity capacity (`edge_embed_rank`
0-32, `node_embed_dim`), identity-as-residual-on-endpoints (`edge_residual_baseline`), LR. Fixed
at config values, never searched: global weight decay, gradient clipping, early stopping. Plus
this session's single-knob diagnostics (WD x10, replace 0.65, LR x0.25/0.5/0.75).

**Never tried anywhere in the code** (grep-confirmed): L1 / elastic net / group lasso on
identity rows, label smoothing, weight EMA / SWA, max-norm on identity rows, noise on identity
embeddings, zero-init of the identity projection, frequency-aware identity regularization
(rare edges), LR warmup, separate LR for the identity table, a multi-seed tuning objective,
and hiding the target's identity (`mask_target_identity` exists, but was run once and never
adopted).

**L2 is effectively off everywhere (computed 2026-09-24).** AdamW's decoupled decay shrinks a
weight by a factor exp(-sum_t lr_t * wd) over training, independent of the gradient. Using each
winner's lr, CosineAnnealingLR, steps/epoch and the epochs actually run (seed 43):

| dataset | identity-table decay | total identity shrink | global decay | total global shrink |
|---|---|---|---|---|
| bitcoin-alpha | 2.8e-4 | 0.30% | 1.0e-4 | 0.11% |
| bitcoin-otc | 3.0e-4 | 0.34% | 1.0e-4 | 0.11% |
| wiki-elec | 3.0e-4 | 0.85% | 3.0e-5 | 0.08% |
| wiki-rfa | 1.1e-4 | 0.48% | 3.0e-5 | 0.13% |
| epinions | 2.5e-3 | 0.63% | 2.9e-4 | 0.07% |
| slashdot090221 | 1.7e-3 | 2.47% | 4.5e-4 | 0.64% |

The search range (1e-4 to 1.0) could reach strong values, but seed-42 validation picked the
bottom of it on every dataset. The global decay (`training.weight_decay`, every parameter except
the identity table) was never searched at all -- inherited from production's configs, where it is
equally negligible. So dropout + token replacement are doing all the regularization.

**Correction to section 15 (user, verified in code):** the target's identity is NOT always visible
during training. `_maybe_apply_edge_identity_replacement` treats every edge-identity token as a
candidate (`x >= old_vocab_size`), targets included. So at train time the target shows its true
identity with probability 1 - edge_replace_prob (wiki-elec 0.50, wiki-rfa 0.59, bitcoin-alpha 0.65,
bitcoin-otc 0.50, epinions 0.80, slashdot 0.32), and a wrong or UNK identity otherwise. At
val/test (no replacement) it's always visible. The memorization route is weakened, not closed. A
decoupled target-only replacement probability is the natural knob between "current" and
`mask_target_identity` (always hidden).

## Open questions / next steps (not yet done)

- ~~Proposed: production-anchored EID~~ -- WITHDRAWN 2026-09-24: it assumed rank=0 turns
  identity off (false) and that production's budget suits EID (false). See section 15.
- wiki-rfa rank=4's collapse to ~random is unexplained (likely the same lr=0.0033 instability;
  would be moot under production's LR).
- Causal confirmation of the attention finding (a `mask_node_tokens` ablation on EID) not done.
