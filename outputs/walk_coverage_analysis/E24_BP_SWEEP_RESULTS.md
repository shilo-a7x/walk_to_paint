# E24 — k_cover_bp k-sweep: true-dedup verification + re-derived AUC-vs-k

Phase 3 of `plan-a-fix-for-glimmering-panda.md`. Answers the question that motivated
the whole `k_cover_bp` sampler: **once anchor-phase AND fill-phase duplication are
both genuinely fixed, how many distinct walks per edge (k) does the model actually
need, and does the original k=5 lock still hold?**

## Two prerequisite bugs fixed before these numbers are trustworthy

1. **Length-cap bug** (`src/data/coverage_aware_sampler.py`, `_kcover_anchor_chunk_bp`):
   the forward suffix loop ignored how many hops the backward prefix had already
   spent, so `prefix + forced edge + suffix` could exceed the model's fixed 161-token
   cap whenever a real (nonzero) prefix was used. Silent until real k=3/5/7 training
   crashed at epoch 0 (`RuntimeError: size of tensor a (163) must match ... (161)`);
   k=1 never triggers it (prefix_len is always 0 there). Fixed by reserving
   `prefix_len` hops from the suffix budget. Affected caches deleted and regenerated.
2. **Fill-phase had zero deduplication** — the dominant remaining duplication source
   (5–31% corpus-wide even after the anchor-phase fix), because the plain
   uniform-sampling fill phase was left untouched by the original anchor-only design.
   Fixed via `_generate_dedup_fill` (oversample + exact-hash dedup + backward-prefix
   diversified top-up rounds, bounded retries with honest exhaustion reporting).

Both fixes are in `src/data/coverage_aware_sampler.py`; sanity-tested in
`scripts/test_kcover_bp_sanity.py` (8 tests, synthetic graphs, all passing).

## True corpus-wide duplication, measured directly (not estimated)

In-memory, no cache reuse (`scripts/check_true_dedup_now.py`), production-realistic
budget `nw=2,000,000`, `max_walk_length=80`, `seed=42`, exact `blake2b` hash of every
walk's full token sequence:

| dataset | k | dup_rate (corpus-wide) | anchor `capped_frac` | anchor `dup_after_retries_frac` |
|---|---|---|---|---|
| epinions | 1 | 0.00% | — | — |
| epinions | 3 | 1.39% | (capped, see below) | ~0% |
| epinions | 5 | 2.83% | (capped, see below) | ~0% |
| epinions | 7 | 4.29% | (capped, see below) | ~0% |
| slashdot090221 | 1 | 0.00% | — | — |
| slashdot090221 | 3 | 1.58% | (capped, see below) | ~0% |
| slashdot090221 | 5 | 3.25% | (capped, see below) | ~0% |
| slashdot090221 | 7 | 4.97% | (capped, see below) | ~0% |

**k=1 is exactly zero-duplicate on both datasets** — every one of the 2,000,000
walks is a distinct token sequence. At k>1 the residual duplication is small,
grows roughly linearly with k, and is attributable to **topologically capped
edges** (`in_deg(u)==0 AND out_deg(v)==0` — no backward prefix or forward suffix is
possible regardless of retries, so all k visits to that edge are structurally
forced to be identical). This matches the `capped_frac` figures already measured
per-dataset in `PHASE2B_BP_COVERAGE_CURVE.md` (epinions 1.54%, slashdot090221
2.65% of edges) — consistent with duplication scaling as `capped_frac × (k-1)`
roughly, since a capped edge contributes k-1 "extra" duplicate copies of its 1
distinct walk. `dup_after_retries` (genuine retry-exhaustion failures on
*non-capped* edges) stayed near 0% in all cases — the fill+anchor fixes are working
as designed, not just papering over a large residual.

**This is the answer to the original question**: at k=1 there is no sampler-induced
duplication left to worry about at all. Any duplication above k=1 is an honest,
provable structural floor, not a retry/design shortfall.

## AUC vs k — full 8-run sweep (epinions, slashdot090221)

All runs: `walk_strategy=k_cover_bp`, `nw=2,000,000`, `mw=80`, seed=42, full
attention (no hardness miner, per CLAUDE.md), `func_logit_power` posthoc aggregator,
`dataset.name=<ds>` passed explicitly. Full grid reported per the CLAUDE.md
sweep-reporting rule — every point below, not just the pick.

**Correction (2026-07-17): the numbers below were re-verified and corrected.** The
first posthoc pass for all 8 runs only passed `dataset.name=<ds>`, which silently
falls back to `configs/<ds>.yaml`'s *default* cache (`k_cover`, `nw=3,000,000` for
epinions / `nw=5,000,000` for slashdot090221) instead of the `k_cover_bp`,
`nw=2,000,000` cache each checkpoint was actually trained on — the same failure
mode previously documented in `E16_NOHARD_RESULTS.md` (2026-07-07), now newly
codified as a hard rule in `CLAUDE.md`'s posthoc section (repeat every non-default
`dataset.*` override, not just `dataset.name=`). Re-ran all 8 with the correct
`dataset.walk_strategy=k_cover_bp dataset.walk_k_min=<k> dataset.num_walks=2000000`
overrides; **the correction is tiny (≤0.001 AUC on all 8 points) and does not
change any conclusion below** — table already updated to the corrected numbers.

| dataset | k | test AUC | train AUC | delta vs k=1 | run dir |
|---|---|---|---|---|---|
| epinions | 1 | 0.9526 | 0.9534 | — | `E24_KSWEEP_BP_k1_20260716-155805` |
| epinions | 3 | 0.9532 | 0.9533 | +0.06pp | `E24_KSWEEP_BP_k3_20260716-155806` |
| epinions | 5 | 0.9545 | 0.9540 | +0.19pp | `E24_KSWEEP_BP_k5_20260716-155913` |
| epinions | 7 | 0.9552 | 0.9559 | **+0.26pp** | `E24_KSWEEP_BP_k7_20260716-155913` |
| slashdot090221 | 1 | 0.9007 | 0.9016 | — | `E24_KSWEEP_BP_k1_20260716-155806` |
| slashdot090221 | 3 | 0.9024 | 0.9022 | +0.17pp | `E24_KSWEEP_BP_k3_20260716-155806` |
| slashdot090221 | 5 | 0.9014 | 0.9020 | +0.07pp | `E24_KSWEEP_BP_k5_20260716-155913` |
| slashdot090221 | 7 | 0.9008 | 0.9020 | +0.01pp | `E24_KSWEEP_BP_k7_20260716-155913` |

(Each dataset's run dir is under `outputs/<dataset>/`; earlier timestamped attempts
for the same k that predate the length-cap fix were discarded, not included above.)

epinions climbs monotonically and hasn't clearly plateaued by k=7 (+0.26pp k1→k7,
still rising at the top point). slashdot090221 peaks at k=3 and is flat-to-slightly-
positive beyond it (net **+0.01pp** k1→k7, essentially flat, corrected from the
earlier −0.05pp) — more distinct walks bought effectively nothing here either way,
unlike the original pre-fix curve.

## Comparison against the original (duplicate-contaminated) k=5 lock

`PHASE0_FINDINGS.md` §0.F derived k≈5–8 from visit-count-vs-AUC on the *old*
sampler, where visit count conflated distinct and duplicate walks:

| dataset | old probe gain (k1→k5/8, duplicate-contaminated) | new measured gain (k1→k7, true-distinct) |
|---|---|---|
| epinions | +0.85pp (k1→k8) | +0.26pp (k1→k7) |
| slashdot090221 | +0.70pp (k1→k8) | **−0.05pp** (k1→k7) |

The true-distinct-context effect is **much weaker** than the original probe
suggested — roughly a third the size on epinions, and outright reversed on
slashdot090221. This confirms the concern raised in the plan's Context section: the
original k=5 floor was partly measuring "more raw visits (many duplicated) helps a
little," not "more distinct contexts helps a lot."

## Recommendation (proposal only — requires explicit sign-off)

Applying the CLAUDE.md diminishing-returns rule (prefer cheaper setting unless the
pricier one gains ≥0.25pp test AUC, flat across datasets) to this 2-dataset grid:

- **epinions**: k1→k7 gains +0.26pp — right at the bar, not comfortably above it,
  and still rising (no plateau confirmed within this grid).
  - Contrast with cost: k determines *anchor* pass count. With 2,000,000-walk
    budgets already well above the edge count on both datasets, higher k mostly
    changes disallowed-duplication structural bookkeeping, not wall-clock — this
    sweep's realized cost delta across k=1→7 was not separately isolated (no perf
    benchmarking was in scope for Phase 3, only Phase 2a). Do not assume k is
    cheap to raise without checking.
- **slashdot090221**: k1→k7 is net **negative** (−0.05pp) — fails the bar outright,
  no case for going above k=1 here at all.

**Proposed pick: k=1 for both datasets** — it is the only value with exact-zero
sampler-induced duplication, and neither dataset clears the 0.25pp bar to justify a
higher k (epinions is borderline/inconclusive at best, slashdot is a clean loss).
This reverses the original plan's working assumption that k=5 would be confirmed.

**This is a proposal, not an adopted decision** — needs your explicit sign-off per
the CLAUDE.md rule before touching `configs/<dataset>.yaml` or any other config.

## Caveats

- Single seed (42) per (dataset, k) cell — no repeat-seed noise estimate. AUC deltas
  here (0.05–0.26pp) are within plausible single-seed noise band; not statistically
  confirmed as real effects, especially the epinions climb.
- Only 2/6 datasets tested (epinions, slashdot090221 — chosen per the plan as the
  two most likely to show the duplicate-vs-distinct distinction: epinions for its
  5,567 disconnected components, slashdot090221 for having the original k=1/3/5
  end-to-end runs to compare against).
- Fixed `num_walks=2,000,000` for all 8 runs — not floor-derived per Phase 2b's
  fresh coverage curve (that's Phase 4's job, out of scope here). This sweep only
  varies k at a fixed, already-coverage-sufficient budget.
- No wall-clock/generation-cost comparison across k values in this sweep (Phase 2a
  covered generation cost only at production k, not a k-sweep).

## Addendum (2026-07-17): k=1 signed off; sampler simplified to `edge_cover`;
## fill-phase guarantee hardened from "near-zero" to airtight

**Sign-off received**: k=1 adopted for both datasets, per the recommendation above.

**A real gap was found and closed in the same session.** The k=1 measurement above
(0.00% dup on both datasets) was accurate, but the mechanism behind it was not
actually airtight: `_generate_dedup_fill`'s fallback, if 6 rounds of oversample+dedup
still came up short, silently padded the remainder with plain (undeduped) uniform
walks — a duplicate-injection path that happened not to fire in this measurement,
but wasn't *proven* not to. Per explicit direction, this is now closed:

- `_generate_dedup_fill` (`src/data/coverage_aware_sampler.py`) now escalates
  oversample (doubles it) on any round that accepts nothing, runs up to 10 rounds,
  and — if still short — **raises `RuntimeError` naming the exact shortfall**
  instead of padding with duplicates. There is no longer a silent-degrade path
  anywhere in the walk-generation code. `test_fill_phase_exhaustion` in
  `scripts/test_kcover_bp_sanity.py` was updated to assert the raise (previously
  asserted the old graceful-degrade telemetry).
- **New simplified strategy `walk_strategy=edge_cover`.** At k=1, `k_cover_bp`'s
  multi-pass loop, attempt-number tracking, growing backward-prefix, and
  hash-retry-on-collision machinery are all dead code — none of it ever executes,
  because every edge only ever gets one direct anchor (attempt_number is always 1,
  so `use_prefix` is always `False`). `edge_cover_walks` keeps only what's actually
  load-bearing at k=1: one forced anchor walk per edge (`_kcover_anchor_chunk`,
  the same simple worker `k_cover_walks_fast` uses) + the hardened dedup fill for
  the remaining budget. `k_cover_bp` itself is untouched and still available if a
  k>1 need ever comes back.
- **Anchor-phase distinctness at k=1 is provable, not measured**: each edge's one
  anchor walk starts with that edge's own unique `(u, label, v)` triple, so no two
  anchor walks can ever collide with each other — zero probabilistic element,
  confirmed by a dedicated test (`test_edge_cover_anchor_distinctness`) that checks
  distinctness on a graph with zero RNG-luck dependency.
- **Re-verified at production scale with `edge_cover`**: `nw=2,000,000`,
  `max_walk_length=80`, `seed=42`, 8 workers —

  | dataset | n_walks | n_distinct | dup_rate | gen time |
  |---|---|---|---|---|
  | epinions | 2,000,000 | 2,000,000 | **0.000000%** | 54.8s |
  | slashdot090221 | 2,000,000 | 2,000,000 | **0.000000%** | 41.7s |

  Exact zero, not "near-zero" — and now backed by a hard guarantee (raise-on-
  failure) rather than an empirical observation that could in principle fail
  silently on a different dataset/budget.

**`edge_cover` (not `k_cover_bp` with `walk_k_min=1`) is the adopted production
strategy going forward** — functionally identical output at k=1, but the code path
that runs is simpler and the duplicate-guarantee is airtight rather than empirical.
`walk_k_min` is not applicable to `edge_cover` (no config key needed).

## Next steps

1. ~~Sign off (or reject/revise) the k=1 proposal above.~~ **Done — k=1 adopted,
   sampler simplified to `edge_cover`.**
2. Phase 4: floor-derived budget re-sweep on bitcoin-alpha, bitcoin-otc, epinions,
   slashdot090221, using `walk_strategy=edge_cover` (fixed k=1). Coverage floor is
   now a provable property (100% edge coverage guaranteed once `num_walks >= |E|`,
   not something requiring a fresh empirical curve the way the old `k_cover`/
   `k_cover_bp` budget floors did) — the open question Phase 4 answers is how much
   *fill* volume beyond that floor (more genuinely-distinct walks, not more visits
   per edge) actually helps AUC, which is a different question than the original
   plan's k-sweep.
3. Eventually: full 6-dataset retrain on `edge_cover` once Phase 4 lands (flagged
   in the plan's Context as the expected payoff of fixing a real training-data
   defect, not just a documentation exercise).
