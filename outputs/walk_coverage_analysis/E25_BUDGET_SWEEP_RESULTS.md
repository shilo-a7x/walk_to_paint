# E25 — edge_cover budget re-sweep (bitcoin-alpha, bitcoin-otc, epinions, slashdot090221)

Phase 4 of `plan-a-fix-for-glimmering-panda.md`. All runs use `walk_strategy=edge_cover`
(k fixed at 1, provable zero-duplication guarantee — see the E24 addendum and
`CLAUDE.md`'s sampler section). Grid per dataset: `{|E| (floor), 1.5×|E|, 3×|E|, 5×|E|}`,
plus the dataset's old production budget as an extra reference point for bitcoin-alpha
and slashdot090221 (per the original plan scope — these two were "never tested past
current budget, AUC still rising at the top of the old grid"). Full attention, no
hardness miner, `func_logit_power` posthoc aggregator, seed=42.

**Correction note:** the first posthoc pass for 8 of these 18 runs (and, separately,
all 8 of Phase 3's runs) used only `dataset.name=<ds>` and silently evaluated against
the *default* production cache instead of the actual training budget — the same bug
documented in `E16_NOHARD_RESULTS.md` (2026-07-07). Caught and corrected before this
doc was written (`CLAUDE.md`'s posthoc section now states the rule explicitly). All
numbers below are from the corrected, matching-cache posthoc pass.

## Full grid — every point, not just winners

| dataset | \|E\| | budget label | num_walks | mult | test AUC | train AUC | run dir |
|---|---|---|---|---|---|---|---|
| bitcoin-alpha | 24,186 | floor | 24,186 | 1.0× | 0.8919 | 0.8904 | `E25_BUDGET_alpha_floor_20260717-162611` |
| bitcoin-alpha | 24,186 | 1.5× | 36,279 | 1.5× | 0.9084 | 0.8918 | `E25_BUDGET_alpha_p1_5x_20260717-162611` |
| bitcoin-alpha | 24,186 | 3× | 72,558 | 3.0× | 0.9184 | 0.9071 | `E25_BUDGET_alpha_x3_20260717-162611` |
| bitcoin-alpha | 24,186 | 5× | 120,930 | 5.0× | 0.9219 | 0.9247 | `E25_BUDGET_alpha_x5_20260717-162755` |
| bitcoin-alpha | 24,186 | old-ref | 5,000,000 | 206.8× | **0.9375** | 0.9356 | `E25_BUDGET_alpha_oldref_20260717-162828` |
| bitcoin-otc | 35,592 | floor | 35,592 | 1.0× | 0.9042 | 0.8933 | `E25_BUDGET_otc_floor_20260717-162957` |
| bitcoin-otc | 35,592 | 1.5× | 53,388 | 1.5× | 0.9117 | 0.8997 | `E25_BUDGET_otc_p1_5x_20260717-163328` |
| bitcoin-otc | 35,592 | 3× | 106,776 | 3.0× | 0.9242 | 0.9137 | `E25_BUDGET_otc_x3_20260717-201634` |
| bitcoin-otc | 35,592 | 5× | 177,960 | 5.0× | **0.9311** | 0.9257 | `E25_BUDGET_otc_x5_20260717-163145` |
| epinions | 840,799 | floor | 840,799 | 1.0× | 0.9527 | 0.9525 | `E25_BUDGET_epinions_floor_20260717-163552` |
| epinions | 840,799 | 1.5× | 1,261,199 | 1.5× | 0.9522 | 0.9529 | `E25_BUDGET_epinions_p1_5x_20260717-202052` |
| epinions | 840,799 | 3× | 2,522,397 | 3.0× | 0.9551 | 0.9568 | `E25_BUDGET_epinions_x3_20260717-163755` |
| epinions | 840,799 | 5× | 4,203,995 | 5.0× | **0.9561** | 0.9566 | `E25_BUDGET_epinions_x5_20260717-165530` |
| slashdot090221 | 549,202 | floor | 549,202 | 1.0× | 0.8807 | 0.8811 | `E25_BUDGET_slashdot_floor_20260717-205047` |
| slashdot090221 | 549,202 | 1.5× | 823,803 | 1.5× | 0.8928 | 0.8920 | `E25_BUDGET_slashdot_p1_5x_20260717-174813` |
| slashdot090221 | 549,202 | 3× | 1,647,606 | 3.0× | **0.9007** | 0.9004 | `E25_BUDGET_slashdot_x3_20260717-190227` |
| slashdot090221 | 549,202 | 5× | 2,746,010 | 5.0× | 0.8996 | 0.9025 | `E25_BUDGET_slashdot_x5_20260717-210303` |
| slashdot090221 | 549,202 | old-ref | 5,000,000 | 9.1× | 0.9004 | 0.9025 | `E25_BUDGET_slashdot_oldref_20260717-180852` |

(Bold = best point per dataset within the grid actually tested.)

## Per-dataset shape

- **bitcoin-alpha**: monotonic, still climbing hard all the way to the old-ref point
  — floor→5× is +3.00pp, and 5×→old-ref (120,930→5,000,000 walks) is *another*
  +1.56pp. No plateau found anywhere in this grid. This dataset genuinely wants a
  large budget; unlike the k-sweep in Phase 3, more distinct walks keep paying off
  here.
- **bitcoin-otc**: also still climbing at the top of the tested grid (floor→5× is
  +2.69pp, no sign of flattening at 5×=177,960). Not tested up to its own old
  production budget (2,000,000) — out of the original plan's scope for this dataset,
  but given the still-rising curve, this is a real gap, not a settled floor.
- **epinions**: mostly flat/noisy — floor→1.5× is actually *slightly negative*
  (−0.05pp), then floor→3× and floor→5× recover to +0.24pp / +0.34pp. This is
  consistent with Phase 3's finding that epinions' extra-context effect is weak and
  somewhat noisy at this seed count (n=1).
- **slashdot090221**: strong gains up to 3× (floor→3× is +2.00pp), then flattens —
  5× is slightly *below* 3× (−0.11pp) and the old-ref point (5,000,000 walks, 9.1×
  the floor) essentially ties 3× (0.9004 vs 0.9007, within noise). **The old
  production budget buys nothing over 3×floor (1,647,606) here** — a real,
  actionable cost-reduction candidate.

## Does the new deduplicated sampler beat the old (duplicated) one at the same budget?

Two of the six grid points allow a clean, matched-budget comparison against this
project's already-reported SOTA (`CLAUDE.md`'s "Ours (no-H, E16)" column, same
architecture/training regime, only the sampler's deduplication differs):

| dataset | budget | old `k_cover` (documented SOTA, duplicated) | new `edge_cover` (this sweep, zero-dup) | delta |
|---|---|---|---|---|
| bitcoin-alpha | 5,000,000 | 0.9212 | **0.9375** | **+1.63pp** |
| slashdot090221 | 5,000,000 | 0.9014 | 0.9004 | −0.10pp |

**bitcoin-alpha shows a large, clean win from fixing sampler duplication at an
identical budget** — same 5M walks, same architecture, same everything else, +1.63pp
just from the walks being genuinely distinct instead of ~duplicated. slashdot090221
shows no difference (within noise) at the same comparison.

epinions and bitcoin-otc don't have an exact matched-budget point in this grid
(epinions' old budget is 3,000,000, between this grid's 3× and 5× points; otc's old
budget is 2,000,000, well above this grid's 5× ceiling of 177,960) — so no clean
comparison is available for those two without an additional run. Worth noting:
epinions' *closest* new point (5×, 4,203,995 walks — more than the old 3M budget) is
0.9561, marginally **below** the old SOTA's 0.9572 (−0.11pp) despite using a larger
budget — inconclusive on its own (different budgets, single seed), but it means
bitcoin-alpha's result should not be assumed to generalize to every dataset without
checking.

**This finding (bitcoin-alpha) is promising but not yet confirmed** — it's a
single-seed comparison, and this project's own training logs have shown val-AUC
epoch-to-epoch swings of ±1–2pp (`E16_NOHARD_RESULTS.md` "Full attention... within
normal epoch-to-epoch noise" caveat) — 1.63pp is above that band but not enormously
so. A repeat-seed check would be needed before treating this as a settled result.

## Recommendation (proposal only — requires explicit sign-off)

Applying the CLAUDE.md diminishing-returns rule (prefer cheaper unless the pricier
point gains ≥0.25pp test AUC):

- **bitcoin-alpha**: every step up gains well above the bar (+1.65pp to +3.00pp
  between adjacent points) — **no cheap pick applies here; the data argues for the
  most expensive point tested** (5,000,000, old production budget), which is also
  the only point clearing the previously-documented SOTA. Not fully explored — still
  rising at the top.
- **bitcoin-otc**: same pattern — floor→5× never dips below the bar (all deltas
  >0.7pp). **Grid doesn't reach a plateau; recommend testing higher (at least up to
  the old 2,000,000 production budget) before picking a final budget.**
- **epinions**: floor→3× is +0.24pp (just under the bar), floor→5× is +0.34pp (just
  over) — borderline, noisy, non-monotonic. **Tentative pick: 3× (2,522,397)** as the
  cheapest point that isn't clearly worse than the best, but this is a weak signal,
  not a confident one.
- **slashdot090221**: floor→3× is +2.00pp (clears the bar easily), 3×→5× and
  3×→old-ref are both slightly *negative*. **Clear pick: 3× (1,647,606)** — matches
  or beats every more expensive point tested, including the old 5,000,000 production
  budget. This is also a **cost-reduction opportunity**: the current production
  config uses 5,000,000 walks for no measured benefit over 1,647,606 (a 3× reduction
  in walk-generation/training cost).

**These are proposals, not adopted decisions** — need your explicit sign-off before
touching `configs/<dataset>.yaml`, and per the diminishing-returns rule's own
exception clause, the headline SOTA table itself is exempt from the cheap-lean (i.e.
if bitcoin-alpha's finding holds up, using its most expensive tested point for the
production SOTA number would be consistent with existing project policy even though
it's the pricier choice).

## Caveats

- Single seed (42) per grid point — no repeat-seed noise estimate anywhere in this
  sweep. Given the observed epoch-to-epoch noise band noted above, none of the
  deltas here should be treated as fully confirmed without a repeat-seed check,
  especially the epinions non-monotonicity and the bitcoin-alpha old-sampler
  comparison.
- bitcoin-otc and epinions were not tested up to their own old production budgets
  (2,000,000 and 3,000,000 respectively) — both showed no plateau within the tested
  range, so their "final" budget question is still open, not just their cheap-end
  question.
- wiki-elec and wiki-rfa remain explicitly out of scope (already documented as
  over-saturating past their current small budget — see the original plan).
- This sweep only varies `num_walks` at `walk_strategy=edge_cover` (k fixed at 1,
  per Phase 3's sign-off) — it does not re-open the k question.

## Next steps (gated on your sign-off, not started)

1. Sign off (or revise) the per-dataset budget proposals above.
2. Consider extending bitcoin-alpha and bitcoin-otc's grids upward (bitcoin-otc to
   at least 2,000,000; bitcoin-alpha's curve is still rising even at 5,000,000, so a
   higher point may be worth one more data point) before finalizing either.
3. If the bitcoin-alpha win holds up under a repeat-seed check, this is a strong
   argument for the full 6-dataset retrain flagged throughout the plan as the
   eventual payoff of fixing sampler duplication — not just a documentation
   exercise.
