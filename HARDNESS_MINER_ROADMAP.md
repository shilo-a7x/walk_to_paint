# Hardness miner — roadmap (ranked by time vs. expected gain)

Living document. Update status inline as items complete; don't create a second copy.
Context/history: `plan-hardness-miner.md` (full investigation log), full Q3/Q5 screen
results: `outputs/walk_coverage_analysis/HARDNESS_MINER_Q3_Q5_SCREEN_RESULTS.md`.
Entropy formulas/directionality: `LEAD4C_EQUATIONS.md`, `LEAD4C_ASYMMETRY.md`.

## Reframing (2026-07-12) — the central question changed

Original framing: "find the best possible hardness map." **That's now mostly answered** —
entropy (role-aware `H_out`/`H_in`, computed from TRAIN+MASK only) is the best signal found,
by a wide margin over the old learned miner, verified three independent ways (Lead4/4a,
Lead4c, and this thread's own role-matched correlation check).

**New central question: is the whole miner mechanism worth keeping at all for the
recommended production path (full attention)?** Early evidence from retraining full
attention on the *best* map found this session (role-aware entropy, E22) is flat-to-mixed
on 4/5 datasets so far (wiki-elec flat, epinions +0.001, bitcoin-otc −0.002, slashdot flat;
bitcoin-alpha + wiki-rfa still running). If that holds, it's a stronger conclusion than Q1's
original "old map doesn't help full attention" — it suggests **H is architecturally
irrelevant to full attention, independent of map quality.** That would mean LocalAttn4+H's
complexity (miner or entropy-map pipeline, plus local attention's own overhead) needs its
own justification: either (a) it beats full-attention-no-H by enough margin to be worth it,
(b) a smarter reweighting *formula* unlocks value a better map alone doesn't, or (c) a
future short-walk main-model regime makes H load-bearing in a way it currently isn't.
Absent one of those, "full attention, no H" remains the simplest, most defensible story —
particularly relevant for how this gets written up.

## Roadmap (reprioritized 2026-07-12, second pass)

LocalAttn4 retrains (old item 6) are **off the table for now** — full attention is the
production path and the question is whether *it* needs H at all; LocalAttn4+H only matters
if a formula tweak or short-walk regime makes it self-justifying, so it's pointless to spend
GPU-hours there before item 7 has an answer. Old item 8 (twohop entropy terms) is **dropped
entirely** — map quality was never the bottleneck once entropy beat the learned miner, and 5/6
E22 results below show the map isn't the limiting factor anyway.

| # | Task | Time | Expected gain | Status |
|---|---|---|---|---|
| 1 | Recompute `src_out`+`tgt_in` entropy from TRAIN+MASK only (leakage-safe) | ~15 min | prerequisite | ✅ done |
| 2 | Predictive-validity screen: entropy vs. every other candidate, both attention variants | ~15 min | free, sets the ceiling | ✅ done — entropy wins clearly everywhere |
| 3 | Can we skip the miner entirely (entropy alone vs. every miner-involving blend)? | ~5 min | high — simplifies pipeline | ✅ done — yes |
| 4 | Build the winning candidate map | ~5 min, no training | prerequisite | ✅ done, then corrected to role-aware (see below) |
| 4b | Role-aware correction: verified "both-sided nodes only" was wrong (checked Lead4/4a + Lead4c's actual code — neither requires it), fixed `lit_model.py` to support separate source/target maps | ~1h (code + verify) | real — up to 8× stronger role-matched correlation, 1.3-4.9× more node coverage | ✅ done |
| 5 | Retrain full attention on the role-aware entropy map (E22), all 6 datasets | ~3h wall-clock | the decisive experiment for the reframed question — does H help full attention AT ALL with the best map available | 🔄 5/6 done, all flat-to-negative (see table below); bitcoin-alpha still running on GPU0 |
| 6 | LocalAttn4 retrain on the role-aware entropy map | ~3h wall-clock | shelved for now per user instruction — revisit once #7 identifies a formula (if any) worth testing there | ⬜ **shelved, not now** |
| 7 | Reweighting-formula pilot on full attention: `hardness_combine=max` (endpoint-max instead of mean) and `hardness_power=2.0` (convex scaling), implemented as `model.hardness_combine`/`model.hardness_power` in `lit_model.py` | pilot ~1-2h (wiki-elec+wiki-rfa fast, epinions slower), full sweep more | **answered** — no formula rescues H; bitcoin-alpha's own formula-sensitivity (3pp swing) exceeds every cross-dataset H effect found | ✅ done — see Final synthesis below |
| 8 | ~~Add twohop_in/twohop_out terms to the entropy map~~ | — | **dropped** — map quality was never the bottleneck | ❌ dropped |
| 9 | Combine D (dynamic-pool)/D+R with the new entropy map | moderate | low priority — same reasoning as dropped #8 | ⬜ low priority |
| 10 | Short-walk main model + miner experiment | large, needs scoping | keep shelved *unless* independently motivated by main-model research priorities — don't chase this just to rescue the miner | ⬜ shelved |
| 11 | Multi-seed variance infra (`plan-stats-rigor.md`) | large, separate infra project | needed eventually, not specific to this thread | ⬜ shelved |
| 12 | Posthoc inference batch size (`run_posthoc.py` reuses training `batch_size=1024` for pure inference, no backward-pass memory constraint — likely 4-8x headroom on a 44GB GPU with a 342K-param model) | ~15 min | minor, quality-of-life speedup, not blocking anything | ⬜ low priority, not urgent |
| 13 | LocalAttn4 final ablation (E27/E28): re-run no-H vs. entropy-H on the current `edge_cover` sampler + NaN-fix, since the old "H is load-bearing for LocalAttn4" finding (E16/E17) predates both | ~1h wall-clock, 12 runs | the decisive experiment for whether LocalAttn4 needs H *at all* anymore, not just which map/formula | 🔄 running — see "E27/E28 LocalAttn4 ablation" below |
| 14 | Source/target mixing weight (`model.hardness_combine=weighted`, `model.hardness_source_weight`): lets `h_left`/`h_right` combine as `w*h_left+(1-w)*h_right` instead of a flat 50/50 mean, implemented in `lit_model.py` | ~15 min code, sweep TBD | motivated by Lead4c's walk-model coefficients (`src_out` β≈−1.01 vs `tgt_in` β≈−0.63, walk hurt ~1.6× more by source) — **but that ratio must NOT be used directly to set `hardness_source_weight`, see caveat below** | ✅ knob implemented, not yet swept — blocked on item 13's keep/scrap verdict |

## E22 full-attention results so far (role-aware entropy map, mean-combine/power=1 — the original formula)

| dataset | E22 (entropy, role-aware) | E17 (old miner) | E16 (no hardness) | delta vs. E17 | delta vs. E16 |
| --- | --- | --- | --- | --- | --- |
| wiki-elec | 0.9020 | 0.9032 | 0.9016 | −0.0012 | +0.0004 |
| epinions | 0.9584 | 0.9573 | 0.9572 | **+0.0011** | +0.0012 |
| bitcoin-otc | 0.9406 | 0.9427 | 0.9434 | −0.0021 | −0.0028 |
| slashdot090221 | 0.9005 | 0.9002 | 0.9014 | +0.0003 | −0.0009 |
| wiki-rfa | 0.8901 | 0.8940 | 0.8917 | −0.0039 | −0.0016 |
| bitcoin-alpha | 0.9177 | 0.9325 | 0.9212 | **−0.0148** | −0.0035 |

**All 6/6 baseline results are in.** bitcoin-alpha is now the clearest and largest negative
result of the whole sweep (−1.5pp vs. the old miner, −0.35pp vs. no hardness at all) — on the
biggest dataset, with the best hardness map found this session. Only epinions shows a real
(if small) win. 5/6 datasets show flat-to-negative deltas vs. both E17 and E16 — this is not a
map-quality problem, it's either a formula problem (item 7, running) or full attention is
genuinely architecturally indifferent/mildly harmed by this reweighting mechanism.

## E23 reweighting-formula pilot (in progress)

`hardness_combine=max`: `walk_w = 1 + λ·max(h_left,h_right)` instead of the mean.
`hardness_power=2.0`: `walk_w = 1 + λ·mean(h_left,h_right)^2` (convex — only strongly
upweights walks where the average is already high, near-zero effect on mildly-hard walks).
Both use the same E22 role-aware entropy maps and λ=1.0, isolating the formula as the only
variable. Results land in `/tmp/e23_logs/gpu{1,2,3}_queue.log` as each run finishes.

| dataset | baseline (mean, pow=1) | MAX | POW2 |
| --- | --- | --- | --- |
| wiki-elec | 0.9020 | 0.9027 (+0.0007) | 0.9013 (−0.0007) |
| wiki-rfa | 0.8901 | 0.8934 (+0.0033) | 0.8929 (+0.0028) |
| epinions | 0.9584 | 0.9571 (−0.0013) | 0.9575 (−0.0009) |
| bitcoin-alpha | 0.9177 | **0.8978 (−0.0199)** | **0.9283 (+0.0106)** |
| bitcoin-otc | 0.9406 | 0.9407 (+0.0001) | 0.9441 (+0.0035) |
| slashdot090221 | 0.9005 | 0.8998 (−0.0007) | 0.8997 (−0.0008) |

**All 18/18 runs done. Pilot complete — verdict: no formula is a universal winner, and
bitcoin-alpha's own within-dataset spread across formulas (0.8978 → 0.9283, a 3.05pp swing on
the identical map/λ/dataset, nothing else changed) is bigger than every cross-dataset "H
helps" signal found anywhere in this whole investigation.** That's the decisive data point —
see "Final synthesis" below.

**Rule for this roadmap: any retrain requires explicit approval before launching — do not
start heavy runs silently.** (Pilot approved 2026-07-12 is complete; no further runs launched
without new approval.)

## Final synthesis (2026-07-13) — full comparison table, all H variants vs. E16 no-hardness

| dataset | E16 (no H) | E17 (old miner) | E22 (entropy, mean/pow1) | E23 MAX | E23 POW2 | best H delta vs E16 |
| --- | --- | --- | --- | --- | --- | --- |
| wiki-elec | 0.9016 | 0.9032 | 0.9020 | 0.9027 | 0.9013 | +0.0016 (E17) |
| epinions | 0.9572 | 0.9573 | 0.9584 | 0.9571 | 0.9575 | +0.0012 (E22) |
| bitcoin-otc | 0.9434 | 0.9427 | 0.9406 | 0.9407 | 0.9441 | +0.0007 (POW2) |
| slashdot090221 | 0.9014 | 0.9002 | 0.9005 | 0.8998 | 0.8997 | **0 — no H variant beats no-H** |
| wiki-rfa | 0.8917 | 0.8940 | 0.8901 | 0.8934 | 0.8929 | +0.0023 (E17) |
| bitcoin-alpha | 0.9212 | 0.9325 | 0.9177 | 0.8978 | 0.9283 | +0.0113 (E17) |

**Reading this straight:**

- 5 of 6 datasets never exceed +0.25pp over no-hardness with *any* map/formula combination
  tried (2 maps × 3 formulas = 6 variants each). That's flat — indistinguishable from noise,
  and on slashdot090221 every single variant is *worse* than doing nothing.
- bitcoin-alpha is the only dataset with a headline-sized gain (+1.13pp, E17 old miner) — but
  it's also the *only* dataset where changing nothing but the combine formula (mean → max)
  swings test AUC by over 3pp in the other direction (0.9283 → 0.8978). A mechanism that's
  this sensitive to an arbitrary formula choice, on the one dataset that supposedly benefits
  most, looks like variance dominating signal, not a real effect the model is reliably using.
  bitcoin-alpha also has by far the smallest test set of the six (2,419 unique edges vs.
  3,560–84,080 for the others) — consistent with it being the noisiest AUC measurement in the
  whole sweep, which is the more likely explanation for both its unusually large positive
  swings (E17, POW2) and its unusually large negative one (MAX).
- **No formula is a universal winner.** MAX/POW2 help wiki-elec/wiki-rfa (where the mean
  baseline hurt worst) and hurt epinions (where the mean baseline helped) and swing
  bitcoin-alpha wildly in both directions — the formula choice reshuffles winners/losers
  rather than raising the floor anywhere.

**Conclusion: H does not reliably help full attention on any dataset.** The one apparent
exception (bitcoin-alpha) fails a basic robustness check (formula-sensitivity bigger than the
effect itself) and is exactly the dataset where measurement noise is expected to be largest.
Recommendation: **drop `hardness_lambda`/`hardness_map_path` from full-attention production
configs** (`configs/<dataset>.yaml` already default to no override — confirm none set it), and
scope the miner pipeline (map building, `compute_hardness_map.py`, entropy map generation) as
LocalAttn4-only going forward, where E16 already showed H is load-bearing (−0.006 to −0.065
without it). No further full-attention H experiments planned unless a new candidate mechanism
(not just a formula tweak) is proposed.

**Not yet done, optional if more rigor is wanted before writing this up:** a genuine
reseed check (`training.seed=<other>`, everything else identical) on bitcoin-alpha specifically,
to directly quantify its run-to-run noise floor rather than inferring it from the formula
swing. Not launched — no approval requested yet.

## E27/E28 LocalAttn4 ablation (2026-07-19, in progress) — item 13

**Why this is needed, not a rehash of E16/E17:** the standing "H is load-bearing for
LocalAttn4" claim (CLAUDE.md, E16: deltas −0.006 to −0.065 without H) predates two things
that have since changed: (a) the walk sampler moved from `k_cover` to the dedup-guaranteed
`edge_cover` (E24–E26), with much smaller per-dataset budgets; (b) a real NaN bug in
LocalAttn4's eval-mode fast path (`MASKING.md`) made *every* prior LocalAttn4 eval number
suspect, fixed 2026-07-19. E27 = fresh no-H LocalAttn4 on the current sampler + fixed eval.
E28 = same, with the E22 role-aware entropy map, mean-combine/power=1 (the plain baseline
formula — no MAX/POW2 pilot here yet, that's a possible follow-up if item 13 says "keep").

| dataset | E27 (no-H) | E28 (entropy-H) | delta |
| --- | --- | --- | --- |
| bitcoin-alpha | 0.9126 | 0.9184 | +0.0058 |
| bitcoin-otc | 0.9390 | 0.9284 | **−0.0106** |
| epinions | 0.9533 | 0.9535 | +0.0002 |
| wiki-elec | 0.9061 | 0.9064 | +0.0003 |
| wiki-rfa | 0.8959 | 0.8966 | +0.0007 |
| slashdot090221 | 0.8981 | running | — |

**Provisional read (5/6 pairs in):** 4/5 flat (under the project's own +0.25pp "not a robust
win" bar) or negative — the same shape the full-attention ablation showed before H got
scrapped there. A real reversal from the E16/E17-era finding: bitcoin-otc's fresh no-H number
(0.9390) already *beats* the old E17 with-hardness SOTA (0.9379) on a much smaller walk
budget — suggesting the old "H recovers 2.9–6.5pp" result was largely compensating for the
stale sampler and/or the (unrelated, but simultaneously present) NaN eval bug, not something
hardness reweighting was actually doing. **Final verdict pending slashdot090221's pair.**

## Source/target mixing weight — item 14, do NOT set from Lead4c

`model.hardness_combine=weighted` + `model.hardness_source_weight=<w in [0,1]>` computes
`combined = w*h_left + (1-w)*h_right` instead of the flat mean (`w=0.5` reproduces "mean"
exactly — verified bit-identical). Implemented as a separate combine mode (not a
generalization of "mean") so every existing "mean"/"max" run/config is completely
unaffected by its presence.

**Motivation:** Lead4c's atomic regression, fit on the walk model's own predictions, found
`src_out` (source rater's out-edge consistency — literally the same statistic as
`hardness_source.pt`) hurts the walk model's accuracy ~1.6× more than `tgt_in` (target's
contested reputation — `hardness_target.pt`): β≈−1.01 vs −0.63 (`lead4_coefficients.md`).
That's a real, well-powered, walk-model-specific finding.

**But the ratio must not be used to hard-set `hardness_source_weight`.** Checked the actual
code: Lead4c's `src_out`/`tgt_in` features are computed from **all edges — train+val+test**
(`scripts/lead4_entropy_heterogeneity.py`, by design, since that analysis is a legitimate
post-hoc diagnostic of an already-frozen model — not a claim it's safe for training-time
reuse), and the regression's outcome variable (prediction correctness) comes from
`test_predictions.pkl` (`baselines/postprocess_canonical.py`) — i.e. **test-set outcomes**.
Any number derived from that regression and fed back into a training-time hyperparameter
that then gets re-evaluated on the same test set is test-set leakage, full stop, regardless
of how clean the *deployed* E22 hardness maps are (those genuinely are computed TRAIN+MASK-only
and are fine). A per-dataset interacted atomic fit (to check whether the wiki-elec/wiki-rfa
entropy-value reversal found in Lead 4 also flips the `src_out`/`tgt_in` *coefficient ranking*
there) does not currently exist — only a shared-slope (pooled) atomic fit was ever run; the
one interacted fit that exists uses a different, legacy feature spec (`marginal3`, likely
`in_in` combo) and can't stand in as an answer. Even if that regression were run, it would
carry the same two leakage issues, so it still wouldn't be usable to set this hyperparameter.

**Correct way to pick `w`:** ordinary hyperparameter sweep (grid or Optuna) selected on
**validation AUC only**, test set touched once at the end — same discipline as any other
sweep in this project (e.g. E23's `hardness_combine`/`hardness_power` pilot). Blocked on
item 13's keep/scrap verdict: no point tuning a mixing weight for a mechanism that might get
dropped entirely.
