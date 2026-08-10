# Hardness miner — Q3 (D/R training regime) + Q5 (hardness definition) screen

Follow-up to `E17_HARDNODE_KCOVER_REMINE_RESULTS.md` (Q1: H helps LocalAttn4, not full
attention) and `plan-hardness-miner.md`. Answers three questions in one pass: (1) does
adding D (dynamic-pool) and/or R (node replacement) to the miner's own training regime
improve the map, mirroring the D+R synergy found for the *main* model in `old_chats/DRH.md`;
(2) is raw accuracy the best per-node hardness *definition*, or do richer signals
(confidence margin, Brier score, per-edge loss) do better; (3) does a zero-training
structural feature beat the learned miner outright. All mining reused the E15 `k_cover`
local-attention-budget caches; all correlations use the validated **predictive-validity**
metric (Spearman ρ between a candidate hardness map and the *real* per-node error rate of
the already-trained E17 LocalAttn4 checkpoint on held-out **TEST** edges — see "Why not
`mean_hardness ∈ [0.10,0.18]`" below). Script: `scripts/hardness_predictive_validity.py`.

## Why not the DRH.md `mean_hardness ∈ [0.10, 0.18]` "sweet spot"

That range was never a validated metric. It comes from exactly **2** downstream-AUC data
points on **1** dataset (wiki-rfa): 5 epochs → mean=0.153 → test AUC=0.8493 (best); 15
epochs → mean=0.067 → test AUC=0.8461 (worse). There's a documented counterexample in the
same file: an 8-epoch dynamic-pool run landed mean_hardness=0.1407 ("squarely in the sweet
spot") yet scored test AUC=0.8310 — clearly worse. Comparing candidate maps by where their
mean falls in that range is not a sound way to judge map quality. All comparisons below use
the predictive-validity correlation instead — grounded in genuine downstream difficulty, no
retraining needed, reusable for any future candidate.

## Q3 — Training-regime axis (D = dynamic-pool, R = node replacement)

R was not previously wired into the miner at all (the miner hand-rolls its own training
loop, bypassing `lit_model.py`); ported `_maybe_apply_node_replacement` into
`compute_hardness_map.py` (training-time only — eval/hardness-collection always sees clean
`input_ids`, so R never contaminates the attribution itself). `--dynamic-pool` already
existed (rotates MASK targets each epoch, mirrors the main model's D).

### Predictive-validity correlation, 4 training-regime variants (mining only, no retrain)

| dataset | E17 static | E18 dyn-pool (D) | E19 static+R | E20 D+R |
|---|---|---|---|---|
| bitcoin-alpha | 0.2999 | 0.3271 | 0.3027 | **0.3720** |
| bitcoin-otc | 0.2837 | 0.3651 | 0.2921 | **0.3663** |
| epinions | 0.1418 | 0.1425 | 0.1462 | 0.1491 (~tied) |
| wiki-elec | 0.2417 | **0.2592** | 0.2310 | 0.2477 |
| wiki-rfa | **0.2173** | 0.1973 | 0.2203 | 0.1975 |
| slashdot090221 | 0.2942 | 0.3418 | 0.2949 | **0.3432** |

R alone barely moves anything (mean_hardness shift <0.01 on most datasets vs static;
rank correlation with static stayed 0.89–0.95 in an earlier check). D dominates; D+R adds
a small increment on top of D on 3/6 datasets, is worse than D-alone on wiki-elec/wiki-rfa.

### Downstream retrain confirmation — D-only (E18), LocalAttn4, all 6 datasets (COMPLETE)

| dataset | E17 static (test AUC) | E18 dyn-pool (test AUC) | Δ |
|---|---|---|---|
| bitcoin-alpha | 0.9325 | **0.9370** | **+0.0045** |
| bitcoin-otc | 0.9379 | **0.9404** | **+0.0025** |
| epinions | 0.9555 | 0.9530 | −0.0025 |
| wiki-elec | 0.9021 | 0.9015 | −0.0006 |
| wiki-rfa | 0.8940 | 0.8917 | −0.0023 |
| slashdot090221 | 0.8982 | 0.8977 | −0.0005 |

**Reading:** directionally consistent with predictive validity on 4/6 datasets — clear wins
on bitcoin-alpha/bitcoin-otc (both metrics agree, both large), a clear loss on wiki-rfa (both
metrics agree). Two mismatches: wiki-elec's predictive validity favored D (0.259 vs 0.242)
but downstream was flat-to-negative (−0.0006, likely just noise given the tiny magnitude);
slashdot's predictive validity clearly favored D (0.342 vs 0.294) but downstream was flat
(−0.0005). **Predictive validity is directionally useful but not a perfect proxy for
downstream AUC magnitude** — treat it as a screening tool to prioritize retrains, not a
guarantee.

**D+R downstream retrain was never run** (paused mid-launch, then superseded by the Q5
screen below finding a more promising direction — see recommendation).

## Q5 — Hardness *definition* axis (accuracy vs. margin/Brier/loss)

Extended `compute_hardness_map.py` (`--save-variants`) to compute 3 additional candidate
hardness definitions from the *same* miner training pass (static recipe, matching E17
exactly — verified: variant run's mean_hardness reproduces E17's numbers to 4 decimal
places on all 6 datasets): margin (`1 − (top1_prob − top2_prob)`), Brier score
(`Σ_c (p_c − onehot_c)²`), per-edge cross-entropy loss.

| dataset | accuracy (=E17) | margin | brier | loss |
|---|---|---|---|---|
| bitcoin-alpha | 0.3012 | 0.2813 | 0.2769 | 0.2810 |
| bitcoin-otc | 0.2833 | 0.2450 | 0.2832 | 0.2856 |
| epinions | 0.1416 | 0.0289 | 0.0561 | 0.0566 |
| wiki-elec | 0.2417 | 0.1950 | 0.2378 | 0.2387 |
| wiki-rfa | 0.2173 | 0.1417 | 0.2096 | 0.2120 |
| slashdot090221 | 0.2939 | 0.2532 | 0.2649 | 0.2664 |

**Clean negative result: raw accuracy beats margin/Brier/loss on every single dataset**,
sometimes by a large margin (epinions: accuracy 0.142 vs. margin 0.029 — nearly 5×).
Likely cause: the miner is deliberately tiny (emb=16/hidden=16) to avoid memorization —
that same weakness probably makes its raw confidence values poorly calibrated, so anything
that leans on confidence (margin, Brier, loss) inherits that noise, while the coarse
correct/incorrect binary is robust to it. **Don't pursue alternative miner-derived metrics
beyond raw accuracy — this axis is closed.**

## Q5 — Zero-training structural baseline vs. the learned miner

Two structural candidates computed directly from the graph (**TRAIN+MASK edges only** —
an earlier pass leaked VAL/TEST edge signs into a node's own structural feature, inflating
the correlation, especially on low-out-degree nodes; fixed before these numbers):
- `struct_inv_outdeg = 1/(1+outdeg)` — low out-degree = less evidence = "harder"
- `struct_ratio_extremity = 1 − |2·pos_frac − 1|` — a node whose outgoing votes are evenly
  split between positive/negative is "harder" than one that's consistently one-sided

| dataset | miner accuracy | struct_inv_outdeg | struct_ratio_extremity |
|---|---|---|---|
| bitcoin-alpha | 0.3012 | −0.2787 | **0.4143** |
| bitcoin-otc | 0.2833 | −0.2036 | **0.3798** |
| epinions | 0.1416 | −0.0231 | **0.3745** |
| wiki-elec | 0.2417 | −0.1452 | **0.3042** |
| wiki-rfa | 0.2173 | −0.1813 | **0.2881** |
| slashdot090221 | 0.2939 | −0.1420 | **0.4093** |

**`struct_ratio_extremity` — a feature that requires zero training — beats the entire
learned-miner pipeline on every single dataset**, often by a wide margin (epinions: 0.375
vs. 0.142, >2.5×). `struct_inv_outdeg` is *negatively* correlated with true difficulty on
every dataset — i.e. **lower**-out-degree nodes tend to be *easier*, not harder, the
opposite of the naive "less evidence = harder" intuition. Plausible read, and a nice tie-back
to this project's own Lead 4c finding (`LEAD4C_ASYMMETRY.md`): nodes with *many* outgoing
edges are the ones whose rating behavior has more room to be inconsistent (Lead 4c's
`src_out` term — "how consistent a rater u is" — was already the single largest
architecture-asymmetry effect found in that thread), so high out-degree tracking with *more*
error, not less, is consistent with prior evidence, not a new coincidence.

### Blend: miner accuracy ⊕ structural ratio-extremity (average rank)

Computed over the (smaller) node subset with both a valid miner-accuracy and a valid
structural value — **not directly comparable in magnitude to the tables above** (different
denominator), but internally consistent for this one comparison:

| dataset | accuracy alone | struct alone | blend (avg rank) |
|---|---|---|---|
| bitcoin-alpha | 0.3476 | **0.4143** | 0.4117 (≈tied w/ struct) |
| bitcoin-otc | 0.3410 | 0.3798 | **0.3960** |
| epinions | 0.3404 | 0.3745 | **0.3992** |
| wiki-elec | 0.2962 | 0.3042 | **0.3525** |
| wiki-rfa | 0.3101 | 0.2881 | **0.3470** |
| slashdot090221 | 0.3906 | 0.4093 | **0.4514** |

**A simple average-rank blend of the miner's accuracy-hardness and the structural
ratio-extremity feature beats BOTH individual signals on 5/6 datasets**, and is
statistically tied with structural-alone on the 6th (bitcoin-alpha). This is the strongest,
cheapest lever found in the whole miner investigation.

## Interim conclusions

1. **H (hardness reweighting) is load-bearing for LocalAttn4, optional for full attention**
   (Q1, unchanged from `E17_HARDNODE_KCOVER_REMINE_RESULTS.md`).
2. **D (dynamic-pool) has a real, dataset-specific effect** — helps bitcoin-alpha/bitcoin-otc
   clearly (confirmed both by predictive validity and downstream retrain), hurts wiki-rfa
   clearly (same), is a wash elsewhere. Not a universal win/loss — use per-dataset.
3. **R (node replacement) is a weak lever for the miner** — unlike its strong, necessary role
   in the main model's D+R synergy (DRH.md), the miner's already-tiny capacity seems to make
   R's "stop relying on node identity" effect largely redundant. D+R modestly beats D-alone
   on 3/6 datasets (bitcoin-alpha/otc/slashdot per predictive validity) but was never
   confirmed downstream.
4. **The hardness *definition* is settled: raw accuracy, not margin/Brier/loss** — richer
   continuous signals are uniformly worse, likely because the miner's confidence is poorly
   calibrated by design (tiny/weak on purpose). Closed question, don't revisit without a
   reason to believe miner calibration itself has changed.
5. **The single best lever found: blend the miner with a trivial structural feature**
   (sign-ratio extremity), not further miner tuning. Structural-alone already beats the
   learned miner outright on every dataset; blended, it beats both components on 5/6.

## Recommendation — before a full 6×(local+full) retrain campaign

Given finding 5, a full retrain campaign should test the **blended hardness map**
(structural ⊕ miner-accuracy, average-rank or similar) on LocalAttn4 first — it's the
best-validated candidate by a clear margin, is cheap (structural feature needs zero
training; blending is a few lines), and per Q5's own pre-filter design this is exactly the
kind of result that should gate expensive retrain compute. Suggested next steps, in order:
1. Implement the blended map as a first-class artifact (not just a scratch analysis) and
   mine it for all 6 datasets.
2. Retrain LocalAttn4 on the blended map for all 6 datasets, compare against E17 static
   (current best) and E18 D-only.
3. Only if the blend clearly wins there, extend to full attention (where H is optional/
   marginal per Q1 — lower expected payoff, still worth a check given E17's finding of small
   positive deltas on bitcoin-alpha/wiki-elec).
4. D+R downstream retrain (bitcoin-alpha/otc/slashdot) is now lower priority than the blend
   — predictive validity gain from R on top of D (e.g. bitcoin-alpha 0.372 D+R vs 0.327 D) is
   smaller than the gain from adding the structural feature to accuracy-D alone would likely
   be (extrapolating from the static-recipe blend numbers above, 0.41 on bitcoin-alpha).
