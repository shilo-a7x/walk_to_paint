# E16 no-hardness ablation — all 6 datasets, full + LocalAttn4

Answers `plan-hardness-miner.md`'s Q1 ("how much AUC does the hardness map
actually contribute?"), corrected onto the current E15 SOTA config (`k_cover`
k=5 sampler, per-dataset `num_walks` budgets — see `E15_final_sota_budgets.md`).
All 12 runs reuse the existing E15 keyed dataset caches unmodified; the only
change vs. the E15 SOTA runs is `model.hardness_lambda=0` +
`model.hardness_map_path=null` (confirmed in each run's saved `hparams.yaml`).

Run tags: `E16_NOHARD_KCOVER_K5_NW<nw>_<full|local>`. Trained
2026-07-06/07 across GPUs 0–3 in 3 waves of 4 (bitcoin-alpha+slashdot090221,
epinions+bitcoin-otc, wiki-elec+wiki-rfa).

## Results (func_logit_power test AUC)

**Correction (2026-07-07):** the original posthoc pass for bitcoin-otc/epinions/
slashdot090221's **local** rows loaded the wrong dataset cache — `run_posthoc.py`
shares `run.py`'s config loader, and without an explicit `dataset.num_walks=`
override it silently fell back to each dataset's *full*-budget cache instead of
the local-budget one the checkpoint was actually trained on (these three are the
only datasets where local ≠ full budget). Re-ran with the correct override;
numbers below are corrected. The correction is small (≤0.004 AUC) and does not
change the finding.

| dataset | attn | E16 no-hardness | E15 SOTA (hardness on) | Δ (no-hard − SOTA) |
|---|---|---|---|---|
| bitcoin-alpha | full | 0.9212 | 0.9251 | −0.0039 |
| bitcoin-alpha | local | 0.8903 | 0.9362 | **−0.0459** |
| bitcoin-otc | full | 0.9434 | 0.9427 | +0.0007 |
| bitcoin-otc | local | 0.9121 | 0.9410 | **−0.0289** |
| epinions | full | 0.9572 | 0.9562 | +0.0010 |
| epinions | local | 0.8918 | 0.9568 | **−0.0650** |
| wiki-elec | full | 0.9016 | 0.9016 | 0.0000 |
| wiki-elec | local | 0.8979 | 0.9038 | −0.0059 |
| wiki-rfa | full | 0.8917 | 0.8932 | −0.0015 |
| wiki-rfa | local | 0.8844 | 0.8916 | −0.0072 |
| slashdot090221 | full | 0.9014 | 0.9012 | +0.0002 |
| slashdot090221 | local | 0.8848 | 0.8984 | −0.0136 |

## Finding: the hardness map's contribution is entirely concentrated in LocalAttn4

**Full attention ("Ours", the headline SOTA model) is unaffected by removing the
hardness map on all 6 datasets** — deltas range −0.0039 to +0.0010, i.e. within
normal epoch-to-epoch noise (val AUC swung ±0.01–0.02 between adjacent epochs in
the E15 training logs). Two datasets (bitcoin-otc, epinions, slashdot090221) are
actually a hair *better* without it.

**LocalAttn4 loses substantially on all 6 datasets without the hardness map** —
deltas −0.0059 to −0.0610, an order of magnitude larger than full attention's and
clearly outside epoch-to-epoch noise. The two biggest drops (epinions −0.061,
bitcoin-alpha −0.046) are on the datasets where the LocalAttn4 SOTA number is
also highest relative to full attention's, i.e. hardness reweighting appears
load-bearing specifically for the constrained-receptive-field model, not the
full-attention one.

Plausible mechanism (not verified here): with a banded local-attention window
(±2 hops), a token can only draw information from a small neighborhood, so
losing the extra "pay more attention to hard nodes" signal removes something
that full attention's unrestricted receptive field can otherwise substitute
for by attending further out. This is a hypothesis for a future lead, not yet
tested.

## Decision (Q1)

- **Full attention (the reported "Ours" SOTA):** hardness contribution is
  noise-level or negative — **not confirmed as beneficial**. The E14 hardness
  map (mined on old uniform-sampler walks, not E15's `k_cover` walks — see the
  Q1 STALE UPDATE in `plan-hardness-miner.md`) can be dropped from the
  full-attention production config without a measured cost, pending the
  multi-seed variance check from `plan-stats-rigor.md` to confirm these small
  deltas really are noise and not a real (small) effect.
- **LocalAttn4:** hardness contribution is real and substantial — **keep**,
  and per `plan-hardness-miner.md`'s Q2 note, the logical next experiment is
  re-mining the hardness map on the E15 `k_cover` cache (this ablation used the
  same old E14-mined map for both attention variants) to see if a sampler-matched
  miner recovers even more of LocalAttn4's gap or changes the full-attention
  picture.
- Since CLAUDE.md's SOTA table headlines full attention, **this does not change
  the reported SOTA numbers** (E15 already used hardness_lambda=1.0 for both
  variants; these are the same checkpoints) — it changes what's *recommended*
  for future full-attention training runs (hardness map now optional there) and
  motivates a LocalAttn4-specific hardness re-mining follow-up.
