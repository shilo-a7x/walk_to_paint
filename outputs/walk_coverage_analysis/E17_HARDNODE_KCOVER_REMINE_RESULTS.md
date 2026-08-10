# E17 — re-mined hardness map on E15 `k_cover` data, corrected recipe

Follow-up to `E16_NOHARD_RESULTS.md` (Q1: does the hardness map help at all) and
`plan-hardness-miner.md`'s methodological history (the miner train/eval design is
intentional, not leaky; the real bug was the mining recipe — see below). Answers:
**with a properly-mined hardness map (matching the current k_cover walk data and
the actual best-known recipe), does LocalAttn4 recover its lost AUC? Does full
attention benefit from a healthier map?**

## Recipe

- Hardness maps re-mined via the newly-optimized `scripts/compute_hardness_map.py`
  (bucket batching + multi-worker loading, ~12x faster — see
  `plan-hardness-miner.md`'s Performance section) on each dataset's E15 `k_cover`
  **local-attention-budget** cache (the smaller/relevant one where local ≠ full).
- **Corrected recipe**: tiny model (emb=16/hidden=16/nlayers=2), 5 epochs, **no**
  `--max-walk-edges` filter — matching `E14_HARDNODE_L10`'s actual production
  config (verified in `run_transformer_incremental_experiments.py`), not the
  filtered variant PROJECT_OVERVIEW.md previously (incorrectly) documented as
  standard. A first mining pass used the filter by mistake; discarded and redone.
- Output: `outputs/<ds>/E17_HARDNODE_KCOVER_REMINE/hardness_map.pt`.

### Mined hardness map stats (mean_hardness, per dataset)

| dataset | mean_hardness | vs. DRH.md's empirical sweet spot (0.10–0.18) |
|---|---|---|
| bitcoin-alpha | 0.0050 | still collapsed — likely a genuine data property (very locally predictable graph) |
| bitcoin-otc | 0.0772 | below |
| epinions | 0.1200 | in range |
| wiki-elec | 0.1911 | just above |
| wiki-rfa | 0.1895 | just above |
| slashdot090221 | 0.0743 | below |

## Results (func_logit_power test AUC) — all three conditions

### Full attention

| dataset | E15 SOTA (old E14 map) | E16 (no hardness) | E17 (new map) | E17−E15 | E17−E16 |
|---|---|---|---|---|---|
| bitcoin-alpha | 0.9251 | 0.9212 | **0.9307** | +0.0056 | +0.0095 |
| bitcoin-otc | 0.9427 | 0.9434 | 0.9427 | 0.0000 | −0.0007 |
| epinions | 0.9562 | 0.9572 | 0.9573 | +0.0011 | +0.0001 |
| wiki-elec | 0.9016 | 0.9016 | **0.9032** | +0.0016 | +0.0016 |
| wiki-rfa | 0.8932 | 0.8917 | 0.8919 | −0.0013 | +0.0002 |
| slashdot090221 | 0.9012 | 0.9014 | 0.9002 | −0.0010 | −0.0012 |

### LocalAttn4

| dataset | E15 SOTA (old E14 map) | E16 (no hardness) | E17 (new map) | E17−E15 | E17−E16 |
|---|---|---|---|---|---|
| bitcoin-alpha | 0.9362 | 0.8903 | **0.9325** | −0.0037 | **+0.0422** |
| bitcoin-otc | 0.9410 | 0.9121 | **0.9379** | −0.0031 | **+0.0258** |
| epinions | 0.9568 | 0.8918 | **0.9555** | −0.0013 | **+0.0637** |
| wiki-elec | 0.9038 | 0.8979 | **0.9021** | −0.0017 | +0.0042 |
| wiki-rfa | 0.8916 | 0.8844 | **0.8940** | **+0.0024** | +0.0096 |
| slashdot090221 | 0.8984 | 0.8848 | **0.8982** | −0.0002 | **+0.0134** |

## Findings

**LocalAttn4: the new map recovers essentially all of the AUC that E16 showed
was lost without hardness reweighting.** On all 6 datasets, E17 lands within
0.004 AUC of E15's SOTA number (using the *old*, pre-k_cover, sometimes-degenerate
E14 map) — and on wiki-rfa it slightly **exceeds** E15. This is a clean, strong
result: it confirms (a) H's contribution to LocalAttn4 found in E16 is real, not
an artifact, and (b) a correctly-mined, sampler-matched hardness map fully
accounts for that contribution — there's no further gap to chase from the mining
side for LocalAttn4.

**Full attention: still no clear benefit, but no longer clean noise either.**
Deltas vs. E15 are small (−0.0013 to +0.0056) and mostly within the
epoch-to-epoch noise band identified in E16. bitcoin-alpha and wiki-elec do show
a small, consistent uptick (+0.0056/+0.0016 over E15, +0.0095/+0.0016 over E16)
— interesting given bitcoin-alpha's E17 map is itself nearly degenerate
(mean_hardness=0.0050), suggesting even a thin, mostly-near-zero hardness signal
might carry a small amount of real information for full attention. Not strong
enough evidence to reverse Q1's "not confirmed beneficial for full attention"
call without the pending multi-seed variance check, but no longer safe to call
it purely noise-level either.

## Decision (updates Q1's provisional call)

- **LocalAttn4: keep H, and adopt the E17 (k_cover-mined) maps as the correct
  ones going forward** — the old E14 maps were mined on stale (pre-k_cover)
  walks; E17 recovers/matches/slightly-beats E15's numbers using data-matched
  maps. Recommend swapping CLAUDE.md's LocalAttn4 hardness map paths to the E17
  ones.
- **Full attention: still optional**, per Q1's original call — the E17 numbers
  don't overturn that, but the small positive deltas on 2/6 datasets are worth
  keeping an eye on once multi-seed variance estimates exist
  (`plan-stats-rigor.md`).
- **Not yet done**: the `--dynamic-pool` isolated test (queued in
  `plan-hardness-miner.md`) — still an open question about whether the miner's
  fixed-MASK-target design costs anything relative to a rotating one. E17's
  strong LocalAttn4 recovery doesn't require answering that question, but it's
  still queued for the broader miner-quality investigation (Q2–Q5).
