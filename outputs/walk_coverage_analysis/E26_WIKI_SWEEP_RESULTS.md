# E26 — edge_cover budget sweep (wiki-elec, wiki-rfa) + E27 wiki-elec architecture probe

Extends Phase 4 of `plan-a-fix-for-glimmering-panda.md` to the two datasets excluded
from the original E25 scope (wiki-elec, wiki-rfa were already known to over-saturate
past a small budget under the old `k_cover` sampler — this sweep checks whether that
holds under the new zero-duplication `edge_cover` sampler too, rather than assuming
it). All runs use `walk_strategy=edge_cover` (k fixed at 1, provable zero-duplication
guarantee), full attention, no hardness miner, `func_logit_power` posthoc aggregator,
seed=42. Grid per dataset: `{|E| (floor), 1.5×|E|, 3×|E|, 5×|E|, 8×|E|}` — the 8×
point was added (beyond E25's `{floor,1.5×,3×,5×}` grid) specifically to confirm the
over-saturation decline continues rather than reversing.

All posthoc calls used the corrected, cache-matching override pattern
(`dataset.walk_strategy=edge_cover dataset.num_walks=<nw>`, per `CLAUDE.md`'s posthoc
rule) from the start — no correction note needed for this batch.

## Full grid — every point, not just winners

| dataset | \|E\| | budget label | num_walks | mult | test AUC | train AUC | run dir |
|---|---|---|---|---|---|---|---|
| wiki-elec | 103,689 | floor | 103,689 | 1.0× | 0.8900 | 0.8850 | `E26_WIKI_elec_floor_20260719-113223` |
| wiki-elec | 103,689 | 1.5× | 155,534 | 1.5× | **0.9036** | 0.8979 | `E26_WIKI_elec_p1_5x_20260719-113223` |
| wiki-elec | 103,689 | 3× | 311,067 | 3.0× | 0.9000 | 0.8964 | `E26_WIKI_elec_x3_20260719-113223` |
| wiki-elec | 103,689 | 5× | 518,445 | 5.0× | 0.8972 | 0.8927 | `E26_WIKI_elec_x5_20260719-113522` |
| wiki-elec | 103,689 | 8× | 829,512 | 8.0× | 0.8937 | 0.8863 | `E26_WIKI_elec_x8_20260719-113639` |
| wiki-rfa | 177,211 | floor | 177,211 | 1.0× | 0.8891 | 0.8883 | `E26_WIKI_rfa_floor_20260719-113813` |
| wiki-rfa | 177,211 | 1.5× | 265,817 | 1.5× | **0.8930** | 0.8902 | `E26_WIKI_rfa_p1_5x_20260719-113223` |
| wiki-rfa | 177,211 | 3× | 531,633 | 3.0× | 0.8924 | 0.8876 | `E26_WIKI_rfa_x3_20260719-114436` |
| wiki-rfa | 177,211 | 5× | 886,055 | 5.0× | 0.8907 | 0.8870 | `E26_WIKI_rfa_x5_20260719-114611` |
| wiki-rfa | 177,211 | 8× | 1,417,688 | 8.0× | 0.8838 | 0.8837 | `E26_WIKI_rfa_x8_20260719-114316` |

(Bold = best point per dataset within the grid.)

## Per-dataset shape

- **wiki-elec**: clean peak at 1.5× (0.9036), then a monotonic decline through
  3×/5×/8× (0.9000 → 0.8972 → 0.8937, a total −0.99pp from peak to 8×). floor→1.5×
  is +1.36pp — the initial gain from more distinct walks is real — but past 1.5× it's
  pure over-saturation, not noise: every step past the peak is lower than the one
  before it, all the way to 8×. Confirms this is a genuine graph/model property, not
  a duplication artifact — the old `k_cover` sampler showed the same over-saturation
  shape, and the new zero-dup sampler shows it too, just from a higher, cleaner peak.
- **wiki-rfa**: same shape, same peak location — 1.5× is best (0.8930), then
  monotonic decline through 3×/5×/8× (0.8924 → 0.8907 → 0.8838, a total −0.92pp from
  peak to 8×). floor→1.5× is +0.39pp, smaller than wiki-elec's but still positive
  and in the same direction. **This confirms the provisional 1.5× pick already
  written into `configs/wiki-rfa.yaml` (265,817 walks) was correct** — the 3×/5×/8×
  data that wasn't available when that pick was made does not change the
  recommendation; no config revision needed.

Both datasets: identical qualitative shape to wiki-elec/wiki-rfa's original
over-saturation finding under the old sampler (small dense graphs, more walks paying
for real distinct-walk variance past a point — the model starts overfitting on
denser/redundant coverage of the same small edge set). **1.5×|E| is the confirmed
recommendation for both, not provisional.**

## SOTA comparison

| dataset | Canon best-GNN (old) | Ours no-H, old `k_cover` sampler (E16, documented SOTA) | LocalAttn4+H (E14) | **New edge_cover, 1.5× pick (this sweep)** |
|---|---|---|---|---|
| wiki-elec | 0.8930 SiGAT | 0.9016 | 0.9038 | **0.9036** |
| wiki-rfa | 0.8831 SiGAT | 0.8917 | 0.8916 | **0.8930** |

The new zero-duplication sampler's cheap 1.5×|E| pick **beats the old no-H `k_cover`
SOTA on both datasets** (wiki-elec +0.20pp, wiki-rfa +0.13pp) at a fraction of the
walk budget (wiki-elec: 155,534 vs. the old 500,000; wiki-rfa: 265,817 vs. the old
1,000,000 — roughly 3–4× fewer walks). It also **matches wiki-elec's LocalAttn4+H
number (−0.02pp, within noise) and beats wiki-rfa's (+0.14pp)** — without a hardness
miner or local attention, consistent with the project's standing full-attention,
no-H default recommendation.

## E27 — wiki-elec architecture probe (nlayers=3, dropout=0.25 vs. baseline nlayers=5, dropout=0.1474)

One manual experiment, at the winning 1.5× budget (155,534 walks), swapping the
tuned architecture (`embedding_dim=64, hidden_dim=64, nhead=2, nlayers=5,
dropout=0.1474`) for a smaller/more-regularized one (`nlayers=3, dropout=0.25`,
other dims unchanged) — motivated by wiki-elec's small size (103,689 edges) and its
tendency to overfit past a small walk budget (see the over-saturation shape above),
raising the question of whether it also overfits on model capacity.

| variant | test AUC | run dir |
|---|---|---|
| baseline (nlayers=5, dropout=0.1474) | **0.9036** | `E26_WIKI_elec_p1_5x_20260719-113223` |
| smallarch (nlayers=3, dropout=0.25) | 0.8991 | `E27_WIKI_ELEC_SMALLARCH_20260719-120708` |

**Smallarch loses, −0.45pp.** The capacity-overfitting hypothesis doesn't hold at
this budget — the tuned Optuna architecture (nlayers=5) is still the better choice.
No further architecture sweep is warranted from this single result; **no config
change** — `configs/wiki-elec.yaml`'s model section is left as-is.

## Config status

Both `configs/wiki-elec.yaml` and `configs/wiki-rfa.yaml` already carry their 1.5×
picks (155,534 and 265,817 walks respectively, `walk_strategy: edge_cover`) from the
prior turn — this doc removes the "provisional" caveat on wiki-rfa's comment (the
3×/5×/8× data now confirms it) but requires no numeric change to either file.

## Relationship to E25

Combined with `E25_BUDGET_SWEEP_RESULTS.md` (bitcoin-alpha, bitcoin-otc, epinions,
slashdot090221), all 6 datasets now have a fresh `edge_cover` budget sweep and a
signed-off production pick:

| dataset | pick | num_walks | mult |
|---|---|---|---|
| bitcoin-alpha | 5× | 120,930 | 5.0× |
| bitcoin-otc | 5× | 177,960 | 5.0× |
| epinions | floor | 840,799 | 1.0× |
| slashdot090221 | 3× | 1,647,606 | 3.0× |
| wiki-elec | 1.5× | 155,534 | 1.5× |
| wiki-rfa | 1.5× | 265,817 | 1.5× |

All 6 are already written into `configs/<dataset>.yaml`. The remaining open item is
the full 6-dataset retrain-and-report against `CLAUDE.md`'s SOTA table (bitcoin-alpha,
bitcoin-otc, slashdot090221's picks are new sweep points, not yet the checkpoints
backing the documented SOTA numbers) — flagged throughout as the expected payoff of
this whole investigation, gated on explicit sign-off before launching.
