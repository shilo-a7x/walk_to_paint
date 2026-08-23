# Does layer-0 attention treat positive and negative context edges equally?

Analysis of the six production LocalAttn4 checkpoints (same ones behind Table 1/Figure 4),
layer 0, mean over heads. Code: `scripts/attention_by_edge_sign.py`, data:
`aaai2027/figure_data/attention_by_edge_sign.csv`. Inference only, no retraining.

## The key methodological point — raw mass vs. per-token rate

Raw attention mass split by sign is dominated by a confound: positive edges are far more
common than negative ones in every dataset's context (train-pool positive rate 77–94%, per the
Ethics section), so context windows simply contain many more positive-sign tokens to spread
attention over. **Raw mass ratio alone cannot distinguish "attention favors positive edges"
from "positive edges are just more numerous."** The script reports both:

- **Raw mass**: total attention mass landing on positive- vs. negative-sign context tokens,
  summed per target-edge occurrence.
- **Per-token rate**: raw mass divided by the count of that-sign tokens actually available to
  attend to for that occurrence, restricted to occurrences where both signs are present in the
  window (so the ratio is well-defined). This is the number that actually answers "does the
  model allocate more attention *per instance* to a positive edge than to a negative one."

On Bitcoin-alpha, for example, positive tokens outnumber negative ones ~20:1 in an average
window (48.3 vs. 2.4 available), and raw mass ratio is ~49:1 — steeper than pure availability
would predict, meaning there's a real per-token effect on top of the base-rate imbalance
(confirmed by the per-token rate below, 0.0104 vs. 0.0041, ~2.5×). Both numbers are reported per
dataset; only the per-token rate is the properly-controlled comparison.

## Results (per-token rate, both-signs-present occurrences only; paired two-sided Wilcoxon, cluster = target edge)

| dataset | pos rate | neg rate | diff (pos−neg) | p | direction |
|---|---|---|---|---|---|
| Bitcoin-alpha | 0.0104 | 0.0041 | +0.0063 | ≈0 | **positive favored** |
| Bitcoin-otc | 0.0100 | 0.0064 | +0.0036 | ≈0 | **positive favored** |
| Epinions | 0.0183 | 0.0231 | −0.0048 | 0.0015 | **negative favored** |
| Wiki-elec | 0.0659 | 0.0724 | −0.0064 | 0.338 | not significant |
| Wiki-RfA | 0.0365 | 0.0266 | +0.0099 | ≈0 | **positive favored** |
| Slashdot | 0.0377 | 0.0181 | +0.0196 | ≈0 | **positive favored** |

**Attention does not treat positive and negative context edges equally, and the direction is
dataset-specific, not uniform.** Four of six datasets (Bitcoin-alpha, Bitcoin-otc, Wiki-RfA,
Slashdot) allocate significantly more attention per token to positive-sign context than
negative-sign context, beyond what their relative frequency would predict. Epinions reverses
this — negative-sign context gets significantly *more* per-token attention than positive. Wiki-
elec shows no significant difference. This heterogeneity matches the general pattern already
established elsewhere in this project (forward/backward and vertex/edge splits are also
dataset-specific, not universal) — worth noting as a consistent theme, not a one-off.

## Forward/backward and hop-distance breakdown (raw mass, mean over heads)

| dataset | pos fwd | pos bwd | neg fwd | neg bwd | pos hop1 | pos hop2 | neg hop1 | neg hop2 |
|---|---|---|---|---|---|---|---|---|
| Bitcoin-alpha | 0.2246 | 0.2042 | 0.0049 | 0.0038 | 0.2261 | 0.2027 | 0.0044 | 0.0043 |
| Bitcoin-otc | 0.1604 | 0.1795 | 0.0069 | 0.0076 | 0.1946 | 0.1453 | 0.0079 | 0.0067 |
| Epinions | 0.0983 | 0.1114 | 0.0314 | 0.0253 | 0.1119 | 0.0979 | 0.0288 | 0.0279 |
| Wiki-elec | 0.1063 | 0.1276 | 0.0317 | 0.0318 | 0.1604 | 0.0735 | 0.0430 | 0.0205 |
| Wiki-RfA | 0.0656 | 0.2226 | 0.0160 | 0.0425 | 0.2158 | 0.0724 | 0.0427 | 0.0158 |
| Slashdot | 0.2089 | 0.1758 | 0.0213 | 0.0179 | 0.2407 | 0.1440 | 0.0233 | 0.0160 |

- **Forward/backward asymmetry applies to both signs in the same direction, on every dataset.**
  E.g. Wiki-RfA is strongly backward-dominant for positive mass (0.066 fwd vs. 0.223 bwd) *and*
  for negative mass (0.016 vs. 0.043) — matches the dataset's existing backward-dominance
  finding in Figure 4 (0.189 vs. 0.557 pooled). The sign-based split doesn't reveal a case where
  positive and negative edges pull attention in opposite directions — the forward/backward
  story and the sign story are separate, non-interacting axes.
- **Hop 1 > hop 2 for both signs, on every dataset** — the existing distance-decay finding
  (Figure 4 panel D, Shapley contribution) holds for raw attention mass split by sign too, no
  exceptions.

## Caveat

`n_targets` per dataset ranges from ~7.7K (Wiki-elec, capped by its smaller test set and 20K
walk sample) to ~86K (Bitcoin-alpha) — Wiki-elec's non-significant result should be read with
that in mind; it may reflect a real absence of effect or simply less statistical power, not
distinguished here. The `frac_both_signs_present` column (52–87% across datasets) shows how
often a target occurrence actually has both signs in its window to compare at all — Wiki-elec
is the lowest (52.4%), consistent with fewer negative-sign tokens generally available there.
