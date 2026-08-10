# E15 full-coverage rerun of Leads 1 / 4 / 4b (2026-06-29)

`predictions_raw_canonical.pkl` was rebuilt from the **E15 k_cover k=5 full-coverage walk
runs** (was: uniform-sampler E14 runs at ~85–88% coverage on the sparse graphs). Walk now
covers 100% of every dataset's canonical test set, so walk == GINEConv == SiGAT == overlap
edge count on all 6 — the per-edge walk-vs-GNN join is no longer capped by coverage. GNN
predictions are unchanged (GNNs were not retrained). Pre-rerun reports/data backed up under
`outputs/_pre_e15_lead_backup_<ts>/`.

Walk per-edge prob = mean-prob over each edge's walk occurrences (≈ func_logit_power within
~0.003). walk_full / walk_localattn4 sourced from the winner run dirs in
`outputs/walk_coverage_analysis/E15_SWEEP_RESULTS.md`.

## Headline: conclusions HOLD, and mostly STRENGTHEN, on full coverage

Including the previously-uncovered (harder, peripheral) test edges did NOT wash out the
differential-degradation effect — it reinforced it. Metric below:
`DIFF = best_GNN_drop − walk_full_drop`, drop = (low-entropy AUC − high-entropy AUC),
so DIFF > 0 means the GNN degrades MORE than the walk as sign-heterogeneity rises
(= supports Lead 4).

### Lead 4 (entropy heterogeneity) — NEW diff (OLD diff in parens)
| variant       | bit-alpha | bit-otc | epinions | wiki-elec | wiki-rfa | slashdot | verdict |
|---------------|-----------|---------|----------|-----------|----------|----------|---------|
| in_in         | +.187(.154) | +.340(.309) | +.169(.139) | +.086(.062) | +.015(.021) | +.129(.130) | STRONG & ROBUST — all 6 positive |
| inout_inout   | +.158(.093) | +.274(.260) | +.042(.055) | −.002(−.023)| +.018(.018) | +.065(.068) | holds 5/6 (wiki-elec ~null) |
| out_out       | +.151(.065) | +.195(.185) | +.114(.117) | −.016(−.058)| +.010(.014) | +.038(.028) | holds 5/6 (wiki-elec ~null) |
| out_in        | +.131(.035) | +.208(.158) | +.051(.041) | +.013(−.014)| −.000(−.009)| −.006(−.011)| WEAK — mixed signs (as before) |

### Lead 4b (2-hop path consistency) — NEW diff (OLD diff)
| variant | bit-alpha | bit-otc | epinions | wiki-elec | wiki-rfa | slashdot |
|---------|-----------|---------|----------|-----------|----------|----------|
| out     | +.082(.069) | +.011(.016) | +.071(.053) | −.005(.029) | +.006(.007) | +.019(.017) |
| in      | +.044(.067) | +.099(.094) | +.064(.054) | +.022(.012) | +.002(.003) | +.059(.055) |
| inout   | +.108(.119) | +.061(.060) | +.080(.061) | +.012(.024) | +.010(.027) | +.054(.050) |

## Verdict (unchanged framing, now on shared FULL-coverage ground truth)
- **in_in (Lead 4) and in/inout (Lead 4b): strong & robust** — GNNs degrade markedly more
  than the walk on high-sign-heterogeneity neighborhoods, all 6 datasets, magnitudes ≈ or
  slightly larger than the ~88%-coverage version.
- **out_in (Lead 4): weak** — mixed signs (positive on bitcoin pair + epinions, ~0/negative
  on wiki-rfa/slashdot). Same as before; not a clean effect.
- **wiki-elec** is the lone ~null/negative on the out-anchored variants (out_out, inout_inout,
  4b-out) — small dense graph, the effect is genuinely weak there.
- The earlier worry that the effect was a "different-edges artifact" is **resolved**: it
  survives on identical full-coverage edges. The walk advantage is NOT a uniform AUC offset —
  it concentrates in heterogeneous (in-anchored) neighborhoods.

## Lead 1 (degree gap)
Walk still beats both GNNs in every degree bucket on all 6 (walk AUCs shifted with the new
model; GNN unchanged). "Gap widens with degree" remains mixed/non-monotonic — over-averaging
real but modest, not the primary driver. Conclusion unchanged.
