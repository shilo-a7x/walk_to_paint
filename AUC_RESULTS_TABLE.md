# Final AUC Results

> **⚠️ STALE / SUPERSEDED (2026-06-29).** The numbers in the "Historical" table below are the
> old **uniform-sampler** results (~85–88% walk coverage on the sparse graphs, posthoc via
> LGBM/mixed aggregators). The current SOTA is the **E15 `k_cover` k=5** full-coverage model
> (~100% node+edge coverage on all 6) with the `func_logit_power` aggregator. The authoritative
> table is in `CLAUDE.md`; full detail in `outputs/walk_coverage_analysis/E15_SWEEP_RESULTS.md`.

## Current SOTA (E15 k_cover k=5, posthoc func_logit_power, test AUC)

Old uniform-E14 numbers in parentheses.

| Dataset | Ours (full attn) | LocalAttn4 |
|---|---:|---:|
| bitcoin-alpha | 0.9251 (0.9131) | 0.9362 (0.9370) |
| bitcoin-otc | 0.9427 (0.9431) | 0.9410 (0.9337) |
| epinions | 0.9562 (0.9311) | 0.9568 (0.9445) |
| wiki-elec | 0.9016 (0.8928) | 0.9038 (0.8917) |
| wiki-rfa | 0.8932 (0.8810) | 0.8916 (0.8882) |
| slashdot090221 | 0.9012 (0.8952) | 0.8984 (0.8958) |

At ~100% coverage the walk model beats every GNN baseline on all 6 datasets, both attention
variants, on identical shared test edges (apples-to-apples).

---

## Historical (old uniform sampler, kept for provenance)

These were our finalized results from an earlier chat.
`Transformer AUC` is the walk-level test AUC, and `LGBM Edge AUC` is the posthoc edge-level aggregated test AUC.

| Dataset | Transformer AUC | LGBM Edge AUC |
|---|---:|---:|
| bitcoin-alpha (5M) | 0.8993 | 0.9129 |
| bitcoin-otc | 0.9229 | 0.9433 |
| wiki-elec | 0.8562 | 0.8826 |
| wiki-rfa | 0.8415 | 0.8815 |
| epinions | 0.9334 | 0.9446 |
| slashdot (5M) | 0.8818 | 0.8954 |

Quick read:
- Aggregation improves over transformer-only for all six datasets in this final set.
- The largest remaining gap is slashdot, while bitcoin-otc, wiki-elec, wiki-rfa, and epinions are the strongest overall.
