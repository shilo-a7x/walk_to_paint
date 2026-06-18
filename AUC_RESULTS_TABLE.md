# Final AUC Results

This table summarizes our finalized evaluation results from this chat.
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
