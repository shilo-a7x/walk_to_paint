# Transformer Incremental Experiments

## Protocol
- Dataset: `wiki-rfa`
- Seed: `42`
- Device: `cuda:1`
- Effective budget: `num_walks=500000`, `max_walk_length=80`, `batch_size=1024`, `epochs=50`
- Shared isolated tmp dir: `/home/eng/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/wiki-rfa_seed42_nw500000_mw80_bs1024_ep50_20260324-115052/tmp_data`
- Optional callbacks forced OFF for all runs

## Data Safety Check
- Original `data/<dataset>` `.pt` artifacts unchanged: `True`

## Results
| Exp ID | Change | Seed | Best Epoch | Val AUC | Test AUC | Train Loss | Val Loss | Gap Loss | Gap AUC | Runtime/Epoch (min) | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E13_DYNAMIC_RESPLIT_PLUS_NODE_REPLACE_P20 | Dynamic train/mask resplit + node replacement p=0.2 (70% UNK / 30% random node) | 42 |  |  | 0.8397 |  |  |  |  | 0.64 | Baseline |

## Next Iteration Recommendation
- No improvement found; revert to baseline and test a different regularization axis.
