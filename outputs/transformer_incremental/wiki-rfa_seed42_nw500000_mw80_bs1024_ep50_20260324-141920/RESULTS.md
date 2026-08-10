# Transformer Incremental Experiments

## Protocol
- Dataset: `wiki-rfa`
- Seed: `42`
- Device: `cuda:1`
- Effective budget: `num_walks=500000`, `max_walk_length=80`, `batch_size=1024`, `epochs=50`
- Shared isolated tmp dir: `/home/eng/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/wiki-rfa_seed42_nw500000_mw80_bs1024_ep50_20260324-141920/tmp_data`
- Optional callbacks forced OFF for all runs

## Data Safety Check
- Original `data/<dataset>` `.pt` artifacts unchanged: `True`

## Results
| Exp ID | Change | Seed | Best Epoch | Val AUC | Test AUC | Train Loss | Val Loss | Gap Loss | Gap AUC | Runtime/Epoch (min) | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E14_HARDNODE_L05 | Hard-node reweight λ=0.5 on E13 stack (miner 16/16/2/2, 5 epochs, simple accuracy) | 42 |  |  | 0.8450 |  |  |  |  | 0.60 | Baseline |
| E14_HARDNODE_L10 | Hard-node reweight λ=1.0 on E13 stack (miner 16/16/2/2, 5 epochs, simple accuracy) | 42 |  |  | 0.8493 |  |  |  |  | 0.61 | Baseline |

## Next Iteration Recommendation
- No improvement found; revert to baseline and test a different regularization axis.
