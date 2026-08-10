# Transformer Incremental Experiments

## Protocol
- Dataset: `bitcoin-otc`
- Seed: `42`
- Device: `cuda:2`
- Effective budget: `num_walks=500000`, `max_walk_length=80`, `batch_size=1024`, `epochs=100`
- Shared isolated tmp dir: `/home/eng/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/bitcoin-otc_seed42_nw500000_mw80_bs1024_ep50_20260419-133621/tmp_data`
- Optional callbacks forced OFF for all runs

## Data Safety Check
- Original `data/<dataset>` `.pt` artifacts unchanged: `True`

## Results
| Exp ID | Change | Seed | Best Epoch | Val AUC | Test AUC | Train Loss | Val Loss | Gap Loss | Gap AUC | Runtime/Epoch (min) | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E14_HARDNODE_L10 | Hard-node reweight λ=1.0 on E13 stack (miner 16/16/2/2, 5 epochs, simple accuracy) | 42 |  |  | 0.9206 |  |  |  |  | 0.53 | Baseline |

## Next Iteration Recommendation
- No improvement found; revert to baseline and test a different regularization axis.
