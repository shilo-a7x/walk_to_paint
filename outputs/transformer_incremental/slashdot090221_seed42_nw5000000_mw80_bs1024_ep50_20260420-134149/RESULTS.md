# Transformer Incremental Experiments

## Protocol
- Dataset: `slashdot090221`
- Seed: `42`
- Device: `cuda:1`
- Effective budget: `num_walks=5000000`, `max_walk_length=80`, `batch_size=1024`, `epochs=50`
- Shared isolated tmp dir: `/home/eng/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/slashdot090221_seed42_nw5000000_mw80_bs1024_ep50_20260420-134149/tmp_data`
- Optional callbacks forced OFF for all runs

## Data Safety Check
- Original `data/<dataset>` `.pt` artifacts unchanged: `False`
- Changes detected:
  - REMOVED: /home/eng/shilo_avital/yolo_lab/walk_to_paint/data/slashdot090221/dataset_cache.pt

## Results
| Exp ID | Change | Seed | Best Epoch | Val AUC | Test AUC | Train Loss | Val Loss | Gap Loss | Gap AUC | Runtime/Epoch (min) | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E14_HARDNODE_L10 | Hard-node reweight λ=1.0 on E13 stack (miner 16/16/2/2, 5 epochs, simple accuracy) | 42 | 20 | 0.9279 |  |  |  |  | 0.0562 | 9.22 | Baseline |

## Next Iteration Recommendation
- No improvement found; revert to baseline and test a different regularization axis.
