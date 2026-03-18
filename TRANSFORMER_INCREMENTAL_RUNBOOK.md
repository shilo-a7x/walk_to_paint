# Transformer Incremental Improvements Runbook

## Prioritized Plan

### Basic (run first)
1. `E0_BASELINE` — baseline lock using current optimized values from `configs/<dataset>.yaml`.
2. `E1_SMALLER_MODEL` — reduce capacity (`embedding_dim=16`, `hidden_dim=16`, `nhead=4`, `nlayers=2`).
3. `E2_DROPOUT_UP` — increase dropout to `0.2`.
4. `E3_WEIGHT_DECAY_UP` — multiply weight decay by `10`.
5. `E4_EARLY_STOP_TIGHTER` — tighten patience to `<=5`.
6. `E5_LR_DOWN` — halve learning rate.

### Moderate (after single-factor winners)
1. Re-run top 2 candidates across 2 additional seeds.
2. Pairwise combinations of top single-factor winners.
3. Validate best pair on one additional dataset.

### Advanced (only if moderate is stable)
1. Add config-gated representation ideas (dynamic masking / node dropout).
2. Evaluate compact bottleneck variants.
3. Run curriculum experiments.

## Safety + Isolation Guarantees

- Every run uses an isolated temporary `dataset.data_dir` under `tmp/transformer_incremental/...`.
- Raw edge file is copied into temp dir; preprocessing cache is built and reused there.
- `dataset.num_walks` is fixed to `500000` for all experiments (speed + clean delta comparisons).
- Outputs are isolated per suite and per experiment under `outputs/transformer_incremental/...`.
- Optional callbacks are always forced OFF:
  - `training.callbacks.enable_prediction_saver=false`
  - `training.callbacks.enable_per_epoch_test_runner=false`
- Preferred GPU is defaulted to `--device 1`.
- Script snapshots original `data/<dataset>/*.pt` before/after and reports any modification.

## Run Commands

Full first wave (YAML baseline values, resilient to disconnect):

```bash
nohup .venv/bin/python scripts/run_transformer_incremental_experiments.py \
  --dataset wiki-rfa \
  --seed 42 \
  --device 1 \
  > outputs/transformer_incremental_wiki_rfa_wave1.log 2>&1 &
```

Fast baseline mode (optional, overrides to `80`/`1024`; `500K` walks is always enforced):

```bash
nohup .venv/bin/python scripts/run_transformer_incremental_experiments.py \
  --dataset wiki-rfa \
  --seed 42 \
  --device 1 \
  --epochs 5 \
  --fast-baseline \
  > outputs/transformer_incremental_wiki_rfa_fast.log 2>&1 &
```

Optional dry-run (prints selected experiments/paths, no training):

```bash
.venv/bin/python scripts/run_transformer_incremental_experiments.py \
  --dataset wiki-rfa \
  --device 1 \
  --dry-run
```

Run only baseline + one ablation:

```bash
nohup .venv/bin/python scripts/run_transformer_incremental_experiments.py \
  --dataset wiki-rfa \
  --device 1 \
  --run-ids E0_BASELINE,E2_DROPOUT_UP \
  > outputs/transformer_incremental_wiki_rfa_e0_e2.log 2>&1 &
```

## Results Table Template

| Exp ID | Change | Seed | Best Epoch | Val AUC | Test AUC | Train Loss | Val Loss | Gap Loss (`val_loss-train_loss`) | Gap AUC (`train_auc-val_auc`) | Runtime/Epoch (min) | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E0_BASELINE | YAML baseline | 42 |  |  |  |  |  |  |  |  | Baseline |
| E1_SMALLER_MODEL | Capacity down | 42 |  |  |  |  |  |  |  |  |  |
| E2_DROPOUT_UP | Dropout up | 42 |  |  |  |  |  |  |  |  |  |
| E3_WEIGHT_DECAY_UP | Weight decay up | 42 |  |  |  |  |  |  |  |  |  |
| E4_EARLY_STOP_TIGHTER | Tighter early stop | 42 |  |  |  |  |  |  |  |  |  |
| E5_LR_DOWN | LR down | 42 |  |  |  |  |  |  |  |  |  |

## Output Artifacts

For each suite run:
- `outputs/transformer_incremental/<suite_tag>/results.csv`
- `outputs/transformer_incremental/<suite_tag>/RESULTS.md`
- Per-run logs: `outputs/transformer_incremental/<suite_tag>/artifacts/<exp_id>/run.log`
