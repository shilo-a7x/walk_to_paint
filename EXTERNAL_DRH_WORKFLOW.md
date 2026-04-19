# External User Guide: Train With D/R/H Without Incremental Runner

This guide is for users who want to run everything manually with their own configs:

- Train a miner model to build hardness map (`H`)
- Train the main model with dynamic resplit (`D`) and node replacement (`R`)
- Inject hardness map into main training (`H`)

No use of `scripts/run_transformer_incremental_experiments.py` is required.

## Concepts

- `D` (dynamic resplit): `model.dynamic_train_masking=true`
- `R` (node replacement):
  - `model.node_context_mode=replace`
  - `model.node_replace_prob`
  - `model.node_replace_unk_ratio`
- `H` (hardness reweighting):
  - `model.hardness_map_path`
  - `model.hardness_lambda`

## Prerequisites

- A dataset config exists, e.g. `configs/my-dataset.yaml`
- Python environment is active
- You know which GPU to use (`--device`)

## Step 1: Ensure dataset cache exists

`compute_hardness_map.py` needs a `dataset_cache.pt`.

Typical location after a training run:

- `<dataset.data_dir>/dataset_cache.pt`

If cache does not exist, run one short warm-up training to generate it:

```bash
/home/dsi/shilo_avital/yolo_lab/walk_to_paint/.venv/bin/python run.py \
  --device 1 \
  --config configs/my-dataset.yaml \
  training.epochs=1 \
  preprocess.use_cache=false \
  preprocess.save=true \
  training.exp_name=cache_warmup
```

## Step 2: Train miner and save hardness map

This run is independent from main training config. Use short walks and longer miner training if desired.

Example: short-walk miner (`<=7` edges) for 15 epochs.

```bash
/home/dsi/shilo_avital/yolo_lab/walk_to_paint/.venv/bin/python scripts/compute_hardness_map.py \
  --cache data/my-dataset/dataset_cache.pt \
  --out outputs/hardness/my_dataset_hardness_short7_e15.pt \
  --device 1 \
  --epochs 15 \
  --batch-size 1024 \
  --max-walk-edges 7 \
  --seed 42
```

Miner knobs:

- `--epochs`: miner training duration
- `--max-walk-edges`: if `>0`, miner uses only samples whose target walk has at most this many edges
- `--batch-size`, `--lr`, `--seed`

Output:

- `hardness_map.pt` tensor of shape `[vocab_size]` with values in `[0, 1]`

## Step 3: Train main model with D+R+H

Use your own config file and inject overrides:

```bash
/home/dsi/shilo_avital/yolo_lab/walk_to_paint/.venv/bin/python run.py \
  --device 1 \
  --config configs/my-dataset.yaml \
  training.exp_name=my_dataset_drh \
  model.dynamic_train_masking=true \
  model.dynamic_train_mask_seed_offset=0 \
  model.node_context_mode=replace \
  model.node_replace_prob=0.2 \
  model.node_replace_unk_ratio=0.7 \
  model.hardness_map_path=outputs/hardness/my_dataset_hardness_short7_e15.pt \
  model.hardness_lambda=1.0
```

## Step 4: Suggested manual ablations

Run these manually to understand contribution per knob:

- `E0`: none of D/R/H
- `D+R`: dynamic resplit + replacement
- `H-only`: hardness only
- `D+R+H`: full stack

Minimal useful lambdas:

- `model.hardness_lambda=0.5`
- `model.hardness_lambda=1.0`

## Step 5: Read outputs

Primary files:

- TensorBoard logs under `logs/` (per your config)
- Checkpoints under `checkpoints/`
- Training console logs for `val_auc_epoch`, `test_auc_epoch`

## Practical tips

- Keep miner and main model seeds fixed for fair comparisons.
- If miner loss is still dropping at final epoch, try more epochs.
- If hardness gives unstable gains, compare `--max-walk-edges 5`, `7`, and `0` (no short-walk filter).
- Keep D/R fixed while tuning H so interpretation remains clean.
