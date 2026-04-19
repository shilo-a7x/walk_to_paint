# D/R/H Training Guide (Dynamic Resplit, Node Replacement, Hardness Reweighting)

This guide explains how to train models with the three main knobs:

- `D`: Dynamic train/mask resplit (`model.dynamic_train_masking`)
- `R`: Node replacement regularization (`model.node_context_mode=replace` + related probs)
- `H`: Hard-node loss reweighting (`model.hardness_map_path` + `model.hardness_lambda`)

The defaults and examples below are designed for practical tuning on a new dataset.

## 1. What each knob does

### D: Dynamic train/mask resplit

Config knobs:

- `model.dynamic_train_masking`: `true|false`
- `model.dynamic_train_mask_seed_offset`: integer offset added to base seed per epoch

Behavior:

- At each epoch, the model resamples which train-pool edges become prediction targets.
- Validation/test splits stay fixed.
- This usually reduces memorization of a static mask subset and can improve generalization.

When to use:

- Start with `true` for medium/large datasets.
- If training becomes unstable on very small datasets, test `false`.

### R: Node replacement

Config knobs:

- `model.node_context_mode`: set to `replace`
- `model.node_replace_prob`: probability of replacing node tokens in training
- `model.node_replace_unk_ratio`: among replacements, fraction replaced by `UNK`; remaining are random node tokens

Behavior:

- Applies only during train stage.
- Reduces over-reliance on specific node identity and pushes model toward structural/context signal.

Reasonable starting values:

- `model.node_replace_prob=0.2`
- `model.node_replace_unk_ratio=0.7`

### H: Hard-node reweighting

Config knobs:

- `model.hardness_map_path`: path to hardness tensor (`[vocab_size]`, float in `[0,1]`)
- `model.hardness_lambda`: non-negative scalar controlling reweight strength

Behavior:

- Training-only loss reweighting.
- Per walk, the loss is multiplied by:

  `1 + lambda * mean(h_left_node, h_right_node)`

- Easy nodes keep weight near `1.0`; hard nodes receive higher weight.

Recommended start:

- `model.hardness_lambda=1.0`

## 2. End-to-end workflow

### Step A: Baseline and D/R stack

Use the experiment runner:

```bash
/home/dsi/shilo_avital/yolo_lab/walk_to_paint/.venv/bin/python \
  scripts/run_transformer_incremental_experiments.py \
  --dataset wiki-rfa --device 1 \
  --run-ids E13_DYNAMIC_RESPLIT_PLUS_NODE_REPLACE_P20
```

### Step B: Run H on top of D/R

```bash
/home/dsi/shilo_avital/yolo_lab/walk_to_paint/.venv/bin/python \
  scripts/run_transformer_incremental_experiments.py \
  --dataset wiki-rfa --device 1 \
  --run-ids E14_HARDNODE_L05,E14_HARDNODE_L10
```

### Step C: Run H-only on E0 baseline (no D/R)

```bash
/home/dsi/shilo_avital/yolo_lab/walk_to_paint/.venv/bin/python \
  scripts/run_transformer_incremental_experiments.py \
  --dataset wiki-rfa --device 1 \
  --run-ids E14_HONLY_E0_L10
```

This gives clean attribution:

- E0 -> baseline
- E13 -> D+R effect
- E14_HONLY_E0_L10 -> H-only effect
- E14_HARDNODE_* -> H on top of D+R

## 3. Config reference for users

Core training knobs (usually tuned first):

- `training.lr`
- `training.weight_decay`
- `training.batch_size`
- `training.epochs`
- `training.early_stopping_patience`

Data knobs:

- `dataset.num_walks`
- `dataset.max_walk_length`
- `dataset.train_ratio`, `dataset.mask_ratio`, `dataset.val_ratio`, `dataset.test_ratio`

Model capacity knobs:

- `model.embedding_dim`
- `model.hidden_dim`
- `model.nhead`
- `model.nlayers`
- `model.dropout`

D/R/H knobs:

- `model.dynamic_train_masking`
- `model.dynamic_train_mask_seed_offset`
- `model.node_context_mode`
- `model.node_replace_prob`
- `model.node_replace_unk_ratio`
- `model.hardness_map_path`
- `model.hardness_lambda`

## 4. Practical defaults for a new dataset

Start with:

- `D`: enabled
- `R`: enabled with `p=0.2`, `unk_ratio=0.7`
- `H`: enabled with `lambda=1.0`
- Keep architecture and optimizer from your best non-DRH baseline

Then do a small ablation matrix:

- `E0`
- `E13` (D+R)
- `E14_HONLY_E0_L10` (H-only)
- `E14_HARDNODE_L10` (D+R+H)

Pick the best by test AUC and stability across seeds.

## 5. Cleanup plan for weak knobs (node mask/noise/dropout style knobs)

If some knobs repeatedly underperform, keep codebase tidy with staged cleanup:

### Stage 1: Freeze but keep (safe)

- Keep implementation, but remove from default experiment grid.
- Mark as "legacy/low-priority" in experiment docs.
- Avoid spending search budget on them.

### Stage 2: Soft deprecate

- Add warning when deprecated mode is used.
- Keep backward compatibility for old configs.

### Stage 3: Hard remove

- Remove unused code paths only after:
  - At least one full ablation cycle confirms no value.
  - No production configs depend on the knob.

Suggested candidates to freeze first if they keep losing:

- `model.node_context_mode=mask_unscaled`
- `model.node_context_mode=noise`

Keep these as primary:

- `dynamic_train_masking`
- `node_context_mode=replace`
- `hardness_*`

## 6. Known caveats

- Current CSV extraction may miss some val/train fields for some suites; use `run.log` as source of truth when needed.
- Checkpoint callback currently saves all epochs plus `last`; testing may use final in-memory model unless explicitly switched to best checkpoint evaluation.

## 7. Quick checklist

- Confirm dataset cache path is writable.
- Run E0, E13, H-only, and D+R+H.
- Compare both val AUC and test AUC.
- Repeat best setup on at least one extra seed before promoting to default.
