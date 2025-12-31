# Checkpoint Loading Fix Summary

## Problem
Existing Optuna checkpoints couldn't reliably reproduce training metrics during eval-only runs because:
1. **No hyperparameters saved**: `LitEdgeClassifier` used `save_hyperparameters(ignore=["cfg"])` which saved nothing
2. **Architecture mismatch**: Loading with different config created wrong model architecture for the saved weights
3. **Wrong best trial**: `configs/wiki-rfa.yaml` had trial #194 params, but best was trial #87 (AUC=0.7779 vs 0.7429)
4. **Extractor bugs**: `extract_optuna_results.py` incorrectly assumed minimize direction and negated AUC

## Solutions Applied

### 1. Fixed `LitEdgeClassifier` to save full config
**File**: `src/model/lit_model.py`

Now saves the complete resolved config in checkpoint:
```python
self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=True)})
```

Allows loading without manual cfg (future checkpoints):
```python
model = LitEdgeClassifier.load_from_checkpoint(ckpt_path)  # Works!
```

For old checkpoints without saved hyperparameters, still pass cfg explicitly:
```python
model = LitEdgeClassifier.load_from_checkpoint(ckpt_path, cfg=cfg)
```

### 2. Updated dataset configs with actual best Optuna hyperparameters

**`configs/wiki-rfa.yaml`** - Trial #87 (AUC=0.7779):
- max_walk_length: 82 (was 86)
- num_walks: 434857 (was 589237)
- hidden_dim: 64 (was 256)
- lr: 0.0016135 (was 0.000533)
- dropout: 0.147 (was 0.205)
- epochs: 20 (was 12)

**`configs/slashdot090221.yaml`** - Trial #0 (AUC=0.685):
- num_walks: 10000 (was 438173)
- batch_size: 32 (was 64)
- epochs: 3 (was 25)

### 3. Fixed `extract_optuna_results.py`
- Respects `study.direction` (MAXIMIZE vs MINIMIZE)
- Sorts trials correctly by direction
- Only negates AUC when study is MINIMIZE and value < 0

### 4. Added helper scripts

**`scripts/dump_trial_overrides.py`**
```bash
python scripts/dump_trial_overrides.py \
  --study outputs/wiki-rfa/.../optuna_study_x.pkl \
  --trial 194 \
  --dotlist
```
Outputs YAML config and dotlist overrides for any trial.

**`scripts/test_checkpoint_loading.py`**
```bash
python scripts/test_checkpoint_loading.py <checkpoint> "dotlist overrides"
```
Verifies checkpoint loads with correct architecture.

## How to Load Old Optuna Checkpoints

### Step 1: Get trial hyperparameters
```bash
python scripts/dump_trial_overrides.py \
  --study outputs/wiki-rfa/wiki-rfa-optuna_20251203-235202/optuna/optuna_study_wiki-rfa-optuna_20251203-235202.pkl \
  --trial 194 \
  --dotlist
```

This outputs:
```yaml
# Trial 194 (AUC=0.7428881525993347)
dataset:
  max_walk_length: 86
  num_walks: 589237
model:
  embedding_dim: 64
  hidden_dim: 256
  ...

# Dotlist overrides:
dataset.max_walk_length=86 dataset.num_walks=589237 model.embedding_dim=64 model.hidden_dim=256 ...
```

### Step 2: Run eval-only with overrides
```bash
python run.py \
  --device 0 \
  dataset.name=wiki-rfa \
  training.eval_only=true \
  training.resume_from_checkpoint=outputs/wiki-rfa/.../trial_194-epoch=07-val_auc_epoch=0.7429.ckpt \
  dataset.max_walk_length=86 \
  dataset.num_walks=589237 \
  model.embedding_dim=64 \
  model.hidden_dim=256 \
  model.nhead=2 \
  model.nlayers=5 \
  model.dropout=0.2046254436963964 \
  training.batch_size=256 \
  training.epochs=12 \
  training.lr=0.0005329354306421037 \
  training.weight_decay=7.3773042780674434e-06 \
  training.gradient_clip_val=0.7679076155943955 \
  training.early_stopping_patience=5
```

**Important**: Must match ALL hyperparameters that affect:
- Data generation (max_walk_length, num_walks)
- Model architecture (embedding_dim, hidden_dim, nhead, nlayers, dropout)

### Step 3 (Optional): Verify correctness
```bash
PYTHONPATH=. python scripts/test_checkpoint_loading.py \
  <checkpoint_path> \
  "<dotlist_overrides>"
```

## Future Checkpoints

All NEW checkpoints (trained after this fix) will be self-describing:
```bash
# No overrides needed!
python run.py \
  --device 0 \
  training.eval_only=true \
  training.resume_from_checkpoint=<new_checkpoint>.ckpt
```

The checkpoint carries its own complete config, so model architecture and data preprocessing parameters are guaranteed to match.

## Available Checkpoints

### wiki-rfa
- **Best overall**: Trial #87 (AUC=0.7779) - **checkpoint deleted** (cleanup kept only last 10 trials)
- **Best available**: Trial #191 (epoch=04, val_auc=0.7432) at `outputs/wiki-rfa/wiki-rfa-optuna_20251203-235202/checkpoints/trial_191-epoch=04-val_auc_epoch=0.7432.ckpt`
- Trial #194 (epoch=07, val_auc=0.7429) - params currently in `configs/wiki-rfa.yaml` before fix

### slashdot090221
- Only last 10 trials preserved; best trial #0 checkpoint deleted
- Update `configs/slashdot090221.yaml` with trial #0 params (AUC=0.685)

### epinions
- No Optuna study pickle found; only checkpoints from trials 51-62
- Trial #62: epoch=03, val_auc=0.9101 (interrupted, not completed)
- Best logged: Trial #31 (AUC=0.9133) - checkpoint deleted

## Recommendation

**Retrain with best hyperparameters** to get checkpoints with:
1. Self-describing configs (no manual overrides needed)
2. Actual best trial params (not suboptimal ones)
3. Full training to convergence

Commands ready:
```bash
# wiki-rfa with trial #87 params (now in configs/wiki-rfa.yaml)
nohup python run.py --device 1 dataset.name=wiki-rfa > logs/retrain_wiki.log 2>&1 &

# slashdot090221 with trial #0 params
nohup python run.py --device 1 dataset.name=slashdot090221 > logs/retrain_slashdot.log 2>&1 &
```
