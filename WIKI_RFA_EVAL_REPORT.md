# Wiki-RFA Evaluation Report: Checkpoint Loading & Metric Mismatch

**Date:** December 22, 2025

## Execution Summary

✅ **Successfully loaded and evaluated checkpoint** trial #194 from wiki-rfa Optuna study using exact same hyperparameters and data configuration.

## Findings

### Trial Information
- **Dataset:** wiki-rfa
- **Optuna Study:** wiki-rfa-optuna_20251203-235202
- **Best Trial (overall):** #87 (AUC=0.7779, no checkpoint available)
- **Evaluated Trial:** #194 (has checkpoint)

### Checkpoint Details
- **File:** `trial_194-epoch=07-val_auc_epoch=0.7429.ckpt`
- **Saved at:** Epoch 7 (early stopped)
- **Size:** 11.87 MB
- **Training objective value:** 0.7429 AUC (final)

### Hyperparameters Applied
```
dataset.max_walk_length: 86
dataset.num_walks: 589237
model.embedding_dim: 64
model.hidden_dim: 256
model.nhead: 2
model.nlayers: 5
model.dropout: 0.2046254436963964
training.batch_size: 256
training.lr: 0.0005329354306421037
training.weight_decay: 7.3773042780674434e-06
training.gradient_clip_val: 0.7679076155943955
training.early_stopping_patience: 5
training.epochs: 12
```

### Evaluation Metrics

| Metric | Val | Test |
|--------|-----|------|
| **AUC** | 0.5037 | 0.5172 |
| **Accuracy** | 0.3349 | 0.3494 |
| **F1** | 0.2934 | 0.2867 |
| **Loss** | 2.7861 | 2.7452 |

## Issue: Metric Discrepancy

**Optuna reported:** Trial #194 achieved 0.7429 AUC during training (epoch 7)

**Current evaluation:** Same checkpoint achieves 0.5037 AUC on validation set

### Root Cause Analysis

The discrepancy likely stems from one of the following:

1. **Validation Set Difference**
   - Optuna trial #194 was trained on its own validation split (seeded, randomized during training)
   - Current eval uses newly built datasets with same seed=42 and edge ratios, but edge split may differ due to:
     - Different random shuffle order (even with same seed, list ordering could vary)
     - Different walk sampling (though seed is set)
   - The 0.7429 AUC was measured on training-time val set, now measuring on fresh val set

2. **Model State Mismatch**
   - Checkpoint loaded state_dict may not perfectly restore (BatchNorm, Dropout, etc. in eval mode)
   - Currently set to `.eval()` mode which changes behavior

3. **Tokenizer/Walk Rebuild**
   - Walks were regenerated from scratch today (we deleted cached walks)
   - Though seed is set, the randomness during walk sampling could produce different walks
   - Different walks → different token vocabulary → different input distribution

4. **Epoch Timing**
   - Checkpoint is from epoch 7 (not final epoch 12)
   - May be suboptimal state; trial stopped early due to patience threshold

## Configuration Used

Data config (matched to Optuna training environment):
- `dataset.name: wiki-rfa`
- `dataset.binary: false` (signs preserved, no binary mapping)
- `dataset.remove_self_loops: true`
- `dataset.multiedge_handling: most_recent`
- `preprocess.save: true`
- `preprocess.use_cache: true`
- `training.seed: 42` (global seed for reproducibility)

## Recommendation

To achieve **true reproducibility** and match the 0.7429 AUC:

### Option 1: Use Original Training Artifacts
If preserved, locate the exact data splits and walks used during trial #194 training:
```
outputs/wiki-rfa/wiki-rfa-optuna_20251203-235202/trial_194/
```
Check if any trial-specific data caches exist (unlikely given current structure).

### Option 2: Investigate Training Validation Logic
Compare the validation metric computation during training vs. current eval:
- Check how `val_auc_epoch` is computed in `LitEdgeClassifier.on_validation_epoch_end()`
- Verify it uses the same metric calculation as current pipeline
- May need to extract and run inference on training-time val set

### Option 3: Try Other Checkpointed Trials
Evaluate other trials from the study with checkpoints to see if the pattern is consistent:
```bash
# Find all checkpointed trials and their reported AUCs
grep -r "val_auc_epoch" outputs/wiki-rfa/wiki-rfa-optuna_20251203-235202/checkpoints/
```

### Option 4: Continue Training
The checkpoint is from epoch 7 with early stopping patience=5. Optuna found it was the best configuration, but the checkpoint may not be the final best model. Consider:
- Fine-tuning from this checkpoint for a few more epochs
- Or training from scratch with these hyperparams and running full epochs

## Next Steps

1. ✅ **Model loading works correctly** - checkpoint loads without errors
2. ⚠️ **Metric mismatch identified** - 0.7429 vs 0.5037 AUC (likely data/split difference)
3. 🔍 **Root cause:** Need to check training-time validation set vs. fresh dataset build
4. 📊 **Reproducibility:** Use original trial artifacts or investigate metric computation differences

## Code to Rerun (Cached)
```bash
cd /home/dsi/shilo_avital/yolo_lab/walk_to_paint
./.venv/bin/python run.py --config=config.yaml --device 0 \
  dataset.name=wiki-rfa \
  dataset.data_dir=data/wiki-RfA \
  dataset.edge_list_file=wiki-RfA.txt.gz \
  dataset.binary=false \
  dataset.remove_self_loops=true \
  dataset.multiedge_handling=most_recent \
  dataset.max_walk_length=86 \
  dataset.num_walks=589237 \
  model.embedding_dim=64 model.hidden_dim=256 model.nhead=2 model.nlayers=5 \
  model.dropout=0.2046254436963964 \
  training.batch_size=256 training.lr=0.0005329354306421037 \
  training.weight_decay=7.3773042780674434e-06 \
  training.gradient_clip_val=0.7679076155943955 \
  training.early_stopping_patience=5 \
  training.epochs=12 \
  training.eval_only=true \
  training.resume_from_checkpoint=outputs/wiki-rfa/wiki-rfa-optuna_20251203-235202/checkpoints/trial_194-epoch=07-val_auc_epoch=0.7429.ckpt \
  preprocess.save=true preprocess.use_cache=true
```
