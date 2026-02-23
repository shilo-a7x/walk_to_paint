# Pre-Retrain Assessment Summary

## Status Overview

### 1. ✅ CONFIGS ARE GOOD

All three datasets have optimized configs from Optuna trials:

- **Wiki-RfA:** Trial #87
- **Slashdot090221:** Trial #12 (AUC=0.8529)
- **Epinions:** Trial #31 (AUC=0.9133)

Configs are already set in `configs/{dataset}.yaml` and are small (~1.2KB each).

### 2. ⚠️ CHECKPOINT PRESERVATION - NEEDS DECISION

**Current issue:** Checkpoints (7-12MB each) are not tracked in git.
**Total expected:** ~620 MB for all three datasets (~20+7+25 epochs)

**Options:**

- A) Keep all checkpoints locally (current approach) - safest, most storage
- B) Commit only best+final checkpoint per dataset + git-track configs
- C) Use Git LFS for checkpoints - requires setup but cleanest

**Recommendation:** Option B is practical - commit only best checkpoint + hparams for each dataset, keep all others locally

### 3. ⚠️ INTEGRATION OPPORTUNITIES - PARTIALLY DONE

**Already in code:**

- ✅ Logs test_auc_epoch per epoch (tensorboard)
- ✅ Saves hyperparameters in checkpoint
- ✅ Class weights from train split only (no data leakage)
- ✅ Reproducible seeding (seed=42)

**Missing/Can Improve:**

- ❌ `eval_batch_size` config parameter (currently uses training batch_size)
- ❌ Per-epoch predictions saved during training
- ❌ Separate JSON export of test_auc_epoch values

**Current DataLoader Implementation:**

```python
train_loader = DataLoader(..., batch_size=batch_size)
val_loader = DataLoader(..., batch_size=batch_size)  # Same as training!
test_loader = DataLoader(..., batch_size=batch_size) # Same as training!
```

**Quick Win:** Add `training.eval_batch_size: 1024` (you verified this fits in GPU memory)

### 4. ✅ TRAINING/EVALUATION APPEARS OPTIMIZED

**Current settings are good:**

```yaml
num_workers: 16              # Parallel data loading
pin_memory: true             # Fast GPU transfer
persistent_workers: true     # Keep workers alive
prefetch_factor: 2           # Prefetch batches
```

**Worker scaling:** Val/test use `num_workers//2` and `prefetch//2` (good - less overhead for non-shuffle)

**Potential improvements:**

- Evaluate impact of `eval_batch_size: 1024` on speed vs memory
- Profile GPU utilization during training vs eval
- Check if we're CPU-bound or GPU-bound

### 5. OTHER CONSIDERATIONS

**Reproducibility:**

- ✅ `seed: 42` locked in config
- ✅ Deterministic seeding per worker
- ✅ Fixed train/val/test splits

**Validation Strategy:**

- ✅ Separate val set for early stopping
- ✅ Test set only used for final evaluation

**Metrics Tracking:**

- ✅ Tensorboard logs all metrics
- ⚠️ No CSV export of per-epoch metrics (can add)

**Data Pipeline:**

- ✅ Class weights computed from train only
- ✅ All randomness seeded
- ✅ Splits are reproducible

---

## Action Items Before Retraining

### Must Decide

1. **Checkpoint storage:** Keep local / Commit best / Use Git LFS?
2. **Eval batch size:** Use 1024 for all datasets? (yes/no/varies)
3. **Add eval_batch_size to config?** (yes - easy win, should we do it now?)

### Optional Enhancements

- [ ] Export per-epoch test_auc to JSON after training?
- [ ] Profile GPU memory during training for each dataset?
- [ ] Add metrics CSV export in trainer?

### Must NOT do

- ❌ Don't change hyperparameters (Optuna already optimized)
- ❌ Don't change seed (keep 42 locked)
- ❌ Don't mix train/val splits

---

## Implementation Plan (Once Decisions Made)

1. Update `config.yaml` to add `training.eval_batch_size` field
2. Update `prepare_data.py` to use `eval_batch_size` for val/test loaders
3. Create `.gitignore` updates for checkpoint structure
4. Retrain all three datasets with current configs
5. Extract per-epoch predictions post-hoc using existing `extract_edge_scores.py`
6. Generate T3 analysis (heatmaps, evolution plots, etc.)

**Expected time:** ~2 hours total (wiki-rfa: 30min, slashdot: 15min, epinions: 45min) + post-processing
