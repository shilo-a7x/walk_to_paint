# Retrain Verification Checklist

Before we retrain the 3 datasets, let's verify these key points:

## 1. CONFIG AGREEMENT ✓

### Wiki-RfA (Best Trial #87)

- `embedding_dim: 64`, `hidden_dim: 64`, `nhead: 2`, `nlayers: 5`, `dropout: 0.147`
- `batch_size: 256`, `epochs: 20`, `lr: 0.001613`, `weight_decay: 2.987e-05`
- `max_walk_length: 82`, `num_walks: 434857`
- **Decisions needed:**
  - Should we use `early_stopping_patience: 5` (from optuna)?
  - Any eval_batch_size override? (currently uses training batch_size)

### Slashdot090221 (Best Trial #12, AUC=0.8529)

- `embedding_dim: 128`, `hidden_dim: 256`, `nhead: 4`, `nlayers: 4`, `dropout: 0.0055`
- `batch_size: 64`, `epochs: 7`, `lr: 0.000124`, `weight_decay: 0.000446`
- `max_walk_length: 76`, `num_walks: 4899286`
- **Decisions needed:**
  - Keep `early_stopping_patience: 17`?
  - Eval batch size?

### Epinions (Best Trial #31, AUC=0.9133)

- `embedding_dim: 32`, `hidden_dim: 32`, `nhead: 8`, `nlayers: 3`, `dropout: 0.0008`
- `batch_size: 256`, `epochs: 25`, `lr: 0.002936`, `weight_decay: 0.000293`
- `max_walk_length: 79`, `num_walks: 4966522`
- **Decisions needed:**
  - Keep `early_stopping_patience: 9`?
  - Eval batch size?

---

## 2. CHECKPOINT PRESERVATION

**Current Status:**

- Average checkpoint size: ~7-12 MB per file
- Total checkpoints directory: 639 MB
- Expected per dataset: wiki-rfa (20 epochs) = ~240 MB, slashdot (7 epochs) = ~84 MB, epinions (25 epochs) = ~300 MB

**Proposal:**

- Keep checkpoints out of git (too large)
- Use `.gitignore` for `checkpoints/` and `outputs/*/checkpoints/`
- Track ONLY essential metadata:
  - Configs (already small YAML files) ✓
  - Training logs (CSV, JSON) ✓
  - Best checkpoint pointers (filename + epoch) ✓
  - Tensorboard event files (optional - can be regenerated)

**Questions:**

1. Should we commit just the best checkpoint per dataset + hparams.yaml for reproducibility?
2. Or keep all checkpoints only locally for analysis?

---

## 3. INTEGRATION DURING TRAINING

**Current Capabilities:**

- ✓ Already logs `test_auc_epoch` per epoch in tensorboard
- ✓ Saves hyperparameters in checkpoint
- ✓ Class weights computed from train split (prevents data leakage)

**What We Want to Add:**

- [ ] Save per-epoch predictions (val + test splits) during training?
- [ ] Save test AUC per epoch separately (JSON) for easier analysis?
- [ ] Add `eval_batch_size` config parameter (currently uses training batch_size)
- [ ] Log evaluation time separately from training time?

**Key Question:** Should we:

1. **Option A:** Save predictions during training (adds I/O overhead, gives us per-epoch predictions automatically)
2. **Option B:** Keep current training clean, extract predictions post-hoc using `extract_edge_scores.py` (cleaner separation of concerns)

**Recommendation:** Option B - keep training focused, extract predictions after for our T3 pipeline

---

## 4. TRAINING/EVALUATION OPTIMIZATION

**Current Settings:**

```yaml
num_workers: 16
pin_memory: true
persistent_workers: true
prefetch_factor: 2
```

**Evaluation Batch Size:**

- Currently uses `training.batch_size` for evaluation
- **Proposal:** Add `training.eval_batch_size: 1024` (as you found, reasonable within GPU memory)
- This doesn't affect training (eval is separate) but speeds up validation/test loops

**Questions:**

1. Should we set `eval_batch_size: 1024` for all three datasets?
2. Or keep eval batch size = training batch size for consistency?
3. Any profiling we should do (training time, memory usage)?

---

## 5. ADDITIONAL CONSIDERATIONS

- [ ] **Reproducibility Lock:** Keep `reproducibility.seed: 42` locked?
- [ ] **Validation Strategy:** Use validation set for early stopping, test set ONLY for reporting?
- [ ] **Metrics Tracking:** Log all metrics to CSV for later analysis?
- [ ] **Checkpoints Cleanup:** Keep only best + last checkpoint per epoch? Or all?
- [ ] **Git LFS:** Use Git LFS for checkpoints if we want version control?
- [ ] **Checkpoint Validation:** Add code to validate checkpoint loading works correctly?

---

## DECISION SUMMARY

Before proceeding, please confirm:

**For each dataset (wiki-rfa, slashdot, epinions):**

- [ ] Config parameters AGREED
- [ ] Early stopping patience: YES / NO / CHANGE TO ___
- [ ] Eval batch size: `1024` / `keep_same_as_training` / `___`

**For checkpoints:**

- [ ] Approach: All local / Best only + git / Git LFS
- [ ] Add eval_batch_size to config.yaml

**For integration:**

- [ ] Extract predictions post-hoc (recommended) / Save during training

**Other:**

- [ ] Any other optimizations or considerations?
