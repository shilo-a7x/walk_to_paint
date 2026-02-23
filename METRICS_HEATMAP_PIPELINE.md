# Metrics Tracking & Heatmap Analysis Pipeline

## TRAINING METRICS - What We Currently Track

### Per-Epoch Metrics (Logged to Tensorboard)

**Training Stage (on_train_epoch_end):**

- ✅ `train_loss` - cross-entropy loss
- ✅ `train_acc_epoch` - accuracy
- ✅ `train_f1_epoch` - F1 score
- ✅ `train_auc_epoch` - ROC AUC
- ✅ `train_confusion_matrix` - confusion matrix plot
- ✅ `train_roc_curve` - ROC curve plot
- ✅ `step` - current epoch number

**Validation Stage (on_validation_epoch_end):**

- ✅ `val_loss` - cross-entropy loss
- ✅ `val_acc_epoch` - accuracy
- ✅ `val_f1_epoch` - F1 score
- ✅ `val_auc_epoch` - ROC AUC ← **USED FOR EARLY STOPPING**
- ✅ `val_confusion_matrix` - confusion matrix plot
- ✅ `val_roc_curve` - ROC curve plot

**Test Stage (on_test_epoch_end):**

- ✅ `test_loss` - cross-entropy loss
- ✅ `test_acc_epoch` - accuracy
- ✅ `test_f1_epoch` - F1 score
- ✅ `test_auc_epoch` - ROC AUC ← **FINAL METRIC WE COMPARE**
- ✅ `test_confusion_matrix` - confusion matrix plot
- ✅ `test_roc_curve` - ROC curve plot

**ALL metrics are tracked BOTH:**

- Tensorboard event files (logs/{run}/events.out.tfevents.*)
- Checkpoint hparams (saved in checkpoint file)

### Per-Step Metrics (NOT saved currently)

- ❌ Per-batch loss (only per-epoch loss saved)
- ❌ Per-batch accuracy predictions
- ❌ Gradient norms, learning rates per step

---

## HEATMAP ANALYSIS PIPELINE - What We Need

```
┌─────────────────────────────────────────────────────────────┐
│ TRAINING (trainer, generates checkpoints per epoch)         │
├─────────────────────────────────────────────────────────────┤
│ ✅ Per-epoch test_auc_epoch logged to tensorboard           │
│ ✅ Checkpoint saved with config + hparams                  │
│ ❌ Per-epoch predictions NOT saved during training          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ EXTRACT EDGE SCORES (for each checkpoint)                  │
├─────────────────────────────────────────────────────────────┤
│ INPUT: checkpoint file + dataset config                    │
│ OUTPUT: predictions pickle with:                           │
│   - edge_id                                                 │
│   - predicted_label                                         │
│   - predicted_probs                                         │
│   - true_label                                              │
│   - correct (1/0)                                           │
│ FILES:                                                      │
│   outputs/predictions/{dataset}/raw_scores/{run_id}/...    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ SAVE TRIPLETS (convert predictions → triplets)             │
├─────────────────────────────────────────────────────────────┤
│ INPUT: predictions pickle + walk data                      │
│ OUTPUT: triplets pickle with:                              │
│   - dist_from_start (walk position from start)             │
│   - dist_from_end (walk position from end)                 │
│   - correct (1/0 prediction)                               │
│ FILES:                                                      │
│   outputs/{dataset}/epoch_analysis/{run_id}/.../           │
│   triplets_val.pkl, triplets_test.pkl                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ PLOT HEATMAP (visualize triplets as 2D grid)               │
├─────────────────────────────────────────────────────────────┤
│ INPUT: triplets pickle                                     │
│ OUTPUT:                                                     │
│   1. heatmap_val.png - visualization (dist_start vs        │
│      dist_end colored by avg accuracy)                     │
│   2. heatmap_val_data.npz - raw grid data:                 │
│      - grid (NaN where sparse)                             │
│      - counts (sample distribution)                        │
│   3. heatmap_val_stats.json - statistics:                  │
│      - sparsity, mean/std/min/max accuracy                 │
│      - boundary vs interior accuracy                       │
│      - position stats (avg distances)                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ EPOCH EVOLUTION (aggregate across all epochs)              │
├─────────────────────────────────────────────────────────────┤
│ INPUT: All checkpoint heatmap_stats.json files             │
│ OUTPUT:                                                     │
│   1. metrics_evolution.png - 3×2 grid showing:             │
│      - train_loss, val_loss, test_loss over epochs         │
│      - train_auc, val_auc, test_auc over epochs            │
│   2. 6 interactive HTML sliders (one per metric):          │
│      - Plotly sliders for navigating epochs                │
│   3. evolution_stats.json - quantified trends              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ COMPREHENSIVE ANALYSIS (cross-dataset interpretation)      │
├─────────────────────────────────────────────────────────────┤
│ INPUT: All heatmap_stats.json + metrics_evolution.json     │
│ OUTPUT:                                                     │
│   1. analysis_report.txt - written analysis covering:       │
│      - Evolution patterns (early epochs vs late)           │
│      - Dataset comparison (wiki vs slashdot vs epinions)   │
│      - Position effects (boundary vs interior learning)    │
│      - Epoch progression patterns                          │
│   2. analysis_evolution.png - cross-dataset comparison     │
│      - Overlaid accuracy curves                            │
│      - Dataset ranking over epochs                         │
└─────────────────────────────────────────────────────────────┘
```

---

## KEY INSIGHT: Epoch vs Per-Epoch Data

**Current Approach:**

- ✅ Train once per epoch → save checkpoint
- ✅ Evaluate on test set → log test_auc_epoch, save predictions
- ✅ Later: Extract per-epoch predictions from checkpoints
- ✅ Generate per-epoch heatmaps
- ✅ Analyze evolution across epochs

**Why Not Save During Training:**

- ❌ Would double I/O overhead (save model + predictions every epoch)
- ❌ Checkpoints already save config + metrics
- ✅ Post-hoc extraction is cleaner separation of concerns
- ✅ Can regenerate from checkpoints anytime

**What DO We Have Per-Epoch:**

- ✅ test_auc_epoch (from tensorboard)
- ✅ Checkpoints at each epoch
- ✅ Can extract any other per-checkpoint data later

---

## MISSING PIECES FOR HEATMAP ANALYSIS

### Currently MISSING

1. **Per-Epoch Predictions Access:**
   - ❌ Currently need to run extract_edge_scores.py separately per checkpoint
   - ❌ This is manual/scripted, not integrated into training
   - ✅ BUT: We have the extraction script working!

2. **Test AUC Export:**
   - ❌ test_auc_epoch stored only in tensorboard
   - ❌ Need to extract from event files or checkpoint names
   - ✅ SOLUTION: Parse event files or use checkpoint naming

3. **Triplet Data During Training:**
   - ❌ Triplets only computed post-hoc
   - ❌ Would require walk data during inference
   - ✅ SOLUTION: Acceptable - compute after training

### What's READY

- ✅ Per-epoch predictions can be extracted from checkpoints
- ✅ All heatmap generation code exists
- ✅ All analysis code exists
- ✅ Evolution plotting exists

---

## ACTION PLAN TO FIX MISSING PIECES

### Option A: MINIMAL (Post-hoc Only)

1. Train all 3 datasets → generates checkpoints
2. Run extract_edge_scores.py for each checkpoint
3. Run save_triplets.py for each
4. Run plot_triplet_heatmap.py for each
5. Run plot_epoch_evolution.py + analyze_all_heatmaps.py
✅ This works already! (The script we built)

### Option B: INTEGRATED (Better for Future)

1. **Add to training code:**
   - After each epoch, save test_auc_epoch to JSON file
   - Log test_auc_epoch to run metadata

2. **Create utility script to extract metrics:**

   ```
   scripts/extract_tensorboard_metrics.py
   - Reads event files from logs/
   - Exports train/val/test AUC per epoch to CSV
   - Exports to metrics_per_epoch.json
   ```

3. **Keep post-hoc extraction for predictions:**
   - Checkpoints save model state
   - Extract predictions from checkpoints afterward
   - More flexible + no extra training I/O

---

## WHAT METRICS ARE USED IN HEATMAP ANALYSIS

**Directly Used:**

1. **test_auc_epoch** - identifies which epoch is "best" for comparison
2. **test_loss** - tracks learning/overfitting
3. **Per-checkpoint predictions** - generates triplets
4. **Walk distances** (dist_from_start, dist_from_end) - X/Y axis of heatmap
5. **Correctness** (predicted == true) - Z axis (color) of heatmap

**Computed from Above:**

- Sparsity (how many cells have data)
- Boundary vs interior accuracy
- Distance effect curves
- Accuracy evolution trends

**NOT Used in Heatmap (But Interesting):**

- ❌ F1 score (only AUC for ranking)
- ❌ Confusion matrix details
- ❌ Per-class metrics
- ❌ Training time per epoch

---

## CURRENT STATUS

**For Retraining + Analysis:**

- ✅ Have all needed extraction scripts
- ✅ Have all needed analysis scripts
- ✅ Training logs to tensorboard per-epoch
- ✅ Checkpoints save per-epoch
- ✅ Can extract per-epoch predictions post-hoc

**Improvements NOT Needed Before Retraining:**

- ❌ Don't need to save predictions during training
- ❌ Don't need to export test_auc_epoch to JSON yet
- ❌ All can be done after training completes

**Ready to Proceed With:**

1. Add eval_batch_size config (easy win for speed)
2. Retrain all 3 datasets
3. Run full epoch analysis pipeline
