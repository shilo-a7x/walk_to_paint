# Complete Retrain + Heatmap Analysis Plan (Optimized)

## GOAL

Retrain 3 datasets AND complete all heatmap analysis without re-running inference.

```
DURING TRAINING:                        AFTER TRAINING:
├─ Run trainer                          ├─ Read predictions from disk
├─ Save predictions per-epoch ✅        ├─ Convert to triplets (CPU)
├─ Save test_auc per-epoch ✅           ├─ Generate heatmaps (CPU)
├─ Save checkpoints (model only)        ├─ Plot evolution (CPU)
└─ Save config + hparams               └─ Comprehensive analysis (CPU)
    (NO MODEL INFERENCE AGAIN!)             (NO MODEL LOADING!)
```

---

## PHASE 1: PREPARE TRAINING CODE (Modifications Needed)

### 1.1 Add Per-Epoch Prediction Saving to LitModel

**Where:** `src/model/lit_model.py` in `on_test_epoch_end()`

**What to add:**

```python
def on_test_epoch_end(self):
    # ... existing code ...
    
    # NEW: Save predictions to disk
    predictions_dir = os.path.join(
        self.cfg.training.checkpoint_dir,
        f"{self.cfg.dataset.name}_predictions",
        f"epoch_{self.current_epoch:03d}"
    )
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Save test predictions
    test_preds_path = os.path.join(predictions_dir, "test_predictions.pkl")
    self._save_epoch_predictions(test_preds_path)
```

**New method to add:**

```python
def _save_epoch_predictions(self, filepath):
    """Save accumulated predictions from test_step to disk."""
    # Get predictions from metrics_manager
    predictions_data = {
        'epoch': self.current_epoch,
        'edge_ids': self.accumulated_edge_ids,      # Need to track these
        'predictions': self.accumulated_predictions,  # argmax labels
        'probabilities': self.accumulated_probs,     # softmax probs
        'targets': self.accumulated_targets,         # true labels
        'correct': self.accumulated_correct,         # pred == target
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(predictions_data, f)
    
    print(f"✓ Saved epoch {self.current_epoch} predictions: {filepath}")
```

**Implementation challenge:** Need to track edge_ids + predictions during test_step

- Modify `_step()` to accumulate predictions
- Store in instance variables
- Save in `on_test_epoch_end()`

### 1.2 Add Per-Epoch Test AUC Export

**Where:** `src/model/lit_model.py` in `on_test_epoch_end()`

**What to add:**

```python
# Save test metrics JSON per epoch
metrics_json = {
    'epoch': self.current_epoch,
    'test_auc': float(results['auc']),
    'test_acc': float(results['accuracy']),
    'test_f1': float(results['f1']),
    'test_loss': float(self.last_test_loss),
}

metrics_dir = os.path.join(
    self.cfg.training.checkpoint_dir,
    f"{self.cfg.dataset.name}_metrics"
)
os.makedirs(metrics_dir, exist_ok=True)

with open(os.path.join(metrics_dir, f"epoch_{self.current_epoch:03d}.json"), 'w') as f:
    json.dump(metrics_json, f)
```

### 1.3 Validate Config Covers Eval Batch Size

**Where:** `config.yaml` and `configs/{dataset}.yaml`

**Add to base config:**

```yaml
training:
    batch_size: 64
    eval_batch_size: 1024  # ← ADD THIS (separate from training)
```

**Update prepare_data.py:**

```python
def make_dataloaders(cfg, train_pack, val_pack, test_pack):
    train_batch_size = int(cfg.training.batch_size)
    eval_batch_size = int(getattr(cfg.training, "eval_batch_size", train_batch_size))
    
    # Use different batch sizes
    train_loader = DataLoader(..., batch_size=train_batch_size)
    val_loader = DataLoader(..., batch_size=eval_batch_size)    # ← Use eval_batch_size
    test_loader = DataLoader(..., batch_size=eval_batch_size)   # ← Use eval_batch_size
```

---

## PHASE 2: UPDATE POST-HOC ANALYSIS SCRIPTS

### 2.1 Create New Script: `scripts/triplets_from_predictions.py`

**Input:** Predictions pickle files from training (FAST - file I/O only)

**Output:** Triplets pickle files (ready for heatmap)

```python
#!/usr/bin/env python3
"""
Convert saved predictions → triplets WITHOUT loading model.

Input:
  checkpoints/{dataset}_predictions/epoch_*/test_predictions.pkl

Output:
  outputs/{dataset}/epoch_analysis/{run_id}/{epoch}/triplets_test.pkl
"""

def load_predictions(pred_path):
    """Load pre-computed predictions from disk."""
    with open(pred_path, 'rb') as f:
        return pickle.load(f)

def predictions_to_triplets(predictions, walks, edge_list):
    """Convert predictions to triplets using walk data.
    
    CPU-only operation - no model inference.
    """
    triplets = {
        'dist_from_start': [],
        'dist_from_end': [],
        'correct': [],
    }
    
    for edge_id, correct in zip(predictions['edge_ids'], predictions['correct']):
        # Look up walk distances for this edge
        # (from pre-computed walk data)
        dist_start = walk_distances[edge_id]['start']
        dist_end = walk_distances[edge_id]['end']
        
        triplets['dist_from_start'].append(dist_start)
        triplets['dist_from_end'].append(dist_end)
        triplets['correct'].append(correct)
    
    return triplets
```

**Key point:** This is just Python/NumPy - NO GPU, NO model loading

### 2.2 Update Script: `scripts/plot_triplet_heatmap.py`

Change input source from:

```python
# OLD: Load triplets from aggregation/strategy_mean/
triplets = load_triplets("outputs/aggregation/dataset/strategy_mean/triplets_test.pkl")
```

To:

```python
# NEW: Load triplets from epoch_analysis (already computed post-training)
triplets = load_triplets(f"outputs/dataset/epoch_analysis/{run_id}/epoch_*/triplets_test.pkl")
```

### 2.3 Create Master Script: `scripts/analyze_retrained_models.py`

**Orchestrates entire post-hoc pipeline:**

```
Input: Predictions saved during training
       ├─ checkpoints/{dataset}_predictions/epoch_*/test_predictions.pkl
       └─ checkpoints/{dataset}_metrics/epoch_*.json

Pipeline:
Step 1: For each epoch:
  ├─ Load predictions (file I/O)
  ├─ Convert to triplets (CPU)
  ├─ Generate heatmap (CPU)
  └─ Save heatmap + stats

Step 2: Aggregate:
  ├─ Load all heatmap_stats.json
  ├─ Generate metrics_evolution.png
  ├─ Generate interactive HTML sliders
  └─ Create comprehensive analysis report

Output: ALL analysis results (no model re-inference needed!)
  ├─ heatmap_*.png (per epoch)
  ├─ heatmap_*_data.npz (per epoch)
  ├─ heatmap_*_stats.json (per epoch)
  ├─ metrics_evolution.png
  ├─ evolution_sliders_*.html
  └─ analysis_report.txt
```

---

## PHASE 3: TRAINING COMMANDS

```bash
# Wiki-RfA (GPU 0)
python run.py \
  --config configs/wiki-rfa.yaml \
  training.epochs=20 \
  training.eval_batch_size=1024

# Slashdot090221 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python run.py \
  --config configs/slashdot090221.yaml \
  training.epochs=7 \
  training.eval_batch_size=1024

# Epinions (GPU 2)
CUDA_VISIBLE_DEVICES=2 python run.py \
  --config configs/epinions.yaml \
  training.epochs=25 \
  training.eval_batch_size=1024
```

**Outputs from training:**

```
checkpoints/
├─ wiki-rfa_predictions/
│  ├─ epoch_000/test_predictions.pkl
│  ├─ epoch_001/test_predictions.pkl
│  └─ ...
├─ wiki-rfa_metrics/
│  ├─ epoch_000.json
│  ├─ epoch_001.json
│  └─ ...
├─ slashdot090221_predictions/...
├─ slashdot090221_metrics/...
├─ epinions_predictions/...
├─ epinions_metrics/...
└─ *.ckpt files (model checkpoints)
```

---

## PHASE 4: POST-HOC ANALYSIS COMMANDS

```bash
# Analyze all three datasets in parallel
parallel --jobs 3 ::: \
  "python scripts/triplets_from_predictions.py --dataset wiki-rfa" \
  "python scripts/triplets_from_predictions.py --dataset slashdot090221" \
  "python scripts/triplets_from_predictions.py --dataset epinions"

# Generate all heatmaps (CPU-only, very fast)
python scripts/analyze_retrained_models.py \
  --datasets wiki-rfa,slashdot090221,epinions

# Result: Complete analysis WITHOUT any model loading!
```

**Outputs from post-hoc:**

```
outputs/
├─ wiki-rfa/epoch_analysis/{run_id}/
│  ├─ epoch_000/
│  │  ├─ heatmap_test.png
│  │  ├─ heatmap_test_data.npz
│  │  └─ heatmap_test_stats.json
│  ├─ epoch_001/...
│  └─ metrics_evolution.png
├─ slashdot090221/epoch_analysis/...
├─ epinions/epoch_analysis/...
└─ analysis_report_{run_id}.txt
```

---

## IMPLEMENTATION CHECKLIST

### During Training (Model Code Changes)

- [ ] **Modify `src/model/lit_model.py`:**
  - [ ] Add instance variables to track edge_ids during test_step
  - [ ] Modify `_step()` to accumulate predictions
  - [ ] Add `_save_epoch_predictions()` method
  - [ ] Call in `on_test_epoch_end()`
  - [ ] Add metrics JSON export

- [ ] **Update `src/data/prepare_data.py`:**
  - [ ] Use `eval_batch_size` from config for val/test loaders

- [ ] **Update config files:**
  - [ ] `config.yaml`: Add `training.eval_batch_size: 1024`
  - [ ] All `configs/{dataset}.yaml`: No changes needed (inherit from base)

- [ ] **Verify tracking edge_ids:**
  - [ ] Ensure edge_ids passed through data pipeline
  - [ ] Verify saved in predictions pickle

### After Training (Post-Hoc Scripts)

- [ ] **Create `scripts/triplets_from_predictions.py`:**
  - [ ] Load predictions pickle (no model loading!)
  - [ ] Convert to triplets using pre-computed walk distances
  - [ ] Save triplets pickle

- [ ] **Create `scripts/analyze_retrained_models.py`:**
  - [ ] Master orchestration script
  - [ ] Calls triplets_from_predictions
  - [ ] Calls plot_triplet_heatmap
  - [ ] Calls plot_epoch_evolution
  - [ ] Calls analyze_all_heatmaps
  - [ ] All CPU-only after initial file reads

- [ ] **Update existing scripts (minimal changes):**
  - [ ] `scripts/plot_triplet_heatmap.py`: Already works, just reads from new location
  - [ ] `scripts/plot_epoch_evolution.py`: Already works
  - [ ] `scripts/analyze_all_heatmaps.py`: Already works

---

## TIMING ESTIMATE

**Training (parallel, 3 GPUs):**

- Wiki-RfA: ~30 min (20 epochs)
- Slashdot: ~15 min (7 epochs)
- Epinions: ~45 min (25 epochs)
- **Total: ~45 min (parallel)**

**Post-Hoc Analysis (CPU-only, no model loading):**

- Convert predictions → triplets: ~5 min (parallel, 3 datasets)
- Generate heatmaps: ~10 min
- Plot evolution: ~2 min
- Comprehensive analysis: ~1 min
- **Total: ~18 min**

**Grand Total: ~60 minutes** (most time is training, analysis is fast)

---

## KEY ADVANTAGES

✅ **No re-inference:** Predictions saved during training, reused in analysis
✅ **Fast post-hoc:** Only file I/O + numpy operations, no GPU needed
✅ **Flexible:** Can re-analyze without retraining
✅ **Checkpointing:** Predictions saved per-epoch for debugging
✅ **Reproducible:** Same randomness as training (no re-runs)
✅ **Scalable:** Easy to add more analysis steps without re-inference

---

## DECISION POINTS

**1. Where to save predictions?**

- Option A: `checkpoints/{dataset}_predictions/epoch_*/` ← Recommended
- Option B: `outputs/{dataset}/predictions/epoch_*/`

**2. Save both val AND test predictions?**

- Option A: Test only (what we need for heatmaps) ← Recommended
- Option B: Both val + test

**3. Include in git tracking?**

- Option A: Keep predictions locally only (large files)
- Option B: Gitignore predictions/ directory ← Recommended
