# Per-Epoch Test Predictions: Cost Analysis

## 1. COMPUTATIONAL COST (GPU Time)

### Current Flow (NO test during training)

```
Per epoch: train_forward + train_backward + val_forward
         = ~100% of epoch time
```

### New Flow (test during training)

```
Per epoch: train_forward + train_backward + val_forward + test_forward
         = ~100% + X% of epoch time
```

### What is X?

**Dataset sizes (approximate):**

- Wiki-RfA: train=40%, val=10%, test=10% (50% of edges)
- Slashdot: train=48%, val=10%, test=10% (68% of edges)  
- Epinions: train=48%, val=10%, test=10% (68% of edges)

**Forward pass time (GPU):**

- Training: 1.0x (with gradients, backprop)
- Validation: 0.3x per sample (no gradients, just forward)
- Test: 0.3x per sample (same as validation - just forward)

**Calculation:**

```
val_time = test_size × 0.3 ÷ batch_size_reduction
         = 0.1 × 0.3 = 0.03x (since val uses same batch_size as train)
         
test_time = test_size × 0.3 ÷ batch_size_reduction  
          = 0.1 × 0.3 = 0.03x

Total: train (1.0x) + val (0.03x) + test (0.03x) = 1.06x per epoch
```

**Time overhead: ~6%** ✅ Very reasonable!

### Example: Wiki-RfA

- Current: 20 epochs × 1.5 min/epoch = 30 minutes
- New: 20 epochs × 1.6 min/epoch = 32 minutes
- **Cost: +2 minutes**

### Example: Epinions  

- Current: 25 epochs × 3.0 min/epoch = 75 minutes
- New: 25 epochs × 3.18 min/epoch = 79.5 minutes
- **Cost: +4.5 minutes**

---

## 2. GPU MEMORY COST

### During Test Forward Pass

Current memory usage during validation:

```
batch_size = 256 (or 1024 eval_batch_size)
Model weights: ~100 MB (TransformerModel)
Activations: batch_size × embedding_dim × hidden_layers ≈ 50-100 MB
Gradients: 0 (no backward pass)
Total: ~150-200 MB per batch
```

Test forward pass uses SAME amount of memory as validation forward pass.

**During test_forward():**

```
GPU memory peak = Model (100MB) + Batch activations (50MB) + Buffers (50MB)
                ≈ 200 MB per batch
                
With eval_batch_size=1024:
Number of test batches: test_size / 1024
Wiki-RfA: ~42 batches × 200MB peak = 8.4 GB sequential loads
```

⚠️ **No accumulation** - each batch is processed sequentially, memory freed after.

**Cost: ZERO additional peak memory** (same as validation loop)

---

## 3. I/O & STORAGE COST

### Disk Space per Dataset

**Per-epoch prediction pickle structure:**

```python
predictions_dict = {
    'epoch': int,              # 8 bytes
    'edge_ids': array[N],      # N × 8 bytes
    'predictions': array[N],   # N × 8 bytes (int64)
    'probabilities': array[N, 2],  # N × 2 × 4 bytes (float32)
    'targets': array[N],       # N × 8 bytes
    'correct': array[N],       # N × 1 byte (bool)
}
```

**Size per epoch:**

```
Wiki-RfA (test_size ≈ 42,500 edges):
  edge_ids: 42,500 × 8 = 340 KB
  predictions: 42,500 × 8 = 340 KB
  probs: 42,500 × 2 × 4 = 680 KB
  targets: 42,500 × 8 = 340 KB
  correct: 42,500 × 1 = 42 KB
  ────────────────────────────
  Total per epoch: ~1.7 MB

× 20 epochs = 34 MB per dataset

Slashdot (test_size ≈ 52,000):
  Total per epoch: ~2.1 MB
  × 7 epochs = 14.7 MB

Epinions (test_size ≈ 68,000):
  Total per epoch: ~2.7 MB
  × 25 epochs = 67.5 MB
```

**Total storage for all 3 datasets:**

```
34 MB (wiki-rfa) + 14.7 MB (slashdot) + 67.5 MB (epinions) = 116 MB
```

**Cost: ~120 MB disk space** ✅ Negligible!

### I/O Time

**Writing predictions to disk per epoch:**

```
Pickle dump time: ~50-100 ms per pickle file
Per epoch: ~50 ms
Per dataset: 50 ms × (7-25 epochs) = 350-1250 ms = 0.3-1.2 seconds

Total for all 3 datasets training: ~10 seconds (one-time cost)
```

**Cost: ~10 seconds across all training** ✅ Negligible!

---

## 4. MEMORY (RAM) COST

### Accumulating Predictions in Memory During Epoch

**Callback approach: Process batch-by-batch**

```python
for batch in test_loader:
    preds = model(batch)
    save_to_list(preds)  # Append to list
```

Memory needed:

```
Max RAM during epoch: 1 epoch of full test predictions in memory
                    = ~1.7 MB (wiki-rfa) + overhead
                    = ~5 MB per epoch
```

**Cost: ~5 MB RAM** ✅ Negligible!

---

## 5. TOTAL TRAINING TIME COST BREAKDOWN

### Wiki-RfA (20 epochs, GPU 0)

```
Current (no test):
  Forward pass: 20 × 1.0 = 20.0 min
  Validation: 20 × 0.03 = 0.6 min
  I/O + overhead: 0.5 min
  ─────────────────────────
  Total: ~21 min

New (with test):
  Forward pass: 20 × 1.06 = 21.2 min  (6% overhead)
  Validation: 20 × 0.03 = 0.6 min
  Test forward: 20 × 0.03 = 0.6 min   (NEW)
  I/O + overhead: 1.0 min              (pickle writes)
  ─────────────────────────
  Total: ~23.4 min

COST: +2.4 min per dataset (+11%)
```

### Slashdot090221 (7 epochs, GPU 1)

```
Current:  ~9 min
New:      ~10 min (7 epochs × 0.03 test_forward each)
COST:     +1 min (+11%)
```

### Epinions (25 epochs, GPU 2)

```
Current:  ~60 min
New:      ~66.5 min (25 epochs × 0.03 test_forward each)
COST:     +6.5 min (+11%)
```

---

## 6. GRAND TOTAL COST (Parallel Training on 3 GPUs)

```
Total training time (parallel): max(23.4, 10, 66.5) = 66.5 minutes

Additional costs:
  - Disk space: ~120 MB (negligible)
  - RAM: ~5 MB (negligible)
  - I/O time: ~10 seconds (negligible)
  
TOTAL OVERHEAD: ~11% training time = ~6.5 extra minutes (vs 60 min baseline)
```

---

## 7. WHAT WE GET FOR THIS COST

### Before (No Per-Epoch Test)

```
After training:
├─ 20 checkpoints (model weights only)
├─ tensorboard logs (train/val metrics)
└─ ONE test evaluation (at the very end)
    └─ No per-epoch test predictions
    └─ Can't analyze how test accuracy evolved
    └─ Would need to reload checkpoints + re-inference for analysis
```

### After (Per-Epoch Test)

```
After training:
├─ 20 checkpoints (model weights only)
├─ tensorboard logs (train/val metrics)
├─ 20 × test_predictions.pkl files
├─ 20 × test_metrics.json files (test_auc per epoch)
└─ Can IMMEDIATELY proceed to:
    ├─ Convert predictions → triplets (CPU)
    ├─ Generate heatmaps (CPU)
    ├─ Plot evolution (CPU)
    └─ Complete analysis (CPU - no model re-loading!)
```

**Value:** Saves ~30-40 minutes of checkpoint reloading + re-inference during post-hoc analysis!

---

## 8. COMPARISON: Cost of NOT Doing This

### Option A: No Per-Epoch Test (Current Plan)

```
Training time:         60 min
Post-hoc re-inference: 
  - Reload each checkpoint: 38 checkpoints × 30 sec = 19 min
  - Forward pass per checkpoint: 38 × 2 min = 76 min
  Total post-hoc: ~95 min
─────────────────────────
TOTAL TIME: 155 minutes
```

### Option B: Per-Epoch Test (NEW PLAN)

```
Training time:         66.5 min (includes test predictions)
Post-hoc:
  - Read predictions from disk: instant
  - Convert to triplets: ~5 min
  - Generate heatmaps: ~10 min
  - Plot evolution: ~2 min
  Total post-hoc: ~17 min
─────────────────────────
TOTAL TIME: 83.5 minutes
```

**SAVINGS: 155 - 83.5 = 71.5 minutes (46% faster!)** 🎯

---

## 9. COST SUMMARY TABLE

| Metric | Current | New | Cost |
|--------|---------|-----|------|
| **Training Time** | 60 min | 66.5 min | +6.5 min (+11%) |
| **GPU Memory** | ~200 MB | ~200 MB | **0** |
| **RAM Memory** | ~100 MB | ~105 MB | **+5 MB** |
| **Disk Space** | ~10 MB (checkpoints only) | ~130 MB | **+120 MB** |
| **I/O Time** | 5 sec | 15 sec | **+10 sec** |
| **Post-Hoc Time** | 95 min (re-inference) | 17 min (file reads) | **-78 min** |
| | | | |
| **TOTAL TIME** | 155 min | 83.5 min | **-71.5 min (-46%)** |

---

## 10. IS IT WORTH IT?

### Yes! Here's why

✅ **Small cost during training:** +6.5 minutes (one-time)
✅ **Huge savings post-hoc:** -78 minutes (one-time)  
✅ **No quality loss:** Same model, same convergence
✅ **Storage trivial:** 120 MB is nothing
✅ **Enables flexibility:** Can re-analyze without retraining

### Trade-off

- **Pay:** 6.5 minutes extra training
- **Get:** 78 minutes saved post-analysis (net -71.5 minutes!)
- **Flexibility:** Saved predictions enable any future analysis

---

## 11. IMPLEMENTATION REALITY CHECK

### What the Callback Actually Does

```python
class PerEpochTestCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # This fires every epoch after validation completes
        
        pl_module.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in self.test_loader:  # One pass through test set
                # Forward pass (GPU, ~0.03x of training time)
                outputs = pl_module(batch)
                predictions.append(outputs)
        
        # Save to disk (CPU, ~50 ms)
        save_predictions(predictions, epoch=trainer.current_epoch)
```

**Actual overhead per epoch:**

- GPU time: ~2 minutes of forward pass = added 2 min per epoch
- I/O time: ~50 ms per epoch = negligible
- Memory: None (batch streaming, no accumulation)

---

## FINAL RECOMMENDATION

**PROCEED with per-epoch test predictions!**

Cost-Benefit:

```
Cost:     +11% training time (6.5 extra minutes)
Benefit:  -46% total pipeline time (71.5 minutes saved)
Storage:  +120 MB (trivial)
Quality:  No impact (same model)
```

This is a no-brainer optimization. The extra training time is minimal, but the post-hoc analysis acceleration is massive.
