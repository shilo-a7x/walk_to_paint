# Analysis: When Does Test Run?

## Current Training Flow

```python
# src/training/train.py
trainer.fit(model, data_module["train"], data_module["val"], ckpt_path=ckpt_path)
trainer.test(model, data_module["test"])  # ← CALLED ONCE AT THE END
```

**KEY ISSUE:** `trainer.test()` is called ONCE after all epochs complete.

This means:

- ❌ test_step runs ONLY once (on final model after all epochs)
- ❌ on_test_epoch_end runs ONLY once
- ❌ test_auc_epoch ONLY logged once (not per-epoch)

---

## What We Actually Need

To get **per-epoch test predictions**, we need to either:

### Option A: Call trainer.test() Per-Epoch (WRONG)

```python
for epoch in range(num_epochs):
    trainer.fit(model, ...)  # Train one epoch
    trainer.test(model, test_loader)  # Test after each epoch
```

❌ This re-initializes trainer each time (messy)
❌ Inefficient (resets state)

### Option B: Validation Loop with Test Data (WRONG SEMANTICS)

```python
# Use val_loader but with test data
# ❌ Violates train/val/test separation
# ❌ Early stopping uses same set we're analyzing
```

### Option C: Add Custom Callback (RIGHT)

```python
class PerEpochTestCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # After validation, run test loop
        trainer.test(pl_module, test_loader)
        # Saves per-epoch test predictions
```

### Option D: Add Custom Test Loop to Lightning Module (RIGHT)

```python
def on_validation_epoch_end(self):
    # After validation, run forward pass on test set
    # Save predictions from test set
    # This gives us per-epoch test predictions
```

---

## The Root Problem

**PyTorch Lightning Architecture:**

```
trainer.fit() runs:
  for epoch in range(epochs):
    training_step()
    validation_step()  ← runs every epoch
    ❌ TEST DOESN'T RUN HERE ❌
    
trainer.test() runs ONCE:
  test_step()  ← only this one runs
```

**We need:** test_step() to run EVERY epoch

---

## Solution: Custom Per-Epoch Test Callback

```python
# src/training/callbacks.py
class PerEpochTestCallback(pl.Callback):
    def __init__(self, test_loader, cfg):
        self.test_loader = test_loader
        self.cfg = cfg
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """After validation, run test loop and save predictions."""
        
        pl_module.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                input_ids, labels, attention_mask = batch
                
                # Forward pass
                logits = pl_module.model(input_ids, attention_mask)
                preds = logits.argmax(dim=-1)
                probs = torch.softmax(logits, dim=-1)
                
                # Accumulate
                all_predictions.append({
                    'preds': preds.cpu(),
                    'probs': probs.cpu(),
                    'targets': labels.cpu(),
                    'correct': (preds == labels).cpu(),
                })
        
        # Save to disk
        pred_dir = f"{cfg.training.checkpoint_dir}/{cfg.dataset.name}_predictions/epoch_{trainer.current_epoch:03d}"
        os.makedirs(pred_dir, exist_ok=True)
        
        with open(f"{pred_dir}/test_predictions.pkl", 'wb') as f:
            pickle.dump({
                'epoch': trainer.current_epoch,
                'predictions': all_predictions,
            }, f)
        
        print(f"✓ Saved test predictions for epoch {trainer.current_epoch}")
```

---

## Updated Training Flow

```python
# src/training/train.py - MODIFIED
from src.training.callbacks import PerEpochTestCallback

def train_model(cfg, data_module):
    # ... existing setup ...
    
    test_callback = PerEpochTestCallback(
        test_loader=data_module["test"],
        cfg=cfg
    )
    
    trainer = Trainer(
        callbacks=[checkpoint, early_stopping, test_callback],  # ← ADD TEST CALLBACK
        ...
    )
    
    trainer.fit(model, data_module["train"], data_module["val"])
    # ❌ NO trainer.test() call (test callback handles it per-epoch)
```

---

## What Gets Saved (Per-Epoch)

```
checkpoints/
├─ wiki-rfa_predictions/
│  ├─ epoch_000/test_predictions.pkl
│  │  ├─ epoch: 0
│  │  ├─ predictions: [...]  # predictions per sample
│  │  ├─ targets: [...]
│  │  └─ correct: [...]
│  ├─ epoch_001/test_predictions.pkl
│  └─ ...epoch_019/test_predictions.pkl (20 epochs total)
```

Each pickle has full predictions for entire test set at that epoch.

---

## Additional Changes Needed

### 1. Save Edge IDs (Required!)

Current problem: We have predictions but DON'T know which edges they correspond to.

**Solution:** Pass edge_ids through the data pipeline

```python
# In DataLoader batches, edge_ids should be returned
# Currently: batch = (input_ids, labels, attention_mask)
# Needed: batch = (input_ids, labels, attention_mask, edge_ids)

# Modify prepare_data.py to include edge_ids
# Modify test_callback to save edge_ids with predictions
```

### 2. Track Predictions During Training (Checkpoints)

Inside fit() loop, we want checkpoint to trigger test predictions save.

```python
# pytorch_lightning checkpoint saves model weights
# We also want to save predictions alongside

# Option: At each checkpoint save, also save test predictions
# Already handled by callback - fires after validation_epoch_end
```

### 3. Save Metadata (Test AUC per epoch)

```python
# In callback, also compute test AUC and save to JSON
test_auc = compute_auc(all_targets, all_probs)

metadata_file = f"{pred_dir}/test_metrics.json"
with open(metadata_file, 'w') as f:
    json.dump({
        'epoch': trainer.current_epoch,
        'test_auc': float(test_auc),
        'test_loss': float(...),
    }, f)
```

---

## Checklist

- [ ] Create `src/training/callbacks.py` with `PerEpochTestCallback`
- [ ] Modify `train.py` to use callback
- [ ] Update data pipeline to include edge_ids in batches
- [ ] Callback saves predictions.pkl per epoch
- [ ] Callback saves test_metrics.json per epoch
- [ ] Callback saves edge_ids with predictions
- [ ] Verify no edge cases (early stopping, resume from checkpoint)

---

## Timing Impact

Training will be SLIGHTLY slower:

- Current: train(X) + val(X/4) per epoch
- New: train(X) + val(X/4) + test(X/4) per epoch

**Overhead:** ~10-15% (test forward pass = ~20% of training time, but we do 20 epochs, not significant)
