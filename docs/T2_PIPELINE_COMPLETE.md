# T2 Pipeline Complete: Class Imbalance with Fair Loss Weighting

**Date**: February 5, 2026  
**Status**: ✅ All Tasks Complete (T2.1 → T2.2 → T2.3)  
**Next Phase**: Ready for clean retraining

---

## Executive Summary

The T2 pipeline successfully implemented **mandatory class-weighted loss** to handle class imbalance fairly:

- ✅ **T2.1 Analysis**: Identified batch-level weighting as suboptimal
- ✅ **T2.2 Implementation**: Global train-only weights stored in config
- ✅ **T2.3 Validation**: All leakage tests pass - no data corruption

Models can now be trained with **fair loss weighting** that eliminates data leakage while maintaining reproducibility.

---

## T2.1: Analysis ✅

**Finding**: Current implementation used batch-level weighting, causing:

- Different loss values for val/test (not comparable)
- Weight variance across batches
- Subtle data leakage risk (val/test affecting their own weights)

**Data**: wiki-rfa dataset showed excellent stratification (78.44% positive in all splits)

**Report**: [docs/LOSS_WEIGHTING_ANALYSIS.md](docs/LOSS_WEIGHTING_ANALYSIS.md)

---

## T2.2: Implementation ✅

**Changes Made**:

### 1. Data Preparation (compute weights)

```python
# In src/data/prepare_data.py
class_weights = compute_class_weights_from_train(
    train_pack, cfg.model.ignore_index, cfg.model.num_classes
)
cfg.model.class_weights = class_weights
```

### 2. Model (read weights from config)

```python
# In src/model/lit_model.py
if hasattr(cfg.model, "class_weights") and cfg.model.class_weights is not None:
    self.class_weights = torch.tensor(cfg.model.class_weights, dtype=torch.float32)
```

### 3. Loss Computation (fixed weights)

```python
# In _step method
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    labels.view(-1),
    weight=self.class_weights,  # Fixed global weights
    ignore_index=self.ignore_index,
)
```

### 4. Config (removed toggle)

```yaml
# Removed: training.use_weighted_loss: true
# Reason: Weighting is now mandatory
```

**Design Pattern**:

- Weights computed during data prep (like `num_classes`, `vocab_size`)
- Stored in `cfg.model.class_weights`
- Read by model from config
- Saved in checkpoint hparams
- No external dependencies at model init

**Details**: [docs/T2_2_IMPLEMENTATION_COMPLETE.md](docs/T2_2_IMPLEMENTATION_COMPLETE.md)

---

## T2.3: Validation ✅

### Test Results

```
[1/4] Testing weights computed from train only...
  ✓ Test passed: weights = [0.8664187157003496, 1.1335812842996504]

[2/4] Testing weights stay constant during training...
  ✓ Test passed: weights remain constant

[3/4] Testing same weights for all stages...
  ✓ Test passed: same weights used for all stages

[4/4] Testing checkpoint loading...
  ✓ Test passed: fallback to uniform weights when class_weights missing

✅ ALL TESTS PASSED - No data leakage detected
```

### Test Coverage

- ✅ Weights computed from train split only (no val/test contamination)
- ✅ Weights remain constant during training (not recomputed per batch)
- ✅ Same weights applied to train/val/test (fair comparison)
- ✅ Fallback behavior for old checkpoints (backward compatible)

**Tests**: [tests/test_no_class_imbalance_leakage.py](tests/test_no_class_imbalance_leakage.py)

---

## Impact Analysis

### Before T2 (Batch-Level Weighting)

```
Problem: Weights computed per batch
Result:  train_loss ≠ val_loss (different loss functions)
Risk:    Val/test influence their own loss
Metrics: Not directly comparable across models
```

### After T2 (Global Train-Only Weighting)

```
Solution: Weights computed ONCE from train split
Result:   train_loss, val_loss, test_loss use same weights
Risk:     None (no data leakage)
Metrics:  Directly comparable across models
```

### Class Distribution (bitcoin-alpha-binary)

```
Train: 6,605 positive / 5,004 negative (56.90%)
Weights: [0.866, 1.134]  (positive gets 1.13x, negative gets 0.87x)
Effect: Balances loss contribution across classes
```

---

## Files Modified

| File | Change |
|------|--------|
| [src/data/prepare_data.py](src/data/prepare_data.py) | Added weight computation from train split |
| [src/model/lit_model.py](src/model/lit_model.py) | Read weights from config (not data_module) |
| [src/training/train.py](src/training/train.py) | Simple `LitEdgeClassifier(cfg)` init |
| [config.yaml](config.yaml) | Removed `use_weighted_loss` toggle |
| [configs/bitcoin-alpha-binary.yaml](configs/bitcoin-alpha-binary.yaml) | Removed `use_weighted_loss` toggle |
| [tests/test_no_class_imbalance_leakage.py](tests/test_no_class_imbalance_leakage.py) | Created validation tests |

---

## Formula Reference

### Inverse Frequency Weighting

For class $i$ in train split:

$$w_i = \frac{N_{total}}{C \cdot N_i}$$

Where:

- $N_{total}$ = total samples in train split
- $C$ = number of classes  
- $N_i$ = samples for class $i$ in train split

Then normalize:
$$w_i' = \frac{w_i}{\sum w_j} \times C$$

---

## Migration Guide

### For Existing Code

1. No changes needed for user code
2. `prepare_data()` now computes weights automatically
3. `LitEdgeClassifier(cfg)` reads from config (not data_module)

### For Old Checkpoints

- Load with uniform weights (fallback)
- Recommended: Retrain with new implementation
- New checkpoints have weights in hparams

### For Config Files

- Remove `use_weighted_loss` if present (ignored)
- No other changes needed

---

## Quick Start: Training with Fair Loss

```python
from src.utils.config import load_config
from src.data.prepare_data import prepare_data
from src.training.train import train_model

# Load config
cfg = load_config("config.yaml", overrides=["dataset.name=wiki-rfa"])

# Prepare data (computes class weights)
data_module = prepare_data(cfg)
print(f"Class weights: {cfg.model.class_weights}")

# Train model (uses weights from config)
model = train_model(cfg, data_module)

# Loss automatically uses fair weights
# train_loss, val_loss, test_loss all use same weights ✓
```

---

## Validation Checklist

- ✅ Weights computed from train split only
- ✅ Weights applied to all stages (train/val/test)
- ✅ No per-batch recomputation
- ✅ Config-based storage (no data_module dependency)
- ✅ Backward compatible (fallback weights)
- ✅ Saved in checkpoints
- ✅ All tests pass
- ✅ No data leakage

---

## Next Steps

### Ready for Production

1. 🔲 **Retrain all datasets** with fair loss
   - bitcoin-alpha
   - epinions
   - slashdot090221
   - wiki-rfa

2. 🔲 **Compare metrics**
   - Before (batch-level): loss metrics may vary
   - After (global): consistent fair evaluation

3. 🔲 **Update documentation**
   - Add loss weighting explanation
   - Update training guide

---

## Success Criteria Met

| Criterion | Status |
|-----------|--------|
| Loss function weighted | ✅ Yes |
| Weights from train only | ✅ Yes |
| Same weights for val/test | ✅ Yes |
| No config toggle | ✅ Removed |
| No data leakage | ✅ Verified |
| Tests pass | ✅ All 4 tests |
| Backward compatible | ✅ Fallback weights |
| Checkpoint safe | ✅ Saved in hparams |

---

## Technical Specifications

### Class Weights Computation

- **Timing**: During data preparation, after building tensors
- **Source**: Train split labels only
- **Formula**: Inverse frequency with normalization
- **Storage**: `cfg.model.class_weights` (list of floats)

### Loss Function

- **Type**: `F.cross_entropy` (PyTorch)
- **Weight parameter**: Fixed global weights
- **Applied to**: All stages (train, val, test)
- **Ignore index**: Configured via `cfg.model.ignore_index`

### Backward Compatibility

- **Old checkpoints**: Use uniform weights if missing
- **New checkpoints**: Have weights in hparams
- **Config**: Safe to add/remove `use_weighted_loss` (ignored)

---

**Pipeline Status**: ✅ Complete and Validated  
**Implementation Date**: February 5, 2026  
**Ready for**: Clean retraining with fair evaluation
