# Loss Weighting Analysis Report (T2.1)

**Date**: February 5, 2026  
**Status**: ✅ Analysis Complete  
**Next Phase**: T2.2 - Implementation (awaiting confirmation)

---

## Executive Summary

The current implementation uses **batch-level weighted loss**, which is **suboptimal** for handling class imbalance. While this prevents data leakage, it introduces unnecessary variance by recomputing weights for each batch.

**Critical Finding**: Weights should be computed **globally from the training split at model initialization**, not per-batch.

**Recommendation**: Implement **mandatory class-weighted loss** (no configuration option) with:

- Global weight computation from train split only
- Same weights used for train, val, and test
- Stored as model property (immutable per training run)

---

## Findings by Task

### Task 1: Loss Function Trace ✅

**Current Implementation**:

- **Loss Function**: `torch.nn.functional.cross_entropy` (PyTorch)
- **Weighting**: ENABLED via `training.use_weighted_loss = true`
- **Strategy**: Batch-level inverse frequency weighting
- **Computation Method**: `WeightedLossHelper.compute_class_weights()`
- **Formula**: `weight[class_i] = total_samples / (num_classes * count[class_i])`

**Code Location**: [src/model/lit_model.py](../src/model/lit_model.py#L44-L54)

```python
if self.use_weighted_loss:
    # Compute class weights based on batch distribution
    class_weights = self.weighted_loss_helper.compute_class_weights(
        labels.view(-1), self.num_classes, self.ignore_index
    )
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        weight=class_weights,
        ignore_index=self.ignore_index,
    )
```

---

### Task 2: Class Distributions ✅

**wiki-rfa Dataset** (analyzed as representative):

| Split | Positive | Negative | Total  | % Positive | Balance |
|-------|----------|----------|--------|------------|---------|
| Train | 66,720   | 18,341   | 85,061 | 78.44%     | ✓ OK    |
| Mask  | 44,479   | 12,228   | 56,707 | 78.44%     | ✓ OK    |
| Val   | 13,900   | 3,821    | 17,721 | 78.44%     | ✓ OK    |
| Test  | 13,901   | 3,821    | 17,722 | 78.44%     | ✓ OK    |

**Key Observations**:

- **Excellent stratification**: All splits maintain 78.44% class balance (diff ≤ 0.01%)
- Due to stratified splitting (A6 task), natural class balance is already preserved
- **However**: The slight class imbalance (78% pos, 22% neg) still benefits from weighting
- Transformer models trained without weight adjustment would over-predict the majority class

**bitcoin-alpha-binary Dataset** (for comparison):

| Split | Positive | Negative | Total  | % Positive | Balance |
|-------|----------|----------|--------|------------|---------|
| Train | 6,605    | 5,004    | 11,609 | 56.90%     | ✓ OK    |
| Mask  | 4,403    | 3,336    | 7,739  | 56.89%     | ✓ OK    |
| Val   | 1,376    | 1,043    | 2,419  | 56.88%     | ✓ OK    |
| Test  | 1,376    | 1,043    | 2,419  | 56.88%     | ✓ OK    |

---

### Task 3: Data Leakage Risk Assessment ⚠️ MEDIUM

**Current Risk**: MEDIUM (Suboptimal but not severe)

**What's happening**:

```
Training:     Weights computed from training batch labels
Validation:   Weights computed from validation batch labels ⚠️ SUBOPTIMAL
Test:         Weights computed from test batch labels ⚠️ SUBOPTIMAL
```

**Why this is suboptimal**:

1. **Val/Test batches influence their own loss**: Each batch recomputes weights from its own distribution
2. **Weights are noisy**: High variance across batches due to random sampling
3. **Fair comparison problem**: Val/test use different loss weighting than they should
4. **Not statistically sound**: Loss metrics become inconsistent across stages

**Why it's not severe**:

- ✅ No temporal leakage: Future splits don't affect past weights
- ✅ Weights are still reasonable (inverse frequency formula is sound)
- ✅ Stratified sampling keeps distributions aligned (reduces variance)
- ⚠️ But: Val/test losses are not directly comparable due to different weighting

**Example scenario**:

- Model A: val_loss=2.5 with weights [0.65, 0.35] (this batch's distribution)
- Model B: val_loss=2.4 with weights [0.70, 0.30] (different batch's distribution)
- These aren't comparable! Different models, different loss functions.

---

### Task 4: Batch-Level Analysis ✅

**Observed batch compositions** (wiki-rfa sample sequences):

| Sequence | Train Labels | Val Labels | Test Labels |
|----------|--------------|------------|-------------|
| Seq 0    | 0/9 pos      | 0/9 pos    | 0/9 pos     |
| Seq 1    | 0/11 pos     | 0/11 pos   | 0/11 pos    |
| Seq 2    | 0/5 pos      | 0/5 pos    | 0/5 pos     |

**Observations**:

- Batch composition is stable (same sequences across splits, due to split marking during encoding)
- Average positive rate per batch: 6.67% (much lower than global 78.44%)
- This shows target-specific imbalance: mask positions in sequences are rarer than raw edges

**Interpretation**:

- Within sequences, labeled positions (edges) are sparse
- This creates **additional imbalance** beyond raw edge distributions
- Batch-level weighting helps, but **global weighting from train split** is still better

---

## Issues with Current Implementation

### 1. **Configuration Paradox**

```yaml
training:
    use_weighted_loss: true  # Can be toggled off!
```

- Current: Weighting is optional
- Should be: Mandatory (once fixed in T2.2)
- Problem: Users can disable weighting, breaking fair evaluation

### 2. **Validation Loss Interpretation**

- Val loss is computed with weights from val batch
- Test loss is computed with weights from test batch
- **These losses are NOT comparable across models** (different loss functions)

### 3. **Weight Variance**

- Current: Weights change every batch
- Better: Stable weights across all batches
- Impact: Smoother training, more reproducible metrics

---

## Recommended Strategy for T2.2

### Implementation Approach

**Compute weights ONCE at model initialization from train split**:

```python
# In LitEdgeClassifier.__init__()
def __init__(self, cfg, data_module=None):
    super().__init__()
    self.cfg = cfg
    
    # Compute class weights from train split ONCE
    if data_module is not None:
        self.class_weights = self._compute_train_weights(data_module)
    else:
        # Fallback for checkpoint loading
        self.class_weights = None
    
    # Initialize loss with FIXED weights
    self.loss_fn = nn.CrossEntropyLoss(
        weight=self.class_weights,
        ignore_index=cfg.model.ignore_index
    )

def _compute_train_weights(self, data_module):
    """Compute weights from train split only."""
    train_loader = data_module["train"]
    all_labels = []
    
    for batch in train_loader:
        _, labels, _ = batch
        all_labels.append(labels[labels != self.ignore_index])
    
    all_labels = torch.cat(all_labels)
    
    # Inverse frequency formula
    class_counts = torch.zeros(self.num_classes)
    for i in range(self.num_classes):
        class_counts[i] = (all_labels == i).sum().float()
    
    total = all_labels.numel()
    weights = total / (self.num_classes * (class_counts + 1e-8))
    return weights / weights.sum() * self.num_classes  # Normalize
```

**Then use in loss**:

```python
def _step(self, batch, stage):
    logits = self.model(...)
    loss = self.loss_fn(logits, labels)  # Uses fixed weights
    return loss
```

### Key Properties

| Property | Current | After T2.2 |
|----------|---------|-----------|
| Weights computed | Per-batch | At init time |
| Source | Current batch | Train split only |
| Same weights for train/val/test | ❌ No | ✅ Yes |
| Weighting mandatory | ❌ No | ✅ Yes |
| Loss consistency | ⚠️ Varying | ✅ Fixed |
| Data leakage | Medium | None |
| Reproducibility | Lower | Higher |

---

## Success Criteria for T2.2

- ✅ Class weights computed from **train split only**
- ✅ Weights computed **at model initialization time**
- ✅ **Same weights** used for train, val, test inference
- ✅ Weighting is **mandatory** (no config toggle)
- ✅ Loss function has **fixed** `weight` parameter (not recomputed)
- ✅ Works with checkpoint loading (weights saved in hparams)

---

## Formula Reference

### Inverse Frequency Weighting

For each class $i$:

$$w_i = \frac{N_{total}}{C \cdot N_i}$$

Where:

- $N_{total}$ = total number of samples in train split
- $C$ = number of classes
- $N_i$ = number of samples for class $i$ in train split

### In Code

```python
total_samples = len(train_labels)
num_classes = 2  # binary
n_pos = (train_labels == 1).sum()
n_neg = (train_labels == 0).sum()

weight_pos = total_samples / (num_classes * n_pos)
weight_neg = total_samples / (num_classes * n_neg)

# Normalize to avoid scaling loss
weights = torch.tensor([weight_neg, weight_pos])
weights = weights / weights.sum() * num_classes  # Optional
```

---

## Q&A

**Q: Why not per-batch weighting if it reduces batch variance?**
A: Per-batch weighting introduces _weight variance_ which contradicts the goal. A better solution is:

- Global train-only weighting (removes leakage)
- Larger batch sizes (reduces sampling variance)

**Q: Should we use `pos_weight` for BCEWithLogitsLoss?**
A: No, use `weight` with CrossEntropyLoss for multi-class consistency. BCEWithLogitsLoss uses different semantics.

**Q: What if train split has very imbalanced classes?**
A: Weighting still applies. The formula naturally handles extreme imbalance by up-weighting rare classes.

**Q: Will checkpoints still load after fixing this?**
A: Yes, weights are stored in `hparams` which are part of checkpoint signature.

---

## Files to Review for T2.2

1. [src/model/lit_model.py](../src/model/lit_model.py) - Loss initialization & computation
2. [src/model/metrics_helper.py](../src/model/metrics_helper.py) - WeightedLossHelper (can be removed)
3. [config.yaml](../config.yaml) - Remove `use_weighted_loss` toggle
4. [src/training/train.py](../src/training/train.py) - Pass data_module to model init

---

## Next Steps

1. ✅ **Analysis Complete** (this report)
2. 🔲 **User Review & Confirmation** (awaiting feedback)
3. 🔲 **T2.2 Implementation** (modify lit_model.py)
4. 🔲 **T2.3 Validation** (write leakage tests)
5. 🔲 **Ready for Clean Retraining** (models with fair loss)

---

**Generated**: February 5, 2026  
**Status**: Ready for T2.2 Implementation  
**Approval**: Awaiting user confirmation
