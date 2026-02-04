# Task A6: Stratified Edge Splitting for Fair Model Evaluation

## Task Overview

Currently, edges are split randomly into train/mask/val/test sets without maintaining class balance. This causes **unfair evaluation** because different stages see different class distributions.

**Goal**: Implement **stratified edge splitting** to ensure all 4 splits maintain the same positive/negative class distribution as the original dataset, enabling fair scientific evaluation.

---

## Understanding the Split Semantics

### The Four Splits and Their Roles

#### TRAIN Split (48% of edges)

- **Semantic Role**: Context edges - always visible context to the model
- **Visibility**:
  - Train stage: ✓ Visible in attention
  - Val stage: ✓ Visible in attention
  - Test stage: ✓ Visible in attention
- **As Targets**: ✗ Never masked, never predicted
- **Labels**: Not used as targets
- **Why**: Provide stable context for the model to learn from

#### MASK Split (32% of edges)

- **Semantic Role**: Training targets - what the model learns to predict
- **Visibility**:
  - Train stage: ✗ Replaced with [MASK] token → **Model predicts these**
  - Val stage: ✓ Visible (context for val/test stages)
  - Test stage: ✓ Visible (context for test stage)
- **As Targets**: Yes, only during training
- **Labels**: Used to train the model
- **Why**: Model learns on MASK edges, so they become known context afterward

#### VAL Split (10% of edges)

- **Semantic Role**: Validation targets - what the model predicts during validation
- **Visibility**:
  - Train stage: ✓ Visible but not used as target (context)
  - Val stage: ✗ Replaced with [MASK] token → **Model predicts these**
  - Test stage: ✓ Visible (context)
- **As Targets**: Yes, only during validation
- **Labels**: Used to evaluate model during validation
- **Why**: New prediction task (different from training)

#### TEST Split (10% of edges)

- **Semantic Role**: Test targets - final held-out test set
- **Visibility**:
  - Train stage: ✗ Hidden - NOT in walk data at all
  - Val stage: ✗ Hidden - NOT in walk data at all
  - Test stage: ✗ Replaced with [MASK] token → **Model predicts these**
- **As Targets**: Yes, only during testing
- **Labels**: Used to evaluate model on final test set
- **Why**: Completely held out until test time, prevents data leakage

### Key Insight: Progressive Information Disclosure

```
TRAIN stage: Can see TRAIN + MASK        (48% + 32% = 80%)
VAL stage:   Can see TRAIN + MASK + VAL  (48% + 32% + 10% = 90%)
TEST stage:  Can see TRAIN + MASK + VAL + TEST (48% + 32% + 10% + 10% = 100%)
```

This is **intentional and fair**:

- During training: model learns with limited context
- During validation: model uses what it learned during training (MASK is now visible context)
- During testing: model has full context from all previous stages
- No data leakage: TEST edges completely hidden until test time

---

## The Class Balance Problem

### Without Stratification (Current Approach)

Original dataset: **10.5% positive, 89.5% negative edges**

Random shuffle might produce:

```
TRAIN (48%):  8.2% positive  ← Model trained on fewer positive examples
MASK (32%):   12.8% positive ← Model learns to predict more positive
VAL (10%):    9.5% positive  ← Different distribution!
TEST (10%):   11.3% positive ← Different distribution!
```

**Problem**: Model trained on one distribution, evaluated on different ones. **Invalid scientific comparison.**

### With Stratification (Required)

```
TRAIN (48%):  10.6% positive ✓ Matches original
MASK (32%):   10.4% positive ✓ Matches original
VAL (10%):    10.7% positive ✓ Matches original
TEST (10%):   10.3% positive ✓ Matches original
```

**Result**: Consistent class distribution across all stages. **Fair and scientific.**

---

## Current Implementation (Problematic)

Location: `src/data/prepare_data.py`

```python
def split_edges(cfg, edges):
    edges_copy = list(edges)
    random.shuffle(edges_copy)  # ← NO STRATIFICATION!
    
    n_train = int(train_ratio * n_total)
    n_mask = int(mask_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_mask - n_val
    
    split = {
        "train": edges_copy[:n_train],
        "mask": edges_copy[n_train:n_train+n_mask],
        "val": edges_copy[n_train+n_mask:n_train+n_mask+n_val],
        "test": edges_copy[n_train+n_mask+n_val:],
    }
```

**Issue**: Random shuffling doesn't preserve class balance.

---

## Implementation Requirements

### Algorithm: Hierarchical Stratified Splitting

A 4-way stratified split cannot be done with a single `train_test_split()`. Instead, use **nested (hierarchical) splits**:

#### Step 1: Separate TRAIN from the rest (48% vs 52%)

```python
train_edges, remaining_edges, _, remaining_labels = train_test_split(
    edges_array, labels,
    train_size=cfg.dataset.train_ratio,
    stratify=labels,
    random_state=seed
)
```

#### Step 2: Separate MASK from temp (32/52% ≈ 61.5% of remaining)

```python
mask_ratio_of_remaining = cfg.dataset.mask_ratio / (1 - cfg.dataset.train_ratio)
mask_edges, temp_edges, _, temp_labels = train_test_split(
    remaining_edges, remaining_labels,
    train_size=mask_ratio_of_remaining,
    stratify=remaining_labels,
    random_state=seed
)
```

#### Step 3: Separate VAL from TEST (50/50 of remaining)

```python
test_ratio_of_temp = cfg.dataset.test_ratio / (cfg.dataset.val_ratio + cfg.dataset.test_ratio)
val_edges, test_edges, _, _ = train_test_split(
    temp_edges, temp_labels,
    test_size=test_ratio_of_temp,
    stratify=temp_labels,
    random_state=seed
)
```

### Key Points

1. **Three stratified splits** (not one random shuffle)
2. **Recalculate ratios** at each step based on remaining proportion
3. **Use same seed** throughout for reproducibility
4. **Maintain labels** through all steps for stratification

---

## Implementation Checklist

- [ ] Locate function: `split_edges()` in `src/data/prepare_data.py`
- [ ] Import `train_test_split` from `sklearn.model_selection`
- [ ] Extract labels from edges: `labels = [e[2] for e in edges]` (3rd element is label)
- [ ] Implement Step 1: TRAIN | remaining (stratify by labels)
- [ ] Implement Step 2: MASK | temp (stratify by remaining_labels)
- [ ] Implement Step 3: VAL | TEST (stratify by temp_labels)
- [ ] Ensure seed is used for reproducibility
- [ ] Return dict with "train", "mask", "val", "test" keys
- [ ] Each value is a list of edge tuples

---

## Validation Checklist

### Edge Count Validation

- [ ] `len(train) / total ≈ 0.48` (within ±0.5%)
- [ ] `len(mask) / total ≈ 0.32` (within ±0.5%)
- [ ] `len(val) / total ≈ 0.10` (within ±0.5%)
- [ ] `len(test) / total ≈ 0.10` (within ±0.5%)
- [ ] `len(train) + len(mask) + len(val) + len(test) == len(edges)`

### Class Balance Validation

For each split (train, mask, val, test):

- [ ] Count positive edges (label == 1): `pos = sum(1 for e in split if e[2] == 1)`
- [ ] Calculate percentage: `pos_pct = pos / len(split) * 100`
- [ ] Compare to original: `original_pct = sum(1 for e in edges if e[2] == 1) / len(edges) * 100`
- [ ] Verify: `|split_pct - original_pct| <= 2.0%` (within ±2%)

### No Overlap Validation

- [ ] Verify no edge appears in multiple splits
- [ ] Convert splits to sets of tuples and check intersections are empty

### Reproducibility Validation

- [ ] Run twice with same seed → get identical splits
- [ ] Run with different seed → get different splits

### End-to-End Validation

- [ ] Prepare data with new stratified splits
- [ ] Train model with new splits
- [ ] Model training completes without errors
- [ ] Verify metrics are computed for train, val, test

---

## Testing: Multi-Dataset Validation

Test implementation on all three datasets:

1. **wiki-rfa** (config.yaml)
   - Dataset: Wikipedia request-for-adminship
   - Graph: ~124K nodes, ~2.3M edges
   - Class balance: ~9.5% positive

2. **epinions** (specific config)
   - Dataset: Epinions trust network
   - Graph: ~131K nodes, ~840K edges
   - Class balance: ~5% positive

3. **slashdot** (specific config)
   - Dataset: Slashdot social network
   - Graph: ~82K nodes, ~500K edges
   - Class balance: ~0.8% positive

For each dataset:

- Prepare data with new splits
- Run training
- Verify all metrics are computed
- Spot-check class balance percentages

---

## Example: What Changed

### Before (Random, Unfair)

```python
edges_copy = list(edges)
random.shuffle(edges_copy)  # Random order, no class balance

# Could result in:
# train: 8.2% positive (too few)
# mask: 12.8% positive (too many)
# val: 9.5% positive (different!)
# test: 11.3% positive (different!)
```

### After (Stratified, Fair)

```python
from sklearn.model_selection import train_test_split

# Step 1: TRAIN | remaining
train, remaining, _, remaining_labels = train_test_split(
    edges, labels, train_size=0.48, stratify=labels, random_state=seed
)
# Result: train ≈ 10.5% positive (matches original!)

# Step 2: MASK | temp  
mask, temp, _, temp_labels = train_test_split(
    remaining, remaining_labels, train_size=0.615, stratify=remaining_labels, random_state=seed
)
# Result: mask ≈ 10.5% positive (matches original!)

# Step 3: VAL | TEST
val, test, _, _ = train_test_split(
    temp, temp_labels, test_size=0.5, stratify=temp_labels, random_state=seed
)
# Result: val ≈ 10.5% positive, test ≈ 10.5% positive (both match!)

# Now: All splits maintain original class balance ✓
```

---

## Files to Modify

**Primary File**: `src/data/prepare_data.py`

- Function: `split_edges(cfg, edges)`
- Current lines: Look for random shuffle approach
- Replace with: Hierarchical stratified splitting

**Testing**: No new test files needed

- Use existing training pipeline
- Validate with real training runs
- Check class balance in output metrics

---

## Success Criteria

✓ **Functional**: All splits created with correct sizes
✓ **Fair**: Class balance maintained (±2%) across all splits
✓ **Reproducible**: Same seed produces same splits
✓ **Scalable**: Works on all three datasets
✓ **Scientific**: Enables fair evaluation across train/val/test stages

---

## References

- **Split Semantics**: See TASK_A6_SPLIT_SEMANTICS.md for detailed walkthrough
- **sklearn Documentation**: <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>
- **Class Imbalance**: <https://imbalanced-learn.org/stable/>
