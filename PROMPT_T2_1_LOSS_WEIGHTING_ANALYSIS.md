# Prompt: T2.1 - Analyze Current Loss Weighting

## Goal

Audit the **current loss weighting strategy** to understand:

1. Is loss currently weighted by class?
2. If yes, how are weights computed?
3. Are there batch-level or epoch-level effects?
4. Is there any data leakage (val/test affecting training weights)?

## Scope

- Analyze current code, **don't change** anything yet
- Document findings
- Prepare recommendations for T2.2

## Files to Examine

### Primary: `src/model/lit_model.py`

Look for:

- Loss function initialization (likely in `__init__`)
- Loss computation (likely in `training_step` or `validation_step`)
- Any `class_weight` or `weight` parameters
- Sample weighting or batch weighting logic

**Key Questions**:

- What loss function? (BCEWithLogitsLoss, CrossEntropyLoss, etc.)
- Any `weight=` parameter passed to loss?
- Any manual sample weighting applied?

### Secondary: `run.py`

Look for:

- How is the model instantiated?
- Are class weights computed before model creation?
- Do they use train/val/test splits?

### Secondary: `src/data/prepare_data.py`

Look for:

- What are the actual class distributions in each split?
- Compute manually if not logged

## What To Check

### 1. Current Loss Function

```python
# In lit_model.py, find the loss definition
# Example patterns to look for:

# Unweighted (BAD for imbalance)
loss_fn = nn.BCEWithLogitsLoss()

# Weighted (GOOD)
pos_weight = torch.tensor([weight_val])
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Or CrossEntropyLoss with weights
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
```

### 2. Where Are Weights Computed?

- If weighted, trace **where** the weights come from
- Are they computed from **train split only**? (✓ Good)
- Or from **current batch**? (? Depends)
- Or from **val/test**? (✗ Data leakage!)

### 3. Batch Composition

- Does DataLoader mix train/val/test in one batch? (check `DataLoader` config)
- Or are they strictly separate? (✓ Good)

### 4. Loss Computation per Split

- Training: what loss is used?
- Validation: same loss or different?
- Test: same loss or different?

## Tasks

### Task 1: Trace Loss Function (30 min)

1. Open `src/model/lit_model.py`
2. Find `__init__` and search for loss function
3. Document:
   - What loss function?
   - Are there weights? If yes, what are they?
   - Where did weights come from?

### Task 2: Check Class Distributions (30 min)

1. Load train/val/test edge splits
2. Count positive and negative edges in each
3. Document:
   - Train: n_pos, n_neg, % positive
   - Val: n_pos, n_neg, % positive
   - Test: n_pos, n_neg, % positive
   - Expected: all ~same %

### Task 3: Audit Data Leakage Risk (30 min)

1. Check if weights are computed at:
   - Model creation time (from train only) ✓
   - Training start time (from train only) ✓
   - Batch time (from current batch) ?
   - Any time val/test stats used? ✗
2. Document leakage risk

### Task 4: Batch-Level Analysis (1h, optional)

1. Sample a few batches from train/val/test
2. Check:
   - How many positive vs negative per batch?
   - Does it match split-level distribution?
   - Any clustering?
3. Document findings

## Deliverables

### Report: `docs/LOSS_WEIGHTING_ANALYSIS.md`

Include:

- Summary of current approach
- Class distributions (table)
- Recommended weighting strategy (to be implemented as **mandatory**)
- Data leakage risk assessment

### Example Report Structure

```markdown
# Loss Weighting Analysis

## Current State
- Loss function: BCEWithLogitsLoss (unweighted)
- Class weighting: None
- Data leakage risk: Low (weights not used)

## Class Distributions
| Split | Pos | Neg | % Pos |
|-------|-----|-----|-------|
| Train | 10k | 90k | 10.0% |
| Val   | 10k | 90k | 10.0% |
| Test  | 10k | 90k | 10.0% |

## Batch-Level Analysis
- All batches sampled from single split
- Class distribution per batch matches split-level
- No obvious data leakage

## Recommended Strategy (for T2.2)
**Implement as mandatory (no config option)**:
1. Compute pos_weight and neg_weight from train split at model init
2. Pass pos_weight to BCEWithLogitsLoss
3. This is the standard behavior, always on
```

## Success Criteria

- ✅ Loss function identified and documented
- ✅ Class distributions computed and logged
- ✅ Data leakage risk assessed
- ✅ Clear strategy recommended for T2.2 (mandatory weighting)
- ✅ Report file created with findings

## Notes

- This is **audit only**, no code changes
- Document everything so T2.2 can fix efficiently
- If unweighted and imbalanced, that's OK (we'll fix in T2.2)
- Focus on facts, not opinions

---

## Execution Plan

1. **Read** `src/model/lit_model.py` (loss function section)
2. **Run** `python scripts/check_class_distributions.py` (or write quick script)
3. **Trace** where weights come from (if any)
4. **Assess** data leakage risk
5. **Write** report with findings
6. **Share** report for review

**Estimated time**: 2-3 hours
**Output**: `docs/LOSS_WEIGHTING_ANALYSIS.md`
