# Prompts: T2.1 → T2.2 → T2.3 - Class Imbalance Pipeline

## Overview

Three sequential tasks to handle class imbalance properly:

```
T2.1 (2-3h)          T2.2 (4-5h)            T2.3 (2-3h)
Analysis      →      Implementation    →     Validation
├─ Audit loss        ├─ Add weighting       ├─ No leakage tests
├─ Check class dist  ├─ Global weights      ├─ Batch tests  
├─ Assess leakage    ├─ Test on one model   └─ Document
└─ Report findings   └─ Measure impact
```

---

## T2.1: Analyze Current Loss Weighting ✅ DETAILED PROMPT READY

**File**: [PROMPT_T2_1_LOSS_WEIGHTING_ANALYSIS.md](PROMPT_T2_1_LOSS_WEIGHTING_ANALYSIS.md)

**Goal**: Understand current state

- Is loss weighted?
- What are class distributions?
- Any data leakage?

**Output**: `docs/LOSS_WEIGHTING_ANALYSIS.md`

**Time**: 2-3 hours

**Next**: Read report → move to T2.2

---

## T2.2: Implement Class-Weighted Loss (To Be Done)

**Goal**: Add **mandatory** class weighting (no config option)

**Strategy** (to be confirmed after T2.1):

1. Compute weights from **train split only** (global, at init time):
   - `weight_pos = n_neg / (n_pos + n_neg)`
   - `weight_neg = n_pos / (n_pos + n_neg)`

2. Add to loss function **permanently**:

   ```python
   # In lit_model.py __init__
   # Compute from train split data
   pos_weight = torch.tensor([weight_pos], device=self.device)
   self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
   ```

3. This is **always on**, not configurable

4. Validate with small test run

**Files to modify**:

- `src/model/lit_model.py` (loss function + weight computation)
- No config changes (this is the standard behavior)

**Output**: Updated model with **mandatory** weighted loss

**Time**: 4-5 hours

**Next**: Verify with T2.3

---

## T2.3: Validate No Data Leakage (To Be Done)

**Goal**: Ensure weights don't leak val/test information

**Tests to implement**:

1. **Weights from train only**:
   - Verify `weight_computation_split == 'train'`
   - Check weights computed before any validation

2. **Batch isolation**:
   - Each batch from single split (no mixing)
   - DataLoader has `drop_last=True` or handles last batch

3. **Loss computation verification**:
   - Train uses same weights throughout
   - Val loss computed with training weights
   - Test loss computed with training weights

4. **Pytest suite**:

   ```python
   def test_weights_from_train_only():
       # Verify weights don't use val/test

   def test_batch_isolation():
       # Verify no split mixing

   def test_no_weight_update_during_val():
       # Verify weights stay constant
   ```

**Output**: `tests/test_no_class_imbalance_leakage.py`

**Time**: 2-3 hours

**Next**: Done! Ready to retrain with fair loss

---

## 🎯 Full T2 Pipeline Execution Plan

### Step 1: Run T2.1 (Analysis)

```bash
# Read PROMPT_T2_1_LOSS_WEIGHTING_ANALYSIS.md
# Follow tasks 1-4
# Generate: docs/LOSS_WEIGHTING_ANALYSIS.md
```

### Step 2: Review T2.1 Report

- Share findings
- Confirm approach for T2.2

### Step 3: Run T2.2 (Implementation)

- Modify `lit_model.py` to add class weighting
- Update config schema
- Test on one model

### Step 4: Run T2.3 (Validation)

- Write leakage tests
- Verify no data leakage
- All tests pass

### Step 5: Ready to Retrain

- Models now trained with fair loss
- Stratified splits (A6) + weighted loss (T2.2) = proper evaluation

---

## 📋 Expected Outcomes

After T2.1-T2.3:

- ✅ Know current loss strategy
- ✅ Have class-weighted loss
- ✅ Know weights don't leak
- ✅ Ready for clean retraining

---

## 🚀 Start Now

1. **Read** [PROMPT_T2_1_LOSS_WEIGHTING_ANALYSIS.md](PROMPT_T2_1_LOSS_WEIGHTING_ANALYSIS.md)
2. **Follow** the 4 tasks
3. **Generate** analysis report
4. **Share** findings

**Ready to start T2.1?**
