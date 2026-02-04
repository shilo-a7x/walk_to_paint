# Task A6: Split Semantics and Progressive Information Disclosure

## Understanding the Four Splits

This document clarifies **what each split is for** and **when edges are visible** during different stages of the model lifecycle.

---

## The Four Splits: Roles and Semantics

### 1. TRAIN Split (48% of edges)

**Semantic Role**: Context edges - background information for the model

**Always Visible**:

- ✓ Train stage: Visible in walk attention
- ✓ Validation stage: Visible in walk attention
- ✓ Test stage: Visible in walk attention

**Ever Masked**: No

- Never replaced with [MASK] token
- Never predicted by model
- Always available as context

**Labels Used As Targets**: ✗ Never

- These edges are context, not targets
- Model doesn't learn to predict TRAIN edges
- They provide stable background signal

**Why This Split Exists**:

- Provides stable context throughout model lifecycle
- Edges the model can always rely on
- Grounds all predictions in common knowledge

**Class Balance Requirement**: **CRITICAL**

- If TRAIN has different class distribution than original
- Model learns biased patterns
- Evaluation becomes unfair

---

### 2. MASK Split (32% of edges)

**Semantic Role**: Training targets - what the model learns to predict during training

**Visibility Timeline**:

- ❌ Train stage: **HIDDEN** - Replaced with [MASK] token
  - Model tries to predict what MASK token represents
  - This is what drives learning
  
- ✓ Validation stage: **VISIBLE** (no longer masked)
  - During validation, we already know what MASK was
  - MASK edges now become context for validation task
  
- ✓ Test stage: **VISIBLE** (context)
  - MASK edges are known, provide context for test task
  - What model learned to predict during training

**Labels Used As Targets**: ✓ **Yes, but only during training**

- During training: `loss = predict_MASK_edges(TRAIN+visible_part_of_MASK)`
- After training: MASK edges are just context
- Not targets during validation or testing

**Why This Split Exists**:

- Training objective: predict these edges given context
- Model learns representations from predicting MASK
- MASK edges become known context afterward

**Progressive Disclosure**:

```
Training:     MASK is hidden  → Model learns what it means
Validation:   MASK is visible → Model has more context
Test:         MASK is visible → Even more context available
```

**Class Balance Requirement**: **CRITICAL**

- If MASK has different class distribution than original
- Model learns biased decision boundaries
- Validation/test evaluation is unfair (different problem)

---

### 3. VAL Split (10% of edges)

**Semantic Role**: Validation targets - unseen during training, predicted during validation

**Visibility Timeline**:

- ✓ Train stage: **VISIBLE** (as context, not as target)
  - VAL edges visible in walks during training
  - But not predicted (not targets)
  - Provide signal about edges coming later
  
- ❌ Validation stage: **HIDDEN** - Replaced with [MASK] token
  - **This is the validation task**
  - Model predicts these edges given TRAIN+MASK context
  - This is where we measure learning
  
- ✓ Test stage: **VISIBLE** (context)
  - During testing, VAL edges are known
  - Provide context for final test task

**Labels Used As Targets**: ✓ **Yes, but only during validation**

- During validation: `loss = predict_VAL_edges(TRAIN+MASK+visible_part_of_VAL)`
- Unseen during training: VAL is new prediction task
- Tests generalization to new edges

**Why This Split Exists**:

- **New prediction task** (different from training)
- Tests if model learned generalizable patterns
- Not training targets (prevents overfitting)
- Tests on distribution same as training

**Progressive Disclosure**:

```
Training:     VAL is visible  → No info leakage, just context
Validation:   VAL is masked   → Measure generalization to new edges
Test:         VAL is visible  → Context for final evaluation
```

**Class Balance Requirement**: **CRITICAL**

- If VAL has different class distribution
- Validates on different problem than trained on
- Results meaningless

---

### 4. TEST Split (10% of edges)

**Semantic Role**: Test targets - held-out final evaluation set

**Visibility Timeline**:

- ❌ Train stage: **COMPLETELY HIDDEN** - Not in data at all
  - Not visible anywhere
  - Not in walks
  - Prevents any data leakage
  
- ❌ Validation stage: **COMPLETELY HIDDEN** - Not in data at all
  - Still hidden during validation
  - Model cannot see these edges
  - Strict data leakage prevention
  
- ❌ Test stage: **INITIALLY HIDDEN**, then **MASKED and predicted**
  - Only at test time: Replace with [MASK] token
  - Model tries to predict these edges
  - **This is the final evaluation**

**Labels Used As Targets**: ✓ **Yes, but only during testing**

- During testing: `loss = predict_TEST_edges(TRAIN+MASK+VAL+visible_part_of_TEST)`
- Completely held-out until test time
- Final rigorous test of model performance

**Why This Split Exists**:

- **Final held-out test** - strictest evaluation
- Completely hidden until needed
- Tests true generalization
- Prevents any training/tuning bias

**Progressive Disclosure**:

```
Training:     TEST is hidden   → No information at all
Validation:   TEST is hidden   → Still hidden
Test:         TEST is masked   → Predict using all context
              → FINAL EVALUATION
```

**Class Balance Requirement**: **CRITICAL**

- If TEST has different class distribution
- Final evaluation is on different problem
- Results cannot be trusted

---

## Information Availability at Each Stage

### Training Stage

```
Visible edges:     TRAIN + MASK (80%)
Hidden edges:      VAL + TEST (20%)
What we predict:   MASK edges (learning targets)
What we evaluate:  Training loss on MASK prediction
```

### Validation Stage

```
Visible edges:     TRAIN + MASK + VAL (90%)
Hidden edges:      TEST (10%)
What we predict:   VAL edges (validation targets)
What we evaluate:  Validation loss/metrics on VAL prediction
                  → Tests generalization to VAL (unseen during training)
```

### Test Stage

```
Visible edges:     TRAIN + MASK + VAL + TEST (100%)
Hidden edges:      None
What we predict:   TEST edges (test targets)
What we evaluate:  Test loss/metrics on TEST prediction
                  → Final evaluation on held-out data
```

---

## Why This Progressive Disclosure is Fair and Scientific

### 1. No Data Leakage

- TEST completely hidden until test time
- Model cannot optimize for TEST
- Prevents overfitting to test distribution

### 2. Incremental Difficulty

- Train: Model learns with 80% context
- Val: Model generalizes with 90% context (unseen edges)
- Test: Model evaluated with 100% context (held-out edges)
- More context at each stage makes sense (more information available)

### 3. Fair Comparison

- **CRITICAL**: If TRAIN/MASK/VAL/TEST have same class distribution as original
  - Model trained on same distribution as evaluated
  - No bias from different class balances
  - Valid scientific comparison

### 4. Prevents Multiple Testing Issues

- VAL not used during training (prevents overfitting)
- TEST hidden until test time (prevents tuning)
- Independent evaluation metrics at each stage

---

## Class Balance Requirement: Why It Matters

### Without Stratification (Unfair)

```
Original: 10.5% positive

Random split:
  TRAIN: 8.2% positive  ← Model trained on fewer positives
  MASK:  12.8% positive ← Different during validation
  VAL:   9.5% positive  ← Different again
  TEST:  11.3% positive ← Different final evaluation

Problem:
- Model optimizes for 8.2% positive distribution
- Validated on 12.8% distribution (different problem)
- Tested on 11.3% distribution (yet different)
- Results meaningless - different problems throughout!
```

### With Stratification (Fair)

```
Original: 10.5% positive

Stratified split:
  TRAIN: 10.6% positive ← Model trained on same distribution
  MASK:  10.4% positive ← Validated on same distribution
  VAL:   10.7% positive ← Same distribution throughout
  TEST:  10.3% positive ← Same distribution at test

Benefit:
- Model optimizes for 10.5% distribution
- Validated on 10.5% distribution (same problem)
- Tested on 10.5% distribution (same problem)
- Results valid - fair scientific comparison!
```

---

## Example: Imbalanced Dataset (wiki-rfa, 9.5% positive)

### What Changes?

```
Original: 9.5% positive

Stratified split maintains:
  TRAIN: 9.5% positive ✓ (same as original)
  MASK:  9.5% positive ✓ (same as original)
  VAL:   9.5% positive ✓ (same as original)
  TEST:  9.5% positive ✓ (same as original)

Same class distribution throughout!
→ Fair evaluation ✓
```

### What If We Didn't Stratify?

```
Original: 9.5% positive

Random split might produce:
  TRAIN: 7.2% positive ← Model trained on fewer positives!
  MASK:  11.8% positive ← Validation much more imbalanced!
  VAL:   8.9% positive  ← Different again!
  TEST:  10.4% positive ← Different for final eval!

→ Model trained on one problem, evaluated on three others!
→ Results meaningless!
```

---

## Example: Highly Imbalanced Dataset (slashdot, 0.8% positive)

### What Changes?

```
Original: 0.8% positive (very rare!)

Stratified split maintains:
  TRAIN: 0.8% positive ✓ (same as original)
  MASK:  0.8% positive ✓ (same as original)
  VAL:   0.8% positive ✓ (same as original)
  TEST:  0.8% positive ✓ (same as original)

Even with extreme imbalance, all splits are identical!
→ Fair evaluation ✓
```

### Stratification is Most Critical Here

- With 0.8% positive, random chunks could have 0-3%
- Huge variations in class balance
- Stratification **prevents this**
- **Especially important for imbalanced data**

---

## Split Design Rationale

### Why 48-32-10-10?

| Ratio | Role | Justification |
|-------|------|---------------|
| 48% | TRAIN | Large, stable context throughout |
| 32% | MASK | Training targets, largest learning task |
| 10% | VAL | Validation targets, separate from test |
| 10% | TEST | Final held-out test, same size as VAL |

**Design Philosophy**:

- TRAIN: Large foundation (48%)
- MASK: Main learning targets (32%)
- VAL + TEST: Equal held-out (10% each)
- Total: 100% of edges

**Why not 50-25-12.5-12.5?**

- 48-32 provides clear separation
- 10-10 is symmetric and simple
- Easy to remember: ~half train, ~third mask, ~fifth test

---

## Summary: Fair Evaluation Requires Stratification

### Key Points

1. **Each split has a role**
   - TRAIN: Context
   - MASK: Training targets
   - VAL: Validation targets (unseen during training)
   - TEST: Final held-out test

2. **Progressive disclosure is intentional**
   - More edges visible as model progresses
   - But no data leakage
   - Fair incremental evaluation

3. **Class balance matters critically**
   - Model trained on one distribution → evaluated on same
   - Stratification ensures all splits maintain original balance
   - ±2% tolerance sufficient for fair evaluation

4. **Stratification makes it fair**
   - Same class distribution across all splits
   - Model trained and evaluated on same problem
   - Results are scientifically valid

5. **Task A6 implements this**
   - Replaces random shuffling
   - Hierarchical stratified splitting
   - ±2% class balance guarantee
   - Fair, reproducible, scientific evaluation

---

## Validation Checklist

When preparing data, verify:

✓ **TRAIN edges**: ~10.5% positive (matches original)  
✓ **MASK edges**: ~10.5% positive (matches original)  
✓ **VAL edges**: ~10.5% positive (matches original)  
✓ **TEST edges**: ~10.5% positive (matches original)  

**If any split differs >2%**: Something is wrong with stratification!

---

## Conclusion

With **Task A6's stratified splitting**, all four splits maintain the original class distribution, enabling fair, scientific model evaluation where the model is trained and evaluated on the same problem across all stages.
