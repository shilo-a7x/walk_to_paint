# Task A6 Verification: Understanding Split Semantics

## Current Split Configuration
```yaml
train_ratio: 0.48  # 48% of edges
mask_ratio: 0.32   # 32% of edges
val_ratio:  0.10   # 10% of edges
test_ratio: 0.10   # 10% of edges
Total:      1.00
```

---

## Split Semantics: What Each Split Contains

### TRAIN Split (48% of edges)
- **Role**: Context edges - known to the model during ALL stages
- **Used during**:
  - Train: Always in attention (visible context)
  - Val: Always in attention (visible context)
  - Test: Always in attention (visible context)
- **Representation in Walk**: Nodes only (not masked)
- **Receives Labels**: NO (they are targets for other splits to predict)
- **Attention Mask**: Always 1 (always visible to model)

### MASK Split (32% of edges)
- **Role**: Training targets - edges for model to predict DURING TRAINING
- **Used during**:
  - Train: **TARGET** (model predicts these) ← Replaced with [MASK] token
  - Val: Context (visible, used for prediction context but not trained on)
  - Test: Context (visible, used for prediction context)
- **Representation in Walk**: 
  - Train stage: Replaced with [MASK] token, receive labels from original tokens
  - Val/Test stages: Unchanged (nodes only), no labels
- **Receives Labels**: YES (during training only)
- **Attention Mask**: During train: target (1 in attn); During val/test: context (1 in attn)

### VAL Split (10% of edges)
- **Role**: Validation targets - edges for model to predict at VALIDATION TIME
- **Used during**:
  - Train: Context (visible for walks but not trained on)
  - Val: **TARGET** (model predicts these) ← Replaced with [MASK] token
  - Test: Context (visible for prediction context)
- **Representation in Walk**:
  - Train stage: Nodes only (unchanged)
  - Val stage: Replaced with [MASK] token, receive labels
  - Test stage: Nodes only (unchanged)
- **Receives Labels**: YES (during validation only)
- **Attention Mask**: During val: target (1 in attn); Others: context (1 in attn) or hidden

### TEST Split (10% of edges)
- **Role**: Test targets - edges for model to predict at TEST TIME
- **Used during**:
  - Train: Hidden (NOT visible to model)
  - Val: Hidden (NOT visible to model)
  - Test: **TARGET** (model predicts these) ← Replaced with [MASK] token
- **Representation in Walk**:
  - Train stage: NOT IN WALK DATA (excluded, hidden)
  - Val stage: NOT IN WALK DATA (excluded, hidden)
  - Test stage: Replaced with [MASK] token, receive labels
- **Receives Labels**: YES (during testing only)
- **Attention Mask**: During test: target (1 in attn); Others: NEVER in attn (0 or excluded)

---

## Data Flow: Train Stage Example

```
Walk: [N_0, E_1, N_2, E_3, N_4]

Split Assignment:
├─ E_1: TRAIN split  (edge (0,2,+1))
├─ E_3: MASK split   (edge (2,4,+1))
└─ Nodes: N_0, N_2, N_4 all get SplitID.BAD

TRAIN STAGE:
═════════════════════════════════════════════════════════

Input Processing:
  allowed_splits = [TRAIN, MASK]
  target_split = MASK

Input IDs:
  [encode(N_0), encode(E_1), encode(N_2), encode(E_3), encode(N_4)]
   = [node_id_0, edge_id_1, node_id_2, edge_id_3, node_id_4]

After Masking:
  [node_id_0, edge_id_1, node_id_2, [MASK], node_id_4]
   ↑ TRAIN edge, visible in input

Labels (only for MASK edges):
  [IGNORE, IGNORE, IGNORE, edge_label_3, IGNORE]
   ↑ Train to predict this edge

Attention Mask:
  [1, 1, 1, 1, 1]  ← All visible in attention
   ↑ TRAIN and MASK both allowed in attention

Output:
  Model sees: [node_0, edge_1 (context), node_2, [MASK], node_4]
  Task: Predict edge_3 from context
```

---

## Data Flow: Val Stage Example

```
Same Walk: [N_0, E_1, N_2, E_3, N_4]

Split Assignment:
├─ E_1: TRAIN split  (edge (0,2,+1))
├─ E_3: VAL split    (edge (2,4,+1))
└─ No TEST edges in this walk

VAL STAGE:
═════════════════════════════════════════════════════════

Input Processing:
  allowed_splits = [TRAIN, MASK, VAL]
  target_split = VAL

Input IDs:
  [node_id_0, edge_id_1, node_id_2, edge_id_3, node_id_4]

After Masking:
  [node_id_0, edge_id_1, node_id_2, [MASK], node_id_4]
   ↑ TRAIN edge, visible    ↑ VAL edge, masked as target

Labels (only for VAL edges):
  [IGNORE, IGNORE, IGNORE, edge_label_3, IGNORE]
   ↑ Val to predict this edge

Attention Mask:
  [1, 1, 1, 1, 1]  ← All visible in attention (TRAIN+MASK+VAL)

Output:
  Model sees: [node_0, edge_1 (context), node_2, [MASK], node_4]
  Task: Predict edge_3 (different from training target!)
  Note: Edge_1 from TRAIN split is visible context
```

---

## Data Flow: Test Stage Example

```
Same Walk: [N_0, E_1, N_2, E_3, N_4]

But imagine we have a full walk with TEST edges:
  [N_0, E_1, N_2, E_3, N_4, E_5, N_6]

Split Assignment:
├─ E_1: TRAIN split  (48%)
├─ E_3: MASK split   (32%)
├─ E_5: TEST split   (10%)
└─ Nodes: all get SplitID.BAD

TEST STAGE:
═════════════════════════════════════════════════════════

Input Processing:
  allowed_splits = [TRAIN, MASK, VAL, TEST]  ← All allowed!
  target_split = TEST

Input IDs Before Masking:
  [node_0, edge_1, node_2, edge_3, node_4, edge_5, node_6]

After Masking (only TEST targets):
  [node_0, edge_1, node_2, edge_3, node_4, [MASK], node_6]
   ↑ TRAIN visible         ↑ MASK visible   ↑ TEST masked

Labels (only for TEST edges):
  [IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, edge_label_5, IGNORE]
   ↑ Test to predict this edge

Attention Mask:
  [1, 1, 1, 1, 1, 1, 1]  ← ALL visible in attention (TRAIN+MASK+VAL+TEST)

Output:
  Model sees: [node_0, edge_1, node_2, edge_3, node_4, [MASK], node_6]
  Task: Predict edge_5
  Context: Can see TRAIN, MASK, and VAL edges
```

---

## Key Insight: Attention Visibility Rule

### Progressive Disclosure of Information
```
TRAIN stage:
  Can see: TRAIN + MASK edges
  Cannot see: VAL, TEST edges (never in walk data, hidden by design)
  ✓ Fair: only sees training targets

VAL stage:
  Can see: TRAIN + MASK + VAL edges
  Cannot see: TEST edges (not in walk data yet)
  ✓ Fair: can see previous targets (they're valid context)

TEST stage:
  Can see: TRAIN + MASK + VAL + TEST edges
  ✓ Fair: can see all edges from training and val
```

### Why This Design?
- During training: model learns to predict MASK edges
- During val: model tries to predict VAL edges (different from training!)
- During test: model predicts TEST edges (most difficult, least seen)
- Each stage progressively includes more edges in attention
- Prevents data leakage (test edges hidden during train/val)

---

## Class Balance Requirement

### Why Stratification Matters

**Without Stratification (Current)**:
```
Original edges: 10.5% positive, 89.5% negative

Random shuffle might give:
├─ TRAIN (48%):  8.2% positive  ← Model sees fewer positive examples
├─ MASK (32%):  12.8% positive  ← Model trained to predict more positive
├─ VAL (10%):    9.5% positive  ← Different distribution than TRAIN!
└─ TEST (10%):  11.3% positive  ← Different distribution than VAL!

Problem:
  Model trained on TRAIN/MASK distribution (8-12%)
  Evaluated on VAL distribution (9.5%)
  Final tested on TEST distribution (11.3%)
  Unfair comparison! Different class balances.
```

**With Stratification (Required)**:
```
Original edges: 10.5% positive, 89.5% negative

Stratified split ensures:
├─ TRAIN (48%):  10.6% positive  ✓ Matches original
├─ MASK (32%):   10.4% positive  ✓ Matches original
├─ VAL (10%):    10.7% positive  ✓ Matches original
└─ TEST (10%):   10.3% positive  ✓ Matches original

Result:
  Model always sees consistent class distribution
  Fair comparison across stages
  Valid scientific evaluation
```

---

## Implementation: Hierarchical Stratification

### Why Hierarchical?

The 4-way split can't be done with a single `train_test_split()`. We need **hierarchical** (nested) stratification:

```python
# Step 1: Separate TRAIN from others (48% vs 52%)
train, remaining = train_test_split(
    edges, 
    train_size=0.48,
    stratify=labels  # ← Maintain class balance
)

# Step 2: From remaining (52%), separate MASK (32/52% = 61.5% of remaining)
mask, temp = train_test_split(
    remaining,
    train_size=0.32/0.52,  # ← Recalculate ratio
    stratify=remaining_labels
)

# Step 3: From temp (remaining after mask), separate VAL (10/20% = 50% of temp)
val, test = train_test_split(
    temp,
    train_size=0.10/0.20,  # ← Recalculate ratio
    stratify=temp_labels
)

Result: train, mask, val, test
  All with stratified class balance!
```

---

## Stratification Validation Checklist

- [ ] **Edge Count**: train + mask + val + test = total edges
- [ ] **Class Balance TRAIN**: class % ≈ original % (±1-2%)
- [ ] **Class Balance MASK**: class % ≈ original % (±1-2%)
- [ ] **Class Balance VAL**: class % ≈ original % (±1-2%)
- [ ] **Class Balance TEST**: class % ≈ original % (±1-2%)
- [ ] **No Overlap**: No edge appears in multiple splits
- [ ] **All Edges**: Every edge in exactly one split
- [ ] **Reproducibility**: Same seed → same split assignment
- [ ] **All Datasets**: Works on wiki-rfa, epinions, slashdot

---

## Current Code (What Needs to Change)

### Before (in src/data/prepare_data.py)
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

### After (What We Need)
```python
def split_edges(cfg, edges):
    from sklearn.model_selection import train_test_split
    
    edges_array = np.array(edges)
    labels = np.array([e[2] for e in edges])
    seed = get_seed(cfg)
    
    # Step 1: TRAIN | remaining
    train_edges, remaining_edges, _, remaining_labels = train_test_split(
        edges_array, labels,
        train_size=cfg.dataset.train_ratio,
        stratify=labels,
        random_state=seed
    )
    
    # Step 2: MASK | temp
    mask_ratio_of_remaining = cfg.dataset.mask_ratio / (1 - cfg.dataset.train_ratio)
    mask_edges, temp_edges, _, temp_labels = train_test_split(
        remaining_edges, remaining_labels,
        train_size=mask_ratio_of_remaining,
        stratify=remaining_labels,
        random_state=seed
    )
    
    # Step 3: VAL | TEST
    test_ratio_of_temp = cfg.dataset.test_ratio / (cfg.dataset.val_ratio + cfg.dataset.test_ratio)
    val_edges, test_edges, _, _ = train_test_split(
        temp_edges, temp_labels,
        test_size=test_ratio_of_temp,
        stratify=temp_labels,
        random_state=seed
    )
    
    split = {
        "train": train_edges.tolist(),
        "mask": mask_edges.tolist(),
        "val": val_edges.tolist(),
        "test": test_edges.tolist(),
    }
    
    return split
```

---

## Summary: Updated A6 Task Requirements

### Understanding ✓
- TRAIN (48%): Context, always visible, never masked, never target
- MASK (32%): Training targets, hidden in VAL/TEST walks, masked during training
- VAL (10%): Validation targets, visible in VAL/TEST walks, masked during validation
- TEST (10%): Test targets, only visible in TEST stage, hidden during train/val
- Class balance is CRITICAL for fair evaluation across all stages

### Implementation ✓
- Use hierarchical stratified `train_test_split()` (3 steps, 3 stratify calls)
- Maintain ±1-2% class balance in all 4 splits vs original
- Ensure reproducibility with `random_state=seed`
- Support both binary and multiclass labels

### Validation ✓
- Verify edge counts: train + mask + val + test = total
- Verify class distribution for each split
- Test on all 3 datasets (wiki-rfa, epinions, slashdot)
- Verify model training completes successfully

