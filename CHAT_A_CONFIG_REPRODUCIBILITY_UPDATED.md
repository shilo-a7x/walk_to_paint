# CHAT A: Config System & Reproducibility Foundation

## Task A1: Config System Overhaul & Full Reproducibility

### Status: ✅ COMPLETE

This task has been completed. See the following documentation files for details:
- [TASK_A1_FINAL_STATUS.md](TASK_A1_FINAL_STATUS.md) - Complete status report with all changes
- [WALK_REPRODUCIBILITY_EXPLAINED.md](WALK_REPRODUCIBILITY_EXPLAINED.md) - Technical explanation of walk algorithm
- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - Configuration system documentation

**Key achievement**: Single unified seed (`reproducibility.seed`) with per-walk deterministic seeding guarantees bit-for-bit identical walk files regardless of worker count.

---

## Task A6: Data Building Validation

### Current Problem
We need to validate that data is built correctly according to our specs:
- Multiedge handling strategy needs verification
- Binary mode must work correctly across all datasets
- Non-integer node IDs (e.g., wiki) need standardization to [0, n-1]
- Walk sampler might assume this node range → need to verify
- **NEW**: Data splits should be **stratified** to ensure class balance across train/mask/val/test

### Your Task

1. **Implement stratified edge splitting** (NEW CRITICAL TASK):
   - Current split method in `split_edges()` uses simple ratio-based splitting: just shuffles edges and cuts at positions
   - Problem: This creates **imbalanced splits** when edge labels are imbalanced
     - Example: If original edges are 10% positive/90% negative, random split might give:
       - train: 8% positive (unlucky)
       - mask: 12% positive (unlucky)
       - val: 9% positive (lucky)
       - test: 11% positive (unlucky)
     - This means different splits see different class distributions → **unfair comparison**
   
   - **Current split structure** (NOT traditional train/val/test):
     ```
     train_ratio=0.48 → context edges (used in walks, known to model)
     mask_ratio=0.32  → training targets (model predicts these during training)
     val_ratio=0.10   → validation targets (model predicts these at validation)
     test_ratio=0.10  → test targets (model predicts these at test)
     
     Total = 1.0 (all edges accounted for)
     ```
   
   - **Solution**: Use hierarchical stratified splitting to ensure each split maintains original class distribution
   
   **Implementation approach**:
   ```python
   from sklearn.model_selection import train_test_split
   import numpy as np
   
   def split_edges(cfg, edges):
       """Split edges into train/mask/val/test with stratified class balance."""
       edges_array = np.array(edges)
       labels = np.array([e[2] for e in edges])  # Extract labels
       
       # Step 1: Split into train + (mask+val+test)
       # Maintain class distribution
       train_ratio = cfg.dataset.train_ratio
       remaining_ratio = 1.0 - train_ratio
       seed = get_seed(cfg)
       
       train_edges, remaining_edges, train_labels, remaining_labels = train_test_split(
           edges_array, labels,
           train_size=train_ratio,
           stratify=labels,  # ← KEY: stratified split
           random_state=seed
       )
       
       # Step 2: Split remaining into mask, val, test
       # Recalculate ratios for remaining edges
       mask_ratio_of_remaining = cfg.dataset.mask_ratio / remaining_ratio
       val_ratio_of_remaining = cfg.dataset.val_ratio / remaining_ratio
       
       mask_edges, temp_edges, _, temp_labels = train_test_split(
           remaining_edges, remaining_labels,
           train_size=mask_ratio_of_remaining,
           stratify=remaining_labels,  # ← KEY: stratified split
           random_state=seed
       )
       
       # Step 3: Split temp into val and test
       test_ratio_of_temp = cfg.dataset.test_ratio / (cfg.dataset.val_ratio + cfg.dataset.test_ratio)
       val_edges, test_edges, _, _ = train_test_split(
           temp_edges, temp_labels,
           test_size=test_ratio_of_temp,
           stratify=temp_labels,  # ← KEY: stratified split
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
   
   **Validation**:
   - Compute label distribution BEFORE split (original)
   - Compute label distribution AFTER split for each subset
   - Verify each split has approximately same class proportions (±1-2%)
   - Example output:
     ```
     Original edges:  87.3% negative (label=0), 12.7% positive (label=1)
     Train split:     87.4% negative, 12.6% positive ✓ (diff < 1%)
     Mask split:      87.2% negative, 12.8% positive ✓ (diff < 1%)
     Val split:       87.5% negative, 12.5% positive ✓ (diff < 1%)
     Test split:      87.3% negative, 12.7% positive ✓ (diff < 1%)
     ```
   - Test on all datasets: wiki-rfa, epinions, slashdot090221
   - For multiclass datasets, verify all classes are balanced

2. **Understand multiedge handling**:
   - How are duplicate (u,v) edges with different labels handled?
   - Check `src/data/datasets.py` (load functions)
   - Verify it matches intent in `config.yaml` (`multiedge_handling: keep|collapse`)
   - Test edge cases

3. **Validate binary mode**:
   - For datasets with `binary: true`, verify edge labels are [0,1]
   - Check that aggregation/training handles binary classification correctly
   - Test with epinions (binary) and slashdot (binary)

4. **Node ID standardization**:
   - Wiki-rfa uses non-sequential node IDs → need mapping to [0, n-1]
   - Check if node loader does this (grep for `relabel`, `remap`, `0-n-1`)
   - If not, add it + verify walk sampler works with this
   - Document the mapping for reproducibility

5. **Walk sampler assumptions**:
   - Read `src/data/walk_sampler.py` carefully
   - Does it assume nodes are [0, n-1]? (check edge sampling logic)
   - If yes, ensure all datasets are preprocessed to this format

6. **Data validation script**:
   - Create `scripts/validate_data.py` that:
     - Loads edge list for each dataset
     - Verifies node IDs are [0, n-1]
     - Checks multiedge handling
     - Validates binary vs multiclass labels
     - **NEW**: Verifies stratified split was applied correctly
     - Prints summary statistics and class distribution per split

### Files to Check/Modify
- `src/data/prepare_data.py` (update `split_edges()` with stratified splitting)
- `src/data/datasets.py` (all loader functions)
- `src/data/walk_sampler.py` (understand assumptions)
- Create `scripts/validate_data.py`
- Possibly: `data/epinions/`, `data/slashdot090221/`, `data/wiki-rfa/` loaders

### Success Criteria
✅ Edge splits use hierarchical stratified sampling (not random shuffle)  
✅ Each split (train/mask/val/test) maintains ±1-2% class balance vs original  
✅ Stratification works for both binary and multiclass labels  
✅ All node IDs are [0, n-1] after loading  
✅ Binary vs multiclass handling is explicit in code  
✅ Multiedge handling matches config intent  
✅ validate_data.py passes for all datasets  
✅ Walk sampler documentation clarifies its assumptions  

### Why This Matters
- Without stratification, train/mask/val/test might have very different class distributions
- This creates **unfair comparison** of model performance
- For example: model might get 95% accuracy on val (mostly negative) but only 70% on test (mostly positive)
- Stratification ensures splits are representative and comparison is fair

---

## Important Notes
- **May trigger retraining**: Stratified splitting changes edge assignment; existing caches will be invalidated
- **Coordinate with Chat D**: After A6, Chat D will review walk building and pipeline optimization
- **Dependency**: Both A1 and A6 must complete before other chats can proceed confidently
