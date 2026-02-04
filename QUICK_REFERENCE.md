# Quick Reference: Updated Task Prompts

## File Locations
- **For A tasks**: Use `CHAT_A_CONFIG_REPRODUCIBILITY_UPDATED.md`
- **For D tasks**: Use `CHAT_D_DATA_PIPELINE_UPDATED.md`
- **For E tasks**: Use `CHAT_E_SEED_CLEANUP_UPDATED.md`
- **B, C, F tasks**: Original files unchanged

## Key Changes at a Glance

### Task A1: ✅ COMPLETE
- See: TASK_A1_FINAL_STATUS.md, WALK_REPRODUCIBILITY_EXPLAINED.md, CONFIG_GUIDE.md
- Result: Unified config, walk reproducibility guaranteed

### Task A6: 🆕 STRATIFIED SPLITTING ADDED
- **What**: Implement stratified edge splitting (train/mask/val/test)
- **Why**: Ensure class balance across splits for fair evaluation
- **How**: Hierarchical train_test_split with stratify=labels
- **Code template**: In CHAT_A_CONFIG_REPRODUCIBILITY_UPDATED.md

### Task D1: 🔄 REFRAMED (Verify + Optimize)
- **Old**: Analyze walk algorithm from scratch
- **New**: Verify A1's solution + benchmark + optimize
- **A1 docs**: WALK_REPRODUCIBILITY_EXPLAINED.md (read this)
- **Your scope**: Verification, benchmarking, optimization identification

### Task E: 🔄 REFRAMED (Verify + Polish)
- **Old**: Major seed overhaul
- **New**: Verify A1's cleanup + test reproducibility
- **Already done**: Unified config, removed scatter, created get_seed()
- **Your scope**: Audit for weird patterns, verify consistency, test end-to-end

### Tasks B, C, F: ✅ No Changes
- Can proceed with original prompts
- Benefit from A1's reproducibility improvements

---

## Critical Stratified Splitting Pattern

```python
# IMPORTANT: This is the four-way split structure
from sklearn.model_selection import train_test_split

def split_edges(cfg, edges):
    edges_array = np.array(edges)
    labels = np.array([e[2] for e in edges])
    seed = get_seed(cfg)
    
    # Step 1: train vs remaining
    train_ratio = cfg.dataset.train_ratio  # 0.48
    remaining_ratio = 1.0 - train_ratio
    
    train_edges, remaining_edges, _, remaining_labels = train_test_split(
        edges_array, labels,
        train_size=train_ratio,
        stratify=labels,  # ← CRITICAL
        random_state=seed
    )
    
    # Step 2: mask vs temp
    mask_ratio_of_remaining = cfg.dataset.mask_ratio / remaining_ratio
    mask_edges, temp_edges, _, temp_labels = train_test_split(
        remaining_edges, remaining_labels,
        train_size=mask_ratio_of_remaining,
        stratify=remaining_labels,  # ← CRITICAL
        random_state=seed
    )
    
    # Step 3: val vs test
    test_ratio_of_temp = cfg.dataset.test_ratio / (cfg.dataset.val_ratio + cfg.dataset.test_ratio)
    val_edges, test_edges, _, _ = train_test_split(
        temp_edges, temp_labels,
        test_size=test_ratio_of_temp,
        stratify=temp_labels,  # ← CRITICAL
        random_state=seed
    )
    
    return {
        "train": train_edges.tolist(),
        "mask": mask_edges.tolist(),
        "val": val_edges.tolist(),
        "test": test_edges.tolist(),
    }
```

---

## Walk Reproducibility Guarantee (Reference)

**Per-walk deterministic seeding**:
```python
for walk_idx in range(start_idx, end_idx):
    rng = np.random.default_rng(base_seed + walk_idx)  # ← Different seed per walk
    start_node = node_arr[rng.integers(0, len(node_arr))]
    # ... walk generation ...
```

**Task sorting**:
```python
results.sort(key=lambda x: x[0])  # Sort by task_id
# Even if tasks finish in random order, walks are in correct order
```

**Result**: walk[0], walk[1], ..., walk[n] in same order, same content, every run.

---

## A1 Completion Checklist (for reference)

- ✅ Unified seed: `reproducibility.seed: 42`
- ✅ Created get_seed(cfg) utility
- ✅ Fixed edge split: `random.seed(seed)` before shuffle
- ✅ Fixed walk sampling: Per-walk seeding + task sorting
- ✅ Cleaned imports: All at top of files
- ✅ Documented: CONFIG_GUIDE.md, WALK_REPRODUCIBILITY_EXPLAINED.md
- ✅ Tested: Walk reproducibility verified with 1,2,4 workers

---

## D1 Scope Change

**Was**: "Analyze and optimize walk sampling algorithm"  
**Now**: "Verify A1's solution + benchmark + identify optimizations"

**Key tasks**:
1. Read WALK_REPRODUCIBILITY_EXPLAINED.md
2. Verify solution with different worker counts
3. Benchmark: throughput, memory, time
4. Identify top 3 optimizations with trade-offs
5. Document in WALK_SAMPLING.md

**Why**: A1 already completed comprehensive algorithm analysis. Your job is verification + optimization search.

---

## E1 Scope Change

**Was**: "Major overhaul: remove weird patterns, centralize seeding"  
**Now**: "Verify cleanup is complete + test reproducibility end-to-end"

**Key tasks**:
1. Audit for remaining weird patterns (% (2**32-1), time.time())
2. Verify get_seed() used consistently
3. Create reproducibility test (same seed → identical outputs)
4. Document in REPRODUCIBILITY.md
5. Test on all 3 datasets

**Why**: A1 already did the heavy lifting. Your job is verification + testing.

---

## Distribution Guide

### For A6 (Stratified Splitting)
→ Use: CHAT_A_CONFIG_REPRODUCIBILITY_UPDATED.md (section "Task A6")

### For D1 (Walk Verification)
→ Use: CHAT_D_DATA_PIPELINE_UPDATED.md (section "Task D1")

### For E (Seed Verification)
→ Use: CHAT_E_SEED_CLEANUP_UPDATED.md (section "Task E1")

### For B (Test Metrics)
→ Use: CHAT_B_TEST_METRICS.md (original, unchanged)

### For C (Prediction Caching)
→ Use: CHAT_C_PREDICTION_CACHING.md (original, unchanged)

### For F (Aggregator Integration)
→ Use: CHAT_F_AGGREGATOR_INTEGRATION.md (original, unchanged)

---

## Reading Order for Understanding

1. **A1_IMPACT_ANALYSIS.md** ← Start here (overview)
2. **WALK_REPRODUCIBILITY_EXPLAINED.md** ← Walk algorithm details
3. **TASK_A1_FINAL_STATUS.md** ← Implementation summary
4. **PROMPT_UPDATES_SUMMARY.md** ← Full context (this doc explains everything)
5. **Specific updated prompt** ← For your task

---

## Key Metrics to Track

### Stratified Splitting (A6)
- Class distribution per split (should be ±1-2% of original)
- Example: Original 10.5% positive → Train 10.3%, Mask 10.6%, Val 10.4%, Test 10.7%

### Walk Reproducibility (D1)
- Same seed + different worker counts → identical walk file
- Test: 1 worker vs 2 worker vs 4 worker vs 8 worker

### Reproducibility Test (E)
- Run training twice with seed=42 → identical metrics
- Example: val_auc run1=0.8547, val_auc run2=0.8547 (exact match)
- Example: test_auc run1=0.8436, test_auc run2=0.8436 (exact match)

---

## Common Issues & Solutions

**Issue**: "Stratified split fails because few examples of minority class"  
**Solution**: Use `stratify=labels` on both splits; if too few examples, allow some imbalance

**Issue**: "Walk reproducibility doesn't match with different num_workers"  
**Solution**: Check that tasks are sorted by task_id BEFORE concatenation; verify per-walk seeding uses base+idx

**Issue**: "Getting different metrics on different runs"  
**Solution**: Verify reproducibility.seed is being set; check no time.time() fallbacks; run test_reproducibility.py

---

## Success Criteria Summary

### A6
✅ Stratified splits implemented  
✅ Class distribution ±1-2% per split  
✅ Works on all 3 datasets  

### D1  
✅ A1 solution verified  
✅ Benchmarks show current performance  
✅ Top 3 optimizations identified  

### E1
✅ No weird patterns remaining  
✅ get_seed() used consistently  
✅ Reproducibility test passes all datasets  

---

## Questions?

Refer to:
- **Walk algorithm**: WALK_REPRODUCIBILITY_EXPLAINED.md
- **Config system**: CONFIG_GUIDE.md
- **Implementation details**: TASK_A1_FINAL_STATUS.md
- **All changes**: PROMPT_UPDATES_SUMMARY.md (this document)
- **Task specifics**: Your assigned updated prompt file
