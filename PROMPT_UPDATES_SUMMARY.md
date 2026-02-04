# Updated Task Prompts: A1 Completion & Stratified Splitting Addition

## Summary of Changes

### Files Created/Updated
1. **A1_IMPACT_ANALYSIS.md** (NEW) - Comprehensive analysis of A1 completion and impact on other tasks
2. **CHAT_A_CONFIG_REPRODUCIBILITY_UPDATED.md** (NEW) - Updated A task prompts with A1 status + stratified splitting requirement
3. **CHAT_D_DATA_PIPELINE_UPDATED.md** (NEW) - Updated D task prompts acknowledging A1 walk reproducibility completion
4. **CHAT_E_SEED_CLEANUP_UPDATED.md** (NEW) - Updated E task reframed as "Verification & Polish"

**Note**: Original files (CHAT_A_*, CHAT_D_*, CHAT_E_*) still exist. New files with _UPDATED suffix contain revisions.

---

## Key Changes by Task

### Task A: Config System & Reproducibility

**A1 Status**: ✅ **COMPLETE**
- Unified seed configuration: `reproducibility.seed: 42`
- Created `get_seed(cfg)` utility function
- Fixed edge split seeding: Uses `random.seed(seed)` before shuffle
- Fixed walk sampling reproducibility: Per-walk deterministic seeding (walk[i] uses seed base+i)
- Guaranteed: Bit-for-bit identical walk files regardless of worker count
- All imports organized at top of files (PEP 8)
- Documentation complete: CONFIG_GUIDE.md, WALK_REPRODUCIBILITY_EXPLAINED.md

**A6 Changes** (NEW): 
- **ADDED**: Stratified edge splitting requirement (CRITICAL)
- **Why**: Current simple random shuffle creates imbalanced splits
  - Example: If 10% positive edges, random split might give train:8%, mask:12%, val:9%, test:11%
  - Different class distributions → unfair model comparison
- **Solution**: Use sklearn.model_selection.train_test_split with `stratify=labels`
  - Hierarchical approach: train/(mask+val+test) → mask/(val+test) → val/test
  - Each split maintains ±1-2% class balance vs original
- **Implementation**: Code example provided in updated prompt
- **Validation**: Verify class distribution per split for all datasets

---

### Task B: Test Metrics Tracking

**Status**: ✅ **No changes needed**
- A1 provides clean seed handling; doesn't affect B's test metrics logging
- B can proceed independently
- Benefits from A1's reproducibility for consistent test baselines

---

### Task C: Prediction Caching & Position Analysis

**Status**: ✅ **No changes needed**
- Depends on A1's reproducibility (satisfied ✓)
- C's predictions will now be truly reproducible (same seed → same predictions)
- No implementation changes needed

---

### Task D: Data Pipeline Review & Optimization

**D1 Changes** (SIGNIFICANT):
- **OLD**: "Analyze walk sampling algorithm from scratch"
- **NEW**: "Verify A1's walk reproducibility solution + optimize"
- **Why**: A1 already completed comprehensive walk algorithm analysis
  - See WALK_REPRODUCIBILITY_EXPLAINED.md for complete explanation
  - Per-walk seeding mechanism (walk[i] uses seed base+i)
  - Sorting by task_id guarantees order preservation
  - Bit-for-bit file reproducibility guaranteed
- **New D1 scope**: 
  - Verify A1's solution with different worker counts
  - Understand the algorithm (read the docs)
  - Benchmark current performance (time, memory, throughput)
  - Identify optimization opportunities
  - Document findings in WALK_SAMPLING.md

**D2 Status**: No changes
- File format comparison still needed
- I/O optimization still needed

---

### Task E: Seed & Reproducibility

**Reframed** (SIGNIFICANT):
- **OLD**: "Remove all weird constructs and simplify propagation" (major overhaul)
- **NEW**: "Verify A1's cleanup is complete + polish remaining issues" (verification task)
- **Why**: A1 already addressed most issues
  - Unified seed config ✓
  - Removed scattered keys ✓
  - Created get_seed() utility ✓
  - Fixed edge split seeding ✓
  - Fixed walk sampling seeding ✓
- **New E1 scope**:
  - Audit for remaining weird patterns (% (2**32-1), time.time() fallbacks)
  - Verify get_seed() used consistently across all entry points
  - Create reproducibility test (run twice with same seed, verify identical outputs)
  - Document reproducibility guarantee in REPRODUCIBILITY.md
  - Test on all 3 datasets

---

### Task F: Aggregator Integration Strategy

**Status**: ✅ **No changes needed**
- Benefits from all other improvements
- Will have cleaner config, verified data, test metrics, etc.

---

## Master Timeline (Updated)

```
COMPLETED:
  ✅ Chat A1: Config unification + walk reproducibility + edge split seeding

READY TO START (can proceed immediately):
  ├─ Chat B: Test metrics (per-epoch monitoring) - depends on A1 ✓
  ├─ Chat C: Prediction caching (C1 raw predictions) - depends on A1 ✓
  ├─ Chat D1: Walk verification + benchmarking - depends on A1 ✓
  ├─ Chat E: Seed verification & polish - depends on A1 ✓
  └─ Chat A6: Stratified splitting (NEW REQUIREMENT) - depends on A1 ✓

SEQUENTIAL AFTER PARALLEL:
  Chat A6 completion → Data pipeline validated with stratified splits
  ↓
  Chat D2: I/O optimization (depends on D1) → D2 can start
  Chat C2/C3: Position analysis (depends on C1) → can start in parallel with D2
  ↓
  Chat F: Integration strategy (depends on all others)
```

---

## Most Important Changes

### 1. **Task A6: Stratified Splitting** (NEW)
This is the most critical new requirement. It ensures fair model evaluation by maintaining class balance across splits.

**Impact**: 
- Changed data splits become more representative
- Model evaluation becomes fairer (not comparing 8% positive train vs 12% positive mask)
- Critical for scientific validity

**Implementation**: Code template provided in CHAT_A_CONFIG_REPRODUCIBILITY_UPDATED.md

---

### 2. **Task A1: Complete** (REFERENCE ONLY)
No action needed; just reference the completed work. The key documents to read:
- WALK_REPRODUCIBILITY_EXPLAINED.md (how and why walk reproducibility works)
- CONFIG_GUIDE.md (how to use the new config system)
- TASK_A1_FINAL_STATUS.md (what was changed)

---

### 3. **Task D1: Verification Focus** (REFRAMED)
Was a major analysis task; now is verification + benchmarking. Much less work.

---

### 4. **Task E: Polish Focus** (REFRAMED)
Was a major overhaul; now is verification + testing. A1 already did the hard work.

---

## File Organization

### Original Files (Keep for reference)
- CHAT_A_CONFIG_REPRODUCIBILITY.md
- CHAT_B_TEST_METRICS.md
- CHAT_C_PREDICTION_CACHING.md
- CHAT_D_DATA_PIPELINE.md
- CHAT_E_SEED_CLEANUP.md
- CHAT_F_AGGREGATOR_INTEGRATION.md

### Updated Files (Use these for new chats)
- CHAT_A_CONFIG_REPRODUCIBILITY_UPDATED.md ← Use this for A tasks
- CHAT_D_DATA_PIPELINE_UPDATED.md ← Use this for D tasks
- CHAT_E_SEED_CLEANUP_UPDATED.md ← Use this for E tasks

### Reference Documents (Consult for understanding)
- A1_IMPACT_ANALYSIS.md ← Read this first to understand all changes
- WALK_SOLUTION_COMPLETE.md (from A1 work)
- WALK_REPRODUCIBILITY_EXPLAINED.md (from A1 work)
- TASK_A1_FINAL_STATUS.md (from A1 work)
- CONFIG_GUIDE.md (from A1 work)

---

## How to Use These Updates

1. **For Chat Leads**:
   - Review A1_IMPACT_ANALYSIS.md (overview)
   - Use the _UPDATED prompt files for each chat
   - Key point: A1 is done; focus on verification/optimization/new tasks

2. **For A6 (Data Validation)**:
   - Read the stratified splitting section in CHAT_A_CONFIG_REPRODUCIBILITY_UPDATED.md
   - Code template provided for implementation
   - Test on all 3 datasets: wiki-rfa, epinions, slashdot

3. **For D1 (Walk Analysis)**:
   - Read WALK_REPRODUCIBILITY_EXPLAINED.md (from A1, it's authoritative)
   - Verify the solution with different worker counts
   - Focus on benchmarking and optimization identification

4. **For E (Seed Cleanup)**:
   - It's now a verification task, not major overhaul
   - Audit for remaining weird patterns
   - Create reproducibility test

---

## Critical Implementation Details

### Stratified Splitting (A6)
The implementation requires hierarchical stratification because we have 4 splits (train/mask/val/test), not 3:

```python
Step 1: Split all edges into train + remaining (stratified)
Step 2: Split remaining into mask + temp (stratified)  
Step 3: Split temp into val + test (stratified)

Result: All 4 splits maintain same class distribution as original
```

See CHAT_A_CONFIG_REPRODUCIBILITY_UPDATED.md for full code.

### Walk Reproducibility Guarantee (A1, referenced in D1)
Walk reproducibility is guaranteed by:

```
Per-walk seeding:  walk[i] uses seed = base_seed + i
Task sorting:      Results sorted by task_id before concatenation
File determinism:  torch.save produces identical bytes

Result: Same seed → identical walks.pkl file, regardless of worker count
```

See WALK_REPRODUCIBILITY_EXPLAINED.md for complete explanation.

---

## Summary Statistics

| Aspect | Before | After |
|--------|--------|-------|
| **Config keys for seed** | 4 (training.seed, walk_seed, etc.) | 1 (reproducibility.seed) |
| **Entry point variants** | Different seeds in different places | Unified via get_seed() |
| **Walk reproducibility** | Uncertain | Guaranteed (per-walk seeding) |
| **Data split reproducibility** | Random shuffle (uncontrolled) | Stratified (balanced) |
| **Weird seed patterns** | % (2**32-1), time.time() | Removed in A1, verified in E |
| **Documentation** | Scattered | Centralized (CONFIG_GUIDE.md, WALK_REPRODUCIBILITY_EXPLAINED.md, REPRODUCIBILITY.md) |

---

## Next Steps

1. **Distribute updated prompts** to relevant chats
2. **Prioritize A6** (stratified splitting) - critical for fair evaluation
3. **Start D1 verification** - confirm A1's walk reproducibility works
4. **Start E verification** - ensure no weird patterns remain
5. **Then proceed with B, C parallel work** as originally planned
6. **Finally F** - integration strategy with all validated components

