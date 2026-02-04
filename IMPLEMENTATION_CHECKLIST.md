# Implementation Checklist & Next Steps

## Documents Created (Ready to Use)

### Reference Documents
- [x] **COMPLETE_ANALYSIS.md** - Comprehensive reference (read this first)
- [x] **PROMPT_UPDATES_SUMMARY.md** - Summary of changes
- [x] **QUICK_REFERENCE.md** - Quick facts and code patterns
- [x] **A1_IMPACT_ANALYSIS.md** - Impact on each task
- [x] **FILES_SUMMARY.md** - File locations and organization

### Updated Task Prompts (Use These)
- [x] **CHAT_A_CONFIG_REPRODUCIBILITY_UPDATED.md** - A1 complete + A6 stratified splitting
- [x] **CHAT_D_DATA_PIPELINE_UPDATED.md** - D1 verification (simplified) + D2 optimization
- [x] **CHAT_E_SEED_CLEANUP_UPDATED.md** - E1 verification & polish (simplified)

### Unchanged Prompts (Still Valid)
- [x] **CHAT_B_TEST_METRICS.md** - Test metrics tracking (independent)
- [x] **CHAT_C_PREDICTION_CACHING.md** - Prediction caching (independent)
- [x] **CHAT_F_AGGREGATOR_INTEGRATION.md** - Integration strategy (depends on all)

---

## Critical Task Checklists

### ✅ Task A6: Stratified Edge Splitting (PRIORITY)

**Preparation**:
- [ ] Read COMPLETE_ANALYSIS.md section "A6 Addition"
- [ ] Read CHAT_A_CONFIG_REPRODUCIBILITY_UPDATED.md (A6 section)
- [ ] Understand 4-way split structure: train(0.48)/mask(0.32)/val(0.10)/test(0.10)
- [ ] Understand why random shuffle creates imbalance

**Implementation**:
- [ ] Update `src/data/prepare_data.py` split_edges() function
- [ ] Add `from sklearn.model_selection import train_test_split` import
- [ ] Implement 3-step hierarchical stratification (see code template)
- [ ] Keep random seeding (already in A1) - seed before shuffle
- [ ] Use stratify=labels in all three train_test_split calls

**Validation**:
- [ ] Compute class distribution before split (original)
- [ ] Compute class distribution after split (each subset)
- [ ] Verify each split is ±1-2% of original distribution
- [ ] Test on wiki-rfa (multiclass/non-integer nodes)
- [ ] Test on epinions (binary)
- [ ] Test on slashdot090221 (binary)
- [ ] Document results with examples

**Success Criteria**:
- ✓ Stratified splits implemented
- ✓ Class distribution ±1-2% per split on all datasets
- ✓ Cache invalidation works (old splits not reused)
- ✓ Data rebuild completes successfully
- ✓ Training with new splits works end-to-end

---

### ✅ Task D1: Walk Reproducibility Verification (SIMPLIFIED)

**Preparation**:
- [ ] Read WALK_REPRODUCIBILITY_EXPLAINED.md (complete understanding)
- [ ] Read CHAT_D_DATA_PIPELINE_UPDATED.md (D1 section)
- [ ] Understand per-walk seeding: walk[i] uses seed = base + i
- [ ] Understand task sorting: results sorted by task_id before concatenation

**Verification**:
- [ ] Review src/data/walk_sampler.py code
- [ ] Trace through code: understand per-walk seeding mechanism
- [ ] Trace through code: understand task sorting logic
- [ ] Run A1's walk reproducibility test with different worker counts (1, 2, 4, 8)
- [ ] Verify walks are bit-for-bit identical
- [ ] Document findings

**Benchmarking**:
- [ ] Create `scripts/benchmark_walks.py` (reusable)
- [ ] Benchmark walk sampling for each dataset
- [ ] Measure: throughput (walks/sec), time, memory
- [ ] Output: `benchmark_results/walk_sampling_{dataset}.csv`
- [ ] Compare: single-threaded vs multi-threaded performance

**Optimization Analysis**:
- [ ] Identify top 3 optimization opportunities
- [ ] For each: explain approach, estimate impact, note trade-offs
- [ ] Examples: vectorization, better caching, memory mapping, etc.
- [ ] Write findings to WALK_SAMPLING.md

**Success Criteria**:
- ✓ A1 solution verified with multiple worker counts
- ✓ Current performance benchmarked (all 3 datasets)
- ✓ Top 3 optimizations identified with impact estimates
- ✓ WALK_SAMPLING.md documents algorithm and opportunities

---

### ✅ Task E1: Seed Verification & Polish (SIMPLIFIED)

**Preparation**:
- [ ] Read CONFIG_GUIDE.md (understand unified config)
- [ ] Read CHAT_E_SEED_CLEANUP_UPDATED.md (E1 section)
- [ ] Understand A1 already completed most work

**Audit Code**:
- [ ] Search codebase for `% (2**32-1)` (should not exist after A1)
- [ ] Search for `int(time.time())` in seed context (should not exist after A1)
- [ ] Search for `getattr(cfg, ..., seed=None)` (should use get_seed() instead)
- [ ] Search for hardcoded seed values (should come from config)
- [ ] Document what you find (if anything)

**Verify Consistency**:
- [ ] Check run.py uses get_seed(cfg) ✓
- [ ] Check optuna_run.py uses get_seed(cfg) ✓
- [ ] Check scripts/extract_edge_scores.py uses get_seed(cfg) ✓
- [ ] Check scripts/train_aggregator.py uses get_seed(cfg) ✓
- [ ] Check all imports at top of files (PEP 8 compliance) ✓
- [ ] No mid-file imports of seed utilities ✓

**Create Reproducibility Test**:
- [ ] Create `scripts/test_reproducibility.py`
- [ ] Function: load config, run training twice with same seed
- [ ] Compare: val_auc, test_auc, predictions (should be identical)
- [ ] Usage: `python scripts/test_reproducibility.py --dataset epinions --seed 42 --num-runs 2`
- [ ] Output: ✓ All predictions identical or ✗ FAILED (with details)

**Test on All Datasets**:
- [ ] Test wiki-rfa (reproducibility test 2 runs, verify identical metrics)
- [ ] Test epinions (reproducibility test 2 runs, verify identical metrics)
- [ ] Test slashdot090221 (reproducibility test 2 runs, verify identical metrics)
- [ ] Document results

**Document Reproducibility**:
- [ ] Create/update `docs/REPRODUCIBILITY.md`
- [ ] Explain: what reproducibility means, why it matters
- [ ] Document: all seed injection points (get_seed() usage)
- [ ] Document: worker_init_fn behavior
- [ ] Clarify: PyTorch vs NumPy seed differences
- [ ] Show: how to test reproducibility
- [ ] Provide: examples of reproducible vs non-reproducible code

**Code Cleanup**:
- [ ] Update comments referencing old seed keys (walk_seed, etc.)
- [ ] Update docstrings to mention reproducibility
- [ ] Add references to REPRODUCIBILITY.md

**Success Criteria**:
- ✓ No weird seed patterns found (or documented if exist)
- ✓ All entry points use get_seed(cfg)
- ✓ Reproducibility test passes all datasets
- ✓ REPRODUCIBILITY.md complete and clear
- ✓ Code comments updated

---

### ✅ Task B: Test Metrics Tracking (INDEPENDENT)

**Use Original Prompt**:
- [ ] CHAT_B_TEST_METRICS.md (no changes)
- [ ] Can proceed independently
- [ ] Benefits from A1's reproducibility

---

### ✅ Task C: Prediction Caching (INDEPENDENT)

**Use Original Prompt**:
- [ ] CHAT_C_PREDICTION_CACHING.md (no changes)
- [ ] Can proceed independently
- [ ] Benefits from A1's reproducibility

---

### ✅ Task F: Aggregator Integration (FINAL)

**Use Original Prompt**:
- [ ] CHAT_F_AGGREGATOR_INTEGRATION.md (no changes)
- [ ] Depends on all other tasks
- [ ] Start after B, C, D, E complete

---

## Execution Plan

### Phase 1: Critical Data Validation (START IMMEDIATELY)
```
Task A6: Stratified Splitting
├─ Priority: CRITICAL (affects all model evaluation)
├─ Time: 2-3 hours
├─ Blockers: None (depends on A1 ✓)
└─ Deliverable: Updated split_edges() with stratified sampling
```

### Phase 2: Verification Tasks (START IMMEDIATELY)
```
Task D1: Walk Verification      Task E: Seed Verification
├─ Priority: HIGH              ├─ Priority: HIGH
├─ Time: 2-3 hours             ├─ Time: 1-2 hours
├─ Blockers: None (A1 ✓)      ├─ Blockers: None (A1 ✓)
└─ Deliverable:                └─ Deliverable:
  WALK_SAMPLING.md               test_reproducibility.py
  walk benchmarks                REPRODUCIBILITY.md
  optimization list
```

### Phase 3: Feature Development (START AFTER A6)
```
Task B: Test Metrics            Task C: Prediction Caching
├─ Priority: MEDIUM             ├─ Priority: MEDIUM
├─ Time: 2-3 hours              ├─ Time: 6 hours (C1+C2+C3)
├─ Blockers: None (A1 ✓)       ├─ Blockers: None (A1 ✓)
└─ Deliverable:                 └─ Deliverable:
  Per-epoch test logging         Raw predictions cached
  TensorBoard curves             Position analysis
                                 Position-aware features
```

### Phase 4: Optimization (START AFTER D1)
```
Task D2: I/O Optimization
├─ Priority: MEDIUM
├─ Time: 3-4 hours
├─ Blockers: D1 complete (benchmarks)
└─ Deliverable:
  Optimized file format
  Improved I/O performance
  Performance report
```

### Phase 5: Final Integration (START AFTER ALL)
```
Task F: Aggregator Integration Strategy
├─ Priority: FINAL
├─ Time: 2-3 hours
├─ Blockers: B, C, D, E complete
└─ Deliverable:
  Architecture decision
  Integration implementation
  Performance comparison
```

---

## Document Reference Guide

| When You Need To... | Read This |
|-------------------|----------|
| Get overview of all changes | COMPLETE_ANALYSIS.md |
| Quick facts and code patterns | QUICK_REFERENCE.md |
| Understand A6 stratified splitting | CHAT_A_CONFIG_REPRODUCIBILITY_UPDATED.md |
| Understand walk reproducibility | WALK_REPRODUCIBILITY_EXPLAINED.md |
| Understand config system | CONFIG_GUIDE.md |
| See what A1 changed | TASK_A1_FINAL_STATUS.md |
| Find file locations | FILES_SUMMARY.md |
| Understand task impacts | A1_IMPACT_ANALYSIS.md |

---

## Success Metrics

### A6: Stratified Splitting
- ✓ Each split maintains ±1-2% class balance
- ✓ Works on all 3 datasets (wiki, epinions, slashdot)
- ✓ Cache invalidation works correctly
- ✓ Model training completes successfully

### D1: Walk Verification
- ✓ Verified with 1, 2, 4, 8 workers (identical output)
- ✓ Benchmarks document current performance
- ✓ Top 3 optimizations identified with estimates
- ✓ WALK_SAMPLING.md complete

### E: Seed Verification
- ✓ No weird seed patterns remain
- ✓ All entry points use get_seed()
- ✓ Test reproducibility passes all datasets
- ✓ REPRODUCIBILITY.md complete
- ✓ Running with seed=42 twice produces identical outputs (bit-level)

---

## Red Flags / Risk Mitigation

### A6 Risks
- ❌ Random shuffle still used (need stratify=labels)
- ❌ Only test on one dataset (must test all 3)
- ❌ Don't verify class distribution (±1-2% check critical)
- ❌ Forget to clear cache (old splits will be reused)

**Mitigation**: Provide code template; verify implementation step-by-step

### D1 Risks
- ❌ Don't understand per-walk seeding (read WALK_REPRODUCIBILITY_EXPLAINED.md)
- ❌ Don't test with different worker counts (1, 2, 4, 8)
- ❌ Benchmark only one dataset (test all 3)

**Mitigation**: Provide detailed algorithm documentation; list specific test cases

### E Risks
- ❌ Skip testing (end-to-end reproducibility critical)
- ❌ Don't audit comprehensively (check all Python files)
- ❌ Assume A1 got everything (still verify)

**Mitigation**: Provide test template; checklist of code patterns to find

---

## Final Status

All documents and updated prompts are ready for distribution.

**Next action**: Assign tasks and distribute appropriate updated prompts.

```
✅ All reference documents created
✅ All updated task prompts created
✅ Implementation checklists provided
✅ Code templates included
✅ Success criteria clearly defined
✅ Timeline and dependencies documented
✅ Ready for chat distribution

→ READY TO PROCEED
```
