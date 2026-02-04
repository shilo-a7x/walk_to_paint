# CHAT E: Seed & Reproducibility Verification & Polish

## Task E1: Verify Seed Cleanup & Remove Any Remaining Weird Constructs

### Status Update: A1 Completed Most Work ✅
Task A1 has already addressed the major reproducibility issues:
- Unified seed configuration: Single `reproducibility.seed` in config
- Removed scattered seed keys: `walk_seed`, `worker_seed`, `training.seed` consolidated
- Created `get_seed()` utility: Centralized, fails loudly if seed missing
- Fixed edge split seeding: `random.seed(seed)` before shuffle
- Fixed walk sampling: Per-walk deterministic seeding (walk[i] uses seed base+i)

Your task is to **verify these changes are complete** and polish any remaining issues.

### Current Problem
Need to ensure:
1. No weird seed constructs remain (% (2**32-1), time.time() fallbacks)
2. All seed operations use the unified `get_seed()` helper
3. Documentation is clear and consistent
4. Reproducibility is truly guaranteed end-to-end

### Your Task

1. **Audit code for remaining weird seed patterns**:
   - Search entire codebase for these suspicious patterns:
     - `% (2**32-1)` — unnecessary modulo (should be removed)
     - `int(time.time())` in seed context — defeats reproducibility (should be removed)
     - `getattr(cfg, ..., seed=None)` with fallback — should use `get_seed(cfg)` instead
     - Any hardcoded seed values (should come from config)
   - Files to check: all Python files that import random/numpy/torch
   - Document what you find (location, type, impact)

2. **Verify get_seed() is used consistently**:
   - Check that all entry points use `get_seed(cfg)`:
     - `run.py` ✓ (likely already done in A1)
     - `optuna_run.py` ✓ (likely already done in A1)
     - `scripts/extract_edge_scores.py` ✓ (likely already done in A1)
     - `scripts/train_aggregator.py` ✓ (likely already done in A1)
   - Any script that loads config should call `get_seed(cfg)` if it uses randomness
   - Verify no mid-file imports of seed utilities (PEP 8 compliance)

3. **Verify reproducibility end-to-end**:
   - Create `scripts/test_reproducibility.py`:
     ```python
     # Run training twice with same seed, verify identical results
     # Usage: python scripts/test_reproducibility.py --dataset epinions --seed 42
     
     def test_reproducibility(dataset, seed, num_epochs=2):
         # Run 1: Train model
         cfg1 = load_config(dataset, seed=seed)
         model1, val_metrics1 = train_model(cfg1, num_epochs=num_epochs)
         preds1 = model1(test_data)
         
         # Run 2: Train same model again
         cfg2 = load_config(dataset, seed=seed)
         model2, val_metrics2 = train_model(cfg2, num_epochs=num_epochs)
         preds2 = model2(test_data)
         
         # Verify identical
         assert torch.allclose(preds1, preds2, atol=1e-7), "Predictions differ!"
         assert abs(val_metrics1['val_auc'] - val_metrics2['val_auc']) < 1e-7, "Metrics differ!"
         return True  # ✓ Reproducible
     ```
   - Run on all datasets (wiki-rfa, epinions, slashdot)
   - Report: ✓ Reproducible or ✗ Failed (with which step)

4. **Document reproducibility guarantee**:
   - Create or update `docs/REPRODUCIBILITY.md`:
     - Explains what "reproducible" means (same seed → identical outputs)
     - Documents all seed injection points
     - Shows how to use `get_seed()` for new code
     - Explains worker_init_fn behavior (per-worker seed generation)
     - Provides testing approach for verifying reproducibility
     - Clarifies PyTorch vs NumPy seed differences (why both needed)
     - Example: Running same training 10x with seed=42 should produce identical val/test metrics

5. **Check config seed validation**:
   - Verify `get_seed(cfg)` fails with clear error if `reproducibility.seed` missing
   - Verify error message is helpful: "Config must have reproducibility.seed set"
   - Verify no silent fallbacks (no defaults, no None handling)
   - Test by running with broken config

6. **Documentation cleanup**:
   - Update any comments in code that reference old seed keys (walk_seed, etc.)
   - Ensure docstrings explain seed behavior
   - Add references to REPRODUCIBILITY.md where appropriate

### Files to Check
- All Python files with import random/numpy.random/torch.manual_seed
- Especially: prepare_data.py, walk_sampler.py, run.py, optuna_run.py, all scripts
- Also: Any config files that mention seed

### Files to Create/Update
- Create `scripts/test_reproducibility.py` (validation test)
- Create/Update `docs/REPRODUCIBILITY.md` (comprehensive guide)
- Update any code comments about seeding

### Success Criteria
✅ No `% (2**32-1)` remaining (or documented reason if exists)  
✅ No `int(time.time())` in seed logic (all seeds come from config)  
✅ All entry points use `get_seed(cfg)` consistently  
✅ test_reproducibility.py passes for all datasets  
✅ REPRODUCIBILITY.md explains the full system  
✅ Code comments updated to reference unified seed  
✅ Running with seed=42 twice produces identical model outputs (bit-level)  

### Testing Reproducibility (validation)
```bash
# Verify full reproducibility
python scripts/test_reproducibility.py --dataset epinions --seed 42 --num-runs 2
# Output: ✓ All predictions identical!

python scripts/test_reproducibility.py --dataset wiki-rfa --seed 42 --num-runs 2
# Output: ✓ All predictions identical!

python scripts/test_reproducibility.py --dataset slashdot090221 --seed 42 --num-runs 2
# Output: ✓ All predictions identical!
```

---

## Integration Notes
- **Depends on**: Chat A (A1 completed most work)
- **No conflicts**: Pure verification and polish, no major API changes
- **Helps**: Makes testing reproducibility easier for other chats
- **Estimated time**: 1-2 hours (mostly verification + test writing)

## Why This Matters
- Reproducibility is **critical** for scientific validity
- Allows other researchers to verify your results
- Enables debugging (same seed = same behavior = easy to track issues)
- Required for many conferences/journals
- Makes experimentation more reliable

## Checklist for Completion
- [ ] Audit: Found and documented all weird seed patterns
- [ ] Verify: All entry points use get_seed(cfg)
- [ ] Test: test_reproducibility.py passes for all 3 datasets
- [ ] Document: REPRODUCIBILITY.md complete and clear
- [ ] Validate: Code comments updated
- [ ] Summary: Report any remaining issues (if any)
