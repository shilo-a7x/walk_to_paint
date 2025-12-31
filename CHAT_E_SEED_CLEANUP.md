# CHAT E: Seed & Reproducibility Cleanup

## Task E1: Remove Weird Seed Constructs & Simplify Propagation

### Current Problem
The codebase has several non-deterministic and confusing seed-related patterns:
- `np.random.seed(seed % (2**32-1))` — why the modulo? It's weird and undocumented
- `seed = int(time.time()) + worker_id` — fallback to time defeats reproducibility
- Multiple seed keys with inconsistent naming conventions
- Silent fallbacks when config keys are missing

### Your Task
1. **Audit all seed-related code**:
   - Find all lines with: `seed`, `random`, `np.random`, `torch.manual_seed`, `torch.cuda`
   - Files to check: `src/data/prepare_data.py`, `src/utils/config.py`, any file with seed logic
   - Document: where is each weird construct used?

2. **Understand why weird constructs exist**:
   - Is there a reason for `% (2**32-1)`? (numpy limit? platform difference?)
   - Is `int(time.time())` fallback intentional or leftover from debugging?
   - Would removing these break anything?

3. **Replace with deterministic alternatives**:
   - Remove `% (2**32-1)` modulo if not necessary
   - Remove time-based fallbacks → use config-required seed instead
   - If certain operations need special seed handling, document it
   - Ensure all paths are deterministic (no randomness in fallbacks)

4. **Simplify seed propagation**:
   - Create single helper function `set_all_seeds(seed)`:
     - Sets numpy, torch, cuda, python random
     - Works for DataLoader worker_init_fn
     - Can be called from run.py, scripts, anywhere
   - Export from `src/utils/reproducibility.py` or similar
   - Use everywhere instead of duplicated seed logic

5. **Update worker_init_fn**:
   - Currently uses custom seed logic in prepare_data.py
   - Should use helper function from step 4
   - No more `int(time.time())` fallbacks

6. **Add comprehensive seeding documentation**:
   - Create `docs/REPRODUCIBILITY.md`:
     - Explains deterministic requirements
     - Documents all seed injection points
     - Shows how to use set_all_seeds()
     - Explains worker_init_fn behavior
     - Provides testing approach

7. **Test reproducibility**:
   - Run training with same seed twice
   - Compare outputs: exact same tensors? (bit-level reproducibility)
   - Check predictions are identical
   - Create `scripts/test_reproducibility.py` for future regression testing

### Files to Modify
- `src/data/prepare_data.py` (worker_init_fn + seed logic)
- Any file with seed operations
- Create `src/utils/reproducibility.py` (centralized seed handling)
- Create `docs/REPRODUCIBILITY.md` (documentation)
- Create `scripts/test_reproducibility.py` (validation)

### Success Criteria
✅ No `% (2**32-1)` modulo (unless documented reason)  
✅ No `int(time.time())` fallbacks (seed required, not optional)  
✅ All seed operations use `set_all_seeds()` helper  
✅ Same seed produces identical training trajectory (bit-level reproducibility)  
✅ REPRODUCIBILITY.md explains full system  
✅ test_reproducibility.py passes for all datasets  

### Testing Reproducibility (validation)
```bash
# Run twice with same seed, compare outputs
python scripts/test_reproducibility.py --dataset epinions --seed 42 --num-runs 2
# Output: All predictions identical ✓
```

---

## Integration Notes
- **Depends on**: Chat A (config unification)
- **No conflicts**: Pure code cleanup, no API changes
- **Helps**: Makes testing reproducibility easier for other chats
- **Estimated time**: 2-3 hours (audit + refactor + documentation + testing)

## Cleanup Checklist
- [ ] Audit: find all seed operations
- [ ] Understand: document why each pattern exists
- [ ] Replace: use set_all_seeds() helper everywhere
- [ ] Document: REPRODUCIBILITY.md with full explanation
- [ ] Test: run reproducibility test, verify bit-level matching
- [ ] Verify: aggregator scripts also pass reproducibility test

---

## Example: Before/After

### Before (weird constructs)
```python
def worker_init_fn(worker_id):
    np.random.seed(seed % (2**32-1))  # Why modulo?
    torch.manual_seed(seed)

def prepare_data():
    if 'walk_seed' not in cfg:
        seed = int(time.time()) + worker_id  # Why time-based?
    np.random.seed(seed)
```

### After (clean)
```python
from src.utils.reproducibility import set_all_seeds

def worker_init_fn(worker_id):
    set_all_seeds(base_seed + worker_id)

def prepare_data():
    set_all_seeds(cfg.reproducibility.seed)
```

---

## Questions to Answer in Documentation
1. Why does PyTorch need different seeds than NumPy?
2. How does worker_init_fn seed differ from main process seed?
3. What does bit-level reproducibility mean and why is it important?
4. How do I verify reproducibility of my trained model?
