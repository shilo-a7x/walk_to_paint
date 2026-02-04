# CHAT A: Config System & Reproducibility Foundation

## Task A1: Config System Overhaul & Full Reproducibility

### Status: ✅ COMPLETE

This task has been completed. See the following documentation files for details:
- [TASK_A1_FINAL_STATUS.md](TASK_A1_FINAL_STATUS.md) - Complete status report with all changes
- [WALK_REPRODUCIBILITY_EXPLAINED.md](WALK_REPRODUCIBILITY_EXPLAINED.md) - Technical explanation of walk algorithm
- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - Configuration system documentation

**Key achievement**: Single unified seed (`reproducibility.seed`) with per-walk deterministic seeding guarantees bit-for-bit identical walk files regardless of worker count.

---

## Task A1: Config System Overhaul & Full Reproducibility (ORIGINAL DETAILS)

### Current Problem
The configuration system is inconsistent across the codebase:
- Seed values scattered: `walk_seed`, `worker_seed`, `training.seed` (different names, same purpose)
- Some cfg keys are never set by users because they have inconsistent names
- Silent defaults when keys are missing
- Entry point `run.py` sets seeds globally, but `scripts/extract_edge_scores.py` etc. don't inherit properly
- Reproducibility issues documented in REPRODUCIBILITY_REVIEW.md

### Your Task
1. **Audit the entire codebase** for seed-related config keys:
   - Find all seed mentions in: `config.yaml`, dataset yamls, `prepare_data.py`, `run.py`, `extract_edge_scores.py`, `train_aggregator.py`, data utilities
   - Document which seed keys are used where
   - Identify naming inconsistencies

2. **Unify seed configuration**:
   - Create single canonical seed source in config (e.g., `reproducibility.seed`)
   - All seed usages must reference the same config key
   - No silent defaults - fail loudly if seed not set

3. **Fix worker/walk configuration**:
   - Ensure `training.num_workers`, `training.persistent_workers`, `preprocess.walk_num_workers` are all documented
   - Make sure they're set with sensible defaults in base config
   - Propagate consistently to prepare_data and walk_sampler

4. **Ensure scripts inherit config properly**:
   - `run.py` sets torch/numpy seeds globally → should work for all downstream
   - `extract_edge_scores.py` when called directly should load config and apply seeds
   - Same for any standalone script

5. **Document the config hierarchy**:
   - Create CONFIG_GUIDE.md explaining:
     - All reproducibility-related keys
     - How they flow through the pipeline
     - What happens when you use scripts directly vs through run.py

### Files to Modify
- `config.yaml` (add/consolidate seed config)
- `configs/epinions.yaml`, `configs/slashdot090221.yaml`, `configs/wiki-rfa.yaml`
- `src/utils/config.py` (if it exists, or create it)
- `src/data/prepare_data.py`
- `scripts/extract_edge_scores.py`
- `scripts/train_aggregator.py`
- `run.py`

### Success Criteria
✅ All seed references use single config key  
✅ No silent defaults for seed-related configs  
✅ extract_edge_scores.py produces identical results whether called from run.py or directly  
✅ CONFIG_GUIDE.md explains full reproducibility model  
✅ All seed values logged at startup  

---

## Task A6: Data Building Validation

### Current Problem
We need to validate that data is built correctly according to our specs:
- Multiedge handling strategy needs verification
- Binary mode must work correctly across all datasets
- Non-integer node IDs (e.g., wiki) need standardization to [0, n-1]
- Walk sampler might assume this node range → need to verify

### Your Task
1. **Understand multiedge handling**:
   - How are duplicate (u,v) edges with different labels handled?
   - Check `src/data/datasets.py` (load functions)
   - Verify it matches intent in `config.yaml` (`multiedge_handling: keep|collapse`)
   - Test edge cases

2. **Validate binary mode**:
   - For datasets with `binary: true`, verify edge labels are [0,1]
   - Check that aggregation/training handles binary classification correctly
   - Test with epinions (binary) and slashdot (binary)

3. **Node ID standardization**:
   - Wiki-rfa uses non-sequential node IDs → need mapping to [0, n-1]
   - Check if node loader does this (grep for `relabel`, `remap`, `0-n-1`)
   - If not, add it + verify walk sampler works with this
   - Document the mapping for reproducibility

4. **Walk sampler assumptions**:
   - Read `src/data/walk_sampler.py` carefully
   - Does it assume nodes are [0, n-1]? (check edge sampling logic)
   - If yes, ensure all datasets are preprocessed to this format

5. **Data validation script**:
   - Create `scripts/validate_data.py` that:
     - Loads edge list for each dataset
     - Verifies node IDs are [0, n-1]
     - Checks multiedge handling
     - Validates binary vs multiclass labels
     - Prints summary statistics

### Files to Check/Modify
- `src/data/datasets.py` (all loader functions)
- `src/data/walk_sampler.py` (understand assumptions)
- Create `scripts/validate_data.py`
- Possibly: `data/epinions/`, `data/slashdot090221/`, `data/wiki-rfa/` loaders

### Success Criteria
✅ All node IDs are [0, n-1] after loading  
✅ Binary vs multiclass handling is explicit in code  
✅ Multiedge handling matches config intent  
✅ validate_data.py passes for all datasets  
✅ Walk sampler documentation clarifies its assumptions  

---

## Important Notes
- **May trigger retraining**: If you need to rebuild node mappings or fix multiedge handling, data cache will be stale
- **Coordinate with Chat D**: After A6, Chat D will review walk building and pipeline optimization
- **Dependency**: Both A1 and A6 must complete before other chats can proceed confidently
