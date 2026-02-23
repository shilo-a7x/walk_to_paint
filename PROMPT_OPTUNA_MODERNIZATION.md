# PROMPT: Modernize Optuna Script for Anti-Overfitting Focus

## Context

You are working with a graph neural network (GNN) project that uses random walks on signed graphs for edge prediction. The project has undergone significant infrastructure updates:

1. **Unified Seeding Strategy** (A1): All randomness uses `reproducibility.seed` via `get_seed(cfg)` utility
2. **Dataset Caching** (T1.3): Unified `.pt` cache format for fast loading (see DATASET_CACHE_COMPLETE.md)
3. **Stratified Splits** (A6): Train/mask/val/test with proper class balance
4. **Class-Weighted Loss** (T2.2): Mandatory weighting computed from train split only

**Current Problem**: Models overfit aggressively - train loss drops but val loss plateaus or increases. Need to update `optuna_run.py` to focus hyperparameter search on **regularization** to combat overfitting.

---

## Files to Examine

### Core Files

- `optuna_run.py` - The Optuna script to update
- `config.yaml` - Base config with `reproducibility.seed` and `preprocess.use_cache`
- `src/utils/config.py` - Contains `get_seed(cfg)` utility function
- `src/data/prepare_data.py` - Data preparation with caching support
- `DATASET_CACHE_COMPLETE.md` - Cache system documentation

### Reference Documents

- `CURRENT_STATUS_ASSESSMENT.md` - Overall project status
- `WALK_REPRODUCIBILITY_EXPLAINED.md` - How seeding works across workers

---

## Task 1: Update Seeding Strategy

### Current State

`optuna_run.py` currently imports and uses:

```python
from pytorch_lightning import seed_everything
```

### Required Changes

1. **Remove**: Direct calls to `seed_everything()` or manual seeding
2. **Add**: Import `get_seed` from `src.utils.config`
3. **Use**: `seed = get_seed(cfg)` to get the canonical seed from `cfg.reproducibility.seed`
4. **Apply**: Use this seed for all randomness sources:
   - `seed_everything(seed)` - PyTorch Lightning
   - `torch.manual_seed(seed)`
   - `np.random.seed(seed)`
   - `random.seed(seed)`

### Example Pattern (from other scripts)

```python
from src.utils.config import get_seed

# In objective function or main
seed = get_seed(cfg)
seed_everything(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
```

**Why**: Ensures all trials use consistent seeding from config, not hardcoded values.

---

## Task 2: Fix Walk Parameters (No Longer Hyperparameters)

### Current State

`optuna_run.py` currently tunes:

```python
OPTUNA_RANGES = {
    "dataset.max_walk_length": (10, 100),
    "dataset.num_walks": (100000, 5000000),
    ...
}

# In objective():
cfg.dataset.max_walk_length = trial.suggest_int("dataset.max_walk_length", ...)
cfg.dataset.num_walks = trial.suggest_int("dataset.num_walks", ...)
```

### Required Changes

1. **Remove** from `OPTUNA_RANGES`:
   - `"dataset.max_walk_length"`
   - `"dataset.num_walks"`

2. **Remove** from `objective()` function:
   - `cfg.dataset.max_walk_length = trial.suggest_int(...)`
   - `cfg.dataset.num_walks = trial.suggest_int(...)`

3. **Use fixed values**:
   - `cfg.dataset.max_walk_length = 80` (FIXED)
   - `cfg.dataset.num_walks = 5000000` (FIXED - 5M walks)

4. **Update seeding logic** (if present in `seed_initial_trials()` or similar):
   - Remove walk parameters from initial trial seeding
   - Keep them as fixed config values

**Why**:

- Walk generation is expensive (caching helps but still slow)
- Changing walks per trial defeats caching benefits
- Focus search on model/training hyperparameters, not data generation
- Reduces search space = faster convergence
- **Fixed values**: walk_length=80, num_walks=5M, batch_size=512

---

## Task 3: Enable Dataset Caching

### Current State

Likely no explicit cache configuration in `optuna_run.py`.

### Required Changes

1. **Ensure** `cfg.preprocess.use_cache = True` before calling `prepare_data()`
2. **Verify** cache is used by checking for cache hits in prepare_data output
3. **Document** in trial logs whether cache was used (for debugging)

### Example

```python
# Before prepare_data() call in objective
cfg.preprocess.use_cache = True  # Ensure caching enabled
cfg.preprocess.save = True       # Save cache if not exists

# Call prepare_data
train_pack, val_pack, test_pack, mask_pack = prepare_data(cfg)
```

**Cache Behavior**:

- First trial with given dataset: builds and saves cache (~40s)
- Subsequent trials: loads from cache (~3-5s)
- Cache key: dataset name + walk parameters (length, num_walks)
- Since walks are now FIXED, all trials share same cache!

**Why**: With fixed walks, every trial can reuse the same cached dataset, massively speeding up search.

---

## Task 4: Focus Hyperparameter Search on Anti-Overfitting

### Problem Analysis

Models overfit = train loss drops but val loss increases/plateaus. This means:

- Model capacity too high for data complexity
- Insufficient regularization
- Learning too fast (high LR)
- Need stronger weight decay, dropout, gradient clipping

### Current Hyperparameter Ranges (OPTUNA_RANGES)

```python
OPTUNA_RANGES = {
    # Training
    "training.lr": (1e-5, 1e-1),              # Too wide
    "training.weight_decay": (1e-8, 1e-1),    # Lower bound too small
    "training.batch_size": [32, 64, 128, 256, 512],
    "training.gradient_clip_val": (0.1, 2.0),
    "training.early_stopping_patience": (5, 20),
    
    # Model architecture
    "model.nhead": [1, 2, 4, 8, 16],
    "model.embedding_dim": [4, 8, 16, 32, 64, 128],
    "model.hidden_dim": [4, 8, 16, 32, 64, 128, 256],
    "model.nlayers": (1, 6),
    "model.dropout": (0.0, 0.7),              # Should start higher
}
```

### Recommended Changes for Anti-Overfitting

#### 1. Learning Rate (Slower = Less Overfit)

```python
# Old: (1e-5, 1e-1) - too wide, high values overfit
# New: Narrow to moderate range
"training.lr": (1e-5, 1e-3),  # Max 0.001 instead of 0.1
```

#### 2. Weight Decay (Stronger Regularization)

```python
# Old: (1e-8, 1e-1) - lower bound too small
# New: Start from meaningful regularization
"training.weight_decay": (1e-5, 1e-1),  # Min 0.00001 instead of 1e-8
```

#### 3. Dropout (Higher = More Regularization)

```python
# Old: (0.0, 0.7) - allows no dropout
# New: Force at least some dropout
"model.dropout": (0.2, 0.7),  # Min 20% dropout
```

#### 4. Model Capacity (Smaller = Less Overfit)

```python
# Old: hidden_dim up to 256
# New: Cap at smaller sizes
"model.hidden_dim": [16, 32, 64, 128],  # Remove 256, remove tiny 4/8
"model.embedding_dim": [16, 32, 64],    # Remove tiny 4/8, cap at 64

# Old: nlayers (1, 6)
# New: Fewer layers
"model.nlayers": (1, 4),  # Cap at 4 layers instead of 6
```

#### 5. Batch Size (FIXED - Not Tuned)

```python
# Old: [32, 64, 128, 256, 512] - was hyperparameter
# New: FIXED at 512 (remove from OPTUNA_RANGES)
# Set in objective(): cfg.training.batch_size = 512
```

#### 6. Early Stopping Patience (More Aggressive)

```python
# Old: (5, 20) - too patient
# New: Stop faster when val loss stops improving
"training.early_stopping_patience": (3, 10),  # Shorter patience
```

### Complete Updated OPTUNA_RANGES

```python
OPTUNA_RANGES = {
    # REMOVED: Walk parameters (now FIXED)
    # dataset.max_walk_length = 80 (FIXED)
    # dataset.num_walks = 5000000 (FIXED)
    # training.batch_size = 512 (FIXED)
    
    # Training - Anti-overfitting focus
    "training.lr": (1e-5, 1e-3),                    # ← Slower learning
    "training.weight_decay": (1e-5, 1e-1),          # ← Stronger L2 reg
    "training.gradient_clip_val": (0.5, 2.0),       # ← Higher min (more clipping)
    "training.early_stopping_patience": (3, 10),    # ← Stop faster
    
    # Model - Smaller capacity
    "model.nhead": [1, 2, 4, 8],                    # ← Remove 16
    "model.embedding_dim": [16, 32, 64],            # ← Remove 4,8,128
    "model.hidden_dim": [16, 32, 64, 128],          # ← Remove 4,8,256
    "model.nlayers": (1, 4),                        # ← Max 4 layers
    "model.dropout": (0.2, 0.7),                    # ← Min 20% dropout
}

# In objective() function, SET FIXED VALUES:
cfg.dataset.max_walk_length = 80           # FIXED
cfg.dataset.num_walks = 5000000            # FIXED (5M)
cfg.training.batch_size = 512              # FIXED
```

---

## Task 5: Update Trial Logging

### Current State

Trial logs likely show walk parameters, old seeding info.

### Required Changes

1. **Log seed source**: "Using seed from cfg.reproducibility.seed: {seed}"
2. **Log cache status**: "Dataset cache: HIT" or "Dataset cache: MISS (building...)"
3. **Remove walk parameter logs** from per-trial output (since they're fixed)
4. **Add overfitting metrics**:
   - `train_loss_final` (from last epoch)
   - `val_loss_final` (from last epoch)
   - `overfit_ratio = val_loss_final / train_loss_final` (>1.0 = overfitting)

### Example

```python
print(f"[Trial {trial.number}]")
print(f"  Seed (from config): {seed}")
print(f"  Dataset cache: {'HIT' if cache_existed else 'MISS'}")
print(f"  LR: {cfg.training.lr:.2e}, WD: {cfg.training.weight_decay:.2e}")
print(f"  Dropout: {cfg.model.dropout:.2f}, Hidden: {cfg.model.hidden_dim}")
print(f"  Results: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
print(f"  Overfit ratio: {val_loss/train_loss:.3f}")  # >1.0 = overfitting
```

---

## Task 6: Verification Steps

After making changes, verify:

1. **Seeding works**:

   ```bash
   # Run same trial twice, should get identical results
   python optuna_run.py --config config.yaml --n-trials 1
   python optuna_run.py --config config.yaml --n-trials 1
   # Check: same val_auc?
   ```

2. **Caching works**:

   ```bash
   # First trial: should build cache
   # Second trial: should load from cache (much faster)
   rm -rf data/bitcoin-alpha-binary/dataset_cache.pt  # Clean slate
   python optuna_run.py --config config.yaml --n-trials 2
   # Check logs: "Building cache..." then "Loading from cache..."
   ```

3. **Walks and batch size are fixed**:

   ```bash
   # Check OPTUNA_RANGES - should NOT contain walk or batch_size parameters
   grep -n "max_walk_length\|num_walks\|batch_size" optuna_run.py
   # Should only appear in: 
   #   - Fixed assignments: cfg.dataset.max_walk_length = 80
   #   - Fixed assignments: cfg.dataset.num_walks = 5000000
   #   - Fixed assignments: cfg.training.batch_size = 512
   #   - Logs (for debugging)
   # Should NOT appear in: trial.suggest_*
   ```

4. **Anti-overfitting ranges**:

   ```bash
   # Run study, check if val_loss / train_loss ratio improves
   python optuna_run.py --config config.yaml --n-trials 20
   # Compare best trial's overfit ratio vs old trials
   ```

---

## Expected Outcomes

### Before (Current State)

- Walk parameters tuned → different cache per trial → SLOW
- Wide hyperparameter ranges → explores high-capacity, overfitting models
- No unified seeding → trials not reproducible
- Models overfit: train_loss << val_loss

### After (Desired State)

- Walk parameters fixed → single cache shared → FAST
- Narrow regularization-focused ranges → explores anti-overfitting models
- Unified seeding via `get_seed(cfg)` → reproducible trials
- Models regularized: train_loss ≈ val_loss (better generalization)

---

## Code Locations Reference

### Main Sections to Edit in optuna_run.py

1. **Line ~20-25**: Imports
   - Add: `from src.utils.config import get_seed`

2. **Line ~30-45**: `OPTUNA_RANGES` dict
   - Remove: walk parameters
   - Update: all ranges per Task 4

3. **Line ~210-270**: `objective()` function
   - Add: `seed = get_seed(cfg)` near top
   - Add: Seeding calls (seed_everything, torch, numpy, random)
   - Remove: walk parameter suggestions
   - Add: Cache status logging

4. **Line ~320-340**: Trial result logging
   - Update: Remove walk params from logs
   - Add: Cache hit/miss status
   - Add: Overfit ratio calculation

5. **Line ~450-500**: `seed_initial_trials()` (if exists)
   - Remove: walk parameter seeding
   - Keep: model/training param seeding

---

## Additional Notes

### Why This Matters

- **Speed**: Fixed walks + caching = 10-20x faster trial iteration
- **Focus**: Regularization-focused search directly addresses overfitting
- **Reproducibility**: Unified seeding ensures consistent results
- **Efficiency**: Smaller search space = faster convergence to good hyperparams

### Integration with Current Workflow

This update complements:

- T2.2 (class-weighted loss) - already helping with class imbalance
- A6 (stratified splits) - already ensuring fair train/val/test splits
- Caching system - now fully leveraged by Optuna

### Next Steps After This Task

Once Optuna script is updated:

1. Run hyperparameter search on one dataset (e.g., bitcoin-alpha-binary)
2. Analyze best trial: does overfit ratio improve?
3. If yes, apply best hyperparams to all 3 datasets
4. Retrain with optimized hyperparams + class weights + stratified splits

---

## Success Criteria

✅ **Seeding**: All trials use `get_seed(cfg)`, no hardcoded seeds  
✅ **Walks**: `max_walk_length` and `num_walks` removed from `OPTUNA_RANGES`  
✅ **Caching**: `cfg.preprocess.use_cache = True` before `prepare_data()`  
✅ **Anti-overfitting**: Updated ranges per Task 4 recommendations  
✅ **Logging**: Shows seed source, cache status, overfit ratio  
✅ **Verification**: Runs successfully, trials faster (cache hit), val/train ratio improves  

---

## Questions to Clarify (If Needed)

1. **Dataset to test**: Which dataset should be used for initial Optuna run? (Suggest: bitcoin-alpha-binary - smallest)
2. **Number of trials**: How many trials for initial test? (Suggest: 20-30 for quick validation)
3. **Metrics**: Primary metric to optimize (currently val_auc)? Keep or change to val_loss?
4. **Pruning**: Keep Optuna pruning enabled or disable for full training per trial?

---

## Final Checklist

Before considering task complete:

- [ ] `get_seed(cfg)` used for all seeding
- [ ] Walk parameters removed from hyperparameter search
- [ ] Caching explicitly enabled
- [ ] `OPTUNA_RANGES` updated with anti-overfitting focus
- [ ] Trial logging shows seed, cache status, overfit ratio
- [ ] Verified on test run: faster trials, cache reuse working
- [ ] Documented changes in code comments or docstring
