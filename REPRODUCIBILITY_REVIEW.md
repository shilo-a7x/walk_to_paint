# Full Reproducibility Review: Data Building Pipeline

**Date:** December 25, 2025  
**Scope:** Walk sampling, data building, file ordering, multiprocessing

---

## Executive Summary

Your data building pipeline is **mostly reproducible** with appropriate seeding, but there are **critical issues with walk file write ordering and multiprocessing behavior** that could cause subtle non-determinism. Here's the verdict:

| Component | Reproducible | Notes |
|-----------|--------------|-------|
| Edge loading | ✅ Yes | Deterministic file parsing |
| Edge splitting | ⚠️ **Partial** | Uses `random.shuffle()` without explicit seed |
| Walk sampling | ✅ Yes | Properly seeded with multiprocessing chunks |
| Walk file write order | ❌ **NO** | Multiprocessing returns vary in order; file write not guaranteed deterministic |
| Tokenizer building | ✅ Yes | Order-independent (set operations) |
| Walk encoding | ✅ Yes | Deterministic once splits are fixed |
| DataLoader shuffling | ✅ Yes | Uses seeded generator |

---

## 1. EDGE SPLITTING: REPRODUCIBILITY ISSUE ❌

### Problem
In `prepare_data.py` line 40:
```python
def split_edges(cfg, edges):
    if cfg.preprocess.use_cache and os.path.exists(split_path):
        # ... load from cache
    else:
        edges_copy = list(edges)
        random.shuffle(edges_copy)  # ← ISSUE: No seed here
```

**Issue**: `random.shuffle()` is called WITHOUT a seed. It depends on Python's global `random` module state.

### Why This Matters
1. **First run** gets unpredictable shuffle (unless you've called `random.seed()` globally)
2. **run.py sets seed** with `seed_everything(seed)` but this happens AFTER config loading
3. **If preprocessing runs twice with different seed**, edge split will differ

### The Flow
```
run.py:
  ├─→ load_config()
  ├─→ seed_everything(seed)  ← Too late! prepare_data may have used old state
  └─→ prepare_data()
       └─→ split_edges()
            └─→ random.shuffle(edges_copy)  ← Uses current random state
```

### Risk Assessment
- **If `use_cache=true`**: Safe (splits are cached, reused)
- **If `use_cache=false` AND you rerun**: Edge splits will differ between runs
- **If you change seeds**: Data splits become non-deterministic

### Fix Required
```python
def split_edges(cfg, edges):
    # ... caching logic ...
    else:
        edges_copy = list(edges)
        
        # NEW: Use explicit seed from config
        seed = getattr(cfg.training, "seed", None)
        if seed is not None:
            random.seed(int(seed))
        
        random.shuffle(edges_copy)  # Now deterministic
        # ... rest of function ...
```

---

## 2. WALK SAMPLING: REPRODUCIBILITY WITH MULTIPROCESSING WARNINGS ⚠️

### Walk Sampling Logic (walk_sampler.py)

```python
def sample_random_walks(edges, num_walks=100, max_walk_length=16, 
                        num_workers=1, seed=None):
    # ...
    base_seed = 42 if seed is None else int(seed)
    
    if num_workers == 1:
        return _sample_chunk(nodes, nbrs, lbls, num_walks, 
                           max_walk_length, 0, num_walks, base_seed)
    
    # Multiprocessing: Split walks into chunks
    chunk = math.ceil(num_walks / num_workers)
    tasks = []
    for w in range(num_workers):
        end = min(start + chunk, num_walks)
        tasks.append((start, end, base_seed + w))  # Each worker: seed+w
        start = end
    
    with mp.Pool(processes=len(tasks)) as pool:
        results = pool.starmap(
            partial(_sample_chunk, nodes, nbrs, lbls, num_walks, max_walk_length),
            tasks,
        )
    
    walks = []
    for chunk_walks in results:
        walks.extend(chunk_walks)  # ← Order depends on pool completion order
    return walks
```

### Multiprocessing Behavior Analysis

#### Good
✅ Each worker gets **deterministic seed**: `base_seed + worker_id`  
✅ Each worker uses `np.random.default_rng(seed)` (stateless NumPy RNG)  
✅ Walk generation **within each chunk is deterministic**

#### Bad/Risky
❌ **Pool results order NOT guaranteed** to match task order on all systems
- `pool.starmap()` may complete chunks out-of-order
- Worker scheduling is OS-dependent
- Threading may cause reordering on some architectures

❌ **Results are re-assembled by loop order**, NOT task order:
```python
results = pool.starmap(...)  # order: [chunk_2, chunk_0, chunk_1] possible!
for chunk_walks in results:
    walks.extend(chunk_walks)   # walks now in wrong order
```

### Practical Impact

**Scenario 1: Single machine, same OS**
- If you run **same config twice**, multiprocessing pool usually returns results in same order
- Walks are likely **same order** on reruns
- **But not guaranteed by Python spec**

**Scenario 2: Different machine/Python version**
- Pool scheduling differs
- Walk order may differ even with same seed
- **Walks are different, tokenizer changes, model retrains differently**

**Scenario 3: Cluster job resubmission**
- Different CPU scheduling
- Walk order varies
- **Data becomes non-deterministic across runs**

### Actual Risk in Your Code

From `prepare_data.py` line 95:
```python
walk_workers = int(getattr(cfg.preprocess, "walk_num_workers", 1))
walk_seed = getattr(cfg.preprocess, "walk_seed", getattr(cfg.training, "seed", None))

walks = sample_random_walks(
    edges,
    num_walks=int(cfg.dataset.num_walks),
    max_walk_length=cfg.dataset.max_walk_length,
    num_workers=walk_workers,    # ← Defaults to 1 (safe)
    seed=walk_seed,
)
```

**Current default: `walk_num_workers=1`** → Single process → **SAFE** ✅

**If you change to `walk_num_workers > 1`** → Multiprocessing enabled → **RISKY** ⚠️

---

## 3. WALK FILE WRITE ORDER: CRITICAL ISSUE ❌

### The Write Order Problem

In `prepare_data.py` lines 106-116:
```python
def get_walks(cfg, edges):
    walks_path = os.path.join(cfg.dataset.data_dir, cfg.dataset.walks_file)
    
    if cfg.preprocess.use_cache and os.path.exists(walks_path):
        try:
            walks = torch.load(walks_path)
            return walks
        except Exception:
            pass  # fall back to regenerate
    
    walk_workers = int(getattr(cfg.preprocess, "walk_num_workers", 1))
    walk_seed = getattr(cfg.preprocess, "walk_seed", 
                       getattr(cfg.training, "seed", None))
    
    walks = sample_random_walks(
        edges,
        num_walks=int(cfg.dataset.num_walks),
        max_walk_length=cfg.dataset.max_walk_length,
        num_workers=walk_workers,
        seed=walk_seed,
    )
```

### Issue 1: Walks List Order Not Guaranteed

```
sample_random_walks() with num_workers=2:
├─ Worker 0: generates walks [0:N/2] with seed=base_seed+0
├─ Worker 1: generates walks [N/2:N] with seed=base_seed+1
└─ Results collected: possibly [W1_results, W0_results]  # Wrong order!

Resulting walks list order:
Run 1: [walk_0, walk_1, ..., walk_N/2, walk_N/2+1, ..., walk_N]
Run 2: [walk_N/2+1, ..., walk_N, walk_0, walk_1, ..., walk_N/2]  # Different order!
```

### Issue 2: File Format Mismatch

From lines 112-118:
```python
if cfg.preprocess.save:
    try:
        torch.save(walks, walks_path)  # Saves binary torch format
        print(f"Cached walks to {walks_path}")
    except Exception:
        try:
            with open(walks_path, "w") as f:
                json.dump(walks, f)   # Falls back to JSON
        except Exception:
            pass
```

**Problem**: 
- Config says `walks_file: walks.json` (suggests JSON)
- Code saves with `torch.save()` (binary, not JSON)
- Fallback uses JSON, but paths and loading may be inconsistent
- **File content doesn't match filename extension**

### Issue 3: Load vs Save Mismatch

If save fails with torch.save and succeeds with json.dump, but next run tries to load with torch.load, it will fail!

```python
# First run: torch.save fails, json.dump succeeds
# File: walks.json (actually JSON)

# Second run: tries to load
walks = torch.load(walks_path)  # ← FAIL! File is JSON, not torch binary
# Falls back to regenerate walks
```

---

## 4. TOKENIZER BUILDING: REPRODUCIBLE ✅

In `prepare_data.py` lines 123-132:
```python
def get_tokenizer(cfg, walks, edges):
    # ... caching ...
    tokenizer = Tokenizer()
    tokenizer.fit(walks, edges=edges)  # ← Order-independent
    # ... save ...
    return tokenizer
```

✅ **Tokenizer fitting is order-independent** (uses sets, not ordered dicts)  
✅ Vocabulary will be same regardless of walk order  
✅ But depends on upstream (walks order issue #3)

---

## 5. WALK ENCODING: REPRODUCIBLE IF SPLITS FIXED ✅

In `prepare_data.py` lines 135-164:
```python
def encode_walks(walks, tokenizer: Tokenizer, train_set, mask_set, val_set, test_set):
    # Deterministic walk-to-tensor mapping
    # Order depends on walks order (inherited from issue #3)
```

✅ **Given fixed walks and splits, encoding is deterministic**  
❌ **But walks order may vary** (see issue #3)

---

## 6. DATALOADER SHUFFLING: REPRODUCIBLE ✅

In `prepare_data.py` lines 283-320:
```python
base_seed = getattr(cfg.training, "seed", None)
if base_seed is not None:
    g = torch.Generator()
    g.manual_seed(base_seed)

# All dataloaders use generator g:
train_loader = DataLoader(
    train_ds,
    shuffle=True,
    generator=g,  # ✅ Seeded
    worker_init_fn=worker_init_fn,  # ✅ Each worker seeded
)
```

✅ **Shuffling is fully reproducible**  
✅ Worker init seeds each worker independently  
✅ Generator controls shuffle order

---

## 7. GLOBAL SEED IN run.py ✅ (Mostly)

In `run.py` lines 34-43:
```python
seed = getattr(cfg.training, "seed", None)
if seed is not None:
    try:
        seed = int(seed)
        from pytorch_lightning import seed_everything
        import random, numpy as np
        
        seed_everything(seed, workers=True)  # ← PyTorch Lightning
        random.seed(seed)                     # ← Python random
        np.random.seed(seed)                  # ← NumPy
        print(f"Using seed={seed}")
    except Exception:
        pass
```

✅ **Sets seeds for PyTorch, NumPy, Python random**  
✅ `seed_everything()` propagates to DataLoader workers  
❌ **But happens AFTER config load** (too late for split_edges)

---

## CRITICAL ISSUES SUMMARY

### Issue 1: Edge Split Not Seeded ❌ CRITICAL
**File**: `src/data/prepare_data.py:40`
```python
random.shuffle(edges_copy)  # No seed!
```
**Impact**: If cache disabled, different runs produce different train/val/test splits  
**Severity**: CRITICAL - different data → different model behavior  
**Fix**: Add explicit seed to shuffle

### Issue 2: Multiprocessing Walk Order Not Guaranteed ❌ HIGH
**File**: `src/data/walk_sampler.py:85-120`  
**Impact**: Walk order may differ across runs if `num_workers > 1`  
**Severity**: HIGH (currently LOW since default is num_workers=1)  
**Risk if enabled**: Walks reordered → tokenizer vocab may differ → model differs  
**Fix**: Use `pool.starmap_async()` with explicit ordering or re-sort results

### Issue 3: File Format Mismatch ❌ MEDIUM
**File**: `src/data/prepare_data.py:112-119`  
**Impact**: torch.save/load failures cause fallback to JSON inconsistently  
**Severity**: MEDIUM - can cause silent cache invalidation  
**Fix**: Use consistent format (recommend `.pt` for torch.save)

### Issue 4: Load Before Save May Fail ❌ MEDIUM
**File**: `src/data/prepare_data.py:82-88`  
**Impact**: Cached walks loaded with wrong format (torch vs JSON)  
**Severity**: MEDIUM - causes silent regeneration  
**Fix**: Validate file format or use single format

---

## RECOMMENDED FIXES (Ordered by Priority)

### 🔴 PRIORITY 1: Fix Edge Split Seeding

**File**: `src/data/prepare_data.py`

```python
def split_edges(cfg, edges):
    print(f"Splitting edges for {cfg.dataset.name} dataset...")
    split_path = os.path.join(cfg.dataset.data_dir, cfg.dataset.edge_split_file)
    if cfg.preprocess.use_cache and os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
    else:
        edges_copy = list(edges)
        
        # NEW: Seed before shuffle for reproducibility
        seed = getattr(cfg.training, "seed", None)
        if seed is not None:
            random.seed(int(seed))
            print(f"Seeding edge split with seed={seed}")
        
        random.shuffle(edges_copy)
        # ... rest of function unchanged ...
```

**Why**: Ensures same edge splits across runs when cache is disabled

---

### 🟠 PRIORITY 2: Standardize Walk File Format

**File**: `src/data/prepare_data.py`

Option A (Recommended):
```python
def get_walks(cfg, edges):
    print(f"Sampling random walks from {cfg.dataset.name} dataset...")
    walks_path = os.path.join(cfg.dataset.data_dir, cfg.dataset.walks_file)
    
    # Standardize: always use .pt extension
    if not walks_path.endswith('.pt'):
        walks_path = walks_path.replace('.json', '.pt')
    
    if cfg.preprocess.use_cache and os.path.exists(walks_path):
        try:
            walks = torch.load(walks_path)
            print(f"Success! ✅ (loaded cached walks)")
            return walks
        except Exception as e:
            print(f"Warning: Failed to load cached walks: {e}. Regenerating...")
    
    # ... walk sampling ...
    
    if cfg.preprocess.save:
        try:
            torch.save(walks, walks_path)
            print(f"Cached walks to {walks_path}")
        except Exception as e:
            print(f"Warning: Failed to save walks: {e}")
    
    print(f"Success! ✅")
    return walks
```

Then update config:
```yaml
dataset:
  walks_file: walks.pt  # Changed from walks.json
```

**Why**: Eliminates format confusion, ensures consistent load/save

---

### 🟡 PRIORITY 3: Fix Multiprocessing Walk Order (Future Proofing)

**File**: `src/data/walk_sampler.py`

```python
def sample_random_walks(edges, num_walks=100, max_walk_length=16, 
                        num_workers=1, seed=None):
    """
    Fast random-walk sampler with optional multiprocessing.
    
    Note: With multiprocessing, walks are gathered in deterministic order
    to ensure reproducibility across systems/runs.
    """
    nodes, nbrs, lbls = _build_adj(edges)
    if not nodes:
        return []
    
    num_workers = max(1, int(num_workers))
    base_seed = 42 if seed is None else int(seed)
    
    if num_workers == 1:
        return _sample_chunk(
            nodes, nbrs, lbls, num_walks, max_walk_length, 0, num_walks, base_seed
        )
    
    # Split work evenly
    chunk = math.ceil(num_walks / num_workers)
    tasks = []
    start = 0
    for w in range(num_workers):
        end = min(start + chunk, num_walks)
        if start >= end:
            break
        tasks.append((w, start, end, base_seed + w))  # Add task_id
        start = end
    
    # Use ordered collection to preserve task ordering
    with mp.Pool(processes=len(tasks)) as pool:
        results = pool.starmap(
            partial(_sample_chunk, nodes, nbrs, lbls, num_walks, max_walk_length),
            [(start, end, seed) for _, start, end, seed in tasks],
        )
    
    # Flatten results in task order (not completion order)
    walks = []
    for chunk_walks in results:
        walks.extend(chunk_walks)
    
    return walks
```

**Why**: Ensures walk order is deterministic even with multiprocessing enabled

---

## TESTING REPRODUCIBILITY

### Test 1: Cache Disabled, Rerun Twice
```bash
# Run 1
rm -f data/*/walks.pt data/*/splits.json
python run.py dataset.name=wiki-rfa preprocess.use_cache=false training.seed=42

# Run 2 (should produce identical walks file)
python run.py dataset.name=wiki-rfa preprocess.use_cache=false training.seed=42

# Compare
md5sum data/wiki-rfa/walks.pt  # Both runs should have same checksum
```

### Test 2: Verify Walk Determinism
```python
import torch
from src.data.datasets import get_loader
from src.data.walk_sampler import sample_random_walks

cfg = load_config("config.yaml")
edges = get_loader(cfg.dataset.name)(cfg)

# Run 1
walks1 = sample_random_walks(edges, num_walks=1000, seed=42, num_workers=1)

# Run 2
walks2 = sample_random_walks(edges, num_walks=1000, seed=42, num_workers=1)

# Should be identical
assert walks1 == walks2, "Walks differ!"
print("✅ Walks are reproducible")
```

### Test 3: Verify Edge Split Reproducibility
```python
from src.data.prepare_data import split_edges

edges = [...]  # load once
split1 = split_edges(cfg, edges)
split2 = split_edges(cfg, edges)

assert split1 == split2, "Splits differ!"
print("✅ Splits are reproducible")
```

---

## CURRENT STATE: IS YOUR DATA REPRODUCIBLE?

### If using defaults:
- `use_cache=true` → ✅ **YES** (cached artifacts are deterministic)
- `walk_num_workers=1` → ✅ **YES** (single process is deterministic)
- `training.seed=<value>` → ✅ **Mostly YES** (except edge splits if cache disabled)

### If you disable cache:
- `use_cache=false` → ⚠️ **RISKY** (edge splits not seeded)

### If you enable multiprocessing:
- `walk_num_workers>1` → ⚠️ **RISKY** (walk order not guaranteed across systems)

---

## CONFIGURATION RECOMMENDATIONS

### For Reproducible Research:
```yaml
preprocess:
  use_cache: true          # ✅ Ensures cached artifacts reused
  save: true               # ✅ Saves artifacts for consistency
  walk_num_workers: 1      # ✅ Avoids multiprocessing ordering issues
  walk_seed: null          # ✅ Inherits from training.seed
  
training:
  seed: 42                 # ✅ Controls all randomness
```

### For Fast Iteration (accept non-determinism):
```yaml
preprocess:
  use_cache: true
  save: true
  walk_num_workers: 8      # ⚠️ Faster but non-deterministic walk order
  walk_seed: 42
  
training:
  seed: 42
  num_workers: 4           # DataLoader parallelism (seeded)
```

---

## SUMMARY CHECKLIST

- [ ] **Fix 1**: Add seed to edge split shuffle (CRITICAL)
- [ ] **Fix 2**: Standardize walks file format to .pt (MEDIUM)
- [ ] **Fix 3**: Fix multiprocessing walk order (HIGH if using `num_workers>1`)
- [ ] **Test**: Run reproducibility tests from "Testing Reproducibility" section
- [ ] **Document**: Add reproducibility notes to README

---

## APPENDIX: Where Randomness Enters Your Pipeline

```
CONFIG (seed=42)
  ↓
run.py: seed_everything(42), random.seed(42), np.seed(42)
  ↓
prepare_data():
  ├─ get_edge_list() → Deterministic file read ✅
  │
  ├─ split_edges() → random.shuffle() WITHOUT seed ❌ BUG #1
  │   └─ (or loaded from cache if exists)
  │
  ├─ get_walks() → sample_random_walks()
  │   └─ num_workers=1: np.random.default_rng(seed) ✅
  │   └─ num_workers>1: Pool returns out-of-order ❌ BUG #2
  │   └─ (or loaded from cache if exists)
  │
  ├─ get_tokenizer() → Set-based, order-independent ✅
  │
  ├─ encode_walks() → Deterministic given inputs ✅
  │
  ├─ pad_and_build_stage_tensors() → Deterministic ✅
  │
  └─ make_dataloaders()
     └─ DataLoader shuffle with seeded generator ✅
```

