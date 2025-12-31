# Data Pipeline Optimization Analysis

## 1. Data File Names & Save Methods

### Current State
- Walks are saved with `torch.save()` → binary format (good!)
- Tokenizer is saved with custom `.save()` method → likely JSON
- Encoded tensors saved with `torch.save()` → binary format (good!)
- Fallback to JSON for walks if torch.save fails

### Analysis
✅ **torch.save() for walks is correct** - This is the best approach:
- Binary format is ~2-3x smaller than JSON
- Much faster to load/save 
- Preserves data types (int64, etc.)

❓ **What about tokenizer?** Check `src/data/tokenizer.py`:
- If it's saving as JSON, consider whether it's needed at runtime
- If used during inference, JSON is fine for small metadata
- Could also be saved as `.pkl` (pickle) if there's complex structure

✅ **Overall: Current approach is good.** Consider only:
- Verify tokenizer format (should be fine either way)
- No changes needed to walks/tensors

---

## 2. Walk Generation: NumPy vs NetworkX vs GPU

### Current Implementation Analysis

**What we have:**
```python
# walk_sampler.py uses:
- defaultdict for adjacency lists (good for sparse graphs)
- NumPy arrays (nbrs, lbls) for vectorized neighbor/label storage
- np.random.default_rng() for fast sampling
- Multiprocessing for parallel chunks
```

### NumPy vs Alternatives

| Aspect | NumPy | NetworkX | GPU |
|--------|-------|----------|-----|
| Speed (random walk) | ⭐⭐⭐⭐⭐ Very fast | ⭐⭐ Slow (lots of Python overhead) | ⭐⭐⭐⭐ Fast but overkill |
| Memory | Efficient | High (networkx objects) | High (GPU memory) |
| Scalability | Excellent to 100M nodes | Poor (Python overhead) | Good but limited by GPU mem |
| Code clarity | Good | Excellent but unnecessary here | Complex |
| CPU util | Good (vectorized) | Poor (Python loops) | N/A |

**Verdict: NumPy is the best choice for your use case.** ✅

**Why NOT NetworkX:**
- Built for graph algorithms, not sampling efficiency
- Massive overhead (object creation per node/edge)
- Pure Python loops = slow
- You're already doing better with NumPy

**Why NOT GPU:**
- Walk generation is memory-bound, not compute-bound
- GPU transfer costs would dwarf speedup
- NumPy on CPU is already near-optimal
- GPU better used for model training

---

## 3. NumPy Chunks + Multiprocessing vs GPU

### Current Implementation
```python
# Splits num_walks across workers, each gets chunk with seeded RNG
chunk = math.ceil(num_walks / num_workers)
# Each worker: _sample_chunk() -> generates independent walks with seed + worker_id
```

### Analysis: Is multiprocessing the best?

| Approach | Speed | Memory | Scalability | Notes |
|----------|-------|--------|-------------|-------|
| **Single process NumPy** | Baseline | Low | OK for <50M walks | Simple, no IPC overhead |
| **Multiprocessing (current)** | 1.5-2.5x | Higher | Good to 8-16 workers | Works well, diminishing returns |
| **Numba JIT** | ⭐⭐⭐⭐ 10-50x | Low | Excellent | **Worth trying** |
| **PyPy** | 2-5x | Similar | Fair | Not typical for ML |
| **GPU (CuPy)** | 5-10x | GPU limited | Moderate | Overkill, not worth it |

### ⭐ Recommendation: Consider Numba

**Numba JIT-compiled version could be 10-50x faster:**

```python
@numba.njit
def _sample_chunk_numba(nodes, nbrs, lbls, num_walks, max_walk_length, seed):
    rng = np.random.default_rng(seed)
    walks = []
    for _ in range(num_walks):
        start = nodes[rng.integers(0, len(nodes))]
        walk_tokens = [f"N_{start}"]
        curr = start
        for _ in range(max_walk_length):
            neigh = nbrs[curr]
            if len(neigh) == 0:
                break
            idx = rng.integers(0, len(neigh))
            walk_tokens.append(f"E_{lbls[curr][idx]}")
            walk_tokens.append(f"N_{neigh[idx]}")
            curr = neigh[idx]
        walks.append(walk_tokens)
    return walks
```

**However, Numba limitations:**
- Can't use dicts directly (need structured arrays)
- String formatting might not work
- Would need refactoring

**Practical recommendation:** ✅ **Current multiprocessing is good enough.** Leave it unless walks generation is a bottleneck. Profile first!

---

## 4. DataLoader Configuration Deep Dive

### Current `make_dataloaders()` Configuration

```python
num_workers=4              # CPU threads for data loading
pin_memory=True           # Pin batch to GPU memory
persistent_workers=False  # Keep workers alive between epochs
prefetch_factor=2         # Batches queued ahead
worker_init_fn=...        # Seed each worker
generator=g               # Deterministic shuffle
```

### Understanding Each Parameter

#### `num_workers` (default: 4)
**What it does:** Spawns N worker processes that load data in parallel
- Each worker loads a batch while main process trains
- Main process doesn't block waiting for I/O

**Is it beneficial or overhead?**
- ✅ **Beneficial** when:
  - Data loading is slow (large datasets, complex transforms)
  - You have CPU cores available
  - Dataset >> GPU batch size
- ❌ **Overhead** when:
  - Data already in RAM
  - Complex pickling cost > loading cost
  - Small batch sizes

**Your case:** ✅ **Useful** - tensors must be loaded from disk/memory, especially on first epoch

**Recommended:** Keep at 4-8 depending on CPU count. Monitor GPU utilization.

---

#### `pin_memory` (default: True)
**What it does:** Pre-allocates pinned (non-pageable) RAM for faster CPU→GPU transfer

**Overhead or beneficial?**
- ✅ **Beneficial** when:
  - Using CUDA/GPU training
  - Batch transfers are bottleneck
  - You have extra RAM
- ❌ **Overhead** when:
  - Data already on GPU
  - Very limited RAM (pins can reduce available memory)

**Your case:** ✅ **Beneficial** - you're using L40S GPU, this helps

**Memory cost:** ~2x batch size in pinned RAM (not huge)

---

#### `persistent_workers` (default: False in your code)
**What it does:** Keeps worker processes alive between epochs instead of killing/recreating them

**Overhead or beneficial?**
- ✅ **Beneficial** when:
  - Many epochs (overhead amortized)
  - Workers have initialization cost (loading models, etc.)
  - num_workers > 0
- ❌ **Overhead** when:
  - Few epochs
  - Workers just load simple data
  - RAM is tight (zombie processes hold memory)

**Your case:** ⚠️ **Marginal benefit.** 
- Set to `True` if running many epochs (e.g., >10)
- Set to `False` if few epochs or tight on RAM

**Recommendation:** Try `True` for long training runs.

---

#### `prefetch_factor` (default: 2 in your code)
**What it does:** Number of batches queued in advance (worker queue size)

**Overhead or beneficial?**
- ✅ **Beneficial** when:
  - Batch loading varies in time
  - You want smooth training (no GPU idle)
- ❌ **Overhead** when:
  - Too high → excess memory usage
  - Batches are large

**Guidelines:**
- `prefetch_factor=2`: Conservative (low memory overhead)
- `prefetch_factor=4-8`: Typical (better GPU utilization)
- `prefetch_factor>8`: Risky (memory bloat)

**Your case:** ✅ **2 is fine.** Consider increasing to 4 if you see GPU stalling.

---

### ⚠️ Seed Management Problem

Your code has **confusing and partly unnecessary seeding**:

```python
# In run.py - Global seeding
seed_everything(seed, workers=True)  # PyTorch Lightning
random.seed(seed)
np.random.seed(seed)

# In prepare_data.py - DataLoader seeding
def worker_init_fn(worker_id):
    seed = base_seed + worker_id if base_seed else int(time.time()) + worker_id
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))  # WHY THIS MODULO?
    torch.manual_seed(seed)

# In walk_sampler.py - Walk generation seeding
base_seed = 42 if seed is None else int(seed)
seed_per_worker = base_seed + w
```

**Problems:**
1. `seed % (2**32 - 1)` is **vestigial** - NumPy's default_rng handles large seeds fine
2. `int(time.time())` as fallback = non-deterministic! Defeats reproducibility
3. Multiple seeding layers doing similar things (confusing)
4. `seed_everything(workers=True)` should handle this, so DataLoader seeding is redundant

---

### ✅ Simplified Seed Strategy

**Replace everything with:**

```python
# run.py - single place to seed
if seed is not None:
    import random
    import numpy as np
    import torch
    from pytorch_lightning import seed_everything
    
    seed = int(seed)
    seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Using seed={seed}")

# prepare_data.py - NO worker_init_fn needed (seed_everything handles it)
# OR minimal version:
def worker_init_fn(worker_id):
    if base_seed is not None:
        seed = base_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    # If base_seed is None, leave workers unseeded (training-only, reproducibility not needed)
```

**Why this works:**
- `seed_everything(workers=True)` from PyTorch Lightning already sets global seeds
- Worker seeding is only needed to ensure each worker generates different data
- Time-based seeding should NEVER be in worker_init (defeats reproducibility)
- Modulo arithmetic is unnecessary for modern NumPy

---

## 5. PyTorch Float32 Matmul Precision

### Current Setting
```python
precision = getattr(cfg, "float32_precision", "medium")
torch.set_float32_matmul_precision(precision)
```

### What does this do?

When you do `matrix @ matrix` operations on NVIDIA Tensor Cores (your L40S has them), PyTorch can trade precision for speed:

| Setting | Precision | Speed | Use Case |
|---------|-----------|-------|----------|
| `"highest"` | float32 (no reduction) | ⭐ Baseline (slowest) | Numerically sensitive algorithms |
| `"high"` | float32 with some speedup | ⭐⭐⭐ 1-5% faster | Most ML (recommended default) |
| `"medium"` | float16 intermediate ops | ⭐⭐⭐⭐ 10-50% faster | **Most common choice** |

### What happens with `"medium"` (your current setting)?

1. Breaks matmul into smaller operations
2. Uses float16 for intermediate results
3. Accumulates back to float32
4. Loss of precision is typically **~1e-6 relative error** = negligible

### ⚠️ Is it risky?

| Scenario | Risk | Notes |
|----------|------|-------|
| Normal ML training | ✅ None | 1e-6 error doesn't affect convergence |
| Classification | ✅ None | Already has classification error >> 1e-6 |
| Regression (MSE) | ⚠️ Low | Error is small but accumulates |
| Numerical algorithms | ❌ High | SVD, matrix inverse, etc. - don't use |

### Recommendation for Your Setup

**Current `"medium"` is good.** Here's the reasoning:

- You're training a graph neural network for edge classification
- Classification task is robust to numerical noise
- 10-50% speedup is significant on L40S
- The precision loss is negligible for your application

**If you want to be conservative:** Use `"high"` (1-5% slower but safer)

**DO NOT use `"highest"`** unless you have numerical instability issues

---

## Summary of Recommendations

| Item | Current | Assessment | Action |
|------|---------|------------|--------|
| **Walks saved with torch.save()** | ✅ | Optimal | Keep as is |
| **NumPy walk generation** | ✅ | Optimal for CPU | Keep as is |
| **Multiprocessing for walks** | ✅ | Good enough | Profile first if bottleneck |
| **pin_memory=True** | ✅ | Beneficial for GPU | Keep |
| **persistent_workers=False** | ⚠️ | Marginal overhead | Set to `True` for long runs |
| **prefetch_factor=2** | ✅ | Conservative | Consider increasing to 4 if GPU stalling |
| **Seed management** | ❌ | Overcomplicated, confusing | Simplify (remove time.time() fallback) |
| **seed % (2**32-1)** | ❌ | Unnecessary | Remove |
| **float32_precision="medium"** | ✅ | Good choice | Keep (10-50% speedup with negligible precision loss) |

---

## Quick Wins (Easy Changes)

### 1. Remove confusing seed modulo in DataLoader:
```python
# BEFORE
np.random.seed(seed % (2**32 - 1))

# AFTER
np.random.seed(seed)
```

### 2. Set persistent_workers=True for long training:
```yaml
# In config
training:
  persistent_workers: true  # If running many epochs
```

### 3. Increase prefetch_factor to 4 (if GPU under-utilized):
```yaml
training:
  prefetch_factor: 4
```

### 4. Remove fallback to time-based seed (forces explicit seeding):
```python
# BEFORE
seed = int(time.time()) + worker_id if base_seed is None else base_seed + worker_id

# AFTER  
seed = base_seed + worker_id if base_seed is not None else None
if seed is not None:
    random.seed(seed)
    # etc.
```

---

## If You Want to Optimize Further

### Profiling First
1. Check if walk generation is actually a bottleneck:
   ```bash
   python -m cProfile -s cumtime run.py --config config.yaml 2>&1 | head -30
   ```
   
2. Check GPU utilization during training:
   ```bash
   nvidia-smi dmon -s pcm  # GPU memory/util over time
   ```

3. If walks generation takes >1 minute, then consider Numba

### Actual Optimizations (if needed)
1. **Move walks→GPU:** Pre-compute walks on GPU (if fits in memory)
2. **Numba JIT walks:** Requires refactoring but could be 10-50x faster
3. **Cached walks in memory:** Load all walks into RAM if <100GB
4. **Distributed data loading:** Multiple machines if >500M walks

