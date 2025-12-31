# Data Pipeline Audit & Optimization Analysis

## 1. File Naming & Saving Issues

### Current Issues
- **Line 101** (`get_walks`): Saves walks as `.json` file using `cfg.dataset.walks_file` but code does `torch.save(walks, walks_path)` which saves binary PyTorch format, NOT JSON
- **Issue**: File is named `.json` but contains binary torch pickle; can cause confusion and potential cross-platform issues

### Recommended Fix
```python
# Option A: Use .pt extension (preferred - clearer intent)
walks_file: walks.pt  # instead of walks.json

# Option B: Update the save logic to respect file extension
if walks_path.endswith('.json'):
    # Use JSON
else:
    # Use torch.save for .pt
```

### Other Artifacts
- `tokenizer.json` - ✅ Actually JSON (verified in tokenizer.py)
- `encoded.pt` - ✅ Correctly torch format
- `meta.json` - ✅ Correctly JSON
- `splits.json` - ✅ Correctly JSON

**Action**: Update config files to use `.pt` for walks

---

## 2. Walk Generation Optimization

### Current Method
- Uses **NumPy random generation** with multiprocessing chunks
- For each walk: random start node → random neighbor selection → repeat

### Analysis
```
Current (NumPy-based):
✅ Fast for 100K-1M walks (vectorized operations)
✅ Multiprocessing available
✅ Reproducible with seed control
❌ CPU-bound only
❌ No GPU acceleration possible (graph structure not GPU-friendly)
```

### Why Not GPU?
- Random walk sampling requires:
  - Irregular graph structure (varied node degrees)
  - Random memory access patterns
  - Sequential path generation
- GPUs excel at batch operations on regular structures, not sparse graph traversal
- **Conclusion**: CPU + NumPy is correct choice

### Multiprocessing Assessment
Current approach:
```python
num_workers = int(getattr(cfg.preprocess, "walk_num_workers", 1))
# Default = 1 (no parallelization)
```

**Optimization**: Default to `num_workers=8` or `num_cpus//2` to use all cores
- For 434K walks on 16-core system: 8 workers would parallelize 50K walks/worker
- Speed improvement: ~4-8x (depends on contention)

---

## 3. DataLoader Configuration Deep Dive

### `pin_memory`
```python
pin_memory = bool(getattr(cfg.training, "pin_memory", True))
```
**What**: Locks tensor memory in RAM (prevents paging to disk)
**When beneficial**: 
- ✅ GPU training (faster CPU→GPU transfer)
- ✅ Large datasets > available RAM
- ❌ Small datasets, CPU-only training

**Default True**: ✅ Good for your GPU setup

---

### `persistent_workers`
```python
persistent_workers = (persistent and num_workers > 0)
```
**What**: Keeps worker processes alive between batches (vs spawning/destroying each time)
**Cost**: More memory (workers hold state)
**Benefit**: Reduced spawn overhead, faster batching
**Recommendation**: 
- ✅ Enable if `num_workers > 4` and dataset > 100MB
- ❌ Disable if `num_workers == 1` (single worker, no parallelism benefit)

**Current**: Default False - reasonable for single worker

---

### `prefetch_factor`
```python
prefetch_factor=prefetch  # default 2
```
**What**: Number of batches to prefetch ahead (worker prefetches while GPU trains on current batch)
**Higher = More memory**, but better GPU utilization (less waiting)
**Recommendation**: 2-4 for typical setups
- Prefetch 1 batch ahead while you process current
- Current setting: 2 ✅ Good default

---

### Seeding Strategy

#### The Weird Seed Code (lines 309-321)
```python
def worker_init_fn(worker_id):
    seed = None
    if base_seed is not None:
        seed = base_seed + worker_id  # ← Each worker gets unique seed
    else:
        seed = int(time.time()) + worker_id  # ← Fallback to time
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))  # ← NumPy seed must be < 2^32
    torch.manual_seed(seed)
```

**Why weird?**
1. **`np.random.seed(seed % (2**32 - 1))`**: NumPy legacy RNG only accepts 32-bit values
   - Your seed might be > 2^32, so modulo wraps it
   - **Better approach**: Use `np.random.default_rng(seed)` (newer, handles arbitrary seeds)

2. **Why per-worker seeding?** 
   - Without it: All workers use identical seed → same batches → defeats parallelization!
   - With it: Each worker gets unique data
   - **Necessary for reproducibility**

#### Better Implementation
```python
def worker_init_fn(worker_id):
    seed = base_seed + worker_id if base_seed is not None else int(time.time() * 1e9) + worker_id
    random.seed(seed)
    np.random.seed(seed)  # new default_rng handles this
    torch.manual_seed(seed % (2**32 - 1))  # torch needs <2^32
```

**Assessment**: Current logic is correct but could be cleaner. Not necessary to change (works fine).

---

## 4. Torch Float32 Matmul Precision

### Current (run.py line ~70)
```python
if cfg.training.use_cuda and torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")
```

### Options
| Mode | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| `"highest"` | Slowest | Max precision (full float32) | Numerical stability required |
| `"high"` | Medium | Good (mixed float32/16) | Default, balanced |
| `"medium"` | Fastest | Acceptable (more aggressive mixed) | Speed-critical |

### Recommendation
- **Current "medium"**: ✅ Good for Transformer training (can tolerate ~1e-4 error)
- **Add config control**: Let users override if needed

### Implementation
```yaml
# config.yaml
training:
  float32_precision: "medium"  # "highest" | "high" | "medium"
```

---

## Summary of Needed Changes

### High Priority (Performance)
1. ✅ Default `walk_num_workers` to 8 (currently 1)
2. ✅ Use `.pt` extension for walks (rename from `.json`)

### Low Priority (Cleanliness)
3. Update NumPy seeding for clarity (not critical, works now)
4. Add `float32_precision` config control

### Already Optimized
- DataLoader config reasonable
- Multiprocessing architecture sound
- CPU-only walk generation is correct
- Caching strategy effective

---

## Retrain Command Template

```bash
# Device 0: wiki-rfa (trial #87 params)
nohup python run.py \
  --device 0 \
  dataset.name=wiki-rfa \
  preprocess.save=true \
  preprocess.use_cache=true \
  preprocess.walk_num_workers=8 \
  > logs/retrain_wiki.log 2>&1 &

# Device 1: epinions (trial #31 params, but use best available)
nohup python run.py \
  --device 1 \
  dataset.name=epinions \
  preprocess.save=true \
  preprocess.use_cache=true \
  preprocess.walk_num_workers=8 \
  > logs/retrain_epinions.log 2>&1 &

# Device 2: slashdot090221 (trial #0 params)
nohup python run.py \
  --device 2 \
  dataset.name=slashdot090221 \
  preprocess.save=true \
  preprocess.use_cache=true \
  preprocess.walk_num_workers=8 \
  > logs/retrain_slashdot.log 2>&1 &
```
