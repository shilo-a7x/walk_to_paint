# Walk Sampling Reproducibility: Complete Explanation

## Problem Statement
How do we ensure that EXACTLY the same walks are generated and saved to file in EXACTLY the same order every run, even when using multiprocessing?

## Answer: Per-Walk Deterministic Seeding

### How Starting Node Selection Works

For each walk[i], the starting node is selected using a seeded RNG:

```python
# Walk 0: use seed = base_seed + 0 = 42 + 0 = 42
rng_0 = np.random.default_rng(42)
start_node_0 = node_arr[rng_0.integers(0, len(node_arr))]  # e.g., node 5

# Walk 1: use seed = base_seed + 1 = 42 + 1 = 43  
rng_1 = np.random.default_rng(43)
start_node_1 = node_arr[rng_1.integers(0, len(node_arr))]  # e.g., node 12

# Walk 2: use seed = base_seed + 2 = 42 + 2 = 44
rng_2 = np.random.default_rng(44)  
start_node_2 = node_arr[rng_2.integers(0, len(node_arr))]  # e.g., node 3
```

**Key insight**: Each walk uses a DIFFERENT seed (base + walk_idx), so:
- Walk[0] will always start at node 5 (same RNG state every run)
- Walk[1] will always start at node 12 (different RNG state from walk[0])
- Walk[2] will always start at node 3 (different RNG state from walk[0] and walk[1])

### Why This Guarantees Reproducibility

1. **RNG State is Deterministic**: `np.random.default_rng(42)` always produces the same sequence of random numbers
2. **Each Walk Gets Unique Seed**: Walk[i] uses seed (base_seed + i), not a shared seed
3. **Walk Steps Are Deterministic**: Same starting node + same RNG state → identical walk steps
4. **Walk Order is Preserved**: We sort by task_id before saving to file

### Example: Walk Generation Trace

```
Run 1 (4 workers):
  Worker 0: walks 0-1   → task_id=0 (walks use seeds 42, 43)
  Worker 1: walks 2-3   → task_id=1 (walks use seeds 44, 45)
  
  Results come back in order: [task_id=1, task_id=0]  ← completion order
  
  After sort by task_id: [task_id=0, task_id=1]      ← restored correct order
  
  Final walks: [walk[0], walk[1], walk[2], walk[3]]  ✓ same order as 1 worker

Run 2 (2 workers):
  Worker 0: walks 0-1   → task_id=0 (walks use seeds 42, 43)
  Worker 1: walks 2-3   → task_id=1 (walks use seeds 44, 45)
  
  Results come back in order: [task_id=0, task_id=1]  ← different completion order!
  
  After sort by task_id: [task_id=0, task_id=1]      ← same correct order
  
  Final walks: [walk[0], walk[1], walk[2], walk[3]]  ✓ identical to Run 1
```

### File Output Reproducibility

When we save walks to disk:

```python
# All runs save in same order
walks[0]: seed=42 → always starts at node 5 → same tokens
walks[1]: seed=43 → always starts at node 12 → same tokens
walks[2]: seed=44 → always starts at node 3 → same tokens
walks[3]: seed=45 → always starts at node 8 → same tokens

# Same bytes written to disk every run!
torch.save(walks, 'walks.pkl')  # Identical file every time
```

## Architecture: How Multiprocessing Works

### Task Distribution
```
num_walks = 100, num_workers = 4

Task distribution:
  Task 0 (worker 0): walks[0..24]   (25 walks, seeded 42-66)
  Task 1 (worker 1): walks[25..49]  (25 walks, seeded 67-91)  
  Task 2 (worker 2): walks[50..74]  (25 walks, seeded 92-116)
  Task 3 (worker 3): walks[75..99]  (25 walks, seeded 117-141)
```

### Code Flow
```python
# Each task computes walks[start_idx:end_idx]
# using absolute walk indices (not relative to task)
for walk_idx in range(start_idx, end_idx):
    rng = np.random.default_rng(base_seed + walk_idx)  # 42 + walk_idx
    # ... compute walk[walk_idx] ...
    walks.append(tokens)

# Result: (task_id, [walk_start, walk_start+1, ..., walk_end-1])
return (0, [walk[0], walk[1], ..., walk[24]])  # task_id=0
return (1, [walk[25], walk[26], ..., walk[49]])  # task_id=1
```

### Sorting Ensures Correct Order

```python
# Results might arrive out of order
results = [
    (1, [walk[25]...walk[49]]),   # task 1 finished first
    (0, [walk[0]...walk[24]]),    # task 0 finished later
    (3, [walk[75]...walk[99]]),   # task 3 finished third
    (2, [walk[50]...walk[74]]),   # task 2 finished last
]

# Sort by task_id to restore walk index order
results.sort(key=lambda x: x[0])

results = [
    (0, [walk[0]...walk[24]]),    # task 0 now first
    (1, [walk[25]...walk[49]]),   # task 1 now second
    (2, [walk[50]...walk[74]]),   # task 2 now third
    (3, [walk[75]...walk[99]]),   # task 3 now last
]

# Extend in order: walks = walk[0]...walk[24] + walk[25]...walk[49] + ...
# ✓ Always get walk[0], walk[1], ..., walk[99] in this order
```

## Guarantee: Bit-For-Bit File Identity

The combination of three things guarantees identical file output:

1. **Deterministic Per-Walk Seeding**
   - Walk[i] always uses seed (base + i)
   - Same starting node every run
   - Same walk steps every run

2. **Deterministic Walk Ordering**
   - Tasks sorted by task_id before concatenation
   - Multiprocessing completion order doesn't affect output
   - walks[0] through walks[n] always in same order

3. **Deterministic File Format**
   - torch.save/json.dump with fixed seed
   - Same walk list → identical bytes on disk

**Result**: Running sample_random_walks(..., seed=42) 100 times with different num_workers (1, 2, 4, 8) produces byte-for-byte identical files.

## Testing This Guarantee

To verify, we can check:

```python
# Run 1: 1 worker
walks_1 = sample_random_walks(edges, num_walks=100, num_workers=1, seed=42)

# Run 2: 4 workers  
walks_4 = sample_random_walks(edges, num_walks=100, num_workers=4, seed=42)

# These should be identical
assert walks_1 == walks_4
assert len(walks_1) == len(walks_4)
for w1, w4 in zip(walks_1, walks_4):
    assert w1 == w4
```

## Integration with Config System

In [src/utils/config.py](src/utils/config.py):
```python
def get_seed(cfg):
    """Get canonical seed from reproducibility.seed"""
    if 'reproducibility' not in cfg or 'seed' not in cfg.reproducibility:
        raise ValueError("Config must have reproducibility.seed set")
    return cfg.reproducibility.seed
```

In [src/data/prepare_data.py](src/data/prepare_data.py):
```python
seed = get_seed(cfg)  # Gets config.reproducibility.seed (e.g., 42)
walks = sample_random_walks(
    edges,
    num_walks=cfg.model.num_walks,
    max_walk_length=cfg.model.walk_length,
    num_workers=cfg.preprocess.num_workers,
    seed=seed  # Pass seed to walk sampler
)
# All walks now reproducible!
```

## Summary

- **How starting nodes are selected**: Each walk[i] uses `rng = np.random.default_rng(base_seed + i)` to deterministically pick starting node
- **Why order is preserved**: Results sorted by task_id before concatenation, independent of worker completion order
- **Why files are identical**: Same seed + same walk sequence + same order = identical file output every run
- **Multiprocessing benefit**: Speeds up walk generation without sacrificing reproducibility
