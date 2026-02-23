# DataLoader Bottleneck Fix

## Root Cause Analysis

**The 268.2s "wait time" bottleneck (53% of training)** was caused by poorly configured validation/test dataloaders:

### Problem Breakdown (from profiling)

```
val_wait statistics:
- Median:  0.67ms  (most batches fast)
- P95:     0.89ms  (still fast)
- MAX:    256.19s  (HUGE spike on first validation batch!)
```

This pattern indicates a **cold-start dataloader problem**:

1. Training completes for one epoch
2. Validation phase starts
3. Validation dataloader workers (8 workers, half of training) spawn fresh
4. Workers build prefetch buffer from scratch (only fetching 1 batch ahead, half of training)
5. First validation batch experiences 256-second stall waiting for data to arrive
6. Subsequent batches run fast (workers already primed)

### Bad Configuration (BEFORE)

`src/data/prepare_data.py` lines 481-490:

```python
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=max(0, int(num_workers // 2)),        # ❌ 8 workers (half)
    pin_memory=pin_memory,
    persistent_workers=(persistent and num_workers > 0),
    prefetch_factor=max(1, prefetch // 2),            # ❌ 1 batch (half)
    ...
)
```

**Plus:** Default `persistent_workers=False` in exception handler (line 444), so workers were terminated between validation phases even though config requested True.

---

## Fixes Applied

### Fix 1: Correct persistent_workers default (line 444)

```python
# BEFORE:
persistent = bool(getattr(cfg.training, "persistent_workers", False))  # ❌ Wrong default

# AFTER:
persistent = bool(getattr(cfg.training, "persistent_workers", True))   # ✅ Match config
```

This ensures validation dataloader workers **stay alive** between validation phases, eliminating cold-start on every validation epoch.

### Fix 2: Use full worker count + prefetch for val/test (lines 481-490, 493-502)

```python
# BEFORE:
val_loader = DataLoader(
    val_ds,
    num_workers=max(0, int(num_workers // 2)),        # ❌ 8 workers
    prefetch_factor=max(1, prefetch // 2),            # ❌ 1 batch
    ...
)

# AFTER:
val_loader = DataLoader(
    val_ds,
    num_workers=num_workers,                          # ✅ 16 workers (same as train)
    prefetch_factor=prefetch,                         # ✅ 2 batches (same as train)
    ...
)
```

Same fix applied to `test_loader`.

**Why this works:**

- Validation is embarrassingly parallel (no dependencies between validation batches)
- Using more workers = more prefetch buffer = data ready before GPU finishes previous batch
- With persistent workers, cost of recreation is eliminated

---

## Expected Improvement

### Current (Broken) Performance

- **val_wait total:** 259.58s per epoch
- **train_model_total:** 506.02s (validation dominates)
- **Per-epoch overhead:** ~376s training + ~120s validation = **~496s per epoch**

### After Fix (Expected)

With persistent workers + full prefetch:

- Warm-up epoch 1: Similar (needs to build buffer)
- Epochs 2-25: `val_wait` drops from **259.58s → ~15-30s** (amortized across 4885 batches at 0.0067s/batch)
- **Savings per epoch (2-25):** ~230-245s
- **Multi-epoch speedup:** 2-5x faster validation after first epoch

### Per-Epoch Time (25-epoch training)

- **Epoch 1:** ~500s (first validation needs warm-up)
- **Epochs 2-25:** ~260-280s each (persistent workers + full prefetch)
- **Total savings:** ~5800s (96 minutes) on 25-epoch run

---

## Test This Fix

Run the profiler again with the updated code:

```bash
python profile_run_pipeline.py dataset.name=epinions training.epochs=3
```

Compare `val_wait` statistics—should see:

- **val_wait median/p95:** Still ~0.67ms-0.89ms (same, fast path works)
- **val_wait max:** Drops from **256s → 5-10s** (much shorter warm-up)
- **val_wait sum total:** Drops from **259s → 50-90s** for 3 epochs

---

## Configuration Details

Your config.yaml already has optimal settings:

```yaml
training:
    num_workers: 16           # ✅ Plenty of workers
    pin_memory: true          # ✅ GPU transfer is fast
    persistent_workers: true  # ✅ Keep workers alive (NOW correctly honored)
    prefetch_factor: 2        # ✅ Good buffer size
```

The bottleneck was in the code not respecting these settings for validation/test dataloaders.
