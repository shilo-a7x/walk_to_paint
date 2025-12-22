# Walk-to-Paint: Optimization & Evaluation Guide

## Status Overview

You have successfully run Optuna studies on 3 datasets:

| Dataset | Trials | Best Score | Best Trial | Study Path |
|---------|--------|-----------|-----------|-----------|
| **Wiki-RFA** | 200 | 0.7779 | #87 | `wiki-rfa-optuna_20251203-235202` |
| **Epinions** | 63 | 0.9133 | #31 | `epinions-optuna_20251203-235202` |
| **Slashdot** | 21 | 0.8529 | #12 | `slashdot090221-optuna_20251203-235202` |

All studies are **resumable** with `load_if_exists=True`.

---

## 1. Resume Optuna Studies

To continue running more trials on any dataset:

```bash
# Add 100 more trials to wiki-rfa
python optuna_run.py --config=config.yaml --n-trials=100 dataset.name=wiki-rfa

# Add 50 more trials to epinions
python optuna_run.py --config=config.yaml --n-trials=50 dataset.name=epinions

# Add 50 more trials to slashdot
python optuna_run.py --config=config.yaml --n-trials=50 dataset.name=slashdot090221
```

The code automatically loads your existing study (`load_if_exists=True`) and continues optimization.

---

## 2. Load Trained Models with Optuna Hyperparams

### Quick Usage:

```bash
# Get best model hyperparams for wiki-rfa
python scripts/load_best_model.py wiki-rfa

# Load the model from checkpoint
python scripts/load_best_model.py wiki-rfa --load-model

# Get hyperparams + resume study
python scripts/load_best_model.py epinions --resume
```

### In Python Code:

```python
from scripts.load_best_model import OptunaBestModel

# Load best model
model = OptunaBestModel("wiki-rfa")
model.find_best_study()
hyperparams = model.get_hyperparms()

print(f"Best hyperparams: {hyperparams}")

# Load the trained model from checkpoint
trained_model = model.load_model(device=2)

# Prepare data with caching
train_loader, val_loader, test_loader = model.prepare_data(use_cache=True)
```

---

## 3. Extract Top Alternative Trials

Find high-scoring trials with **fewer walks** or **shorter walk length** — useful for production where speed matters:

```bash
# Find 10 trials within 3% of best score but with FEWEST WALKS
python scripts/extract_top_trials.py wiki-rfa --metric num_walks --tolerance 3

# Find trials with SHORTEST WALKS (while keeping score high)
python scripts/extract_top_trials.py epinions --metric max_walk_length --tolerance 5
```

### Output Example:
```
Trial #87 (best): score=0.7779, num_walks=434,857, max_walk_length=82
  → Can use this if you need the best score

Trial #102: score=0.7720, num_walks=639,012, max_walk_length=81
  → Score is 0.62% lower but uses 47% more walks (trade-off)

Trial #35: score=0.7719, num_walks=662,661, max_walk_length=75
  → Score is 0.77% lower but walks are 8.5% shorter
```

**Use Case:** If you need fast inference, Trial #35 might be better even though the score is slightly lower.

---

## 4. Dataset Caching to Avoid Rebuilds

Caching is now **enabled by default** and writes all intermediates on the first run.

### Caching settings (config.yaml)
```yaml
preprocess:
  use_cache: true    # read cached artifacts when present
  save: true         # write artifacts on first run
  walk_num_workers: 8  # parallel walk sampler workers
  walk_seed: 42        # walk sampler seed (inherits training.seed if absent)
```

### What Gets Cached (binary for speed)
- `{data_dir}/walks.pt` (or `walks_file`) — sampled walks (torch.save)
- `{data_dir}/{dataset_name}_encoded.pt` — encoded walks and labels
- `{data_dir}/{dataset_name}_meta.json` — tokenizer vocab, class info

### First vs. subsequent runs
```bash
# First run: builds and saves cache
python scripts/evaluation_pipeline.py wiki-rfa --device 2

# Subsequent runs: loads from cache (~seconds)
python scripts/evaluation_pipeline.py wiki-rfa --device 2
```

**Huge speedup:** From 300+ seconds to <10 seconds for large datasets.

---

## 5. Fast Evaluation Pipeline

Complete pipeline for evaluation with models and cached datasets:

```bash
# Evaluate best model on test set (uses cache)
python scripts/evaluation_pipeline.py wiki-rfa

python scripts/evaluation_pipeline.py epinions --device 3

python scripts/evaluation_pipeline.py slashdot090221 --device 0
```

---

## 6. Data Building Optimization

### Current Bottleneck:

From profiling (Slashdot, 500k walks, length 80):
- **get_walks (random walk sampling): ~306s** ← BOTTLENECK
- encode_walks: ~13s
- pad_and_build: ~3s

The **random walk sampling** dominates. Options to speed it up:

### Option A: Reduce `num_walks` (easiest, currently done by Optuna)

The best wiki-rfa trial used **434k walks** instead of 500k — only 0.62% score drop.

Slashdot's best trial used **~1.5-2M walks** — you could try 1M and see if score holds.

### Option B: Optimize walk sampling algorithm

Current: `sample_random_walks()` in `src/data/walk_sampler.py` likely uses Python loops.

**Potential improvements:**
1. **Vectorize walk sampling** — use NumPy batch operations
2. **Use Numba/Cython** — JIT compile the hot loop
3. **Parallelize** — split walks across CPU cores using multiprocessing

### Option C: Use a faster backend

- **GPU-accelerated graph sampling** (PyG's neighbor sampler)
- **DuckDB or Polars** instead of pure Python for edge manipulation

---

## 7. DataLoader Optimization

Current config (from your setup):
```yaml
training:
  num_workers: 16
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
  batch_size: [32, 64, 128, 256, 512]  # Optuna samples these
```

### Analysis for Your Case:

**Good choices:**
- ✅ `persistent_workers: true` — keeps workers alive across epochs (no fork overhead)
- ✅ `pin_memory: true` — locks data in GPU-accessible CPU RAM
- ✅ `num_workers: 16` — reasonable for 128-core machine

**Potential improvements:**

1. **Increase `num_workers`** to 24-32 (if CPU isn't maxed out)
   ```yaml
   training:
     num_workers: 32  # Try this
   ```

2. **Tune `prefetch_factor`** (currently 2, try 4-8)
   ```yaml
   training:
     prefetch_factor: 4  # More prefetching = less GPU stalls
   ```

3. **Larger batches** during training (reduces DataLoader overhead)
   - Current: max batch_size=512
   - Try: 1024 or 2048 if GPU memory allows

### How to Test:

Add to `optuna_run.py`:
```python
# In OPTUNA_RANGES
"training.num_workers": [16, 24, 32],
"training.prefetch_factor": [2, 4, 8],
```

Then run a small HPO sweep to find optimal values for your GPU/CPU combination.

---

## 8. Edge Score Aggregation Strategies

### The Problem:

Each edge can appear in **multiple walks** with different predicted labels:

```
Edge (u, v):
  - Walk 1: predicts class=1
  - Walk 5: predicts class=0
  - Walk 23: predicts class=1
  
How to aggregate into ONE final score?
```

### Available Aggregation Methods:

1. **Mean Probability** (recommended for soft labels)
   ```python
   edge_score = mean([p_walk1, p_walk5, p_walk23])
   ```

2. **Majority Vote** (for hard labels)
   ```python
   edge_score = mode([class_walk1, class_walk5, class_walk23])
   ```

3. **Weighted by Position in Walk**
   ```python
   # Weight edges near middle of walk higher (more central = more important)
   weights = [0.8, 0.9, 1.0, 0.9, 0.8] if walk_length=5
   edge_score = weighted_mean(scores, weights)
   ```

4. **Learned Aggregation** (advanced)
   ```python
   # Train a small MLP to learn how to combine walk-level predictions
   final_score = MLP([scores_all_walks]) → single output
   ```

5. **Length-Aware Aggregation** (novel idea!)
   ```python
   # Walks of length 3 (2 edges) might predict differently than length 50
   # Group by walk_length, aggregate within group, then combine
   score_short = mean(predictions_from_walks_len_1_to_10)
   score_long = mean(predictions_from_walks_len_40_to_100)
   final_score = weighted_combo(score_short, score_long, weights=[0.4, 0.6])
   ```

### Implementation:

The `evaluation_pipeline.py` has a ready-to-use aggregation function:

```python
from scripts.evaluation_pipeline import aggregate_edge_scores_across_walks

predictions_per_edge = {
    123: [0.9, 0.8, 0.85],  # Edge 123 scored 0.9, 0.8, 0.85 across 3 walks
    456: [0.2, 0.3, 0.25],  # Edge 456 scored 0.2, 0.3, 0.25
}

# Simple mean
result = aggregate_edge_scores_across_walks(predictions_per_edge, "mean")
# {123: 0.85, 456: 0.25}

# Majority vote
result = aggregate_edge_scores_across_walks(predictions_per_edge, "majority_vote")

# Weighted by position
result = aggregate_edge_scores_across_walks(predictions_per_edge, "weighted_by_walk_length")
```

### Recommended Next Step:

1. **Start with mean aggregation** — simplest, often works well
2. **Experiment with majority_vote** — if you want hard labels
3. **Try length-aware** — might help since walk length affects prediction confidence
4. **Eventually build a learned aggregator** — fine-tune MLP to optimize for your task

---

## 9. Creating Pre-Built Datasets & Models

For fast evaluation without rebuilding:

```bash
# Step 1: Enable cache and prepare data for all datasets
python -c "
from scripts.evaluation_pipeline import EvaluationPipeline
for dataset in ['wiki-rfa', 'epinions', 'slashdot090221']:
    pipe = EvaluationPipeline(dataset)
    pipe.prepare_data(use_cache=True)  # First run: builds cache
    print(f'✅ Cached {dataset}')
"

# Now all future runs are fast!
python scripts/evaluation_pipeline.py wiki-rfa  # < 10 seconds
python scripts/evaluation_pipeline.py epinions  # < 10 seconds
```

Cache locations:
```
outputs/wiki-rfa/wiki-rfa_encoded.pt      (large file, ~500MB-2GB)
outputs/wiki-rfa/wiki-rfa_meta.json       (small, ~1KB)
outputs/epinions/epinions_encoded.pt
outputs/epinions/epinions_meta.json
outputs/slashdot090221/slashdot090221_encoded.pt
outputs/slashdot090221/slashdot090221_meta.json
```

---

## Quick Action Items

### Immediate (today):
- [ ] Run `python scripts/analyze_optuna_studies.py` — see your results
- [ ] Run `python scripts/extract_top_trials.py wiki-rfa --metric num_walks` — find faster alternatives
- [ ] Run `python scripts/evaluation_pipeline.py epinions` — test evaluation pipeline

### Short-term (this week):
- [ ] Enable `use_cache: true` in config.yaml
- [ ] Run evaluation on all 3 datasets to build caches
- [ ] Design your edge aggregation strategy (start with mean)

### Medium-term (optimization):
- [ ] Profile walk sampling — identify exact slow lines
- [ ] Try num_workers = 24/32 in Optuna HPO
- [ ] Implement learned aggregation for edges

---

## File Reference

| Script | Purpose |
|--------|---------|
| `scripts/analyze_optuna_studies.py` | Summarize all Optuna studies |
| `scripts/load_best_model.py` | Load best models + hyperparams |
| `scripts/extract_top_trials.py` | Find high-scoring trials with fewer walks |
| `scripts/evaluation_pipeline.py` | Complete evaluation with caching |
| `src/data/prepare_data.py` | Data preprocessing with caching logic |

---

## Next Steps

Once you have cached datasets and trained models, the focus shifts to:

1. **Edge score aggregation** — determine best way to combine walk-level predictions
2. **Aggregation evaluation** — measure how different aggregation methods affect downstream task performance
3. **Optimization** — if aggregation is slow, parallelize or use GPU

This is the **core problem** you want to solve next, and you now have the infrastructure to do fast iteration.
