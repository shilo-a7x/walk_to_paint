# Configuration Guide: Reproducibility & Config System

## Overview

The walk_to_paint configuration system is designed around **single-source-of-truth** principles to ensure reproducibility and clarity across the entire pipeline.

## Core Principle: Unified Seed Configuration

**All randomness in the codebase is controlled by a single seed:**

```yaml
reproducibility:
    seed: 42
```

This seed is **MANDATORY** and must be present in all configurations. The system fails **loudly** with a clear error message if the seed is missing—there are no silent defaults.

### What This Seed Controls

The `reproducibility.seed` controls all sources of randomness:

- **PyTorch**: Model weight initialization, DataLoader shuffling, dropout layers
- **NumPy**: Random walk sampling, data splitting, aggregator training
- **Python `random` module**: Edge shuffling during preprocessing
- **Worker processes**: Deterministic seeding for parallel data loaders (seed = base_seed + worker_id)

## Configuration Hierarchy

The configuration system uses a three-level hierarchy:

```
Base Config (config.yaml)
    ↓
Dataset-Specific Config (configs/<dataset>.yaml)
    ↓
CLI Overrides (--key=value)
```

Each level **overrides** the previous one. Final config is validated before use.

### 1. Base Configuration (`config.yaml`)

Contains defaults that apply to all datasets.

**Key Sections:**

#### Reproducibility
```yaml
reproducibility:
    seed: 42  # MANDATORY - all randomness controlled by this
```

#### Preprocessing
```yaml
preprocess:
    save: true           # Cache preprocessed data
    use_cache: true      # Reuse cached data if available
    num_workers: 8       # Parallel workers for walk sampling
```

#### Dataset (defaults)
```yaml
dataset:
    name: bitcoin-alpha-binary
    binary: false
    multiedge_handling: keep
    train_ratio: 0.48
    mask_ratio: 0.32
    val_ratio: 0.1
    test_ratio: 0.1
```

#### Training (includes DataLoader config)
```yaml
training:
    # Model training
    batch_size: 64
    epochs: 25
    lr: 0.001
    
    # DataLoader configuration
    num_workers: 16          # Parallel workers (0 = main process only)
    persistent_workers: true # Keep workers alive between epochs
    pin_memory: true         # Pin GPU memory for faster transfer
    prefetch_factor: 2       # Batches prefetched per worker
```

### 2. Dataset-Specific Configs (`configs/<dataset>.yaml`)

Each dataset (epinions, slashdot090221, wiki-rfa) has its own config that overrides base defaults.

**Example: `configs/epinions.yaml`**
```yaml
dataset:
    name: epinions
    binary: true              # Override base config
    max_walk_length: 79
    num_walks: 4966522

training:
    batch_size: 256           # Override base config
    lr: 0.002936
```

**Note**: Dataset-specific configs do NOT override `reproducibility.seed` or `preprocess.num_workers`. They inherit these from the base config.

### 3. CLI Overrides

Override any config value from the command line:

```bash
# Override seed for a single run
python run.py --reproducibility.seed=123 --dataset.name=epinions

# Override training parameters
python run.py --training.epochs=50 --training.lr=0.0001

# Override preprocessing workers
python run.py --preprocess.num_workers=16
```

Multiple overrides can be chained:
```bash
python run.py \
  --reproducibility.seed=42 \
  --dataset.name=wiki-rfa \
  --training.epochs=30 \
  --training.batch_size=128
```

## How the Pipeline Uses Configuration

### 1. Entry Point: `run.py`

```python
def main():
    cfg = load_config("config.yaml", overrides=sys.argv[1:])
    
    # Get seed - fails if not configured
    seed = get_seed(cfg)
    
    # Set all random seeds
    seed_everything(seed, workers=True)  # PyTorch Lightning
    random.seed(seed)                    # Python random
    np.random.seed(seed)                 # NumPy
    
    # Proceed with training
    train_model(cfg)
```

**Key Property**: The seed set here automatically propagates to all downstream code.

### 2. Data Preprocessing: `prepare_data.py`

```python
def split_edges(cfg, edges):
    # CRITICAL: Seed before shuffle for reproducibility
    seed = get_seed(cfg)
    random.seed(seed)
    
    edges_copy = list(edges)
    random.shuffle(edges_copy)  # Now deterministic
    # ... split into train/mask/val/test

def get_walks(cfg, edges):
    # Get seed from unified config
    walk_seed = get_seed(cfg)
    walk_workers = int(cfg.preprocess.num_workers)
    
    # Walk sampling uses seed + worker_id for deterministic parallelism
    # Results are sorted by task ID to ensure consistent ordering
    walks = sample_random_walks(
        edges,
        num_walks=cfg.dataset.num_walks,
        max_walk_length=cfg.dataset.max_walk_length,
        num_workers=walk_workers,
        seed=walk_seed,
    )
    return walks

def make_dataloaders(cfg, ...):
    # Get seed and worker config
    seed = get_seed(cfg)
    num_workers = cfg.training.num_workers
    
    # Create seeded generator for deterministic shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Each worker gets seed + worker_id
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32 - 1))
        torch.manual_seed(worker_seed)
    
    return DataLoader(..., generator=generator, worker_init_fn=worker_init_fn)
```

### 3. Standalone Scripts

All standalone scripts (extract_edge_scores.py, train_aggregator.py) must load config and get seed:

```python
def main():
    cfg = load_config(args.config, overrides=[])
    
    # Get seed - fails loudly if not configured
    seed = get_seed(cfg)
    
    # Apply seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Proceed with work
```

**Critical**: Standalone scripts called directly (not via run.py) must load their own config. They do NOT inherit the global seed that run.py sets—they need to load and set seeds independently.

## Reproducibility Guarantees

### When Run Through `run.py`

```bash
python run.py --reproducibility.seed=42
```

✅ **Fully reproducible**. All randomness is seeded:
- Walk sampling uses seed 42
- Data splitting uses seed 42
- DataLoader shuffling uses seed 42
- Model initialization uses seed 42
- All worker processes use seed 42 + worker_id

### When Running Standalone Scripts

```bash
python scripts/extract_edge_scores.py --config config.yaml --checkpoint model.ckpt --output scores.pkl
```

✅ **Reproducible if config has seed set**. The script loads `config.yaml`, gets seed from `reproducibility.seed`, and applies it.

❌ **Not reproducible if seed is missing**. The script will exit with:
```
❌ ERROR: reproducibility.seed is not set in config.
```

### When Using CLI Overrides

```bash
python run.py --reproducibility.seed=123 --training.epochs=20
```

✅ **Reproducible**. The CLI override takes precedence over config file.

## DataLoader Configuration: Threading vs Multiprocessing

The `training` section controls parallel data loading:

```yaml
training:
    num_workers: 16          # Number of worker processes
    persistent_workers: true # Keep workers alive between epochs
    pin_memory: true         # Pin GPU memory
    prefetch_factor: 2       # Batches to prefetch per worker
```

### Guidelines

- **`num_workers: 0`**: Single process, slower, but simpler debugging
- **`num_workers: 4-8`**: Good for most cases, balances speed and memory
- **`num_workers: 16+`**: For large datasets, high-throughput training

### Reproducibility With Multiple Workers

When `num_workers > 0`, the system ensures deterministic behavior:

1. **Main process** sets seed = base_seed
2. **Each worker** is initialized with seed = base_seed + worker_id
3. **Generator** for shuffling uses base_seed
4. Result: Identical randomness regardless of num_workers setting

## Walk Sampling: Preprocess Workers

Walk sampling runs in parallel:

```yaml
preprocess:
    num_workers: 8  # Parallel walk sampling processes
```

**IMPORTANT: Per-Walk Deterministic Seeding**

The walk sampler ensures perfect reproducibility by using **per-walk seeding**. Each walk uses its absolute index (0 to num_walks-1) to seed its RNG:

```python
for walk_idx in range(start_idx, end_idx):
    rng = np.random.default_rng(base_seed + walk_idx)
    # Generate walk with this specific seed
```

This guarantees that:
- **walk[0] is always identical** regardless of num_workers
- **walk[i] is always identical** for any i, regardless of which worker processes it
- Walks are reproducible across different machines, OS, Python versions, and CPU scheduling

**Why this matters:**
- Previous approach: Each worker chunk got a different seed → walk order depended on chunk boundaries
- Current approach: Each walk gets its own seed based on absolute index → walk[i] is deterministic

**Testing confirms:**
```python
walks_1worker = sample_random_walks(..., num_workers=1, seed=42)
walks_4worker = sample_random_walks(..., num_workers=4, seed=42)
assert walks_1worker == walks_4worker  # ✅ IDENTICAL
```

## Configuration Validation

The system validates configuration at load time:

```python
from src.utils.config import load_config, get_seed

cfg = load_config("config.yaml")

# This WILL raise ValueError if reproducibility.seed is missing
seed = get_seed(cfg)
```

Error messages are descriptive:

```
ValueError: reproducibility.seed is not set in config.
Please add 'reproducibility:
  seed: 42' to your config.yaml 
or set it via CLI: --reproducibility.seed=42
```

## Common Tasks

### Set a Different Seed for Experimentation

```bash
python run.py --reproducibility.seed=999
```

### Use More Workers for Faster Training

```bash
python run.py --training.num_workers=32 --preprocess.num_workers=16
```

### Train on Different Dataset

```bash
python run.py --dataset.name=epinions
```

### Use Dataset Config + Override One Parameter

```bash
python run.py --dataset.name=wiki-rfa --training.epochs=50
```

### Extract Edge Scores with Matching Seed

```bash
# Ensure extract_edge_scores.py uses same seed as training
python scripts/extract_edge_scores.py \
  --config config.yaml \
  --checkpoint checkpoints/model.ckpt \
  --output scores.pkl
```

(Seed comes from config.yaml's `reproducibility.seed`)

### Reproduce a Specific Optuna Trial

```bash
# Optuna trials use the global seed + trial-specific hyperparams
# To reproduce trial #87, use same seed:
python run.py --reproducibility.seed=42 --dataset.name=wiki-rfa
```

## Migration Guide: From Old Seed System

**Old Config (deprecated):**
```yaml
preprocess:
    walk_seed: 42        # ❌ No longer used

dataset:
    seed: 42             # ❌ No longer used

training:
    seed: 42             # ❌ No longer used
```

**New Config (current):**
```yaml
reproducibility:
    seed: 42             # ✅ Single canonical seed

preprocess:
    num_workers: 8       # ✅ Renamed from walk_num_workers
```

**Code Migration:**
```python
# Old (no longer works)
seed = getattr(cfg.training, "seed", None)

# New (required)
from src.utils.config import get_seed
seed = get_seed(cfg)  # Fails loudly if not set
```

## Summary

- **Single seed source**: `reproducibility.seed` controls all randomness
- **No silent defaults**: Missing seed causes immediate, clear error
- **Three-level hierarchy**: Base config → Dataset config → CLI overrides
- **Parallel workers are deterministic**: seed + worker_id ensures consistency
- **Standalone scripts are independent**: Each must load its own config and set seeds

This design ensures that:
1. ✅ Experiments are fully reproducible
2. ✅ Configuration mistakes are caught immediately
3. ✅ The seed's role is crystal clear
4. ✅ Parallelism doesn't compromise reproducibility
