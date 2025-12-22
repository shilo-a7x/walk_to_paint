# Scripts Reference

Complete toolkit for managing Optuna studies, loading trained models, and evaluating with edge score aggregation.

## Quick Start

```bash
python scripts/quickstart.py
```

This runs all setup steps and shows recommendations.

---

## Scripts by Task

### 1. Analyze & Monitor Optuna Studies

#### `analyze_optuna_studies.py`
Summarize all Optuna studies across all datasets.

```bash
python scripts/analyze_optuna_studies.py
```

**Output:**
```
📊 DATASET: wiki-rfa
  📁 wiki-rfa-optuna_20251203-235202
     Study: walk_to_paint_study_wiki-rfa-optuna_20251203-235202
     Trials: 200 total
       ✅ Completed: 198
       ❌ Failed: 0
       ⏸️  Pruned: 2
     Best: Trial #87 with MAXIMIZE = 0.777932
```

---

### 2. Load & Inspect Trained Models

#### `load_best_model.py`
Load the best trained model for a dataset with all Optuna hyperparams.

```bash
# Get best hyperparams
python scripts/load_best_model.py wiki-rfa

# Load the model from checkpoint
python scripts/load_best_model.py wiki-rfa --load-model

# Show hyperparams + resume study
python scripts/load_best_model.py epinions --resume
```

**Usage in Python:**
```python
from scripts.load_best_model import OptunaBestModel

model_loader = OptunaBestModel("wiki-rfa")
model_loader.find_best_study()
hyperparams = model_loader.get_hyperparms()
trained_model = model_loader.load_model(device=2)
```

---

### 3. Find Better Hyperparameter Tradeoffs

#### `extract_top_trials.py`
Find high-scoring trials with **fewer walks** or **shorter walks** — useful for production where speed matters.

```bash
# Find 10 trials within 3% of best score but FEWEST WALKS
python scripts/extract_top_trials.py wiki-rfa --metric num_walks --tolerance 3

# Find trials with SHORTEST WALKS
python scripts/extract_top_trials.py epinions --metric max_walk_length --tolerance 5

# Looser tolerance (5% score drop allowed)
python scripts/extract_top_trials.py slashdot090221 --metric num_walks --tolerance 5
```

**Output:**
```
Trial #87 (best): score=0.7779, num_walks=434,857 ← Best accuracy
  Trial #102: score=0.7720, num_walks=639,012 ← 0.62% slower but 47% more walks
  Trial #35: score=0.7719, num_walks=662,661 ← 0.77% slower but 8.5% shorter walks
```

**Use Case:** Choose Trial #35 if you prioritize inference speed over max accuracy.

---

### 4. Complete Evaluation Pipeline

#### `evaluation_pipeline.py`
End-to-end evaluation with:
- Best model loading
- Cached dataset loading (fast!)
- Inference on test set
- Edge score aggregation across multiple walks

```bash
# Evaluate on test set (uses cache if available)
python scripts/evaluation_pipeline.py wiki-rfa

python scripts/evaluation_pipeline.py epinions --device 3

python scripts/evaluation_pipeline.py slashdot090221 --device 0
```

**Usage in Python:**
```python
from scripts.evaluation_pipeline import EvaluationPipeline, aggregate_edge_scores_across_walks

# Load model and prepare data
pipe = EvaluationPipeline("wiki-rfa")
pipe.find_best_study()
pipe.load_model(device=2)
pipe.prepare_data(use_cache=True)  # Cache saves/loads are on by default

# Evaluate
results = pipe.evaluate_on_test_set()

# Aggregate scores across walks
predictions_per_edge = {123: [0.9, 0.8, 0.85], 456: [0.2, 0.3]}
aggregated = aggregate_edge_scores_across_walks(predictions_per_edge, method="mean")
```

---

### 5. Optimize Data Building Speed

#### `profile_data_building.py`
Profile how long walk sampling takes for different `num_walks` and `max_walk_length` combinations. Walk caches are saved/loaded by default; tune `preprocess.walk_num_workers` and `preprocess.walk_seed` in `config.yaml` if you need faster builds or reproducibility.

```bash
# Quick profile (3x2 = 6 combinations)
python scripts/profile_data_building.py wiki-rfa --quick

# Full profile (5x4 = 20 combinations)
python scripts/profile_data_building.py epinions --full

# Output saved to profiling_<dataset>.json
```

**Output:**
```
⏱️  Time Estimate for wiki-rfa:
   num_walks: 434,857
   max_walk_length: 82
   Walk sampling: 127.3s
   Other stages: ~50s
   TOTAL: 177.3s (2.96 min)
```

**Use this to:**
- Identify which hyperparams cause slow preprocessing
- Find fast alternatives (fewer walks, shorter length)
- Make informed decisions about batch-size vs. speed tradeoffs

---

## Typical Workflow

### Phase 1: Initial Setup
```bash
# 1. See what you have
python scripts/analyze_optuna_studies.py

# 2. Find fast alternatives
python scripts/extract_top_trials.py wiki-rfa --metric num_walks --tolerance 3
```

### Phase 2: Prepare for Evaluation
```bash
# 3. Caching already enabled (config.yaml)
# 4. Build caches (takes 5-10 min first time)
python scripts/evaluation_pipeline.py wiki-rfa
python scripts/evaluation_pipeline.py epinions
python scripts/evaluation_pipeline.py slashdot090221
```

### Phase 3: Fast Iteration
```bash
# Now evaluate in <10 seconds:
python scripts/evaluation_pipeline.py wiki-rfa  

# Try different aggregation strategies by modifying evaluation_pipeline.py
# Continue Optuna if needed
python optuna_run.py --config=config.yaml --n-trials=50 dataset.name=wiki-rfa
```

---

## Command Cheat Sheet

```bash
# Analyze
python scripts/analyze_optuna_studies.py

# Load model
python scripts/load_best_model.py <dataset> [--load-model] [--resume]

# Extract alternatives
python scripts/extract_top_trials.py <dataset> --metric [num_walks|max_walk_length] --tolerance 5

# Evaluate
python scripts/evaluation_pipeline.py <dataset> [--device N]

# Profile
python scripts/profile_data_building.py <dataset> [--quick|--full]

# Quick start (all of above)
python scripts/quickstart.py
```

---

## Key Files

| File | Purpose |
|------|---------|
| `analyze_optuna_studies.py` | Summarize studies across datasets |
| `load_best_model.py` | Load best model + hyperparams for a dataset |
| `extract_top_trials.py` | Find high-scoring trials with speed tradeoffs |
| `evaluation_pipeline.py` | Complete eval pipeline with caching |
| `profile_data_building.py` | Profile walk sampling speed |
| `quickstart.py` | Run all setup steps |

---

## Next: Edge Score Aggregation

Once you have cached datasets and trained models, the next step is implementing smart edge score aggregation. See `OPTIMIZATION_GUIDE.md` section 8 for strategies.

Available methods:
- **Mean** — simple average of predictions across walks
- **Majority Vote** — most common class
- **Weighted by Position** — weight edges based on where they appear in walk
- **Learned Aggregation** — train a small neural network to combine scores
- **Length-Aware** — group by walk length, aggregate within group

Test each method by modifying `evaluation_pipeline.py` and measuring how it affects final task performance.
