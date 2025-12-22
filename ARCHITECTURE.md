# Walk-to-Paint: System Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    WALK-TO-PAINT PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

INPUT: Graph with edges & labels
   │
   ├─→ [1] WALK SAMPLING
   │   └─→ Generate random walks for each edge
   │   └─→ Returns: walks for entire graph
   │
   ├─→ [2] WALK ENCODING
   │   └─→ Tokenize node/edge sequences
   │   └─→ Create input sequences for model
   │   └─→ Returns: tokenized walk sequences
   │
   ├─→ [3] OPTUNA OPTIMIZATION (HPO)
   │   └─→ Hyperparameter search over:
   │       - num_walks, max_walk_length
   │       - model architecture (hidden_dim, nhead, nlayers, dropout)
   │       - training (lr, batch_size, gradient_clip)
   │   └─→ Returns: best hyperparams + trained model
   │
   ├─→ [4] MODEL TRAINING (PyTorch Lightning)
   │   └─→ Transformer-based edge classifier
   │   └─→ Loss: edge classification (binary or multi-class)
   │   └─→ Validation: AUC / Accuracy
   │   └─→ Returns: trained model checkpoint
   │
   ├─→ [5] INFERENCE ON TEST SET
   │   └─→ Get model predictions for each edge
   │   └─→ Returns: predictions_per_walk[edge_id] = [pred1, pred2, ...]
   │
   └─→ [6] EDGE SCORE AGGREGATION ← YOUR CORE PROBLEM
       └─→ Combine predictions across multiple walks
       └─→ Strategies: Mean, Majority Vote, Learned MLP, etc.
       └─→ Returns: final_score[edge_id] (single aggregated score)

OUTPUT: Final edge scores for downstream tasks
```

## System Components

```
┌──────────────────────────────────────────────────────────────────┐
│                        SOURCE CODE                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  src/                                                            │
│  ├── data/                                                       │
│  │   ├── datasets.py ............ Edge loaders (gzip, wiki-rfa) │
│  │   ├── prepare_data.py ........ Full pipeline + caching      │
│  │   ├── tokenizer.py ........... Encode walks                 │
│  │   └── walk_sampler.py ........ Sample random walks          │
│  │                                                              │
│  ├── model/                                                      │
│  │   └── lit_model.py ........... LitEdgeClassifier (Lightning) │
│  │                                                              │
│  └── utils/                                                      │
│      ├── config.py ............. Config loading + merging       │
│      └── paths.py .............. Output directory management    │
│                                                                  │
│  optuna_run.py ................. HPO orchestration              │
│  run.py ........................ Single model training          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                      TOOLS & SCRIPTS                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  scripts/                                                        │
│  ├── analyze_optuna_studies.py ... See all results              │
│  ├── load_best_model.py ......... Load trained models           │
│  ├── extract_top_trials.py ...... Find speed/accuracy tradeoffs │
│  ├── evaluation_pipeline.py ..... End-to-end eval + aggregate   │
│  ├── profile_data_building.py ... Benchmark preprocessing       │
│  └── quickstart.py ............. Full setup                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                      DATA & STORAGE                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  outputs/                                                        │
│  ├── wiki-rfa/                                                   │
│  │   ├── wiki-rfa-optuna_20251203-235202/          [STUDY 1]    │
│  │   │   ├── optuna/optuna_study.log               [JOURNAL]    │
│  │   │   ├── checkpoints/trial_*.ckpt              [MODELS]     │
│  │   │   ├── logs/                                 [TENSORBOARD]│
│  │   │   └── plots/                                [METRICS]    │
│  │   └── wiki-rfa_encoded.pt                       [CACHE]      │
│  │                                                              │
│  ├── epinions/                                                   │
│  │   └── epinions-optuna_20251203-235202/          [STUDY 2]    │
│  │                                                              │
│  └── slashdot090221/                                             │
│      └── slashdot090221-optuna_20251203-235202/    [STUDY 3]    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow: Training

```
config.yaml
    ↓
load_config()
    ↓ (+ CLI overrides)
    ↓
optuna_run.py
    ├─→ JournalStorage(optuna_study.log)
    │   └─→ Load existing study or create new
    │
    ├─→ prepare_data(cfg)
    │   ├─→ get_edge_list() [from datasets.py]
    │   ├─→ split_edges()
    │   ├─→ sample_random_walks() [BOTTLENECK: ~300s]
    │   ├─→ get_tokenizer()
    │   ├─→ encode_walks()
    │   ├─→ pad_and_build_stage_tensors()
    │   └─→ make_dataloaders()
    │       └─→ Returns: train_loader, val_loader, test_loader
    │
    ├─→ For each trial:
    │   ├─→ suggest_hyperparams()
    │   ├─→ build_trainer()
    │   ├─→ trainer.fit(train_loader, val_loader)
    │   ├─→ Save checkpoint if best
    │   └─→ Report trial result to study
    │
    └─→ Output: best_trial with checkpoint
```

## Data Flow: Evaluation

```
evaluation_pipeline.py
    │
    ├─→ find_best_study()
    │   └─→ Load from optuna_study.log
    │
    ├─→ load_model()
    │   ├─→ Get best trial hyperparams
    │   └─→ Load checkpoint
    │
    ├─→ prepare_data(use_cache=True)
    │   ├─→ Check if cache exists (encoded.pt, meta.json)
    │   ├─→ If yes: load from cache (< 10 sec)
    │   └─→ If no: full preprocessing (5-10 min) + save cache
    │
    ├─→ Inference loop:
    │   └─→ For each batch in test_loader:
    │       ├─→ model(batch) → logits
    │       ├─→ Collect predictions by edge
    │       └─→ predictions_per_edge[edge_id].append(pred)
    │
    └─→ aggregate_edge_scores()
        ├─→ For each edge, combine predictions across walks
        ├─→ Methods: mean, majority_vote, learned, etc.
        └─→ Returns: final_score[edge_id]
```

## Optimization Flow

```
PROBLEM: Preprocessing too slow (300+ seconds)
    │
    ├─→ Profile with profile_data_building.py
    │   └─→ Identify: walk_sampling is 90% of time
    │
    ├─→ SOLUTION 1: Reduce num_walks
    │   ├─→ Use extract_top_trials.py
    │   ├─→ Find: Trial #35 uses 662k walks, only 0.77% slower
    │   └─→ Recommendation: Use fewer walks in practice
    │
    ├─→ SOLUTION 2: Enable caching
    │   ├─→ First run: cache preprocessed data
    │   ├─→ Subsequent runs: load in < 10 seconds
    │   └─→ 100x speedup!
    │
    ├─→ SOLUTION 3: Tune DataLoaders
    │   ├─→ num_workers: 16 → 32
    │   ├─→ prefetch_factor: 2 → 4-8
    │   └─→ persistent_workers: true (already enabled)
    │
    └─→ SOLUTION 4: Optimize walk sampling algorithm
        ├─→ Vectorize with NumPy
        ├─→ JIT compile with Numba/Cython
        └─→ Parallelize with multiprocessing
```

## Key Decisions & Tradeoffs

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE DECISIONS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. WHY RANDOM WALKS?                                           │
│     → Captures local neighborhood structure                     │
│     → Each edge gets multiple context windows                   │
│     → Allows aggregation across walks                           │
│                                                                 │
│  2. WHY OPTUNA HPO?                                             │
│     → Efficient hyperparameter search                           │
│     → Pruning eliminates bad trials early                       │
│     → Reproducibility per-trial                                │
│                                                                 │
│  3. WHY PYTORCH LIGHTNING?                                      │
│     → Abstraction over training boilerplate                     │
│     → Multi-GPU support built-in                               │
│     → Structured logging and checkpointing                      │
│                                                                 │
│  4. WHY EDGE AGGREGATION?                                       │
│     → Edges appear in multiple walks                            │
│     → Model predictions vary across walks                       │
│     → Final label requires combining evidence                   │
│     → This is the core unsolved problem                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

```
TIMING BREAKDOWN (per dataset)

walk_sampling........... ~300s (300 walks/sec for 100k walks)
tokenizer .............. ~1-5s
encode_walks............ ~13s
padding/building........ ~3s
DataLoader creation..... < 1s
────────────────────────────────────
TOTAL FIRST RUN ........ ~320s (~5 minutes)

WITH CACHING:
load cache ............. ~5s
DataLoader creation..... < 1s
────────────────────────────────────
TOTAL CACHED ........... ~6s (50x faster!)

INFERENCE:
forward pass per batch.. ~50-100ms (depends on batch_size)
N batches in test set... ~100-1000 (depends on dataset size)
total inference......... ~5-100s
aggregation............ ~1-5s
────────────────────────────────────
TOTAL EVALUATION ....... ~5-105s
```

## Storage Footprint

```
Per dataset (approximate):

Edge list ......................... 50-500 MB (raw graph)
Preprocessed walks (encoded)...... 500 MB - 2 GB
Tokenizer state................... < 1 MB
Model checkpoints................. 50-200 MB per checkpoint
TensorBoard logs.................. 1-10 GB
────────────────────────────────────
TOTAL PER DATASET................ 2-15 GB

Optuna journal file............... 1-50 MB
────────────────────────────────────
TOTAL FOR 3 DATASETS............ ~15-50 GB
```

## Next Architecture Addition: Edge Aggregation

```
CURRENT:                          NEW:

model(walk)                       model(walk)
   ↓                                 ↓
prediction[edge]               prediction[edge]
   ↓                                 ↓
output (one per walk)          collect_across_walks()
                                   ↓
                              predictions_per_edge[edge]
                              = [pred_walk1, pred_walk2, ...]
                                   ↓
                              aggregate() ← CHOOSE METHOD
                              /            \
                            mean    majority_vote
                            /                  \
                     weighted_mean         consensus
                         /                      \
                   learned_mlp          position_weighted
                         /                      \
                   length_stratified          ...
                                   ↓
                            final_score[edge]
                                   ↓
                            downstream tasks
```

---

This architecture supports:
- ✅ Multiple datasets
- ✅ Hyperparameter optimization
- ✅ Reproducibility (per-trial seeds)
- ✅ Fast evaluation (caching)
- ✅ Flexible aggregation strategies

Your focus: implement edge aggregation (step 6).
