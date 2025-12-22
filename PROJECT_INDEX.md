# Project Index & Navigation Guide

Quick navigation to all documentation and tools.

## 📘 Main Documentation (READ FIRST)

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [COMEBACK_SUMMARY.md](COMEBACK_SUMMARY.md) | **START HERE** — Welcome back overview, checklist, quick start | 10 min |
| [EDGE_AGGREGATION_GUIDE.md](EDGE_AGGREGATION_GUIDE.md) | Your core next problem: how to combine edge predictions across walks | 15 min |
| [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) | Comprehensive optimization reference: caching, DataLoaders, HPO | 20 min |
| [scripts/README.md](scripts/README.md) | Command reference for all tools and typical workflows | 10 min |

## 🛠️ Tools & Scripts

### Quick Start
```bash
# Run full setup with recommendations
python scripts/quickstart.py
```

### By Task

#### Analyze Your Results
```bash
# See all optuna studies across datasets
python scripts/analyze_optuna_studies.py
```

#### Load & Inspect Models
```bash
# Get best hyperparams for a dataset
python scripts/load_best_model.py wiki-rfa

# Load the actual trained model
python scripts/load_best_model.py wiki-rfa --load-model

# With resume info
python scripts/load_best_model.py epinions --resume
```

#### Find Speed Tradeoffs
```bash
# Find trials with fewer walks but similar score
python scripts/extract_top_trials.py wiki-rfa --metric num_walks --tolerance 3

# Find trials with shorter walks
python scripts/extract_top_trials.py epinions --metric max_walk_length --tolerance 5
```

#### Evaluate Models
```bash
# Complete evaluation with caching
python scripts/evaluation_pipeline.py wiki-rfa --device 2

python scripts/evaluation_pipeline.py epinions
```

#### Profile Data Building
```bash
# Quick profile (6 configurations)
python scripts/profile_data_building.py wiki-rfa --quick

# Full profile (20 configurations)
python scripts/profile_data_building.py epinions --full
```

## 📊 Current Project State

### Optuna Studies
| Dataset | Trials | Best Score | Best Trial | Study Folder |
|---------|--------|-----------|-----------|--------------|
| **Wiki-RFA** | 200 | 0.7779 | #87 | `wiki-rfa-optuna_20251203-235202` |
| **Epinions** | 63 | 0.9133 | #31 | `epinions-optuna_20251203-235202` |
| **Slashdot** | 21 | 0.8529 | #12 | `slashdot090221-optuna_20251203-235202` |

**All studies are resumable** with:
```bash
python optuna_run.py --config=config.yaml --n-trials=<N> dataset.name=<dataset>
```

### Available Infrastructure
- ✅ Trained models (checkpoints saved for all best trials)
- ✅ Reproducible hyperparameters (stored in Optuna study)
- ✅ Dataset caching (available but currently disabled)
- ✅ Fast evaluation pipeline (ready to use)
- ✅ Data profiling tools (understand bottlenecks)

### Next Focus
→ **Edge Score Aggregation** (see `EDGE_AGGREGATION_GUIDE.md`)

## 🚀 Quick Command Reference

```bash
# See all results
python scripts/analyze_optuna_studies.py

# Find fast alternatives
python scripts/extract_top_trials.py <dataset> --metric num_walks --tolerance 3

# Load trained model
python scripts/load_best_model.py <dataset> --load-model

# Evaluate
python scripts/evaluation_pipeline.py <dataset>

# Profile preprocessing
python scripts/profile_data_building.py <dataset> --quick

# Full setup
python scripts/quickstart.py
```

## 📁 File Organization

```
walk_to_paint/
├── COMEBACK_SUMMARY.md           ← START HERE
├── EDGE_AGGREGATION_GUIDE.md     ← Core problem explained
├── OPTIMIZATION_GUIDE.md         ← Reference guide
├── PROJECT_INDEX.md              ← This file
│
├── scripts/
│   ├── README.md                 ← Script reference
│   ├── analyze_optuna_studies.py
│   ├── load_best_model.py
│   ├── extract_top_trials.py
│   ├── evaluation_pipeline.py
│   ├── profile_data_building.py
│   └── quickstart.py
│
├── src/
│   ├── data/prepare_data.py      ← Has caching support
│   ├── model/lit_model.py
│   └── utils/config.py
│
├── config.yaml                   ← Edit to enable caching
├── optuna_run.py                 ← Resume with this
├── run.py                        ← Training entrypoint
│
└── outputs/
    ├── wiki-rfa/
    ├── epinions/
    └── slashdot090221/
        └── <exp>-optuna_<timestamp>/
            ├── optuna/optuna_study.log    ← Your studies
            ├── checkpoints/               ← Trained models
            ├── logs/                      ← TensorBoard logs
            └── plots/                     ← Metrics plots
```

## 🎯 Recommended Workflow

### Phase 1: Explore (30 minutes)
```bash
# Understand what you have
python scripts/analyze_optuna_studies.py
python scripts/extract_top_trials.py wiki-rfa --metric num_walks --tolerance 3
python scripts/load_best_model.py wiki-rfa
```

### Phase 2: Setup (1-2 hours)
```bash
# Edit config.yaml
preprocess:
  use_cache: true  # Enable caching

# Build cached datasets (first time only)
python scripts/evaluation_pipeline.py wiki-rfa
python scripts/evaluation_pipeline.py epinions
python scripts/evaluation_pipeline.py slashdot090221

# Now evaluation is FAST!
python scripts/evaluation_pipeline.py wiki-rfa  # < 10 seconds
```

### Phase 3: Core Work (this month)
1. Read `EDGE_AGGREGATION_GUIDE.md` (understand the problem)
2. Choose aggregation strategy (start with Mean)
3. Implement in `evaluation_pipeline.py`
4. Test and measure impact
5. Iterate on best method

### Phase 4: Optimization (ongoing)
- Continue Optuna trials if needed: `python optuna_run.py --n-trials=50 dataset.name=wiki-rfa`
- Try different aggregation methods
- Measure downstream task performance

## 📖 How to Read the Documentation

**First Time Here?**
1. Start with [COMEBACK_SUMMARY.md](COMEBACK_SUMMARY.md)
2. Run the quick start commands
3. Read [EDGE_AGGREGATION_GUIDE.md](EDGE_AGGREGATION_GUIDE.md) sections 1-3

**Ready to Code?**
1. Check [scripts/README.md](scripts/README.md) for command syntax
2. Review [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) section 2-4
3. Start implementing edge aggregation

**Hitting Issues?**
1. Check [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) troubleshooting
2. Look at script docstrings (they have usage examples)
3. Run scripts with available flags/help

## 💡 Key Concepts

**Optuna Studies** — Hyperparameter optimization runs for each dataset
- All resumable with `load_if_exists=True`
- Best trials selected and models saved

**Dataset Caching** — Preprocessed data saved to disk
- First load: 5-10 minutes (builds cache)
- Subsequent loads: <10 seconds (loads from cache)
- Enable: change `preprocess.use_cache: true` in config.yaml

**Edge Score Aggregation** — Core algorithmic problem
- Each edge appears in multiple walks
- Model predicts different classes across walks
- Need to combine into one final label
- 7 strategies from simple (Mean) to advanced (Learned MLP)

**Speed/Accuracy Tradeoffs** — Time vs. Score
- Fewer walks = faster, slightly lower score
- Find optimal using `extract_top_trials.py`
- Example: 0.77% score drop → 47% faster

## 🔗 External Links

- **Optuna Documentation:** https://optuna.readthedocs.io/
- **PyTorch Lightning:** https://lightning.ai/
- **DGL (Graph Learning):** https://www.dgl.ai/

## 📞 Need Help?

All scripts have docstrings with usage examples. Try:

```bash
# See help for any script
python scripts/load_best_model.py --help

# Check docstring
python -c "from scripts.load_best_model import OptunaBestModel; help(OptunaBestModel.find_best_study)"

# Read the README
cat scripts/README.md
```

## ✅ Next Immediate Steps

1. [ ] Read [COMEBACK_SUMMARY.md](COMEBACK_SUMMARY.md)
2. [ ] Run `python scripts/analyze_optuna_studies.py`
3. [ ] Run `python scripts/extract_top_trials.py wiki-rfa --metric num_walks --tolerance 3`
4. [ ] Read [EDGE_AGGREGATION_GUIDE.md](EDGE_AGGREGATION_GUIDE.md)
5. [ ] Choose aggregation strategy
6. [ ] Enable caching and build datasets
7. [ ] Implement aggregation in `evaluation_pipeline.py`

---

**Welcome back! Everything is ready for you.** 🚀
