# 📊 TASK STATUS UPDATE (Feb 5, 2026)

## ✅ COMPLETED TASKS

### Phase 0: Cleanup

- ✅ **T0.1**: Cleaned up old documentation (40+ files removed)

### Phase 2: Prediction Infrastructure (JUST COMPLETED 🎉)

- ✅ **T3.1**: Save Predictions (with walk metadata)
  - Scripts: `extract_edge_scores.py` + `save_predictions.py`
  - Output: `outputs/predictions/<dataset>/raw_scores/<run_id>_<split>.pkl`
  - Status: Validated on wiki-rfa, full run in progress

- ✅ **T3.2**: Save Aggregator Triplets
  - Script: `scripts/save_triplets.py`
  - Output: `outputs/aggregation/<dataset>/strategy_mean/triplets_<split>.pkl`
  - Triplets: (dist_from_start, dist_from_end, correct_flag) per occurrence
  - Status: Validated on wiki-rfa, full run in progress

- ✅ **T3.3**: Heatmap Visualization
  - Script: `scripts/plot_triplet_heatmap.py`
  - Output: `outputs/aggregation/<dataset>/strategy_mean/heatmap_<split>.png`
  - Heatmap: average correct grouped by (dist_from_start, dist_from_end)
  - Status: Validated on wiki-rfa, full run in progress

---

## 🟡 IN PROGRESS / BLOCKED

### Current Full Run

```
nohup bash scripts/run_t3_full.sh > heatmap_all.out 2>&1 &
```

- Datasets: wiki-rfa, slashdot090221, epinions
- Splits: val + test
- Status: **Running** (check with `tail -f heatmap_all.out`)

---

## 🟡 NEXT PHASE: Research & Analysis (Phase 3)

Now that we have triplets and heatmaps, we have two paths:

### Option A: Continue with Aggregation Research (T4.1 → T4.3)

**Tasks**:

- **T4.1**: Implement MLP/Logistic aggregation (learned)
- **T4.2**: Triplet-based analysis of different aggregation strategies
- **T4.3**: Compare aggregation strategies

**Why**: Answer "what's the best way to aggregate edge predictions?"

**Time**: ~15-20 hours

---

### Option B: Foundation & Infrastructure (Phase 1 - Critical Path)

Before expanding research, build robust infrastructure:

- **T0.2**: Robust Config System (validation, schema)
- **T0.3**: Output Directory Structure (strict separation)
- **T1.1**: Verify Data Reproducibility
- **T1.2**: Optimize Data Stages
- **T1.3**: Implement Caching + Train-Ready Loading
- **T2.1-T2.3**: Class Imbalance Analysis & Loss Weighting

**Why**:

- These enable clean, reproducible experimentation
- Currently using current model checkpoints (not retrained with stratified splits)
- T2 (class imbalance) is critical before new training

**Time**: ~35-45 hours

---

## 🎯 Recommendation

### Short-term (Next 1-2 days)

1. ✅ Wait for T3 full run to complete (check `heatmap_all.out`)
2. 📊 Examine heatmaps from all 3 datasets
3. 📝 Document insights from heatmaps (where model does well/poorly)

### Medium-term (Next week)

**Choose Path A or B**:

- **Path A (Research-focused)**: Implement T4.1 (MLP aggregation) to see if we can improve
  - Quick wins, publishable results
  - Requires 15-20 hours
  
- **Path B (Infrastructure-focused)**: Do T0.2 → T0.3 → T1.x → T2.x first
  - Professional foundation
  - Enables reproducible experimentation
  - Required for future retraining
  - Requires 35-45 hours

---

## 📋 Current Status by Phase

| Phase | Task | Status | Time |
|-------|------|--------|------|
| **0** | T0.1 Cleanup | ✅ Done | 0.5h |
| **1** | T0.2 Config | 🟡 Not started | 4-6h |
| **1** | T0.3 Outputs | 🟡 Not started | 3-4h |
| **1** | T1.1 Data Verify | 🟡 Not started | 3-4h |
| **1** | T1.2 Data Optimize | 🟡 Not started | 4-5h |
| **1** | T1.3 Data Cache | 🟡 Not started | 5-6h |
| **1** | T2.1 Loss Analyze | 🟡 Not started | 2-3h |
| **1** | T2.2 Loss Weight | 🟡 Not started | 4-5h |
| **1** | T2.3 Leakage Check | 🟡 Not started | 2-3h |
| **2** | T3.1 Predictions | ✅ Done | 3-4h |
| **2** | T3.2 Triplets | ✅ Done | 3-4h |
| **2** | T3.3 Heatmap | ✅ Done | 3-4h |
| **2** | T4.1 MLP/Logistic | 🟡 Ready | 5-6h |
| **3** | T4.2 Analysis | 🟡 Ready | 3-4h |
| **3** | T4.3 Compare | 🟡 Ready | 4-5h |
| **3** | T5.1 Train Opt | 🟡 Ready | 4-5h |
| **3** | T5.2 Multiclass | 🟡 Ready | 2-3h |
| **3** | T5.3 Metrics | 🟡 Ready | 4-5h |

---

## 🎓 Professor's Requirements - Status

✅ **Done**:

- Save predictions with metadata
- Save triplets (dist_from_start, dist_from_end, correct_flag)
- Generate heatmaps showing model performance by position

🔄 **Could be next**:

- Analyze heatmaps for insights
- Try different aggregation strategies
- Compare results

---

## 💬 What Would You Like To Do Next?

**Option 1: Analyze Current Results** (1-2 hours)

- Check heatmaps from all 3 datasets
- Document what you see (where does model do well/poorly?)
- Write summary of insights

**Option 2: Implement MLP Aggregation** (5-6 hours)

- Try learned aggregation (logistic + MLP)
- Compare vs. mean aggregation
- See if we can improve edge predictions

**Option 3: Build Infrastructure** (35-45 hours)

- Start Phase 1 tasks (config, data, loss weighting)
- More robust and reproducible setup
- Required for proper retraining

**Which would you prefer?**
