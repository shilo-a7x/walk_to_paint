# 🎯 Prioritized Task List - Visual Summary

## Current Status
- ✅ A1: Seed config unified (complete)
- ✅ A6: Stratified splits (complete)
- 🟡 14 new tasks identified
- 📋 17 total tasks across all phases

---

## 🚨 CRITICAL PATH (Must Do These First)

### Week 1: Foundation & Infrastructure

#### **T0.2: Robust Config System** 🔴 START HERE
- **Time**: 4-6 hours
- **Why**: Everything depends on config
- **Blocks**: Everything else
- **Impact**: 🟢 Foundational

**Single Most Important**:
```
Current: cfg.dataset.name scattered, no validation
Target:  Unified schema, validated, type hints
```

#### **T0.3: Output Directory Restructuring**
- **Time**: 3-4 hours  
- **Why**: Clear separation (data, models, predictions, aggregation)
- **Blocks**: T3.x (prediction saving)
- **Impact**: 🟢 Organizational

#### **T1.1 → T1.2 → T1.3: Data Pipeline** (Sequence)
- **T1.1** (Verify): 3-4h
  - Understand current pipeline stages
  - Document reproducibility status
  
- **T1.2** (Optimize): 4-5h  
  - Merge unnecessary file I/O
  - Reduce disk usage
  
- **T1.3** (Caching): 5-6h
  - Enable fast retraining
  - Cache + load mechanism

#### **T2.1 → T2.2 → T2.3: Class Imbalance** (Sequence)
- **T2.1** (Analyze): 2-3h
  - How is loss currently weighted?
  - Any batch imbalance?
  
- **T2.2** (Implement): 4-5h
  - Add class-weighted loss
  - Global, per-epoch, or per-batch?
  
- **T2.3** (Validate): 2-3h
  - No data leakage
  - Verify val/test don't affect weights

**Total Week 1**: ~35-45 hours

---

### Week 2: Prediction Infrastructure

#### **T3.1: Save Model Predictions**
- **Time**: 3-4 hours
- **What**: Raw scores, labels, metadata
- **Blocks**: T3.2, T3.3
- **Impact**: 🟠 Foundation for analysis

#### **T3.2: Save Aggregator Triplets**
- **Time**: 3-4 hours
- **What**: Prof's requirement (dist_start, dist_end, correct_flag)
- **Blocks**: T3.3, T4.1
- **Impact**: 🟠 Prof's core request

#### **T3.3: Heatmap Analysis Tool**
- **Time**: 3-4 hours
- **What**: 2D heatmap (dist_start, dist_end) → avg_correct
- **Prof Wants**: YES, explicitly
- **Impact**: 🟠 Prof's visualization

#### **T4.1: MLP/Logistic Aggregation**
- **Time**: 5-6 hours
- **What**: Learned aggregation strategies
- **Blocks**: T4.2, T4.3
- **Impact**: 🟠 Research direction

**Total Week 2**: ~20-25 hours

---

### Week 3: Refinement & Analysis

#### **T4.2: Triplet-Based Analysis**
- **Time**: 3-4 hours
- **What**: Generate heatmaps for each strategy
- **Prof Wants**: Heatmap comparison
- **Impact**: 🟠 Insights

#### **T4.3: Compare Strategies**
- **Time**: 4-5 hours
- **What**: Mean vs. Logistic vs. MLP (metrics, speed)
- **Impact**: 🟡 Medium

#### **T5.1: Training Optimization**
- **Time**: 4-5 hours
- **What**: Speed, memory, GPU utilization
- **Impact**: 🟡 Performance

#### **T5.2: Binary/Multiclass**
- **Time**: 2-3 hours
- **What**: Support both task types
- **Impact**: 🟡 Flexibility

#### **T5.3: Comprehensive Metrics**
- **Time**: 4-5 hours
- **What**: AUC, F1, per-class metrics, curves
- **Impact**: 🟡 Evaluation

**Total Week 3**: ~22-27 hours

---

## 🎯 What You Get by End of Each Phase

### After Week 1 (T0-T2)
```
✅ Config system: Type-safe, validated, extensible
✅ Output structure: Clean separation by content type
✅ Data pipeline: Reproducible, optimized, cached
✅ Loss function: Class-weighted, no leakage
✅ Ready to: Retrain models easily, save outputs

Gain: Professional infrastructure
```

### After Week 2 (T3-T4.1)
```
✅ Predictions saved: All runs preserved for analysis
✅ Triplets generated: Prof's data ready
✅ Heatmaps created: Visualizations ready
✅ Aggregation: MLP working (strategy comparison ready)
✅ Ready to: Analyze, compare, visualize

Gain: Research capability + Prof's analysis
```

### After Week 3 (T4.2-T5.3)
```
✅ Strategy comparison: Know which aggregation is best
✅ Training optimized: 2-3x faster training
✅ Multiclass support: Flexible to other datasets
✅ Metrics comprehensive: Professional reporting
✅ Ready to: Publish, productize, iterate

Gain: Publication-ready + Production-ready
```

---

## 📊 By the Numbers

| Phase | Tasks | Hours | Benefit |
|-------|-------|-------|---------|
| **Phase 0** | 1 (cleanup) | 0.5 | Clean repo |
| **Phase 1** | 8 tasks (T0,T1,T2) | 35-45 | Infrastructure |
| **Phase 2** | 4 tasks (T3,T4.1) | 20-25 | Research ready |
| **Phase 3** | 5 tasks (T4.2-T5.3) | 22-27 | Publication ready |
| **TOTAL** | 17 tasks | 70-97 | Complete system |

---

## ⚡ Immediate Next Steps (Pick One)

### Option A: Start Cleanup (Fast Start - 30 min)
```bash
bash cleanup_old_docs.sh
```
Then pick either B or C below

### Option B: Start with T0.2 (Config System) - RECOMMENDED
- Most foundational
- Blocks everything else
- Cleanest to start fresh
- Will guide T0.3, T1.x, etc.

### Option C: Start with T1.1 (Data Verification) - ALTERNATIVE
- If you want to verify things work first
- Understand current pipeline better
- Then can design T0.2 around real requirements

---

## 🚀 Recommendation

```
1. Run: bash cleanup_old_docs.sh (30 min)
2. Start: T0.2 (Config System) (4-6 hours)
3. Then: T0.3 (Output Dirs) in parallel with T1.1
4. Follow priority list strictly

Why this order?
- Config drives everything (shape others)
- Outputs must follow config (cleaner)
- Data verification validates config design
- All others depend on these three
```

---

## 📞 Questions Before We Start?

1. **Cleanup**: Want me to run the cleanup script?
2. **Starting Point**: T0.2 (config) or T1.1 (verify data)?
3. **Scope**: Any tasks you want to skip or combine?
4. **Timeline**: How much time per day can you dedicate?

**Let me know and I'll create detailed implementation prompt for chosen task!**

