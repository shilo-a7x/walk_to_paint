# 📚 Core Documentation Reference

**Start here to navigate the new task structure.**

---

## 🗺️ Navigation Guide

### For Understanding the Big Picture
1. **This file** (you are here)
2. [COMPREHENSIVE_RETHINKING.md](COMPREHENSIVE_RETHINKING.md) - Full task definitions
3. [PRIORITY_SUMMARY.md](PRIORITY_SUMMARY.md) - Visual priority chart

### For Implementation
- [Detailed task prompts] (will be created for chosen task)
- Code examples in implementation docs

### For Reference (A1-A6 Completed)
- [WALK_REPRODUCIBILITY_EXPLAINED.md](WALK_REPRODUCIBILITY_EXPLAINED.md) - A1 details
- [WALK_SOLUTION_COMPLETE.md](WALK_SOLUTION_COMPLETE.md) - A1 summary
- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - Current config (will be updated by T0.2)

---

## 🎯 The 14 Requirements Mapped to Tasks

| Requirement | Task | Hours |
|------------|------|-------|
| 1. Verify data reproducibility | T1.1 | 3-4 |
| 2. Optimize data stages | T1.2 | 4-5 |
| 3. Train-ready loading, easy retrain | T1.3 | 5-6 |
| 4. Train optimization | T5.1 | 4-5 |
| 5. Save predictions | T3.1 | 3-4 |
| 6. Save aggregator info (walks, positions) | T3.2 | 3-4 |
| 7. Class imbalance rethinking | T2.1-T2.3 | 8-11 |
| 8. Robust config system | T0.2 | 4-6 |
| 9. Output structure cleanup | T0.3 | 3-4 |
| 10. Easy for current experiments | T1.3 + T3.1 | - |
| 11. Binary + multiclass option | T5.2 | 2-3 |
| 12. Aggregator experimentation | T3.1-T4.1 | - |
| 13. Triplet analysis + heatmaps | T3.2-T3.3 | 6-8 |
| 14. Keep relevant old tasks | T5.3 (metrics) | 4-5 |

**Total**: 70-97 hours (~2-3 weeks full-time)

---

## 📋 Task Dependencies Graph

```
T0.1 Cleanup
├─ (no dependencies)

T0.2 Config ⚙️
├─ T0.1 (optional, cleanup first)
└─ BLOCKS: T0.3, T1.x, T2.x, T3.x

T0.3 Outputs 📁
├─ T0.2 (config must be ready)
└─ BLOCKS: T3.1, T3.2, T4.x

T1.1 Data Verify 🔍
├─ T0.2 (config for reproducibility settings)
└─ FEEDS: T1.2

T1.2 Data Optimize 🚀
├─ T1.1 (must understand pipeline first)
└─ FEEDS: T1.3

T1.3 Data Cache 💾
├─ T1.2 (must know what to cache)
├─ T0.3 (output structure)
└─ ENABLES: T3.1, T4.1 (fast experiments)

T2.1 Loss Analysis 📊
├─ T0.2 (config validation)
└─ FEEDS: T2.2

T2.2 Loss Weighting ⚖️
├─ T2.1 (understand current)
└─ FEEDS: T2.3

T2.3 Leakage Validation ✅
├─ T2.2 (implement weighting)
└─ VALIDATES: No data leakage

T3.1 Save Predictions 💾
├─ T0.3 (output structure)
├─ T1.3 (data loading)
└─ FEEDS: T3.2, T3.3, T4.1

T3.2 Save Triplets 📍
├─ T3.1 (must save predictions first)
└─ FEEDS: T3.3, T4.1

T3.3 Heatmap Tool 🗺️
├─ T3.2 (triplet data)
└─ PRODUCES: Prof's visualization

T4.1 MLP/Logistic 🧠
├─ T3.2 (triplet features)
├─ T0.3 (output structure)
└─ FEEDS: T4.2, T4.3

T4.2 Triplet Analysis 📈
├─ T4.1 (aggregation results)
├─ T3.3 (heatmap tool)
└─ PRODUCES: Insights

T4.3 Compare Strategies 📊
├─ T4.1 (all strategies)
├─ T4.2 (analysis)
└─ RECOMMENDS: Best approach

T5.1 Train Optimization ⚡
├─ T1.3 (optimized data)
└─ MEASURES: Speed, memory

T5.2 Binary/Multiclass 🎛️
├─ T0.2 (config schema)
├─ T2.2 (loss weighting)
└─ ENABLES: Multiclass support

T5.3 Metrics 📉
├─ T1.3 (validation setup)
├─ T0.3 (output structure)
└─ PRODUCES: Comprehensive metrics
```

---

## 🔴 CRITICAL BLOCKERS (Do These First!)

If you only have 1 day:
1. **T0.2** (Config) - 4h
   - Everything depends on good config
   - Saves time later (no cfg.typo errors)

If you have 3 days:
1. **T0.2** (Config) - 4h
2. **T0.3** (Outputs) - 3h
3. **T1.1** (Data Verify) - 3h

If you have 1 week:
1. **T0.2** (Config) - 4h
2. **T0.3** (Outputs) - 3h
3. **T1.1** (Data Verify) - 3h
4. **T1.2** (Data Optimize) - 4h
5. **T1.3** (Data Cache) - 5h
6. **T2.1** (Loss Analyze) - 2h
7. **T2.2** (Loss Weight) - 4h
8. **T2.3** (Leakage Check) - 2h
(Total: ~27h)

---

## 🚀 Quick Decision Tree

**Q: Where should I start?**
```
├─ "I want clean infrastructure first"
│  └─ START: T0.2 (Config System)
│
├─ "I want to verify things work"
│  └─ START: T1.1 (Data Verify)
│
├─ "I want to do research/analysis"
│  └─ START: T3.1 (Save Predictions)
│
└─ "I want everything done properly"
   └─ START: T0.2 → T0.3 → T1.1 (sequence)
```

**Q: How long will everything take?**
- With 3-4h/day: ~3-4 weeks
- With 4-5h/day: ~2-3 weeks
- With 8h/day: ~1-2 weeks

**Q: Can I skip some tasks?**
- T0.1: Yes, optional (cleanup)
- T0.2: NO (everything depends on it)
- T0.3: NO (needed for T3.x)
- T1.1-T1.3: NO (needed for T2.x, T3.x)
- T2.1-T2.3: NO (critical for fair evaluation)
- T3.x-T4.x: YES (if not doing research)
- T5.x: YES (if in hurry)

---

## 📞 Status of Each Task

| Task | Status | Ready? |
|------|--------|--------|
| T0.1 Cleanup | 📋 Design ready | ✅ Yes |
| T0.2 Config | 📋 Design ready | ✅ Ready for prompt |
| T0.3 Outputs | 📋 Design ready | ✅ Ready for prompt |
| T1.1 Verify | 📋 Design ready | ✅ Ready for prompt |
| T1.2 Optimize | 📋 Design ready | ✅ Ready for prompt |
| T1.3 Cache | 📋 Design ready | ✅ Ready for prompt |
| T2.1 Analyze | 📋 Design ready | ✅ Ready for prompt |
| T2.2 Weight | 📋 Design ready | ✅ Ready for prompt |
| T2.3 Validate | 📋 Design ready | ✅ Ready for prompt |
| T3.1 Predictions | 📋 Design ready | ✅ Ready for prompt |
| T3.2 Triplets | 📋 Design ready | ✅ Ready for prompt |
| T3.3 Heatmap | 📋 Design ready | ✅ Ready for prompt |
| T4.1 Aggregation | 📋 Design ready | ✅ Ready for prompt |
| T4.2 Analysis | 📋 Design ready | ✅ Ready for prompt |
| T4.3 Compare | 📋 Design ready | ✅ Ready for prompt |
| T5.1 Optimize | 📋 Design ready | ✅ Ready for prompt |
| T5.2 Multiclass | 📋 Design ready | ✅ Ready for prompt |
| T5.3 Metrics | 📋 Design ready | ✅ Ready for prompt |

**All tasks designed and documented. Ready to implement!**

---

## 💡 Pro Tips

1. **Start small**: Do T0.2 first (config system)
   - Most foundational
   - Affects everything else
   - Takes 4-6h, high value

2. **T1.1 is free validation**: 
   - Before spending time on T1.2/T1.3
   - Understanding the pipeline is key
   - Document it thoroughly

3. **Prof's requirement (T3.2-T3.3)**:
   - Don't skip! This is explicitly requested
   - Heatmap analysis is prof's visualization
   - Implement early for credibility

4. **T2 (Class Imbalance)**:
   - Critical! Don't rush
   - A6 stratified splits is good
   - But loss weighting is equally important
   - Data leakage is easy to miss

5. **Save work often**:
   - Each task should be reviewable
   - Create tests as you go
   - Document assumptions

---

## 🎯 Final Checklist Before Starting

- [ ] Read [COMPREHENSIVE_RETHINKING.md](COMPREHENSIVE_RETHINKING.md) (your detailed task Bible)
- [ ] Read [PRIORITY_SUMMARY.md](PRIORITY_SUMMARY.md) (visual overview)
- [ ] Decide which task to start with (T0.2 recommended)
- [ ] Ask me to create detailed implementation prompt for chosen task
- [ ] Ready to code!

**Next step: Pick a task and let's create its implementation prompt!**

