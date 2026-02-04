# ⚡ Quick Start - What To Do First

## 🎯 TL;DR - 30 Second Version

You've done **substantial work** across 5 chat sessions:
- ✅ **A1 Config System**: Complete (unified `reproducibility.seed`)
- ✅ **Walk Reproducibility**: Complete (deterministic seeding verified)
- ✅ **DataLoader Tuning**: Complete (all settings validated)
- ✅ **Repo Design**: Complete (multi-dataset structure designed)
- 🚫 **Stratified Splitting**: Identified but NOT done

**Next**: 
1. **Verify A1 works** (30 min)
2. **Implement stratified splitting** (2-3 hours)
3. **Test everything** (2 hours)

**Estimated time to "fully working": 6-8 hours**

---

## 🚀 Step-by-Step: What To Do RIGHT NOW

### Step 1: Read the Summary (5 minutes)
- [ ] Read this file (you're doing it!)
- [ ] Skim `CHAT_HISTORY_SUMMARY.md` for visual overview
- [ ] Read `WALK_REPRODUCIBILITY_EXPLAINED.md` for understanding

### Step 2: Verify A1 Implementation (20 minutes)
```bash
# Check that key files have the changes
grep -n "reproducibility.seed" config.yaml  # Should exist
grep -n "get_seed" src/utils/config.py      # Should exist
grep -n "def get_seed" src/utils/config.py  # Should exist function

# Run a quick reproducibility test
python run.py --config config.yaml dataset.name=toy seed=42 2>&1 | grep -i "seed\|walk"
# Should see seed initialization messages
```

### Step 3: Run Reproducibility Test (10 minutes)
```bash
# Run 1
cd /home/dsi/shilo_avital/yolo_lab/walk_to_paint
python run.py --config config.yaml dataset.name=toy seed=42 max_epochs=1
md5sum data/toy/walks.pkl > run1.md5

# Run 2 (same seed)
rm -rf data/toy/walks.pkl
python run.py --config config.yaml dataset.name=toy seed=42 max_epochs=1
md5sum data/toy/walks.pkl > run2.md5

# Compare
diff run1.md5 run2.md5
# If they match: ✅ A1 works!
```

### Step 4: Plan Next Steps (5 minutes)
Based on test result:
- **If test PASSES**: Proceed to "Stratified Splitting Implementation" below
- **If test FAILS**: Debug using `WALK_REPRODUCIBILITY_EXPLAINED.md`

---

## 📌 Current Status: What You Know

From your 5 chat sessions:

| Item | Status | Evidence |
|------|--------|----------|
| A1 Config unified | ✅ DONE | Unified to `reproducibility.seed` |
| Walk sampling seeding | ✅ DONE | Per-walk (base_seed + i) implemented |
| Sorting for multiprocessing | ✅ DONE | Sort by task_id verified safe |
| DataLoader tuning | ✅ ANALYZED | All settings validated good |
| Repo multi-dataset | ✅ DESIGNED | Structure documented |
| **Stratified splitting** | ❌ NOT DONE | Class imbalance identified |

---

## 🎯 Top 3 Priority Items

### Priority 1: Stratified Splitting (2-3 hours)
**Why**: Fair evaluation requires balanced class distributions

**What to do**:
```python
# In src/data/prepare_data.py, replace:
train, rest = train_test_split(edges, test_size=0.75, random_state=seed)

# With:
from sklearn.model_selection import train_test_split
labels = np.array([edge_label[e] for e in edges])
train, rest = train_test_split(
    edges, test_size=0.75, 
    stratify=labels,  # ← This line is key
    random_state=seed
)
# Then continue with mask/val/test splits similarly
```

**Test**:
```python
# Verify class balance in each split
for split_name, split_edges in [('train', train), ('mask', mask), ('val', val), ('test', test)]:
    split_labels = np.array([edge_label[e] for e in split_edges])
    pos_ratio = np.mean(split_labels)
    print(f"{split_name}: {pos_ratio:.2%}")
# All should be within ±1-2% of original ratio
```

### Priority 2: Full Reproducibility Test (1 hour)
**Why**: Confirm seed system works across all cases

**Test script** to run:
```bash
#!/bin/bash
# Test 1: Same seed produces identical walks
echo "Test 1: Reproducibility..."
rm -rf data/toy/walks.pkl
python run.py dataset.name=toy seed=42 max_epochs=1
md5sum data/toy/walks.pkl > check1.md5
rm -rf data/toy/walks.pkl
python run.py dataset.name=toy seed=42 max_epochs=1
md5sum data/toy/walks.pkl > check2.md5
diff check1.md5 check2.md5 && echo "✅ PASS" || echo "❌ FAIL"

# Test 2: Different seed produces different walks
echo "Test 2: Different seed..."
rm -rf data/toy/walks.pkl
python run.py dataset.name=toy seed=99 max_epochs=1
md5sum data/toy/walks.pkl > check3.md5
diff check1.md5 check3.md5 && echo "❌ FAIL (should differ)" || echo "✅ PASS"

# Test 3: Multi-worker safety
echo "Test 3: Worker count invariance..."
for workers in 2 4; do
    rm -rf data/toy/walks.pkl
    python run.py dataset.name=toy seed=42 max_epochs=1 training.num_workers=$workers
    md5sum data/toy/walks.pkl > check_w${workers}.md5
done
diff check_w2.md5 check_w4.md5 && echo "✅ PASS" || echo "❌ FAIL"
```

### Priority 3: Standalone Script Updates (1 hour)
**Why**: Prevent bugs when scripts run outside main pipeline

**Scripts to check**:
- `extract_edge_scores.py`
- `scripts/train_aggregator.py`
- `optuna_run.py`

**For each, add**:
```python
import sys
from src.utils.config import get_seed

# Load config
cfg = load_config()  # However you load it

# Initialize seeds
seed = get_seed(cfg)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Now rest of script...
```

---

## 📂 Key Files You Should Know

```
/home/dsi/shilo_avital/yolo_lab/walk_to_paint/
├── GET_BACK_TO_WORK_PLAN.md                      ← Full plan
├── CHAT_HISTORY_SUMMARY.md                       ← Visual summary
├── WALK_REPRODUCIBILITY_EXPLAINED.md             ← Deep dive
├── WALK_SOLUTION_COMPLETE.md                     ← A1 details
├── QUICK_REFERENCE.md                            ← Quick lookup
├── COMPLETE_ANALYSIS.md                          ← Comprehensive ref
├── CONFIG_GUIDE.md                               ← Config reference
├── IMPLEMENTATION_CHECKLIST.md                   ← Detailed checklist
│
├── config.yaml                                   ← Main config (HAS: reproducibility.seed)
├── configs/
│   ├── bitcoin-alpha-binary.yaml
│   ├── wiki-rfa.yaml
│   ├── epinions.yaml
│   └── slashdot090221.yaml
│
├── src/
│   ├── utils/config.py                           ← HAS: get_seed() function
│   ├── data/
│   │   ├── prepare_data.py                       ← UPDATE: stratified splitting
│   │   └── walk_sampler.py                       ← HAS: per-walk seeding
│   └── model/
│       └── lit_model.py
├── run.py                                        ← HAS: seed initialization
├── optuna_run.py                                 ← CHECK: seed initialization
├── extract_edge_scores.py                        ← CHECK: seed initialization
└── scripts/
    └── train_aggregator.py                       ← CHECK: seed initialization
```

---

## 🎬 Immediate Action Plan

### Today (1-2 hours):
```
1. [ ] Read CHAT_HISTORY_SUMMARY.md (15 min)
2. [ ] Skim WALK_REPRODUCIBILITY_EXPLAINED.md (15 min)
3. [ ] Run reproducibility test (20 min)
4. [ ] Check if A1 implementation is complete (10 min)
```

### Tomorrow (3-4 hours):
```
1. [ ] Implement stratified splitting (2 hours)
2. [ ] Write test to verify class balance (30 min)
3. [ ] Run on all datasets (1 hour)
```

### Day 3 (2-3 hours):
```
1. [ ] Update standalone scripts (1 hour)
2. [ ] Run full reproducibility test suite (1 hour)
3. [ ] Archive old results if switching datasets (30 min)
```

---

## 🐛 Troubleshooting: If Things Go Wrong

### Issue: Reproducibility test fails (walks differ)
**Possible causes**:
1. A1 not fully implemented - Check all files for `get_seed()` usage
2. Different seed being used - Check config loading
3. Non-deterministic operation - Search for `random.` without seed setup

**Debug steps**:
```bash
# Check if get_seed is being called
grep -rn "get_seed" src/
# Should see calls in prepare_data.py, run.py

# Check if seeds are being set
grep -n "torch.manual_seed\|np.random.seed\|random.seed" run.py
# Should see all three

# Check walk_sampler.py has per-walk seeding
grep -A5 "for.*walk" src/data/walk_sampler.py
# Should see seed = base_seed + walk_idx
```

### Issue: Stratified splitting fails
**Possible cause**: Labels not properly extracted

**Debug**:
```python
# In prepare_data.py, add debug output
labels = np.array([edge_label[e] for e in edges])
print(f"Label shape: {labels.shape}, unique: {np.unique(labels)}")
print(f"Positive ratio: {np.mean(labels):.2%}")
```

### Issue: Different worker counts produce different results
**Possible cause**: Missing sort by task_id

**Check**:
```python
# In walk_sampler.py, look for:
results.sort(key=lambda x: x[0])  # Sort by task_id
# If not there, add it before concatenating
```

---

## 📊 Success Criteria

### A1 Complete ✅
- [ ] `reproducibility.seed` in `config.yaml`
- [ ] `get_seed()` function in `src/utils/config.py`
- [ ] Reproducibility test passes (same seed → same walks)
- [ ] Different seeds → different walks

### Stratified Splitting Complete ✅
- [ ] Class distribution in train/mask/val/test within ±1-2%
- [ ] Works on all datasets
- [ ] Labels properly extracted
- [ ] Test cases created

### Full Pipeline Works ✅
- [ ] Run 2x with same seed → identical results
- [ ] Different workers → identical results
- [ ] All standalone scripts initialize seeds
- [ ] All test cases pass

---

## 💡 Pro Tips

1. **Use small dataset for testing**: `dataset.name=toy` runs fast
2. **Check MD5 sums**: `md5sum <file>` for quick reproducibility check
3. **Save timestamps**: Keep logs of runs for reference
4. **Test incrementally**: Test each change before moving to next
5. **Archive results**: Before switching datasets, archive old outputs

---

## ❓ Quick Reference: What Each File Does

| File | Purpose | Read If... |
|------|---------|-----------|
| GET_BACK_TO_WORK_PLAN.md | Full implementation plan | You want detailed steps |
| CHAT_HISTORY_SUMMARY.md | Visual overview of all work | You want big picture |
| WALK_REPRODUCIBILITY_EXPLAINED.md | Deep dive into seeding | You want to understand walks |
| QUICK_REFERENCE.md | Facts and code snippets | You need quick lookup |
| CONFIG_GUIDE.md | Config system explanation | You're confused about config |
| WALK_SOLUTION_COMPLETE.md | A1 completion details | You're debugging A1 |
| COMPLETE_ANALYSIS.md | Comprehensive reference | You want everything in one place |
| IMPLEMENTATION_CHECKLIST.md | Detailed checklist | You like structured checklists |

---

## 🏁 Where To Start

**If you have 30 minutes**: Read this file + skim CHAT_HISTORY_SUMMARY.md

**If you have 1 hour**: Read this + WALK_REPRODUCIBILITY_EXPLAINED.md

**If you have 2 hours**: Read this + run reproducibility test + review QUICK_REFERENCE.md

**If you have a full day**: Do everything above + start Phase 1 (Stratified Splitting)

---

## 📞 Key Contacts/Files

- **Config issues**: See CONFIG_GUIDE.md
- **Seed issues**: See WALK_REPRODUCIBILITY_EXPLAINED.md
- **Reproducibility test**: See QUICK_REFERENCE.md (Test section)
- **Stratified splitting**: See IMPLEMENTATION_CHECKLIST.md (Phase 2)
- **Standalone scripts**: See GET_BACK_TO_WORK_PLAN.md (Phase 3)

---

## ✨ Final Thought

You've done excellent foundational work across multiple dimensions. The main remaining items are:
1. Verify everything works (fast - 1 hour)
2. Implement stratified splitting (medium - 2-3 hours)
3. Test the full pipeline (fast - 2 hours)

**Total time to "fully production-ready": 6-8 hours**

You're in great shape. Let's go! 🚀

---

**Updated**: February 2, 2026  
**Status**: Ready to execute Phase 1  
**Time to complete**: 6-8 hours estimated
