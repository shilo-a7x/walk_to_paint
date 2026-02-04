# 🎯 Comprehensive Project Rethinking & Task Reorganization

**Date**: February 4, 2026  
**Based on**: 14-point requirements review

---

## 📋 Part 1: Your 14 Requirements → Core Task Areas

### Group 1: Data Building & Reproducibility

- **Req 1**: Verify data building reproducibility
- **Req 2**: Optimize data building stages (merge/separate files)
- **Req 3**: Train-ready data loading, easy retraining without rebuild

### Group 2: Data Saving & Analysis

- **Req 5**: Save predictions (raw scores + labels)
- **Req 6**: Save aggregator info (walk ID, position, distance from start/end, label)
- **Req 13**: Analysis capability: triplets (dist_from_start, dist_from_end, correct/incorrect) → heatmap

### Group 3: Class Imbalance Strategy

- **Req 7**: Full rethink on loss weighting given stratified splits

### Group 4: System Architecture

- **Req 8**: Robust config system (verification, structure, organization)
- **Req 9**: Strict separation of outputs (data, predictions, checkpoints, logs, optuna)
- **Req 12**: Easy aggregator experimentation (save outputs, identify edge locations)

### Group 5: Code Robustness

- **Req 4**: Training optimization
- **Req 10**: Easy for current experiments
- **Req 11**: Binary focus + multiclass option
- **Req 14**: Keep previous relevant tasks (evaluation metrics, etc.)

---

## 🗂️ New Task Structure (Reorganized)

### PHASE 0: Foundation & Cleanup (Prerequisite)

- **T0.1**: Clean up old documentation (remove redundant MD files)
- **T0.2**: Standardize config system (robust, verified)
- **T0.3**: Restructure output directories (strict separation)

### PHASE 1: Data Pipeline (Core)

- **T1.1**: Verify data reproducibility + document stages
- **T1.2**: Optimize data stages (merge unnecessary file I/O)
- **T1.3**: Implement caching + train-ready loading

### PHASE 2: Class Imbalance Strategy (Critical)

- **T2.1**: Analyze current loss weighting
- **T2.2**: Implement class-weighted loss (per-batch or per-epoch)
- **T2.3**: Validate against data leakage

### PHASE 3: Prediction Saving & Analysis (Research Foundation)

- **T3.1**: Save model predictions (raw + labels) with walk metadata
- **T3.2**: Save aggregator triplets (dist_start, dist_end, correct flag)
- **T3.3**: Heatmap analysis tool for triplet data

### PHASE 4: Aggregator Experiments (Research)

- **T4.1**: Implement MLP/Logistic aggregation
- **T4.2**: Triplet-based analysis (prof's requirement)
- **T4.3**: Compare aggregation strategies with heatmap insights

### PHASE 5: Optimization & Polish (Final)

- **T5.1**: Training optimization
- **T5.2**: Binary/multiclass flexibility
- **T5.3**: Evaluation metrics (comprehensive)

---

## 🔴 Priority Ordering (What to Do First)

### CRITICAL PATH (Must Do First)

1. **T0.2 → T0.3** (Config + Outputs): Foundational for everything else
2. **T1.1 → T1.3** (Data Pipeline): Must have reproducible, optimized data
3. **T2.1 → T2.3** (Class Imbalance): Must fix before retraining
4. **T3.1 → T3.2** (Prediction Saving): Foundation for research

### HIGH PRIORITY (Do Next)

5. **T3.3** (Heatmap Tool): Needed for prof's analysis
2. **T4.1 → T4.2** (Aggregator): Core research direction

### MEDIUM PRIORITY (Do After)

7. **T4.3** (Compare strategies)
2. **T5.1 → T5.3** (Optimization & Polish)

---

## 📝 Detailed Task Definitions

### PHASE 0: Foundation & Cleanup

#### T0.1: Clean Up Old Documentation

**Status**: Ready  
**Time**: 30 minutes  
**Why**: Repo has 50+ MD files, most outdated

**Files to Delete** (old/redundant):

```
CHAT_A_CONFIG_REPRODUCIBILITY.md        (old, has _UPDATED version)
CHAT_A_CONFIG_REPRODUCIBILITY_UPDATED.md (old, replaced by T0.2)
CHAT_B_TEST_METRICS.md                  (old, merged into T5.3)
CHAT_C_PREDICTION_CACHING.md            (old, merged into T1.3)
CHAT_D_DATA_PIPELINE.md                 (old, has _UPDATED)
CHAT_D_DATA_PIPELINE_UPDATED.md         (old, merged into T1.x)
CHAT_E_SEED_CLEANUP.md                  (old, A1 done)
CHAT_E_SEED_CLEANUP_UPDATED.md          (old, A1 done)
CHAT_F_AGGREGATOR_INTEGRATION.md        (old, merged into T4.x)
COMEBACK_SUMMARY.md                     (old context)
CHAT_HISTORY_SUMMARY.md                 (old context)
PROJECT_INDEX.md                        (old index)
COMPLETE_ANALYSIS.md                    (old analysis)
FINAL_SUMMARY.md                        (old summary)
PROMPT_UPDATES_SUMMARY.md               (old)
QUICK_REFERENCE.md                      (old)
FILES_SUMMARY.md                        (old)
IMPLEMENTATION_CHECKLIST.md             (old)
GET_BACK_TO_WORK_PLAN.md                (old)
INDEX.md                                (old)
VISUAL_SUMMARY.md                       (old)
ARCHITECTURE.md                         (old)
WIKI_RFA_EVALUATION.md                  (old)
WIKI_RFA_EVAL_REPORT.md                 (old)
CHECKPOINT_LOADING_FIX.md               (old, merged)
DATA_PIPELINE_ANALYSIS.md               (old, merged into T1.x)
DATA_OPTIMIZATION_ANALYSIS.md           (old, merged into T1.x)
A1_IMPACT_ANALYSIS.md                   (old, A1 done)
TASK_A1_REPRODUCIBILITY_FIXES.md        (old, A1 done)
TASK_A1_IMPLEMENTATION_SUMMARY.md       (old, A1 done)
TASK_A1_FINAL_STATUS.md                 (old, A1 done)
AGGREGATOR_CONFIG.md                    (old, merged)
OPTIMIZATION_GUIDE.md                   (old, merged into T5.x)
EDGE_AGGREGATION_GUIDE.md               (old, merged into T4.x)
REPRODUCIBILITY_REVIEW.md               (old)
TASK_A6_PROMPT.md                       (old, A6 done)
TASK_A6_SPLIT_SEMANTICS.md              (old, A6 done)
TASK_A6_SPLIT_SEMANTICS_EXPLAINED.md    (old, A6 done)
TASK_A6_VERIFICATION_CHECKLIST.md       (old, A6 done)
TASK_A6_STRATIFIED_SPLITTING_IMPLEMENTATION.md (old, A6 done)
TASK_A6_COMPLETE.md                     (old, A6 done)
TASK_A6_IMPLEMENTATION_SUMMARY.md       (old, A6 done)
WALK_REPRODUCIBILITY_EXPLAINED.md       (keep - A1 reference)
WALK_SOLUTION_COMPLETE.md               (keep - A1 reference)
CONFIG_GUIDE.md                         (keep but update for T0.2)
FULL_PROJECT_STATUS.md                  (old, will replace)
```

**Files to Keep**:

```
README.md                      (main entry point)
WALK_REPRODUCIBILITY_EXPLAINED.md (A1 reference)
WALK_SOLUTION_COMPLETE.md      (A1 reference)
CONFIG_GUIDE.md                (update for T0.2)
(new files will be created for T0.2, T1.1, etc.)
```

**Deliverable**: Clean repo with only essential docs

---

#### T0.2: Standardize Config System

**Status**: Design needed  
**Time**: 4-6 hours  
**Dependencies**: None  
**Priority**: 🔴 CRITICAL (blocks everything)

**Current Problems**:

- `cfg.dataset.name`, `cfg.model.hidden_dim`, etc. scattered throughout code
- No schema validation - missing fields silently fail
- Config loading ad-hoc (base → override → merge)
- Dataset-specific configs mixed with base config

**Requirements**:

1. **Config Schema**: Define and validate all expected fields
   - Data schema: dataset fields, preprocessing options, cache settings
   - Model schema: architecture, dropout, etc.
   - Training schema: learning rate, batch size, etc.
   - Evaluation schema: metrics, thresholds, etc.

2. **Config Loading Pipeline**:
   - Load base config (defaults)
   - Load dataset-specific overrides from `configs/<dataset>.yaml`
   - Validate against schema
   - Provide clear errors if fields missing

3. **Config Access Utilities**:
   - `get_config()` - returns validated config object
   - `validate_config(cfg)` - check completeness
   - Type hints throughout

4. **Documentation**:
   - Config schema explanation
   - Example configs for each dataset
   - Troubleshooting guide

**Implementation**:

- Create `src/config_schema.py` - Schema definitions + validation
- Update `src/utils/config.py` - Loading pipeline
- Create `configs/schema.yaml` - Config structure documentation
- Create `configs/defaults.yaml` - All defaults
- Create `configs/<dataset>.yaml` - Per-dataset overrides

**Success Criteria**:

- ✅ Cannot run with missing config fields (clear error)
- ✅ Can easily add new config option (update schema only)
- ✅ All datasets use same loading mechanism
- ✅ Type hints prevent cfg.typo errors
- ✅ Documentation clear

---

#### T0.3: Restructure Output Directory System

**Status**: Design needed  
**Time**: 3-4 hours  
**Dependencies**: T0.2  
**Priority**: 🔴 CRITICAL

**Current Problem**:

```
outputs/
├── bitcoin-alpha-binary/optuna/
├── wiki-rfa/optuna/
└── (unclear structure, mixed content types)
```

**Desired Structure**:

```
outputs/
├── data/                         # Data building outputs
│   ├── wiki-rfa/
│   │   ├── preprocessed/
│   │   ├── walks/
│   │   ├── splits.json           # train/mask/val/test edge IDs
│   │   └── metadata.json         # reproducibility info
│   ├── epinions/
│   └── slashdot090221/
│
├── models/                       # Trained models & checkpoints
│   ├── wiki-rfa/
│   │   ├── optuna_study_20260204.log
│   │   ├── trial_123/            # Single trial
│   │   │   ├── best.ckpt         # Best checkpoint
│   │   │   ├── hparams.yaml      # Trial hyperparams
│   │   │   └── logs/             # TensorBoard
│   │   └── trial_456/
│   ├── epinions/
│   └── slashdot090221/
│
├── predictions/                  # Model predictions (for analysis)
│   ├── wiki-rfa/
│   │   ├── raw_scores/           # Raw model outputs
│   │   │   ├── trial_123_train.pkl
│   │   │   ├── trial_123_val.pkl
│   │   │   └── trial_123_test.pkl
│   │   ├── labels/               # Predicted labels
│   │   ├── metadata/             # Walk info, positions, etc.
│   │   └── analysis/             # Heatmaps, plots, triplets
│   ├── epinions/
│   └── slashdot090221/
│
├── aggregation/                  # Aggregator experiments
│   ├── wiki-rfa/
│   │   ├── strategy_mean/
│   │   │   ├── triplets.pkl
│   │   │   ├── heatmap.png
│   │   │   └── results.json
│   │   ├── strategy_logistic/
│   │   └── strategy_mlp/
│   ├── epinions/
│   └── slashdot090221/
│
└── optuna/                       # Optuna studies (global)
    ├── wiki-rfa-study.log
    ├── epinions-study.log
    └── slashdot090221-study.log
```

**Implementation**:

- Create utility: `utils/output_paths.py` - Manage output directory structure
- Create utility: `utils/output_manager.py` - Handle saving with proper organization
- Update config: Add output paths to schema

**Success Criteria**:

- ✅ Clear separation of concerns
- ✅ Easy to find any output type
- ✅ Extensible for new output types
- ✅ All downstream code uses output manager

---

### PHASE 1: Data Pipeline (Core)

#### T1.1: Verify Data Reproducibility & Document Stages

**Status**: Analysis needed  
**Time**: 3-4 hours  
**Dependencies**: T0.2, T0.3  
**Priority**: 🔴 CRITICAL

**Current Implementation**: A1 already did per-walk seeding  
**Your Requirement**: Verify it works + document each stage

**Tasks**:

1. Audit data building pipeline:
   - Graph loading
   - Edge stratified splitting
   - Walk sampling
   - Feature engineering
   - File saving

2. For each stage, answer:
   - Is this deterministic? (Given seed, same output)
   - Where is randomness? (List all RNG calls)
   - File dependency? (Must be separate or can merge?)
   - Performance? (How long?)
   - Memory? (What's loaded?)

3. Create reproducibility verification:
   - Run with seed=42 → save all intermediates
   - Run with seed=42 again → verify bit-for-bit identical
   - Run with seed=123 → verify different output
   - Document findings

4. Document findings in: `docs/DATA_BUILDING_PIPELINE.md`
   - Diagram showing each stage
   - Reproducibility status of each
   - File I/O strategy
   - Recommendations

**Deliverable**:

- Verification report
- `docs/DATA_BUILDING_PIPELINE.md`
- Updated data building code with comments

---

#### T1.2: Optimize Data Stages (Merge/Separate Files)

**Status**: Implementation needed  
**Time**: 4-5 hours  
**Dependencies**: T1.1  
**Priority**: 🔴 CRITICAL

**Goal**: Minimize file I/O while maintaining ability to reuse stages

**Analysis from T1.1 will reveal**:

- Which stages must be separate files (needed by different code)
- Which stages can be merged (only sequential access)
- Where data can be streamed vs. loaded entirely

**Likely Recommendations**:

```
Current (probably inefficient):
1. Raw graph → graph.pkl
2. Graph → edges → edges.pkl
3. Edges → stratified split → train.pkl, mask.pkl, val.pkl, test.pkl
4. Each split → walks → walks_<split>.pkl
5. Walks + graph → features → features_<split>.pkl

Optimized (likely):
1. Raw graph → preprocessed (one file or memory-mapped)
2. Edges → stratified split → splits.json (IDs only, <1MB)
3. Walks generated on-the-fly from graph + split IDs (no file saved)
4. Features cached after first pass → features_<split>.pkl
   OR: Features generated on-the-fly during DataLoader

Decision framework:
├─ Must save separately if used by different processes
├─ Can merge if sequential
├─ Can skip if computed fast enough
└─ Stream from disk if too large for memory
```

**Deliverable**:

- Updated `prepare_data.py` with optimized pipeline
- Benchmark: before vs. after (file I/O time, disk usage, memory)

---

#### T1.3: Implement Caching + Train-Ready Loading

**Status**: Implementation needed  
**Time**: 5-6 hours  
**Dependencies**: T1.2  
**Priority**: 🔴 CRITICAL

**Requirement**:

- First run: build all data, takes time
- Subsequent runs: load from cache, fast
- Easy to retrain without rebuilding

**Implementation**:

1. **Cache System**:
   - Check if cached data exists (per dataset + config hash)
   - If exists: load from cache
   - If not: build + save to cache

2. **Config Hash**:
   - Hash of config fields that affect data (graph params, walk params, etc.)
   - If config changes → invalidate cache

3. **DataLoader Interface**:

   ```python
   # New interface
   dataloader = get_dataloader(cfg, split='train', use_cache=True)
   # Returns: already loaded, ready for training
   # No need for prepare_data.py calls
   ```

4. **Caching Options**:

   ```yaml
   data:
     cache_enabled: true
     cache_dir: ./outputs/data/
     cache_invalidation: config_hash
   ```

5. **Memory Management**:
   - Option to stream large datasets from disk
   - Option to preload small datasets to memory
   - Profile memory usage

**Deliverable**:

- Updated DataLoader with caching
- Benchmarks: cold start vs. warm cache
- Documentation: how to clear cache, invalidation strategy

---

### PHASE 2: Class Imbalance Strategy (Critical)

#### T2.1: Analyze Current Loss Weighting

**Status**: Analysis needed  
**Time**: 2-3 hours  
**Dependencies**: T0.2  
**Priority**: 🔴 CRITICAL

**Questions**:

- How is loss currently computed? (raw cross-entropy? weighted?)
- If weighted, how? (class weights? sample weights?)
- How are weights computed? (per-batch frequency? per-epoch? global?)
- Is there data leakage? (Val loss should not affect train weights)

**Tasks**:

1. Audit `src/model/lit_model.py`:
   - Find loss computation
   - Check for class weighting
   - Check for batch-level effects

2. Test current behavior:
   - Train on 80% positive edges → does model overfit positive?
   - Train on 20% negative edges → does model overfit negative?
   - Current loss weights if any → what's the formula?

3. Analyze stratified splits:
   - A6 ensures splits have same class distribution
   - But within each split, what's the distribution?
   - Is there batch-level imbalance?

4. Document findings: `docs/CLASS_IMBALANCE_ANALYSIS.md`

**Deliverable**:

- Analysis report
- Code comments explaining current approach
- Recommendations for T2.2

---

#### T2.2: Implement Class-Weighted Loss

**Status**: Implementation needed  
**Time**: 4-5 hours  
**Dependencies**: T2.1  
**Priority**: 🔴 CRITICAL

**Strategy**:

- Compute class weights from training split only
- Weight loss: `loss = weighted_cross_entropy(pred, label, class_weights)`
- Do NOT use validation/test splits for weight computation

**Implementation Options**:

1. **Global Weighting** (per dataset):

   ```python
   # Computed once from train split
   n_pos = count(label == 1 in train split)
   n_neg = count(label == 0 in train split)
   weight_pos = n_neg / (n_pos + n_neg)
   weight_neg = n_pos / (n_pos + n_neg)
   # Use same weights throughout training
   ```

2. **Per-Epoch Weighting** (from current batch):

   ```python
   # Recompute weights each epoch
   pos_count = count(label == 1 in current epoch)
   neg_count = count(label == 0 in current epoch)
   # Dynamically adjust weights
   ```

3. **Per-Batch Weighting** (from current batch):

   ```python
   # Recompute for every batch
   # Higher variance, might help or hurt
   ```

**Recommendation**: Start with global weighting (safest)

**Config**:

```yaml
training:
  loss_weighting: 'global'  # or 'per_epoch', 'per_batch'
  weight_computation_split: 'train'  # Use train split only
```

**Deliverable**:

- Updated loss function with class weighting
- Benchmarks: with vs. without weighting (metrics on val/test)
- Documentation: loss weighting strategy

---

#### T2.3: Validate Against Data Leakage

**Status**: Validation needed  
**Time**: 2-3 hours  
**Dependencies**: T2.2  
**Priority**: 🔴 CRITICAL

**Potential Leakage Points**:

1. Using val/test split to compute class weights → LEAKAGE
2. Using val/test edges in loss weighting → LEAKAGE
3. Computing weights from batches that mix train/val → LEAKAGE

**Validation Tests**:

1. **Weights Only from Train Split**:
   - Verify class weights computed only from train split edges
   - Check that val/test splits never affect weight computation

2. **Batch Composition**:
   - Verify DataLoader doesn't mix splits in single batch
   - Each batch should have only one split

3. **Loss Computation**:
   - Trace loss backward → verify no val/test gradients flow to weights
   - Manual check: compute loss on val split → should not affect next training step

**Deliverable**:

- Leakage validation tests (pytest)
- Documentation: data leakage prevention strategy
- Confidence report: "No leakage detected"

---

### PHASE 3: Prediction Saving & Analysis

#### T3.1: Save Model Predictions (Raw + Labels)

**Status**: Implementation needed  
**Time**: 3-4 hours  
**Dependencies**: T0.3, T1.3  
**Priority**: 🟠 HIGH

**What to Save**:

```
For each edge:
├─ Raw prediction score (0.0-1.0)
├─ Predicted label (0 or 1)
├─ True label (ground truth)
├─ Confidence (max logit)
└─ Metadata:
   ├─ Walk ID
   ├─ Position in walk (0, 1, 2, ...)
   ├─ Distance from start (position / walk_length)
   ├─ Distance from end ((walk_length - position) / walk_length)
   └─ Walk length
```

**Format**: Pickle (per split)

```python
predictions = {
    'edge_ids': [(u1, v1), (u2, v2), ...],  # Edge as tuple
    'scores': [0.92, 0.45, ...],            # Raw scores
    'labels': [1, 0, ...],                  # Predicted labels
    'true_labels': [1, 0, ...],             # Ground truth
    'walk_ids': [123, 456, ...],            # Which walk
    'positions': [0, 2, ...],               # Position in walk
    'distances_from_start': [0.0, 0.4, ...],
    'distances_from_end': [1.0, 0.6, ...],
    'walk_lengths': [5, 10, ...],
    'confidences': [0.92, 0.55, ...],
    'split': 'test',
    'dataset': 'wiki-rfa',
    'trial_id': 87,
    'timestamp': '2026-02-04T10:00:00',
}
```

**Implementation**:

- Update `evaluation_pipeline.py` to save predictions
- Create `utils/prediction_saver.py` - Handle pickle I/O with metadata

**Output Path** (using T0.3):

```
outputs/predictions/<dataset>/raw_scores/trial_<id>_<split>.pkl
outputs/predictions/<dataset>/metadata/trial_<id>_<split>.pkl
```

**Deliverable**:

- Updated evaluation pipeline
- Verified pickle files contain all required fields

---

#### T3.2: Save Aggregator Triplets

**Status**: Implementation needed  
**Time**: 3-4 hours  
**Dependencies**: T3.1  
**Priority**: 🟠 HIGH

**What to Save** (Prof's requirement):

```
For each UNIQUE edge (after aggregation):
├─ Triplet:
│  ├─ distance_from_start (average across all walks)
│  ├─ distance_from_end (average across all walks)
│  └─ correct (1 if predicted==true, 0 otherwise)
└─ Metadata:
   ├─ edge_id
   ├─ num_occurrences (how many walks had this edge)
   ├─ predictions (all raw scores from all walks)
   ├─ walk_ids (which walks had this edge)
   └─ positions (positions in each walk)
```

**Format**: Pickle (aggregated per split)

```python
triplets = {
    'edges': [(u1, v1), (u2, v2), ...],
    'distances_from_start': [0.2, 0.5, ...],  # avg
    'distances_from_end': [0.8, 0.5, ...],    # avg
    'correct_flags': [1, 0, ...],             # correct/incorrect
    'num_occurrences': [3, 5, ...],           # how many walks
    'raw_predictions': [[0.9, 0.85, 0.95], [0.2, 0.3], ...],
    'walk_ids': [[123, 456, 789], [111, 222], ...],
    'positions_in_walks': [[0, 2, 4], [1, 3], ...],
}
```

**Implementation**:

- Create `utils/aggregator.py` - Base aggregator class
- Implement aggregation strategies (Mean, Majority, etc.)
- Save triplets after aggregation

**Output Path**:

```
outputs/aggregation/<dataset>/strategy_<name>/triplets_<split>.pkl
```

**Deliverable**:

- Triplet generation and saving
- Verified pickle files contain all required fields

---

#### T3.3: Heatmap Analysis Tool

**Status**: Implementation needed  
**Time**: 3-4 hours  
**Dependencies**: T3.2  
**Priority**: 🟠 HIGH

**What Prof Wants**:

```
"Produce triplets (distance from start, distance from end, 0/1 flag of correct)
Then plot the average of the 0/1 flag as a function of the two first and plot a heatmap"
```

**Interpretation**:

- X-axis: distance from start (0 to 1)
- Y-axis: distance from end (0 to 1)
- Color: average correct flag (0 to 1)
- Result: Heatmap showing where model performs well/poorly

**Implementation**:

```python
def create_triplet_heatmap(triplets, bins=10):
    """
    Create 2D heatmap: (dist_from_start, dist_from_end) → avg_correct
    
    Args:
        triplets: dict with distances_from_start, distances_from_end, correct_flags
        bins: number of bins for each axis (default 10x10 grid)
    
    Returns:
        heatmap (bins x bins), xaxis, yaxis
    """
    # Bin the distances
    start_bins = np.digitize(triplets['distances_from_start'], 
                             np.linspace(0, 1, bins+1))
    end_bins = np.digitize(triplets['distances_from_end'], 
                           np.linspace(0, 1, bins+1))
    
    # Create 2D grid
    heatmap = np.zeros((bins, bins))
    counts = np.zeros((bins, bins))
    
    # Aggregate correct flags by bin
    for i, (start_bin, end_bin, correct) in enumerate(
        zip(start_bins, end_bins, triplets['correct_flags'])):
        heatmap[end_bin-1, start_bin-1] += correct
        counts[end_bin-1, start_bin-1] += 1
    
    # Average
    heatmap = np.divide(heatmap, counts, where=counts>0)
    
    return heatmap

# Usage
heatmap = create_triplet_heatmap(triplets, bins=10)
plt.imshow(heatmap, cmap='RdYlGn', origin='lower', aspect='auto')
plt.xlabel('Distance from Start')
plt.ylabel('Distance from End')
plt.colorbar(label='Avg Correct')
plt.savefig('heatmap.png')
```

**Deliverable**:

- `utils/heatmap_analysis.py` - Heatmap generation
- Heatmap plots for each dataset/strategy/split
- Saved to: `outputs/predictions/<dataset>/analysis/heatmap_<split>.png`

---

### PHASE 4: Aggregator Experiments

#### T4.1: Implement MLP/Logistic Aggregation

**Status**: Implementation needed  
**Time**: 5-6 hours  
**Dependencies**: T3.2  
**Priority**: 🟠 HIGH

**Strategies**:

1. **Mean**: Average of all predictions
2. **Logistic**: Learned logistic regression on aggregation features
3. **MLP**: Multi-layer perceptron on aggregation features

**Features for Learning**:

```
For each edge, aggregate features from all walks:
├─ mean_score
├─ std_score
├─ min_score
├─ max_score
├─ median_score
├─ num_walks
├─ mean_dist_from_start
├─ mean_dist_from_end
├─ dist_from_start_std
├─ dist_from_end_std
└─ Other position-based features
```

**Implementation**:

- Create `aggregators/mean.py` - Simple mean
- Create `aggregators/logistic.py` - Learned logistic
- Create `aggregators/mlp.py` - Learned MLP
- Train on train split, evaluate on val/test

**Config**:

```yaml
aggregator:
  strategy: 'mlp'  # 'mean', 'logistic', 'mlp'
  mlp_hidden_dims: [64, 32]
  mlp_dropout: 0.2
```

**Deliverable**:

- Aggregator implementations
- Trained models for each strategy
- Comparison metrics (val/test AUC)

---

#### T4.2: Triplet-Based Analysis & Heatmaps

**Status**: Implementation needed  
**Time**: 3-4 hours  
**Dependencies**: T3.3, T4.1  
**Priority**: 🟠 HIGH

**Analysis**:

- Generate triplet heatmaps for each aggregation strategy
- Compare heatmaps visually
- Identify patterns: where does model do well/poorly?
- Correlate with walk properties

**Deliverable**:

- Heatmaps for all strategies
- Analysis report: insights from heatmaps
- Visualization comparing strategies

---

#### T4.3: Compare Aggregation Strategies

**Status**: Implementation needed  
**Time**: 4-5 hours  
**Dependencies**: T4.1, T4.2  
**Priority**: 🟠 MEDIUM

**Comparison**:

- Mean vs. Majority vs. Weighted Mean vs. Logistic vs. MLP
- Metrics: AUC, F1, Precision, Recall
- Speed: inference time per edge
- Interpretability: which is easiest to understand?

**Report**: `docs/AGGREGATION_COMPARISON.md`

**Deliverable**:

- Comparison table (metrics × strategies)
- Best strategy recommendation
- Analysis of trade-offs

---

### PHASE 5: Optimization & Polish

#### T5.1: Training Optimization

**Status**: Design needed  
**Time**: 4-5 hours  
**Dependencies**: T1.3  
**Priority**: 🟡 MEDIUM

**Areas to Optimize**:

1. DataLoader: prefetch, pin_memory, num_workers
2. Model: mixed precision, gradient checkpointing
3. Training: learning rate scheduling, early stopping
4. Hardware: GPU utilization, batch size tuning

**Deliverable**:

- Optimization report
- Updated training loop
- Benchmarks: before vs. after (time, memory, metrics)

---

#### T5.2: Binary/Multiclass Flexibility

**Status**: Implementation needed  
**Time**: 2-3 hours  
**Dependencies**: T0.2  
**Priority**: 🟡 MEDIUM

**Changes**:

- Add `dataset.task: 'binary'` or `'multiclass'` to config
- Update loss function: use `BCEWithLogitsLoss` for binary, `CrossEntropyLoss` for multiclass
- Update metrics: handle multiple classes
- Update aggregator: handle class probabilities

**Deliverable**:

- Binary/multiclass support throughout codebase
- Config option to switch
- Documentation: how to use multiclass mode

---

#### T5.3: Comprehensive Evaluation Metrics

**Status**: Implementation needed  
**Time**: 4-5 hours  
**Dependencies**: T0.2, T1.3  
**Priority**: 🟡 MEDIUM

**Metrics to Implement**:

- **Classification**: Precision, Recall, F1, Specificity, Sensitivity
- **ROC/PR**: AUC-ROC, AUC-PR, ROC curves, Precision-Recall curves
- **Class-wise**: Per-class metrics (important for imbalanced)
- **Threshold analysis**: Vary threshold, plot metrics
- **Confusion matrix**: Visualize errors

**Format**:

- CSV export: all metrics × splits × trials
- Plots: ROC, PR, confusion matrices
- JSON: raw results for downstream analysis

**Deliverable**:

- `utils/metrics.py` - All metric computations
- Evaluation report with all metrics
- Visualization suite

---

## 📋 Summary: New Task Ordering by Priority

### CRITICAL PATH (Do First - ~2 weeks)

1. **T0.2** (Config System): 4-6h - Foundation
2. **T0.3** (Output Dirs): 3-4h - Organization
3. **T1.1** (Data Verification): 3-4h - Understand pipeline
4. **T1.2** (Data Optimization): 4-5h - Optimize I/O
5. **T1.3** (Data Caching): 5-6h - Easy retraining
6. **T2.1** (Loss Analysis): 2-3h - Understand current
7. **T2.2** (Loss Weighting): 4-5h - Implement weights
8. **T2.3** (Leakage Check): 2-3h - Validate

### HIGH PRIORITY (Next - ~1 week)

9. **T3.1** (Save Predictions): 3-4h - Foundation for analysis
2. **T3.2** (Save Triplets): 3-4h - Prof's requirement
3. **T3.3** (Heatmap Tool): 3-4h - Visualization
4. **T4.1** (MLP/Logistic): 5-6h - Learned aggregation
5. **T4.2** (Heatmap Analysis): 3-4h - Insights

### MEDIUM PRIORITY (Finish - ~1 week)

14. **T4.3** (Compare Strategies): 4-5h - Choose best
2. **T5.1** (Training Optimization): 4-5h - Performance
3. **T5.2** (Binary/Multiclass): 2-3h - Flexibility
4. **T5.3** (Eval Metrics): 4-5h - Comprehensive

**Total**: ~70-85 hours (~2-3 weeks of full-time work)

---

## ✅ Next Step

Pick one and we'll create a detailed implementation prompt:

- [ ] T0.1 (Cleanup) - 30 min
- [ ] T0.2 (Config) - 4-6h **← RECOMMEND HERE**
- [ ] T0.3 (Outputs) - 3-4h

Which one should we start with?
