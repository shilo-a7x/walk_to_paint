# Task 3 - Next Steps: Production Retraining & Post-Hoc Analysis

**Date**: February 8, 2026  
**Status**: POC Complete - Ready for Production

---

## Overview

Task 3 POC has been successfully completed and verified on binary toy dataset. Now we need to:

1. Retrain all 3 production datasets with optimized hyperparameters
2. Make post-hoc analysis scripts dataset-agnostic
3. Build edge-level aggregator from walk-level predictions

---

## Task 3.1: Production Dataset Retraining

### Objective

Retrain wiki-rfa, slashdot090221, and epinions with:

- Best Optuna hyperparameters
- Per-epoch prediction saving enabled
- Complete TensorBoard logging
- Reproducible seeds

### Datasets Configuration Summary

#### 1. wiki-rfa (Trial #87, AUC=0.7779)

```yaml
Dataset:
  - binary: true
  - remove_self_loops: true
  - multiedge_handling: most_recent
  - max_walk_length: 82
  - num_walks: 434,857

Model:
  - embedding_dim: 64
  - hidden_dim: 64
  - nhead: 2
  - nlayers: 5
  - dropout: 0.147

Training:
  - epochs: 20
  - batch_size: 512
  - lr: 0.001614
  - weight_decay: 2.987e-05
  - gradient_clip_val: 0.539
  - early_stopping_patience: 5
```

#### 2. slashdot090221 (Trial #12, AUC=0.8529)

```yaml
Dataset:
  - binary: false (3-class)
  - remove_self_loops: true
  - multiedge_handling: keep
  - max_walk_length: 76
  - num_walks: 4,899,286

Model:
  - embedding_dim: 128
  - hidden_dim: 256
  - nhead: 4
  - nlayers: 4
  - dropout: 0.00555

Training:
  - epochs: 7
  - batch_size: 512
  - lr: 0.0001237
  - weight_decay: 0.0004465
  - gradient_clip_val: 0.950
  - early_stopping_patience: 17
```

#### 3. epinions (Trial #31, AUC=0.9133)

```yaml
Dataset:
  - binary: false (3-class)
  - remove_self_loops: true
  - multiedge_handling: keep
  - max_walk_length: 79
  - num_walks: 4,966,522

Model:
  - embedding_dim: 32
  - hidden_dim: 32
  - nhead: 8
  - nlayers: 3
  - dropout: 0.000861

Training:
  - epochs: 25
  - batch_size: 256
  - lr: 0.002936
  - weight_decay: 0.000293
  - gradient_clip_val: 0.1072
  - early_stopping_patience: 9
```

### Implementation Requirements

#### Bash Script: `retrain_all_datasets_task3.sh`

**Features**:

- Run all 3 datasets in parallel (one per GPU device 0, 1, 2)
- Use existing config files: `configs/{dataset}.yaml`
- Set reproducibility seed: `reproducibility.seed=42`
- Enable preprocessing cache: `preprocess.use_cache=true`
- Disable eval-only mode (default is false)
- Save nohup logs per dataset

**Device Assignment**:

```bash
Device 0: wiki-rfa (20 epochs, binary, ~6-8 hours estimated)
Device 1: epinions (25 epochs, 3-class, ~8-10 hours estimated)
Device 2: slashdot090221 (7 epochs, 3-class, ~4-6 hours estimated)
```

**Command Template**:

```bash
python run.py \
  --config configs/{dataset}.yaml \
  --device {gpu_id} \
  reproducibility.seed=42 \
  preprocess.save=true \
  preprocess.use_cache=true
```

**Verification Checklist**:

- [ ] Configs have correct `binary` flag (wiki-rfa: true, others: false)
- [ ] Configs have correct `multiedge_handling` (wiki-rfa: most_recent, others: keep)
- [ ] Configs have correct `remove_self_loops` (all: true)
- [ ] Callbacks registered: PerEpochPredictionSaver + PerEpochTestRunner
- [ ] Class weights computed from train split
- [ ] TensorBoard logging enabled for all splits

**Expected Outputs (per dataset)**:

```
outputs/{dataset}/{exp_name}_{timestamp}/
├── checkpoints/
│   ├── {dataset}_predictions/
│   │   ├── epoch_000/
│   │   │   ├── train_predictions.pkl
│   │   │   ├── val_predictions.pkl
│   │   │   └── test_predictions.pkl
│   │   ├── epoch_001/...
│   │   └── epoch_{N}/...
│   └── {dataset}-{exp}-epoch={best}-val_loss={X.XX}.ckpt
└── logs/
    └── {dataset}-{exp}/
        └── version_0/
            └── events.out.tfevents.*
```

**Monitoring**:

```bash
# Watch training progress
tail -f outputs/{dataset}/retrain_device{N}_nohup.log

# Check TensorBoard
tensorboard --logdir outputs/{dataset}/{run}/logs/

# Check disk space (predictions can be large)
du -sh outputs/*/
```

---

## Task 3.2: Dataset-Agnostic Post-Hoc Analysis

### Objective

Make `test_posthoc_analysis.py` work with any dataset, not just toy.

### Current Limitations

- Hardcoded to `outputs/toy/` directory
- Assumes latest run in toy folder
- No command-line arguments

### Required Changes

#### Command-Line Interface

```bash
python test_posthoc_analysis.py \
  --dataset {dataset_name} \
  --run_dir {path_to_run} \
  [--epochs {comma_separated_list}] \
  [--splits train,val,test]
```

**Arguments**:

- `--dataset`: Dataset name (wiki-rfa, slashdot090221, epinions, toy)
- `--run_dir`: Path to specific run directory (e.g., `outputs/wiki-rfa/wiki-rfa-run_20260208-140000`)
- `--epochs` (optional): Which epochs to process (default: all)
- `--splits` (optional): Which splits to process (default: train,val,test)
- `--output_dir` (optional): Where to save post-hoc results (default: `{run_dir}/posthoc_analysis`)

#### Auto-Discovery Mode

If `--run_dir` not provided:

- Find latest run in `outputs/{dataset}/`
- Or use best checkpoint based on val_auc

#### Functionality

1. **Triplet Extraction**:
   - Load predictions from `{run_dir}/checkpoints/{dataset}_predictions/epoch_{N}/{split}_predictions.pkl`
   - Extract: edge_ids, walk_ids, positions, walk_lengths, dist_from_start, dist_from_end, correct
   - Save: `{output_dir}/epoch_{N}_{split}_triplets.pkl`

2. **Heatmap Generation**:
   - Create 2D heatmap: avg(correct) as function of (dist_from_start, dist_from_end)
   - Handle sparse grids for large datasets (e.g., max_walk_length=82)
   - Save: `{output_dir}/epoch_{N}_{split}_heatmap.png`

3. **Summary Report**:
   - Generate: `{output_dir}/posthoc_summary.txt`
   - Include:
     - Dataset name, run directory, epochs processed
     - Per-epoch, per-split statistics: AUC, accuracy, avg correctness
     - Heatmap statistics: grid size, filled cells, correctness range
     - File sizes

#### Multi-Class Support

- Works for both binary (wiki-rfa) and 3-class (slashdot, epinions)
- Correctness computed as `predictions == targets` (binary flag)

#### Output Structure

```
{run_dir}/posthoc_analysis/
├── epoch_000_train_triplets.pkl
├── epoch_000_train_heatmap.png
├── epoch_000_val_triplets.pkl
├── epoch_000_val_heatmap.png
├── epoch_000_test_triplets.pkl
├── epoch_000_test_heatmap.png
├── epoch_001_train_triplets.pkl
├── ...
└── posthoc_summary.txt
```

---

## Task 3.3: Edge-Level Aggregator Training

### Objective

Train a simple aggregator model that predicts edge labels from multiple walk-level predictions.

**Rationale**: Each edge appears in multiple walks at different positions. The transformer predicts sign/label at the token level (per occurrence). We want to aggregate these to get a single edge-level prediction.

### Problem Formulation

**Input**: For each edge `(u, v)`, we have K occurrences across different walks:

- `walk_lengths`: [L₁, L₂, ..., Lₖ]
- `positions`: [p₁, p₂, ..., pₖ]
- `predictions`: [ŷ₁, ŷ₂, ..., ŷₖ]
- `probabilities`: [P₁, P₂, ..., Pₖ] (shape K × num_classes)
- `targets`: [y, y, ..., y] (same ground truth for all occurrences)

**Derived Features** (per occurrence):

- `dist_from_start = position`
- `dist_from_end = walk_length - 1 - position`

**Goal**: Predict edge label `y` from occurrence-level features.

### Aggregator Architecture Options

#### Option 1: Simple MLP (Recommended for POC)

**Per-occurrence features** (2 features):

- dist_from_start (int)
- dist_from_end (int)

**Per-edge aggregation**:

- mean(dist_from_start), std(dist_from_start), min(dist_from_start), max(dist_from_start)
- mean(dist_from_end), std(dist_from_end), min(dist_from_end), max(dist_from_end)
- Total occurrences (count)
- → **9 features per edge**

**Model**:

- MLP with 1-2 hidden layers (e.g., 9 → 16 → num_classes)
- Activation: ReLU
- Output: Softmax (for 3-class) or Sigmoid (for binary)
- Loss: CrossEntropyLoss with class weights

#### Option 2: Logistic Regression (Baseline)

**Same features as Option 1** (9 features per edge)

**Model**:

- sklearn LogisticRegression with `class_weight='balanced'`
- C=1.0 (L2 regularization)
- Solver: lbfgs

#### Option 3: Position-Aware Weighted Average (Heuristic)

**No training required** - use transformer probabilities directly:

- For each edge, average probabilities weighted by position:
  - `weight_i = 1 / (1 + dist_from_start_i + dist_from_end_i)`
  - `P_edge = Σ(weight_i × P_i) / Σ(weight_i)`
- Predict: `argmax(P_edge)`

### Data Splits

**Important**: Aggregator uses different splits than transformer!

```
Transformer Splits:
  - train: used for transformer training
  - val: used for transformer validation
  - test: used for transformer testing

Aggregator Splits (from transformer predictions):
  - agg_train: 80% of transformer val edges
  - agg_val: 20% of transformer val edges
  - agg_test: 100% of transformer test edges
```

**Rationale**:

- Transformer's validation set is "unseen" during transformer training
- We split it further for aggregator training/tuning
- Transformer's test set becomes aggregator's test set

### Implementation Requirements

#### Script: `train_edge_aggregator.py`

**Command**:

```bash
python train_edge_aggregator.py \
  --predictions_dir outputs/{dataset}/{run}/checkpoints/{dataset}_predictions \
  --dataset {dataset_name} \
  --epoch {epoch_num} \
  --model {mlp|logistic|weighted} \
  --output_dir outputs/{dataset}/{run}/aggregator_results
```

**Arguments**:

- `--predictions_dir`: Directory with epoch_XXX folders
- `--dataset`: Dataset name
- `--epoch`: Which epoch to use (e.g., "best" or specific number)
- `--model`: Aggregator type (mlp, logistic, weighted)
- `--output_dir`: Where to save aggregator model and results
- `--seed`: Random seed (default: 42)

**Workflow**:

1. **Load predictions** from `epoch_{N}/val_predictions.pkl` and `epoch_{N}/test_predictions.pkl`
2. **Group by edge_id**: Collect all occurrences per unique edge
3. **Extract features** per edge:
   - Aggregate dist_from_start statistics
   - Aggregate dist_from_end statistics
   - Count occurrences
4. **Split val edges**: 80% agg_train, 20% agg_val
5. **Train model** on agg_train, tune on agg_val
6. **Evaluate** on agg_test (transformer test edges)
7. **Save**:
   - Model: `aggregator_{model}.pkl`
   - Results: `aggregator_{model}_results.txt`
   - Confusion matrix: `aggregator_{model}_confusion.png`
   - Feature importance (if applicable): `aggregator_{model}_features.png`

**Metrics**:

- Accuracy, F1 (macro), AUC (macro for 3-class)
- Per-class precision/recall
- Confusion matrix

### Expected Performance

**Hypothesis**: Aggregator should match or improve upon simple averaging strategies because:

1. Edges near walk boundaries might be less reliable
2. Walk length might correlate with prediction confidence
3. Multiple occurrences provide robustness

**Baseline Comparison**:

- **Transformer per-token**: Already evaluated (saved in predictions)
- **Simple averaging**: Average probabilities per edge (ignore position)
- **Weighted averaging**: Position-aware weighted average
- **Aggregator (MLP/Logistic)**: Learned from features

### Output Files

```
outputs/{dataset}/{run}/aggregator_results/
├── epoch_{N}_val_edge_features.pkl          # Aggregated features for val edges
├── epoch_{N}_test_edge_features.pkl         # Aggregated features for test edges
├── aggregator_mlp.pkl                       # Trained MLP model
├── aggregator_mlp_results.txt               # Metrics, classification report
├── aggregator_mlp_confusion.png             # Confusion matrix plot
├── aggregator_logistic.pkl                  # Trained logistic model
├── aggregator_logistic_results.txt          # Metrics, classification report
├── aggregator_logistic_confusion.png        # Confusion matrix plot
└── comparison_summary.txt                   # Compare all strategies
```

---

## Task Execution Order

### Phase 1: Preparation ✅

- [x] Task 3 POC complete
- [x] Binary toy dataset verified
- [x] Post-hoc analysis working on toy

### Phase 2: Infrastructure (To Do)

1. **Create retraining bash script**
   - File: `retrain_all_datasets_task3.sh`
   - Verify configs, hyperparameters, device assignments
   - Add monitoring/logging commands

2. **Make post-hoc analysis dataset-agnostic**
   - Refactor `test_posthoc_analysis.py`
   - Add command-line arguments
   - Support auto-discovery of runs
   - Test on toy dataset first

3. **Implement edge aggregator training**
   - Create `train_edge_aggregator.py`
   - Implement feature extraction from predictions
   - Implement MLP, Logistic, Weighted strategies
   - Create evaluation comparison

### Phase 3: Execution (To Do)

1. **Launch retraining** (parallel on 3 GPUs)
   - Estimated time: 10-12 hours for all datasets
   - Monitor logs, check disk space

2. **Run post-hoc analysis** on completed runs
   - Generate triplets and heatmaps for all epochs
   - Verify correctness patterns across positions

3. **Train edge aggregators** on best/final epoch
   - Compare strategies (simple avg vs weighted vs learned)
   - Evaluate on test split
   - Report if aggregation improves performance

### Phase 4: Reporting (To Do)

1. **TensorBoard analysis**
   - Compare learning curves across datasets
   - Check test metrics per epoch

2. **Aggregator comparison report**
   - Which strategy works best per dataset
   - Does position matter? Does walk length matter?

3. **Final summary document**
   - Task 3 complete end-to-end results
   - Recommendations for production use

---

## Success Criteria

### Retraining

- [ ] All 3 datasets complete training without errors
- [ ] Per-epoch predictions saved for all splits
- [ ] TensorBoard has complete logs (scalars + images)
- [ ] Final test AUC matches or exceeds Optuna trial performance

### Post-Hoc Analysis

- [ ] Works on all 3 datasets + toy
- [ ] Triplets extracted for all epochs/splits
- [ ] Heatmaps generated showing position effects
- [ ] Summary report generated

### Aggregator

- [ ] Edge-level features extracted from walk-level predictions
- [ ] MLP and Logistic models trained successfully
- [ ] Test AUC reported for all strategies
- [ ] Comparison shows which aggregation works best

---

## Estimated Timeline

- **Script preparation**: 2-3 hours
- **Retraining (parallel)**: 10-12 hours (overnight)
- **Post-hoc analysis**: 1-2 hours
- **Aggregator training**: 2-3 hours
- **Analysis & reporting**: 2-3 hours

**Total**: ~1.5-2 days end-to-end

---

## Notes & Considerations

### Disk Space

- Predictions can be large (especially epinions/slashdot with 5M walks)
- Estimate: ~100-500 MB per epoch per split
- Monitor: `du -sh outputs/*/`

### Memory

- Loading all predictions at once might be memory-intensive
- Consider batch processing for aggregator feature extraction

### Reproducibility

- Use `reproducibility.seed=42` for all runs
- Document exact commit hash and config versions

### Validation Strategy

- Compare aggregator test AUC to transformer test AUC
- If aggregator << transformer: position features not useful
- If aggregator ≈ transformer: simple averaging sufficient
- If aggregator > transformer: learned aggregation adds value!

---

**Document prepared for orchestrator review.**  
**Ready for implementation approval.**
