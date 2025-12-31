# CHAT C: Prediction Caching & Position Analysis

## Task Overview
Enable deeper analysis of transformer predictions by caching raw predictions with metadata, then analyzing position-based biases.

---

## Task C1: Cache Raw Predictions

### Current Problem
`scripts/extract_edge_scores.py` aggregates predictions per edge (percentiles, mean, std). This is fine for aggregator training, but loses detailed information:
- Which walk gave this prediction?
- Where in the walk (center vs edges)?
- What was the walk length?
- How confident was the model?

This makes it hard to understand what the aggregator is learning.

### Your Task
1. **Extend edge feature extraction** to include raw predictions:
   - Cache every prediction: `(edge_id, walk_id, position_in_walk, walk_length, predicted_probs, true_label)`
   - Store as pickle or parquet in `aggregator_features/{dataset}/raw_predictions/`
   - Separate files for val_raw.pkl, test_raw.pkl

2. **Maintain aggregated features**:
   - Keep existing percentile/mean/std aggregation (needed for aggregator training)
   - Add raw predictions to same output dictionary
   - Update output schema in pickle header

3. **Add caching for speed**:
   - Recompute aggregates from raw predictions (don't duplicate work)
   - If raw predictions already exist, reuse them
   - Add `--recompute-raw` flag to force rebuild

### Files to Modify
- `scripts/extract_edge_scores.py` (major refactor)
- Create `scripts/analyze_predictions.py` (new utility for C2/C3)

### Success Criteria
✅ Raw predictions cached with all metadata  
✅ Aggregated features computed from raw (one source of truth)  
✅ Separate val/test raw files  
✅ Can reload and reuse for analysis  
✅ extract_edge_scores.py still produces same aggregator features  

---

## Task C2: Position-in-Walk Analysis

### Current Problem
Different positions in a walk may have different prediction qualities:
- Center nodes (high signal) might have better predictions
- Edge nodes (neighbors only) might be noisier
- This could bias aggregation toward edge positions

### Your Task
1. **Analyze position effects**:
   - Load raw predictions from C1
   - Group by `position_in_walk` (0 = center, ±1 = neighbors, etc.)
   - For each position, compute:
     - Accuracy vs ground truth
     - AUC per position
     - Prediction confidence (max prob)
     - Confusion matrix per position

2. **Create visualizations**:
   - Plot: accuracy vs position_in_walk (with error bars)
   - Plot: AUC vs position (separate for val/test)
   - Heatmap: confusion matrix per position bin

3. **Generate report** `scripts/analyze_position_effects.py`:
   - Runs on all datasets
   - Outputs analysis to `analysis_results/position_effects_{dataset}.csv`
   - Saves plots to `analysis_results/plots/`

### Files to Create
- `scripts/analyze_position_effects.py` (new)

### Success Criteria
✅ Position-based accuracy/AUC computed for all datasets  
✅ Visualizations show clear trends (or no trends)  
✅ Report identifies best and worst positions  
✅ Ready to feed into C3 (feature engineering)  

---

## Task C3: Position-Aware Aggregator Features

### Current Problem
Current aggregator features use flat percentiles; they don't distinguish position quality. We can do better by:
- Weighting predictions by position quality
- Adding position-based features to aggregator input

### Your Task
1. **Add position-aware features**:
   - For each edge, add separate feature sets:
     - Features from center predictions only (position=0)
     - Features from neighbor predictions only (position=±1)
     - Features from all positions (current)
   - Compute same percentiles for each subset

2. **Retrain aggregators with new features**:
   - Use new feature set in `scripts/train_aggregator.py`
   - Compare performance: flat vs position-aware
   - Log results to `aggregator_results_position_aware/`

3. **Implement weighted aggregation**:
   - Optional: weight predictions by position accuracy (from C2)
   - Compare to uniform weighting
   - Document impact

### Files to Modify
- `scripts/extract_edge_scores.py` (add position-aware features)
- `scripts/train_aggregator.py` (support multiple feature sets)
- Create `scripts/compare_aggregator_variants.py`

### Success Criteria
✅ Position-aware features extracted for all datasets  
✅ Aggregators retrained with new feature set  
✅ Performance comparison: position-aware vs baseline  
✅ Report shows whether position helps or not  

---

## Integration Points
- **Depends on**: Chat A (config) + extract_edge_scores.py baseline
- **Feeds into**: Chat F (aggregator integration strategy)
- **Output format**: All analyses in `analysis_results/` directory

## Example Output (C2 Report)
```
Position Analysis for wiki-rfa (test set):
  Center (pos=0):       accuracy=0.75, auc=0.82
  Neighbors (pos=±1):   accuracy=0.68, auc=0.78
  All positions:        accuracy=0.71, auc=0.80

Best strategy: Weight center > neighbors 2:1
```

---

## Notes
- **High value**: C1+C2 will reveal prediction patterns; C3 shows if we can exploit them
- **Time estimate**: C1 (2h), C2 (2h), C3 (2h) = ~6h total
- **Can parallelize**: C2/C3 can run while other chats work on B/D/E
