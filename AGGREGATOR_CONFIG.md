# Aggregator Configuration

## Overview
The aggregator meta-classifier learns to predict edge labels from transformer walk scores. It uses transformer's val/test splits where edges have ground truth labels.

## Data Split Mapping

| Aggregator Split | Source | Purpose |
|---|---|---|
| **agg_train** | 80% of transformer val edges | Training the meta-classifier |
| **agg_val** | 20% of transformer val edges | Validation and hyperparameter tuning |
| **agg_test** | 100% of transformer test edges | Final evaluation (unseen by both models) |

## Feature Engineering

- **Extraction**: For each edge, collect all predictions across walks where that edge appears
- **Aggregation**: Per-edge score statistics → 10 features
  - Percentiles: p5, p10, p25, p50, p75, p90, p95
  - Statistics: mean, std, count (number of walks containing this edge)
- **Scaling**: StandardScaler (mean=0, std=1) applied per feature

## Class Weighting Strategy

**Matches transformer training** (both use inverse frequency):
$$\text{weight}_i = \frac{n\_total}{n\_classes \times n\_i}$$

where $n_i$ is the count of class $i$ in the training set.

### Implementation

**Logistic Regression:**
- Parameter: `class_weight='balanced'` (sklearn's inverse frequency)
- Applies uniform weighting to all samples based on global class distribution

**MLP:**
- Manually computed class weights (sklearn doesn't support `class_weight` for MLPClassifier)
- Same formula applied
- Printed during training for verification

## Hyperparameter Defaults

### Shared
- `class_weight`: 'balanced' (inverse frequency)

### Logistic Regression
- `max_iter`: 1000 (iterations for solver convergence)
- `C`: 1.0 (regularization strength; lower = more regularization)
- `solver`: 'lbfgs' (stable for both binary and multiclass)

### MLP
- `hidden_layers`: (128, 64) → two hidden layers with 128 and 64 units
- `learning_rate`: 0.001 (initial learning rate for SGD)
- `max_iter`: 1000 (max training epochs)
- `batch_size`: 32 (samples per gradient update)
- `early_stopping_patience`: 20 (stop if validation loss doesn't improve for 20 epochs)

## Command-Line Override Examples

```bash
# Use MLP with custom architecture
python scripts/train_aggregator.py \
  --features aggregator_features/epinions_features.pkl \
  --output_dir aggregator_results/epinions/mlp_custom \
  --model mlp \
  --mlp_hidden_layers "256,128,64" \
  --mlp_early_stopping_patience 30 \
  --mlp_learning_rate 0.0005

# Use logistic with stronger regularization
python scripts/train_aggregator.py \
  --features aggregator_features/epinions_features.pkl \
  --output_dir aggregator_results/epinions/logistic_l2 \
  --model logistic \
  --logistic_C 0.1
```

## Key Decisions

1. **Why no weighted loss during MLP training?**
   - sklearn MLPClassifier doesn't support `class_weight` parameter
   - Instead, we compute and print class weights for reference and understanding
   - (Advanced: could use sample_weight in fit() if needed)

2. **Why 80/20 split of transformer val?**
   - Provides sufficient training data (aggregator needs many edges)
   - Clean val set for tuning without affecting test
   - Transformer already saw these edges during training

3. **Why match transformer's weighting formula?**
   - Consistency: both models handle class imbalance the same way
   - Learned from transformer's training experience
   - Inverse frequency is well-established for imbalanced datasets

4. **Why StandardScaler?**
   - Features have different ranges (percentiles 0-1, count unbounded)
   - Both logistic regression and MLP benefit from normalized features
   - Applied independently to agg_train (fit) and agg_val/test (transform)

## Expected Behavior

- **Logistic regression**: Fast, interpretable, good baseline
- **MLP**: Potentially captures nonlinear relationships in score distributions
- **Both**: Should help balance transformer's predictions across different edge types

## Output

For each model, the script saves:
- `aggregator_{model}.pkl`: Trained classifier
- `scaler.pkl`: Feature scaler (must be applied to new data identically)
- `results.pkl`: All metrics and predictions
- `summary.txt`: Human-readable results
- `*_confusion_matrix.png`: Visualizations
