# Edge Score Aggregation: Deep Dive

## Problem Statement

Your walk-based edge classification model makes predictions for individual edges **within walks**. However, real-world edges appear in **multiple walks** with potentially different predictions.

### Example

```
Graph Edge (u, v):
  - Appears in Walk #1 (length 47): MODEL PREDICTS class=positive (confidence 0.92)
  - Appears in Walk #5 (length 13): MODEL PREDICTS class=negative (confidence 0.45)
  - Appears in Walk #23 (length 67): MODEL PREDICTS class=positive (confidence 0.89)
  - Appears in Walk #101 (length 3): MODEL PREDICTS class=positive (confidence 0.61)

How do we combine these 4 predictions into ONE final label for edge (u, v)?
```

This is **the core problem** you need to solve before downstream tasks.

---

## Why This Matters

### Impact on Performance

Different aggregation strategies can lead to different outcomes:

1. **Mean aggregation:** (0.92 + 0.45 + 0.89 + 0.61) / 4 = **0.72** → positive
2. **Majority vote:** 3 positive, 1 negative → **positive**
3. **Weighted by walk length:** longer walks (47, 67) have higher weight → **positive (0.90)**
4. **Max confidence:** max(0.92, 0.45, 0.89, 0.61) = **0.92** → positive
5. **Learned aggregation:** train MLP([0.92, 0.45, 0.89, 0.61]) → may output different result

All 5 methods give the same result here, but they won't always.

### Edge Cases

What about edges that appear in very few walks? Or edges that get contradictory predictions?

```
Edge (x, y):
  - Walk #3 (length 7): POSITIVE (0.95)
  - Walk #12 (length 9): NEGATIVE (0.98)
  - Walk #40 (length 5): POSITIVE (0.52)

Mean: (0.95 + 0.98 + 0.52) / 3 = 0.82 → positive (uncertain)
Majority vote: positive (2 vs 1)
Max: 0.98 → CONTRADICTS the mean!
```

---

## Available Aggregation Strategies

### 1. Simple Mean

**What:** Average prediction across all walks

```python
def mean_aggregation(predictions):
    return np.mean(predictions)

# Example: [0.92, 0.45, 0.89, 0.61] → 0.7175
```

**Pros:**
- Simple, fast, interpretable
- Works well if predictions are normally distributed

**Cons:**
- Ignores confidence differences
- Sensitive to outliers

**When to use:**
- Baseline / starting point
- When you have many walks per edge

---

### 2. Weighted Mean (by Confidence)

**What:** Weight predictions by model confidence

```python
def weighted_mean(predictions, confidences):
    return np.average(predictions, weights=confidences)

# Example: 
#   predictions = [0.92, 0.45, 0.89, 0.61]
#   confidences = [0.8, 0.3, 0.85, 0.5]  # From softmax max prob
#   → weighted average favors high-confidence predictions
```

**Pros:**
- Respects model uncertainty
- Downweights uncertain predictions

**Cons:**
- Requires confidence scores (not always available)

**When to use:**
- When you have softmax probabilities, not just class labels
- When model uncertainty is important

---

### 3. Majority Vote

**What:** Take the most common class label

```python
def majority_vote(class_labels):
    from collections import Counter
    return Counter(class_labels).most_common(1)[0][0]

# Example: [1, 0, 1, 1] → 1 (positive)
```

**Pros:**
- Simple, interpretable
- Works with hard labels

**Cons:**
- Loses confidence information
- Breaks ties arbitrarily

**When to use:**
- When you have hard class labels, not probabilities
- Binary classification

---

### 4. Consensus + Confidence

**What:** Only accept predictions if they're confident AND consistent

```python
def consensus_with_threshold(predictions, confidences, 
                              conf_threshold=0.7, agreement_threshold=0.8):
    # Only use high-confidence predictions
    confident = [(p, c) for p, c in zip(predictions, confidences) 
                 if c >= conf_threshold]
    
    if not confident:
        return np.mean(predictions)  # Fallback to mean
    
    # Check if confident predictions agree
    confident_preds = [p for p, c in confident]
    if np.std(confident_preds) < (1 - agreement_threshold):
        return np.mean(confident_preds)
    else:
        return None  # Conflicting confident predictions
```

**Pros:**
- Ignores uncertain predictions
- Flags conflicting information

**Cons:**
- May discard useful information
- Requires tuning thresholds

**When to use:**
- When you need high-confidence final labels
- When conflicting predictions are problematic

---

### 5. Position-Weighted Aggregation

**What:** Weight predictions by position in walk (edges near middle might be more "central")

```python
def position_weighted(predictions, walk_lengths):
    """
    For each prediction, assign a weight based on where the edge
    appeared in the walk (position / walk_length).
    """
    weights = []
    for pos_in_walk, walk_len in walk_lengths:
        # Central positions (0.4 to 0.6 of walk) have higher weight
        normalized_pos = pos_in_walk / walk_len
        weight = np.exp(-4 * (normalized_pos - 0.5)**2)  # Gaussian, peak at 0.5
        weights.append(weight)
    
    weights = np.array(weights)
    weights /= weights.sum()  # Normalize
    
    return np.average(predictions, weights=weights)
```

**Pros:**
- Captures structure of walks
- May be more robust

**Cons:**
- Requires tracking positions (more data)
- Assumption that central edges are more important (might not be true)

**When to use:**
- If you suspect edge position in walk matters
- Longer walks with diverse edge positions

---

### 6. Length-Stratified Aggregation

**What:** Group by walk length, aggregate within each group, then combine

```python
def length_stratified(predictions, walk_lengths):
    """
    Short walks (e.g., length 3-10) may have different characteristics
    than long walks (length 50-100).
    Aggregate separately, then combine.
    """
    short = [p for p, l in zip(predictions, walk_lengths) if l <= 20]
    medium = [p for p, l in zip(predictions, walk_lengths) if 20 < l <= 50]
    long = [p for p, l in zip(predictions, walk_lengths) if l > 50]
    
    short_score = np.mean(short) if short else 0.5
    medium_score = np.mean(medium) if medium else 0.5
    long_score = np.mean(long) if long else 0.5
    
    # Weight by frequency or confidence
    n_short, n_medium, n_long = len(short), len(medium), len(long)
    total = n_short + n_medium + n_long
    
    return (n_short * short_score + n_medium * medium_score + n_long * long_score) / total
```

**Pros:**
- Accounts for walk length heterogeneity
- Interpretable groups

**Cons:**
- Arbitrary group boundaries
- May fragment data

**When to use:**
- If you suspect walk length affects edge labeling reliability
- When you have enough data to stratify

---

### 7. Learned Aggregation (Advanced)

**What:** Train a small neural network to learn optimal aggregation

```python
class LearnedAggregator(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, 1),
            torch.nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, predictions, confidences=None):
        """
        predictions: tensor of shape [n_walks]
        confidences: optional, tensor of shape [n_walks]
        
        Returns: aggregated score (0 or 1)
        """
        if confidences is not None:
            x = torch.stack([predictions, confidences], dim=1)
        else:
            x = predictions.unsqueeze(1)
        
        return self.mlp(x).squeeze()

# Training would use:
# - Inputs: predictions from multiple walks for each edge
# - Targets: true edge labels
# - Loss: binary cross-entropy
```

**Pros:**
- Learns optimal strategy from data
- Can discover non-obvious patterns
- Generalizes to new edges

**Cons:**
- Requires labeled data for edges
- More complex, potential overfitting
- Slower inference

**When to use:**
- You have plenty of labeled data
- Simple aggregations underperform
- You want state-of-the-art performance

---

## Comparative Table

| Method | Complexity | Speed | Data Needed | Best For |
|--------|-----------|-------|-------------|----------|
| Mean | ⭐ | ⭐⭐⭐⭐⭐ | Predictions | Baseline |
| Weighted Mean | ⭐⭐ | ⭐⭐⭐⭐ | Predictions + Confidence | Uncertainty matters |
| Majority Vote | ⭐ | ⭐⭐⭐⭐⭐ | Hard labels | Binary, simple |
| Consensus | ⭐⭐ | ⭐⭐⭐⭐ | Predictions + Confidence | High-confidence labels |
| Position-Weighted | ⭐⭐ | ⭐⭐⭐⭐ | Predictions + Positions | Structure matters |
| Length-Stratified | ⭐⭐ | ⭐⭐⭐⭐ | Predictions + Walk lengths | Walk length affects prediction |
| Learned | ⭐⭐⭐⭐ | ⭐⭐ | Predictions + True labels | State-of-the-art |

---

## Practical Recommendation

### Phase 1: Baseline (This Week)
Start with **Mean Aggregation**:
```python
# In evaluation_pipeline.py, modify eval loop:
edges_predictions = defaultdict(list)
for walk in walks:
    for edge in walk:
        pred = model(edge_in_walk_context)
        edges_predictions[edge].append(pred)

# Final scores
for edge, predictions in edges_predictions.items():
    final_score[edge] = np.mean(predictions)
```

Measure: How does this affect your downstream task performance?

### Phase 2: Experiment (Next 2 Weeks)
Try 2-3 methods:
1. **Weighted Mean** (if you have softmax probs)
2. **Majority Vote** (if using hard labels)
3. **Length-Stratified** (if walk length varies a lot)

Measure: Which improves performance the most?

### Phase 3: Production (Ongoing)
Once you know what works:
- Use that method as baseline
- Optionally train Learned Aggregator if performance plateau
- Document your choice for reproducibility

---

## Implementation in Your Codebase

The `evaluation_pipeline.py` already has aggregation functions ready:

```python
from scripts.evaluation_pipeline import aggregate_edge_scores_across_walks

# After running model on all walks, you have:
predictions_per_edge = {
    edge_id: [score_from_walk1, score_from_walk2, ...]
    for edge_id in edges
}

# Try different methods:
result_mean = aggregate_edge_scores_across_walks(
    predictions_per_edge, 
    aggregation_method="mean"
)

result_max = aggregate_edge_scores_across_walks(
    predictions_per_edge, 
    aggregation_method="max"
)

result_majority = aggregate_edge_scores_across_walks(
    predictions_per_edge, 
    aggregation_method="majority_vote"
)
```

---

## Next Steps

1. **Choose a baseline** — recommend Mean Aggregation
2. **Implement it** — modify `evaluation_pipeline.py`
3. **Measure impact** — how does it affect your task?
4. **Iterate** — try other methods, compare
5. **Document** — record which method works best and why

Good luck! This is the **core algorithmic problem** of your project.
