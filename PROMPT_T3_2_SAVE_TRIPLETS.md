# Prompt: T3.2 Save Aggregator Triplets (Proof of Concept)

## Goal

Produce **triplets** for the professor’s analysis:

```
(distance_from_start, distance_from_end, correct_flag)
```

then save them to disk. This is **per occurrence**, not per unique edge.

## Dependency

Run after T3.1 (predictions saved with walk metadata).

## Input

Load predictions file from T3.1:

- `outputs/predictions/<dataset>/raw_scores/<run_id>_<split>.pkl`

## Required Triplet Definition

For each **edge occurrence**:

- `dist_from_start`: raw edge count to the left (from T3.1)
- `dist_from_end`: raw edge count to the right (from T3.1)
- `correct_flag`: 1 if prediction for that occurrence matches ground truth, else 0

## Output

- `outputs/aggregation/<dataset>/strategy_mean/triplets_<split>.pkl`

## Minimal Data Structure

```python
triplets = {
  "edge_id": [...],            # keep for traceability
  "walk_id": [...],            # keep for traceability
  "position": [...],           # edge position
  "walk_len": [...],
  "dist_from_start": [...],    # raw edge count
  "dist_from_end": [...],      # raw edge count
  "correct": [...],            # 0/1 per occurrence
}
```

## Algorithm (POC)

1. Load predictions (val/test) from T3.1.
2. For each prediction record:

- correct = int(pred_label == true_label)
- dist_from_start = position
- dist_from_end = walk_len - 1 - position

3. Save triplets pickle (one per occurrence).

## Success Criteria

- ✅ One triplet per **edge occurrence**
- ✅ Distances are **raw edge counts**
- ✅ Correct flag is 0 or 1
- ✅ Saved file loads successfully

## Notes

- We **do not aggregate** by edge here.
- We **do** keep `edge_id` and `walk_id` for traceability.
