# Prompt: T3.1 Save Predictions (Proof of Concept)

## Goal

Implement a proof‑of‑concept pipeline that **saves model predictions** (raw scores + predicted labels) **with walk metadata** so downstream analysis is possible without retraining.

## Scope (POC)

- Save predictions for **validation and test** splits.
- Preserve enough metadata for later aggregation & analysis.
- Keep implementation minimal and isolated (no refactor of config/output system yet).

## Where to Implement

- Primary entry: `scripts/extract_edge_scores.py` (current file opened by user)
- If predictions are produced elsewhere, add a minimal call or reuse utilities.

## What To Save (Per Edge Occurrence)

Store one record **per edge occurrence in a walk** (not per unique edge):

Fields:

- `edge_id`: tuple `(u, v)` or any unique edge identifier used in code
- `score`: raw model score (float)
- `pred_label`: predicted label (0/1)
- `true_label`: ground truth (0/1)
- `walk_id`: integer ID of the walk
- `position`: index in walk sequence (edge position)
- `walk_len`: length of walk (edges or tokens; be consistent)
- `dist_from_start`: raw edge count to the left (`position`)
- `dist_from_end`: raw edge count to the right (`walk_len - 1 - position`)
- `split`: train/val/test
- `dataset`: dataset name

## Output Format

- Use `pickle` (`.pkl`) for simplicity and fast I/O.
- One file per split:
  - `outputs/predictions/<dataset>/raw_scores/<run_id>_<split>.pkl`

`run_id` can be:

- timestamp or
- trial id

## Minimal Data Structure

```python
payload = {
  "edge_id": [...],
  "score": [...],
  "pred_label": [...],
  "true_label": [...],
  "walk_id": [...],
  "position": [...],
  "walk_len": [...],
  "dist_from_start": [...],  # raw edge count
  "dist_from_end": [...],    # raw edge count
  "split": "val" or "test",
  "dataset": "wiki-rfa",
}
```

## Success Criteria

- ✅ Files saved successfully for **val and test**
- ✅ Loaded files match number of edge occurrences
- ✅ Distances are **raw edge counts**
- ✅ POC runs on one dataset

## Minimal Validation Script (optional)

- Load file, check lengths match, print 5 samples.

## Notes

- Don’t integrate with caching or new output system yet.
- Don’t change training. Just extract from existing prediction output path.
- Keep logic isolated and easy to delete later.
