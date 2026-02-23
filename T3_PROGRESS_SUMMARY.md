# T3 Pipeline Progress Summary (Feb 5, 2026)

## Overview

Implemented and validated the full T3 proof‑of‑concept pipeline across:

- **T3.1** Save per‑occurrence predictions with walk metadata
- **T3.2** Save per‑occurrence triplets for aggregation analysis
- **T3.3** Heatmap visualization from triplets

## Key Implementations

### T3.1: Save Predictions

- Added **preds‑only** and **max‑batches** options for fast validation.
- Vectorized extraction and precomputed node ID mapping for speed.
- Output format: `outputs/predictions/<dataset>/raw_scores/<run_id>_<split>.pkl`.

### T3.2: Save Triplets

- New script: `scripts/save_triplets.py`.
- Input: predictions from T3.1.
- Output: `outputs/aggregation/<dataset>/strategy_mean/triplets_<split>.pkl`.
- Verified correct lengths and sample values on wiki‑rfa.

### T3.3: Heatmap Visualization

- New script: `scripts/plot_triplet_heatmap.py`.
- Output: `outputs/aggregation/<dataset>/strategy_mean/heatmap_<split>.png`.
- Verified heatmap generation on wiki‑rfa (val split).

## Performance & GPU

- Vectorized extraction greatly reduced Python‑loop overhead.
- Device override via `--device` confirmed (tested on GPU 3).
- Empirical memory probe (wiki‑rfa, batch 1024): **~650 MiB peak GPU**.
- Chosen overrides for full run: `--batch-size 1024`, `training.num_workers=16`, `training.prefetch_factor=4`, `training.persistent_workers=true`, `training.pin_memory=true`.

## Validation Results (Smoke Tests)

- Predictions saved correctly for val/test with expected fields and lengths.
- Triplets saved with correct distances and `correct` flags.
- Heatmap file saved successfully.

## Full Run Command (Started)

User launched:

```
nohup bash scripts/run_t3_full.sh > heatmap_all.out 2>&1 &
```

This runs predictions on 3 datasets (wiki‑rfa, slashdot090221, epinions) on separate GPUs, then triplets, then heatmaps.

## Artifacts Created

- `scripts/save_triplets.py`
- `scripts/plot_triplet_heatmap.py`
- `scripts/run_t3_full.sh`
- Smoke test outputs under:
  - `outputs/predictions/wiki-rfa/raw_scores/`
  - `outputs/aggregation/wiki-rfa/strategy_mean/`

## Status

- Implementation complete and validated on wiki‑rfa.
- Full multi‑dataset run in progress via nohup.
