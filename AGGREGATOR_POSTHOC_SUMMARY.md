# Aggregator + Posthoc Q&A Summary

Date: 2026-02-23

## Decision Update (Latest)

- ✅ `lgbm` should be included across aggregator flows.
- ✅ Optuna should save full loadable model config (tuned + fixed params).
- 📝 Keep current auxiliary artifacts for now; revisit for production profile later.
- 📝 Rename/restructure plan only for now (no module moves yet).

## 1) Are aggregator model/scaler saved and reloadable?

Yes.

Current save points:

- `run_posthoc.py` saves per-model artifacts to:
  - `outputs/<dataset>/<exp>/posthoc/<run_id>/aggregator/<model>/model.pkl`
  - `outputs/<dataset>/<exp>/posthoc/<run_id>/aggregator/<model>/summary.txt`
- `scripts/agg_train_edge.py` saves to:
  - `<output-dir>/model.pkl`
  - `<output-dir>/summary.txt`
- `scripts/agg_optuna_tune.py` saves tuned models to:
  - `<output-dir>/logistic_model.pkl`
  - `<output-dir>/xgboost_model.pkl`
  - `<output-dir>/lgbm_model.pkl`
  - `<output-dir>/optuna_summary.txt`

Model pickle format is a dict:

```python
{"model": trained_model, "scaler": fitted_scaler}
```

Reload example:

```python
import pickle

with open(".../model.pkl", "rb") as f:
    bundle = pickle.load(f)
model = bundle["model"]
scaler = bundle["scaler"]
```

---

## 2) Which aggregator models are currently supported?

By pipeline/script:

- `run_posthoc.py` (`--agg-models`):
  - `logistic`, `xgboost`, `lgbm`
- `scripts/agg_train_edge.py` (`--model`):
  - `logistic`, `xgboost`, `lgbm`
- `scripts/agg_optuna_tune.py` (fixed tuned set):
  - `logistic`, `xgboost`, `lgbm`

Also included in:

- `scripts/run_aggregator_new.sh`
- `scripts/compare_aggregator.py`

---

## 3) Can posthoc run with a specific model and hyperparameters?

Current state:

- Specific model: **yes** via `--agg-models logistic,xgboost,lgbm`.
- Specific hyperparameters: **not yet** in `run_posthoc.py` CLI. It uses hardcoded defaults.

Current best practice:

1. Run Optuna tuner:

   ```bash
   source .venv/bin/activate
   python scripts/agg_optuna_tune.py --input <features.pkl> --output-dir <out_dir> --n-trials 30
   ```

2. Use saved model + full JSON config from that output.

New Optuna persisted files per output dir:

- `<model>_model.pkl`
- `<model>_config.json`  ← full loadable config (tuned + fixed + context)
- `optuna_summary.txt`
- `optuna_summary.json`  ← full multi-model summary

Recommended next change (small):

- Add CLI options to `run_posthoc.py`, e.g.:
  - `--agg-model logistic --agg-params '{"C":1.2,"max_iter":1500}'`
  - `--agg-model xgboost --agg-params '{...}'`
  - `--agg-model lgbm --agg-params '{...}'`

---

## 4) If transformer was trained without callbacks, how do I generate prediction PKLs?

Use `run_posthoc.py` with `predictions` artifact. It re-runs inference from checkpoint and saves prediction PKLs.

Example:

```bash
source .venv/bin/activate
python run_posthoc.py \
  --config config.yaml \
  --exp-dir outputs/wiki-rfa/<exp_name> \
  --checkpoint-choice best \
  --splits train,val,test \
  --artifacts predictions
```

Saved files:

- `outputs/wiki-rfa/<exp_name>/checkpoints/wiki-rfa_predictions/epoch_XXX/train_predictions.pkl`
- `outputs/wiki-rfa/<exp_name>/checkpoints/wiki-rfa_predictions/epoch_XXX/val_predictions.pkl`
- `outputs/wiki-rfa/<exp_name>/checkpoints/wiki-rfa_predictions/epoch_XXX/test_predictions.pkl`

Then you can run triplets/heatmaps/aggregator using those PKLs.

---

## 5) Proposed scripting cleanup and single-entry integration

Suggested target structure:

- `src/posthoc/`
  - `checkpoint_selector.py`
  - `prediction_export.py`
  - `triplet_builder.py`
  - `heatmap.py`
  - `aggregator.py`
  - `optuna_tune.py`
- Keep CLI wrappers in `scripts/` as thin launchers only.

Suggested single entrypoint pattern:

- Keep `run.py` for training.
- Add `pipeline.py` with subcommands:
  - `pipeline.py train ...`
  - `pipeline.py posthoc ...`
  - `pipeline.py agg-tune ...`
  - `pipeline.py report ...`

Why this helps:

- Less duplicated feature/aggregation logic.
- Reuse same config schema across train/posthoc/tune.
- Cleaner maintenance and easier orchestration.

---

## 6) Minimal end-to-end command recipes

### A. Train a dataset

```bash
source .venv/bin/activate
python run.py --config config.yaml dataset.name=wiki-rfa training.epochs=20
```

### B. Generate predictions + posthoc artifacts from best checkpoint

```bash
source .venv/bin/activate
python run_posthoc.py \
  --config config.yaml \
  --exp-dir outputs/wiki-rfa/<exp_name> \
  --checkpoint-choice best \
  --splits train,val,test \
  --artifacts predictions,triplets,heatmaps,aggregator \
  --agg-models logistic,xgboost
```

### C. Run Optuna tuning for aggregator models

```bash
source .venv/bin/activate
bash scripts/run_optuna_aggregator.sh
```

### D. Quick best-vs-transformer report

```bash
source .venv/bin/activate
python scripts/quick_summary.py
```

---

## Data representation reminder

Each prediction row is a walk-edge occurrence sample:

- features used by aggregator: `[dist_from_start, walk_length, predicted_prob]`
- target: edge label
- edge-level score: mean of walk-level probabilities grouped by `edge_id`

This is why one edge can have many rows and still get one final edge prediction.
