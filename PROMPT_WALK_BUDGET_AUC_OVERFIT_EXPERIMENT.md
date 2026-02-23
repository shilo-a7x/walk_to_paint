# Prompt for New Chat: Walk Budget Sensitivity by AUC + Overfit (5 Epoch Cap)

You are working in `walk_to_paint`.
Design and run a controlled experiment to compare `num_walks` / `max_walk_length` settings by **model quality** (AUC on train/val/test), not only coverage statistics.

## What to optimize for

Find settings that give the best tradeoff between:

1. strong `val/test` AUC,
2. low overfit gap,
3. acceptable train performance,
4. reduced walk budget when possible.

We explicitly want candidates where train AUC may drop a bit, but val/test generalization improves.

---

## Hard Constraints (Must Follow)

1. **Epoch cap**: maximum `training.epochs=5` for this experiment.
2. **Rebuild data from scratch every run**:
   - `preprocess.use_cache=false`
   - `preprocess.save=false`
3. **Do NOT overwrite existing cached artifacts in current `data/` dirs**.
4. Reproducibility:
   - use canonical seed path (`reproducibility.seed`, via existing config flow)
   - keep same seed across compared runs (at least for first pass)
5. Keep all non-budget hyperparameters fixed initially (single-factor comparison).

---

## Important Safety Requirement: No writes to existing data artifacts

Even with cache/save disabled, ensure no accidental writes over existing dataset artifacts.

### Required approach

Use an **isolated temporary dataset directory per run** and point config there:

- Override `dataset.data_dir` to something like:
  - `tmp/walk_budget_auc/<dataset>/<run_tag>/data`
- Copy only required raw edge-list input file(s) into that temp dir.
- Run preprocessing/training using that isolated `dataset.data_dir`.

This guarantees existing `data/<dataset>/*.pt` files are untouched.

If any code path still tries to save artifacts, they go to temp dir, not production data dir.

---

## Baseline and Candidate Grid

Use dataset-specific defaults from existing configs as baseline anchor.

### Suggested candidate sets (start small)

- `max_walk_length`: `[60, 80]`
- `num_walks`: `[1_000_000, 2_000_000, 3_500_000, 5_000_000]`

If total runs are too many, do staged filtering:

1. Fix `max_walk_length=80`, sweep `num_walks`.
2. Keep top 2 budgets by val/test + gap.
3. Compare those top budgets with `max_walk_length=60` vs `80`.

---

## Metrics to collect (per run)

At minimum collect:

- `train_auc`
- `val_auc`
- `test_auc`
- `train_loss`
- `val_loss`
- best epoch (within 5)
- runtime

Overfit indicators:

- `auc_gap = train_auc - val_auc`
- `loss_gap = val_loss - train_loss`
- optional: `test_minus_val = test_auc - val_auc`

Ranking emphasis:

1. higher `val_auc`
2. then higher `test_auc`
3. then lower `auc_gap`
4. then lower compute cost (`num_walks`, runtime)

---

## Execution Plan

### Phase 1: Runner preparation

Create/adjust a script (or command generator) to:

- iterate candidate grid,
- create isolated temp data dir for each run,
- copy raw input file(s) only,
- run training with overrides:
  - `training.epochs=5`
  - `preprocess.use_cache=false`
  - `preprocess.save=false`
  - `dataset.num_walks=<candidate>`
  - `dataset.max_walk_length=<candidate>`
  - `dataset.data_dir=<temp_path>`
- save logs and parsed metrics to a summary CSV.

### Phase 2: Run experiments

Start with one target dataset (fastest representative) to validate protocol.
Then expand to other datasets if time permits.

### Phase 3: Analysis

Produce final comparison table and shortlist:

- best generalization candidate,
- best efficiency candidate,
- conservative candidate (closest to baseline quality).

---

## Output artifacts expected

1. Experiment table (CSV + markdown) with columns:
   - dataset, seed, num_walks, max_walk_length, best_epoch,
   - train_auc, val_auc, test_auc,
   - train_loss, val_loss,
   - auc_gap, loss_gap,
   - runtime_minutes
2. Clear recommendation section:
   - winner for quality,
   - winner for anti-overfit,
   - winner for compute/quality balance.
3. Confirmation that production `data/<dataset>/*.pt` files were not modified.

---

## Acceptance Criteria

- Runs complete with epoch cap = 5.
- All runs rebuild preprocessing from scratch.
- Existing data artifacts remain untouched.
- Results include train/val/test AUC and overfit-gap analysis.
- Recommendation balances quality and generalization (not train metric alone).

---

## Notes from existing project context

- Previous walk-budget report focused on coverage/distribution sensitivity.
- This new task extends it to **model quality sensitivity**.
- Aggregator work is currently non-urgent and can be ignored in this task.
