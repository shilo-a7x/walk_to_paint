# Prompt for New Chat: Incremental Transformer Improvements (Overfit + Representation)

You are working in `walk_to_paint`.
Your job is to **gradually** improve transformer performance, with main focus on:

1) reducing overfitting, 2) improving representation quality.

## Current Project Context (Important)

- Seeding strategy exists and must stay canonical via `reproducibility.seed` + `get_seed(cfg)`.
- Stratified splits and class-weighted loss were added previously.
- Dataset cache system exists and should be used (`preprocess.use_cache=true`) to speed experiments.
- Posthoc + aggregator tooling exists (`run_posthoc.py`, scripts under `scripts/`).
- Aggregator summary exists in `AGGREGATOR_POSTHOC_SUMMARY.md`.

## Ground Rules (Must Follow)

1. **One small change at a time** (single-factor experiments first).
2. Keep runs reproducible (fixed seed, same split strategy, same eval protocol).
3. Do not break existing train/posthoc pipelines.
4. Prefer minimal, local code changes; avoid large refactors.
5. Keep data budget fixed at first (no changing walk budget in early steps unless explicitly in a curriculum experiment).
6. For every experiment, compare against baseline using same dataset + seed.

## Success Criteria

Primary:

- Improve validation/test metrics (AUC/F1/accuracy as applicable).
- Reduce generalization gap:
  - `gap_loss = val_loss - train_loss`
  - `gap_auc = train_auc - val_auc` (if train AUC exists)

Secondary:

- Stability across 2-3 seeds for top candidates.
- No major training slowdown or memory blowups.

## Deliverables From This Chat

Produce:

1. A short prioritized experiment plan (basic → moderate → advanced).
2. Minimal code changes for the first wave (basic anti-overfit improvements).
3. Run commands for each experiment.
4. A compact results table template and filled rows for completed runs.
5. Recommendation for next iteration based on evidence.

---

## Phase 0 — Baseline Lock (Do First)

1. Pick one dataset for fast iteration (prefer the fastest representative available in this repo).
2. Run baseline training with current default config and fixed seed.
3. Record:
   - best epoch
   - train/val/test loss
   - val/test AUC (or task metrics)
   - overfit gaps (`gap_loss`, `gap_auc`)
   - runtime per epoch

Do not proceed until baseline is reproducible.

---

## Phase 1 — Basic Anti-Overfit Sweep (Low Risk)

Run ablations one by one (not combined initially):

1. **Capacity down**
   - lower `embedding_dim`, `hidden_dim`, maybe `nlayers`
2. **Regularization up**
   - increase dropout moderately
   - increase weight decay within reasonable range
3. **Training control**
   - stricter early stopping patience
   - learning rate reduction or schedule adjustment (if scheduler exists)
4. **Label smoothing** (if classification head supports it)

For each experiment:

- change one knob (or one coherent mini-group),
- run,
- compare to baseline,
- keep only if it improves val metric and/or reduces gap.

---

## Phase 2 — Representation-Focused Experiments (Moderate Risk)

After Phase 1 winners, test representation ideas incrementally:

1. **Dynamic masking schedule**
   - start with lower masking, increase through epochs (or inverse; test 2 variants)
2. **Node dropout**
   - probabilistically drop node tokens in walk context (careful with special tokens)
3. **Fake node tokens / structural sentinel tokens**
   - add minimal token-type augmentation to encode role/position/segment
4. **Even lower dims + stronger bottleneck**
   - test compact representations to force generalizable features
5. **Curriculum learning (walk complexity)**
   - begin with shorter/easier sequences, then increase to full complexity

Important: gate each idea behind config flags so experiments are easy to toggle.

---

## Phase 3 — Combine Winners (Controlled)

Take top 2-3 winning ideas and combine them carefully.

- Start with pairwise combinations.
- Only then try triple combination if pairwise results are stable.
- Validate on at least one additional dataset before claiming improvement.

---

## Implementation Constraints

- Reuse existing config pattern and avoid hardcoded constants.
- Preserve compatibility with caching and posthoc scripts.
- If a new feature is added, add safe defaults so old configs still run.
- Avoid changing output directory semantics unless required.

---

## Experiment Tracking Format

Use a compact table like:

| Exp ID | Change | Seed | Best Epoch | Val AUC | Test AUC | Train Loss | Val Loss | Gap Loss | Runtime/Epoch | Verdict |
|-------:|--------|-----:|-----------:|--------:|---------:|-----------:|---------:|---------:|--------------:|--------|

Verdict rules:

- **Keep**: better val/test and same or lower overfit gap.
- **Maybe**: small metric gain but higher variance.
- **Drop**: no gain or worse overfitting.

---

## Suggested First 6 Experiments

1. `E0_BASELINE` — current config, fixed seed.
2. `E1_SMALLER_MODEL` — reduce hidden/embedding/layers.
3. `E2_DROPOUT_UP` — increase dropout only.
4. `E3_WEIGHT_DECAY_UP` — increase WD only.
5. `E4_EARLY_STOP_TIGHTER` — lower patience only.
6. `E5_DYNAMIC_MASK_V1` — introduce simple masking schedule.

Then select best 2 for combination trial `E6_COMBO_TOP2`.

---

## Decision Protocol

At each step, answer:

1. Did val/test improve?
2. Did overfit gap shrink?
3. Is result stable across rerun/seed?
4. Is compute cost acceptable?

If 2+ answers are “no”, rollback and try next idea.

---

## Final Output Expected from This Chat

- Updated code + config toggles for tested ideas.
- Commands used to run each experiment.
- Results table with clear keep/drop decision.
- Recommendation of next small batch of experiments.
