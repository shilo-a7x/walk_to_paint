# Prompt for New Chat: Full T0.2 Config Hardening (Fail-Fast, Safe, Incremental)

You are working in `walk_to_paint`.
Your task is to implement **T0.2 Robust Config System** with minimal risk to ongoing experiments.

Current context:

- Transformer-improvement experiments are running in parallel chats.
- Walk-budget/AUC experiments are running in parallel chats.
- Aggregator integration is currently deferred.
- We need config hardening now to prevent silent misconfigurations.

Goal:

- Add a **fail-fast validation layer** and safer config UX without breaking existing scripts.
- Start minimal and robust (not a huge refactor).

---

## Scope (What to implement now)

### 1) Add centralized config validation

Implement in `src/utils/config.py`:

- New function: `validate_config(cfg, context: str = "train") -> None`
- It should raise `ValueError` with clear actionable messages on invalid config.
- Keep existing `load_config()` behavior intact (base + dataset override + CLI overrides).
- Keep `get_seed(cfg)` as canonical seed source.

Validation should cover at least:

#### Required structure/keys

- `reproducibility.seed`
- `dataset.name`
- `dataset.data_dir`
- `dataset.max_walk_length`
- `dataset.num_walks`
- `training.epochs`
- `training.batch_size`
- `training.lr`
- `training.weight_decay`
- `model.embedding_dim`
- `model.hidden_dim`
- `model.nhead`
- `model.nlayers`
- `model.dropout`

#### Type/range checks

- `seed`: int
- `dataset.max_walk_length > 0`
- `dataset.num_walks > 0`
- `training.epochs > 0`
- `training.batch_size > 0`
- `training.lr > 0`
- `training.weight_decay >= 0`
- `0.0 <= model.dropout < 1.0`
- `model.embedding_dim > 0`, `model.hidden_dim > 0`, `model.nlayers > 0`, `model.nhead > 0`

#### Consistency checks

- split ratios (if present): `train_ratio + mask_ratio + val_ratio + test_ratio == 1.0 ± 1e-6`
- transformer divisibility: `embedding_dim % nhead == 0`
- if `training.use_cuda=true` but CUDA unavailable, print warning (do not hard fail)
- class weights if present: length must match `num_classes` (if both exist)

#### Cache flags checks

- `preprocess.use_cache` and `preprocess.save` must be bool if present.

Error message style:

- Include field path + current value + expected constraint.
- Example: `Invalid config: model.dropout=1.2 (expected 0.0 <= dropout < 1.0)`

---

### 2) Validate at all main entrypoints

Add validation call immediately after loading config in:

- `run.py`
- `optuna_run.py`
- `run_posthoc.py`

Pattern:

```python
cfg = load_config(...)
validate_config(cfg, context="train")  # or "optuna" / "posthoc"
```

For posthoc context:

- still validate core fields needed for inference/posthoc.
- allow training-only fields to be optional only if truly unused.

---

### 3) Add config-only validation mode in `run.py`

Add CLI flag:

- `--dry-run-config`

Behavior:

- load config + apply overrides + validate
- print success summary
- exit 0 without preparing data/training

Example:

```bash
python run.py --config config.yaml --dry-run-config dataset.name=wiki-rfa
```

Expected output:

- `✅ Config validation passed (context=train)`

---

### 4) Keep backward compatibility

Do NOT break:

- current YAML structure
- CLI dotlist overrides
- dataset override merge via `configs/<dataset>.yaml`

No aggressive migration in this task.
If a field is missing but can be reasonably defaulted by existing code, keep behavior stable unless it causes silent critical issues.

---

### 5) Add tests for validator

Create lightweight tests in `tests/` for:

- valid config passes
- missing required key fails
- bad range fails (e.g., dropout=1.5)
- split ratio mismatch fails
- embedding_dim not divisible by nhead fails

Keep tests fast and focused.

---

## Out of scope (Do not do now)

- Full typed dataclass/pydantic migration of entire config tree.
- Rewriting all scripts around a new config abstraction.
- Major refactor of path/output architecture.

This task is a hardening pass, not a full redesign.

---

## Files to modify (expected)

- `src/utils/config.py` (main validator)
- `run.py` (dry-run + validate)
- `optuna_run.py` (validate)
- `run_posthoc.py` (validate)
- `tests/test_config_validation.py` (new)

Optional small docs update:

- brief note in `README.md` with `--dry-run-config` usage.

---

## Implementation quality requirements

- Keep code concise and readable.
- Use explicit checks with precise errors.
- Avoid silent fallback on clearly invalid values.
- Preserve existing seed strategy through `get_seed(cfg)`.

---

## Verification checklist

Run these after changes:

```bash
# 1) Syntax
python -m py_compile run.py optuna_run.py run_posthoc.py src/utils/config.py

# 2) Validator tests
pytest -q tests/test_config_validation.py

# 3) Dry-run success
python run.py --config config.yaml --dry-run-config dataset.name=wiki-rfa

# 4) Dry-run expected failure (example)
python run.py --config config.yaml --dry-run-config model.dropout=1.5
# Should fail with clear validation error

# 5) Entry points still launch
python run_posthoc.py --help
python optuna_run.py --help
```

---

## Acceptance criteria

- `validate_config()` exists and is used by all key entrypoints.
- Invalid configs fail early with clear messages.
- `run.py --dry-run-config` works.
- Tests cover key validation logic.
- No regressions in existing config loading/override flow.

---

## Final output expected from this chat

1. Code changes implementing validator + integration.
2. New/updated tests and their results.
3. Short summary of what was validated and any known limitations.
