# Prompt: Config/CLI/YAML Overhaul (Requirements First, Staged Execution)

You are working in this repo to overhaul configuration reliability without breaking current workflows.

Important: Start by implementing only Stage 1 unless explicitly asked to continue.

---

## 0) Why this task exists

Current system is flexible and practical (base config + dataset overrides + CLI overrides), but needs stronger safety:
- no silent failures,
- no hidden key typos,
- explicit handling of runtime-derived keys,
- checkpoint load behavior must be deterministic and explainable.

---

## 1) Requirements to lock BEFORE coding (must satisfy all)

### R1. Layered config precedence must be explicit and stable
Target precedence:
1. root base config (`config.yaml`)
2. dataset override (`configs/<dataset>.yaml`)
3. optional profile layer (future stage)
4. CLI dotlist overrides (highest)

Result: every run has deterministic resolved config.

### R2. Two config classes are first-class citizens
- Input/user keys: expected from YAML/CLI.
- Runtime-derived keys: produced during pipeline (data prep / path resolver / metadata).

Runtime-derived keys must not be treated as required user inputs.

### R3. No silent failure policy
- Unknown keys in CLI or YAML must be caught (error or strict warning policy).
- Invalid ranges/types must hard fail before expensive work starts.
- Missing required keys must fail with clear actionable message.

### R4. Phase-aware validation contract
- Preflight validation: only input keys required.
- Post-prepare validation: derived keys must exist and be consistent.

### R5. Checkpoint compatibility contract
When loading model from checkpoint:
- Model architecture and label-space critical fields must be validated.
- If runtime-derived keys differ between checkpoint and current run, behavior must be explicit:
  - either strict fail, or
  - explicit compatibility mode with warnings and rules.

No hidden mismatch between checkpoint model head and current dataset/tokenizer state.

### R6. Reproducibility + provenance
Each run must save:
- final resolved input config,
- derived runtime config snapshot,
- compatibility checks summary.

### R7. Backward compatibility for active experiments
- Keep existing config loading style working.
- No forced large migration in Stage 1.

### R8. CLI ergonomics
- Add config dry-run mode.
- Add config explain mode showing final resolved values and sources when possible.

---

## 2) Clarified checkpoint logic requirement (critical)

Current behavior in this codebase includes dynamic fields such as:
- `model.vocab_size`, `model.num_classes`, `model.pad_id`, `model.ignore_index`, `model.class_weights`,
- path fields (`training.checkpoint_dir`, `training.log_dir`).

These are produced at runtime by data preparation and path resolution.

### Required checkpoint rule
On checkpoint load, the system must verify that checkpoint-embedded config and current runtime config are compatible for:
- `num_classes`
- tokenizer/label-space assumptions (`ignore_index`, potentially vocab-sensitive fields)
- model architecture (`embedding_dim`, `hidden_dim`, `nhead`, `nlayers`, dropout where relevant)

If incompatible, fail fast with exact mismatch details.

---

## 3) Staged implementation plan

## Stage 1 (Now): Safety Hardening (minimum risk)
Deliverables:
1. Central validator with preflight + post-prepare modes.
2. Entry-point enforcement in:
   - `run.py`
   - `optuna_run.py`
   - `run_posthoc.py`
3. `--dry-run-config` support in training entrypoint.
4. Unknown key detection policy (strict for CLI keys).
5. Unit tests for validator and key mismatch cases.

Out of scope Stage 1:
- full schema framework migration,
- splitting all configs by domain,
- advanced profile layering.

## Stage 2: Checkpoint Compatibility Layer
Deliverables:
1. Utility `validate_checkpoint_compatibility(ckpt_cfg, runtime_cfg)`.
2. Explicit compatibility report (pass/fail + field diffs).
3. Optional strictness flag:
   - strict mode default for training resume,
   - configurable for posthoc inference only.
4. Test matrix for mismatch scenarios.

## Stage 3: Config Architecture Upgrade (optional, later)
Deliverables:
1. optional profile layer (`configs/profiles/*.yaml`)
2. source-aware config explain output
3. stronger typed schema or pydantic/dataclass contract
4. frozen run manifest written to output dir

---

## 4) Required behavior details

### 4.1 Validation modes
- preflight-train
- preflight-optuna
- preflight-posthoc
- postprepare-train
- postprepare-posthoc

Each mode can require slightly different key sets.

### 4.2 Unknown key policy
- CLI unknown keys: hard error.
- YAML unknown keys: warning in Stage 1, hard error optional in Stage 2/3.

### 4.3 Dynamic key ownership
Derived keys are owned by runtime pipeline.
If user sets derived keys manually, either:
- warning and overwrite, or
- strict error (mode-dependent).

Must be explicit and consistent.

---

## 5) Testing requirements

Minimum tests:
1. valid config passes preflight.
2. missing required key fails.
3. bad range/type fails.
4. split ratio inconsistency fails.
5. transformer divisibility fails (`embedding_dim % nhead != 0`).
6. post-prepare missing derived keys fails.
7. checkpoint mismatch fails with precise diff.
8. dry-run returns success/failure correctly.

---

## 6) Acceptance criteria (must all pass)

1. No silent bad config proceeds to training.
2. Dynamic keys are handled by phase-aware validation, not mistaken as YAML-required inputs.
3. Checkpoint load path has explicit compatibility checks and clear failure modes.
4. Existing experiment workflows still run with minimal/no config migration.
5. Debugging usability improved via dry-run/explain outputs.

---

## 7) Execution instruction to the implementing chat

1. Implement Stage 1 fully.
2. Run tests and py_compile checks.
3. Show concise diff summary and any compatibility caveats.
4. Stop and request approval before Stage 2.

---

## 8) Notes for this repository context

- This repo already mixes static and dynamic config keys, so two-phase validation is mandatory.
- Aggregator integration is currently lower priority; do not expand scope into aggregator refactors.
- Keep changes surgical and low-risk while other experiment chats continue.
