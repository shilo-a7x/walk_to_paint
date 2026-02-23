# Master Task Status Review (Feb 23, 2026)

Purpose: one source of truth you can mark quickly so we can align on what is truly done vs pending.

How to use:

- For each task, update USER_CONFIRM to one of: DONE / PARTIAL / NOT_DONE / DEFERRED
- Add short notes in USER_NOTES
- If done, include evidence path (log/script/output)

---

## A) Foundation + Reproducibility

### T0.1 Cleanup old docs

- AGENT_STATUS: DONE
- EVIDENCE: cleanup artifacts referenced in prior status docs
- NEXT_ACTION: none
- USER_CONFIRM: DONE
- USER_NOTES:

### T0.2 Robust config system (validation/schema/safer loading)

- AGENT_STATUS: LIKELY PARTIAL (high-priority gap still reported in latest status docs)
- EVIDENCE: current docs still flag this as critical gap
- NEXT_ACTION:
  1) decide target approach (lightweight validator vs typed schema)
  2) enforce required keys + type/range checks
  3) fail-fast errors before training starts
- USER_CONFIRM: NOT_DONE
- USER_NOTES: This change should be done carefully as a lots of scripts and py files scattered across reop use the current config system

### T0.3 Output directory structure hardening

- AGENT_STATUS: PARTIAL
- EVIDENCE: outputs are structured by dataset/exp, but normalization rules may still be inconsistent across scripts
- NEXT_ACTION:
  1) define canonical run layout
  2) enforce in training + posthoc + tuning scripts
  3) add one resolver utility and use everywhere
- USER_CONFIRM: PARTIAL
- USER_NOTES:

### A1 Seed unification (canonical reproducibility.seed + get_seed)

- AGENT_STATUS: DONE
- EVIDENCE: src/utils/config.py get_seed present and used in prior work
- NEXT_ACTION: verify all newly added scripts still follow canonical seed path
- USER_CONFIRM: DONE
- USER_NOTES:

### A6 Stratified split strategy

- AGENT_STATUS: DONE (from prior implementation threads)
- EVIDENCE: documented as completed in multiple status docs
- NEXT_ACTION: regression check when changing data pipeline
- USER_CONFIRM: DONE
- USER_NOTES:

---

## B) Data Pipeline + Performance

### T1.1 Data reproducibility verification

- AGENT_STATUS: PARTIAL
- EVIDENCE: prior verification docs exist, but recent changes suggest re-check needed as guardrail
- NEXT_ACTION:
  1) run reproducibility smoke tests on at least one production dataset
  2) compare deterministic artifacts/checksums where applicable
- USER_CONFIRM: DONE
- USER_NOTES:

### T1.2 Data stage optimization

- AGENT_STATUS: PARTIAL to DONE (likely substantial progress)
- EVIDENCE: tokenizer optimization summary exists
- NEXT_ACTION: confirm remaining bottlenecks from latest profiling runs
- USER_CONFIRM: DONE
- USER_NOTES:

### T1.3 Dataset cache system

- AGENT_STATUS: DONE
- EVIDENCE: DATASET_CACHE_COMPLETE.md and active use of preprocess.use_cache in workflows
- NEXT_ACTION: ensure all experiment entry points enable cache consistently
- USER_CONFIRM: DONE
- USER_NOTES:

### Tokenizer optimization

- AGENT_STATUS: DONE
- EVIDENCE: TOKENIZER_OPTIMIZATION_SUMMARY.md with benchmarks
- NEXT_ACTION: optional follow-up only if profiling shows tokenizer as bottleneck again
- USER_CONFIRM: DONE
- USER_NOTES:

---

## C) Fairness / Imbalance

### T2.1 Analyze class imbalance + weighting strategy

- AGENT_STATUS: DONE
- EVIDENCE: prior analysis artifacts and implementation chain
- NEXT_ACTION: none
- USER_CONFIRM: DONE
- USER_NOTES:

### T2.2 Mandatory class-weighted loss

- AGENT_STATUS: DONE
- EVIDENCE: implementation completion doc + integrated config flow
- NEXT_ACTION: verify continues to work with latest training configs
- USER_CONFIRM: DONE
- USER_NOTES:

### T2.3 Leakage validation tests

- AGENT_STATUS: PARTIAL (tests exist, closure status unclear)
- EVIDENCE: referenced as ready in prior status docs
- NEXT_ACTION:
  1) run leakage tests in current code state
  2) record pass/fail and any edge cases
- USER_CONFIRM: DONE
- USER_NOTES:

---

## D) Prediction / Posthoc / Aggregation

### T3.1 Save per-occurrence predictions with metadata

- AGENT_STATUS: DONE
- EVIDENCE: implemented and documented in T3 progress docs
- NEXT_ACTION: none
- USER_CONFIRM: DONE
- USER_NOTES:

### T3.2 Save per-occurrence triplets

- AGENT_STATUS: DONE
- EVIDENCE: scripts/save_triplets.py and prior validation notes
- NEXT_ACTION: none
- USER_CONFIRM: DONE
- USER_NOTES:

### T3.3 Heatmap generation

- AGENT_STATUS: DONE
- EVIDENCE: scripts/plot_triplet_heatmap.py and produced outputs
- NEXT_ACTION: none
- USER_CONFIRM: DONE
- USER_NOTES:

### T4.1 Aggregator modeling (logistic/xgboost/lgbm paths)

- AGENT_STATUS: DONE for current practical scope
- EVIDENCE: AGGREGATOR_POSTHOC_SUMMARY.md + quick summary tooling + compile checks passed
- NEXT_ACTION: DEFER integration polish until needed
- USER_CONFIRM: DONE (*)
- USER_NOTES:

### T4.2/T4.3 Aggregator strategy analysis + comparison

- AGENT_STATUS: PARTIAL
- EVIDENCE: reporting scripts exist and results are available
- NEXT_ACTION: defer full formal comparison report until integration phase
- USER_CONFIRM: PARTIAL
- USER_NOTES: full integration and lightweight implementation in the future

---

## E) Training / Transformer Improvement Track (Current Priority)

### E1 Baseline lock for transformer experiments

- AGENT_STATUS: IN_PROGRESS (you are actively experimenting in another chat)
- EVIDENCE: current workflow indicates active profiling/training/log review
- NEXT_ACTION:
  1) freeze one baseline config per target dataset
  2) capture baseline metrics table (train/val/test + overfit gaps)
- USER_CONFIRM: IN_PROGRESS
- USER_NOTES:

### E2 Incremental anti-overfit experiments

- AGENT_STATUS: IN_PROGRESS
- EVIDENCE: prepared prompt exists for incremental experiments
- NEXT_ACTION:
  1) run single-factor ablations first (capacity, dropout, wd, patience)
  2) keep only winners by val/test + gap reduction
- USER_CONFIRM: IN_PROGRESS
- USER_NOTES:

### E3 Representation experiments

- AGENT_STATUS: READY (not yet consolidated)
- EVIDENCE: ideas queued (dynamic masking, fake tokens, node dropout, curriculum)
- NEXT_ACTION:
  1) implement feature flags for each idea
  2) test one idea at a time after anti-overfit basics
- USER_CONFIRM: IN_PROGRESS
- USER_NOTES:

---

## F) Recommended Priority Order From Now

1. Keep transformer experiment loop active (baseline + anti-overfit basics).
2. Close T2.3 leakage validation with explicit pass/fail evidence.
3. Stabilize T0.2 config hardening (minimum fail-fast validator).
4. Keep aggregator work deferred unless integration is required.
5. After top transformer gains are reproducible, run multi-dataset confirmation.

---

## Fast Reconciliation Table (Fill This)

| Task Group | Your Verdict | Notes |
|---|---|---|
| Foundation (T0.x + A1/A6) |  |  |
| Data pipeline (T1.x + tokenizer/cache) |  |  |
| Fairness (T2.1-T2.3) |  |  |
| Posthoc/Aggregator (T3/T4) |  |  |
| Transformer improvement track |  |  |
