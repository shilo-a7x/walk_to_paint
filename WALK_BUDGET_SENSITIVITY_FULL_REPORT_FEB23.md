# Walk Budget Sensitivity — Full Consolidated Report (Feb 23, 2026)

## 1) Scope and Constraints

This report consolidates **all experiments and outcomes from this conversation only**.

- No new runs were executed for this report.
- All metrics are derived from current cached `dataset_cache.pt` files under `data/`.
- Datasets analyzed:
  - `wiki-rfa`
  - `epinions`
  - `slashdot090221`

Primary question:

- Can walk count be reduced from **5M** to **1M or lower** while preserving similar:
  - node coverage
  - walk-length distribution
  - overall behavior

---

## 2) Experiment Inventory (What Was Done)

### A. Node coverage at current cache scale (5M)

Script used:

- `verify_node_coverage.py`

Purpose:

- Compare all graph nodes (from edges) vs nodes visited in sampled walks.

### B. Graph connectivity analysis

Script used:

- `verify_graph_connectivity.py`

Purpose:

- Determine whether each graph is connected or fragmented.

### C. Walk-length distribution analysis (5M)

Script used:

- `analyze_walk_lengths.py`

Purpose:

- Characterize distribution of edge-count per walk and termination behavior.

### D. Budget sensitivity sweep vs 5M baseline

Script used:

- `scripts/analyze_walk_budget_sensitivity.py`

Purpose:

- Compare reduced budgets to baseline using:
  - coverage %
  - mean/median edges per walk
  - short-walk rate (`<=2` edges)
  - max-length hit rate (`==80` edges)

Sweeps executed:

1. Broad low-budget sweep:
   - 1,000,000 / 750,000 / 500,000 / 250,000 / 100,000
2. Intermediate sweep:
   - 3,000,000 / 2,500,000 / 2,000,000 / 1,750,000 / 1,500,000 / 1,250,000 / 1,000,000
3. High-budget sweep:
   - 4,500,000 / 4,000,000 / 3,500,000 / 3,000,000

---

## 3) Baseline Results at 5M Walks

## 3.1 Node Coverage (5M)

| Dataset | Covered / Total Nodes | Coverage | Uncovered |
|---|---:|---:|---:|
| wiki-rfa | 11,256 / 11,256 | 100.00% | 0 |
| epinions | 131,406 / 131,580 | 99.87% | 174 |
| slashdot090221 | 81,759 / 82,140 | 99.54% | 381 |

Notes from analysis:

- Uncovered nodes in `epinions` and `slashdot090221` are predominantly low-degree (often 1–3).

## 3.2 Connectivity

| Dataset | Connected? | #Components | Main Component Size |
|---|---|---:|---:|
| wiki-rfa | Yes | 1 | 11,256 (100.00%) |
| epinions | No | 5,568 | 119,130 (90.54%) |
| slashdot090221 | Yes | 1 | 82,140 (100.00%) |

Additional `epinions` detail:

- 5,567 non-main components contain 12,450 nodes total.
- This fragmentation strongly explains persistent uncovered nodes at any finite walk budget.

## 3.3 Walk-Length Distribution (5M)

(Lengths below are **edges per walk**, not token count.)

| Dataset | Mean | Median | P(short <=2) | P(hit max=80) |
|---|---:|---:|---:|---:|
| wiki-rfa | 9.56 | 6.0 | 21.37% | 1.19% |
| epinions | 14.62 | 8.0 | 21.70% | 4.47% |
| slashdot090221 | 6.18 | 3.0 | 47.64% | 2.87% |

Interpretation:

- `epinions` supports longer traversals (higher mean, higher max-hit rate).
- `slashdot090221` terminates quickly very often (high short-walk rate).
- `wiki-rfa` is moderate and stable.

---

## 4) Budget Sensitivity Results vs 5M Baseline

### Similarity rule used during analysis

Strict rule:

- `|Δcoverage| <= 0.10 percentage points`
- `|Δmean_edges| <= 0.50`

This rule was used to classify “near-equivalent” budgets.

---

## 4.1 Broad Low-Budget Sweep (1M to 100k)

### wiki-rfa

| Budget | Coverage | ΔCoverage (pp) | Mean Edges | ΔMean |
|---:|---:|---:|---:|---:|
| 5,000,000 | 100.00% | +0.000 | 9.56 | +0.000 |
| 1,000,000 | 99.99% | -0.009 | 9.55 | -0.009 |
| 750,000 | 99.99% | -0.009 | 9.55 | -0.009 |
| 500,000 | 99.97% | -0.027 | 9.55 | -0.003 |
| 250,000 | 99.95% | -0.053 | 9.58 | +0.020 |
| 100,000 | 99.85% | -0.151 | 9.61 | +0.053 |

### epinions

| Budget | Coverage | ΔCoverage (pp) | Mean Edges | ΔMean |
|---:|---:|---:|---:|---:|
| 5,000,000 | 99.87% | +0.000 | 14.62 | +0.000 |
| 1,000,000 | 98.87% | -0.994 | 14.61 | -0.011 |
| 750,000 | 98.37% | -1.496 | 14.63 | +0.010 |
| 500,000 | 97.22% | -2.645 | 14.67 | +0.045 |
| 250,000 | 91.97% | -7.900 | 14.69 | +0.062 |
| 100,000 | 74.67% | -25.195 | 14.63 | +0.006 |

### slashdot090221

| Budget | Coverage | ΔCoverage (pp) | Mean Edges | ΔMean |
|---:|---:|---:|---:|---:|
| 5,000,000 | 99.54% | +0.000 | 6.18 | +0.000 |
| 1,000,000 | 94.63% | -4.907 | 6.19 | +0.006 |
| 750,000 | 92.97% | -6.562 | 6.18 | -0.002 |
| 500,000 | 90.28% | -9.257 | 6.17 | -0.008 |
| 250,000 | 84.72% | -14.814 | 6.17 | -0.006 |
| 100,000 | 74.07% | -25.471 | 6.18 | -0.001 |

Key observation from low-budget sweep:

- Distribution shape metrics (mean/median/short/max-hit) are stable.
- Coverage is the dominant failure mode at low budgets for `epinions` and `slashdot090221`.

---

## 4.2 Intermediate Sweep (3.0M down to 1.0M)

### wiki-rfa

| Budget | Coverage | ΔCoverage (pp) | Mean Edges | ΔMean |
|---:|---:|---:|---:|---:|
| 3,000,000 | 99.99% | -0.009 | 9.56 | +0.002 |
| 2,500,000 | 99.99% | -0.009 | 9.56 | +0.001 |
| 2,000,000 | 99.99% | -0.009 | 9.55 | -0.001 |
| 1,750,000 | 99.99% | -0.009 | 9.55 | -0.002 |
| 1,500,000 | 99.99% | -0.009 | 9.55 | -0.005 |
| 1,250,000 | 99.99% | -0.009 | 9.54 | -0.010 |
| 1,000,000 | 99.99% | -0.009 | 9.55 | -0.009 |

### epinions

| Budget | Coverage | ΔCoverage (pp) | Mean Edges | ΔMean |
|---:|---:|---:|---:|---:|
| 3,000,000 | 99.72% | -0.148 | 14.62 | -0.003 |
| 2,500,000 | 99.65% | -0.217 | 14.62 | -0.003 |
| 2,000,000 | 99.54% | -0.328 | 14.62 | -0.005 |
| 1,750,000 | 99.45% | -0.414 | 14.62 | -0.003 |
| 1,500,000 | 99.36% | -0.512 | 14.61 | -0.013 |
| 1,250,000 | 99.18% | -0.689 | 14.61 | -0.010 |
| 1,000,000 | 98.87% | -0.994 | 14.61 | -0.011 |

### slashdot090221

| Budget | Coverage | ΔCoverage (pp) | Mean Edges | ΔMean |
|---:|---:|---:|---:|---:|
| 3,000,000 | 98.70% | -0.834 | 6.18 | -0.000 |
| 2,500,000 | 98.24% | -1.297 | 6.18 | -0.002 |
| 2,000,000 | 97.57% | -1.971 | 6.18 | -0.000 |
| 1,750,000 | 97.10% | -2.432 | 6.18 | +0.000 |
| 1,500,000 | 96.55% | -2.983 | 6.18 | -0.002 |
| 1,250,000 | 95.75% | -3.784 | 6.17 | -0.005 |
| 1,000,000 | 94.63% | -4.907 | 6.19 | +0.006 |

Intermediate sweep finding:

- `epinions` improves substantially above 3M but still misses strict 0.10pp at 3M.
- `slashdot090221` remains clearly sensitive below ~4.5M.

---

## 4.3 High-Budget Sweep (4.5M to 3.0M)

### wiki-rfa

| Budget | Coverage | ΔCoverage (pp) | Mean Edges | ΔMean |
|---:|---:|---:|---:|---:|
| 4,500,000 | 100.00% | +0.000 | 9.55 | -0.002 |
| 4,000,000 | 100.00% | +0.000 | 9.56 | +0.000 |
| 3,500,000 | 99.99% | -0.009 | 9.56 | +0.001 |
| 3,000,000 | 99.99% | -0.009 | 9.56 | +0.002 |

### epinions

| Budget | Coverage | ΔCoverage (pp) | Mean Edges | ΔMean |
|---:|---:|---:|---:|---:|
| 4,500,000 | 99.84% | -0.027 | 14.62 | -0.002 |
| 4,000,000 | 99.81% | -0.062 | 14.62 | -0.007 |
| 3,500,000 | 99.77% | -0.099 | 14.61 | -0.009 |
| 3,000,000 | 99.72% | -0.148 | 14.62 | -0.003 |

### slashdot090221

| Budget | Coverage | ΔCoverage (pp) | Mean Edges | ΔMean |
|---:|---:|---:|---:|---:|
| 4,500,000 | 99.38% | -0.153 | 6.18 | +0.003 |
| 4,000,000 | 99.25% | -0.286 | 6.18 | +0.004 |
| 3,500,000 | 99.00% | -0.533 | 6.18 | +0.000 |
| 3,000,000 | 98.70% | -0.834 | 6.18 | -0.000 |

High-budget finding:

- `epinions` crosses strict threshold at **3.5M**.
- `slashdot090221` does **not** reach strict threshold even at 4.5M.

---

## 5) Consolidated Interpretation

## 5.1 Why 1M works for one dataset but not others

- `wiki-rfa`
  - connected and comparatively compact
  - coverage saturates quickly

- `epinions`
  - strongly fragmented (5,568 components)
  - lower budgets undersample tails/small components more aggressively

- `slashdot090221`
  - connected but high short-walk termination rate (47.64% at baseline)
  - coverage is sensitive to walk count despite stable mean/median distribution

## 5.2 Distribution similarity vs coverage similarity

A critical distinction from these experiments:

- Walk-length distribution statistics can stay very stable while node coverage drops materially.
- Therefore, **coverage must be a first-class acceptance criterion**, not just mean/median length.

---

## 6) Recommendations

## 6.1 Strict recommendation (preserve baseline behavior)

Using strict criterion (`|Δcoverage| <= 0.10pp`, `|Δmean| <= 0.50`):

- `wiki-rfa`: **1.0M** is acceptable.
- `epinions`: **3.5M** is minimum acceptable from tested points.
- `slashdot090221`: keep **5.0M** (no tested lower point satisfied strict coverage threshold).

## 6.2 Relaxed option (if small coverage loss is acceptable)

If up to ~0.20pp coverage loss is acceptable:

- `epinions`: 3.0M may be acceptable (`-0.148pp`).
- `slashdot090221`: 4.5M still at `-0.153pp` (closest tested reduction).

## 6.3 Single global walk budget

If one budget must be used for all datasets:

- Choose **5.0M** to avoid quality regression on `slashdot090221` and `epinions` tails.

If per-dataset budget is allowed:

- Recommended:
  - `wiki-rfa`: 1,000,000
  - `epinions`: 3,500,000
  - `slashdot090221`: 5,000,000

---

## 7) Proposed Config Changes (Not Applied Yet)

Per your instruction, no YAML files were patched.

Candidate values to apply later (when approved):

- `configs/wiki-rfa.yaml` → `dataset.num_walks: 1000000`
- `configs/epinions.yaml` → `dataset.num_walks: 3500000`
- `configs/slashdot090221.yaml` → keep `dataset.num_walks: 5000000`

---

## 8) Methodological Notes and Caveats

1. **All analysis uses existing cache ordering**
   - Reduced-budget comparisons used the first `N` walks from cached 5M sequences.
   - Because walk generation is deterministic with seeded indexing, this is consistent and reproducible.

2. **No retraining included**
   - This report evaluates preprocessing/coverage/distribution behavior only.
   - Final model quality impact (AUC/F1/etc.) should be validated before production rollout.

3. **Coverage floor is dataset-dependent**
   - Fragmentation (`epinions`) and quick-termination dynamics (`slashdot090221`) drive stronger sensitivity.

---

## 9) Artifacts Created in This Conversation

- `verify_node_coverage.py`
- `verify_graph_connectivity.py`
- `analyze_walk_lengths.py`
- `scripts/analyze_walk_budget_sensitivity.py`
- This report: `WALK_BUDGET_SENSITIVITY_FULL_REPORT_FEB23.md`

---

## 10) Final Decision Summary

- Reducing all datasets to 1M is **not** recommended.
- Best currently supported per-dataset choice from available evidence:
  - `wiki-rfa`: 1.0M
  - `epinions`: 3.5M
  - `slashdot090221`: 5.0M
- No configuration files changed yet, per request.
