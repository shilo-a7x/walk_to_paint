# Lead 4 — Sign-Entropy Heterogeneity (parts A, B, C)

One consolidated report for all three Lead-4 sub-investigations. **Part C (atomic
regression, 2026-06-29) is the current headline and supersedes the earlier
"it's the target node's in-edge entropy" framing of A/B/C-v1.** Equations for
everything here: [`LEAD4C_EQUATIONS.md`](LEAD4C_EQUATIONS.md).

## Central question

The walk-Transformer beats every GNN/SGNN baseline by 3–5 pp AUC on all 6 datasets
even though edge-sign mutual information collapses 10–1000× beyond 1 hop. Lead 4
asks: **is the walk's advantage explained by how the models cope with *sign
heterogeneity* (contested / inconsistent neighborhoods)?**

## TL;DR (the bottom line, from Part C)

**The walk's edge is direction-specific, not blanket robustness to entropy.**

- GNNs are hurt **more** than the walk only on **`tgt_in`** — the entropy of the
  *target* node's incoming signs ("how contested is v's reputation"): pooled
  β walk ≈ −1.9 vs GNN ≈ −3.0, the gap is negative on **100 % of datasets**
  (sign-test p = 0.031).
- On **`src_out`** — the entropy of the *source* node's outgoing signs ("how
  inconsistent a rater u is") — which is the **largest** entropy effect of all,
  the **walk is hurt *more*** than GINEConv (β walk ≈ −2.7 vs GINEConv ≈ −1.3).
- `tgt_out` and `src_in` are ≈ null; **2-hop/path entropy barely moves any model.**

So the phenomenon is a **source/target directional asymmetry**, *not* a uniform
"entropy hurts GNNs more." Averaging the entropy variants together cancels it
(see the composite caveat in Part C), which is why earlier single-number framings
were misleading.

## Shared methodology (all parts)

- **Same-edge ground truth.** Per-edge predictions for all 4 models
  (`walk_full`, `walk_localattn4`, `GINEConv`, `SiGAT`) come from
  `outputs/lead4_entropy_heterogeneity/predictions_raw_canonical.pkl`, restricted
  to the **shared edge set** (intersection of `(u,v)` = the E15 full-coverage test
  edges) via `load_shared_predictions`. Every comparison is on identical edges with
  identical ground truth and identical per-cell `n`.
- **Binary Shannon entropy** `H(p) = −p·log₂p − (1−p)·log₂(1−p)` (bits), `p` =
  positive-sign fraction. `H=0` homogeneous, `H=1` 50/50. Computed over **all
  edges** (train+val+test) of the canonical edge list — a *diagnostic grouping of
  already-trained models' predictions*, not a training feature → no leakage.
- Provenance/caveats: `CANONICAL_RERUN_FINDINGS.md`, `FABRICATED_REVERSE_EDGES.md`,
  `WALK_COVERAGE.md`.

---

## Part A — Node sign-entropy heterogeneity (bucket-AUC)

**`scripts/lead4_entropy_heterogeneity.py` → `outputs/lead4_entropy_heterogeneity/`**

Bin each edge by (source-node entropy × target-node entropy) into a 2D grid and
compare per-cell AUC across models. Four node-variants by direction:

| variant | source H uses | target H uses |
|---|---|---|
| `out_out` | u's out-edges | v's out-edges |
| `in_in` | u's in-edges | v's in-edges |
| `out_in` | u's out-edges | v's in-edges |
| `inout_inout` | all of u | all of v |

**Verdict (bucket-AUC):** the "GNNs degrade more than the walk as heterogeneity
rises" effect is **variant-dependent** — strong & robust for `in_in` (GINEConv's
AUC drop 2–10× the walk's, all bucket sizes, 5/6 datasets), weak/mixed for
`out_in`, intermediate for `out_out`/`inout_inout`. Survives the E15 full-coverage
rerun (resolving the earlier coverage-artifact worry). The puzzle this left —
*why `in_in`, mechanistically the less-obvious variant?* — is answered by Part C.

---

## Part B — 2-hop sign-path-consistency entropy

**`scripts/lead4_twohop_path_consistency.py` → `outputs/lead4_twohop_path_consistency/`**

Give each edge a heterogeneity score from 2-hop path-sign consistency (a path
`n→m→k` is *consistent* if both hop-signs match; `p` = consistent fraction →
`H(p)`). Variants: `out` (forward from target v), `in` (backward into source u),
`inout` (pool the counts, then entropy).

**Verdict:** on its own, 2-hop consistency entropy **does not corroborate** the
`in_in` node story — its variants are noisy and direction-mixed. Part C confirms
this quantitatively: the 2-hop terms are the smallest effects and `twohop_out`
even runs slightly *positive* for GNNs.

---

## Part C — Atomic-direction logistic regression (HEADLINE, 2026-06-29)

**`scripts/lead4c_entropy_logit_regression.py` → `outputs/lead4c_entropy_logit_regression/`**
(zip: `lead4c_atomic_outputs.zip`). Equations: `LEAD4C_EQUATIONS.md`.

### The combo problem, and the fix

A/B used **4 node-variants × 3 two-hop variants = 12 combos**, with no principled
reason to privilege any one and an awkward "null control" (`out_out`). But the 12
combos are just overlapping pairings of **6 atomic directional entropies** — node:
`src_out, src_in, tgt_out, tgt_in`; two-hop: `twohop_in@u, twohop_out@v` (the
`inout` ones are count-level pools of these). E.g. `in_in`, `out_in`,
`inout_inout` all re-use `tgt_in`.

**Atomic model:** enter all 6 atoms in ONE logistic regression of per-edge
correctness, per (dataset, model) and pooled (dataset fixed effects), two-way
cluster-robust SE on (u,v), BH-FDR. Each β is then a partial coefficient and
"which direction matters" is an *empirical output* — no combo choice, no assumed
control.

### Headline results (pooled, shared slope; β = log-odds of *correct* per +1 bit)

| atomic term | walk (full) | GNN (GINEConv) | who is hurt more | cross-dataset (GNN-more-neg) |
|---|---:|---:|---|---|
| **src_out** (u's out / inconsistent rater) | **−2.69** | −1.34 | **walk** (largest effect overall) | 17 % (n.s.) |
| src_in (others rate u) | ≈0 (n.s.) | −0.65 | gnn slightly | 83 % (n.s.) |
| tgt_out (v rates others) | −0.11 | ≈0 (n.s.) | ~null | 33 % (n.s.) |
| **tgt_in** (others rate v / contested reputation) | −1.93 | **−3.52** | **GNN** | **100 % (p=0.031)** |
| twohop_in (s→t→u) | −0.40 | −0.75 | gnn slightly | 33 % (n.s.) |
| twohop_out (v→m→k) | −0.12 | +0.17 | opposite signs | 0 % (p=0.031) |

All entropy effects are negative-or-null *within* a model (heterogeneity rarely
helps); the *story is the walk-vs-GNN gap per direction*. Figures:
`atomic_forest.png` (the one-glance figure, annotated), `atomic_heatmap.png`,
`atomic_binned_accuracy.png` (model-free), `atomic_explainer.png` (neutral schematic).

### Compact 2-number companion (count-pooled, NOT a variant average)

One node-entropy term (entropy of sign-counts pooled across both endpoints'
incident edges) and one path-entropy term (pooled 2-hop counts):
`composite_forest.png`, report §1b.

| term | walk | GNN |
|---|---:|---:|
| `b_node` | −3.3 | −4.3 |
| `b_path` | +0.22 | +0.51 |

**Caveat (important):** `b_node` pools `src_out` (walk-worse) with `tgt_in`
(GNN-worse). Because `tgt_in` dominates, the pooled node number looks "GNN-worse"
and **cancels** the asymmetry. Read it *with* the atomic forest, never alone.

### Pooling — shared vs interacted slope

- **Shared slope** (the headline table): one β per term for all datasets; datasets
  shift only the intercept (dataset fixed effects) → "is the effect the same
  everywhere?"
- **Interacted slope**: each dataset gets its own β (= the per-dataset fits) →
  "does it vary by dataset?" `atomic_pooling_caterpillar.png` overlays the two for
  all 6 terms.

### Model-free cross-check

Spearman ρ and binned MI between each atomic entropy and per-edge correctness
(`correlations.csv`, `correlation_heatmap.png`) reproduce the regression β signs —
the effect is not a logistic functional-form artifact.

### Appendix — the 12-combo sweep, explained

`appendix_combo_ranking.png`: the legacy `b_tgt` gap concentrates entirely in
combos whose target term uses v's IN-edges (= `tgt_in`); the `b_src` side is the
mirror image (walk-worse where source uses out-edges). The combo sweep is just the
atoms re-packaged. Full per-combo numbers: `fit_results.csv` (`spec == "marginal3"`).

### How Part C revises A/B

- The Part-A `in_in` result is real, but it was carrying the `tgt_in` signal
  (GNNs worse at contested target reputation) — that part holds.
- The earlier claim **"source entropy is weak / not where the effect lives" was an
  over-generalization** from the `in_in` headline combo (whose source term uses
  *in*-edges, genuinely ≈0). Atomically — and in the old `out_*`/`inout` combos —
  `src_out` is the single **largest** entropy effect, with the **walk** hurt more.
- Net: not "GNNs are worse with entropy," but a **directional asymmetry**
  (target-reputation contestedness hurts GNNs; source rater-inconsistency hurts the
  walk; paths barely matter).

### Reproduce

```
python scripts/lead4c_entropy_logit_regression.py --mode all   # compute → fit → plot
# fit_results.csv columns: spec ∈ {marginal3, atomic, composite}; per-term β, robust SE,
# p, p_fdr, beta_std, OR, pseudo_r2, n; pooled_spec ∈ {shared_slope, interacted}
```

Open follow-ups (Phase 2): formal `is_GNN × term` interaction (one differential
coefficient), degree controls (is `tgt_in` a hub-ness proxy?), per-term model
selection (AIC/BIC/LR).
