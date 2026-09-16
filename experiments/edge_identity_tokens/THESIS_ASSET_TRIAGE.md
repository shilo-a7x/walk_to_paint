# Thesis asset triage — WSDM paper figures/tables vs. EID

Per `~/.claude/plans/plan-eid-multiseed-thesis.md` Phase 3a. For every current figure/table
asset in `aaai2027/`, this triages whether it needs to be re-derived against EID
(`experiments/edge_identity_tokens/`) instead of production, and what script work that would
take. **Read-only investigation — no files in `aaai2027/` or `experiments/edge_identity_tokens/`
(other than this one) were touched.**

Grounding: every row below is based on actually opening the named `extract_<name>.py` script
and reading what it loads, not inferred from the figure's title. Row names mirror the paper's
own figure/table names (see `aaai2027/PEWTER_ASSETS_CHECKLIST.md`).

## Key constraints established this pass

- **EID's Phase 1 posthoc (`experiments/edge_identity_tokens/eid_posthoc.py`) saves the SAME
  raw per-walk-occurrence prediction pickle schema and path convention as production's
  `run_posthoc.py`** (it literally imports `prediction_path`/`run_aggregator` from
  `run_posthoc.py` and writes `pickle.dump(pred, f)` with fields `edge_ids`, `walk_ids`,
  `positions`, `walk_lengths`, `predictions`, `probabilities`, `targets`, `correct`,
  `dist_from_start`, `dist_from_end`, `auc` — an exact match to what production's
  `test_predictions.pkl` contains). This means **any figure/table that only needs raw
  per-edge/per-walk predictions (not multiple aggregator summaries) can be reproduced against
  EID with just a glob-pattern change** (`outputs/<ds>/EID_MULTISEED_s<seed>_noablation_*/...`
  instead of `outputs/<ds>/MULTISEED_s<seed>_local_*/...`), once Phase 1's campaign lands.
- **EID's campaign only runs `run_aggregator` with `--agg-models func_logit_power`** — so
  `posthoc/multiseed_agg/aggregator/<func>/summary.txt` only exists for `func_logit_power`,
  not the other 10 functions production computed. Any figure/table needing multiple
  aggregators (Ablation B) needs a **new posthoc rerun** with `--agg-models` set to all 11 —
  this is cheap (aggregator fitting is closed-form numpy over already-saved raw predictions,
  no retraining/no GPU-heavy step) but is real, not-yet-done work beyond Phase 1 as currently
  scoped.
- **Attention-weight introspection (`scripts/attention_directionality.py`,
  `scripts/shap_edge_directionality.py`) is architecturally portable but not yet ported.**
  `EdgeIdentityTransformerModel` (`experiments/edge_identity_tokens/eid_src/model/model.py`)
  subclasses production's `TransformerModel` and reuses `self.transformer` (the same
  `LocalAttentionEncoderLayer` stack) unchanged — so the attention-weight-recording subclass
  trick (`LocalAttentionRecordingEncoderLayer`) would work mechanically the same way. But the
  loader is different (`EIDLitEdgeClassifier.load_from_checkpoint(ckpt_path, cfg=cfg, ...)`
  needs an explicit `cfg`, unlike production's `LitEdgeClassifier.load_from_checkpoint(ckpt_path)`),
  and EID's token sequence has a third channel not present in production — edge-identity
  tokens with their own embedding path (`_content_embed`'s `is_edge_tok` branch, separate from
  ordinary node/edge-sign tokens). The existing node-vs-edge attention-mass bucketing logic in
  `attention_directionality.py` was written before this channel existed and has not been
  checked against it. This is real, non-trivial porting work, not a path swap.

## Table 1 (main results, `tab:result1`)

- **EID-dependent?** Yes.
- **Why**: `extract_result1_walk_aucs.py` reads Pewter's own posthoc summaries directly
  (`outputs/<ds>/E31_PY314_MIGRATION_*/...` for full attention, `E32_PY314_LOCALATTN4_*` for
  LocalAttn4), both `func_logit_power` only. `table1_paired_significance.py` additionally
  pulls Pewter's 10-seed AUCs (`MULTISEED_s<seed>_{full,local}_*`) paired against the best
  baseline per dataset. Both are entirely "our model" data — the baseline side (SiGAT/SNEA/
  GCN/GAT/etc.) is untouched.
- **Reproduction script(s)**: `scripts/paper_figures/extract_result1_walk_aucs.py` (needs a new
  EID-specific variant, or a parameterized glob swap, pointed at
  `outputs/<ds>/EID_MULTISEED_s42_noablation_*/posthoc/multiseed_agg/aggregator/
  func_logit_power/summary.txt`), and `experiments/edge_identity_tokens/eid_significance.py`
  (already written, `--mode vs_production`-style machinery generalizes directly to "EID vs.
  best baseline" — needs a new mode/script mirroring `table1_paired_significance.py`, reusing
  `eid_seed_auc()`).
- **Blocking gaps**: none beyond Phase 1 completing (needs all 10 seeds × 6 datasets ×
  `noablation` condition). Data is `func_logit_power` only, but Table 1 only ever reported
  `func_logit_power` for production too, so no aggregator gap here.

## Ablation A (full vs. local attention + vertex/edge/direction, `tab:ablationA` region)

- **EID-dependent?** Partial / not directly applicable as currently scoped.
- **Why**: `extract_ablationA_full_vs_local_significance.py` compares production's full-attention
  vs. LocalAttn4 variants (`E31_PY314_MIGRATION` vs. `E32_PY314_LOCALATTN4` tags) — EID has no
  "full attention" variant in Phase 1's scope (its architecture search already picked a fixed
  per-dataset architecture+budget, not a full-vs-local axis), so this specific comparison has
  no EID analogue unless the thesis chooses to re-run an EID full-attention condition (not
  planned in Phase 1/1's plan). `extract_ablation_masknode_maskedge_significance.py` compares
  production's `ABLATION_MASKNODE`/`ABLATION_MASKEDGE` single-flag ablations against the local
  baseline — EID's analogous ablations are `mask_context_edges` and `mask_context_sign_only`
  (Phase 0's replacement for `scramble_edge_signs`), a different, EID-native design, not a
  reuse of the same flags (`mask_edge_tokens` was explicitly dropped in favor of
  `mask_context_edges`, per the plan file). `randomize_walk_direction`/dirflip and
  `mask_node_tokens` are both confirmed portable to EID with **no new code** (per the plan's
  Phase 3b note — inherited unmodified from `LitEdgeClassifier`/implemented at the data-loading
  level) but are explicitly **not yet run** (Phase 3b, scoped later than 3a).
- **Reproduction script(s)**: EID's own ablation table needs a **new script**, structurally
  mirroring `extract_ablation_masknode_maskedge_significance.py` but pointed at
  `outputs/<ds>/EID_MULTISEED_s<seed>_{maskcontextedges,signonly}_*/posthoc/multiseed_agg/
  aggregator/func_logit_power/summary.txt` vs. `..._noablation_*`. Statistical test machinery:
  `experiments/edge_identity_tokens/eid_significance.py --mode ablation` is already written and
  directly usable for `mask_context_edges`/`mask_context_sign_only` once Phase 1 lands.
- **Blocking gaps**: the full-vs-local comparison itself doesn't carry over conceptually (no
  EID full-attention run exists or is planned). The dirflip/mask_node_tokens ports are
  Phase-3b work — real training jobs not yet launched, not just a reproduction-script gap.

## Ablation B (`tab:ablationB`, all 11 aggregators)

- **EID-dependent?** Yes.
- **Why**: `extract_ablationB_multiseed.py` reads all 11 `func_*` aggregator summaries per
  seed from Pewter's LocalAttn4 multiseed posthoc dirs
  (`MULTISEED_s<seed>_local_*/posthoc/{ablationB_e32,multiseed_agg}/aggregator/<func>/
  summary.txt`). Purely "our model" data, no baseline involved.
- **Reproduction script(s)**: needs a new EID-specific variant of
  `extract_ablationB_multiseed.py`, **AND a new posthoc rerun on EID checkpoints with
  `--agg-models` set to all 11 functions** (`eid_posthoc.py --agg-models func_uniform,
  func_conf_power,...`) — Phase 1's campaign only computed `func_logit_power`. This rerun is
  cheap (closed-form aggregator fit over already-saved raw predictions, no retraining) but has
  not been done.
- **Blocking gaps**: the 11-aggregator posthoc rerun on EID's `noablation` checkpoints (60
  seed×dataset cells) is not yet done — this is the single most concrete "needs new EID-side
  work" item found in this triage, and it's cheap/short relative to retraining.

## Empirical Confirmation — Panel A (entropy asymmetry boxplot)

- **EID-dependent?** No.
- **Why**: `extract_empconf_panelA_entropy_boxplot.py` computes per-node H_out/H_in directly
  from the canonical raw edge list (`load_edges_canonical`, `build_sign_dicts`,
  `entropy_lookup`) — pure graph-structural sign entropy, no trained-model predictions
  anywhere in the script.
- **Reproduction script(s)**: none needed; reuse as-is.
- **Blocking gaps**: none.

## Empirical Confirmation — Panel B (MI decay with BFS distance)

- **EID-dependent?** No.
- **Why**: `extract_empconf_panelB_mi_decay.py` is a pure read-only transcription of
  `outputs/mi_analysis_package.zip`'s pre-computed MI-vs-distance report — a structural MI
  computation over the graph, no model predictions.
- **Reproduction script(s)**: none needed; reuse as-is.
- **Blocking gaps**: none.

## Empirical Confirmation — Panel C (GNN entropy heatmap: SiGAT + GINEConv)

- **EID-dependent?** No.
- **Why**: `extract_empconf_panelC_gnn_entropy_heatmap.py` compares only SiGAT (10-split, via
  `sigat_raw_seed`/`per_seed_grids`) and GINEConv (single-split, from
  `outputs/lead4_entropy_heterogeneity/computed_data.pkl`). Neither Pewter nor EID appears in
  this panel at all — it's explicitly the "GNN-only" comparison (Result 2 / Panel D of the
  attndir figure is where the walk model's own version lives, see below).
- **Reproduction script(s)**: none needed; reuse as-is.
- **Blocking gaps**: none.

## Empirical Confirmation — Panel D (4-way sign-agreement AUC)

- **EID-dependent?** No.
- **Why**: `extract_empconf_panelD_signagreement_auc.py` buckets predictions by neighbor
  sign-agreement using only SiGAT's 10-seed predictions (`sigat_raw_seed`/
  `sigat_raw_seed_cached`) against the real canonical edge list for adjacency. No walk-model
  (production or EID) predictions are loaded anywhere in this script.
- **Reproduction script(s)**: none needed; reuse as-is.
- **Blocking gaps**: none.

## Empirical Confirmation — Panel E (SiGAT logistic-regression coefficients)

- **EID-dependent?** No.
- **Why**: `extract_empconf_panelE_coefficients.py` reshapes an already-computed SiGAT-only
  multiseed regression export (`outputs/lead4c_sigat_multiseed_node4_export/results/
  aggregated_summary.csv`). No Pewter/EID data involved.
- **Reproduction script(s)**: none needed; reuse as-is.
- **Blocking gaps**: none.

## Figure 3 — Delta heatmap (Pewter vs. SiGAT, entropy-binned) + entropy regression

- **EID-dependent?** Yes.
- **Why**: `extract_multiseed_entropy_heatmaps.py`'s `walk_raw_seed()` reads Pewter LocalAttn4's
  raw per-walk predictions (`outputs/<ds>/MULTISEED_s<N>_local_*/checkpoints/<ds>_predictions/
  epoch_*/test_predictions.pkl`) and computes `pewter_sigat_delta_heatmap.csv` (delta = walk AUC
  − SiGAT AUC per entropy cell). `delta_heatmap_entropy_regression.py` (the statistical
  regression backing the figure's claim) and `extract_figure3_pewter_vs_sigat_significance.py`
  (the paired-significance companion test) both build directly on the same `walk_raw_seed()`
  and Pewter posthoc-summary reads. SiGAT's side is unaffected (cached raw predictions, no
  Pewter/EID involvement there).
- **Reproduction script(s)**: because `eid_posthoc.py` writes the exact same
  `test_predictions.pkl` schema/path convention as production (see "Key constraints" above),
  this is a **glob-pattern-only change** in `walk_raw_seed()` (or a small EID-specific sibling
  function) pointed at `EID_MULTISEED_s<N>_noablation_*` instead of `MULTISEED_s<N>_local_*`,
  and `pewter_seed_auc()` in `extract_figure3_pewter_vs_sigat_significance.py` similarly
  redirected via `eid_seed_auc()`-style lookup (already implemented in
  `experiments/edge_identity_tokens/eid_significance.py`). No new attention/model-internals code
  needed — this whole figure operates purely on scored probabilities per edge, not attention
  weights.
- **Blocking gaps**: none beyond Phase 1 completing (`noablation` condition, all 10 seeds, all
  6 datasets).

## Figure 4 — Attention Directionality Panels A/B/C (+D) and SHAP

- **EID-dependent?** Yes, and this is the deepest-blocked group in the whole triage.
- **Why**: Panels A, B, C all read `outputs/attention_directionality/
  attention_directionality_<ds>_local_result.pkl`, produced by `scripts/
  attention_directionality.py`, which loads a checkpoint via
  `LitEdgeClassifier.load_from_checkpoint(ckpt_path, map_location="cpu")` and monkeypatches
  `LocalAttentionEncoderLayer`/`nn.MultiheadAttention` subclasses
  (`LocalAttentionRecordingEncoderLayer`, `RecordingMultiheadAttention`-equivalent) to capture
  real `[B, nhead, S, S]` attention weight tensors during a forward pass, then buckets mass by
  token-offset/direction/node-vs-edge. Panel D
  (`extract_attndir_panelD_pewter_entropy_heatmap.py`) reads Pewter's own entropy-binned AUC
  heatmap from `outputs/lead4_entropy_heterogeneity_e32/computed_data.pkl` (built via
  `baselines/postprocess_canonical_e32.py` + `lead4_entropy_heterogeneity.py --mode compute`,
  Pewter LocalAttn4 raw predictions). The SHAP companion
  (`extract_shap_edge_directionality.py`) reads `outputs/shap_edge_directionality/
  shap_directionality_<ds>_result.pkl` from `scripts/shap_edge_directionality.py`, which does
  exact Shapley attribution over LocalAttn4's local-window attention context — same
  `LitEdgeClassifier`/`LocalAttentionEncoderLayer` dependency chain as Panels A/B/C, imported
  directly (`from scripts.attention_directionality import load_model_and_dataset`).
- **Reproduction script(s)**: none exist yet for EID. Would need:
  1. An EID-specific loader mirroring `load_model_and_dataset()`, using
     `EIDLitEdgeClassifier.load_from_checkpoint(ckpt_path, cfg=cfg, ...)` (note: needs an
     explicit `cfg` argument unlike production's plain call — see `eid_posthoc.py`'s pattern
     for how to recover it from the checkpoint).
  2. The attention-recording subclass trick is mechanically portable — confirmed
     `EdgeIdentityTransformerModel` (`experiments/edge_identity_tokens/eid_src/model/model.py`)
     subclasses `TransformerModel` and reuses `self.transformer`'s
     `LocalAttentionEncoderLayer` stack unchanged (same `self_attn` structure) — but has not
     been tried.
  3. **The node-vs-edge/token-offset bucketing logic needs re-examination**: EID's token
     sequence has a third content channel (edge-identity tokens, `is_edge_tok` branch in
     `EdgeIdentityTransformerModel._content_embed`) not present in production's plain
     node/edge-sign alternation. Whether `attention_directionality.py`'s existing node/edge
     classification still partitions EID's sequence correctly (or silently mis-buckets
     identity-token positions) has not been checked and would need real verification, not
     just a mechanical port.
  4. Panel D and SHAP additionally need an EID-side rebuild of the
     `lead4_entropy_heterogeneity`-style computed_data.pkl / raw prediction pipeline pointed at
     EID's predictions (straightforward once (1)-(3) exist, since it only needs scored
     probabilities, not attention weights).
- **Blocking gaps**: this whole figure group needs **new code**, not just a checkpoint swap —
  the single biggest "needs new EID-side work" item in the triage, larger than the Ablation B
  aggregator rerun. Should be scoped as its own task, not assumed to fall out of Phase 1 data.

## Pipeline schematic (Figure, `figures/pipeline_schematic.png`)

- **EID-dependent?** Yes (conceptually), but not script-driven.
- **Why**: this is a hand-built `.drawio` diagram (`aaai2027/figures/pipeline_schematic.drawio`)
  illustrating production's token layout/architecture, not something an `extract_*.py` script
  generates. If EID becomes "production" in the thesis, the diagram would need a manual redraw
  to show the edge-identity-token channel — not a script/data reproduction task at all.
- **Reproduction script(s)**: n/a — manual diagram edit, out of scope for this triage's
  script-reproduction framing.
- **Blocking gaps**: none blocking Phase 1/3a data work; this is a separate manual-design task
  whenever the thesis write-up gets to that figure.

## Dataset stats / test-set counts (supporting data, not a headline figure but feeds SE bars)

- **EID-dependent?** No.
- **Why**: `compute_dataset_stats.py` (size/density/degree/sign-balance/reciprocity/
  connectivity) and `extract_test_set_counts.py` (n_pos/n_neg per dataset, feeding the
  Hanley-McNeil SE used throughout Table 1) both read only the canonical edge list / canonical
  split file (`baselines/splits_canonical/<ds>.pt`) — no trained-model predictions.
- **Reproduction script(s)**: none needed; reuse as-is.
- **Blocking gaps**: none. (Note: EID's own posthoc summaries already carry a Hanley-McNeil-
  style SE if computed the same way `extract_result1_walk_aucs.py` does — the n_pos/n_neg
  input doesn't change since it's the same canonical split.)

## Ablation-adjacent extras: K-ablation, F1 scores (`extract_ablation_kwalks.py`,
## `extract_ablation_f1.py`)

- **EID-dependent?** Yes, if the thesis wants to reproduce these for EID (not yet listed as a
  named figure/table in the checklist, but present in `scripts/paper_figures/` and referenced
  from Ablation A's own text).
- **Why**: both read raw per-walk `test_predictions.pkl` files from Pewter's
  `MULTISEED_*`/`ABLATION_*` run directories directly (`edge_ids`, `probabilities`, `targets`,
  plus `model_config.json`'s fitted `theta_star` for `extract_ablation_f1.py`'s weighted
  aggregation).
- **Reproduction script(s)**: same story as the Figure 3 delta heatmap — since EID's raw
  prediction pickles share the same schema/path convention, these need only a glob-pattern
  change (`EID_MULTISEED_s<seed>_<cond_tag>_*` in place of `MULTISEED_s<seed>_local_*` /
  `ABLATION_<TAG>_s<seed>_*`), plus (for `extract_ablation_f1.py`) EID's `func_logit_power`
  `model_config.json`, which Phase 1 already produces.
- **Blocking gaps**: none beyond Phase 1 completing, for the `noablation`/`mask_context_edges`/
  `mask_context_sign_only` conditions Phase 1 actually runs. (Production's `DIRFLIP`/
  `SIGNSCRAMBLE` rows obviously have no EID analogue until Phase 3b's ports are run.)

---

## Summary

**Ready to reproduce once Phase 1 finishes (no blockers beyond having the `noablation`
10-seed multiseed data on disk — glob-pattern/path changes only, using data/schema Phase 1
already produces):**
- Table 1 headline numbers (needs a small new extract variant + a paired-significance script
  modeled on `table1_paired_significance.py`, reusing `eid_significance.py`'s machinery).
- Figure 3 delta heatmap + its entropy regression + its paired-significance companion (glob
  swap in `walk_raw_seed()`/`pewter_seed_auc()`-equivalent functions).
- K-ablation and F1-score ablation extras (glob swap only).
- Every Empirical Confirmation panel (A, B, C, D, E) — **not EID-dependent at all**, these are
  either pure graph-structural computations or SiGAT/GINEConv-only comparisons that never touch
  Pewter/EID predictions; reuse unchanged, zero thesis-side work.
- Dataset stats / test-set counts — likewise architecture-independent, reuse unchanged.

**Needs new EID-side work beyond Phase 1's campaign as currently scoped:**
1. **Attention Directionality Panels A/B/C + Panel D + SHAP (Figure 4)** — the single biggest
   gap. Needs a ported attention-weight-recording loader for `EIDLitEdgeClassifier`
   (mechanically feasible, since EID's transformer stack is inherited unchanged from
   production) and, more importantly, a real check of whether the existing node/edge
   attention-bucketing logic correctly handles EID's third token channel (edge-identity
   tokens) — not yet attempted at all.
2. **Ablation B (all-11-aggregator table)** — needs a cheap but real posthoc rerun on EID's
   `noablation` checkpoints with `--agg-models` set to all 11 functions; Phase 1 only computed
   `func_logit_power`.
3. **Ablation A's vertex/edge/direction ports** (`randomize_walk_direction`, `mask_node_tokens`)
   — confirmed to need no new code, but the actual 10-seed training runs are Phase 3b work, not
   yet launched; EID's own `mask_context_edges`/`mask_context_sign_only` ablations are a
   different (not directly equivalent) design and need their own new significance-table script,
   even though the underlying data (Phase 1) already exists.
4. **Pipeline schematic** — a manual diagram redraw, not a script/data problem, whenever the
   thesis write-up reaches that figure.
