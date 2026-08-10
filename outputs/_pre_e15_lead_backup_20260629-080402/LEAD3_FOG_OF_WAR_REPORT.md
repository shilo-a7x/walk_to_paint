# Lead 3 — First-hop signal swamping ("Fog of War")

Plan: `plan-research-leads.md` Lead 3. Hypothesis: GNNs' mean/sum aggregation
forces a strong 1-hop sign signal to share a slot with weak 2-hop+ signal,
numerically suppressing ("swamping") the latter — distinct from dilution-by-count
(Lead 1) and from the compression bottleneck (Lead 2). The walk-model's attention
gives every token its own weighted slot, so it shouldn't suffer this.

## Step 1 — Synthetic SNR test (`scripts/lead3_synthetic_swamping.py`)

Model-free: k "1-hop-like" messages at magnitude s1 + noise, one "2-hop-like"
target message at fixed magnitude s2=1 + noise. Compare mean aggregation
(GNN-like) vs. concatenation (attention-like) recoverability of the target via
5-fold CV logistic-regression probe AUC.

**Result:** concat_AUC stays flat (~0.91–0.93) across all (k, s1). mean_AUC
degrades monotonically with both k (dilution-by-count) and s1 (swamping-by-
magnitude), down to ≈0.48 (chance) at k=16, s1=16. Sanity checks confirmed the
two effects are separable (s1=0,k=0 → mean==concat; s1=0,k=4 → mean<concat from
count alone). **Swamping is real and severe in the idealized case.**
Full table: `outputs/lead3_swamping/swamping_report.txt`.

## Step 2 — Real-GNN magnitude-ratio check (`scripts/lead3_real_gnn_swamping_check.py`)

No retraining: takes Lead 2's already-measured per-edge `contribution_share`
(degree-bucketed, 6 datasets × {CSG, GINEConv}), inverts it to
`k_eff = 1/mean_share` (effective # of equal-weight diluters), and looks up the
predicted mean_AUC on Step 1's synthetic curve at two assumed s1/s2 ratios
(1× and 4×, the latter motivated by Lead 1/2's 10–1000× d=1-vs-d=2 MI gap).

**Result:** consistent across all 6 datasets and both architectures — only the
lowest-degree bucket lands in "partial" recoverability; every other
degree bucket is "destroyed" (predicted AUC ≈ chance) under both ratio
assumptions. Real GNN dilution levels are high enough that, if the swamping
mechanism from Step 1 applies, signal loss should be severe outside the
lowest-degree regime. Full table: `outputs/lead3_swamping/real_gnn_swamping_check.txt`.

## Step 3 — Walk-model attention vs. 1-hop ambiguity (`scripts/lead3_attention_ambiguity.py`)

Tests the specific *adaptive-compensation* claim: does the walk-model's
attention shift mass beyond 1-hop more when the 1-hop evidence at the target
node v is locally ambiguous (its training out-edge signs disagree), vs.
generic/unconditional far-reach? `ambiguity_score(v)` = agreement rate among
v's training-split out-edge signs (1.0 = unanimous, 0.5 = maximally mixed);
nodes with <2 training out-edges are dropped (degenerate score). Probes the
existing E14_HARDNODE_L10 checkpoint, no retraining.

**Bug caught and fixed during this run:** the first implementation sourced
training out-edges from `baselines/splits/<ds>.pt`, which remaps node ids to a
contiguous `0..num_nodes-1` range (`baselines/CopulaLSP/loader.py:remap_node_id`,
`sorted(unique_nodes)`-based). The tokenizer's `N_<id>` tokens use each
dataset's **raw original** node ids instead — a different id space. For
bitcoin-alpha/otc/wiki-elec/epinions the two ranges happened to overlap enough
that lookups silently "succeeded" against the wrong nodes; for wiki-rfa (whose
raw ids are large account numbers) the spaces don't overlap at all, so every
target was dropped and the script crashed — that crash is what surfaced the
bug. Fixed by reading `dataset_cache.pt["splits"]["train"]` directly, which is
already `(src, dst, sign)` in the same raw-id space the tokenizer uses — no
remapping needed. All 6 datasets re-run after the fix; numbers below are
post-fix and verified (`re-aggregated mean eff_dist` exactly matches the
dataset-level value reported by `attention_analysis.py` for every dataset).

| Dataset | n retained | Pearson(ambiguity_score, frac_beyond_1hop) | Spearman |
|---|---|---|---|
| bitcoin-alpha | 56,097 | +0.1932 | +0.1534 |
| bitcoin-otc | 42,910 | +0.2311 | +0.2086 |
| epinions | 18,940 | +0.1132 | +0.1017 |
| wiki-elec | 3,982 | +0.0156 | +0.0239 |
| wiki-rfa | 13,241 | +0.0320 | +0.0156 |
| slashdot090221 | 4,492 | +0.1111 | +0.0826 |

Sign convention: `ambiguity_score` is **higher when 1-hop evidence is LESS
ambiguous** (1.0 = unanimous). The adaptive-compensation hypothesis predicts a
**negative** correlation (more attention beyond 1-hop precisely when 1-hop
evidence is weak/ambiguous). All 6 datasets instead show a small-to-moderate
**positive** correlation: attention reaches further beyond the 1-hop ring
*more*, not less, when the 1-hop evidence is already clear/unambiguous.

**Conclusion: no evidence of adaptive/selective compensation for 1-hop
ambiguity** — consistent direction (positive, opposite of the hypothesized
sign) across all 6 datasets, magnitude modest (0.01–0.23). This doesn't
contradict Steps 1–2's broader swamping-avoidance evidence (concat vs. mean is
still structurally different regardless of whether attention is locally
adaptive), but it does rule out the *specific* mechanism Step 3 set out to
test: the walk-model isn't dynamically reallocating attention mass in response
to per-target 1-hop sign ambiguity. Full per-dataset tables:
`outputs/lead3_swamping/attention_ambiguity_<dataset>.txt`.

## Overall Lead 3 conclusion

- Swamping-by-magnitude is a real, severe effect in the idealized synthetic
  model (Step 1), and real-GNN dilution levels (Step 2) are high enough that
  it should bite outside the lowest-degree bucket on all 6 datasets/both
  architectures if the mean-aggregation mechanism applies as modeled.
- The walk-model's attention does NOT compensate for this adaptively at the
  per-target level (Step 3) — its far-reach is generic, not ambiguity-
  conditioned. The walk-model's robustness to swamping (if any) comes from the
  structural fact that concatenation/independent-attention-slots never force a
  shared sum in the first place, not from a learned adaptive mechanism.
- Step 4 (eval-time attention-window sweep) remains optional/secondary per the
  plan and was not run — Steps 1–3 already give a complete, verified answer to
  the primary question.
