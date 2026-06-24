# Walk-to-Paint — CLAUDE.md

## What this project is

Edge sign prediction in directed signed graphs via random-walk Transformer.
Converts graphs into token sequences (alternating node/edge tokens), trains a
Transformer encoder with masked-edge-sign prediction (MLM-style).

6 datasets: bitcoin-alpha, bitcoin-otc, epinions, wiki-elec, wiki-rfa, slashdot090221.
Beats every GNN/SGNN baseline on all 6 datasets on the same canonical splits.

## Hardware

4× NVIDIA L40S (44 GB each), GPUs 0–3.

## Key commands

### Training

```
CUDA_VISIBLE_DEVICES=<N> python run.py dataset.name=<ds> \
    model.hardness_map_path=<path> model.hardness_lambda=1.0 \
    exp_name=<tag>
```

### Posthoc aggregation — MUST pass dataset.name= override

```
python run_posthoc.py \
    --exp-dir outputs/<ds>/<run>/ \
    --artifacts predictions,aggregator \
    --agg-models func_logit_power \
    --device <N> --run-id <tag> \
    dataset.name=<ds>        # critical, else defaults to bitcoin-alpha
```

### Hardness map (miner)

```
python scripts/compute_hardness_map.py --dataset <ds> --device <N>
```

### Optuna search

```
python optuna_run.py --dataset <ds> --n-trials 100 --device <N>
```

(Note: optuna_run.py is suspected stale — see plan-stats-rigor.md.)

## Canonical splits

train:val:test = 0.8:0.1:0.1, seed=42.
baselines/ uses identical splits (verified) — comparison is apples-to-apples.

## Walk encoding

max_walk_length=80 HOPS (not tokens). Full walk = up to 161 tokens.
Token layout: N_u0, E_s1, N_u1, E_s2, ... (alternating node/edge).
1 graph-hop = 2 token positions. window=4 tokens = ±2 graph-hops.

## Current SOTA (func_logit_power, test AUC)

| Dataset         | Baseline (best GNN) | Ours  | LocalAttn4 |
|-----------------|---------------------|-------|------------|
| bitcoin-alpha   | 0.8804 (GSGNN+SGA)  | 0.9131| 0.9370     |
| bitcoin-otc     | 0.9086 (GSGNN+SGA)  | 0.9431| 0.9337     |
| epinions        | 0.9113 (CSG-GSGNN)  | 0.9311| 0.9445     |
| wiki-elec       | 0.8840 (CSG-GSGNN)  | 0.8928| 0.8917     |
| wiki-rfa        | 0.8673 (CSG-GSGNN)  | 0.8810| 0.8882     |
| slashdot090221  | 0.8845 (SNEA)       | 0.8952| 0.8958     |

Experiment tag for current SOTA: E14_HARDNODE_L10
LocalAttn4 tag: E14_HARDNODE_L10_LOCALATTN4_20260615-150214

## Training feature flags (confirmed beneficial — already config.yaml defaults)

- **D — Dynamic resplit** (`dynamic_train_masking: true`): re-assigns supervised targets each
  epoch within TRAIN+MASK. Prevents memorizing which edges get predicted.
- **R — Node token replacement** (`node_context_mode: replace`, `node_replace_prob: 0.2`):
  randomly replaces node tokens with [UNK] or another node. Reduces reliance on node identity.
- **H — Hard-node reweighting** (`hardness_lambda: 1.0`, needs `hardness_map_path`): upweights
  loss near nodes the miner model finds hard. Currently used in E14_HARDNODE_L10 SOTA.
- **L — Local attention window** (`local_attention_window`): null=full attention (default);
  4=±2-hop banded mask (LocalAttn4 experiment, competitive/better on 4/6 datasets).

D and R are load-bearing defaults, not ablation toggles — don't disable them without reason.

## Performance philosophy — READ BEFORE ADDING ANY NEW FEATURE

Significant engineering time went into making data handling/training fast: bucket batching
and CSR ragged-tensor caching (see git history of src/data/, "Implement ragged (CSR) format
for dataset caching and loading"). **Any new feature, experiment, or training mechanic must
consider hardware/time efficiency from the start** — don't bolt something on that silently
reintroduces O(n²) padding or defeats the CSR caching. When in doubt, benchmark before and
after. See plan-performance.md for an open investigation into whether local-attention's
masked-SDPA path is actually fast.

## Key file locations

- Config: config.yaml, configs/<dataset>.yaml (partial overlays merged via dataset.name=<ds>)
- Model: src/model/model.py, src/model/lit_model.py
- Data: src/data/datasets.py, src/data/walk_sampler.py, src/data/tokenizer.py
- Shared edge loader: scripts/balance_theory_paths.py → load_edges_canonical()
- Hardness miner: scripts/compute_hardness_map.py
- Analysis scripts: scripts/edge_sign_mi_vs_distance_v3.py, scripts/node_mi_structural_embedding.py,
                    scripts/attention_analysis.py, scripts/lead4_entropy_heterogeneity.py
- MI reports: outputs/mi_analysis_package.zip (full), outputs/mi_vs_dist/, outputs/node_mi_structural/,
              outputs/attention_analysis/
- Lead 4 report/data: outputs/lead4_entropy_heterogeneity/report.md (+ raw_data.txt, computed_data.pkl,
              predictions_raw.pkl — the last holds per-edge (u,v,y,p) for all 4 models × 6 datasets,
              reusable for any future per-edge investigation without rerunning models)
- Baselines: baselines/all_results.csv, baselines/<model>/results_our_splits/

## Hardness map paths (E14 production)

bitcoin-alpha: outputs/transformer_incremental/bitcoin-alpha_seed42_nw5000000_mw80_bs1024_ep75_20260423-111407/artifacts/E14_HARDNODE_L10/hardness_map.pt
(others: same structure, check outputs/<dataset>/<run>/artifacts/E14_HARDNODE_L10/hardness_map.pt)

## Research status (as of 2026-06-23)

COMPLETED:

- MI analysis: edge-sign MI collapses 10–1000× at d=1→2 on all 6 datasets
- Node-feature MI: embeddings are 1-hop local; HITS shows weak structural MI at d>1
- Attention analysis: full-attention model attends far (mean 9–16 tokens) despite empty signal
- LocalAttn4 experiment: restricting to ±2 hops is better on 4/6 datasets (+0.004 mean AUC)
- **Lead 1 (GNN over-averaging):** Comprehensive cross-dataset analysis (6 datasets, 4 measurement steps) shows sign-heterogeneity-driven cancellation is universal (r≈−0.96) but modest (~5–9% norm loss). Degree-stratified gap shows no consistent pattern. Aggregator choice is dataset-dependent (sum on small/sparse, mean on large/dense), not primarily cancellation-driven. Depth beyond 2 is universally harmful. See LEAD1_GNN_OVER_AVERAGING_REPORT.md for full analysis. **Conclusion: Walk model's advantage is not primarily explained by over-averaging; Leads 2 & 3 remain stronger candidates.**
- **Lead 2 (GNN bottleneck):** Comprehensive cross-dataset analysis (6 datasets, both GINEConv/CSG) shows the 1-hop bottleneck is real and measurable (`h_v^(1)` retains genuine but partial, NMI 0.008–0.27, neighbor-sign info; per-edge dilution confirmed monotonic with degree on all 6). But bypassing it (oracle ablation) only helps AUC on GINEConv (6/6, substantially on epinions/slashdot090221) — CSG gains on just 1/6. Walk-transformer's own relay-token MI doesn't track GINEConv's oracle-gain pattern either (Step 2's relay-MI numbers were corrected post-hoc for a walk-distance-vs-true-BFS-distance bug — slashdot090221 0.28→0.1717, epinions 0.10→0.0505; `attention_analysis.py` flagged as likely sharing the same bug, not yet fixed). Synthetic "inverted fog" stress-test graph (Step 4) was calibrated successfully but abandoned at Step 5: GNN baselines use fixed non-trainable random node features, so they can't learn the synthetic graph's signal regardless of bottleneck, while the walk-transformer's trainable per-node embeddings would confound any comparison. **Follow-up (attention-weight diagnostic):** integrated SiGAT (GAT-style, 38-channel) as a new canonical-splits baseline and as a literal-attention-weight probe. Bucketed MI shows attention weight tracks true downstream importance cleanly on the **positive**-edge channel (monotonic, all 6 datasets) but breaks down on the **negative**-edge channel for bitcoin-otc and epinions specifically — a low-mid-attention bucket carries MORE real neighbor-sign information than the highest-attention bucket (bitcoin-otc 0.137 vs 0.052 NMI, epinions 0.127 vs 0.101 NMI) — a genuine "good information through an under-weighted pipe" case, sign-asymmetric and dataset-specific, not universal. See LEAD2_GNN_BOTTLENECK_STATUS.md and outputs/lead2_gnn_bottleneck/SUMMARY.txt for full analysis. **Conclusion: the bottleneck is a real information-loss mechanism but not sufficient alone to explain a uniform walk-model AUC advantage across datasets/architectures — contributes specifically to GINEConv's gap on a subset of datasets. The attention-weight follow-up adds a second, narrower finding: even where the model uses literal attention, negative-edge importance can be systematically under-weighted on specific datasets.**
- **Lead 3 (first-hop signal swamping / "fog of war"):** Synthetic SNR test confirms mean aggregation severely degrades a weak target signal as diluter count/magnitude grow (down to chance AUC), while concatenation stays flat — swamping is real in the idealized case. Plugging Lead 2's measured real-GNN dilution levels into that synthetic curve shows all 6 datasets/both architectures land in "destroyed" recoverability outside the lowest-degree bucket. But the walk-model's attention shows NO adaptive compensation for it: attention-mass-beyond-1-hop vs. 1-hop sign-agreement ambiguity is weakly *positively* correlated on all 6 datasets (Pearson 0.02–0.23) — opposite of the predicted sign, meaning attention reaches further when 1-hop evidence is already unambiguous, not when it's weak. See LEAD3_FOG_OF_WAR_REPORT.md for full analysis (includes a caught/fixed id-space bug: tokenizer node ids are raw dataset ids, not `baselines/splits/*.pt`'s remapped contiguous ids — read training out-edges from `dataset_cache.pt["splits"]["train"]` instead). **Conclusion: swamping is a real GNN failure mode the walk-model structurally avoids (independent attention slots vs. forced shared sum), but the avoidance is not an adaptive/learned compensation mechanism — it's a byproduct of the architecture, not evidence of selective far-attention.**
- **Lead 4 (node sign-entropy heterogeneity vs. AUC):** Bins test edges by source/target node sign-entropy (4 directional variants: out_out, in_in, out_in, inout_inout; binary Shannon entropy over ALL edges train+val+test, not train-only) and compares per-cell AUC across `walk_full`, `walk_localattn4`, GINEConv, SiGAT on all 6 datasets. Walk-model predictions use the same `func_logit_power` posthoc aggregator as the project's reported SOTA (theta read from existing `run_posthoc.py` artifacts — `*_posthoc_enc_fixed` for full attention, `localattn4_posthoc` for LocalAttn4 — never refit; verified to exactly reproduce the SOTA AUC table above). Preliminary pattern (bitcoin-alpha checked in detail, full sweep done for all 6): GNN baselines (GINEConv, SiGAT) degrade sharply on high-sign-entropy (heterogeneous) nodes — e.g. bitcoin-alpha GINEConv 0.93→0.69 AUC, SiGAT 0.76→0.16 AUC low- vs. high-entropy — while walk models stay near-ceiling in the cells where AUC is even defined (many high-entropy cells are single-class or below the n=20 sample floor, so "n/a" there isn't yet a clean comparison). Not yet given a final causal conclusion — see outputs/lead4_entropy_heterogeneity/report.md for full per-dataset tables/heatmaps/raw_data.txt before drawing one.

OPEN WORKSTREAMS (see plan files in ~/.claude/plans/):

- plan-research-leads.md       ← active, top priority
- plan-stats-rigor.md          ← optuna rewrite + multiple splits + cross-validation
- plan-hardness-miner.md       ← improve miner model, calibrate hardness_lambda
- plan-performance.md          ← efficiency of local attention + model implementation
- plan-side-quests-misc.md     ← config cleanup, repo hygiene, walk-length sweep, OWL removal

## Session management tips

- Start a new chat for each independent workstream.
- In new chat: CLAUDE.md loads automatically. Say "read ~/.claude/plans/plan-<X>.md and do step N".
- Use /compact when finishing a sub-task within a session.
- Update this CLAUDE.md after completing a workstream (update status table above).
- Never paste long files — reference by path.
