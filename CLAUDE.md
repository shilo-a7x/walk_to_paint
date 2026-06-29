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

train:val:test = 0.8:0.1:0.1, seed=42 (walk model: nested 4-way train0.48/mask0.32/val0.1/test0.1).

**CORRECTION (was wrong):** the old claim "baselines/ uses identical splits — apples-to-apples"
does NOT hold. The legacy `baselines/splits/*.pt` and SGA CSVs were generated *independently* of
the walk split (different RNG + fabricated reverse edges), so walk-test and GNN-test overlapped
only ~10%. Fixed by `baselines/prepare_splits.py::build_canonical_split`, which re-derives the
baseline artifacts from the frozen walk split (`data/<ds>/dataset_cache.pt["splits"]`) into
isolated `baselines/splits_canonical/`. Full story: `SPLIT_PROVENANCE.md` (+ `FABRICATED_REVERSE_EDGES.md`).

**Walk coverage caveat — RESOLVED (2026-06-29) by the E15 k_cover k=5 sampler.** The old
uniform sampler only predicted a test edge if it appeared in a sampled walk, so on sparse
graphs it evaluated a *subset* of the nominal test split (bitcoin-alpha/otc 100%, slashdot 98%,
epinions 88%, wiki-rfa 86%, wiki-elec 85%) while GNNs saw all of it. The new edge-anchored
`k_cover` sampler (`walk_strategy=k_cover`, `walk_k_min=5`) drives **node AND edge coverage to
~100% on all 6** while matching/beating uniform AUC (and improving it on the same edges where
coverage was low). The SOTA table below is now the E15 full-coverage model — no coverage
footnote. Old uniform-E14 numbers are in parentheses. Details: `WALK_COVERAGE.md`,
`outputs/walk_coverage_analysis/E15_SWEEP_RESULTS.md`,
`~/.claude/plans/plan-a-fix-for-glimmering-panda.md`.

## Walk encoding

max_walk_length=80 HOPS (not tokens). Full walk = up to 161 tokens.
Token layout: N_u0, E_s1, N_u1, E_s2, ... (alternating node/edge).
1 graph-hop = 2 token positions. window=4 tokens = ±2 graph-hops.

## Current SOTA (func_logit_power, test AUC)

**Walk numbers are now the E15 k_cover k=5 full-coverage model** (~100% node+edge coverage
on all 6). Old uniform-E14 numbers in parentheses (had the ~85–88% coverage caveat on the
sparse graphs). The baseline column shows the **canonical-split** best GNN (re-run on the
unified walk-derived split, `baselines/all_results_canonical.csv`); pre-canonical best-GNN in
its own parentheses. **On the identical shared test edges the walk model beats EVERY GNN on all
6 datasets, both attention variants** (apples-to-apples; full matched table in
`outputs/walk_coverage_analysis/E15_SWEEP_RESULTS.md`, older canonical detail in
`CANONICAL_RERUN_FINDINGS.md` §1a). With full coverage the walk now also wins on each-model's-
own-full-test on all 6 — the prior wiki-elec/wiki-rfa "SiGAT marginally higher" exception was
purely a coverage artifact and is gone.

| Dataset         | Canon best-GNN (old)              | Ours (E14)      | LocalAttn4 (E14) |
|-----------------|-----------------------------------|-----------------|------------------|
| bitcoin-alpha   | 0.9051 SGA-GSGNN (0.8804)         | 0.9251 (0.9131) | 0.9362 (0.9370)  |
| bitcoin-otc     | 0.8972 SNEA (0.9086)              | 0.9427 (0.9431) | 0.9410 (0.9337)  |
| epinions        | 0.9146 SiGAT (0.9113)             | 0.9562 (0.9311) | 0.9568 (0.9445)  |
| wiki-elec       | 0.8930 SiGAT (0.8840)             | 0.9016 (0.8928) | 0.9038 (0.8917)  |
| wiki-rfa        | 0.8831 SiGAT (0.8673)             | 0.8932 (0.8810) | 0.8916 (0.8882)  |
| slashdot090221  | 0.8587 SiGAT (0.8845)             | 0.9012 (0.8952) | 0.8984 (0.8958)  |

E15 per-dataset best budgets (k_cover k=5, mw80): alpha 5M/5M, otc 2M/1M, epinions 3M/2M,
wiki-elec 0.5M/0.5M, wiki-rfa 1M/1M, slashdot 5M/3M (full/local). wiki-elec & wiki-rfa AUC
*drops* past the minimum covering budget (over-saturation on small dense graphs) — ship the
smallest covering budget there.

Apples-to-apples (shared edges) best GNN is always lower still — e.g. epinions GINE
0.8642 / SiGAT 0.9109, slashdot GINE 0.7869 / SiGAT 0.8571. SE-SGformer excluded from
"best GNN": its KNN discriminator emits hard labels, so its AUC is really balanced
accuracy (~0.57–0.73, same as pre-canonical — not a regression; see findings doc §3).

Experiment tag for current SOTA: E15_SWEEP_k5 / E15_COVERAGE_KCOVER_K5 (k_cover k=5; isolated
keyed caches `data/<ds>/dataset_cache__k_cover_k5_nw<nw>_mw80_seed42.pt`; winner run dirs listed
in `E15_SWEEP_RESULTS.md`). Prior uniform SOTA was E14_HARDNODE_L10 /
E14_HARDNODE_L10_LOCALATTN4_20260615-150214 (artifacts untouched).

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
after. **2026-06-28: local attention's masked-SDPA path WAS slow (37% wall-clock / 4x memory
vs. full attention) — fixed in `src/model/model.py` (`LocalAttentionEncoderLayer`), see
plan-performance.md for the diagnosis.**

## Key file locations

- Config: config.yaml, configs/<dataset>.yaml (partial overlays merged via dataset.name=<ds>)
- Model: src/model/model.py, src/model/lit_model.py
- Data: src/data/datasets.py, src/data/walk_sampler.py, src/data/tokenizer.py
- Shared edge loader: scripts/balance_theory_paths.py → load_edges_canonical()
- Hardness miner: scripts/compute_hardness_map.py
- Analysis scripts: scripts/edge_sign_mi_vs_distance_v3.py, scripts/node_mi_structural_embedding.py,
                    scripts/attention_analysis.py, scripts/lead4_entropy_heterogeneity.py,
                    scripts/lead4_twohop_path_consistency.py
- MI reports: outputs/mi_analysis_package.zip (full), outputs/mi_vs_dist/, outputs/node_mi_structural/,
              outputs/attention_analysis/
- Lead 4 report/data: outputs/lead4_entropy_heterogeneity/report.md (+ raw_data.txt, computed_data.pkl).
              Now CANONICAL-NATIVE (shared-edge): reads predictions_raw_canonical.pkl (built by
              baselines/postprocess_canonical.py), restricts every model to the shared (walk-covered)
              edge set so all 4 models are bucketed over IDENTICAL edges with identical per-cell n.
              predictions_raw_canonical.pkl holds per-edge (u,v,y,p) for all 4 models × 6 datasets in
              one raw (u,v) id space — reusable for any future per-edge investigation without rerunning
              models. Bucket sweep is powers of 2 (--n-buckets 2 4 8 16 32).
- Lead 4b report/data: outputs/lead4_twohop_path_consistency/report.md (+ raw_data.txt, computed_data.pkl) —
              reuses the same predictions_raw_canonical.pkl via lead4_entropy_heterogeneity.load_shared_predictions.
              Now has THREE path-direction variants for edge (u,v): `out` = forward 2-hop consistency from
              target v (v->m->k), `in` = backward 2-hop consistency into source u (s->t->u), `inout` = pool
              the out (from v) and in (into u) (total,consistent) counts into ONE tally for that edge before
              taking entropy (not a separate undirected traversal). Bars grow from a 0.5 baseline, walk=blue
              vs GNN=orange, shared n shown under each x-tick, bucket sweep powers of 2 (2 4 8 16 32).
- Baselines: baselines/all_results.csv, baselines/<model>/results_our_splits/
- **FABRICATED_REVERSE_EDGES.md** — read before using `baselines/splits/<ds>.pt`'s `edge_index` as
              "all edges of the graph" for any per-edge/per-node diagnostic: 14–48% of its edges
              (worse on epinions/wiki-elec/wiki-rfa/slashdot090221) are fabricated reverse mirrors with
              no real counterpart. Use `build_real_dense_edge_set()` (in lead4_entropy_heterogeneity.py)
              to filter them out first. Does NOT affect the SOTA table below. **Also documents a
              SECOND, UNRESOLVED issue**: even after that fix, the walk model's test split and the GNN
              baselines' test split are ~independent random samples of the same edge pool (≈10%
              overlap on all 6 datasets, not a subset) — any edge-level walk-vs-GNN bucket comparison
              (Lead 4, Lead 4b, likely Lead 1) has no shared ground truth underneath it. Needs a
              deliberate decision (intersect vs. re-evaluate vs. accept+caveat), not a quick filter.

## Hardness map paths (E14 production)

bitcoin-alpha: outputs/transformer_incremental/bitcoin-alpha_seed42_nw5000000_mw80_bs1024_ep75_20260423-111407/artifacts/E14_HARDNODE_L10/hardness_map.pt
(others: same structure, check outputs/<dataset>/<run>/artifacts/E14_HARDNODE_L10/hardness_map.pt)

## Research status (as of 2026-06-26)

**Central question:** the walk-Transformer beats every GNN/SGNN baseline by
3–5pp AUC on all 6 datasets despite edge-sign MI collapsing 10–1000× beyond
1 hop. Leads 1–4b below investigate why. **Full synthesis, always read this
first:** `RESEARCH_LEADS_SUMMARY.md` (supervisor-facing, kept in sync with
the per-lead files below). Per-lead detail lives in its own file — don't
duplicate findings here, just point to them:

**2026-06-26 canonical-split rerun — `CANONICAL_RERUN_FINDINGS.md`.** All Leads
were re-derived on the unified canonical split (shared ground truth, not the old
~10%-overlap edge samples). Headline strengthens (walk beats every GNN on identical
edges, all 6); Leads 1/2/3 hold; Lead 4/4b is **entropy-variant-dependent** — weak for
`out_in`, strong/robust for `in_in` (NOT the uniform retraction first reported).

**2026-06-29 E15 full-coverage rerun — `outputs/lead4_entropy_heterogeneity/E15_FULLCOVERAGE_RERUN_NOTE.md`.**
`predictions_raw_canonical.pkl` rebuilt from the E15 k_cover full-coverage walk runs (walk now
100% covers every test set; GNNs unchanged) and Leads 1/4/4b rerun on identical full-coverage
edges. **Lead 4/4b variant-dependent picture HOLDS and slightly STRENGTHENS** — the `in_in`
differential degradation survives on the now-complete edge set (resolving the "different-edges
artifact" worry); `out_in` stays weak. Walk advantage is NOT a uniform offset — it concentrates
in heterogeneous in-anchored neighborhoods. Pre-rerun reports backed up under
`outputs/_pre_e15_lead_backup_<ts>/`.

| Lead | One-line verdict | Detail |
|---|---|---|
| MI / attention baseline | Edge-sign MI collapses 10–1000× at d=1→2; full-attention model still attends far (mean 9–16 tokens, **provisional**, see Lead 2's bug note below); LocalAttn4 (±2 hop) competitive/better on 4/6 datasets | `outputs/mi_analysis_package.zip`, `outputs/attention_analysis/` |
| Lead 1 — GNN over-averaging | Cancellation is real, universal (r≈−0.96), but modest (~5–9%) — not the primary driver | `LEAD1_GNN_OVER_AVERAGING_REPORT.md` |
| Lead 2 — GNN bottleneck (+ SiGAT attention-weight follow-up) | Bottleneck real (NMI 0.008–0.27) but oracle-bypass gain is architecture/dataset-dependent (GINEConv 6/6, CSG 1/6); SiGAT attention-weight diagnostic found negative-edge "good info through bad pipe" on bitcoin-otc/epinions | `LEAD2_GNN_BOTTLENECK_STATUS.md` |
| Lead 3 — fog of war / swamping | Swamping is real and severe in theory, but walk-model attention shows no adaptive compensation for it — robustness is structural, not learned | `LEAD3_FOG_OF_WAR_REPORT.md` |
| Lead 4 / 4b — entropy heterogeneity | **Variant-dependent, CONFIRMED on E15 full coverage (2026-06-29).** GNN baselines degrade more than the walk as sign-heterogeneity rises for the `in_in` variant — strong & robust on all 6 (best-GNN drop − walk drop = +0.015..+0.34, ≥ the prior ~88%-cov values); `out_out`/`inout_inout` positive 5/6 (wiki-elec ~null); `out_in` weak/mixed. Survives on identical full-coverage edges ⇒ the walk advantage concentrates in heterogeneous in-anchored neighborhoods, NOT a uniform offset. (Earlier "RETRACTED / uniform offset" reading was the `out_in`-only view.) | `outputs/lead4_entropy_heterogeneity/E15_FULLCOVERAGE_RERUN_NOTE.md`, `outputs/lead4_entropy_heterogeneity/`, `outputs/lead4_twohop_path_consistency/` |

**Known data-quality caveats (read before any new edge-level diagnostic):**
- `FABRICATED_REVERSE_EDGES.md` — `baselines/splits/<ds>.pt`'s `edge_index`
  has 14–48% fabricated reverse-mirror edges; use `build_real_dense_edge_set()`.
  Also documents an unresolved second issue: walk-model vs. GNN test splits
  are ~independent samples of the same edge pool (~10% overlap), affecting
  any edge-level walk-vs-GNN bucket comparison (Leads 1, 4, 4b).
- Lead 2's Step 2 found a walk-token-position-vs-true-BFS-distance bug
  (fixed there); `scripts/attention_analysis.py` likely shares the same bug
  and has **not** been fixed — the "attends far despite empty signal"
  number above is provisional pending that fix.

OPEN WORKSTREAMS (see plan files in ~/.claude/plans/):

- plan-research-leads.md       ← active, top priority (Leads 1–4b above)
- Lead 5/6 subplans            ← ensemble effect + trainable-features/capacity/training-regime parity (post Lead 4b follow-ups, see plan index in ~/.claude/plans/)
- plan-stats-rigor.md          ← optuna rewrite + multiple splits + cross-validation
- plan-hardness-miner.md       ← improve miner model, calibrate hardness_lambda
- plan-performance.md          ← Issue 1 (local attn slower than full attn) RESOLVED 2026-06-28,
  fixed in src/model/model.py; Issues 2/3 deprioritized, see plan file
- plan-side-quests-misc.md     ← config cleanup, repo hygiene, walk-length sweep, OWL removal

## Session management tips

- Start a new chat for each independent workstream.
- In new chat: CLAUDE.md loads automatically. Say "read ~/.claude/plans/plan-<X>.md and do step N".
- Use /compact when finishing a sub-task within a session.
- Update this CLAUDE.md after completing a workstream (update status table above).
- Never paste long files — reference by path.
