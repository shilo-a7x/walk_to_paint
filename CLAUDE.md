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

All commands below assume the project venv (`.venv/bin/python`, or
`source .venv/bin/activate` first) — the bare system `python`/`python3` has no
`torch` installed and fails with `ModuleNotFoundError` (confirmed 2026-07-06).

**Environment migrated 2026-08-02:** `.venv` is now Python 3.14.6 + torch
2.13.0+cu126 + numpy 2.5.1 + scipy 1.18.0 + pandas 3.0.5 + scikit-learn 1.9.0 +
lightning/pytorch-lightning 2.6.5 (was Python 3.9.25 + torch 2.7.1 + numpy
2.0.2). Full 6-dataset retrain + `func_logit_power` posthoc validation passed
(deltas −0.34pp to +0.89pp vs. the prior stack, within the established
same-environment noise band — see E31 in the SOTA table below). The old
environment is archived at `.venv-py39-archive/`, untouched, for reference.
Reproduction steps for this whole migration (or the next one): see
`PYTHON_MIGRATION_GUIDE.md`.

**Python 3.14 multiprocessing gotcha:** Python 3.14 changed the default
`multiprocessing` start method on Linux from `fork` to `forkserver`, which
requires anything passed to a worker process (e.g. a DataLoader's
`collate_fn`) to be picklable — a local/nested closure is not, since pickling
only works by recording an importable module-level name. This broke
`ragged_collate_fn` (`src/data/stage_dataset.py`), fixed by returning a
module-level `_RaggedCollate` class instance instead of a closure. **Any new
multiprocessing code (DataLoader workers, `Pool`/`ProcessPoolExecutor`
targets) must use a module-level function/class, not a closure** — this is
now a standing rule, not just a one-time fix. (Everywhere else in the repo
already followed this pattern, or explicitly pins `mp.get_context("fork")`,
and was unaffected.)

### Training

**GPU pinning gotcha (confirmed 2026-07-06):** `run.py` unconditionally runs
`os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)` (line 97), where
`args.device` is the `--device` CLI flag, **defaulting to 0 if omitted**. This
overwrites any shell-level `CUDA_VISIBLE_DEVICES=<N>` export before the process
actually touches the GPU — omitting `--device` on parallel runs silently piles
every process onto physical GPU 0 (verified: 4 processes launched with
`CUDA_VISIBLE_DEVICES=0/1/2/3` and no `--device` all landed on GPU 0). **Always
pass `--device <N>` explicitly**; the shell `CUDA_VISIBLE_DEVICES=<N>` prefix is
redundant/ineffective for pinning in this codebase, don't rely on it alone.

Also note: `exp_name=<tag>` (no `training.` prefix) is a **silent no-op** — the
config key is `training.exp_name`, and a bare `exp_name=` dotlist override just
creates an unused top-level key, so `run.py` falls back to auto-generating
`<dataset>-run_<timestamp>` instead. Always use `training.exp_name=<tag>`.

```
.venv/bin/python run.py --device <N> dataset.name=<ds> training.exp_name=<tag>
```

configs/<dataset>.yaml already default to the E25/E26 `edge_cover` sampler
(replaced `k_cover k=5` on 2026-07-19 — see "Walk sampler" below) and its
per-dataset `num_walks` budget (see Current SOTA below) — no override needed for
walk sampling either way. **Hardness reweighting (H) is scrapped everywhere as of
2026-07-19 — never set `hardness_lambda`/`hardness_map_path` on any run, full
attention or LocalAttn4** (see the H bullet in Training feature flags below).
**Current default is LocalAttn4** (`model.local_attention_window=4`) — **settled,
2026-07-20**, see "Attention variant: full vs. local" below. Not chosen for a clean
AUC win (the numbers are mixed, 4/6 vs 2/6) but confirmed as the mechanistically
correct way to realize the PEWTER paper's "proximal" claim: a genuine ±2-hop
attention window on a long walk preserves real local context ~85-91% of the time,
while literally truncating the walk to match that window's span (L=2) starves
context down to ~45-57% because a 2-edge walk rarely has room for both a target and
a neighbor. Full attention + short walks is no longer being pursued as the
alternative production path.

**`config.yaml`'s `model.local_attention_window` default was `null` (full attention)
from 2026-07-20 until 2026-08-04** despite the "settled default" claim above — real
production LocalAttn4 runs only got there via a manually-added
`model.local_attention_window=4` CLI override that was never actually part of the
documented training command below, which silently trained full attention instead. This
bit the `E31_PY314_MIGRATION` retrain (see "CORRECTED 2026-08-04" under "Current SOTA").
**Fixed 2026-08-04: `config.yaml`'s default is now `local_attention_window: 4`** — the
plain command below now gets real LocalAttn4 with no override needed. To get full
attention (e.g. to reproduce an E25/E26-style run), pass
`model.local_attention_window=null` explicitly.

**Training-regime defaults updated 2026-07-19** (opinion requested, decided
item-by-item, see `~/.claude/plans/plan-a-fix-for-glimmering-panda.md`
addendum): `training.early_stopping_min_delta: 0.001` is now set in
`config.yaml` (was implicit 0 — noise-level upticks no longer reset the
early-stopping patience counter); `epochs: 75` for epinions and
slashdot090221 only (both showed still-climbing val AUC at the old 50-epoch
cap in convergence curves — the other 4 datasets were unaffected and keep
their existing epoch caps). `CosineAnnealingLR`'s horizon stays coupled to
`cfg.training.epochs` (left as-is, no change).

### Posthoc aggregation — auto-recovers training-time config from the checkpoint (fixed 2026-07-19)

`run_posthoc.py` now loads the full resolved training-time cfg straight from
the checkpoint (`LitEdgeClassifier.save_hyperparameters()` already saves it)
and uses that as the base for both the data pipeline and the model — CLI
dotlist overrides are only needed for deliberate, explicit changes, not to
reproduce the training-time setup. Verified against real checkpoints
(epinions LocalAttn4 and full-attention) to reproduce their known reference
AUCs exactly when posted with nothing but `dataset.name=<ds>`. Before this
fix, any non-default `dataset.*` or `model.local_attention_window` override
had to be retyped verbatim on the posthoc call or it silently evaluated the
wrong walk cache/architecture — bit 3 times historically (see `MASKING.md`'s
re-verification list for detail). Checkpoints saved before
`save_hyperparameters` existed fall back to the old config.yaml+CLI behavior
with a loud warning — the old "repeat every `dataset.*` override" rule still
applies only to those.

```
.venv/bin/python run_posthoc.py \
    --exp-dir outputs/<ds>/<run>/ \
    --artifacts predictions,aggregator \
    --agg-models func_logit_power \
    --device <N> --run-id <tag> \
    dataset.name=<ds>   # no longer required (recovered from the checkpoint), harmless to keep
```

### Hardness map (miner)

The `--dataset`/`--device`-only form below is **stale** — the script's actual
args are `--cache` (path to a `dataset_cache*.pt`) and `--out` (output
`hardness_map.pt` path), both required, plus `--device`/`--epochs`/etc.:

```
.venv/bin/python scripts/compute_hardness_map.py \
    --cache data/<ds>/dataset_cache.pt --out <path>/hardness_map.pt --device <N>
```

**Leave `--max-walk-edges` unset/0 (default).** An earlier E14-era doc claimed
`--max-walk-edges=7` was part of the production recipe; it isn't — the actual
`E14_HARDNODE_L10`/E17 config leaves it at 0, and the one E17 mining pass that
used the filter by mistake had to be discarded and redone (corrected
2026-07-07, see `PROJECT_OVERVIEW.md`'s hardness-miner section and
`plan-hardness-miner.md`'s methodological history / `old_chats/DRH.md`).

### Optuna search

```
.venv/bin/python optuna_run.py --dataset <ds> --n-trials 100 --device <N>
```

(Note: optuna_run.py is suspected stale — never re-verified against the current
pipeline (edge_cover sampler, LocalAttn4 default, current config.yaml schema). Flagged
with an inline marker at the top of the file itself as of 2026-08-20; not a current
priority to fix, needs a real rewrite/re-audit when it is picked up.)

## Canonical splits

train:val:test = 0.8:0.1:0.1, seed=42 (walk model: nested 4-way train0.48/mask0.32/val0.1/test0.1).

**CORRECTION (was wrong):** the old claim "baselines/ uses identical splits — apples-to-apples"
does NOT hold. The legacy `baselines/splits/*.pt` and SGA CSVs were generated *independently* of
the walk split (different RNG + fabricated reverse edges), so walk-test and GNN-test overlapped
only ~10%. Fixed by `baselines/prepare_splits.py::build_canonical_split`, which re-derives the
baseline artifacts from the frozen walk split (`data/<ds>/dataset_cache.pt["splits"]`) into
isolated `baselines/splits_canonical/`. Full story: `SPLIT_PROVENANCE.md` (+ `FABRICATED_REVERSE_EDGES.md`).

**Walk coverage caveat — RESOLVED (2026-06-29), sampler since SUPERSEDED (2026-07-19)
by `edge_cover`.** Old uniform sampler evaluated only a subset of the nominal test split
on sparse graphs (85–98% coverage) while GNNs saw all of it; fixed by an edge-anchored
sampler. That sampler (`k_cover`) was itself replaced by **`walk_strategy=edge_cover`**
— one forced, provably-globally-distinct anchor walk per edge, zero duplicate walks
(measured 0.000000% at production scale), vs. `k_cover`'s 29%-on-wiki-elec duplicate
rate. **`edge_cover` is now the default in every `configs/<dataset>.yaml`**, budgets
freshly re-swept per dataset — see "Walk sampler" below. Coverage remains ~100% on all 6
(provable, since `edge_cover` guarantees coverage once `num_walks >= |E|`). Details:
`WALK_COVERAGE.md`, `outputs/walk_coverage_analysis/{E15_SWEEP_RESULTS,E24_BP_SWEEP_RESULTS,
E25_BUDGET_SWEEP_RESULTS,E26_WIKI_SWEEP_RESULTS}.md`,
`~/.claude/plans/plan-a-fix-for-glimmering-panda.md`.

## Walk sampler

**Current default: `edge_cover` (k=1, zero-duplication guarantee)** — adopted
2026-07-19, replacing `k_cover k=5`. Every edge gets exactly one forced anchor walk;
since each edge's own `(u, label, v)` triple is unique, anchor-anchor collision is
impossible *by construction*, provable not just empirical. Remaining budget beyond
`|E|` is filled via a dedup-retry phase that raises `RuntimeError` (not silent
duplicate-padding) if it can't find enough genuinely distinct walks. `k_cover`/
`k_cover_bp` (k>1-capable, backward-prefix anchors) remain available as a fallback,
just no longer the default.

Per-dataset production budgets (re-swept fresh under `edge_cover`, `{floor(=|E|),
1.5×, 3×, 5×}` grid, diminishing-returns pick, signed off 2026-07-17/19 — full grids
with every point tested, not just winners, in `E25_BUDGET_SWEEP_RESULTS.md` /
`E26_WIKI_SWEEP_RESULTS.md`):

| dataset | \|E\| | pick | num_walks | why |
|---|---|---|---|---|
| bitcoin-alpha | 24,186 | 5× | 120,930 | still climbing at 5×; old 5M-walk budget (207×) gains +1.56pp more but at 41× the walks — not yet re-tested past 5× |
| bitcoin-otc | 35,592 | 5× | 177,960 | still climbing at 5×, not yet swept past it |
| epinions | 840,799 | floor (1×) | 840,799 | floor→5× is flat/noisy (0.34pp range) — cheapest point taken |
| slashdot090221 | 549,202 | 3× | 1,647,606 | peaks at 3×, 5×/old-ref (9.1×) both flat-to-negative past this point |
| wiki-elec | 103,689 | 1.5× | 155,534 | over-saturates past 1.5× (confirmed genuine, not a dedup artifact — monotonic decline through 3×/5×/8×) |
| wiki-rfa | 177,211 | 1.5× | 265,817 | same over-saturation shape, confirmed by a full 5-point sweep incl. 3×/5×/8× |

bitcoin-alpha and bitcoin-otc's picks are **not yet a confirmed final floor** — both
were still rising at the top of their tested grid (5×); only bitcoin-alpha has a
higher reference point (5M walks, +1.56pp over 5×), and it hasn't been re-tested
between 5× and that point. Treat those two as "current best swept point," not "proven
plateau," unlike epinions/slashdot090221/wiki-elec/wiki-rfa which do show a real
peak/plateau within their tested grids.

**Retrain status: DONE (2026-07-19, re-verified 2026-07-30).** The full 6-dataset
retrain-and-report against the SOTA table is complete and re-verified by exact numeric
match against the underlying sweep docs — see `SOTA_HISTORY.md` for the verification
detail. Re-running Leads 1/4/4b on the new predictions remains a separate, not-yet-done
follow-up — see "Open threads" in Research status below.

## Walk encoding

max_walk_length=80 HOPS (not tokens). Full walk = up to 161 tokens.
Token layout: N_u0, E_s1, N_u1, E_s2, ... (alternating node/edge).
1 graph-hop = 2 token positions. window=4 tokens = ±2 graph-hops.

## Current SOTA (test AUC)

**Canonical numbers: 10-seed mean±std** (`func_logit_power`, seeds 42–51, driver
`scripts/run_multiseed_pewter.py`) — this is what `WSDM_format_revised.tex` Table 1
uses. An older single-split table (E25/E26/E31/E32) is superseded; full detail and the
old per-experiment-tag numbers are archived in `SOTA_HISTORY.md`.

| dataset | full attention | local attention | GINEConv (10-split) |
|---|---|---|---|
| bitcoin-alpha | 0.9146 ± 0.0102 | 0.9134 ± 0.0173 | 0.8497 ± 0.0189 |
| bitcoin-otc | 0.9315 ± 0.0073 | 0.9317 ± 0.0065 | 0.8877 ± 0.0089 |
| epinions | 0.9523 ± 0.0012 | 0.9536 ± 0.0015 | 0.8612 ± 0.0043 |
| wiki-elec | 0.9008 ± 0.0030 | 0.9023 ± 0.0031 | 0.8730 ± 0.0073 |
| wiki-rfa | 0.8923 ± 0.0025 | 0.8914 ± 0.0046 | 0.8606 ± 0.0055 |
| slashdot090221 | 0.8989 ± 0.0015 | 0.8968 ± 0.0016 | 0.7864 ± 0.0082 |

All 11 aggregator functions were computed per seed, so Ablation B reuses this data
(`aaai2027/figure_data/ablationB_multiseed.csv`). Real split-to-split std (0.001–0.017)
is consistently wider than the old analytic Hanley–McNeil SE (~0.001–0.007) — the old SE
understated true uncertainty. Bitcoin-alpha local attention has one unexplained low
outlier (seed 50, AUC 0.8718 across every aggregator) — check
`logs/multiseed/bitcoin-alpha_local_s50.train.log` before citing this as a stable number.

**Table 1 baselines:** GCN/GAT/SGCN/GSGNN are published-only numbers (no local repro).
SiGAT is our own 10-split canonical repro
(`baselines/SGA/results_our_splits_canonical/<ds>/SiGAT/seed{42..51}/score.csv`,
`tst_auc`): bitcoin-alpha 0.869±.016, bitcoin-otc 0.878±.008, epinions 0.909±.005,
wiki-elec 0.888±.005, wiki-rfa 0.878±.004, slashdot090221 0.859±.005. SNEA/CopulaLSP
are still single-split. **Convention: unqualified "SiGAT" always means raw SiGAT, not
SGA-augmented, everywhere in this codebase/paper.** CSG/CSG-GSGNN are dropped from
the table pending their own 10-split rerun (numbers still on disk, `baselines/CSG/
results_our_splits_canonical/`). node2vec is the one non-GNN baseline included
(weaker than every GNN everywhere, safe reference row); POLE (infeasible, O(n²)
memory), SLF (beat SiGAT on 3/6 but no real val-based model selection), and SIGNet
(reference C++ implementation has a real stopping-condition bug) were tried and
rejected — don't add their numbers without redoing that investigation, detail in
`aaai2027/PAPER_CLOSEOUT_LOG.md`.

**Paired significance:** Pewter's winning variant (per-dataset, matches Table 1's bold
marks) beats the best baseline on all 10/10 seeds, all 6 datasets (60/60 total),
one-sided paired Wilcoxon $p=0.00098$ throughout. Script:
`scripts/paper_figures/table1_paired_significance.py`, data:
`aaai2027/figure_data/table1_paired_significance.csv`.

**Standing methodology rules for any future multiseed figure:**
- Entropy-heatmap multi-split convention: average each split's own per-cell AUC
  (mean±std), not pooled predictions — matches the mean±std convention used
  everywhere else in the paper. Decided 2026-08-10, full rationale in `SOTA_HISTORY.md`.
- Figure panel-letter labels are uppercase (A)/(B)/(C)/... everywhere — captions,
  in-prose refs, and rendered PNGs.
- Dataset display names are canonicalized everywhere in rendered text
  (Bitcoin-alpha/Bitcoin-otc/Epinions/Slashdot/Wiki-elec/Wiki-RfA — never the raw
  `slashdot090221` key), including every `plot_*.py` script's `DISPLAY_LABEL` dict.

**SHAP edge directionality** (companion to the Attention Directionality figure): exact
Shapley causal contribution of each context edge to a masked target edge's predicted
P(positive), on the LocalAttn4 checkpoint, restricted to the local attention window
(±2 hops, ≤16 subsets/instance, exact not sampled). Efficiency-property check passed to
float precision (~1e-16) on all 6 datasets. **Finding:** mean |SHAP| decays from hop 1
to hop 2 on all 6/6 datasets, well outside cluster-robust SEs — a real distance-decay
result. Not yet placed in the paper. `scripts/paper_figures/{extract,plot}_shap_edge_
directionality.py`, `aaai2027/figure_data/shap_edge_directionality.csv`.

Full day-by-day build history (10-seed migration, SiGAT campaign, entropy-heatmap
methodology decision, readability pass, per-figure fixes): `SOTA_HISTORY.md`.

## Hardness reweighting (H): scrapped everywhere (2026-07-19, final)

**Not in use, full stop — neither full attention nor LocalAttn4.** Exhaustively tested
across both architectures, 2 hardness maps (old learned miner, role-aware entropy),
3 reweighting formulas (mean, max, power=2), and an asymmetric source/target mixing
weight (`hardness_combine=weighted`) — no combination gives a robust, reproducible win
on more than 1 dataset at a time, and the one repeated "apparent win" (bitcoin-alpha)
fails a formula-robustness check every time it's tried (see `HARDNESS_MINER_ROADMAP.md`
"Final synthesis" and the E27/E28/E29 tables there for the complete numbers, including
the asymmetric-weight pilot). **Production configs must not set `hardness_lambda`/
`hardness_map_path`/`hardness_source_map_path`/`hardness_target_map_path` anywhere** —
already the default in every `configs/<dataset>.yaml`. The `hardness_source_weight`
knob (`lit_model.py`) stays implemented for any future revisit but is not swept further
absent a new candidate mechanism — do not resume tuning it on the strength of the
Lead4c coefficients alone (test-set leakage, see the roadmap doc). Old hardness-map
artifact paths (E14/E17/E22), kept on disk for provenance only, not used anywhere:
`SOTA_HISTORY.md`.

## Attention variant: full vs. local — SETTLED 2026-07-20, LocalAttn4 is the confirmed default

**LocalAttn4 (`model.local_attention_window=4`) is the production default and the paper's
primary vehicle for the "proximal" claim.** Settled by a direct mechanism check, not just
AUC or narrative preference.

An E30 pilot tried the paper's own proposed cleaner-looking alternative — literally
shortening `dataset.max_walk_length` instead of masking attention on a long walk. Short
walks (L=8/L=16) looked competitive or better than LocalAttn4 on raw AUC, but a follow-up
measurement (`scripts/measure_local_context_availability.py`, replicates the real
dynamic-masking target sampling against cached walk data) found the confound: at L=2 (the
length that actually matches LocalAttn4's ±2-hop window), a target edge has *zero* labeled
context to condition on 43–55% of the time — a 2-edge walk rarely has room for both a
target and a neighbor, and raising the target-selection ratio to compensate makes this
worse, not better. LocalAttn4 on a long walk avoids this because the walk keeps wandering
through the real graph instead of being cut off:

| L | bitcoin-alpha: local-window context available | epinions: local-window context |
|---|---|---|
| 2 | 56.8% | 45.4% |
| 8 | 84.8% | 82.3% |
| 80 (LocalAttn4) | 90.6% | — |

**Verdict:** full-attention-with-short-walks is not pursued further as the production
alternative — it can't cleanly demonstrate the proximal-context mechanism the paper claims,
regardless of masking-rate tuning. LocalAttn4 is the mechanistically honest way to get
±2-hop locality without conflating "restrict information" with "restrict the walk sample
itself." Not extended past bitcoin-alpha/epinions — the question it was meant to resolve
is answered. Full pilot detail/derivation: `SOTA_HISTORY.md`.

## Training feature flags (confirmed beneficial — already config.yaml defaults)

- **D — Dynamic resplit** (`dynamic_train_masking: true`): re-assigns supervised targets each
  epoch within TRAIN+MASK. Prevents memorizing which edges get predicted.
- **R — Node token replacement** (`node_context_mode: replace`, `node_replace_prob: 0.2`):
  randomly replaces node tokens with [UNK] or another node. Reduces reliance on node identity.
- **H — Hard-node reweighting**: **scrapped everywhere, final (2026-07-19).** See
  "Hardness reweighting (H): scrapped everywhere" above for the full picture (both
  architectures, all maps/formulas, including the old "load-bearing for LocalAttn4"
  claim's retraction). Production configs must not set `hardness_lambda`/
  `hardness_map_path`/`hardness_source_map_path`/`hardness_target_map_path` anywhere.
- **L — Local attention window** (`local_attention_window`): null=full attention;
  4=±2-hop banded mask (LocalAttn4, **settled default, 2026-07-20** — see "Attention
  variant: full vs. local" above; confirmed via direct context-availability
  measurement, not just AUC or narrative preference). **`config.yaml`'s actual default
  only matched this claim starting 2026-08-04** — before that it was `null` and every
  real LocalAttn4 run relied on a manual CLI override (see the "Key commands" note
  above); now enforced at the config level, no override needed.

D and R are load-bearing defaults, not ablation toggles — don't disable them without reason.
H is scrapped everywhere — do not enable it on either attention variant. L defaults to
LocalAttn4, settled (see above) — full-attention+short-walks is not the production
alternative.

## Ablation campaign — vertex/edge/direction (started 2026-08-21, in progress)

Three new **ablation-only** flags (not production defaults, all default `false`), added to
answer `aaai2027/WSDM_format_revised.tex`'s own line-331 marker ("randomize the edge
direction... remove the vertex token... remove the edge token..."). Pilot-validated
(single-seed, no crashes/NaN, sensible AUC drops) before launching the full campaign.

- **`model.mask_node_tokens`**: every vertex/node token → `<UNK>` (train+eval). Isolates how
  much signal edge signs alone carry. Implemented in `model.py`/`lit_model.py`
  (`_maybe_apply_token_masking`).
- **`model.mask_edge_tokens`**: every edge/sign token → `<MASK>` (train+eval). Isolates how
  much signal vertex identity alone carries. Same implementation site.
- **`model.randomize_walk_direction`**: NOT a per-token swap (that fabricates edges between
  node pairs that were never adjacent — interior walk positions are shared between two
  edges, so swapping one edge's flanking pair independently corrupts its neighbor into a
  fictitious edge; verified concretely, see `aaai2027/PAPER_CLOSEOUT_LOG.md`'s 2026-08-21
  entry). Instead: one **fixed, per-walk_id** coin flip (not per-draw-random, not a single
  global "always flip" — see the log for why each of those is wrong), computed once and
  shared by reference across the train/val/test dataset views (`create_stage_dataloaders`
  in `src/data/stage_dataset.py`), reusing `get_seed(cfg)` + a fixed offset (mirrors
  `dynamic_train_mask_seed_offset`'s pattern). If flagged, the whole walk is reversed
  (`input_ids`/`edge_split_mask`/`edge_ids` all flipped together, before any split/label/
  masking logic runs) — real edges keep their real endpoints and signs, just presented in
  reversed reading order. Because a single edge appears in many different walks (mean
  occurrences/edge range ~6 to ~216 depending on dataset — see Ablation~\ref{abl:singlewalk}
  in the tex), each with its own independent flip decision, the same edge ends up presented
  forward in some occurrences and reversed in others, genuinely destroying usable
  directionality at the edge level without ever fabricating an edge within any single walk.
  No graph-level changes, no new dataset cache, no walk-sampler changes — implemented
  entirely in `src/data/stage_dataset.py` (`StageViewDataset._getitem_ragged`).

**Campaign**: `scripts/run_ablation_campaign.py`, launched under nohup, logs at
`logs/ablation_campaign/`. 4-GPU queue-worker pattern (same as `run_multiseed_pewter.py`):
one worker thread per GPU, each does train→posthoc→next-job with zero idle time. 3
ablations × 6 datasets × 10 seeds (42-51) = 180 jobs, local attention only, compared against
the existing Table 1 `Pewter (local attention)` numbers (no new baseline retrain needed).
Order: `mask_node_tokens` block (60 jobs) → `mask_edge_tokens` block (60) → `randomize_walk_
direction` block (60), each block ordered easy-to-heavy by *measured* wall-clock time (wiki-
elec, bitcoin-alpha, bitcoin-otc, wiki-rfa, epinions, slashdot090221 — not the walk-budget
order used elsewhere, which doesn't track measured time). Posthoc computes `func_logit_power`
only (not all 11 aggregators — this ablation isn't about aggregator choice). No dataset cache
is touched by any of the three ablations.

**Live-extendable, no restart needed**: the driver never exits on its own — a background
thread polls `logs/ablation_campaign/extra_jobs/` every 30s for new `*.json` job-batch files
and pushes them onto the same live queue. Use `scripts/enqueue_extra_ablation_jobs.py` to add
a batch to a *running* driver. A `logs/ablation_campaign/STOP` file is the clean-shutdown
signal. **Caveat, confirmed this session**: killing the driver process does NOT stop an
in-flight training subprocess (plain `subprocess.run`, no process-group isolation — the
child is a separate PID, gets orphaned/reparented, keeps running to completion) — but it
DOES lose the driver's own bookkeeping for that job, since `job_train_and_posthoc` always
retrains from scratch rather than resuming from a checkpoint, so a restarted driver would
dispatch a duplicate of whatever was in-flight at kill time. Don't restart the driver
casually; let it drain or use the STOP file.

**Status as of 2026-08-21 session end**: MASKNODE 43/60 done (0 failures), MASKEDGE and
DIRFLIP not yet started (queued behind MASKNODE). No numbers written into the tex yet —
wait for the campaign to finish before filling in the line-331 marker.

## Performance philosophy — READ BEFORE ADDING ANY NEW FEATURE

Significant engineering time went into making data handling/training fast: bucket batching
and CSR ragged-tensor caching (see git history of src/data/, "Implement ragged (CSR) format
for dataset caching and loading"). **Any new feature, experiment, or training mechanic must
consider hardware/time efficiency from the start** — don't bolt something on that silently
reintroduces O(n²) padding or defeats the CSR caching. When in doubt, benchmark before and
after. **2026-06-28: local attention's masked-SDPA path WAS slow (37% wall-clock / 4x memory
vs. full attention) — fixed in `src/model/model.py` (`LocalAttentionEncoderLayer`), see
plan-performance.md for the diagnosis.**

## Cost/performance tradeoff default: prefer cheap unless the gain is real

When sweeping a cost knob on an *already-adopted* mechanism (walk budget, retry
counts, epoch count — not "should we adopt mechanism X," a separate judgment call),
the default lean is toward the cheaper/faster setting when the pricier one gains
**<0.25pp test AUC**, flat across all 6 datasets (generalizes the bar the H-ablation
section already used — "never exceed +0.25pp" = not a robust win — rather than
inventing a new threshold). **This is a lean, not an auto-apply rule: any time it
would pick the cheaper/lower-AUC setting over a more expensive one, get explicit user
sign-off before adopting it** — surface the comparison (cost delta + AUC delta),
don't silently swap configs. **Exception:** the headline six-dataset SOTA table is
exempt — squeezing the max within an already-validated grid is its explicit purpose.

**Sweep/ablation reporting: always keep the full table, not just the winner.** Every
config tested in a sweep or ablation must be recorded in the results doc (numbers,
not just the pick) — even the ones not chosen. This is what lets future work push
toward either extreme (max performance, ignoring cost; or max cheapness, accepting
more loss) using data already in hand, instead of re-running the sweep. Already the
de facto style in this project's `outputs/walk_coverage_analysis/*.md` docs (see e.g.
`E15_SWEEP_RESULTS.md`) — this makes it an explicit requirement, not just a habit.

## Key file locations

- **`SOTA_HISTORY.md`** — archived detail behind "Current SOTA" and "Attention variant"
  above: the old superseded single-split SOTA table, the 10-seed migration/SiGAT-campaign
  build history, entropy-heatmap methodology decision, SHAP figure detail, E30 short-walk
  pilot derivation, and old hardness-map artifact paths.
- **`aaai2027/PAPER_CLOSEOUT_LOG.md`** — archived day-by-day session log behind "Open
  threads" and "PEWTER paper" below: the WSDM closeout punch-list history, per-figure
  rebuild/bug-fix narrative (Panel A/C/D/E, delta-heatmap regression, K-ablation, etc.).
- Config: config.yaml, configs/<dataset>.yaml (partial overlays merged via dataset.name=<ds>)
- Model: src/model/model.py, src/model/lit_model.py
- **`MASKING.md`** — full picture of all masking (split masking, key padding
  mask, local attention window), the NaN bug found/fixed in
  `LocalAttentionEncoderLayer` 2026-07-19 and its root cause, and the
  merge-once design. Regression test/benchmark: scripts/test_local_attention_masking.py.
- `scripts/measure_local_context_availability.py` — read-only diagnostic (no training):
  replicates the exact dynamic-masking target sampling against a real cached walk file
  and measures whether target edges actually have a labeled neighbor edge visible as
  context, either within a LocalAttn4-style window or anywhere in the walk. This is what
  settled the LocalAttn4-vs-short-walk-truncation question — see "Attention variant: full
  vs. local" above.
- Data: src/data/datasets.py, src/data/walk_sampler.py, src/data/tokenizer.py
- **`aaai2027/DATASET_STATS.md`** — cheap graph statistics (size, density, degree
  distribution, sign balance, reciprocity, weak connectivity/giant-component fraction,
  sampled eccentricity/clustering coefficient) for all 6 canonical datasets, plus a
  Spearman-correlation cross-reference against Panel B's MI/phi "bump" size — generated
  so these numbers don't get recomputed from scratch every time a new diagnostic needs
  them. Regenerate via `scripts/paper_figures/compute_dataset_stats.py` (also writes the
  machine-readable `aaai2027/figure_data/dataset_stats.csv`); don't hand-edit the `.md`.
- Shared edge loader: scripts/balance_theory_paths.py → load_edges_canonical() — a thin
  wrapper around the same `get_loader`/`load_config` production functions
  `src/data/prepare_data.py` uses, so it loads real canonical training-identical edges.
  **Known duplication (harmless, worth cleaning up eventually):** `scripts/
  node_mi_structural_embedding.py` defines an independent, currently-identical copy of the
  same `DATASET_CONFIGS`/`load_edges_canonical()`. **New scripts should import
  `get_loader`/`load_config` directly from `src/data/datasets.py`/`src/utils/config.py`**
  rather than reaching into either analysis script's copy — don't add a third copy.
- Hardness miner: scripts/compute_hardness_map.py. Candidate-metric screening
  (Q5 pre-filter, correlates a candidate hardness signal against a trained
  checkpoint's real per-node error rate without retraining):
  scripts/hardness_predictive_validity.py — see plan-hardness-miner.md.
  **`HARDNESS_MINER_ROADMAP.md`** — ranked, time/gain-estimated task list for
  the hardness-miner investigation (entropy-based map, reweighting formula,
  short walks, etc.) — check here before starting new miner work; any retrain
  item on it requires explicit approval before launching.
- Analysis scripts: scripts/edge_sign_mi_vs_distance_v3.py, scripts/node_mi_structural_embedding.py,
                    scripts/attention_analysis.py, scripts/lead4_entropy_heterogeneity.py,
                    scripts/lead4_twohop_path_consistency.py, scripts/lead4c_entropy_logit_regression.py
- MI reports: outputs/mi_analysis_package.zip (full), outputs/mi_vs_dist/, outputs/node_mi_structural/,
              outputs/attention_analysis/
- Lead 4/4b/4c (entropy directionality — headline research thread, see Research status above):
              outputs/lead4_entropy_heterogeneity/ (Lead 4, bucketed AUC drop, predictions_raw_canonical.pkl
              shared across all leads), outputs/lead4_twohop_path_consistency/ (Lead 4b, 2-hop path
              consistency), outputs/lead4c_entropy_logit_regression/ (Lead 4c, atomic + composite +
              marginal3 logistic regression, fit_results.csv, figures, script
              scripts/lead4c_entropy_logit_regression.py). Consolidated report: LEAD4_ENTROPY_REPORT.md.
              Equations: LEAD4C_EQUATIONS.md. Running notes/Q&A: LEAD4C_ASYMMETRY.md. Handoff:
              LEAD4C_HANDOFF.md. Coefficients: lead4_coefficients.csv/.md.
- Baselines: baselines/all_results.csv, baselines/<model>/results_our_splits/. Convention:
  unqualified "SiGAT" always means raw SiGAT, not SGA-augmented, everywhere in this
  codebase/paper unless explicitly marked otherwise.
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

## Research status (leads content as of 2026-07-07, cross-checked against plan files 2026-07-19 — no changes needed, all still accurate)

**Central question:** the walk-Transformer beats every GNN/SGNN baseline by 3–5pp AUC on all 6
datasets despite edge-sign MI collapsing 10–1000× beyond 1 hop. Full history: `RESEARCH_LEADS_SUMMARY.md`.
Leads 1–3 (over-averaging, bottleneck, swamping) are each real but individually insufficient — one-liners
below. **Lead 4/4c is the live thread and the strongest signal found so far; a paper draft is in prep
around it.**

| Lead | Verdict | Detail |
|---|---|---|
| MI/attention baseline | MI collapses 10–1000× at d=1→2; full-attention model still attends far (provisional, Lead 2 bug caveat) | `outputs/mi_analysis_package.zip` |
| 1 — over-averaging | Real (cancellation r≈−0.96) but modest (5–9% norm loss) — not primary | `LEAD1_GNN_OVER_AVERAGING_REPORT.md` |
| 2 — bottleneck | Real (NMI 0.008–0.27) but oracle-bypass gain is architecture-dependent (GINEConv 6/6, CSG 1/6) | `LEAD2_GNN_BOTTLENECK_STATUS.md` |
| 3 — swamping | Severe in theory, but walk attention shows no *learned* compensation — avoidance is structural | `LEAD3_FOG_OF_WAR_REPORT.md` |

### Lead 4/4b/4c — entropy directionality (headline result)

**The finding:** decomposing node sign-entropy into 6 atomic directions and fitting them jointly
(logistic regression, `correct ~ src_out+src_in+tgt_out+tgt_in+twohop_in+twohop_out`, cluster-robust
SEs) reveals a **source/target asymmetry**, not a uniform "GNNs worse with entropy" story:
- **`tgt_in`** (how contested v's reputation already is): hurts **GNNs** more, on 100% of datasets (p=0.031).
- **`src_out`** (how consistent a rater u is): hurts the **walk model** more — the single largest effect measured.
- `tgt_out`/`src_in` ≈ null; 2-hop terms negligible.
- Architectural explanation, corroborated independently two ways (entropy regression + a separate
  `log(outdeg(u))`/`log(indeg(v))` degree regression): GNNs (GINEConv) build v's embedding purely
  from message-passing over v's in-neighbors, so **u's own outgoing behavior is structurally invisible
  to them** — the walk model and SiGAT (attention-based) can both see and use it, GINEConv can't
  (β≈0, n.s., on out-degree(u) specifically).
- Ruled out: forward-walk-sampling context asymmetry (measured directly — left/right context around
  test edges is symmetric, 55.6 vs 55.5 mean tokens, anchor walks only 2.9%). The asymmetry is about
  how each architecture *uses* context, not how much of it exists.

**External validation (professor's independent information-theoretic argument, 2026-07-05):**
predicted that a node's outgoing-edge sign entropy should be inherently lower than its incoming-edge
entropy (raters are self-consistent; received opinions are noisier) — confirmed on
bitcoin-alpha/bitcoin-otc/epinions/slashdot (p≤5e-4), **reversed on wiki-elec/wiki-rfa** (p=1.00
against his direction — these are vote/election graphs where a few admin-candidate nodes concentrate
large, genuinely mixed in-vote counts).

**This split is not noise — it cleanly tracks the SOTA gap.** The 2 datasets where the entropy
asymmetry reverses (wiki-elec, wiki-rfa) are exactly the 2 smallest walk-vs-GNN AUC gaps (1.0–1.1pp);
the 4 where it holds are the 4 largest gaps (3.1–4.6pp) — clean separation, no overlap (Spearman
ρ≈0.71–0.77, n=6). **Where the underlying sign data has real directional asymmetry to exploit, the
walk model's advantage is large; where the data itself is direction-symmetric (or reversed), the
achievable advantage shrinks toward zero.** This is the strongest causal-adjacent evidence in the
whole investigation — it ties the architectural mechanism to an inherent property of the data, not
just a model artifact.

**Confirmed further (2026-07-06): the reversal on wiki is a genuine flip, not just an absent effect.**
An explicit reversed-direction test (H_out > H_in) is itself significant on wiki-elec/wiki-rfa
(p=1.7e-10, p=3.4e-14). It also shows up independently in the original pre-atomic bucketed-AUC-drop
analysis: on the `out_in` bucket variant, the walk model drops *more* than GINEConv on both wiki
datasets (reversed from every other dataset), while the `in_in` variant still shows GNN dropping
more (unflipped) — reconciled by the atomic model: `tgt_in`'s GNN-worse gap shrinks a lot on wiki but
doesn't reverse; `src_out`'s walk-worse gap (which never reverses on any dataset) becomes relatively
dominant once `tgt_in`'s gap shrinks, flipping the *combined* bucket's ranking without any single
atomic term actually changing sign.

Full writeup + equations + Q&A: `LEAD4C_ASYMMETRY.md`, `LEAD4C_EQUATIONS.md`, `LEAD4_ENTROPY_REPORT.md`.
Coefficients: `lead4_coefficients.csv`/`.md`. Script: `scripts/lead4c_entropy_logit_regression.py`.
Handoff doc: `LEAD4C_HANDOFF.md`.

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

### Open threads (not yet started)

**Leads 5 and 6 — scrapped (2026-08-20), never started.** Lead 5 (per-walk prediction
variance / ensemble effect) is superseded by the PEWTER paper's own K-ablation
(`Ablation~\ref{abl:singlewalk}` in `aaai2027/WSDM_format_revised.tex`), which already
answers the question Lead 5 was going to investigate ("is the gain just ensembling?") —
a single walk with no aggregation already beats the best baseline on all six datasets.
Lead 6 (training-regime confound: D/R/H tricks, capacity mismatch, missing
trainable/spectral features in GNN baselines) is dropped with no replacement, per
explicit user decision. `plan-lead5-ensemble-effect.md`/`plan-lead6-trainable-
features.md` are not deleted (left on disk for provenance) but neither is active work.

**PEWTER paper (`aaai2027/`) — ACTIVE, top priority.** Now WSDM-targeted
(`aaai2027/WSDM_format_revised.tex`), in closeout/punch-list phase. Resume via
`~/.claude/plans/adaptive-watching-ember.md`. Live checklist:
`aaai2027/PEWTER_ASSETS_CHECKLIST.md` — read before touching any figure/table. Standing
rules from the professor's 2026-08-18 direct-edit pass, apply to all future work on this
paper:
1. `pewter_references.bib` is canonical as-is — never edit an entry, only fix
   `\cite`/`\citep` calls to match existing keys; any bib add/remove needs explicit user
   sign-off first.
2. Math-proof findings (Propositions/Lemma/Corollary/Appendix) must be presented and
   confirmed with the user before any edit, never silently patched.
3. GINE is dropped from the paper (Table 1, baselines text) — `run_with_our_splits.py`'s
   `train_edge_attr` is the raw unembedded sign scalar, a training-time self-referential
   shortcut specific to GINEConv. Only restore if a learned-edge-embedding experiment
   succeeds (not yet run).
4. `aaai2027/STATISTICAL_TESTS_AUDIT.md` catalogs every empirical claim in the paper
   against what statistical test backs it — read before adding or touching any
   significance claim. `aaai2027/STATISTICS_ELI5_GUIDE.md` is the plain-language
   reference for the tests/terms used.

(See "Current SOTA" above for the dataset-name / panel-letter canonicalization rules.)

**Complexity claim (Sec 6.6, O(L²d)→O(Lwd) for local attention):** correct as a
*theoretical* claim (professor signed off) but not realized by the current
`LocalAttentionEncoderLayer` implementation — it computes full dense L×L attention and
applies the window as a post-hoc mask, so no real speedup exists today (see `MASKING.md`'s
benchmark table). Future-work item, not scheduled: implement genuine sparse/windowed
attention. Don't start without explicit user go-ahead.

**Everything else** (10-seed SiGAT/SNEA/CopulaLSP campaigns, entropy-heatmap methodology,
SHAP directionality figure, per-figure rebuild/bug-fix history, K-ablation, cluster-robust
SE work, two-way ANOVA, delta-heatmap regression, Ablation B multiseed rebuild, Panel
A/C/D/E rebuild history) — done, full day-by-day log: `aaai2027/PAPER_CLOSEOUT_LOG.md`.

**Other open plans** (not urgent, see `~/.claude/plans/`):
- `plan-stats-rigor.md` — **CLOSED 2026-08-20.** Multi-seed variance / cross-validation /
  significance testing is now the de facto standard throughout the paper (10-seed
  campaign, paired Wilcoxon tests everywhere a comparison is claimed) — the goal this
  plan was tracking is done, achieved through the paper closeout work itself rather than
  by executing this plan file directly. `optuna_run.py` is still stale (see its own
  top-of-file marker) but that's now tracked as a separate, low-priority item, not
  gating this closure.
- `plan-side-quests-misc.md` — docs/config/repo-hygiene/research-follow-up backlog. 3/10
  items done (doc fix, git cleanup, SiGAT+SGA baseline); rest still open, lowest priority.
- `plan-lead5-ensemble-effect.md` / `plan-lead6-trainable-features.md` — **scrapped
  2026-08-20**, see "Open threads" above.
- `plan-a-fix-for-glimmering-panda.md` — CLOSED 2026-07-19 (walk sampler fix, see "Walk
  sampler" above).
- `plan-cleanup-and-local-attention-prompts.md` — not a research plan, a pair of
  self-contained session-starter prompts for two future sessions: (1) production cleanup
  of this repo into a new, minimal, double-blind-compliant public repo for the paper's
  anonymous code link, (2) a genuine sparse/windowed local-attention implementation
  (current `LocalAttentionEncoderLayer` is dense-masked, not actually sub-quadratic —
  see `MASKING.md`'s benchmark). Neither started; paste either prompt into a fresh
  session to begin.

## PEWTER paper (aaai2027/) — file map and conventions

Most day-to-day session time goes into `aaai2027/`, not new modeling experiments — see
"Open threads" above for current status/priority. This section is the fast-lookup index
for the file layout and script conventions.

**File map:**
- `aaai2027/pewter_aaai.tex` — the paper source. `aaai2027/PEWTER_ASSETS_CHECKLIST.md` — the
  live, row-per-marker/figure checklist (status + pointers); this is the first thing to read
  when resuming any paper subtask, before re-deriving anything from scratch.
- `scripts/paper_figures/` — one script pair per figure/table, named
  `extract_<name>.py` / `plot_<name>.py`:
  - **extract**: recomputes real numbers from the actual source data (checkpoints, cached
    predictions, `computed_data.pkl`, canonical edge loaders) and writes a CSV to
    `aaai2027/figure_data/<name>.csv`. This is the only step that touches raw data — rerun it
    only when the underlying data/checkpoint changes.
    - **plot**: pure rendering — reads the CSV, writes a PNG to `aaai2027/figures/<name>.png`.
    Safe to edit freely (colors/labels/layout) without recomputation.
  - `combine_<name>.py` — stacks two or more already-rendered PNGs into one multi-panel image
    (so LaTeX treats a multi-panel figure as a single float instead of drifting apart on the
    page). Rerun after re-plotting any panel it combines.
  - `aaai2027/figure_data/*.csv` and `aaai2027/figures/*.png` are both regenerable — safe to
    delete/regenerate, not hand-edited.
  - **Standing rule (made explicit 2026-08-11, was only implicit before): if an extract
    script's expensive step (loading/fitting raw per-seed predictions, refitting a
    classifier, etc.) is separable from a cheap cosmetic parameter (a binning threshold,
    bin edges, which columns to write), cache the expensive step's output to disk on its
    own, separately from the final CSV** — so changing the cheap parameter later never
    re-pays the expensive cost. Bit by this exact gap 2026-08-11:
    `extract_multiseed_entropy_heatmaps.py` only cached the final binned CSV, so bumping
    the delta heatmap's `MIN_CELL_N` (a pure re-thresholding of already-computed
    predictions) forced a full ~75min rerun of the SiGAT LogisticRegression refits (60
    fits) and PEWTER prediction reloads. Fixed by splitting `per_seed_records()`
    (expensive: load+fit, cached to `outputs/cache/multiseed_entropy_records/<ds>__<model>.pkl`)
    from `grids_from_records()` (cheap: bin at any threshold, no cache needed) — delete a
    cache file to force a real recompute for that dataset/model (e.g. after a checkpoint
    changes), same "safe to delete/regenerate" convention as the figure_data CSVs
    themselves.
- **Rule going forward: every `PEWTER_ASSETS_CHECKLIST.md` row that reaches DONE/NEEDS-FIGURE
  status must name its generating script(s) and output path(s)** (the `Scripts:`/`Data:`
  pattern already used in rows #21/#23/#25/#29/#30/#30b) — not just "done", so a future session
  (or this one, post-compaction) can jump straight to the code instead of rediscovering which
  script produced a given figure. Add the pointer in the same edit that changes the status.
- **Logging rule (added 2026-08-18, after the CLAUDE.md cleanup that created this file):
  day-by-day session narrative for paper closeout work goes in
  `aaai2027/PAPER_CLOSEOUT_LOG.md`, not CLAUDE.md.** When a session does paper work (fixes
  a figure bug, reruns a stats test, makes a presentation call, tries and rejects an
  approach, etc.), append the dated writeup — what changed, why, what was tried — directly
  to `PAPER_CLOSEOUT_LOG.md`. Only touch CLAUDE.md when something needs to change at the
  *standing-rule/current-status* level (a new convention every future session must know at
  a glance, a table number, a status flip) — not to narrate how it got there. This is what
  keeps CLAUDE.md from re-bloating the way it did before this split; if a CLAUDE.md edit is
  starting to read like a story instead of a fact, it belongs in the log instead.

**Figure rework history** — Figure 1 (Empirical Confirmation panels) rebuild, Panel B's
post-minimum "bump" investigation (real, mechanism still being pinned down — recommended
next step: deduplicate by distinct context edge, not yet implemented), Panel A schematic
rebuild, Panel C/D rework, and all associated bug fixes: done, full history in
`aaai2027/PAPER_CLOSEOUT_LOG.md`.

## Session management tips

- Start a new chat for each independent workstream.
- In new chat: CLAUDE.md loads automatically. Say "read ~/.claude/plans/plan-<X>.md and do step N".
- Use /compact when finishing a sub-task within a session.
- Update this CLAUDE.md after completing a workstream (update status table above).
- Never paste long files — reference by path.
