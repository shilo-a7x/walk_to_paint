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

**Historical bug, now fixed.** `run_posthoc.py` used to rebuild `cfg` purely
from `configs/<ds>.yaml` + CLI dotlist overrides, completely ignoring the
checkpoint. Any non-default `dataset.*` override used at training time (or,
found later, `model.local_attention_window`) had to be retyped verbatim on
the posthoc call, or it silently evaluated the wrong walk cache/architecture.
This bit three times: twice on `dataset.*` (`E16_NOHARD_RESULTS.md`
2026-07-07 — LocalAttn4 rows used the full-budget cache instead of the local
one; E24/E25 walk-dedup sweeps 2026-07-17, caught before being reported —
`outputs/walk_coverage_analysis/E24_BP_SWEEP_RESULTS.md` correction note) and
once on `model.local_attention_window` (2026-07-19 — LocalAttn4 checkpoints
were silently re-evaluated as full attention unless the flag was manually
repeated; see `MASKING.md`'s re-verification list).

**Fix:** `LitEdgeClassifier.save_hyperparameters()` (`src/model/lit_model.py`)
already saves the *full* resolved training-time cfg into every checkpoint —
nothing was ever lost at save time, `run_posthoc.py` just never read it back.
It now loads that saved cfg first and uses it as the base for both the data
pipeline and the model; CLI positional overrides are merged on top only for
deliberate, explicit changes and are no longer required to reproduce the
training-time setup. (Same change also fixed a latent `validate_config` bug:
`model.class_weights` round-trips through the checkpoint as an OmegaConf
`ListConfig`, which the old `isinstance(x, (list, tuple))` check rejected —
never triggered before because `class_weights` is attached to `cfg` *after*
training's own `validate_config` call, so posthoc's new checkpoint-cfg path
was the first caller to ever see it; fixed in `src/utils/config.py` by
allowing `ListConfig` too.)

**Verified against real checkpoints (2026-07-19):** an epinions LocalAttn4
checkpoint (`E16_NOHARD_KCOVER_K5_NW2000000_local`) posted with only
`dataset.name=epinions` (the old under-specified invocation that used to
silently break) now correctly recovers `walk_strategy=k_cover,
num_walks=2000000, local_attention_window=4` straight from the checkpoint —
edge test AUC=0.9570, 0% NaN, matching `MASKING.md`'s independently-verified
reference number for this exact checkpoint. A full-attention checkpoint
posted the same way reproduced the existing E16 number exactly (0.9572),
confirming zero regression on the already-correct common path. Checkpoints
saved before `save_hyperparameters` existed (if any) fall back to the old
config.yaml+CLI behavior with a loud warning — the old "repeat every
`dataset.*` override" rule still applies only to those.

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

(Note: optuna_run.py is suspected stale — see `plan-stats-rigor.md` (restored
2026-07-19 after going missing from `~/.claude/plans/`, see OPEN WORKSTREAMS below;
nothing in it has been executed yet).)

## Canonical splits

train:val:test = 0.8:0.1:0.1, seed=42 (walk model: nested 4-way train0.48/mask0.32/val0.1/test0.1).

**CORRECTION (was wrong):** the old claim "baselines/ uses identical splits — apples-to-apples"
does NOT hold. The legacy `baselines/splits/*.pt` and SGA CSVs were generated *independently* of
the walk split (different RNG + fabricated reverse edges), so walk-test and GNN-test overlapped
only ~10%. Fixed by `baselines/prepare_splits.py::build_canonical_split`, which re-derives the
baseline artifacts from the frozen walk split (`data/<ds>/dataset_cache.pt["splits"]`) into
isolated `baselines/splits_canonical/`. Full story: `SPLIT_PROVENANCE.md` (+ `FABRICATED_REVERSE_EDGES.md`).

**Walk coverage caveat — RESOLVED (2026-06-29) by the E15 k_cover k=5 sampler; sampler
itself since SUPERSEDED (2026-07-19) by `edge_cover`.** The old uniform sampler only
predicted a test edge if it appeared in a sampled walk, so on sparse graphs it evaluated
a *subset* of the nominal test split (bitcoin-alpha/otc 100%, slashdot 98%, epinions 88%,
wiki-rfa 86%, wiki-elec 85%) while GNNs saw all of it. The E15 edge-anchored `k_cover`
sampler (`walk_strategy=k_cover`, `walk_k_min=5`) fixed that, driving node AND edge
coverage to ~100% on all 6 — but its anchor walks turned out to contain a lot of *exact
duplicate* walks (dead-end targets collapse all k=5 "visits" to 1 repeated context;
29% of all walks on wiki-elec, measured directly). This didn't affect coverage or cause
leakage, but it's bad training practice (repeated identical gradient steps) and
undermined the "k=5 distinct views" premise. Investigated and fixed in
`~/.claude/plans/plan-a-fix-for-glimmering-panda.md`: re-deriving k under a
duplication-free sampler found k=1 is sufficient (k=5's marginal AUC gain was weak/
negative once duplicates were removed), so the sampler was simplified to
**`walk_strategy=edge_cover`** — one forced, provably-globally-distinct anchor walk per
edge (no `walk_k_min`), with a hardened dedup-fill phase that **hard-fails
(`RuntimeError`)** rather than silently padding with duplicates if a budget can't be
honored. Measured 0.000000% duplication at production scale. **`edge_cover` is now the
default in every `configs/<dataset>.yaml`**, with budgets freshly re-swept per dataset
(not copied from the old `k_cover` absolute walk counts) — see "Walk sampler" below.
Coverage remains ~100% on all 6 (provable, not just measured, since `edge_cover`
guarantees coverage once `num_walks >= |E|`). Details: `WALK_COVERAGE.md`,
`outputs/walk_coverage_analysis/{E15_SWEEP_RESULTS,E24_BP_SWEEP_RESULTS,
E25_BUDGET_SWEEP_RESULTS,E26_WIKI_SWEEP_RESULTS}.md`,
`~/.claude/plans/plan-a-fix-for-glimmering-panda.md`.

## Walk sampler

**Current default: `edge_cover` (k=1, zero-duplication guarantee)** — adopted
2026-07-19, replacing `k_cover k=5`. Every edge gets exactly one forced anchor walk;
since each edge's own `(u, label, v)` triple is unique, anchor-anchor collision is
impossible *by construction*, not just unlikely — this is what makes the
zero-duplication guarantee provable rather than empirical. Remaining budget beyond
`|E|` is filled via a dedup-retry phase that raises `RuntimeError` (not silent
duplicate-padding) if it can't find enough genuinely distinct walks. `k_cover`/
`k_cover_bp` (k>1-capable, backward-prefix anchors) are untouched and remain
available as a fallback, just no longer the default.

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

**Retrain status: DONE (2026-07-19), contradicting an earlier version of this note —
corrected 2026-07-30.** The full 6-dataset retrain-and-report against the SOTA table
below is complete: the "Current SOTA" section's Full-attn (E25/E26) and LocalAttn4
(E27) columns already reflect the `edge_cover` sampler at each dataset's adopted
production budget, not the old `k_cover` checkpoints. Re-verified 2026-07-30 by exact
numeric match against `outputs/walk_coverage_analysis/E25_BUDGET_SWEEP_RESULTS.md`
(bitcoin-alpha 5×=0.9219, matches the SOTA table exactly), `E26_WIKI_SWEEP_RESULTS.md`
(wiki-elec 1.5×=0.9036, matches exactly), and `HARDNESS_MINER_ROADMAP.md`'s E27 table
(all 6 LocalAttn4 values match the SOTA table exactly). Re-running Leads 1/4/4b on the
new predictions remains a separate, not-yet-done follow-up (unaffected by this
correction) — see "Open threads" in Research status below.

## Walk encoding

max_walk_length=80 HOPS (not tokens). Full walk = up to 161 tokens.
Token layout: N_u0, E_s1, N_u1, E_s2, ... (alternating node/edge).
1 graph-hop = 2 token positions. window=4 tokens = ±2 graph-hops.

## Current SOTA (func_logit_power, test AUC)

**Superseded 2026-08-10: the table below is single-split (E25/E26/E31/E32), kept for
history — the WSDM paper's Table 1 (`aaai2027/WSDM_format_revised.tex`) no longer uses
it.** A 10-seed multi-seed campaign (seed 42 reused + 43–51 new, driver
`scripts/run_multiseed_pewter.py`) now provides real mean±std AUC for both PEWTER
attention variants and GINEConv on all 6 datasets, replacing the single-split
Hanley–McNeil numbers in the paper table. New canonical PEWTER numbers (mean±std
across 10 splits, `func_logit_power`):

| dataset | full attention | local attention | GINEConv (10-split) |
|---|---|---|---|
| bitcoin-alpha | 0.9146 ± 0.0102 | 0.9134 ± 0.0173 | 0.8497 ± 0.0189 |
| bitcoin-otc | 0.9315 ± 0.0073 | 0.9317 ± 0.0065 | 0.8877 ± 0.0089 |
| epinions | 0.9523 ± 0.0012 | 0.9536 ± 0.0015 | 0.8612 ± 0.0043 |
| wiki-elec | 0.9008 ± 0.0030 | 0.9023 ± 0.0031 | 0.8730 ± 0.0073 |
| wiki-rfa | 0.8923 ± 0.0025 | 0.8914 ± 0.0046 | 0.8606 ± 0.0055 |
| slashdot090221 | 0.8989 ± 0.0015 | 0.8968 ± 0.0016 | 0.7864 ± 0.0082 |

All 11 aggregator functions were computed per seed (not just `func_logit_power`), so
Ablation B can be rebuilt to the same 10-split standard without a second training
pass — not yet done. Real split-to-split std (0.001–0.017 depending on
dataset/variant) is consistently wider than the old analytic Hanley–McNeil SE
(~0.001–0.007), confirming the old SE was understating true uncertainty (expected —
it only captures within-split sampling noise, not split-to-split variance). Bitcoin-alpha
local attention shows one low outlier (seed 50, AUC 0.8718 across every aggregator
function) not yet root-caused — check `logs/multiseed/bitcoin-alpha_local_s50.train.log`
before citing this as a stable number.

**WSDM Table 1 baseline rows, same 2026-08-10 update:** added GCN, GAT, SGCN, GSGNN, SGA,
and **SiGAT** using numbers the user supplied from external publications (not reproduced
locally for GCN/GAT — no local run exists at all; SGCN/GSGNN are distinct from this
repo's curriculum-augmented CSG/CSG-GSGNN reproductions, not treated as equivalent per
explicit user call, "for now we dont include the curriculum"). **SiGAT correction
(2026-08-10, same day):** Table 1's SiGAT row initially showed our own single-split
canonical reproduction (0.882/0.876/0.915/0.893/0.883/0.859) instead of the published
numbers the user had already supplied in the same message as GCN/GAT/SGCN/GSGNN/SGA —
caught by the user ("why do you need SiGAT? i already gave the nums from published"),
fixed by switching the row to the published values (0.855/0.883/0.891/0.880/0.871/0.846,
±SE as given). **Our own local SiGAT reproduction is still used elsewhere and stays on
disk** — it's the per-edge-prediction source for the entropy-vs-AUC analyses (Empirical
Confirmation panels, Attention Directionality) in the WSDM paper, which need real
per-edge predictions a published aggregate AUC can't provide; it's just no longer what
Table 1 itself reports for SiGAT, matching the other published-only rows. CSG and
CSG-GSGNN **removed from the table** (not deleted from the repo — `baselines/CSG/
results_our_splits_canonical/` still has the single-split numbers) pending their own
10-split rerun, for consistency with the same standard now applied to PEWTER/GINEConv.
SNEA/CopulaLSP remain single-split for now; a 10-split campaign for both was launched the
same day (`scripts/run_multiseed_snea_copulalsp.py`, reuses `baselines/prepare_splits.py`'s
new `save_canonical_split_for_seed()` — same per-seed canonical-split fix built for
GINEConv's multiseed run). Both SNEA and CopulaLSP are undirected/pair-level models in
this codebase's implementation (`run_with_our_splits.py` builds `uni_edge_index` for
both), confirmed **not** a leakage/apples-to-apples problem: `prepare_splits.py`'s
`uni_tst_mask` only marks a canonical pair "clean test" if every real direction of it is
in the walk model's own test split, asserted in `_assert_canonical`.

**Sanity check against published numbers (2026-08-10):** the one clean same-model match
available, raw SiGAT (our canonical reproduction vs. the user's supplied published
SiGAT), came back within 0.7–3.7pp on all 6 datasets — no red flags. No such comparison
was attempted for SGCN/GSGNN vs. CSG/CSG-GSGNN (different models, curriculum added) or
for GCN/GAT (no local run exists).

**SiGAT 10-split campaign (launched 2026-08-10, DONE same day, 54/54 jobs, 0 failed).**
Motivation: the entropy-vs-AUC binned heatmaps (`scripts/paper_figures/
extract_empconf_panelC_gnn_entropy_heatmap.py`, `extract_attndir_panelD_pewter_entropy_
heatmap.py`) need real 10-split SiGAT predictions, not just a 10-split aggregate AUC, so
the SiGAT reproduction had to be re-run regardless of what Table 1 showed. Driver:
`scripts/run_multiseed_sigat.py` (54 new jobs: 6 datasets × 9 new seeds, seed 42 reuses
the existing canonical reproduction), same `save_canonical_split_for_seed()` infra as
SNEA/CopulaLSP, `baselines/SGA/run_with_our_splits.py` (env `sga_env`). **Table 1's SiGAT
row now shows our own 10-split mean±std** (same standard as PEWTER/GINEConv), replacing
the published number used earlier the same day: bitcoin-alpha 0.869±.016, bitcoin-otc
0.878±.008, epinions 0.909±.005, wiki-elec 0.888±.005, wiki-rfa 0.878±.004, slashdot090221
0.859±.005 (`baselines/SGA/results_our_splits_canonical/<ds>/SiGAT/seed{42,43..51}/
score.csv`, `tst_auc` column). Notably higher than the published numbers on 5/6 datasets
(epinions +1.8pp) — not flagged as a red flag, just a real split/hyperparameter difference
between our canonical-split reproduction and the original paper's own split.

**Entropy-heatmap multi-split methodology — decided 2026-08-10: Option 2 (mean±std of
per-split cell AUCs), not pooled test-edge predictions.** Two ways to combine the 10
SiGAT/PEWTER splits into one binned heatmap were discussed: (1) pool all 10 splits' test
predictions per cell then compute one AUC (uses ~10x the edges per cell, the only option
that can de-noise the naturally-thin high-entropy "hard node" corner cells, but mixes
predictions from 10 different trained model instances); (2) compute each split's cell AUC
independently on its own ~10% slice, then average — matches the mean±std convention
already used everywhere else in this paper (Table 1, GINEConv), but doesn't fix small-N
noise in rare cells (a cell below `MIN_CELL_N` in one split stays below it in every
split). **User picked (2) explicitly, "much simpler and reliable."** Confirmed: the fixed
4×4 entropy bins (`empconf_panelC`) are already bin-edges-on-entropy-value, not
percentile-based, and the per-node entropy values themselves (`src_ent`/`tgt_ent` in
`lead4_entropy_heterogeneity.py`) are computed once from the fixed real dense edge set,
independent of split — so averaging the per-cell AUC across the 10 splits is a
straightforward loop-and-average over the existing binning code, no re-derivation of bin
edges needed. **Both heatmaps built same day.** Extract:
`scripts/paper_figures/extract_multiseed_entropy_heatmaps.py` — pulls fresh per-seed
predictions for both models (SiGAT: `sigat_raw_seed()`, fits a fresh LogisticRegression
per seed on that seed's own `best_epoch_artifacts.pkl` + `splits_canonical[_seed{N}]`;
PEWTER local: `walk_raw_seed()`, mean-prob aggregation over each edge's walk occurrences
from `MULTISEED_s{seed}_local_*`'s `test_predictions.pkl`, mapped to raw (u,v) via that
seed's own keyed cache). **Bug caught and fixed before the numbers were trusted:** seed
42 predates the `MULTISEED_*` naming (it's the pre-existing, already-adopted
`E32_PY314_LOCALATTN4` checkpoint per `run_multiseed_pewter.py`'s own docstring, "seed 42
already trained, reused/backfilled") — the first pass silently found 0 matches for it and
every dataset's PEWTER average was only over 9/10 splits; fixed by special-casing seed 42
to pull from `scripts/attention_directionality.py`'s `LOCAL_RUN_INFO` pins instead of
globbing for a directory that doesn't exist. Output CSVs:
`aaai2027/figure_data/empconf_panelC_sigat_10split.csv`,
`aaai2027/figure_data/pewter_sigat_delta_heatmap.csv`. Plot:
`scripts/paper_figures/plot_multiseed_entropy_heatmaps.py` →
`aaai2027/figures/empconf_panelC_sigat_10split.png`,
`aaai2027/figures/pewter_sigat_delta_heatmap.png` (diverging `RdBu`, **blue = PEWTER
higher, red = SiGAT higher** — per the user's explicit color choice; the first render used
`RdBu_r` which put PEWTER-higher on red, caught and flipped). Also restyled to match the
original single-split Panel C exactly: `RdYlGn`/grey-`n/a` for the SiGAT-alone heatmap,
white gridlines, same text-color threshold logic — not a different look just because the
data source changed. **Finding:** PEWTER (local) beats SiGAT in nearly every cell on 5/6
datasets (bitcoin-otc's high-entropy corner is the single largest gap, +0.26 AUC);
bitcoin-alpha is the one dataset with a real mixed picture — PEWTER wins the low-entropy
corner (+0.16) but loses the (low-src, mid-tgt) region (-0.12), a genuine reversal worth a
closer look before citing as a clean "PEWTER wins everywhere" story. Kept as a standalone
pair of figures, **not yet placed in the paper** — an earlier attempt to insert them as two
new `\begin{figure}` blocks was explicitly reverted by the user ("i only asked you to make
the existing sigat heatmap to be updated with the 10 split ones so just fig1 update").

**Figure 1 (`fig:empconf-panels`, the 5-panel Empirical Confirmation figure) — Panel C
updated in place with the 10-split SiGAT data, 2026-08-10, no new figures, no tex changes.**
`scripts/paper_figures/extract_empconf_panelC_gnn_entropy_heatmap.py` was edited so SiGAT's
cells now come from the same 10-split mean computation as the standalone heatmap above
(reusing `sigat_raw_seed()`/`per_seed_grids()`); GINEConv's cells are untouched (still
single-split from `computed_data.pkl` — GINEConv was already dropped from this panel's
*plot* back on 2026-07-27, kept only in the CSV/appendix table, so the panel itself stays
SiGAT-only, 1 row, exactly as before). Rerun order: extract → `plot_empconf_panelC_gnn_
entropy_heatmap.py` (unchanged) → `combine_empconf_panels_abcde.py` (unchanged) —
`aaai2027/figures/empconf_panels_abcde_combined.png` is the only file that changed;
`fig:empconf-panels`'s tex reference was already correct and needed no edit.

**Readability pass, same day, across both combined figures (`empconf_panels_abcde_
combined.png` and `attndir_panels_abc_combined.png`):** several panel scripts had long
in-image titles that just repeated content already spelled out in the external LaTeX
caption (e.g. Panel D's old title: "SiGAT AUC by target-edge / neighbor-edge sign
agreement (pooled, 6 datasets; in = v's other in-edges, out = u's other out-edges)" — every
word of the parenthetical is already in `fig:empconf-panels`'s caption) — eating vertical
space that forced every other font in the panel to stay small once the panel got
compressed into the combined grid. Shortened titles and raised font sizes in
`plot_empconf_panelC_gnn_entropy_heatmap.py`, `plot_empconf_panelD_signagreement_auc.py`,
`plot_empconf_panelE_coefficients.py`, `plot_attndir_panelA_headgrid.py`,
`plot_attndir_panelB_direction.py`, `plot_attndir_panelC_nodeedge.py`. Panel C needed a
noticeably larger bump than the others (title 12→17pt, cell text 9.5→13pt) since it's a
6-column-wide panel that gets compressed harder than the narrower panels when the combine
script scales everything to the same figure width — same absolute font size reads smaller
there than in a 2-column panel. Panels A/B of the empconf figure were left alone (no
suptitle to begin with, already fine).

**New figure built the same day, independent of the above: SHAP edge directionality
(`scripts/shap_edge_directionality.py`, `scripts/paper_figures/{extract,plot}_shap_edge_
directionality.py`, `aaai2027/figures/shap_edge_directionality.png`).** Companion to the
Attention Directionality figure — measures actual causal contribution (exact Shapley, not
raw attention weight) of each context edge to a masked target edge's predicted
P(positive), on the LocalAttn4 checkpoint, restricted to the local attention window.
`local_attention_window=4` is a token-distance threshold and 1 hop = 2 token positions, so
the window covers **two hops each side** (4 context-edge features: fwd/bwd × hop1/hop2),
not one — exact Shapley over ≤16 subsets per instance, no sampling approximation needed.
Masking reuses the model's own `<MASK>` token (same one used for the target edge itself),
keeping every constructed input in-distribution. Efficiency-property check
(`sum(shap) == value(full) − value(empty)`) passed to float precision (~1e-16) on all 6
datasets — the computation is verified correct, not just plausible. **Finding: mean |SHAP|
decays from hop 1 to hop 2 on all 6/6 datasets, both directions, well outside cluster-robust
SEs** — a real, robust distance-decay result. Direction asymmetry splits into two groups:
bitcoin-alpha/bitcoin-otc/epinions show forward≈backward; slashdot090221/wiki-elec/wiki-rfa
show forward > backward at hop 1. Raw per-dataset results:
`outputs/shap_edge_directionality/shap_directionality_<ds>_result.pkl`; summary table:
`aaai2027/figure_data/shap_edge_directionality.csv`. Placement in the paper not yet
decided (candidate: near the existing Attention Directionality figure).

**Table below reflects the current, adopted `edge_cover` sampler** at each dataset's
production `num_walks` budget (see "Walk sampler" above) — confirmed reproducible via
a plain `dataset.name=<ds>` run, since `edge_cover` and its adopted budget are already
the default in every `configs/<dataset>.yaml`. (Corrected 2026-07-30: this section
previously and incorrectly claimed the table was still on the old `k_cover k=5`
checkpoints pending a separate retrain — that claim was stale leftover text from
before the "Rewritten 2026-07-19" note below the table, which already documented the
table as edge_cover-based; the retrain this contradicted was in fact already done the
same day. See the retrain-status note under "Walk sampler" above for the
re-verification.) Old uniform-E14 numbers in parentheses (had the ~85–88% coverage
caveat on the sparse graphs). The baseline column shows the **canonical-split** best GNN (re-run on the
unified walk-derived split, `baselines/all_results_canonical.csv`); pre-canonical best-GNN in
its own parentheses. **On the identical shared test edges the walk model beats EVERY GNN on all
6 datasets, both attention variants** (apples-to-apples; full matched table in
`outputs/walk_coverage_analysis/E15_SWEEP_RESULTS.md`, older canonical detail in
`CANONICAL_RERUN_FINDINGS.md` §1a). With full coverage the walk now also wins on each-model's-
own-full-test on all 6 — the prior wiki-elec/wiki-rfa "SiGAT marginally higher" exception was
purely a coverage artifact and is gone.

| Dataset         | Canon best-GNN (old)              | Full attn, no-H (E25/E26) | LocalAttn4, no-H (E32) — **current default** |
|-----------------|-----------------------------------|-----------------------------|------------------------------------------------|
| bitcoin-alpha   | 0.9051 SGA-GSGNN (0.8804)         | **0.9219**                  | 0.9188                                          |
| bitcoin-otc     | 0.8972 SNEA (0.9086)              | 0.9311                      | **0.9358**                                      |
| epinions        | 0.9146 SiGAT (0.9113)             | 0.9527                      | **0.9535**                                      |
| wiki-elec       | 0.8930 SiGAT (0.8840)             | 0.9036                      | **0.9075**                                      |
| wiki-rfa        | 0.8831 SiGAT (0.8673)             | 0.8930                      | **0.8976**                                      |
| slashdot090221  | 0.8587 SiGAT (0.8845)             | **0.9007**                  | 0.8981                                          |

**Corrected 2026-08-04/2026-08-05: `E31_PY314_MIGRATION` is full attention, not
LocalAttn4.** Root cause: `config.yaml`'s `model.local_attention_window` default was
`null`, not `4`, so the plain documented training command silently trained full
attention — confirmed by inspecting the checkpoint config directly
(`local_attention_window=None` on all 6) and by the AUCs matching the Full-attn
(E25/E26) column almost exactly. **Fixed at the root**: `config.yaml`'s default is now
`local_attention_window: 4` (full attention now requires an explicit
`model.local_attention_window=null`). `E31_PY314_MIGRATION` checkpoints are kept on
disk (`outputs/<ds>/E31_PY314_MIGRATION_*`) as an incidental extra full-attention data
point under the Python 3.14 stack — **not adopted into this table's Full-attn column**,
which stays on its original Python-3.9-stack (E25/E26) numbers; do not use E31 as
LocalAttn4 for anything.

**`E32_PY314_LOCALATTN4`** is the corrected, properly-configured post-migration
LocalAttn4 retrain (verified via direct checkpoint inspection to carry
`local_attention_window=4` on all 6 datasets) and is what the LocalAttn4 column above
reflects. Deltas vs. the old Python-3.9-stack E27 numbers: alpha +0.62pp, otc -0.32pp,
epinions +0.02pp, wiki-elec +0.14pp, wiki-rfa +0.17pp, slashdot090221 0.00pp -- all
within the established noise band; Local-vs-Full keeps the same 4/6-win split as
before (otc, epinions, wiki-elec, wiki-rfa win for Local; alpha, slashdot090221 win
for Full).

Entropy-hardness numbers (E28, no longer recommended -- see the H flag below) for
provenance: alpha 0.9184, otc 0.9284, epinions 0.9535, wiki-elec 0.9064, wiki-rfa
0.8966, slashdot 0.8976 -- flat-to-negative vs. no-H on 5/6.

Apples-to-apples (shared edges) best GNN is always lower still -- e.g. epinions GINE
0.8642 / SiGAT 0.9109, slashdot GINE 0.7869 / SiGAT 0.8571. SE-SGformer excluded from
"best GNN": its KNN discriminator emits hard labels, so its AUC is really balanced
accuracy (~0.57-0.73).

Experiment tags: `E25_BUDGET_SWEEP`/`E26_WIKI_SWEEP` (full attention, Python-3.9-stack
-- this table's canonical Full-attn column) and `E32_PY314_LOCALATTN4` (LocalAttn4,
Python-3.14-stack, canonical as of 2026-08-05), both on the `edge_cover` sampler at
each dataset's production `num_walks` (isolated keyed caches
`data/<ds>/dataset_cache__edge_cover_nw<nw>_mw80_seed42.pt`, environment-independent,
reused across the migration). Prior LocalAttn4 tag `E27_NOHARD_EDGECOVER_LOCALATTN4`
(Python-3.9-stack) and prior `k_cover`-sampler SOTA (E15/E14) kept for provenance only
-- `git log` this file or `HARDNESS_MINER_ROADMAP.md`'s history.

**Paper figures/tables rebuilt 2026-08-05** against the corrected checkpoints (full
attn = `E31_PY314_MIGRATION`, LocalAttn4 = `E32_PY314_LOCALATTN4`) -- Result 1 table,
Result 2 heatmap, Attention Directionality figure, Ablations A and B (formerly C). See
`aaai2027/PEWTER_ASSETS_CHECKLIST.md` rows #20-25 for scripts/data pointers; nothing
pending on this front.

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
Lead4c coefficients alone (test-set leakage, see the roadmap doc).

## Attention variant: full vs. local — SETTLED 2026-07-20, LocalAttn4 is the confirmed default

**LocalAttn4 (`model.local_attention_window=4`) is the production default and the paper's
primary vehicle for the "proximal" claim.** This was an open question through E30 (below);
it's now closed by a direct mechanism check, not just AUC or narrative preference.

**E30 pilot (2026-07-19)** tested the paper's own proposed cleaner ablation — literally
shortening `dataset.max_walk_length` instead of masking attention within a long walk — on
bitcoin-alpha + epinions, full attention, no hardness, `max_walk_length` ∈ {2, 4, 8, 16}:

| dataset | L=2 | L=4 | L=8 | L=16 | L=80 (reference, E25/E26) |
|---|---|---|---|---|---|
| bitcoin-alpha | 0.9112 | 0.9196 | 0.9200 | **0.9305** | 0.9219 |
| epinions | 0.9480 | 0.9515 | **0.9528** | 0.9491 | 0.9527 |

L=8/L=16 looked like a clean win for short-walk truncation over LocalAttn4 (matches or beats
L=80 at 1/10th the length). **But digging into *why* L=2 underperforms found the walk-length
axis is confounded at the short end, and the confound gets worse — not better — the closer
you push toward LocalAttn4's own ±2-hop span.**

**The confound:** dynamic masking selects this epoch's supervised targets *globally* per
edge_id (`_sample_epoch_targets`, `src/model/lit_model.py:237`), not per-walk — so a 2-edge
walk has a real chance none of its edges are selected this epoch (measured: 43.9%/45.3% of
L=2 walks on alpha/epinions contribute zero gradient). Raising `model.dynamic_target_ratio`
(an override added for this investigation, decoupled from `dataset.mask_ratio`/`train_ratio`
so it doesn't confound the actual data split) fixes that for free — same walks, same
compute, just a different fraction of the pool selected as targets each epoch — and L=2
bitcoin-alpha jumps from 0.9112 to **0.9251** (ratio=0.7), beating the L=80 reference.

**But that "fix" trades one problem for a worse one.** Measured directly against the real
cached walk data (`scripts/measure_local_context_availability.py`, replicating the exact
`_sample_epoch_targets` sampling): a 2-edge walk essentially never has room for both a
target *and* a labeled neighbor edge to condition on. At the untouched baseline (ratio=0.4),
a target edge already has zero labeled context 43.2% of the time on bitcoin-alpha (54.6% on
epinions) purely because there usually isn't a second pool edge in such a short walk —
raising the ratio to fix the empty-walk waste makes this *strictly worse* (67% zero-context
at ratio=0.7, 100% at ratio=1.0, since every pool edge becomes a target every epoch and pool
edges can then never appear as each other's context). **So an L=2-truncated walk cannot
cleanly demonstrate "the model conditions on nearby labeled edges" — the walk almost never
contains a nearby labeled edge to condition on, independent of any ratio tuning.** L=8/L=16
are NOT affected by this (only 0.5–2.1% of walks have zero context anywhere at those
lengths) — their sufficiency result stands, uncounfounded; it's specifically the short end
that breaks, which happens to be exactly the length range that would make the cleanest
paper story.

**The resolution: LocalAttn4 on a long walk gets the same ±2-hop locality restriction
without the starvation, because the underlying walk keeps wandering through the real graph
instead of being cut off.** Measured on the L=80 cache with the *default* ratio (no tuning
needed): a target's ±2-hop window (`local_attention_window=4`) contains real labeled context
90.6% of the time on bitcoin-alpha, 82.3% at L=8 on epinions — vastly better than literal
L=2 truncation's 56.8%/45.4%. Full table (`scripts/measure_local_context_availability.py`,
ratio=0.4 default throughout):

| L | bitcoin-alpha: local-window context available | bitcoin-alpha: zero context anywhere | epinions: local-window context | epinions: zero context anywhere |
|---|---|---|---|---|
| 2 | 56.8% | 43.2% | 45.4% | 54.6% |
| 4 | 78.2% | 15.5% | — | — |
| 8 | 84.8% | 2.1% | 82.3% | 5.9% |
| 16 | 88.0% | 0.5% | — | — |
| 80 | 90.6% | 0.1% | — | — |

**Verdict: full-attention-with-short-walks is not being pursued further as the production
alternative.** L=2 (the length that would actually match LocalAttn4's window for an
apples-to-apples comparison) is structurally unable to exercise the proximal-context
mechanism the paper claims, regardless of masking-rate tuning — its AUC "win" there is real
but is likely reflecting node-identity/structural signal, not proximal labeled context.
LocalAttn4 is the mechanistically honest way to get ±2-hop locality: same restriction, but
without conflating "restrict information" with "restrict the walk sample itself." **Not
extending E30 to the remaining 4 datasets** — the open question it was meant to resolve
(flip to full-attention+short-walks?) is answered. The paper's Ablation A should be anchored
on full-vs-local attention (already-run E25/E26 vs E27), optionally citing this
context-collapse-under-truncation finding as supporting mechanism evidence.

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
- Shared edge loader: scripts/balance_theory_paths.py → load_edges_canonical() — **verified
  2026-07-28: this is not a separate/divergent implementation**, it's a thin wrapper doing
  exactly `load_config(overrides=[f"dataset.name={ds}"]) ` + `get_loader(ds_name)(cfg)`, where
  `get_loader` is imported directly `from src.data.datasets import get_loader` — the same
  function `src/data/prepare_data.py` (the real training-data-cache builder) calls. So it does
  load the real, canonical, training-identical edges, just via an extra hop.
  **Known duplication (harmless today, worth cleaning up eventually):** `scripts/
  node_mi_structural_embedding.py` defines an independent, currently-identical copy of the
  same `DATASET_CONFIGS`/`load_edges_canonical()` — neither script is GNN-specific despite
  the historical naming confusion (both are walk-model-side Lead-investigation analysis
  scripts, not baseline loaders); the duplication is just organic accumulation, not a design
  choice. **Going forward: new scripts should import `get_loader`/`load_config` directly
  from `src/data/datasets.py`/`src/utils/config.py`** (the actual production modules) rather
  than reaching into either analysis script's copy — don't add a third copy. If it ever
  becomes actively annoying, consolidate into one small shared module (e.g.
  `scripts/canonical_data.py`) that both existing files import from, rather than duplicating
  again.
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
- Baselines: baselines/all_results.csv, baselines/<model>/results_our_splits/. **Convention
  (corrected 2026-08-04): unqualified "SiGAT" always means raw SiGAT, not SGA-augmented,
  everywhere in this codebase/paper unless explicitly marked otherwise** — an earlier version
  of `aaai2027/PEWTER_ASSETS_CHECKLIST.md` (#21/#29/#34) wrongly claimed the SiGAT number used
  in the entropy-vs-AUC analyses was SGA-augmented (`baselines/SGA/sigat_SGA.py`); confirmed
  with the user it's raw SiGAT throughout, docs fixed.
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

## Hardness map paths (historical — H is scrapped, not used anywhere as of 2026-07-19)

Kept for provenance only. Do not point any new run at these — see "Hardness reweighting
(H): scrapped everywhere" above.

`outputs/<dataset>/E22_HARDNODE_ENTROPY/hardness_{source,target}.pt` — role-aware entropy
maps, the best-quality map found (`HARDNESS_MINER_ROADMAP.md`), used in the final E28/E29
ablation that led to the scrap decision.
`outputs/<dataset>/E17_HARDNODE_KCOVER_REMINE/hardness_map.pt` — old learned-miner map,
re-mined on the (now superseded) `k_cover` sampler.
`outputs/transformer_incremental/bitcoin-alpha_.../artifacts/E14_HARDNODE_L10/hardness_map.pt`
— original pre-`k_cover` miner map (others: same structure under each dataset's run dir).

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

- Lead 5 — per-walk prediction variance / ensemble effect (cheapest untested candidate mechanism).
- Lead 6 — training-regime confound (D/R/H tricks) + capacity mismatch + missing trainable/spectral
  features in GNN baselines.

OPEN WORKSTREAMS — audited 2026-07-19 (see plan files in ~/.claude/plans/):

**Active / not started, real open work, roughly priority order:**
- `hello-so-i-have-unified-valiant.md` — **PEWTER paper (aaai2027/), ACTIVE, top priority,
  deadline-critical.** Abstract due 2026-07-28, full paper 2026-07-31 — as of this audit
  that's 9 days out. Per the plan's own 2026-07-16 status update, most of
  Results/Discussion/bib/Supplementary/final-QA tiers are NOT STARTED. Confirmed
  in-progress and real (2026-07-19) — take this as the top-priority workstream when
  triaging session time against everything else below.
- **LocalAttn4 H/no-H re-ablation — CLOSED 2026-07-19.** `HARDNESS_MINER_ROADMAP.md` items
  13/14 (`E27`/`E28`/`E29`, current sampler+masking, all 6 datasets + the asymmetric
  source/target weight probe on bitcoin-alpha/otc) all complete. Final verdict: **H
  scrapped everywhere** (see "Hardness reweighting (H): scrapped everywhere" above) — do
  not cite the old "LocalAttn4+H (E14)" SOTA-table column or the E16/E17 "H is
  load-bearing for LocalAttn4" claim as current guidance, both retracted.
- **Short-walk ablation (E30) — CLOSED 2026-07-20.** Tracked ad hoc in CLAUDE.md's
  "Attention variant: full vs. local" section above. Pilot (bitcoin-alpha + epinions,
  `max_walk_length` ∈ {2,4,8,16}) initially looked like a clean win for full-attention
  short walks, but a follow-up context-availability measurement found L=2 (the length
  that would actually match LocalAttn4's window) structurally starves target edges of
  any labeled neighbor to condition on, regardless of masking-rate tuning — so it can't
  cleanly demonstrate the paper's proximal-context claim even though its raw AUC looked
  good. LocalAttn4 confirmed as the settled default; full-attention+short-walks is not
  being pursued further as the production alternative. Not extending to the remaining 4
  datasets. Reusable measurement script:
  `scripts/measure_local_context_availability.py`.
- `plan-a-fix-for-glimmering-panda.md` ← **CLOSED 2026-07-19.** Walk sampler fix done —
  `edge_cover` adopted, budgets swept and signed off on all 6 datasets (see "Walk sampler"
  above), and LocalAttn4 viability confirmed via `E27` (all 6 datasets). Remaining item
  (full-attention `edge_cover` retrain against the SOTA table) tracked separately, not
  blocking this plan's closure.
- `plan-stats-rigor.md` ← multi-seed variance / cross-validation / significance testing.
  Restored 2026-07-19 (see its own recovery note) — now higher priority than when written,
  since the PEWTER paper needs defensible, variance-aware results, not single-run point
  estimates. No K-split infra exists yet; `optuna_run.py` is still suspected stale.
- `plan-side-quests-misc.md` ← docs/config/repo-hygiene/research-follow-up backlog. Restored
  2026-07-19 with a fresh relevance check against the live codebase (see its own recovery
  note) — 3 of 10 original items are already done (doc fix, git cleanup, SiGAT+SGA baseline),
  the rest (config-loading bug, dependency pinning, OWL dead-code removal, walk-length-sweep
  write-up, 2 unreconfirmed LightGBM anomalies) are still open. Lowest priority of this list,
  doesn't block anything else.
- Lead 5/6 subplans (`plan-lead5-ensemble-effect.md`, `plan-lead6-trainable-features.md`) ←
  ensemble effect + trainable-features/capacity/training-regime parity; neither started
  (no `outputs/lead5_ensemble/` or `outputs/lead6_trainable_features/` on disk).
## PEWTER paper (aaai2027/) — repo-to-paper phase, file map and conventions

**As of 2026-07-19 the project entered a second phase: turning this repo's findings into the
PEWTER AAAI-27 submission.** Most day-to-day session time now goes into `aaai2027/`, not new
modeling experiments — see "OPEN WORKSTREAMS" above (`hello-so-i-have-unified-valiant.md` is
top priority, deadline-critical: abstract 2026-07-28, full paper 2026-07-31). This section is
the fast-lookup index for that work; the full plan/status detail lives in the plan file, not
here.

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
- **Rule going forward: every `PEWTER_ASSETS_CHECKLIST.md` row that reaches DONE/NEEDS-FIGURE
  status must name its generating script(s) and output path(s)** (the `Scripts:`/`Data:`
  pattern already used in rows #21/#23/#25/#29/#30/#30b) — not just "done", so a future session
  (or this one, post-compaction) can jump straight to the code instead of rediscovering which
  script produced a given figure. Add the pointer in the same edit that changes the status.

**Figure 1 rework (2026-07-28) — DONE except Panel B's final presentation call.** The
Empirical Confirmation 3-panel figure (checklist #12, now `fig:empconf-panels`,
`figures/empconf_panels_abc_combined.png`): Panel C converted from the smoothed
Gaussian-kernel grid to discrete 4×4 entropy bins with per-cell AUC annotations and a
larger panel (`scripts/paper_figures/{extract,plot}_empconf_panelC_gnn_entropy_heatmap.py`).
Panel A (entropy asymmetry, previously text-only/deferred) is now a real boxplot
($\Hh_\outdeg$ vs.\ $\Hh_\indeg$ per dataset, mean-diamond markers since 4/6 datasets have
median+IQR collapsed to 0, paired $t$-test in the caption) —
`scripts/paper_figures/{extract,plot}_empconf_panelA_entropy_boxplot.py`, reusing the same
per-node entropy arrays as `scripts/lead4c_directionality_answers.py::claim1_for_dataset`.
Combine script: `scripts/paper_figures/combine_empconf_panels_abc.py` (supersedes the old
2-panel `combine_empconf_panels_bc.py`).

**Panel B's post-minimum "bump" (distance 4–6) — investigated in full 2026-07-28, verdict:
real (not a bug, not pure noise), mechanism still being pinned down.** Checked and ruled out: BFS-correctness
(shell values matched `networkx` ground truth exactly, 0/30 mismatches; per-anchor edge
counts matched an independent brute-force enumeration exactly, 0/8 mismatches) and
numerical instability (integer counts throughout, no overflow, guarded `log2`). Confirmed
via a `--shuffle-signs` null control (already implemented on the extractor) that the bump
survives null-subtraction by 2–3 orders of magnitude on most datasets at distance 5–6 (not
pure estimator noise), but traced its cause to two compounding, measured mechanisms: (1) a
degree confound — nodes reached only at the outer BFS shells have collapsed degree (median
degree 44.5→2 across shells on bitcoin-alpha; exactly 1 by shell 3–4 on wiki-elec), hence
mechanically low entropy; (2) heavy edge reuse at the tail — the top 10 distinct context
edges account for ~40% of all (anchor, context) pairs at the farthest distance on
bitcoin-alpha vs. 0.2% at distance 1, so the naive per-pair-independent contingency table
is overconfident there. Extending `d_max` to 9 (real + null, all 6 datasets) showed the
effect does **not** keep growing — bitcoin-alpha/otc's graphs are essentially exhausted by
distance 7 (n_pairs collapses to ~5–7K and real/null become indistinguishable or reverse),
and wiki-elec/wiki-rfa hit **zero** remaining pairs by distance 6–7 (their graphs are simply
that small) — ruling out "it's a truncation artifact that would keep climbing if we looked
further."

**Degree-filtering was then tried as the candidate fix and FAILED** — a pilot on
bitcoin-alpha (5,000-anchor sample, requiring both context-edge endpoints to have degree
>=5) left distance-6 NMI essentially unchanged (0.000534 filtered vs. 0.000472 unfiltered,
same order of magnitude). So despite degree genuinely collapsing at the outer shells,
removing low-degree nodes does not kill the bump — degree is a correlate, not the
operative mechanism. **The more likely operative mechanism, per the edge-reuse
measurement, is non-independence/clustering of the (anchor, context) pairs**: a small
number of distinct context edges get counted many times over (top 10 distinct edges =
~40% of pairs at the tail vs. 0.2% at distance 1), violating the naive contingency table's
implicit "each pair is independent" assumption exactly where the bump appears — the
effective sample size at the tail is far smaller than the nominal n_pairs. **Recommended
next step (not yet implemented):** deduplicate by distinct context edge before forming the
contingency table (weight each distinct edge once, not once per anchor that reaches it), or
run a cluster-aware significance test (bootstrap over distinct edges, not raw pairs) instead
of trusting the naive point estimate at the tail bins. This supersedes the "degree confound"
framing as the leading candidate mechanism (degree collapse is real but not sufficient by
itself). The current `pewter_aaai.tex` Panel B paragraph's inline `%%` comment still
describes the superseded degree-confound framing and needs a follow-up edit once the
clustering fix is implemented or a final presentation decision is made.

## Session management tips

- Start a new chat for each independent workstream.
- In new chat: CLAUDE.md loads automatically. Say "read ~/.claude/plans/plan-<X>.md and do step N".
- Use /compact when finishing a sub-task within a session.
- Update this CLAUDE.md after completing a workstream (update status table above).
- Never paste long files — reference by path.
