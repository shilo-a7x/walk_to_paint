# E15 final SOTA budgets — k_cover k=5, all 6 datasets

Machine-readable version: [`E15_final_sota_budgets.csv`](E15_final_sota_budgets.csv)
(one row per dataset × attention-variant). Full sweep history/derivation:
[`E15_SWEEP_RESULTS.md`](E15_SWEEP_RESULTS.md) (34-cell grid, all intermediate findings).

## Final decision

All 6 datasets adopt `walk_strategy=k_cover`, `walk_k_min=5` (saturation floor locked
in Phase 0.F of `~/.claude/plans/plan-a-fix-for-glimmering-panda.md`). This supersedes
the interim "keep bitcoin-alpha/otc on uniform" note that appears partway through
`E15_SWEEP_RESULTS.md` — the FINAL 34-cell sweep (bottom of that file) showed k_cover
ties-or-beats uniform on bitcoin-alpha/otc too once budget is adequate, so a single
sampler strategy across all 6 is simpler and still 100%-coverage everywhere.

`max_walk_length=80` (hops) unchanged on all 6, matching the pre-E15 SOTA.

## Per-dataset, per-attention-variant budgets (test AUC, func_logit_power)

| dataset        | attn  | num_walks | test AUC (flp) | coverage |
|----------------|-------|-----------|----------------|----------|
| bitcoin-alpha   | full  | 5,000,000 | 0.9251 | 100% |
| bitcoin-alpha   | local | 5,000,000 | 0.9362 | 100% |
| bitcoin-otc     | full  | 2,000,000 | 0.9427 | 100% |
| bitcoin-otc     | local | 1,000,000 | 0.9410 | 100% |
| epinions        | full  | 3,000,000 | 0.9562 | 100% |
| epinions        | local | 2,000,000 | 0.9568 | 100% |
| wiki-elec       | full  |   500,000 | 0.9016 | 100% |
| wiki-elec       | local |   500,000 | 0.9038 | 100% |
| wiki-rfa        | full  | 1,000,000 | 0.8932 | 100% |
| wiki-rfa        | local | 1,000,000 | 0.8916 | 100% |
| slashdot090221  | full  | 5,000,000 | 0.9012 | 100% |
| slashdot090221  | local | 3,000,000 | 0.8984 | 100% |

"full" = `model.local_attention_window=null` (default, "Ours" in CLAUDE.md's SOTA table).
"local" = `model.local_attention_window=4` (LocalAttn4 experiment).

Budget shape notes (from the sweep): epinions/otc plateau by 1-2M; bitcoin-alpha and
slashdot were still climbing at 5M (haven't confirmed the ceiling); wiki-elec/wiki-rfa
AUC **drops** past their minimum covering budget (over-saturation on small, dense
graphs) — do not increase those two past the listed budget without re-validating.

## How to reproduce a run

Dataset configs (`configs/<ds>.yaml`) now default to the **full-attention** budget
and `walk_strategy: k_cover` / `walk_k_min: 5` (see edits below). To reproduce:

**Corrected commands (2026-07-06)** — the previous version of this snippet had
three bugs, confirmed live while launching the E16 no-hardness ablation (see
`~/.claude/plans/plan-hardness-miner.md`): (1) it used the bare system `python`,
which has no `torch` installed — use `.venv/bin/python`; (2) it relied on shell
`CUDA_VISIBLE_DEVICES=<N>`, but `run.py` unconditionally overwrites that env var
from its `--device` flag (default 0) — omitting `--device` silently piles every
parallel run onto physical GPU 0, so pass `--device <N>` explicitly and treat the
`CUDA_VISIBLE_DEVICES=` prefix as redundant; (3) `exp_name=<tag>` is a silent
no-op (wrong key — the config path is `training.exp_name`), so runs launched with
the shorthand auto-generate `<dataset>-run_<timestamp>` instead of the intended
tag. **Also note:** the actual winner runs in the table above all had
`model.hardness_lambda=1.0` and `model.hardness_map_path=<E14 hardness
map>` set (confirmed via their saved `hparams.yaml`) — omitted below for brevity,
but the with-hardness result in this doc is not a hardness-free baseline.

```
# full attention (Ours) — uses config.yaml defaults, no override needed beyond dataset.name
.venv/bin/python run.py --device <N> dataset.name=<ds> training.exp_name=<tag>

# LocalAttn4 — override attention window AND num_walks where they differ from the config default
.venv/bin/python run.py --device <N> dataset.name=<ds> \
    model.local_attention_window=4 dataset.num_walks=<local_nw_from_table> \
    training.exp_name=<tag>_LOCALATTN4
```

Caches are keyed by strategy/budget (`data/<ds>/dataset_cache__k_cover_k5_nw<nw>_mw80_seed42.pt`),
so switching `num_walks` for the local-attention run builds/reads its own cache file —
it will not clobber the full-attention cache or overwrite `data/<ds>/dataset_cache.pt`
(the old uniform-sampler SOTA cache, kept byte-unchanged per Phase 1 of the coverage plan).

## Winner run dirs / caches (for per-edge prediction extraction, Lead reruns, etc.)

| dataset        | attn  | run dir | cache file |
|----------------|-------|---------|------------|
| bitcoin-alpha   | full  | `outputs/bitcoin-alpha/E15_SWEEP_k5_nw5000000_full` | `data/bitcoin-alpha/dataset_cache__k_cover_k5_nw5000000_mw80_seed42.pt` |
| bitcoin-alpha   | local | `outputs/bitcoin-alpha/E15_SWEEP_k5_nw5000000_local` | same cache as full (same nw) |
| bitcoin-otc     | full  | `outputs/bitcoin-otc/E15_COVERAGE_KCOVER_K5_NW2000000_20260628-173849` | `data/bitcoin-otc/dataset_cache__k_cover_k5_nw2000000_mw80_seed42.pt` |
| bitcoin-otc     | local | `outputs/bitcoin-otc/E15_SWEEP_k5_nw1000000_local` | `data/bitcoin-otc/dataset_cache__k_cover_k5_nw1000000_mw80_seed42.pt` |
| epinions        | full  | `outputs/epinions/E15_SWEEP_k5_nw3000000_full` | `data/epinions/dataset_cache__k_cover_k5_nw3000000_mw80_seed42.pt` |
| epinions        | local | `outputs/epinions/E15_SWEEP_k5_nw2000000_local` | `data/epinions/dataset_cache__k_cover_k5_nw2000000_mw80_seed42.pt` |
| wiki-elec       | full  | `outputs/wiki-elec/E15_COVERAGE_KCOVER_K5_20260628-153111` | `data/wiki-Elec/dataset_cache__k_cover_k5_nw500000_mw80_seed42.pt` |
| wiki-elec       | local | `outputs/wiki-elec/E15_SWEEP_k5_nw500000_local` | same cache as full (same nw) |
| wiki-rfa        | full  | `outputs/wiki-rfa/E15_COVERAGE_KCOVER_K5_20260628-153111` | `data/wiki-RfA/dataset_cache__k_cover_k5_nw1000000_mw80_seed42.pt` |
| wiki-rfa        | local | `outputs/wiki-rfa/E15_SWEEP_k5_nw1000000_local` | same cache as full (same nw) |
| slashdot090221  | full  | `outputs/slashdot090221/E15_COVERAGE_KCOVER_K5_20260628-160751` | `data/slashdot090221/dataset_cache__k_cover_k5_nw5000000_mw80_seed42.pt` |
| slashdot090221  | local | `outputs/slashdot090221/E15_SWEEP_k5_nw3000000_local` | `data/slashdot090221/dataset_cache__k_cover_k5_nw3000000_mw80_seed42.pt` |

Prediction files: `<run_dir>/checkpoints/<ds>_predictions/epoch_*/test_predictions.pkl`
(all 12 dirs above confirmed to exist on disk 2026-07-06).

Note directory-name casing: the wiki datasets' data dirs are `data/wiki-Elec` and
`data/wiki-RfA` (capitalized), not `wiki-elec`/`wiki-rfa` — `dataset.name` is lowercase
(`wiki-elec`, `wiki-rfa`) but `dataset.data_dir` keeps the original casing.
