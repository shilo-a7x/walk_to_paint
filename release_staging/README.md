# Random-Walk Transformer for Signed Edge Prediction

Edge sign prediction in directed signed graphs via a random-walk Transformer.

Graphs are converted into token sequences (alternating node/edge tokens) by sampling
random walks over the signed graph; a Transformer encoder is then trained with a
masked-edge-sign objective (MLM-style: some edges in each walk are hidden and the
model predicts their sign from surrounding context).

Evaluated on 6 signed-graph benchmarks: bitcoin-alpha, bitcoin-otc, epinions,
wiki-elec, wiki-rfa, slashdot090221.

## Install

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.
`pyproject.toml` pins torch to the CUDA 12.6 wheel index, so a single `uv sync`
installs everything, including a CUDA-enabled torch:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if not already installed

uv sync
```

If your GPU driver doesn't support CUDA 12.6, check its ceiling first:

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
# nvidia-smi's "CUDA Version: X" line (top-right of the normal output) is the
# driver's maximum supported CUDA runtime.
```

...then edit the `cu126` in both `[tool.uv.sources]` and `[[tool.uv.index]]` at the
bottom of `pyproject.toml` to a version at or below that ceiling, and re-run `uv sync`.

## Reproducing the headline results

Each of the 6 datasets ships as a small raw edge-list file under `data/<dataset>/`.
Training automatically builds and caches the walk-sampled dataset the first time it
runs (`src/data/prepare_data.py`) -- no separate data-preparation step is needed.

Single run, one dataset:

```bash
uv run run.py --device <gpu-id> dataset.name=<dataset> training.exp_name=<tag>
```

`<dataset>` is one of: `bitcoin-alpha`, `bitcoin-otc`, `epinions`, `wiki-elec`,
`wiki-rfa`, `slashdot090221`. `--device` pins the CUDA device (required -- see the
gotcha below). This writes checkpoints and logs to
`outputs/<dataset>/<tag>_<timestamp>/` (the timestamp is appended automatically,
so check `outputs/<dataset>/` after training to get the exact directory name).

Post-hoc evaluation (predictions + aggregation across a walk's multiple occurrences
of the same edge):

```bash
uv run run_posthoc.py --exp-dir outputs/<dataset>/<tag>_<timestamp>/ \
    --artifacts predictions,aggregator --agg-models func_logit_power \
    --device <gpu-id> --run-id <tag>
```

A 10-seed campaign (mean +/- std AUC across seeds) is driven by
`scripts/run_multiseed.py` (reads/writes under `outputs/` and `logs/multiseed/`;
edit `GPUS` at the top of the script for your machine).

**Gotcha:** `run.py` always sets `CUDA_VISIBLE_DEVICES` from `--device` (default 0
if omitted) -- always pass `--device <N>` explicitly on multi-GPU machines, a bare
shell-level `CUDA_VISIBLE_DEVICES=<N>` export is not sufficient on its own.

## Model configuration

- `config.yaml` -- base config; `configs/<dataset>.yaml` -- per-dataset overrides,
  merged automatically via `dataset.name=<dataset>`.
- `model.local_attention_window` -- `4` (default) restricts self-attention to a
  +/-2-hop window around each token; `null` gives full attention.
- Walk sampling uses `edge_cover`: one forced anchor walk per edge (guaranteeing
  100% edge coverage), with any remaining walk budget filled by genuinely distinct
  walks (a dedup-retry phase that raises rather than silently padding with
  duplicates). See `src/data/walk_sampler.py::edge_cover_walks`'s docstring for
  the full mechanism.

## Repository structure

```text
run.py                  # training entrypoint
run_posthoc.py          # post-hoc prediction + aggregation
config.yaml             # base config
configs/                # per-dataset config overrides
pyproject.toml, uv.lock # dependencies
src/
  data/                 # data loading, walk sampling, tokenization, dataset caching
  model/                # Transformer model + LightningModule
  training/             # training callbacks
  utils/                # config loading, misc utilities
data/<dataset>/         # ships one raw edge-list file per dataset; training adds
                        # a dataset_cache__*.pt file here the first time it runs
scripts/
  run_multiseed.py      # 10-seed campaign driver
outputs/                # created by run.py: checkpoints, logs, posthoc results
logs/                   # created by scripts/run_multiseed.py
```

## Canonical splits

train:val:test = 0.8:0.1:0.1, nested 4-way (train/mask/val/test) for the walk model's
masked-edge objective.
