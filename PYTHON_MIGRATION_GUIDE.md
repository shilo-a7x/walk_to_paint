# Python/stack migration — self-serve guide

How to reproduce the Python 3.9 → 3.14 + numpy/scipy/pandas/scikit-learn/lightning/torch
migration from scratch, or redo it later (e.g. for the next round of version bumps). Written
as exact terminal commands, no memorization required. Everything here is additive/isolated —
none of it touches the existing `.venv/`.

## Result of this pass (2026-08-02)

| Component | Old (`.venv`) | New (`.venv314`) |
|---|---|---|
| Python | 3.9.25 | 3.14.6 |
| torch | 2.7.1 (cu126 default) | 2.13.0 (`+cu126`, explicitly pinned) |
| numpy | 2.0.2 | 2.5.1 |
| scipy | 1.13.1 | 1.18.0 |
| pandas | 2.3.1 | 3.0.5 |
| scikit-learn | 1.6.1 | 1.9.0 |
| lightning / pytorch-lightning | 2.5.3 / 2.5.2 | 2.6.5 / 2.6.5 |

Validated: all 6 datasets, full-length training + `func_logit_power` posthoc, deltas −0.34pp to
+0.89pp vs. the CLAUDE.md reference table — within the established same-environment noise band.
One code fix was required and applied (see "Known gotcha" below) before training would run at
all under the new interpreter.

## 1. Install `uv` (no sudo required)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# lands in ~/.local/bin — confirm it's on PATH
uv --version
```

`uv` does two jobs here: fetching a specific Python interpreter version (prebuilt binary, no
compilation — this is what avoids the "no readline module" failure a from-source/pyenv build
can hit with no sudo to install `libreadline-dev`), and installing packages
(`uv pip install`/`uv pip freeze`, a drop-in fast replacement for `pip`). This does **not**
convert the repo into a `uv`-native project — no `pyproject.toml`/`uv.lock`/`uv run`. Every
existing `.venv<N>/bin/python run.py ...` invocation keeps working exactly as documented in
CLAUDE.md, just pointed at whichever venv path you choose.

## 2. Fetch the interpreter

```bash
uv python install 3.14
uv python list | grep 3.14   # confirm the exact patch version resolved
```

Sanity-check stdlib modules that a from-source build can silently miss without sudo:

```bash
uv run -p 3.14 python -c "import readline, ssl, sqlite3, zlib, lzma, bz2, ctypes; print('stdlib ok')"
```

## 3. Create an isolated venv (repo root, sibling to `.venv/`)

```bash
cd /home/eng/shilo_avital/yolo_lab/walk_to_paint
uv venv --python 3.14 .venv314
```

Never overwrites or touches `.venv/`. Rollback at any point is `rm -rf .venv314`.

## 4. Install torch with an explicit CUDA-version-pinned index

**Do not `uv pip install torch` bare** — recent torch releases default their wheel index to
whatever the newest bundled CUDA runtime is (drifts upward over time, e.g. CUDA 13.0 as of
torch ~2.11+), which can exceed what an older NVIDIA driver supports. Check your driver's
ceiling first:

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader   # this box: 575.57.08
# nvidia-smi's "CUDA Version: X" line (top-right of the normal nvidia-smi output) is the
# driver's MAXIMUM SUPPORTED CUDA runtime, not a separately-installed toolkit.
# CUDA 13.0 requires driver >=580.65.06. If your driver is below that, pin to a cu12x index.
```

Then install torch against an explicit CUDA-12.x index (confirm the exact torch version you
want still publishes wheels there — `--dry-run` costs nothing):

```bash
uv pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126 \
    --python .venv314/bin/python --dry-run   # remove --dry-run once it looks right
```

## 5. Install the rest of the stack

Write (or reuse) a requirements file — this repo's is `requirements-target-py314.txt` — with
everything except torch (already installed in step 4):

```
numpy==2.5.1
scipy==1.18.0
pandas==3.0.5
scikit-learn==1.9.0
lightning>=2.6,<2.7
pytorch-lightning>=2.6,<2.7
omegaconf
tensorboard
matplotlib
tqdm
optuna
optuna-integration
seaborn
statsmodels
xgboost
lightgbm
tbparse
plotly
kaleido
pytest
pytest-timeout
networkx
PyYAML
simplejson
```

```bash
uv pip install -r requirements-target-py314.txt --python .venv314/bin/python
```

Freeze a lockfile for later diffing/reproducibility:

```bash
uv pip freeze --python .venv314/bin/python > requirements-frozen-py314-$(date +%Y%m%d).txt
```

(`uv venv` does not install `pip` into the venv itself — always use
`uv pip freeze --python <venv>/bin/python`, not `<venv>/bin/python -m pip freeze`.)

## 6. Smoke checks before touching real data

```bash
# GPU + core libs
.venv314/bin/python -c "
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device='cuda') @ torch.randn(1024, 1024, device='cuda')
print('matmul ok', x.shape)
import numpy, scipy, pandas, sklearn
print(numpy.__version__, scipy.__version__, pandas.__version__, sklearn.__version__)
"

# structural: config validates for every dataset
for ds in bitcoin-alpha bitcoin-otc epinions wiki-elec wiki-rfa slashdot090221; do
  .venv314/bin/python run.py --device 0 --dry-run-config dataset.name=$ds
done

# full import smoke check
.venv314/bin/python -c "
import importlib, pkgutil
import src.data, src.model, src.training, src.utils
for pkg in [src.data, src.model, src.training, src.utils]:
    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + '.'):
        importlib.import_module(modname)
print('ALL IMPORTS OK')
"
```

## 7. Known gotcha: Python 3.14 changed the Linux multiprocessing default

**Symptom:** any `DataLoader(..., num_workers>0, collate_fn=<closure>)` crashes with a
`PicklingError`/`AttributeError` about not being able to find a local function, the moment
training actually starts iterating.

**Cause:** Python 3.14 changed the default `multiprocessing` start method on Linux from `fork`
to `forkserver` (confirmed via `multiprocessing.get_context().get_start_method()` — returns
`'fork'` under the old interpreter, `'forkserver'` under 3.14). `fork` copies the parent
process's memory directly, so a worker inherits closures for free. `forkserver`/`spawn` instead
**pickle** whatever's passed to the worker, and pickling a function only records its
module-qualified name for later re-import — impossible for a function that only exists as a
local/nested object (e.g. a closure returned by another function), since there's no importable
name pointing at it.

This repo's `ragged_collate_fn` (`src/data/stage_dataset.py`) used to return exactly such a
closure. **Already fixed** (2026-08-02) — it now returns an instance of a module-level
`_RaggedCollate` class instead, which pickles fine under any start method. If you're
reproducing this migration against an older commit that predates the fix, apply the same
pattern to any closure passed as `collate_fn`/`Pool` target/`Process` target: replace it with a
module-level class holding the closed-over values as `__init__` attributes and a `__call__`.

Everywhere else in the repo, multiprocessing already uses either module-level worker functions
(`functools.partial(_module_level_fn, ...)` — picklable regardless of start method) or
explicitly requests `mp.get_context("fork").Pool(...)` (immune to the default changing since it
never relies on the default). Neither of those needed any change.

## 8. Validate against production numbers

Uses each dataset's already-built `edge_cover` production cache — no new walk sampling needed,
training loads straight from the existing `.pt` cache file. This is safe against accidentally
overwriting a cache with different content: `_keyed_cache_path()`
(`src/data/prepare_data.py`) encodes only `walk_strategy`/`num_walks`/`max_walk_length`/`seed`
in the filename (no environment fingerprint), and the load path returns before ever reaching
the save path once a matching file is found.

```bash
# train (repeat per dataset, spread across free GPUs — check with nvidia-smi first)
.venv314/bin/python run.py --device <N> dataset.name=<ds> training.exp_name=<tag>

# posthoc — this is what CLAUDE.md's SOTA table actually reports (NOT the raw run.py test AUC)
.venv314/bin/python run_posthoc.py \
    --exp-dir outputs/<ds>/<tag>_<timestamp>/ \
    --artifacts predictions,aggregator \
    --agg-models func_logit_power \
    --device <N> --run-id <tag>
```

Compare the printed `Test AUC` against CLAUDE.md's "Current SOTA" table for the same dataset.
Expect small, mixed-sign drift (this pass saw −0.34pp to +0.89pp across all 6 datasets) — the
`Trainer` isn't built with `deterministic=True` and uses TF32 (`float32_precision: "medium"`),
so exact bit-reproducibility was never guaranteed even before any migration. A same-environment
back-to-back rerun of one dataset gives you a noise-floor number to compare against if a delta
looks larger than expected.

## 9. Adopting the new environment (separate, explicit step — not implied by validation passing)

Only do this once you've decided to actually switch:

```bash
# Option A: keep both paths, just start using .venv314/bin/python in commands
# Option B: make it the canonical `.venv`
mv .venv .venv-py39-archive
mv .venv314 .venv
```

If you do Option B, update CLAUDE.md's command examples (`.venv/bin/python ...` — already
correct if you rename) and note the Python 3.14 multiprocessing default change in the "Key
commands" section so nobody reintroduces a closure-based `collate_fn`/`Pool` target later
without realizing it'll break.

## 10. Rollback

At any point before step 9: `rm -rf .venv314` and the new requirements/lockfile artifacts.
`.venv/` was never modified, so there is nothing to undo there.
