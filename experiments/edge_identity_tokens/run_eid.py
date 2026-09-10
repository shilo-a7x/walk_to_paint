"""Edge-identity-token experiment launcher -- mirrors run.py's structure/flow,
using the REAL production config system (load_config/validate_config,
configs/<dataset>.yaml overlay, CLI dotlist overrides) and the real PL Trainer
via eid_src/training/train.py, instead of a hand-rolled training loop.

Two real production functions are called directly, unmodified: `prepare_data`
(the actual walk-sampling/tokenizer/cache-building pipeline -- ensures the EID
cache is built from a genuinely production-identical base cache, see
build_cache.py's docstring) only if the EID cache doesn't exist yet, and then
`prepare_eid_data`/`train_eid_model` (this experiment's own thin loader + Trainer
wrapper, see eid_src/ for what's copied/subclassed vs. reused unmodified).

Usage:
  .venv/bin/python experiments/edge_identity_tokens/run_eid.py \
      dataset.name=bitcoin-alpha training.exp_name=EID_PILOT --device 3 \
      [model.edge_replace_prob=0.2]
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
from pytorch_lightning import seed_everything

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from src.utils.config import load_config, get_seed, validate_config
from src.utils.paths import resolve_outputs_dirs

from experiments.edge_identity_tokens.eid_src.data.prepare_eid_data import prepare_eid_data
from experiments.edge_identity_tokens.eid_src.training.train import train_eid_model

EID_CACHE_PATH = "experiments/edge_identity_tokens/cache/{dataset}_nw{num_walks}_eid.pt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=int, default=0)
    # nargs="*" (not REMAINDER): REMAINDER swallows every token after the first
    # positional-looking one, including a later --device flag -- confirmed to
    # silently strand args.device at its default (0) whenever --device is passed
    # after the dotlist overrides (as run_eid_queue.py's command construction
    # does). "*" lets argparse interleave --device/--config with overrides in
    # either order, since dotlist entries never start with "-" here.
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def ensure_eid_cache(cfg, eid_cache_path):
    if os.path.isfile(eid_cache_path):
        print(f"EID cache already exists at {eid_cache_path}, reusing it.")
        return

    # Build (or load, if it already exists on disk) the real, unmodified production
    # cache first -- same walk sampler, same edge_cover budget, same tokenizer as
    # every other run of this dataset -- then post-process it into the EID format.
    # This guarantees the walks/splits underneath this experiment are byte-identical
    # to production's, never a separately-sampled or hand-rolled substitute.
    from src.data.prepare_data import prepare_data as prod_prepare_data, _keyed_cache_path

    base_cache_path = _keyed_cache_path(cfg)
    if not os.path.isfile(base_cache_path):
        print(f"Base production cache not found at {base_cache_path} -- building it "
              "via the real src.data.prepare_data.prepare_data pipeline...")
        prod_prepare_data(cfg)  # builds+saves the base cache as a side effect
    else:
        print(f"Reusing existing base production cache at {base_cache_path}.")

    from experiments.edge_identity_tokens.build_cache import build as build_eid_cache
    print(f"Post-processing into EID cache: {base_cache_path} -> {eid_cache_path}")
    build_eid_cache(base_cache_path, eid_cache_path)


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)

    try:
        validate_config(cfg, context="train")
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # dynamic_train_masking is now supported (sign-only-hide override, see
    # eid_src/model/lit_model.py's module docstring) -- production's config.yaml
    # default (True, the "D" flag) now flows through unmodified. scramble_edge_signs
    # is ALSO now supported (EIDLitEdgeClassifier._maybe_apply_eid_sign_scramble) --
    # this used to force it off here, silently invalidating every scramble_edge_signs
    # run (confirmed: EID_ABL_SCRAMBLESIGN_300000 trained and evaluated with the flag
    # force-reset to False the whole time). Removed 2026-09-10.

    try:
        current_exp = getattr(cfg.training, "exp_name", None)
    except Exception:
        current_exp = None
    if not current_exp or current_exp in ("walk_to_paint_experiment", "experiment"):
        cfg.training.exp_name = f"{getattr(cfg.dataset, 'name', 'dataset')}-EID_PILOT"
        print(f"Auto-generated exp_name: {cfg.training.exp_name}")

    try:
        seed = get_seed(cfg)
        seed_everything(seed, workers=True)
        random.seed(seed)
        np.random.seed(seed)
        print(f"Reproducibility enabled: seed={seed}")
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    resolved = resolve_outputs_dirs(cfg)
    print(f"Outputs -> exp_dir: {resolved['exp_dir']}")

    try:
        if cfg.training.use_cuda and torch.cuda.is_available():
            precision = getattr(cfg, "float32_precision", "medium")
            torch.set_float32_matmul_precision(precision)
    except Exception:
        pass

    if cfg.training.use_cuda and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        print(f"Using CUDA device {args.device}")
    else:
        print("Using CPU")

    eid_cache_path = EID_CACHE_PATH.format(dataset=cfg.dataset.name, num_walks=int(cfg.dataset.num_walks))
    ensure_eid_cache(cfg, eid_cache_path)

    data_module = prepare_eid_data(cfg, eid_cache_path)
    train_eid_model(cfg, data_module)


if __name__ == "__main__":
    main()
