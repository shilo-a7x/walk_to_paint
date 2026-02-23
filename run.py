# run.py
import os
import sys
import random
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from src.data.prepare_data import prepare_data
from src.training.train import train_model
from src.utils.paths import resolve_outputs_dirs
from src.utils.config import load_config, get_seed, validate_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device id (default: 0)"
    )
    parser.add_argument(
        "--dry-run-config",
        action="store_true",
        help="Load + validate config then exit without data prep or training",
    )
    parser.add_argument(
        "overrides", nargs=argparse.REMAINDER, help="Override config values"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load and merge config (supports `configs/<dataset>.yaml` and CLI dotlist overrides)
    cfg = load_config(args.config, overrides=args.overrides)

    try:
        validate_config(cfg, context="train")
    except ValueError as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)

    if args.dry_run_config:
        print("✅ Config validation passed (context=train)")
        return

    # Auto-generate exp_name when placeholder or absent
    try:
        current_exp = getattr(cfg.training, "exp_name", None)
    except Exception:
        current_exp = None
    if not current_exp or current_exp in ("walk_to_paint_experiment", "experiment"):
        try:
            note = getattr(cfg.training, "exp_note", None)
        except Exception:
            note = None
        note_part = f"-{note}" if note else ""
        cfg.training.exp_name = (
            f"{getattr(cfg.dataset, 'name', 'dataset')}-run{note_part}"
        )
        print(f"Auto-generated exp_name: {cfg.training.exp_name}")

    # Set global seeds for reproducibility using canonical config.reproducibility.seed
    try:
        seed = get_seed(cfg)
        seed_everything(seed, workers=True)
        random.seed(seed)
        np.random.seed(seed)
        print(f"✅ Reproducibility enabled: seed={seed}")
    except ValueError as e:
        print(f"❌ ERROR: {e}")
        print("Cannot proceed without a valid seed.")
        sys.exit(1)

    # Resolve outputs directories (namespaced by dataset and exp_name)
    resolved = resolve_outputs_dirs(cfg)
    print(f"Outputs -> exp_dir: {resolved['exp_dir']}")

    # Set float32 matmul precision based on config
    try:
        if cfg.training.use_cuda and torch.cuda.is_available():
            precision = getattr(cfg, "float32_precision", "medium")
            torch.set_float32_matmul_precision(precision)
            print(
                f"Set torch.float32 matmul precision to '{precision}' (Tensor Cores enabled)"
            )
    except Exception:
        pass

    # Handle device
    if cfg.training.use_cuda and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        print(f"Using CUDA device {args.device}")
    else:
        print("Using CPU")

    # Prepare data
    data_module = prepare_data(cfg)

    # Train
    train_model(cfg, data_module)


if __name__ == "__main__":
    main()
