# run.py
import os
import torch
import argparse
from omegaconf import OmegaConf
from src.data.prepare_data import prepare_data
from src.training.train import train_model
from src.utils.paths import resolve_outputs_dirs
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device id (default: 0)"
    )
    parser.add_argument(
        "overrides", nargs=argparse.REMAINDER, help="Override config values"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load and merge config (supports `configs/<dataset>.yaml` and CLI dotlist overrides)
    cfg = load_config(args.config, overrides=args.overrides)

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

    # Set global seeds for reproducibility (if provided in config)
    seed = getattr(cfg.training, "seed", None)
    if seed is not None:
        try:
            seed = int(seed)
            from pytorch_lightning import seed_everything
            import random, numpy as np

            seed_everything(seed, workers=True)
            random.seed(seed)
            np.random.seed(seed)
            print(f"Using seed={seed}")
        except Exception:
            pass

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
