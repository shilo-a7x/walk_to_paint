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

    # Resolve outputs directories (namespaced by dataset and exp_name)
    resolved = resolve_outputs_dirs(cfg)
    print(f"Outputs -> exp_dir: {resolved['exp_dir']}")

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
