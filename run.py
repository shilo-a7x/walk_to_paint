# run.py
import os
import torch
import argparse
from omegaconf import OmegaConf
from src.data.prepare_data import prepare_data
from src.training.train import train_model


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

    # Load and override config
    cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_dotlist(args.overrides)
    cfg = OmegaConf.merge(cfg, cli_cfg)

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
