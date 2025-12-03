#!/usr/bin/env python3
"""
Profile data creation for Slashdot dataset with given walk length and num_walks.
Writes timing output to stdout.
"""
import argparse
from src.utils.config import load_config
from src.data.prepare_data import prepare_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--walk_length", type=int, required=True)
    p.add_argument("--num_walks", type=int, required=True)
    p.add_argument("--num_workers", type=int, default=None)
    args = p.parse_args()

    overrides = [
        "dataset.name=slashdot090221",
        f"dataset.max_walk_length={args.walk_length}",
        f"dataset.num_walks={args.num_walks}",
    ]
    if args.num_workers is not None:
        overrides.append(f"training.num_workers={args.num_workers}")
    # ensure persistent workers true for profiling per user's request
    overrides.append("training.persistent_workers=true")

    cfg = load_config(args.config, overrides=overrides)
    print(f"Profiling data creation with max_walk_length={args.walk_length}, num_walks={args.num_walks}")
    dl = prepare_data(cfg)
    print("Prepared dataloaders:")
    for k, v in dl.items():
        try:
            print(f"  {k}: batches={len(v)}")
        except Exception:
            print(f"  {k}: {v}")

if __name__ == '__main__':
    main()
