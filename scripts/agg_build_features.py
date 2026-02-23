#!/usr/bin/env python3
"""
Build per-walk features for aggregator training from saved prediction PKLs.

Each walk produces a triplet:
    (dist_from_start, walk_length, predicted_prob)

Targets:
    edge ground-truth label

Data sources:
    - Transformer val predictions -> used for agg train/val split (80/20)
    - Transformer test predictions -> used for agg test
"""

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np


def _load_predictions(exp_dir, dataset, epoch, split):
    pred_file = os.path.join(
        exp_dir,
        "checkpoints",
        f"{dataset}_predictions",
        f"epoch_{epoch:03d}",
        f"{split}_predictions.pkl",
    )
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")
    with open(pred_file, "rb") as f:
        return pickle.load(f)


def _walk_features_from_predictions(preds):
    edge_ids = preds["edge_ids"].astype(int)
    dist_start = preds["dist_from_start"].astype(float)
    walk_lengths = preds["walk_lengths"].astype(float)
    targets = preds["targets"].astype(int)

    if "probabilities" in preds and preds["probabilities"].ndim == 2:
        pred_prob = preds["probabilities"][:, 1].astype(float)
    else:
        pred_prob = preds["predictions"].astype(float)

    X = np.stack([dist_start, walk_lengths, pred_prob], axis=1)
    y = targets

    return {
        "edge_ids": edge_ids,
        "X": X,
        "y": y,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build aggregator features from prediction PKLs"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--exp-dir", required=True, help="Experiment directory")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch number (best = 0)")
    parser.add_argument("--output", required=True, help="Output pickle path")
    args = parser.parse_args()

    val_preds = _load_predictions(args.exp_dir, args.dataset, args.epoch, "val")
    test_preds = _load_predictions(args.exp_dir, args.dataset, args.epoch, "test")

    val_features = _walk_features_from_predictions(val_preds)
    test_features = _walk_features_from_predictions(test_preds)

    output = {
        "val": val_features,
        "test": test_features,
        "feature_names": ["dist_from_start", "walk_length", "predicted_prob"],
        "meta": {
            "dataset": args.dataset,
            "epoch": args.epoch,
            "exp_dir": args.exp_dir,
            "val_edges": int(len(np.unique(val_features["edge_ids"]))),
            "test_edges": int(len(np.unique(test_features["edge_ids"]))),
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(output, f)

    print(f"Saved features: {args.output}")
    print(f"  Val edges: {len(val_features)}")
    print(f"  Test edges: {len(test_features)}")


if __name__ == "__main__":
    main()
