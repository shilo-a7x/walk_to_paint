#!/usr/bin/env python3
"""
Compare aggregator results with transformer baseline.

Reads:
  - Transformer metrics from experiment directories
  - Aggregator metrics from summary.txt files

Outputs:
  - Comparison table (AUC: transformer vs aggregator for each dataset/split/model)
"""

import os
import pickle
import sys
from collections import defaultdict

import numpy as np


def _load_transformer_metrics(exp_dir, dataset, epoch=0):
    """Load transformer val/test AUC from predictions."""
    metrics = {}

    for split in ["val", "test"]:
        pred_file = os.path.join(
            exp_dir,
            "checkpoints",
            f"{dataset}_predictions",
            f"epoch_{epoch:03d}",
            f"{split}_predictions.pkl",
        )
        if not os.path.exists(pred_file):
            continue

        with open(pred_file, "rb") as f:
            preds = pickle.load(f)

        targets = preds["targets"].astype(int)
        if "probabilities" in preds and preds["probabilities"].ndim == 2:
            probs = preds["probabilities"][:, 1].astype(float)
        else:
            probs = preds["predictions"].astype(float)

        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(targets, probs)
        metrics[split] = auc

    return metrics


def _load_aggregator_metrics(results_dir, model):
    """Load aggregator AUC from summary.txt."""
    summary_file = os.path.join(results_dir, model, "summary.txt")
    if not os.path.exists(summary_file):
        return None

    metrics = {}
    with open(summary_file, "r") as f:
        lines = f.readlines()

    in_edge_level_section = False
    for line in lines:
        if "Edge-level aggregated AUC" in line:
            in_edge_level_section = True
            continue

        if in_edge_level_section:
            if "Train AUC:" in line:
                # Extract AUC from: "  Train AUC: 0.8436 (17574 edges)"
                metrics["train"] = float(line.split("AUC:")[1].split("(")[0].strip())
            elif "Test  AUC:" in line or "Test AUC:" in line:
                # Extract AUC from: "  Test  AUC: 0.8444 (17607 edges)"
                metrics["test"] = float(line.split("AUC:")[1].split("(")[0].strip())

    return metrics


def main():
    base_dir = "/home/dsi/shilo_avital/yolo_lab/walk_to_paint"
    results_dir = os.path.join(base_dir, "aggregator_results_new")

    exp_dirs = {
        "wiki-rfa": os.path.join(
            base_dir, "outputs/wiki-rfa/wiki-rfa-run_20260208-145124"
        ),
        "epinions": os.path.join(
            base_dir, "outputs/epinions/epinions-run_20260208-145126"
        ),
        "slashdot090221": os.path.join(
            base_dir, "outputs/slashdot090221/slashdot090221-run_20260208-145128"
        ),
    }

    datasets = ["wiki-rfa", "epinions", "slashdot090221"]
    models = ["logistic", "xgboost", "lgbm"]

    print("\n" + "=" * 120)
    print(
        "AGGREGATOR vs TRANSFORMER COMPARISON (Walk-Edge Samples, Balanced Class Weights)"
    )
    print("=" * 120 + "\n")

    # Transformer baseline
    print("TRANSFORMER BASELINE:")
    print("-" * 120)
    print(f"{'Dataset':<20} {'Train AUC':<15} {'Test AUC':<15}")
    print("-" * 120)

    transformer_metrics = {}
    for dataset in datasets:
        exp_dir = exp_dirs[dataset]
        metrics = _load_transformer_metrics(exp_dir, dataset, epoch=0)
        transformer_metrics[dataset] = metrics

        train_auc = metrics.get("val", np.nan)
        test_auc = metrics.get("test", np.nan)
        print(f"{dataset:<20} {train_auc:<15.4f} {test_auc:<15.4f}")

    print("\n" + "=" * 120)
    print("AGGREGATOR MODELS:")
    print("=" * 120 + "\n")

    for model in models:
        print(f"\n{model.upper()} MODEL:")
        print("-" * 120)
        print(
            f"{'Dataset':<20} {'Agg Train AUC':<15} {'Trans Train AUC':<15} {'Delta':<15} {'Agg Test AUC':<15} {'Trans Test AUC':<15} {'Delta':<15}"
        )
        print("-" * 120)

        for dataset in datasets:
            agg_dir = os.path.join(results_dir, dataset)
            agg_metrics = _load_aggregator_metrics(agg_dir, model)
            trans_metrics = transformer_metrics[dataset]

            if agg_metrics is None:
                print(
                    f"{dataset:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}"
                )
                continue

            agg_train = agg_metrics.get("train", np.nan)
            agg_test = agg_metrics.get("test", np.nan)
            trans_train = trans_metrics.get("val", np.nan)
            trans_test = trans_metrics.get("test", np.nan)

            delta_train = agg_train - trans_train
            delta_test = agg_test - trans_test

            print(
                f"{dataset:<20} {agg_train:<15.4f} {trans_train:<15.4f} {delta_train:+.4f}       {agg_test:<15.4f} {trans_test:<15.4f} {delta_test:+.4f}"
            )

    print("\n" + "=" * 120 + "\n")


if __name__ == "__main__":
    main()
