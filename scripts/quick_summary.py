#!/usr/bin/env python3
"""
Quick summary: Best aggregator AUC vs Transformer baseline AUC
"""

import os
import pickle

import numpy as np


def _load_optuna_summary(results_dir):
    """Load Optuna results."""
    summary_file = os.path.join(results_dir, "optuna_summary.txt")
    if not os.path.exists(summary_file):
        return None

    results = {}
    with open(summary_file, "r") as f:
        lines = f.readlines()

    current_model = None
    for line in lines:
        line = line.strip()
        if line.isupper() and ":" in line:
            current_model = line.split(":")[0].lower()
            results[current_model] = {}
        elif current_model and "Test AUC:" in line:
            test_auc = float(line.split()[-1])
            results[current_model]["test_auc"] = test_auc

    return results


def _load_transformer_baseline():
    """Load transformer baseline AUCs from predictions."""
    base_dir = "/home/dsi/shilo_avital/yolo_lab/walk_to_paint"

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

    from sklearn.metrics import roc_auc_score

    baseline = {}
    for dataset, exp_dir in exp_dirs.items():
        pred_file = os.path.join(
            exp_dir,
            "checkpoints",
            f"{dataset}_predictions",
            "epoch_000",
            "test_predictions.pkl",
        )
        if os.path.exists(pred_file):
            with open(pred_file, "rb") as f:
                preds = pickle.load(f)
            targets = preds["targets"].astype(int)
            if "probabilities" in preds and preds["probabilities"].ndim == 2:
                probs = preds["probabilities"][:, 1].astype(float)
            else:
                probs = preds["predictions"].astype(float)
            auc = roc_auc_score(targets, probs)
            baseline[dataset] = auc

    return baseline


def main():
    base_dir = "/home/dsi/shilo_avital/yolo_lab/walk_to_paint"
    optuna_results_dir = os.path.join(base_dir, "aggregator_optuna_results")

    datasets = ["wiki-rfa", "epinions", "slashdot090221"]
    baseline = _load_transformer_baseline()

    print("\n" + "=" * 90)
    print("AGGREGATOR SUMMARY: Best Model per Dataset vs Transformer Baseline")
    print("=" * 90 + "\n")

    print(
        f"{'Dataset':<20} {'Best Model':<15} {'Agg Test AUC':<15} {'Trans Test AUC':<15} {'Improvement':<15}"
    )
    print("-" * 90)

    all_results = {}
    for dataset in datasets:
        dataset_dir = os.path.join(optuna_results_dir, dataset)
        optuna_results = _load_optuna_summary(dataset_dir)

        if optuna_results is None:
            print(f"{dataset:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
            continue

        # Find best model by test AUC
        best_model = max(
            optuna_results.keys(),
            key=lambda m: optuna_results[m].get("test_auc", -np.inf),
        )
        best_auc = optuna_results[best_model].get("test_auc", 0)

        trans_auc = baseline.get(dataset, 0)
        improvement = best_auc - trans_auc

        all_results[dataset] = {
            "model": best_model,
            "agg_auc": best_auc,
            "trans_auc": trans_auc,
            "improvement": improvement,
        }

        improvement_str = f"{improvement:+.4f}" if improvement != 0 else "0.0000"
        print(
            f"{dataset:<20} {best_model:<15} {best_auc:<15.4f} {trans_auc:<15.4f} {improvement_str:<15}"
        )

    print("-" * 90)

    # Summary statistics
    improvements = [all_results[d]["improvement"] for d in datasets if d in all_results]
    avg_improvement = np.mean(improvements)
    print(f"{'AVERAGE':<20} {'':<15} {'':<15} {'':<15} {avg_improvement:+.4f}")

    print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    main()
