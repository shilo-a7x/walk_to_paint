#!/usr/bin/env python3
"""
Compare Optuna tuned models and show best model per dataset.
"""

import os
import pickle

import numpy as np


def _load_optuna_summary(results_dir):
    """Load Optuna results from summary.txt."""
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
        elif current_model and "Val AUC:" in line:
            val_auc = float(line.split()[-1])
            results[current_model]["val_auc"] = val_auc
        elif current_model and "Train AUC:" in line:
            train_auc = float(line.split()[-1])
            results[current_model]["train_auc"] = train_auc

    return results


def main():
    base_dir = "/home/dsi/shilo_avital/yolo_lab/walk_to_paint"
    results_dir = os.path.join(base_dir, "aggregator_optuna_results")

    datasets = ["wiki-rfa", "epinions", "slashdot090221"]

    print("\n" + "=" * 100)
    print("OPTUNA TUNED AGGREGATORS - BEST MODEL PER DATASET")
    print("=" * 100 + "\n")

    print(
        f"{'Dataset':<20} {'Model':<12} {'Val AUC':<12} {'Train AUC':<12} {'Test AUC':<12}"
    )
    print("-" * 100)

    for dataset in datasets:
        dataset_dir = os.path.join(results_dir, dataset)
        results = _load_optuna_summary(dataset_dir)

        if results is None:
            print(f"{dataset:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            continue

        # Find best by test AUC
        best_model = max(
            results.keys(), key=lambda m: results[m].get("test_auc", -np.inf)
        )
        best_result = results[best_model]

        val_auc = best_result.get("val_auc", 0)
        train_auc = best_result.get("train_auc", 0)
        test_auc = best_result.get("test_auc", 0)

        print(
            f"{dataset:<20} {best_model:<12} {val_auc:<12.4f} {train_auc:<12.4f} {test_auc:<12.4f}"
        )

    print("\n" + "=" * 100)
    print("DETAILED RESULTS PER MODEL:")
    print("=" * 100 + "\n")

    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        print("-" * 100)
        print(f"{'Model':<12} {'Val AUC':<12} {'Train AUC':<12} {'Test AUC':<12}")
        print("-" * 100)

        dataset_dir = os.path.join(results_dir, dataset)
        results = _load_optuna_summary(dataset_dir)

        if results is None:
            print(f"{'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            continue

        for model in ["logistic", "xgboost", "lgbm"]:
            if model in results:
                res = results[model]
                val_auc = res.get("val_auc", 0)
                train_auc = res.get("train_auc", 0)
                test_auc = res.get("test_auc", 0)
                print(
                    f"{model:<12} {val_auc:<12.4f} {train_auc:<12.4f} {test_auc:<12.4f}"
                )

    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()
