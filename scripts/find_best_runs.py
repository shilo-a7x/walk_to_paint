#!/usr/bin/env python3
"""Find best runs by test AUC from tensorboard logs."""

import os
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd


def extract_metrics_from_run(log_dir):
    """Extract metrics from a single run's tensorboard logs."""
    try:
        ea = event_accumulator.EventAccumulator(str(log_dir))
        ea.Reload()

        metrics = {}

        # Get all scalar tags
        tags = ea.Tags().get("scalars", [])

        for tag in tags:
            events = ea.Scalars(tag)
            if events:
                # Get the last value for each metric
                metrics[tag] = events[-1].value
                # Also store all values for plotting later
                metrics[f"{tag}_all"] = [(e.step, e.value) for e in events]

        return metrics
    except Exception as e:
        print(f"Error reading {log_dir}: {e}")
        return None


def find_best_runs(base_dir="outputs"):
    """Find best runs for each dataset based on test AUC."""

    datasets = ["wiki-rfa", "slashdot090221", "epinions"]
    results = []

    for dataset in datasets:
        dataset_dir = Path(base_dir) / dataset
        if not dataset_dir.exists():
            continue

        print(f"\n{'='*60}")
        print(f"Analyzing {dataset}")
        print(f"{'='*60}")

        # Find all run directories with tensorboard logs
        for run_dir in dataset_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name in [
                "edge_scores",
                "optuna",
                "plots",
            ]:
                continue

            # Look for tensorboard logs - try different paths
            tb_dir = None
            # Try logs/{run_name}/version_0
            potential_tb_dir = run_dir / "logs"
            if potential_tb_dir.exists():
                # Find any version_0 directory under logs
                for subdir in potential_tb_dir.rglob("version_0"):
                    if subdir.is_dir():
                        tb_dir = subdir
                        break

            # Try lightning_logs/version_0
            if tb_dir is None:
                potential_tb_dir = run_dir / "lightning_logs" / "version_0"
                if potential_tb_dir.exists():
                    tb_dir = potential_tb_dir

            if tb_dir is None:
                continue

            print(f"\nProcessing: {run_dir.name}")
            metrics = extract_metrics_from_run(tb_dir)

            if metrics is None:
                continue

            # Look for test AUC metrics
            test_auc = None
            test_loss = None
            val_auc = None
            val_loss = None

            for key in metrics:
                if "test" in key.lower() and "auc" in key.lower() and "_all" not in key:
                    test_auc = metrics[key]
                    print(f"  Test AUC: {test_auc:.4f}")
                elif (
                    "test" in key.lower()
                    and "loss" in key.lower()
                    and "_all" not in key
                ):
                    test_loss = metrics[key]
                elif (
                    "val" in key.lower() and "auc" in key.lower() and "_all" not in key
                ):
                    val_auc = metrics[key]
                elif (
                    "val" in key.lower() and "loss" in key.lower() and "_all" not in key
                ):
                    val_loss = metrics[key]

            # Find all checkpoints for this run
            ckpt_dir = run_dir / "checkpoints"
            checkpoints = []
            if ckpt_dir.exists():
                checkpoints = sorted([str(p) for p in ckpt_dir.glob("*.ckpt")])

            results.append(
                {
                    "dataset": dataset,
                    "run_name": run_dir.name,
                    "run_path": str(run_dir),
                    "test_auc": test_auc,
                    "test_loss": test_loss,
                    "val_auc": val_auc,
                    "val_loss": val_loss,
                    "num_checkpoints": len(checkpoints),
                    "checkpoints": checkpoints,
                    "metrics": metrics,
                }
            )

    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(results)

    # Find best run per dataset
    print(f"\n{'='*60}")
    print("BEST RUNS BY TEST AUC")
    print(f"{'='*60}")

    for dataset in datasets:
        dataset_runs = df[df["dataset"] == dataset].copy()
        if dataset_runs.empty:
            print(f"\n{dataset}: No runs found")
            continue

        # Sort by test AUC (descending)
        dataset_runs = dataset_runs.sort_values("test_auc", ascending=False)

        print(f"\n{dataset}:")
        print("-" * 60)
        for idx, row in dataset_runs.iterrows():
            print(f"  {row['run_name']}")
            print(
                f"    Test AUC:  {row['test_auc']:.4f}"
                if row["test_auc"]
                else "    Test AUC:  N/A"
            )
            print(
                f"    Test Loss: {row['test_loss']:.4f}"
                if row["test_loss"]
                else "    Test Loss: N/A"
            )
            print(
                f"    Val AUC:   {row['val_auc']:.4f}"
                if row["val_auc"]
                else "    Val AUC:   N/A"
            )
            print(
                f"    Val Loss:  {row['val_loss']:.4f}"
                if row["val_loss"]
                else "    Val Loss:  N/A"
            )
            print(f"    Checkpoints: {row['num_checkpoints']}")

        # Show best run
        if not dataset_runs.empty and dataset_runs.iloc[0]["test_auc"]:
            best = dataset_runs.iloc[0]
            print(f"\n  ✓ BEST: {best['run_name']}")
            print(f"    Path: {best['run_path']}")
            print(f"    Test AUC: {best['test_auc']:.4f}")
            print(f"    Checkpoints ({best['num_checkpoints']}):")
            for ckpt in best["checkpoints"][:3]:
                print(f"      - {Path(ckpt).name}")
            if best["num_checkpoints"] > 3:
                print(f"      ... and {best['num_checkpoints'] - 3} more")

    return df


if __name__ == "__main__":
    df = find_best_runs()

    # Save results
    df.to_pickle("outputs/best_runs_analysis.pkl")
    print("\n\nResults saved to: outputs/best_runs_analysis.pkl")
