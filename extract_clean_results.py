"""
Clean extraction of training results from logs and checkpoints.
Ignores tqdm noise and extracts only clean metrics.
"""

import os
import re
from pathlib import Path


def safe_float(s):
    """Safely convert string to float, stripping trailing dots"""
    return float(str(s).rstrip("."))


def extract_results_from_checkpoint_names(exp_dir, dataset_name):
    """Extract metrics from checkpoint filenames (they contain val_loss)"""
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")

    if not os.path.exists(checkpoint_dir):
        return None

    # Pattern: dataset-name-epoch=00-val_loss=1.30.ckpt
    checkpoints = []
    for fname in os.listdir(checkpoint_dir):
        if fname.endswith(".ckpt"):
            # Parse: epoch and val_loss
            match = re.search(r"epoch=(\d+)-val_loss=([\d.]+)\.ckpt", fname)
            if match:
                epoch = int(match.group(1))
                try:
                    val_loss = safe_float(match.group(2))
                    checkpoints.append((epoch, val_loss, fname))
                except ValueError:
                    pass

    if checkpoints:
        # Sort by epoch
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints
    return None


def extract_early_stopping_info(log_file):
    """Extract early stopping and best score from log"""
    if not os.path.exists(log_file):
        return None

    results = {
        "best_score": None,
        "best_score_epoch": None,
        "stopped_epoch": None,
        "patience": None,
    }

    with open(log_file, "r", errors="ignore") as f:
        for line in f:
            # Look for: "Best score: 0.826"
            if "Best score:" in line:
                match = re.search(r"Best score:\s+([\d.]+)", line)
                if match:
                    try:
                        results["best_score"] = safe_float(match.group(1))
                    except ValueError:
                        pass

            # Look for: "did not improve in the last 15 records"
            if "did not improve in the last" in line:
                match = re.search(r"did not improve in the last (\d+) records", line)
                if match:
                    results["patience"] = int(match.group(1))

                # Extract current epoch from line
                match = re.search(r"Epoch (\d+):", line)
                if match:
                    results["stopped_epoch"] = int(match.group(1))

    return results


def extract_final_test_results(log_file):
    """Extract final test results from log"""
    if not os.path.exists(log_file):
        return None

    results = {
        "test_auc": None,
        "test_acc": None,
        "test_f1": None,
        "test_loss": None,
    }

    with open(log_file, "r", errors="ignore") as f:
        content = f.read()

        # Look for final test results section
        # Pattern: "test_acc_epoch         0.7322"
        match = re.search(r"test_acc_epoch\s+([\d.]+)", content)
        if match:
            try:
                results["test_acc"] = safe_float(match.group(1))
            except ValueError:
                pass

        match = re.search(r"test_auc_epoch\s+([\d.]+)", content)
        if match:
            try:
                results["test_auc"] = safe_float(match.group(1))
            except ValueError:
                pass

        match = re.search(r"test_f1_epoch\s+([\d.]+)", content)
        if match:
            try:
                results["test_f1"] = safe_float(match.group(1))
            except ValueError:
                pass

        match = re.search(r"test_loss\s+([\d.]+)", content)
        if match:
            try:
                results["test_loss"] = safe_float(match.group(1))
            except ValueError:
                pass

    return results


def extract_best_epoch_metrics(log_file):
    """
    Extract metrics at best epoch by looking for the pattern where
    early stopping message appears - that's when best checkpoint was reached.
    """
    if not os.path.exists(log_file):
        return None

    metrics = {
        "val_auc": None,
        "test_auc": None,
        "epoch": None,
    }

    with open(log_file, "r", errors="ignore") as f:
        lines = f.readlines()

    # Look backwards from the end for early stopping message
    for i in range(len(lines) - 1, max(0, len(lines) - 200), -1):
        line = lines[i]

        if "did not improve in the last" in line:
            # Found early stopping, extract metrics from this line
            match = re.search(r"val_auc_epoch=([\d.]+)", line)
            if match:
                try:
                    metrics["val_auc"] = safe_float(match.group(1))
                except ValueError:
                    pass

            match = re.search(r"test_auc_epoch=([\d.]+)", line)
            if match:
                try:
                    metrics["test_auc"] = safe_float(match.group(1))
                except ValueError:
                    pass

            match = re.search(r"Epoch (\d+):", line)
            if match:
                metrics["epoch"] = int(match.group(1))

            break

    return metrics


def main():
    datasets = {
        "wiki-rfa": {
            "exp_dir": "outputs/wiki-rfa/wiki-rfa-run_20260208-145124",
            "log_file": "nohup_logs/wiki-rfa_20260208-145121.log",
        },
        "epinions": {
            "exp_dir": "outputs/epinions/epinions-run_20260208-145126",
            "log_file": "nohup_logs/epinions_20260208-145121.log",
        },
        "slashdot090221": {
            "exp_dir": "outputs/slashdot090221/slashdot090221-run_20260208-145128",
            "log_file": "nohup_logs/slashdot090221_20260208-145121.log",
        },
    }

    print("\n" + "=" * 90)
    print(" " * 20 + "TRAINING RESULTS ANALYSIS - CLEANED")
    print("=" * 90)

    all_results = {}

    for dataset_name, paths in datasets.items():
        print(f"\n{dataset_name.upper()}")
        print("-" * 90)

        exp_dir = paths["exp_dir"]
        log_file = paths["log_file"]

        # 1. Extract from checkpoints
        checkpoints = extract_results_from_checkpoint_names(exp_dir, dataset_name)
        if checkpoints:
            print(f"\n  Checkpoints saved: {len(checkpoints)} epochs")
            print(
                f"    - First checkpoint: epoch {checkpoints[0][0]:02d}, val_loss={checkpoints[0][1]:.4f}"
            )
            print(
                f"    - Last checkpoint:  epoch {checkpoints[-1][0]:02d}, val_loss={checkpoints[-1][1]:.4f}"
            )
            print(
                f"    - Best val_loss:    epoch {min(checkpoints, key=lambda x: x[1])[0]:02d}, val_loss={min(checkpoints, key=lambda x: x[1])[1]:.4f}"
            )

        # 2. Extract early stopping info
        early_stop_info = extract_early_stopping_info(log_file)
        if early_stop_info["best_score"] is not None:
            print(f"\n  Early Stopping Info:")
            print(f"    - Best val_auc: {early_stop_info['best_score']:.4f}")
            print(f"    - Patience: {early_stop_info['patience']} epochs")
            print(f"    - Stopped at epoch: {early_stop_info['stopped_epoch']}")

        # 3. Extract best epoch metrics
        best_metrics = extract_best_epoch_metrics(log_file)
        if best_metrics["val_auc"] is not None:
            print(f"\n  Metrics at Best Epoch:")
            print(f"    - Epoch: {best_metrics['epoch']}")
            print(f"    - Val AUC: {best_metrics['val_auc']:.4f}")
            print(f"    - Test AUC: {best_metrics['test_auc']:.4f}")

        # 4. Extract final test results
        final_test = extract_final_test_results(log_file)
        if final_test["test_auc"] is not None:
            print(f"\n  Final Test Results (on best checkpoint):")
            print(f"    - Test AUC: {final_test['test_auc']:.4f}")
            print(f"    - Test ACC: {final_test['test_acc']:.4f}")
            print(f"    - Test F1:  {final_test['test_f1']:.4f}")
            print(f"    - Test Loss: {final_test['test_loss']:.4f}")

        all_results[dataset_name] = {
            "checkpoints": checkpoints,
            "early_stopping": early_stop_info,
            "best_epoch": best_metrics,
            "final_test": final_test,
        }

    # Summary table
    print("\n" + "=" * 90)
    print(" " * 25 + "SUMMARY TABLE")
    print("=" * 90)
    print(f"\n{'Dataset':<20} {'Best Epoch':<15} {'Val AUC':<15} {'Test AUC':<15}")
    print("-" * 90)

    for dataset_name, results in all_results.items():
        if results["best_epoch"]["epoch"] is not None:
            epoch = results["best_epoch"]["epoch"]
            val_auc = results["best_epoch"]["val_auc"]
            test_auc = results["best_epoch"]["test_auc"]
            print(f"{dataset_name:<20} {epoch:<15} {val_auc:<15.4f} {test_auc:<15.4f}")

    print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    main()
