#!/usr/bin/env python3
"""
Generate heatmaps from saved predictions showing prediction patterns
by distance from start/end of walks.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re


def find_best_checkpoint_epoch(exp_dir):
    """Find epoch with lowest val_loss from checkpoint filenames"""
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    best_epoch = None
    best_val_loss = float("inf")

    for fname in os.listdir(checkpoint_dir):
        if fname.endswith(".ckpt"):
            match = re.search(r"epoch=(\d+)-val_loss=([\d.]+)\.ckpt", fname)
            if match:
                epoch = int(match.group(1))
                val_loss = float(str(match.group(2)).rstrip("."))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch

    if best_epoch is None:
        raise ValueError("Could not find best checkpoint from checkpoint filenames")

    print(f"✓ Best checkpoint: epoch {best_epoch} with val_loss={best_val_loss:.4f}")
    return best_epoch


def load_predictions(dataset_name, exp_dir, epoch, split="test"):
    """Load predictions for a specific epoch and split"""
    pred_file = os.path.join(
        exp_dir,
        "checkpoints",
        f"{dataset_name}_predictions",
        f"epoch_{epoch:03d}",
        f"{split}_predictions.pkl",
    )

    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")

    with open(pred_file, "rb") as f:
        return pickle.load(f)


def generate_heatmap(predictions, dataset_name, epoch, output_dir, n_bins=20):
    """
    Generate heatmap showing prediction correctness by position in walk.

    X-axis: distance from start
    Y-axis: distance from end
    Color: proportion of correct predictions
    """
    dist_start = predictions["dist_from_start"]
    dist_end = predictions["dist_from_end"]
    correct = predictions["correct"]

    print(f"  Total predictions: {len(correct):,}")
    print(f"  Accuracy: {correct.mean():.4f}")
    print(f"  Distance from start range: [{dist_start.min()}, {dist_start.max()}]")
    print(f"  Distance from end range: [{dist_end.min()}, {dist_end.max()}]")

    # Create 2D bins
    max_dist = max(dist_start.max(), dist_end.max())

    # Compute 2D histogram of correctness
    H_correct, xedges, yedges = np.histogram2d(
        dist_start,
        dist_end,
        bins=n_bins,
        range=[[0, max_dist], [0, max_dist]],
        weights=correct,
    )

    H_total, _, _ = np.histogram2d(
        dist_start, dist_end, bins=n_bins, range=[[0, max_dist], [0, max_dist]]
    )

    # Compute proportion correct (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy_grid = np.divide(H_correct, H_total)
        accuracy_grid[H_total == 0] = np.nan

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(
        accuracy_grid.T,
        origin="lower",
        extent=[0, max_dist, 0, max_dist],
        aspect="auto",
        cmap="cividis",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    ax.set_xlabel(
        "Distance from START node (walk position)\n← At start (d=0) | Far from start →",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Distance from END node (walk position)\n← At end (d=0) | Far from end →",
        fontsize=13,
        fontweight="bold",
    )

    title_text = (
        f"{dataset_name.upper()} - Epoch {epoch:03d} - PREDICTION ACCURACY HEATMAP\n"
        f"Overall Accuracy: {correct.mean():.4f} | Total Predictions: {len(correct):,}\n"
        f"BLACK = High Accuracy (correct predictions) | WHITE = Low Accuracy (incorrect)\n"
        f"Each cell shows: What % of predictions were CORRECT at that (d_start, d_end) position"
    )
    ax.set_title(title_text, fontsize=12, fontweight="bold", pad=20)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(
        "Prediction Accuracy\n(Black=1.0=Perfect, White=0.0=All Wrong)",
        fontsize=12,
        fontweight="bold",
    )
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    output_file = os.path.join(
        output_dir, f"{dataset_name}_epoch{epoch:03d}_heatmap.png"
    )
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Heatmap saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate heatmaps from saved predictions"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--exp-dir", type=str, required=True, help="Experiment directory"
    )
    parser.add_argument(
        "--epoch", type=str, default="best", help="Epoch to analyze (best/last/number)"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Data split (test/val/train)"
    )
    parser.add_argument(
        "--n-bins", type=int, default=20, help="Number of bins for heatmap"
    )

    args = parser.parse_args()

    print("=" * 70)
    print(f"POST-HOC HEATMAP ANALYSIS: {args.dataset}")
    print("=" * 70)
    print(f"  Experiment dir: {args.exp_dir}")
    print(f"  Epoch: {args.epoch}")
    print(f"  Split: {args.split}")
    print(f"  Bins: {args.n_bins}x{args.n_bins}")
    print()

    # Determine epoch
    if args.epoch == "best":
        epoch = find_best_checkpoint_epoch(args.exp_dir)
    elif args.epoch == "last":
        # Find last epoch from predictions
        pred_dir = os.path.join(
            args.exp_dir, "checkpoints", f"{args.dataset}_predictions"
        )
        epochs = sorted(
            [
                int(d.split("_")[1])
                for d in os.listdir(pred_dir)
                if d.startswith("epoch_")
            ]
        )
        epoch = epochs[-1]
        print(f"✓ Last epoch: {epoch}")
    else:
        epoch = int(args.epoch)
        print(f"✓ Using specified epoch: {epoch}")

    # Load predictions
    print(f"\nLoading predictions for epoch {epoch}...")
    predictions = load_predictions(args.dataset, args.exp_dir, epoch, args.split)

    # Create output directory
    output_dir = os.path.join(args.exp_dir, "posthoc_heatmaps")
    os.makedirs(output_dir, exist_ok=True)

    # Generate heatmap
    print(f"\nGenerating heatmap...")
    generate_heatmap(predictions, args.dataset, epoch, output_dir, args.n_bins)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
