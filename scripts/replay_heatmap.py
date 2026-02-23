#!/usr/bin/env python3
"""
Replay heatmap visualizations from saved raw data.
Allows customizing plots without recomputing everything.

Usage:
  python scripts/replay_heatmap.py --data-path outputs/.../heatmap_val_data.npz --output-dir plots/
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_heatmap_data(data_path):
    """Load saved heatmap data from npz file."""
    data = np.load(data_path)
    return data["grid"], data["counts"]


def plot_heatmap_custom(
    grid, counts, output_path, cmap="RdYlGn", vmin=0, vmax=1, title=None
):
    """Plot heatmap with customizable parameters."""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    plt.colorbar(im, label="Avg Correctness", ax=ax)
    ax.set_xlabel("Distance from Start", fontsize=12)
    ax.set_ylabel("Distance from End", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_heatmap_with_counts(grid, counts, output_path, title=None):
    """Plot heatmap with count information."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy heatmap
    im1 = ax1.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    plt.colorbar(im1, label="Avg Correctness", ax=ax1)
    ax1.set_xlabel("Distance from Start", fontsize=11)
    ax1.set_ylabel("Distance from End", fontsize=11)
    ax1.set_title("Accuracy Heatmap", fontweight="bold")

    # Counts heatmap (log scale for visibility)
    im2 = ax2.imshow(
        np.log1p(counts),
        origin="lower",
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )
    plt.colorbar(im2, label="log(Count + 1)", ax=ax2)
    ax2.set_xlabel("Distance from Start", fontsize=11)
    ax2.set_ylabel("Distance from End", fontsize=11)
    ax2.set_title("Sample Counts Heatmap", fontweight="bold")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_diagonal_accuracy(grid, output_path, title=None):
    """Plot accuracy along main diagonal (equal distances from start/end)."""
    diagonal = np.diag(grid)

    fig, ax = plt.subplots(figsize=(10, 6))
    valid_idx = ~np.isnan(diagonal)
    ax.plot(
        np.where(valid_idx)[0],
        diagonal[valid_idx],
        "o-",
        linewidth=2,
        markersize=8,
        color="steelblue",
    )
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% accuracy")
    ax.set_xlabel("Distance from Start = Distance from End", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_distance_effects(grid, output_path, title=None):
    """Plot how accuracy changes with total distance from both ends."""
    distance_stats = {}

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                total_dist = i + j
                if total_dist not in distance_stats:
                    distance_stats[total_dist] = []
                distance_stats[total_dist].append(grid[i, j])

    # Compute mean and std
    distances = sorted(distance_stats.keys())
    means = [np.mean(distance_stats[d]) for d in distances]
    stds = [np.std(distance_stats[d]) for d in distances]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        distances,
        means,
        yerr=stds,
        fmt="o-",
        linewidth=2,
        markersize=8,
        capsize=5,
        capthick=2,
        color="steelblue",
        label="Mean ± Std",
    )
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% accuracy")
    ax.set_xlabel("Total Distance from Both Ends", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Replay heatmap visualizations from saved data"
    )
    parser.add_argument(
        "--data-path", required=True, help="Path to heatmap_data.npz file"
    )
    parser.add_argument("--output-dir", default=".", help="Output directory for plots")
    parser.add_argument(
        "--all", action="store_true", help="Generate all alternative plots"
    )
    parser.add_argument("--cmap", default="RdYlGn", help="Colormap for heatmap")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    grid, counts = load_heatmap_data(args.data_path)
    base_name = Path(args.data_path).stem.replace("_data", "")

    if args.all:
        # Generate all plots
        plot_heatmap_custom(
            grid,
            counts,
            os.path.join(args.output_dir, f"{base_name}_default.png"),
            cmap=args.cmap,
        )
        plot_heatmap_with_counts(
            grid, counts, os.path.join(args.output_dir, f"{base_name}_with_counts.png")
        )
        plot_diagonal_accuracy(
            grid, os.path.join(args.output_dir, f"{base_name}_diagonal.png")
        )
        plot_distance_effects(
            grid, os.path.join(args.output_dir, f"{base_name}_distance_effects.png")
        )
    else:
        # Default plot
        plot_heatmap_custom(
            grid,
            counts,
            os.path.join(args.output_dir, f"{base_name}.png"),
            cmap=args.cmap,
        )


if __name__ == "__main__":
    main()
