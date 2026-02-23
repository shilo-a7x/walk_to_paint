#!/usr/bin/env python3
"""
Test post-hoc analysis: triplets and heatmaps from saved predictions.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def load_predictions(pred_file):
    """Load predictions from new format."""
    with open(pred_file, "rb") as f:
        return pickle.load(f)


def extract_triplets(predictions):
    """
    Extract triplets from predictions.

    Returns dict with:
        - dist_from_start
        - dist_from_end
        - correct (0 or 1)
        - edge_ids, walk_ids, positions (for traceability)
    """
    # predictions already has dist_from_start and dist_from_end computed
    triplets = {
        "edge_ids": predictions["edge_ids"],
        "walk_ids": predictions["walk_ids"],
        "positions": predictions["positions"],
        "walk_lengths": predictions["walk_lengths"],
        "dist_from_start": predictions["dist_from_start"],
        "dist_from_end": predictions["dist_from_end"],
        "correct": predictions["correct"].astype(int),
    }

    return triplets


def create_heatmap(triplets, output_path):
    """
    Create heatmap: avg(correct) as function of (dist_from_start, dist_from_end).
    """
    dist_start = triplets["dist_from_start"]
    dist_end = triplets["dist_from_end"]
    correct = triplets["correct"]

    # Get unique values to create proper grid
    unique_starts = np.unique(dist_start)
    unique_ends = np.unique(dist_end)

    # Create mapping from distance to grid index
    start_to_idx = {v: i for i, v in enumerate(unique_starts)}
    end_to_idx = {v: i for i, v in enumerate(unique_ends)}

    # Initialize grid
    grid = np.full((len(unique_ends), len(unique_starts)), np.nan)
    counts = np.zeros((len(unique_ends), len(unique_starts)), dtype=int)

    # Accumulate correct flags
    for ds, de, c in zip(dist_start, dist_end, correct):
        idx_s = start_to_idx[ds]
        idx_e = end_to_idx[de]
        if np.isnan(grid[idx_e, idx_s]):
            grid[idx_e, idx_s] = 0
        grid[idx_e, idx_s] += c
        counts[idx_e, idx_s] += 1

    # Compute average
    with np.errstate(divide="ignore", invalid="ignore"):
        grid = np.where(counts > 0, grid / counts, np.nan)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", origin="lower")

    # Set tick labels to actual distance values
    ax.set_xticks(range(len(unique_starts)))
    ax.set_xticklabels(unique_starts)
    ax.set_yticks(range(len(unique_ends)))
    ax.set_yticklabels(unique_ends)

    ax.set_xlabel("Distance from Start (edges)", fontsize=12)
    ax.set_ylabel("Distance from End (edges)", fontsize=12)
    ax.set_title("Average Correctness by Position in Walk", fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Avg Correct", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Heatmap saved to: {output_path}")
    print(f"  Grid shape: {grid.shape}")
    print(f"  Filled cells: {np.sum(~np.isnan(grid))}")
    print(f"  Avg correctness range: [{np.nanmin(grid):.3f}, {np.nanmax(grid):.3f}]")


def main():
    # Find the latest toy run
    toy_dir = Path("outputs/toy")
    runs = sorted([d for d in toy_dir.iterdir() if d.is_dir()])
    if not runs:
        print("No toy runs found!")
        return

    latest_run = runs[-1]
    print(f"Using latest run: {latest_run.name}")

    # Find predictions directory
    pred_dir = latest_run / "checkpoints" / "toy_predictions"
    if not pred_dir.exists():
        print(f"Predictions directory not found: {pred_dir}")
        return

    # Process each epoch and split
    output_dir = latest_run / "posthoc_analysis"
    output_dir.mkdir(exist_ok=True)

    for epoch_dir in sorted(pred_dir.iterdir()):
        if not epoch_dir.is_dir():
            continue

        epoch_num = epoch_dir.name.split("_")[1]
        print(f"\n=== Epoch {epoch_num} ===")

        for split in ["train", "val", "test"]:
            pred_file = epoch_dir / f"{split}_predictions.pkl"
            if not pred_file.exists():
                continue

            print(f"\nProcessing {split} split...")

            # Load predictions
            predictions = load_predictions(pred_file)
            print(f"  Loaded {len(predictions['predictions'])} predictions")
            print(f"  AUC: {predictions['auc']:.4f}")

            # Extract triplets
            triplets = extract_triplets(predictions)
            print(f"  Extracted {len(triplets['correct'])} triplets")
            print(f"  Avg correctness: {triplets['correct'].mean():.4f}")

            # Save triplets
            triplet_file = output_dir / f"epoch_{epoch_num}_{split}_triplets.pkl"
            with open(triplet_file, "wb") as f:
                pickle.dump(triplets, f)
            print(f"  ✓ Saved triplets to: {triplet_file.name}")

            # Create heatmap
            heatmap_file = output_dir / f"epoch_{epoch_num}_{split}_heatmap.png"
            create_heatmap(triplets, heatmap_file)

    print(f"\n✅ Post-hoc analysis complete!")
    print(f"   Output directory: {output_dir}")


if __name__ == "__main__":
    main()
