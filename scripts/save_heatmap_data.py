#!/usr/bin/env python3
"""
Save heatmap raw data for later analysis and visualization.

For each heatmap, saves:
  - heatmap_data.npz: numpy arrays (grid, counts, metadata)
  - heatmap_stats.json: statistics (sparsity, mean accuracy, etc)
  - triplets_summary.json: triplet field distributions
"""

import argparse
import os
import json
import pickle
import numpy as np
from pathlib import Path


def compute_heatmap_stats(grid, counts, triplets_data):
    """Compute various statistics about the heatmap."""

    # Sparsity: fraction of non-empty cells
    total_cells = grid.size
    non_empty_cells = np.sum(~np.isnan(grid))
    sparsity = 1.0 - (non_empty_cells / total_cells)

    # Accuracy statistics
    valid_vals = grid[~np.isnan(grid)]
    if len(valid_vals) > 0:
        mean_acc = float(np.mean(valid_vals))
        std_acc = float(np.std(valid_vals))
        min_acc = float(np.min(valid_vals))
        max_acc = float(np.max(valid_vals))
    else:
        mean_acc = std_acc = min_acc = max_acc = None

    # Position statistics
    dist_from_start = triplets_data.get("dist_from_start", [])
    dist_from_end = triplets_data.get("dist_from_end", [])
    total_dist = np.array(dist_from_start) + np.array(dist_from_end)

    stats = {
        "grid_shape": tuple(grid.shape),
        "total_cells": int(total_cells),
        "non_empty_cells": int(non_empty_cells),
        "sparsity": float(sparsity),
        "num_triplets": len(triplets_data.get("correct", [])),
        "accuracy": {
            "mean": mean_acc,
            "std": std_acc,
            "min": min_acc,
            "max": max_acc,
        },
        "position_stats": {
            "avg_dist_from_start": (
                float(np.mean(dist_from_start)) if len(dist_from_start) > 0 else None
            ),
            "avg_dist_from_end": (
                float(np.mean(dist_from_end)) if len(dist_from_end) > 0 else None
            ),
            "avg_total_distance": (
                float(np.mean(total_dist)) if len(total_dist) > 0 else None
            ),
        },
    }

    return stats


def compute_triplet_distributions(triplets_data):
    """Compute distributions of triplet fields."""

    dist_from_start = triplets_data.get("dist_from_start", [])
    dist_from_end = triplets_data.get("dist_from_end", [])
    walk_len = triplets_data.get("walk_len", [])
    correct = triplets_data.get("correct", [])

    if len(correct) == 0:
        return {}

    correct_arr = np.array(correct)

    distributions = {
        "accuracy_by_position": {
            "start": (
                float(
                    np.mean(
                        [
                            c
                            for i, c in enumerate(correct_arr)
                            if dist_from_start[i] == 0
                        ]
                    )
                )
                if any(d == 0 for d in dist_from_start)
                else None
            ),
            "end": (
                float(
                    np.mean(
                        [c for i, c in enumerate(correct_arr) if dist_from_end[i] == 0]
                    )
                )
                if any(d == 0 for d in dist_from_end)
                else None
            ),
            "middle": (
                float(
                    np.mean(
                        [
                            c
                            for i, c in enumerate(correct_arr)
                            if dist_from_start[i] > 0 and dist_from_end[i] > 0
                        ]
                    )
                )
                if any(d > 0 for d in dist_from_start)
                and any(d > 0 for d in dist_from_end)
                else None
            ),
        },
        "walk_length_stats": {
            "min": int(np.min(walk_len)) if len(walk_len) > 0 else None,
            "max": int(np.max(walk_len)) if len(walk_len) > 0 else None,
            "mean": float(np.mean(walk_len)) if len(walk_len) > 0 else None,
        },
        "overall_accuracy": float(np.mean(correct_arr)),
    }

    return distributions


def save_heatmap_data(triplets_file, heatmap_grid, heatmap_counts, output_dir):
    """Save raw heatmap data and statistics."""

    os.makedirs(output_dir, exist_ok=True)

    # Load triplets for stats computation
    with open(triplets_file, "rb") as f:
        triplets_data = pickle.load(f)

    # Compute statistics
    stats = compute_heatmap_stats(heatmap_grid, heatmap_counts, triplets_data)
    distributions = compute_triplet_distributions(triplets_data)

    # Save numpy data
    npz_path = os.path.join(output_dir, "heatmap_data.npz")
    np.savez(
        npz_path,
        grid=heatmap_grid,
        counts=heatmap_counts,
    )

    # Save statistics
    stats_path = os.path.join(output_dir, "heatmap_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Save distributions
    dist_path = os.path.join(output_dir, "triplet_distributions.json")
    with open(dist_path, "w") as f:
        json.dump(distributions, f, indent=2)

    return npz_path, stats_path, dist_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save heatmap raw data for analysis")
    parser.add_argument(
        "--triplets-file", required=True, help="Path to triplets pickle file"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save heatmap data"
    )

    args = parser.parse_args()

    # This is called from plot_triplet_heatmap.py
    # Just showing the function signatures here
    print("Use save_heatmap_data() function from plot_triplet_heatmap.py")
