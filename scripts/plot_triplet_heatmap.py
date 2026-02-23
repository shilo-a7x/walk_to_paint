#!/usr/bin/env python3
"""
Plot heatmap of average correctness over (dist_from_start, dist_from_end).

Input:
  outputs/aggregation/<dataset>/strategy_mean/triplets_<split>.pkl

Output:
  outputs/aggregation/<dataset>/strategy_mean/<run_id>/heatmap_<split>.png
  outputs/aggregation/<dataset>/strategy_mean/<run_id>/heatmap_<split>_data.npz
  outputs/aggregation/<dataset>/strategy_mean/<run_id>/heatmap_<split>_stats.json
"""

import argparse
import os
import pickle
import json

import numpy as np
import matplotlib.pyplot as plt


def load_triplets(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_heatmap(triplets):
    dist_start = np.array(triplets["dist_from_start"], dtype=np.int64)
    dist_end = np.array(triplets["dist_from_end"], dtype=np.int64)
    correct = np.array(triplets["correct"], dtype=np.float32)

    if dist_start.size == 0:
        return np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.int32)

    max_x = int(dist_start.max())
    max_y = int(dist_end.max())

    sum_grid = np.zeros((max_y + 1, max_x + 1), dtype=np.float32)
    cnt_grid = np.zeros((max_y + 1, max_x + 1), dtype=np.int32)

    for x, y, c in zip(dist_start, dist_end, correct):
        sum_grid[y, x] += c
        cnt_grid[y, x] += 1

    heatmap = np.full_like(sum_grid, np.nan, dtype=np.float32)
    mask = cnt_grid > 0
    heatmap[mask] = sum_grid[mask] / cnt_grid[mask]

    return heatmap, cnt_grid


def plot_heatmap(heatmap, output_path):
    plt.figure(figsize=(8, 6))
    img = plt.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    plt.colorbar(img, label="Avg Correct")
    plt.xlabel("Distance from Start (edges)")
    plt.ylabel("Distance from End (edges)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compute_heatmap_stats(heatmap, counts, triplets):
    """Compute statistics about the heatmap."""
    valid_vals = heatmap[~np.isnan(heatmap)]

    dist_start = np.array(triplets.get("dist_from_start", []))
    dist_end = np.array(triplets.get("dist_from_end", []))
    correct = np.array(triplets.get("correct", []))

    stats = {
        "grid_shape": tuple(heatmap.shape),
        "total_cells": int(heatmap.size),
        "non_empty_cells": int(np.sum(~np.isnan(heatmap))),
        "sparsity": float(1.0 - (np.sum(~np.isnan(heatmap)) / heatmap.size)),
        "num_triplets": len(correct),
        "accuracy": {
            "mean": float(np.mean(valid_vals)) if len(valid_vals) > 0 else None,
            "std": float(np.std(valid_vals)) if len(valid_vals) > 0 else None,
            "min": float(np.min(valid_vals)) if len(valid_vals) > 0 else None,
            "max": float(np.max(valid_vals)) if len(valid_vals) > 0 else None,
        },
        "position_stats": {
            "avg_dist_from_start": (
                float(np.mean(dist_start)) if len(dist_start) > 0 else None
            ),
            "avg_dist_from_end": (
                float(np.mean(dist_end)) if len(dist_end) > 0 else None
            ),
            "boundary_accuracy": (
                float(np.mean(correct[np.logical_or(dist_start == 0, dist_end == 0)]))
                if np.any(np.logical_or(dist_start == 0, dist_end == 0))
                else None
            ),
            "interior_accuracy": (
                float(np.mean(correct[np.logical_and(dist_start > 0, dist_end > 0)]))
                if np.any(np.logical_and(dist_start > 0, dist_end > 0))
                else None
            ),
        },
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Plot heatmap of avg correctness over distances"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--split", type=str, required=True, help="Split name")
    parser.add_argument(
        "--run-id",
        type=str,
        default="strategy_mean",
        help="Run identifier for organizing outputs",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs/aggregation",
        help="Base directory for triplet inputs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/aggregation",
        help="Base directory for heatmap outputs",
    )

    args = parser.parse_args()

    # Include run_id in paths to avoid overwriting
    input_path = os.path.join(
        args.input_dir,
        args.dataset,
        "strategy_mean",
        args.run_id,
        f"triplets_{args.split}.pkl",
    )
    output_path = os.path.join(
        args.output_dir,
        args.dataset,
        "strategy_mean",
        args.run_id,
        f"heatmap_{args.split}.png",
    )

    triplets = load_triplets(input_path)
    heatmap, counts = build_heatmap(triplets)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_heatmap(heatmap, output_path)

    print(f"Saved heatmap to: {output_path}")

    # Save raw data (numpy arrays)
    data_path = output_path.replace(".png", "_data.npz")
    np.savez(data_path, grid=heatmap, counts=counts)
    print(f"Saved heatmap data to: {data_path}")

    # Save statistics
    stats = compute_heatmap_stats(heatmap, counts, triplets)
    stats_path = output_path.replace(".png", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved heatmap stats to: {stats_path}")


if __name__ == "__main__":
    main()
