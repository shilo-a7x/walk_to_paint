#!/usr/bin/env python3
"""
Generate epoch evolution visualizations:
- Loss/AUC vs epoch (static plots)
- ROC curves with slider (Plotly interactive)
- Heatmaps with slider (Plotly interactive)
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorboard.backend.event_processing import event_accumulator
import re


def extract_epoch_number(ckpt_name):
    """Extract epoch number from checkpoint name."""
    match = re.search(r"epoch[=_](\d+)", ckpt_name)
    if match:
        return int(match.group(1))
    return None


def load_tensorboard_metrics(run_dir):
    """Load metrics from tensorboard logs."""
    tb_dir = None
    for version_dir in Path(run_dir).rglob("version_0"):
        if version_dir.is_dir():
            tb_dir = version_dir
            break

    if tb_dir is None:
        return None

    ea = event_accumulator.EventAccumulator(str(tb_dir))
    ea.Reload()

    metrics = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        metrics[tag] = [(e.step, e.value) for e in events]

    return metrics


def plot_metrics_evolution(run_dirs, output_dir):
    """Plot loss and AUC evolution across epochs."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Training Evolution: Loss and AUC Across Epochs", fontsize=16, fontweight="bold"
    )

    datasets = list(run_dirs.keys())

    for idx, (dataset, run_dir) in enumerate(run_dirs.items()):
        print(f"Plotting metrics for {dataset}...")
        metrics = load_tensorboard_metrics(run_dir)

        if metrics is None:
            print(f"  WARNING: No metrics found for {dataset}")
            continue

        # Plot losses
        ax_loss = axes[0, idx]
        if "train_loss" in metrics:
            epochs, values = zip(*metrics["train_loss"])
            ax_loss.plot(epochs, values, "o-", label="Train", markersize=4)
        if "val_loss" in metrics:
            epochs, values = zip(*metrics["val_loss"])
            ax_loss.plot(epochs, values, "s-", label="Val", markersize=4)
        if "test_loss" in metrics:
            epochs, values = zip(*metrics["test_loss"])
            ax_loss.plot(epochs, values, "^-", label="Test", markersize=4)

        ax_loss.set_xlabel("Epoch", fontsize=11)
        ax_loss.set_ylabel("Loss", fontsize=11)
        ax_loss.set_title(f"{dataset} - Loss", fontweight="bold")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        # Plot AUC
        ax_auc = axes[1, idx]
        if "train_auc_epoch" in metrics:
            epochs, values = zip(*metrics["train_auc_epoch"])
            ax_auc.plot(epochs, values, "o-", label="Train", markersize=4)
        if "val_auc_epoch" in metrics:
            epochs, values = zip(*metrics["val_auc_epoch"])
            ax_auc.plot(epochs, values, "s-", label="Val", markersize=4)
        if "test_auc_epoch" in metrics:
            epochs, values = zip(*metrics["test_auc_epoch"])
            ax_auc.plot(epochs, values, "^-", label="Test", markersize=4)

        ax_auc.set_xlabel("Epoch", fontsize=11)
        ax_auc.set_ylabel("AUC", fontsize=11)
        ax_auc.set_title(f"{dataset} - AUC", fontweight="bold")
        ax_auc.legend()
        ax_auc.grid(True, alpha=0.3)
        ax_auc.set_ylim([0.4, 1.0])

    plt.tight_layout()
    output_file = output_dir / "metrics_evolution.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Saved metrics evolution: {output_file}")
    plt.close()


def create_heatmap_slider(dataset, run_id, split, output_dir):
    """Create interactive heatmap with epoch slider using Plotly."""
    print(f"Creating heatmap slider for {dataset} {split}...")

    base_dir = Path(f"outputs/{dataset}/aggregation/strategy_mean")

    # Find all triplet files for this run_id (now in subdirectories)
    pattern = f"{run_id}/*/triplets_{split}.pkl"
    triplet_files = sorted(base_dir.glob(pattern))

    if not triplet_files:
        print(f"  WARNING: No triplet files found for {dataset} {split}")
        return

    # Load all heatmaps
    heatmaps = []
    epochs = []

    for triplet_file in triplet_files:
        # Extract epoch from path
        ckpt_name = triplet_file.parent.name
        epoch = extract_epoch_number(ckpt_name)
        if epoch is None:
            continue

        with open(triplet_file, "rb") as f:
            data = pickle.load(f)

        # Build heatmap
        dist_start = np.array([d[0] for d in data["triplets"]])
        dist_end = np.array([d[1] for d in data["triplets"]])
        correct = np.array([d[2] for d in data["triplets"]])

        max_dist = max(dist_start.max(), dist_end.max()) + 1
        grid = np.full((max_dist, max_dist), np.nan)
        counts = np.zeros((max_dist, max_dist))

        for ds, de, c in zip(dist_start, dist_end, correct):
            grid[ds, de] = (
                np.nanmean([grid[ds, de], c]) if not np.isnan(grid[ds, de]) else c
            )
            counts[ds, de] += 1

        heatmaps.append({"epoch": epoch, "grid": grid, "counts": counts})
        epochs.append(epoch)

    if not heatmaps:
        print(f"  WARNING: No valid heatmaps for {dataset} {split}")
        return

    # Sort by epoch
    sorted_data = sorted(zip(epochs, heatmaps), key=lambda x: x[0])
    epochs = [e for e, _ in sorted_data]
    heatmaps = [h for _, h in sorted_data]

    # Create Plotly figure with slider
    fig = go.Figure()

    # Add traces for each epoch
    for i, hmap in enumerate(heatmaps):
        fig.add_trace(
            go.Heatmap(
                z=hmap["grid"],
                x=list(range(hmap["grid"].shape[1])),
                y=list(range(hmap["grid"].shape[0])),
                colorscale="RdYlGn",
                zmin=0,
                zmax=1,
                visible=(i == 0),
                name=f"Epoch {hmap['epoch']}",
                hovertemplate="Distance from start: %{y}<br>Distance from end: %{x}<br>Avg correctness: %{z:.3f}<extra></extra>",
            )
        )

    # Create slider steps
    steps = []
    for i, epoch in enumerate(epochs):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(epochs)},
                {"title": f"{dataset} {split} - Epoch {epoch}"},
            ],
            label=str(epoch),
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)

    sliders = [
        dict(
            active=0,
            yanchor="top",
            y=-0.15,
            xanchor="left",
            currentvalue=dict(prefix="Epoch: ", visible=True, xanchor="center"),
            pad=dict(b=10, t=10),
            len=0.9,
            x=0.05,
            steps=steps,
        )
    ]

    fig.update_layout(
        sliders=sliders,
        title=f"{dataset} {split} - Epoch {epochs[0]}",
        xaxis_title="Distance from end",
        yaxis_title="Distance from start",
        width=900,
        height=800,
    )

    output_file = output_dir / f"heatmap_evolution_{dataset}_{split}.html"
    fig.write_html(output_file)
    print(f"✓ Saved interactive heatmap: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate epoch evolution visualizations"
    )
    parser.add_argument("--run-id", required=True, help="Run ID for epoch analysis")
    args = parser.parse_args()

    # Best runs (from find_best_runs.py output)
    run_dirs = {
        "wiki-rfa": "outputs/wiki-rfa/wiki-rfa-run_20251225-094820",
        "slashdot090221": "outputs/slashdot090221/slashdot090221-run_20251224-214709",
        "epinions": "outputs/epinions/epinions-run_20251224-214709",
    }

    output_dir = Path(f"outputs/epoch_evolution_{args.run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating epoch evolution visualizations")
    print("=" * 60)
    print()

    # 1. Plot metrics evolution (static matplotlib)
    plot_metrics_evolution(run_dirs, output_dir)
    print()

    # 2. Create interactive heatmap sliders (Plotly)
    for dataset in run_dirs.keys():
        for split in ["val", "test"]:
            create_heatmap_slider(dataset, args.run_id, split, output_dir)
        print()

    print("=" * 60)
    print(f"✓ All visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
