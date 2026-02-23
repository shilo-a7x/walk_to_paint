"""
General post-hoc analysis script for any dataset.
Generates triplets and heatmaps from saved predictions.
- Black-white colormap
- Smaller bins for finer granularity
- Configurable epoch selection (best, last, or all)
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import argparse


@dataclass
class PostHocConfig:
    """Configuration for post-hoc analysis"""

    dataset_name: str
    exp_dir: str
    epoch_selection: str = "best"  # "best", "last", "all", or specific epoch number
    # For "best": find checkpoint with lowest val_loss
    # For "last": use the final trained epoch
    # For "all": generate for all epochs
    # For number: use specific epoch (e.g., "0", "5", "10")

    # Heatmap config
    n_bins: int = 20  # Smaller bins = finer granularity
    figsize: tuple = (12, 10)

    @property
    def predictions_dir(self):
        return os.path.join(
            self.exp_dir, "checkpoints", f"{self.dataset_name}_predictions"
        )

    @property
    def analysis_dir(self):
        return os.path.join(self.exp_dir, "posthoc_analysis")


def get_all_epochs(config: PostHocConfig):
    """Get list of all saved epoch directories"""
    if not os.path.exists(config.predictions_dir):
        raise FileNotFoundError(f"Predictions dir not found: {config.predictions_dir}")

    epochs = []
    for d in sorted(os.listdir(config.predictions_dir)):
        if d.startswith("epoch_"):
            epoch_num = int(d.split("_")[1])
            epochs.append((epoch_num, d))
    return epochs


def find_best_checkpoint_epoch(config: PostHocConfig):
    """Find epoch with lowest val_loss from checkpoint filenames"""
    import re

    checkpoint_dir = os.path.join(config.exp_dir, "checkpoints")

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

    print(f"Best checkpoint: epoch {best_epoch} with val_loss={best_val_loss:.4f}")
    return best_epoch


def select_epochs_to_analyze(config: PostHocConfig):
    """Select which epochs to analyze based on config"""
    all_epochs = get_all_epochs(config)

    if config.epoch_selection == "all":
        return [e[0] for e in all_epochs]
    elif config.epoch_selection == "last":
        return [all_epochs[-1][0]]  # Last epoch
    elif config.epoch_selection == "best":
        # Find true best checkpoint by lowest val_loss
        return [find_best_checkpoint_epoch(config)]
    else:
        # Try to parse as specific epoch number
        try:
            specific_epoch = int(config.epoch_selection)
            if any(e[0] == specific_epoch for e in all_epochs):
                return [specific_epoch]
            else:
                raise ValueError(f"Epoch {specific_epoch} not found in predictions")
        except ValueError:
            raise ValueError(
                f"Unknown epoch_selection: {config.epoch_selection}. Use 'best', 'last', 'all', or a specific epoch number."
            )


def load_predictions(config: PostHocConfig, epoch: int, split: str = "test"):
    """Load predictions for a specific epoch and split"""
    pred_file = os.path.join(
        config.predictions_dir, f"epoch_{epoch:03d}", f"{split}_predictions.pkl"
    )

    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")

    with open(pred_file, "rb") as f:
        return pickle.load(f)


def generate_heatmap(
    config: PostHocConfig, predictions: dict, epoch: int, split: str = "test"
):
    """
    Generate black-white heatmap from predictions.
    Shows agreement of predictions at different distances from start/end.
    """
    # Extract predictions with distance metadata
    # predictions['predictions'] = {edge_id: [scores...]}
    # predictions['metadata'] = {edge_id: [(dist_from_start, dist_from_end, walk_length)...]}

    edge_predictions = predictions["predictions"]  # edge_id -> list of walk scores
    edge_metadata = predictions[
        "metadata"
    ]  # edge_id -> list of (dist_start, dist_end, length)

    # Create 2D histogram: dist_from_start vs dist_from_end
    all_dist_start = []
    all_dist_end = []
    all_agreement = []

    for edge_id, scores in edge_predictions.items():
        if edge_id not in edge_metadata:
            continue

        metadata = edge_metadata[edge_id]

        for i, (d_start, d_end, length) in enumerate(metadata):
            if i < len(scores):
                # Convert distance to normalized coordinates (0-1)
                if length > 0:
                    norm_start = d_start / length
                    norm_end = d_end / length

                    all_dist_start.append(norm_start)
                    all_dist_end.append(norm_end)
                    all_agreement.append(scores[i])

    if not all_dist_start:
        print(f"  ⚠ No valid predictions found for {split} split, epoch {epoch}")
        return None

    # Create 2D bins
    hist_agreement, xedges, yedges = np.histogram2d(
        all_dist_start,
        all_dist_end,
        bins=config.n_bins,
        weights=all_agreement,
        range=[[0, 1], [0, 1]],
    )

    hist_count, _, _ = np.histogram2d(
        all_dist_start, all_dist_end, bins=config.n_bins, range=[[0, 1], [0, 1]]
    )

    # Normalize by count
    with np.errstate(divide="ignore", invalid="ignore"):
        hist_normalized = np.divide(
            hist_agreement,
            hist_count,
            where=hist_count > 0,
            out=np.full_like(hist_agreement, np.nan),
        )

    # Create figure with black-white colormap
    fig, ax = plt.subplots(figsize=config.figsize)

    im = ax.imshow(
        hist_normalized.T,
        origin="lower",
        cmap="binary",
        aspect="auto",
        extent=[0, 1, 0, 1],
        interpolation="nearest",
    )

    ax.set_xlabel("Distance from Start (normalized)", fontsize=12)
    ax.set_ylabel("Distance from End (normalized)", fontsize=12)
    ax.set_title(
        f"{config.dataset_name} - {split} split - Epoch {epoch}\n"
        f"Prediction Agreement Heatmap ({config.n_bins}x{config.n_bins} bins)",
        fontsize=14,
        fontweight="bold",
    )

    cbar = plt.colorbar(im, ax=ax, label="Mean Prediction Score")

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    return fig


def generate_triplets(
    config: PostHocConfig, predictions: dict, epoch: int, split: str = "test"
):
    """
    Generate triplet statistics from predictions.
    Shows how model predictions vary across walks for same edge.
    """
    edge_predictions = predictions["predictions"]

    triplets = {
        "num_walks_per_edge": [],
        "prediction_std": [],
        "prediction_mean": [],
        "edge_count": 0,
    }

    for edge_id, scores in edge_predictions.items():
        scores_arr = np.array(scores)
        triplets["num_walks_per_edge"].append(len(scores))
        triplets["prediction_std"].append(np.std(scores_arr))
        triplets["prediction_mean"].append(np.mean(scores_arr))
        triplets["edge_count"] += 1

    # Print summary
    print(f"\n  {split} split - Epoch {epoch}:")
    print(f"    - Total edges: {triplets['edge_count']}")
    print(f"    - Avg walks per edge: {np.mean(triplets['num_walks_per_edge']):.1f}")
    print(f"    - Avg prediction std: {np.mean(triplets['prediction_std']):.4f}")
    print(f"    - Avg prediction mean: {np.mean(triplets['prediction_mean']):.4f}")

    return triplets


def save_analysis_results(config: PostHocConfig, epoch: int, results: dict):
    """Save analysis results"""
    os.makedirs(config.analysis_dir, exist_ok=True)

    results_file = os.path.join(config.analysis_dir, f"epoch_{epoch:03d}_analysis.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    print(f"  ✓ Saved analysis to {results_file}")


def main(args):
    config = PostHocConfig(
        dataset_name=args.dataset,
        exp_dir=args.exp_dir,
        epoch_selection=args.epoch_selection,
        n_bins=args.n_bins,
    )

    print(f"\n{'='*70}")
    print(f"POST-HOC ANALYSIS: {config.dataset_name}")
    print(f"{'='*70}")
    print(f"  Experiment dir: {config.exp_dir}")
    print(f"  Predictions dir: {config.predictions_dir}")
    print(f"  Epoch selection: {config.epoch_selection}")
    print(f"  Heatmap bins: {config.n_bins}x{config.n_bins}")

    # Select epochs
    epochs_to_analyze = select_epochs_to_analyze(config)
    print(f"  Analyzing epochs: {epochs_to_analyze}")

    os.makedirs(config.analysis_dir, exist_ok=True)

    # Analyze each epoch
    for epoch in epochs_to_analyze:
        print(f"\n  Processing epoch {epoch}...")

        epoch_results = {"epoch": epoch, "splits": {}}

        # Analyze each split
        for split in ["train", "val", "test"]:
            try:
                predictions = load_predictions(config, epoch, split)

                # Generate triplets
                triplets = generate_triplets(config, predictions, epoch, split)
                epoch_results["splits"][split] = {"triplets": triplets}

                # Generate heatmap
                fig = generate_heatmap(config, predictions, epoch, split)
                if fig is not None:
                    heatmap_file = os.path.join(
                        config.analysis_dir, f"epoch_{epoch:03d}_{split}_heatmap.png"
                    )
                    fig.savefig(heatmap_file, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print(f"    ✓ Saved {split} heatmap")

            except FileNotFoundError as e:
                print(f"    ⚠ Skipping {split}: {e}")
                continue

        # Save results
        save_analysis_results(config, epoch, epoch_results)

    print(f"\n✓ Analysis complete! Results saved to {config.analysis_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="General post-hoc analysis for any dataset"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name (e.g., wiki-rfa)"
    )
    parser.add_argument(
        "--exp-dir", type=str, required=True, help="Experiment directory path"
    )
    parser.add_argument(
        "--epoch-selection",
        type=str,
        default="last",
        choices=["best", "last", "all"],
        help="Which epochs to analyze",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=20,
        help="Number of bins for heatmap (higher = finer granularity)",
    )

    args = parser.parse_args()
    main(args)
