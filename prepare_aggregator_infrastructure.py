"""
Aggregator Training Infrastructure Setup

This module prepares data for training edge-level aggregators that learn to
combine multiple walk-level predictions into a single edge prediction.

Key concepts:
1. Per-walk predictions: Model predicts score for each walk passing through an edge
2. Per-edge predictions: We want to aggregate multiple walks → single edge label
3. Aggregator inputs: (distance_from_start, distance_from_end, walk_prediction)
4. Aggregator target: Ground truth edge label
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


class AggregatorDatasetBuilder:
    """Build datasets for training edge-level aggregators"""

    def __init__(self, config):
        self.dataset_name = config["dataset_name"]
        self.exp_dir = config["exp_dir"]
        self.predictions_dir = os.path.join(
            self.exp_dir, "checkpoints", f"{self.dataset_name}_predictions"
        )
        self.output_dir = os.path.join(self.exp_dir, "aggregator_data")
        os.makedirs(self.output_dir, exist_ok=True)

    def build_aggregator_training_data(self, epoch: int, split: str = "train"):
        """
        Build training data for aggregator from walk predictions.

        Data structure:
        {
            'edge_id': {
                'ground_truth': 0 or 1,
                'walks': [
                    {
                        'distance_from_start': float,
                        'distance_from_end': float,
                        'walk_length': int,
                        'prediction': float,
                    },
                    ...
                ]
            },
            ...
        }
        """
        pred_file = os.path.join(
            self.predictions_dir, f"epoch_{epoch:03d}", f"{split}_predictions.pkl"
        )

        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Predictions not found: {pred_file}")

        with open(pred_file, "rb") as f:
            predictions = pickle.load(f)

        # predictions format:
        # {
        #     'predictions': {edge_id: [scores...]},
        #     'metadata': {edge_id: [(d_start, d_end, length)...]},
        #     'ground_truth': {edge_id: label}
        # }

        edge_predictions = predictions["predictions"]
        edge_metadata = predictions["metadata"]
        ground_truth = predictions.get("ground_truth", {})

        aggregator_data = {}

        for edge_id, scores in edge_predictions.items():
            if edge_id not in edge_metadata:
                continue

            metadata = edge_metadata[edge_id]
            label = ground_truth.get(edge_id, None)

            # Build walk records
            walks = []
            for i, (d_start, d_end, length) in enumerate(metadata):
                if i < len(scores):
                    walks.append(
                        {
                            "distance_from_start": d_start,
                            "distance_from_end": d_end,
                            "walk_length": length,
                            "prediction": scores[i],
                        }
                    )

            aggregator_data[edge_id] = {
                "ground_truth": label,
                "walks": walks,
            }

        return aggregator_data

    def get_aggregator_train_val_test_splits(self, epoch: int):
        """
        Get aggregator training/validation/test splits.

        Question: What edges should we train aggregator on?
        - Train aggregator: edges from TRAIN split of original data
        - Val aggregator: edges from VAL split of original data
        - Test aggregator: edges from TEST split of original data

        Reasoning:
        - Train split has ground truth labels during original training
        - Val/Test splits have out-of-training-set ground truth
        - This prevents data leakage
        """

        # Build data from each split
        aggregator_splits = {}

        for split_name in ["train", "val", "test"]:
            print(f"  Building aggregator data for {split_name} split...")
            agg_data = self.build_aggregator_training_data(epoch, split_name)
            aggregator_splits[split_name] = agg_data

            # Statistics
            total_edges = len(agg_data)
            edges_with_walks = sum(1 for e in agg_data.values() if e["walks"])
            total_walks = sum(len(e["walks"]) for e in agg_data.values())

            print(f"    - Total edges: {total_edges}")
            print(f"    - Edges with walks: {edges_with_walks}")
            print(f"    - Total walks: {total_walks}")
            print(f"    - Avg walks/edge: {total_walks / max(edges_with_walks, 1):.1f}")

        return aggregator_splits

    def save_aggregator_data(self, epoch: int, aggregator_splits: Dict):
        """Save aggregator training data"""
        output_file = os.path.join(
            self.output_dir, f"epoch_{epoch:03d}_aggregator_data.pkl"
        )

        with open(output_file, "wb") as f:
            pickle.dump(aggregator_splits, f)

        print(f"✓ Saved aggregator data to {output_file}")


def describe_aggregator_training_setup():
    """
    Describe the aggregator training setup and edge/split strategy.
    """
    description = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                    AGGREGATOR TRAINING INFRASTRUCTURE                         ║
╚════════════════════════════════════════════════════════════════════════════════╝

WHAT ARE WE TRAINING?
─────────────────────
An edge-level aggregator that learns to combine multiple walk predictions into
a single edge prediction. This handles the case where many random walks pass
through the same edge with different positions.

INPUTS TO AGGREGATOR (per-walk basis)
─────────────────────────────────────
For each walk through an edge:
  1. distance_from_start:  How far along the walk the edge appears (0-indexed)
  2. distance_from_end:    How far from the end the edge appears
  3. walk_length:          Total length of the walk
  4. walk_prediction:      Model's prediction for this walk

These inputs capture position information that might affect prediction quality.

POSSIBLE AGGREGATOR STRATEGIES
──────────────────────────────
1. SIMPLE AVERAGING (baseline):
   - Take mean/median of all walk predictions
   - No learnable parameters

2. POSITION-WEIGHTED AVERAGING:
   - Weight walks by distance metrics (e.g., favor edges near start/end)
   - Still simple but captures position effects

3. LEARNED AGGREGATOR (MLP):
   - Train MLP: (d_start, d_end, length, pred) → aggregated_pred
   - Learn how much to trust each walk based on position
   - Compare against ground truth labels

4. ATTENTION-BASED:
   - Learn to attend to most informative walks
   - More complex, may overfit with small edge count

WHICH EDGES FOR TRAINING?
──────────────────────────
Question: On which edges do we train/val/test the aggregator?

Answer (CRITICAL for no data leakage):
  ✓ Train aggregator on: edges from original TRAIN split
    - These edges have ground truth labels from training time
    - Aggregator learns position-weighted combination strategy

  ✓ Val aggregator on: edges from original VAL split
    - Different edges, unseen during aggregator training
    - Validates generalization

  ✓ Test aggregator on: edges from original TEST split
    - Completely unseen edges
    - Final evaluation

TRAIN/VAL/TEST STRATEGY
───────────────────────
For aggregator training:
  - X (features): [distance_from_start, distance_from_end, walk_length, walk_pred]
  - Y (target): ground_truth label for the edge
  
  For each edge:
    - Collect ALL walks that pass through it from the split
    - Use walks as samples to train the aggregator
    - Target is the ground truth edge label (same for all walks of same edge)

Example:
  Edge A in train split:
    - Walk 1 passes with: (d_start=5, d_end=10, length=20, pred=0.7)
    - Walk 2 passes with: (d_start=3, d_end=15, length=20, pred=0.8)
    - Walk 3 passes with: (d_start=10, d_end=5, length=20, pred=0.6)
    - Ground truth: 1 (positive edge)
  
  Training samples:
    - Sample 1: [5, 10, 20, 0.7] → 1
    - Sample 2: [3, 15, 20, 0.8] → 1
    - Sample 3: [10, 5, 20, 0.6] → 1

DATA LEAKAGE PREVENTION
───────────────────────
✓ No leakage: Each split is independent
  - Train aggregator on train edges only
  - Val/test edges were never seen during original model training
  - Ground truth for val/test from original data splits

✗ Would be leakage:
  - Using test split edges to train aggregator
  - Using same edges for training and testing

EXPECTED BENEFITS
─────────────────
1. Better edge predictions by using all walk information
2. Understanding which positions matter most (distance effects)
3. Comparison: learned vs simple aggregation
4. Analysis: how much does edge position affect prediction quality?
"""

    print(description)


def main():
    parser = argparse.ArgumentParser(
        description="Setup infrastructure for aggregator training"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name (e.g., wiki-rfa)"
    )
    parser.add_argument(
        "--exp-dir", type=str, required=True, help="Experiment directory"
    )
    parser.add_argument(
        "--epoch", type=int, default=-1, help="Epoch to use (-1 for last)"
    )
    parser.add_argument(
        "--describe-only",
        action="store_true",
        help="Only print description, do not build data",
    )

    args = parser.parse_args()

    # Print description
    # describe_aggregator_training_setup()

    if args.describe_only:
        return

    # Build aggregator data
    config = {
        "dataset_name": args.dataset,
        "exp_dir": args.exp_dir,
    }

    builder = AggregatorDatasetBuilder(config)

    print(f"\n{'='*80}")
    print(f"BUILDING AGGREGATOR DATA: {args.dataset}")
    print(f"{'='*80}")
    print(f"Experiment: {args.exp_dir}")
    print(f"Epoch selection: {args.epoch} (-1 = last)")

    # For now, use epoch -1 (last)
    # In practice, you'd select best epoch from metrics
    epoch_to_use = args.epoch
    if epoch_to_use == -1:
        # Find last epoch
        epochs = []
        for d in os.listdir(builder.predictions_dir):
            if d.startswith("epoch_"):
                epochs.append(int(d.split("_")[1]))
        epoch_to_use = max(epochs)

    print(f"\nUsing epoch: {epoch_to_use}")

    # Get splits
    aggregator_splits = builder.get_aggregator_train_val_test_splits(epoch_to_use)

    # Save
    builder.save_aggregator_data(epoch_to_use, aggregator_splits)


if __name__ == "__main__":
    main()
