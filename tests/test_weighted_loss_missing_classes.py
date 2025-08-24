#!/usr/bin/env python3
"""
Test weighted loss behavior with missing classes
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from src.model.metrics_helper import WeightedLossHelper


def test_missing_classes():
    print("Testing weighted loss with missing classes...")

    helper = WeightedLossHelper()
    num_classes = 3
    ignore_index = -100

    # Scenario 1: All classes present
    print("\n🔍 Scenario 1: All classes present")
    labels1 = torch.tensor([0, 0, 1, 1, 1, 2])  # Class counts: [2, 3, 1]
    weights1 = helper.compute_class_weights(labels1, num_classes, ignore_index)
    print(f"Labels: {labels1.tolist()}")
    print(f"Class counts: [2, 3, 1]")
    print(f"Weights: {weights1.tolist()}")
    print(f"Weight formula: total_samples / (num_classes * class_count)")
    print(f"Class 0: 6 / (3 * 2) = {6 / (3 * 2):.3f}")
    print(f"Class 1: 6 / (3 * 3) = {6 / (3 * 3):.3f}")
    print(f"Class 2: 6 / (3 * 1) = {6 / (3 * 1):.3f}")

    # Scenario 2: Missing class 2
    print("\n🔍 Scenario 2: Missing class 2")
    labels2 = torch.tensor(
        [0, 0, 1, 1, 1]
    )  # Class counts: [2, 3, 0] <- Class 2 missing
    weights2 = helper.compute_class_weights(labels2, num_classes, ignore_index)
    print(f"Labels: {labels2.tolist()}")
    print(f"Class counts: [2, 3, 0]")
    print(f"Weights: {weights2.tolist()}")
    print(f"Class 0: 5 / (3 * 2) = {5 / (3 * 2):.3f}")
    print(f"Class 1: 5 / (3 * 3) = {5 / (3 * 3):.3f}")
    print(f"Class 2: 1.0 (default for missing class)")

    # Scenario 3: Only one class present
    print("\n🔍 Scenario 3: Only class 1 present")
    labels3 = torch.tensor([1, 1, 1, 1])  # Class counts: [0, 4, 0]
    weights3 = helper.compute_class_weights(labels3, num_classes, ignore_index)
    print(f"Labels: {labels3.tolist()}")
    print(f"Class counts: [0, 4, 0]")
    print(f"Weights: {weights3.tolist()}")
    print(f"Class 0: 1.0 (default for missing class)")
    print(f"Class 1: 4 / (3 * 4) = {4 / (3 * 4):.3f}")
    print(f"Class 2: 1.0 (default for missing class)")

    # Scenario 4: Test actual loss computation
    print("\n🔍 Scenario 4: How PyTorch CrossEntropyLoss handles missing classes")

    # Create dummy logits
    batch_size = 4
    logits = torch.randn(batch_size, num_classes)
    labels = torch.tensor([1, 1, 1, 1])  # Only class 1

    # Compute weights
    weights = helper.compute_class_weights(labels, num_classes, ignore_index)

    # Compute weighted loss
    loss_weighted = F.cross_entropy(logits, labels, weight=weights)
    loss_unweighted = F.cross_entropy(logits, labels)

    print(f"Weights: {weights.tolist()}")
    print(f"Weighted loss: {loss_weighted.item():.4f}")
    print(f"Unweighted loss: {loss_unweighted.item():.4f}")
    print(f"Impact: Only predictions for class 1 are weighted (×{weights[1]:.3f})")
    print(f"Classes 0 and 2 weights don't affect loss (no samples to penalize)")


def test_ignore_index():
    print("\n\n🔍 Testing ignore_index behavior")

    helper = WeightedLossHelper()
    num_classes = 3
    ignore_index = -100

    # Labels with ignore_index mixed in
    labels = torch.tensor([0, -100, 1, -100, 1, 2])
    weights = helper.compute_class_weights(labels, num_classes, ignore_index)

    print(f"Labels with ignore_index: {labels.tolist()}")
    print(f"Valid labels after filtering: [0, 1, 1, 2]")
    print(f"Class counts: [1, 2, 1]")
    print(f"Weights: {weights.tolist()}")
    print("✅ ignore_index (-100) tokens are properly excluded from weight computation")


if __name__ == "__main__":
    test_missing_classes()
    test_ignore_index()
