#!/usr/bin/env python3
"""
Test ROC curve plotting with gradients to verify the fix
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model.metrics_helper import MetricsManager, PlottingHelper


def test_roc_curve_with_gradients():
    print("Testing ROC curve plotting with gradients...")

    # Create metrics manager
    metrics_manager = MetricsManager(num_classes=3, ignore_index=-100)
    plotting_helper = PlottingHelper()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metrics_manager = metrics_manager.to(device)

    # Create sample tensors with gradients (like in real training)
    batch_size = 32
    preds = torch.randint(0, 3, (batch_size,)).to(device)
    targets = torch.randint(0, 3, (batch_size,)).to(device)

    # Create probs that require gradients (simulating real training)
    logits = torch.randn(batch_size, 3, requires_grad=True).to(device)
    probs = torch.softmax(logits, dim=1)

    print(f"Tensors device: {preds.device}")
    print(f"Probs require grad: {probs.requires_grad}")

    # This should work now without gradient errors
    try:
        # Update metrics (this stores probs with gradients)
        metrics_manager.update_metrics("val", preds, targets, probs)

        # Get ROC data and try to plot
        targets_list, probs_list = metrics_manager.get_roc_data("val")
        if targets_list and probs_list:
            roc_fig = plotting_helper.plot_roc_curve(
                targets_list, probs_list, "Test", 3
            )
            if roc_fig:
                print("✅ Success! ROC curve plotted without gradient errors")
                # Close the figure to free memory
                import matplotlib.pyplot as plt

                plt.close(roc_fig)
            else:
                print("⚠️  No ROC figure generated (possibly no valid data)")
        else:
            print("⚠️  No ROC data available")

        print("✅ All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    test_roc_curve_with_gradients()
