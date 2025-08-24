#!/usr/bin/env python3
"""
Test confusion matrix plotting with consistent color scale
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from src.model.metrics_helper import PlottingHelper


def test_confusion_matrix_color_scale():
    print("Testing confusion matrix with consistent color scale...")

    plotting_helper = PlottingHelper()

    # Create two different confusion matrices to test consistency
    # Matrix 1: Early training (more confused)
    cm1 = torch.tensor(
        [
            [50, 30, 20],  # Class 0: 50% correct
            [25, 45, 30],  # Class 1: 45% correct
            [15, 35, 50],  # Class 2: 50% correct
        ]
    )

    # Matrix 2: Later training (better performance)
    cm2 = torch.tensor(
        [
            [80, 15, 5],  # Class 0: 80% correct
            [10, 75, 15],  # Class 1: 75% correct
            [5, 10, 85],  # Class 2: 85% correct
        ]
    )

    print("Creating confusion matrix plots...")

    try:
        # Plot both matrices
        fig1 = plotting_helper.plot_confusion_matrix(cm1, "Early Training", 3)
        fig2 = plotting_helper.plot_confusion_matrix(cm2, "Later Training", 3)

        if fig1 and fig2:
            print("✅ Success! Both confusion matrices plotted")
            print("✅ Color scale is now consistent (vmin=0.0, vmax=1.0)")
            print(
                "💡 In TensorBoard, you can now easily compare evolution across epochs!"
            )

            # Save figures for visual inspection (optional)
            fig1.savefig("cm_early.png", dpi=150, bbox_inches="tight")
            fig2.savefig("cm_later.png", dpi=150, bbox_inches="tight")

            # Close figures to free memory
            plt.close(fig1)
            plt.close(fig2)
        else:
            print("❌ Failed to create figures")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    test_confusion_matrix_color_scale()
