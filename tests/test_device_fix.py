#!/usr/bin/env python3
"""
Quick test to verify the device fix for MetricsManager
"""

import torch
from src.model.metrics_helper import MetricsManager


# Test device handling
def test_device_fix():
    print("Testing MetricsManager device handling...")

    # Create metrics manager
    metrics_manager = MetricsManager(num_classes=3, ignore_index=-100)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metrics_manager = metrics_manager.to(device)

    # Create sample tensors on the same device
    batch_size = 32
    preds = torch.randint(0, 3, (batch_size,)).to(device)
    targets = torch.randint(0, 3, (batch_size,)).to(device)
    probs = torch.softmax(torch.randn(batch_size, 3), dim=1).to(device)

    print(f"Tensors device: {preds.device}")
    print(f"Metrics device: {next(metrics_manager.parameters()).device}")

    # This should work now without device errors
    try:
        metrics_manager.update_metrics("val", preds, targets, probs)
        results = metrics_manager.compute_and_reset_metrics("val")
        print("✅ Success! No device mismatch errors")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1: {results['f1']:.4f}")
        print(f"AUC: {results['auc']:.4f}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


if __name__ == "__main__":
    test_device_fix()
