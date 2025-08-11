#!/usr/bin/env python3
import torch
import json
import numpy as np


def analyze_dataset():
    print("Loading encoded data...")
    data = torch.load("data/chess/encoded.pt")

    print("Data structure:", type(data))
    if isinstance(data, (list, tuple)):
        print("Data length:", len(data))
        print("First few items:")
        for i, item in enumerate(data[:3]):
            if hasattr(item, "shape"):
                print(f"  Item {i}: shape={item.shape}, dtype={item.dtype}")
            else:
                print(f"  Item {i}: type={type(item)}")
    else:
        print("Data keys:", list(data.keys()) if hasattr(data, "keys") else "No keys")

    # Load splits to map indices
    with open("data/chess/splits.json", "r") as f:
        splits = json.load(f)

    print("\nAnalyzing class distribution...")

    # Assuming data is a tuple of (input_ids, labels)
    if isinstance(data, (list, tuple)) and len(data) >= 2:
        input_ids, labels = data[0], data[1]
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Labels shape: {labels.shape}")

        # Analyze each split
        for split_name, indices in splits.items():
            if isinstance(indices, list) and len(indices) > 0:
                split_labels = labels[indices]

                # Remove ignore_index (-100) tokens
                valid_labels = split_labels[split_labels != -100]
                print(f"\n{split_name.upper()} SET:")
                print(f"  Total tokens: {split_labels.numel()}")
                print(f"  Valid tokens: {valid_labels.numel()}")
                print(f"  Padding ratio: {(split_labels == -100).float().mean():.3f}")

                if valid_labels.numel() > 0:
                    unique, counts = torch.unique(valid_labels, return_counts=True)
                    total_valid = valid_labels.numel()

                    print("  Class distribution:")
                    for u, c in zip(unique, counts):
                        percentage = c / total_valid * 100
                        print(f"    Class {u}: {c} ({percentage:.1f}%)")

                    # Calculate class weights for balancing
                    class_weights = total_valid / (len(unique) * counts.float())
                    print("  Suggested class weights:")
                    for u, w in zip(unique, class_weights):
                        print(f"    Class {u}: {w:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    analyze_dataset()
