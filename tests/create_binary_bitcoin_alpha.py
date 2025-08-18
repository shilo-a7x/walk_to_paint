#!/usr/bin/env python3
"""
Convert Bitcoin Alpha dataset to binary labels.
Converts ratings to binary: positive ratings -> 1, negative/zero ratings -> 0
"""

import pandas as pd
import os


def create_binary_bitcoin_alpha():
    """Convert bitcoin alpha dataset to binary labels."""

    os.makedirs("data/bitcoin-alpha-binary", exist_ok=True)
    input_path = "data/bitcoin-alpha/soc-sign-bitcoinalpha.csv"
    output_path = "data/bitcoin-alpha-binary/soc-sign-bitcoinalpha-binary.csv"

    print("🔄 Converting Bitcoin Alpha to binary labels...")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # Read the original data
    # Format: source,target,rating,timestamp
    df = pd.read_csv(
        input_path, header=None, names=["source", "target", "rating", "timestamp"]
    )

    print(f"Original data shape: {df.shape}")
    print(f"Rating range: {df['rating'].min()} to {df['rating'].max()}")

    # Convert to binary labels
    # Positive ratings (> 0) -> 1, Non-positive ratings (<= 0) -> 0
    df["binary_rating"] = (df["rating"] > 0).astype(int)

    # Create output dataframe with binary ratings
    binary_df = df[["source", "target", "binary_rating", "timestamp"]].copy()

    # Show conversion statistics
    original_pos = (df["rating"] > 0).sum()
    original_neg = (df["rating"] <= 0).sum()
    binary_pos = (binary_df["binary_rating"] == 1).sum()
    binary_neg = (binary_df["binary_rating"] == 0).sum()

    print(f"\n📊 Conversion Statistics:")
    print(f"Original positive ratings: {original_pos}")
    print(f"Original negative/zero ratings: {original_neg}")
    print(f"Binary positive labels (1): {binary_pos}")
    print(f"Binary negative labels (0): {binary_neg}")

    # Save binary version
    binary_df.to_csv(output_path, header=False, index=False)

    print(f"\n✅ Binary dataset saved to: {output_path}")
    print(f"Format: source,target,binary_label,timestamp")

    # Show sample of converted data
    print(f"\n📝 Sample of converted data:")
    print(binary_df.head(10).to_string(index=False))

    return output_path


if __name__ == "__main__":
    create_binary_bitcoin_alpha()
