#!/usr/bin/env python3
"""
Convert Bitcoin OTC dataset to binary labels based on most dominant class.
Most dominant class is rating=1 (56.9% of dataset).
Binary split: rating=1 -> 1, rating≠1 -> 0
"""

import pandas as pd
import os


def create_dominant_binary_bitcoin_otc():
    """Convert bitcoin otc dataset to binary labels based on most dominant class."""

    os.makedirs("data/bitcoin-otc-binary", exist_ok=True)
    input_path = "data/bitcoin-otc/soc-sign-bitcoinotc.csv"
    output_path = "data/bitcoin-otc-binary/soc-sign-bitcoinotc-binary.csv"

    print("🔄 Converting Bitcoin OTC to dominant-class binary labels...")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # Read the original data
    # Format: source,target,rating,timestamp
    df = pd.read_csv(
        input_path, header=None, names=["source", "target", "rating", "timestamp"]
    )

    print(f"Original data shape: {df.shape}")
    print(f"Rating range: {df['rating'].min()} to {df['rating'].max()}")

    # Find most dominant class
    rating_counts = df["rating"].value_counts()
    most_dominant = rating_counts.idxmax()
    dominant_count = rating_counts.max()

    print(
        f"Most dominant rating: {most_dominant} ({dominant_count} edges, {dominant_count/len(df)*100:.1f}%)"
    )

    # Convert to binary labels based on dominant class
    # Dominant class (rating = most_dominant) -> 1, Others -> 0
    df["binary_rating"] = (df["rating"] == most_dominant).astype(int)

    # Create output dataframe with binary ratings
    binary_df = df[["source", "target", "binary_rating", "timestamp"]].copy()

    # Show conversion statistics
    dominant_class_count = (df["rating"] == most_dominant).sum()
    other_class_count = (df["rating"] != most_dominant).sum()
    binary_pos = (binary_df["binary_rating"] == 1).sum()
    binary_neg = (binary_df["binary_rating"] == 0).sum()

    print(f"\n📊 Conversion Statistics:")
    print(
        f"Rating = {most_dominant} (dominant): {dominant_class_count:5d} edges ({dominant_class_count/len(df)*100:.1f}%)"
    )
    print(
        f"Rating ≠ {most_dominant} (others):   {other_class_count:5d} edges ({other_class_count/len(df)*100:.1f}%)"
    )
    print(
        f"Binary positive labels (1): {binary_pos:5d} edges ({binary_pos/len(df)*100:.1f}%)"
    )
    print(
        f"Binary negative labels (0): {binary_neg:5d} edges ({binary_neg/len(df)*100:.1f}%)"
    )

    # Show what ratings map to each binary class
    print(f"\n🔢 Binary Mapping:")
    print(f"Class 1 (binary=1): rating = {most_dominant}")
    print(f"Class 0 (binary=0): rating ≠ {most_dominant}")

    # Show distribution of "other" ratings
    other_ratings = (
        df[df["rating"] != most_dominant]["rating"].value_counts().sort_index()
    )
    print(f"\nDistribution of 'other' ratings (binary=0):")
    for rating, count in other_ratings.head(10).items():
        percentage = count / len(df) * 100
        print(f"  Rating {rating:3d}: {count:5d} edges ({percentage:5.1f}%)")
    if len(other_ratings) > 10:
        print(f"  ... and {len(other_ratings)-10} more ratings")

    # Save binary version
    binary_df.to_csv(output_path, header=False, index=False)

    print(f"\n✅ Dominant-class binary dataset saved to: {output_path}")
    print(f"Format: source,target,binary_label,timestamp")

    # Show sample of converted data
    print(f"\n📝 Sample of converted data:")
    print(binary_df.head(10).to_string(index=False))

    return output_path


if __name__ == "__main__":
    create_dominant_binary_bitcoin_otc()
