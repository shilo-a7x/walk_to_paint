#!/usr/bin/env python3
"""
Analyze walk length distribution for datasets.
"""

import os
import sys
import numpy as np
from collections import Counter
from omegaconf import OmegaConf

from src.data.datasets import get_loader
from src.data.walk_sampler import sample_random_walks
from src.utils.config import get_seed


def get_edge_list(cfg):
    """Load edges for a dataset."""
    data = get_loader(cfg.dataset.name)(cfg)
    return data


def count_nodes_and_edges(walk):
    """Count nodes and edges in a walk."""
    num_nodes = sum(1 for token in walk if token.startswith("N_"))
    num_edges = sum(1 for token in walk if token.startswith("E_"))
    return num_nodes, num_edges


def analyze_walks(dataset_name, cfg):
    """Analyze walk length distribution."""
    print(f"\n{'='*70}")
    print(f"Walk Length Analysis for {dataset_name}")
    print(f"{'='*70}")

    # Load edges
    print(f"Loading {dataset_name} dataset...")
    edges = get_edge_list(cfg)
    print(f"✓ Loaded {len(edges)} edges")

    # Generate walks
    print(f"Sampling random walks...")
    walk_workers = int(
        getattr(
            cfg.preprocess,
            "num_workers",
            getattr(cfg.preprocess, "walk_num_workers", 1),
        )
    )
    walk_seed = get_seed(cfg)
    max_walk_length = cfg.dataset.max_walk_length

    walks = sample_random_walks(
        edges,
        num_walks=int(cfg.dataset.num_walks),
        max_walk_length=max_walk_length,
        num_workers=walk_workers,
        seed=walk_seed,
    )
    print(f"✓ Sampled {len(walks)} walks (max_walk_length={max_walk_length})")

    # Analyze walk lengths
    print(f"\nAnalyzing walk lengths...")

    # Count tokens (total length)
    token_lengths = [len(walk) for walk in walks]

    # Count edges per walk
    edge_counts = []
    node_counts = []
    for walk in walks:
        num_nodes, num_edges = count_nodes_and_edges(walk)
        node_counts.append(num_nodes)
        edge_counts.append(num_edges)

    # Statistics
    token_lengths = np.array(token_lengths)
    edge_counts = np.array(edge_counts)
    node_counts = np.array(node_counts)

    print(f"\n📊 Token Length Statistics (nodes + edges):")
    print(f"  Mean:     {np.mean(token_lengths):.2f}")
    print(f"  Median:   {np.median(token_lengths):.2f}")
    print(f"  Std Dev:  {np.std(token_lengths):.2f}")
    print(f"  Min:      {np.min(token_lengths)}")
    print(f"  Max:      {np.max(token_lengths)}")
    print(f"  Percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        val = np.percentile(token_lengths, p)
        print(f"    {p:2d}th: {val:.2f}")

    print(f"\n📊 Edge Count Statistics:")
    print(f"  Mean:     {np.mean(edge_counts):.2f}")
    print(f"  Median:   {np.median(edge_counts):.2f}")
    print(f"  Std Dev:  {np.std(edge_counts):.2f}")
    print(f"  Min:      {np.min(edge_counts)}")
    print(f"  Max:      {np.max(edge_counts)}")

    # Check walks that hit max length
    max_possible_tokens = (
        1 + 2 * max_walk_length
    )  # start node + (edge + node) * max_walk_length
    walks_at_max = np.sum(edge_counts == max_walk_length)
    pct_at_max = 100.0 * walks_at_max / len(walks)
    print(f"\n🎯 Walks reaching max length ({max_walk_length} edges):")
    print(f"  Count: {walks_at_max} / {len(walks)} ({pct_at_max:.2f}%)")

    # Check very short walks (early termination)
    short_walks = np.sum(edge_counts <= 2)
    pct_short = 100.0 * short_walks / len(walks)
    print(f"\n⚠️  Very short walks (≤2 edges):")
    print(f"  Count: {short_walks} / {len(walks)} ({pct_short:.2f}%)")

    # Distribution histogram
    print(f"\n📈 Edge Count Distribution (histogram):")
    edge_counter = Counter(edge_counts)

    # Group into buckets for display
    buckets = {}
    for edges_in_walk, count in edge_counter.items():
        if edges_in_walk == 0:
            bucket = "0"
        elif edges_in_walk <= 5:
            bucket = "1-5"
        elif edges_in_walk <= 10:
            bucket = "6-10"
        elif edges_in_walk <= 20:
            bucket = "11-20"
        elif edges_in_walk <= 40:
            bucket = "21-40"
        elif edges_in_walk <= max_walk_length - 1:
            bucket = f"41-{max_walk_length-1}"
        else:
            bucket = f"{max_walk_length}+"

        buckets[bucket] = buckets.get(bucket, 0) + count

    # Display buckets in order
    bucket_order = [
        "0",
        "1-5",
        "6-10",
        "11-20",
        "21-40",
        f"41-{max_walk_length-1}",
        f"{max_walk_length}+",
    ]
    for bucket in bucket_order:
        if bucket in buckets:
            count = buckets[bucket]
            pct = 100.0 * count / len(walks)
            bar = "█" * int(pct / 2)  # Scale bar
            print(f"  {bucket:12s}: {count:8d} ({pct:5.2f}%) {bar}")

    # Top 10 most common exact lengths
    print(f"\n🔢 Top 10 Most Common Edge Counts:")
    most_common = edge_counter.most_common(10)
    for edges_in_walk, count in most_common:
        pct = 100.0 * count / len(walks)
        print(f"  {edges_in_walk:3d} edges: {count:8d} walks ({pct:5.2f}%)")

    return {
        "dataset": dataset_name,
        "num_walks": len(walks),
        "max_walk_length": max_walk_length,
        "mean_edges": np.mean(edge_counts),
        "median_edges": np.median(edge_counts),
        "pct_at_max": pct_at_max,
        "pct_short": pct_short,
    }


def main():
    """Analyze walk lengths for multiple datasets."""

    dataset_names = ["wiki-rfa", "epinions", "slashdot090221"]

    # Base config path
    config_path = "config.yaml"

    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found at {config_path}")
        sys.exit(1)

    # Load base config
    base_cfg = OmegaConf.load(config_path)

    results = []

    for dataset_name in dataset_names:
        try:
            # Load dataset-specific config
            config_file = os.path.join("configs", f"{dataset_name}.yaml")
            if not os.path.exists(config_file):
                print(f"\n⚠️  Config file not found: {config_file}")
                continue

            dataset_cfg = OmegaConf.load(config_file)
            cfg = OmegaConf.merge(base_cfg, dataset_cfg)

            # Verify the dataset exists
            dataset_dir = cfg.dataset.data_dir
            if not os.path.exists(dataset_dir):
                print(f"\n⚠️  Dataset directory not found: {dataset_dir}")
                continue

            result = analyze_walks(dataset_name, cfg)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print(f"WALK LENGTH SUMMARY")
    print(f"{'='*70}")
    for result in results:
        print(f"{result['dataset']:20s}:")
        print(f"  Max length: {result['max_walk_length']:3d} edges")
        print(f"  Mean:       {result['mean_edges']:6.2f} edges")
        print(f"  Median:     {result['median_edges']:6.2f} edges")
        print(f"  At max:     {result['pct_at_max']:5.2f}%")
        print(f"  Very short: {result['pct_short']:5.2f}%")
        print()


if __name__ == "__main__":
    main()
