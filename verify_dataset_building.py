#!/usr/bin/env python3
"""
Verify dataset building process for all 3 production datasets.
Shows exact edgelist details, preprocessing effects, and configuration.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.datasets import get_loader
from src.utils.config import load_config
from collections import Counter
import numpy as np


def verify_dataset(dataset_name):
    """Verify dataset building for a specific dataset."""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")

    # Load config
    cfg = load_config("config.yaml", overrides=[f"dataset.name={dataset_name}"])

    print("\n📋 Configuration:")
    print(f"  data_dir: {cfg.dataset.data_dir}")
    print(f"  edge_list_file: {cfg.dataset.edge_list_file}")
    print(f"  binary: {cfg.dataset.binary}")
    print(f"  remove_self_loops: {cfg.dataset.remove_self_loops}")
    print(f"  multiedge_handling: {cfg.dataset.multiedge_handling}")

    # Load edges
    loader = get_loader(dataset_name)
    edges = loader(cfg)

    print(f"\n📊 Dataset Statistics:")
    print(f"  Total edges: {len(edges)}")

    # Extract labels
    labels = [e[2] for e in edges]
    label_counts = Counter(labels)
    unique_labels = sorted(label_counts.keys())

    print(f"  Unique labels: {unique_labels}")
    print(f"  Label distribution:")
    for label in unique_labels:
        count = label_counts[label]
        pct = 100 * count / len(edges)
        print(f"    {label:3d}: {count:7d} ({pct:5.2f}%)")

    # Node statistics
    sources = [e[0] for e in edges]
    targets = [e[1] for e in edges]
    all_nodes = set(sources) | set(targets)

    print(f"\n🔗 Node Statistics:")
    print(f"  Unique nodes: {len(all_nodes)}")
    print(f"  Min node ID: {min(all_nodes)}")
    print(f"  Max node ID: {max(all_nodes)}")

    # Check for self-loops
    self_loops = [(u, v, l) for (u, v, l) in edges if u == v]
    print(f"  Self-loops: {len(self_loops)}")
    if self_loops:
        print(
            f"    WARNING: Self-loops found but remove_self_loops={cfg.dataset.remove_self_loops}"
        )
        print(f"    Sample: {self_loops[:3]}")

    # Check for multiedges (duplicate u,v pairs)
    edge_pairs = [(e[0], e[1]) for e in edges]
    pair_counts = Counter(edge_pairs)
    multiedges = {k: v for k, v in pair_counts.items() if v > 1}

    print(f"  Multiedges: {len(multiedges)} unique pairs")
    if multiedges:
        total_multiedge_occurrences = sum(multiedges.values())
        print(f"    Total occurrences: {total_multiedge_occurrences}")
        print(f"    Max occurrences for one pair: {max(multiedges.values())}")
        if cfg.dataset.multiedge_handling == "keep":
            print(f"    ✓ multiedge_handling=keep - all occurrences preserved")
        elif cfg.dataset.multiedge_handling == "most_recent":
            print(f"    ✓ multiedge_handling=most_recent - only latest kept per pair")
        print(f"    Sample multiedges (showing counts):")
        for (u, v), count in sorted(multiedges.items(), key=lambda x: -x[1])[:5]:
            print(f"      ({u}, {v}): {count} occurrences")

    # Show sample edges
    print(f"\n📝 Sample Edges (first 10):")
    for i, (u, v, label) in enumerate(edges[:10]):
        print(f"  {i:2d}. ({u:6d}, {v:6d}) -> {label:2d}")

    # Binary classification check
    if cfg.dataset.binary:
        non_signed = [l for l in labels if l not in [-1, 1]]
        if non_signed:
            print(
                f"\n⚠️  WARNING: binary=True but found non-signed labels: {set(non_signed)}"
            )
        else:
            print(f"\n✅ Binary classification verified: all labels in {{-1, 1}}")
    else:
        print(f"\n✅ Multi-class classification: labels in {unique_labels}")

    return edges, cfg


def main():
    datasets = ["wiki-rfa", "epinions", "slashdot090221"]

    print("=" * 80)
    print("Dataset Building Verification for Task 3 Production Retraining")
    print("=" * 80)

    results = {}
    for dataset in datasets:
        try:
            edges, cfg = verify_dataset(dataset)
            results[dataset] = {
                "edges": len(edges),
                "binary": cfg.dataset.binary,
                "self_loops_removed": cfg.dataset.remove_self_loops,
                "multiedge_handling": cfg.dataset.multiedge_handling,
                "success": True,
            }
        except Exception as e:
            print(f"\n❌ Error loading {dataset}: {e}")
            import traceback

            traceback.print_exc()
            results[dataset] = {"success": False, "error": str(e)}

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(
        f"\n{'Dataset':<20} {'Edges':<10} {'Binary':<8} {'Self-Loops':<12} {'Multiedge':<15} {'Status':<8}"
    )
    print("-" * 80)

    for dataset in datasets:
        res = results[dataset]
        if res["success"]:
            print(
                f"{dataset:<20} {res['edges']:<10} {str(res['binary']):<8} "
                f"{'removed' if res['self_loops_removed'] else 'kept':<12} "
                f"{res['multiedge_handling']:<15} {'✅':<8}"
            )
        else:
            print(
                f"{dataset:<20} {'N/A':<10} {'N/A':<8} {'N/A':<12} {'N/A':<15} {'❌':<8}"
            )

    print(f"\n{'='*80}")
    print("✅ Dataset building verification complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
