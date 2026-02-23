#!/usr/bin/env python3
"""
Verify that random walks cover all nodes in the graph for selected datasets.
Checks wiki-rfa, epinions, and slashdot090221 datasets.
"""

import os
import sys
from collections import defaultdict
from omegaconf import OmegaConf

from src.data.datasets import get_loader
from src.data.walk_sampler import sample_random_walks
from src.utils.config import get_seed


def get_edge_list(cfg):
    """Load edges for a dataset."""
    data = get_loader(cfg.dataset.name)(cfg)
    return data


def extract_nodes_from_edges(edges):
    """Extract all unique nodes from edge list."""
    nodes_from_edges = set()
    for u, v, *_ in edges:
        nodes_from_edges.add(u)
        nodes_from_edges.add(v)
    return nodes_from_edges


def extract_nodes_from_walks(walks):
    """Extract all unique nodes visited in walks."""
    nodes_from_walks = set()
    for walk in walks:
        for token in walk:
            if token.startswith("N_"):
                node_id = int(token[2:])
                nodes_from_walks.add(node_id)
    return nodes_from_walks


def verify_coverage(dataset_name, cfg):
    """Verify node coverage for a dataset."""
    print(f"\n{'='*70}")
    print(f"Verifying node coverage for {dataset_name}")
    print(f"{'='*70}")

    # Load edges
    print(f"Loading {dataset_name} dataset...")
    edges = get_edge_list(cfg)
    print(f"✓ Loaded {len(edges)} edges")

    # Get nodes from edges
    nodes_from_edges = extract_nodes_from_edges(edges)
    print(f"✓ Found {len(nodes_from_edges)} unique nodes in graph")

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

    walks = sample_random_walks(
        edges,
        num_walks=int(cfg.dataset.num_walks),
        max_walk_length=cfg.dataset.max_walk_length,
        num_workers=walk_workers,
        seed=walk_seed,
    )
    print(f"✓ Sampled {len(walks)} walks")

    # Get nodes from walks
    nodes_from_walks = extract_nodes_from_walks(walks)
    print(f"✓ Found {len(nodes_from_walks)} unique nodes visited in walks")

    # Check coverage
    total_nodes = len(nodes_from_edges)
    covered_nodes = len(nodes_from_walks)
    coverage_pct = 100.0 * covered_nodes / total_nodes if total_nodes > 0 else 0

    print(f"\nCoverage statistics:")
    print(f"  Total nodes in graph: {total_nodes}")
    print(f"  Nodes covered by walks: {covered_nodes}")
    print(f"  Coverage: {coverage_pct:.2f}%")

    # Find uncovered nodes
    uncovered_nodes = nodes_from_edges - nodes_from_walks
    if uncovered_nodes:
        print(f"\n⚠️  {len(uncovered_nodes)} nodes NOT covered by walks")
        # Show first 20 uncovered nodes
        sample_uncovered = sorted(list(uncovered_nodes))[:20]
        print(f"  Sample uncovered nodes: {sample_uncovered}")

        # Analyze why they're uncovered
        print(f"\nAnalyzing uncovered nodes...")
        degree_info = defaultdict(int)
        for u, v, *_ in edges:
            degree_info[u] += 1
            degree_info[v] += 1

        degrees = [degree_info[n] for n in uncovered_nodes]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0

        print(f"  Uncovered node degrees:")
        print(f"    Min: {min_degree}, Max: {max_degree}, Avg: {avg_degree:.2f}")
    else:
        print(f"\n✅ All nodes are covered by walks!")

    # Calculate walk statistics
    walk_lengths = [len(walk) for walk in walks]
    total_tokens = sum(walk_lengths)
    avg_walk_len = total_tokens / len(walks) if walks else 0
    min_walk_len = min(walk_lengths) if walk_lengths else 0
    max_walk_len = max(walk_lengths) if walk_lengths else 0

    print(f"\nWalk statistics:")
    print(f"  Total walks: {len(walks)}")
    print(f"  Avg walk length: {avg_walk_len:.2f} tokens")
    print(f"  Min/Max walk length: {min_walk_len}/{max_walk_len} tokens")
    print(f"  Total tokens: {total_tokens}")

    return {
        "dataset": dataset_name,
        "total_nodes": total_nodes,
        "covered_nodes": covered_nodes,
        "coverage_pct": coverage_pct,
        "uncovered_count": len(uncovered_nodes),
        "num_walks": len(walks),
        "avg_walk_length": avg_walk_len,
    }


def main():
    """Verify coverage for multiple datasets."""

    # Datasets to check
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

            # Merge configs
            cfg = OmegaConf.merge(base_cfg, dataset_cfg)

            # Verify the dataset exists in data/ dir
            dataset_dir = cfg.dataset.data_dir
            if not os.path.exists(dataset_dir):
                print(f"\n⚠️  Dataset directory not found: {dataset_dir}")
                continue

            result = verify_coverage(dataset_name, cfg)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for result in results:
        status = "✅" if result["coverage_pct"] == 100.0 else "⚠️"
        print(
            f"{status} {result['dataset']:20s}: {result['covered_nodes']:5d}/{result['total_nodes']:5d} nodes "
            f"({result['coverage_pct']:6.2f}%) - {result['uncovered_count']} uncovered"
        )


if __name__ == "__main__":
    main()
