#!/usr/bin/env python3
"""
Verify if datasets are connected and analyze connected components.
"""

import os
import sys
from collections import defaultdict, deque
from omegaconf import OmegaConf

from src.data.datasets import get_loader


def get_edge_list(cfg):
    """Load edges for a dataset."""
    data = get_loader(cfg.dataset.name)(cfg)
    return data


def build_undirected_graph(edges):
    """Build undirected graph from edges."""
    graph = defaultdict(set)
    all_nodes = set()
    for u, v, *_ in edges:
        graph[u].add(v)
        graph[v].add(u)  # undirected
        all_nodes.add(u)
        all_nodes.add(v)
    return graph, all_nodes


def find_connected_components(graph, all_nodes):
    """Find connected components using BFS."""
    visited = set()
    components = []

    for start_node in all_nodes:
        if start_node in visited:
            continue

        component = set()
        queue = deque([start_node])
        visited.add(start_node)

        while queue:
            node = queue.popleft()
            component.add(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        components.append(component)

    # Sort by size (largest first)
    components.sort(key=len, reverse=True)
    return components


def verify_connectivity(dataset_name, cfg):
    """Verify if dataset is connected."""
    print(f"\n{'='*70}")
    print(f"Connectivity Analysis for {dataset_name}")
    print(f"{'='*70}")

    # Load edges
    print(f"Loading {dataset_name} dataset...")
    edges = get_edge_list(cfg)
    print(f"✓ Loaded {len(edges)} edges")

    # Build graph
    print(f"Building graph...")
    graph, all_nodes = build_undirected_graph(edges)
    print(f"✓ Graph has {len(all_nodes)} nodes")

    # Find connected components
    print(f"Finding connected components...")
    components = find_connected_components(graph, all_nodes)
    print(f"✓ Found {len(components)} connected component(s)")

    # Analyze components
    print(f"\nConnected Components Analysis:")
    total = 0
    for i, comp in enumerate(components, 1):
        comp_size = len(comp)
        total += comp_size
        pct = 100.0 * comp_size / len(all_nodes)
        if i <= 10 or i == len(components):
            status = "✓ MAIN" if i == 1 else ""
            print(f"  Component {i:3d}: {comp_size:7d} nodes ({pct:6.2f}%) {status}")
        elif i == 11:
            print(f"  ... ({len(components) - 10} more components)")

    # Is graph connected?
    is_connected = len(components) == 1
    status = "✅ CONNECTED" if is_connected else "❌ DISCONNECTED"
    print(f"\nGraph status: {status}")

    if not is_connected:
        main_comp_size = len(components[0])
        other_nodes = total - main_comp_size
        print(
            f"  Main component: {main_comp_size} nodes ({100.0*main_comp_size/len(all_nodes):.2f}%)"
        )
        print(
            f"  Other components: {len(components)-1} components with {other_nodes} nodes total"
        )

    return {
        "dataset": dataset_name,
        "total_nodes": len(all_nodes),
        "total_edges": len(edges),
        "num_components": len(components),
        "is_connected": is_connected,
        "component_sizes": [len(c) for c in components],
    }


def main():
    """Verify connectivity for multiple datasets."""

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

            result = verify_connectivity(dataset_name, cfg)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print(f"CONNECTIVITY SUMMARY")
    print(f"{'='*70}")
    for result in results:
        status = "✅" if result["is_connected"] else "❌"
        comp_info = (
            f"({result['num_components']} components)"
            if result["num_components"] > 1
            else ""
        )
        print(
            f"{status} {result['dataset']:20s}: {result['total_nodes']:7d} nodes, {result['total_edges']:7d} edges {comp_info}"
        )


if __name__ == "__main__":
    main()
