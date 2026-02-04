import argparse
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.data.datasets import get_loader


def _list_dataset_names(root: str):
    configs_dir = os.path.join(root, "configs")
    names = []
    if os.path.isdir(configs_dir):
        for fname in os.listdir(configs_dir):
            if fname.endswith(".yaml"):
                names.append(fname[:-5])
    if not names:
        cfg = load_config(os.path.join(root, "config.yaml"))
        try:
            names = [cfg.dataset.name]
        except Exception:
            names = []
    return sorted(set(names))


def _compute_stats(edges):
    total_edges = len(edges)
    nodes = set()
    loops = 0
    pair_counts = Counter()
    for e in edges:
        if len(e) < 2:
            continue
        u, v = e[0], e[1]
        nodes.add(u)
        nodes.add(v)
        if u == v:
            loops += 1
        pair_counts[(u, v)] += 1

    unique_directed = len(pair_counts)
    multi_edge_pairs = sum(1 for c in pair_counts.values() if c > 1)
    multi_edge_extra = sum(c - 1 for c in pair_counts.values() if c > 1)

    pair_set = set(pair_counts.keys())
    bidirectional_pairs = 0
    reciprocal_edges = 0
    symmetric_multiplicity_pairs = 0
    for u, v in pair_set:
        if u == v:
            continue
        if (v, u) in pair_set:
            reciprocal_edges += 1
            if u < v:
                bidirectional_pairs += 1
                if pair_counts[(u, v)] == pair_counts[(v, u)]:
                    symmetric_multiplicity_pairs += 1

    directed = not all((v, u) in pair_set for (u, v) in pair_set if u != v)

    unique_undirected = len({((u, v) if u <= v else (v, u)) for (u, v) in pair_set})

    reciprocal_ratio = 100.0 * reciprocal_edges / max(1, unique_directed - loops)

    return {
        "nodes": len(nodes),
        "edges_total": total_edges,
        "edges_unique_directed": unique_directed,
        "edges_unique_undirected": unique_undirected,
        "self_loops": loops,
        "multi_edge_pairs": multi_edge_pairs,
        "multi_edge_extra": multi_edge_extra,
        "bidirectional_pairs": bidirectional_pairs,
        "reciprocal_edges": reciprocal_edges,
        "reciprocal_ratio": reciprocal_ratio,
        "symmetric_multiplicity_pairs": symmetric_multiplicity_pairs,
        "directed": directed,
    }


def _format_stats(name: str, stats: dict) -> str:
    graph_type = "directed" if stats["directed"] else "undirected"
    lines = [f"Dataset: {name}"]
    lines.append(f"  Type: {graph_type}")
    lines.append(f"  Nodes: {stats['nodes']}")
    lines.append(
        f"  Edges: {stats['edges_total']} (unique directed: {stats['edges_unique_directed']}, "
        f"unique undirected: {stats['edges_unique_undirected']})"
    )
    lines.append(f"  Self-loops: {stats['self_loops']}")
    lines.append(
        f"  Multiedges: {stats['multi_edge_pairs']} pairs, {stats['multi_edge_extra']} extra edges"
    )
    lines.append(
        f"  Bidirectional pairs: {stats['bidirectional_pairs']} "
        f"(reciprocal directed edges: {stats['reciprocal_edges']}, "
        f"{stats['reciprocal_ratio']:.2f}% of non-loop directed edges)"
    )
    lines.append(
        f"  Symmetric multiplicity pairs: {stats['symmetric_multiplicity_pairs']}"
    )
    return "\n".join(lines)


def main():
    root = ROOT

    parser = argparse.ArgumentParser(
        description="Compute basic graph properties for datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset names to analyze (default: all configs/*.yaml)",
    )
    args = parser.parse_args()

    dataset_names = args.datasets or _list_dataset_names(root)
    if not dataset_names:
        raise SystemExit("No datasets found.")

    for name in dataset_names:
        cfg = load_config(
            os.path.join(root, "config.yaml"),
            overrides=[
                f"dataset.name={name}",
                "dataset.remove_self_loops=false",
                "dataset.multiedge_handling=keep",
            ],
        )
        loader = get_loader(name)
        edges = loader(cfg)
        stats = _compute_stats(edges)
        print(_format_stats(name, stats))
        print("-" * 60)


if __name__ == "__main__":
    main()
