#!/usr/bin/env python3
"""Analyze how lowering walk count affects coverage and walk-length distribution.

Uses existing `dataset_cache.pt` files in each dataset folder, so results are based on
current cached 5M-walk artifacts.
"""

import gc
import os
import sys
import argparse
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.datasets import get_loader


DATASETS = ["wiki-rfa", "epinions", "slashdot090221"]
DEFAULT_BUDGETS = [1_000_000, 750_000, 500_000, 250_000, 100_000]
DEFAULT_BASELINE = 5_000_000


def get_cfg(dataset_name: str):
    base_cfg = OmegaConf.load("config.yaml")
    ds_cfg = OmegaConf.load(os.path.join("configs", f"{dataset_name}.yaml"))
    return OmegaConf.merge(base_cfg, ds_cfg)


def load_all_nodes(cfg) -> set:
    edges = get_loader(cfg.dataset.name)(cfg)
    nodes = set()
    for u, v, *_ in edges:
        nodes.add(int(u))
        nodes.add(int(v))
    return nodes


def walk_stats(walks: List[List[str]], all_nodes: set) -> Dict[str, float]:
    visited = set()
    edge_counts = np.zeros(len(walks), dtype=np.int32)

    for i, walk in enumerate(walks):
        edges_in_walk = 0
        for token in walk:
            if token.startswith("N_"):
                visited.add(int(token[2:]))
            elif token.startswith("E_"):
                edges_in_walk += 1
        edge_counts[i] = edges_in_walk

    total_nodes = len(all_nodes)
    covered_nodes = len(visited)

    return {
        "num_walks": int(len(walks)),
        "covered_nodes": int(covered_nodes),
        "total_nodes": int(total_nodes),
        "coverage_pct": 100.0 * covered_nodes / total_nodes if total_nodes else 0.0,
        "uncovered_nodes": int(total_nodes - covered_nodes),
        "mean_edges": float(np.mean(edge_counts)),
        "median_edges": float(np.median(edge_counts)),
        "p90_edges": float(np.percentile(edge_counts, 90)),
        "p95_edges": float(np.percentile(edge_counts, 95)),
        "pct_short_le2": float(100.0 * np.mean(edge_counts <= 2)),
        "pct_at_max80": float(100.0 * np.mean(edge_counts == 80)),
    }


def compare_to_baseline(
    cur: Dict[str, float], base: Dict[str, float]
) -> Dict[str, float]:
    return {
        "delta_coverage_pp": cur["coverage_pct"] - base["coverage_pct"],
        "delta_mean_edges": cur["mean_edges"] - base["mean_edges"],
        "delta_median_edges": cur["median_edges"] - base["median_edges"],
        "delta_short_pp": cur["pct_short_le2"] - base["pct_short_le2"],
        "delta_at_max_pp": cur["pct_at_max80"] - base["pct_at_max80"],
    }


def print_row(dataset: str, budget: int, s: Dict[str, float], d: Dict[str, float]):
    print(
        f"{dataset:15s} {budget:9d} | "
        f"cov={s['coverage_pct']:6.2f}% ({s['covered_nodes']:6d}/{s['total_nodes']:6d}) "
        f"mean={s['mean_edges']:5.2f} med={s['median_edges']:4.1f} "
        f"short={s['pct_short_le2']:5.2f}% max80={s['pct_at_max80']:5.2f}% | "
        f"Δcov={d['delta_coverage_pp']:+6.3f}pp Δmean={d['delta_mean_edges']:+6.3f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Walk budget sensitivity analysis from cached walks"
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default=",".join(str(x) for x in DEFAULT_BUDGETS),
        help="Comma-separated walk budgets (e.g. 3000000,2500000,2000000)",
    )
    parser.add_argument(
        "--baseline",
        type=int,
        default=DEFAULT_BASELINE,
        help="Baseline walk count for comparison",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    budgets = sorted(set(budgets), reverse=True)
    baseline_target = int(args.baseline)

    print("Using cached walks from current dataset_cache.pt files")
    print("Metric format: coverage / mean edges / median / short(<=2) / max80")
    print("-" * 150)

    for dataset_name in DATASETS:
        cfg = get_cfg(dataset_name)
        cache_path = os.path.join(cfg.dataset.data_dir, "dataset_cache.pt")

        if not os.path.exists(cache_path):
            print(f"{dataset_name}: missing cache at {cache_path}")
            continue

        print(f"\n[{dataset_name}] loading cache: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
        walks = cache["walks"]
        total_cached = len(walks)
        all_nodes = load_all_nodes(cfg)

        print(f"  cached walks={total_cached:,}, graph nodes={len(all_nodes):,}")

        if total_cached < baseline_target:
            baseline_n = total_cached
        else:
            baseline_n = baseline_target

        baseline_stats = walk_stats(walks[:baseline_n], all_nodes)

        header = "dataset         budget    | stats                                                                 | delta vs baseline"
        print(header)
        print("-" * len(header))

        # baseline row
        base_delta = compare_to_baseline(baseline_stats, baseline_stats)
        print_row(dataset_name, baseline_n, baseline_stats, base_delta)

        for budget in budgets:
            if budget > total_cached:
                continue
            s = walk_stats(walks[:budget], all_nodes)
            d = compare_to_baseline(s, baseline_stats)
            print_row(dataset_name, budget, s, d)

        # candidate selection: keep very similar behavior
        print("  candidate rule: |Δcoverage|<=0.10pp and |Δmean_edges|<=0.50")
        chosen = None
        for budget in sorted([b for b in budgets if b <= total_cached], reverse=True):
            s = walk_stats(walks[:budget], all_nodes)
            d = compare_to_baseline(s, baseline_stats)
            if (
                abs(d["delta_coverage_pp"]) <= 0.10
                and abs(d["delta_mean_edges"]) <= 0.50
            ):
                chosen = (budget, s, d)
        if chosen is None:
            print(
                "  recommended: keep baseline (no tested lower budget met strict similarity rule)"
            )
        else:
            b, s, d = chosen
            print(
                f"  recommended lower budget: {b:,} "
                f"(Δcoverage={d['delta_coverage_pp']:+.3f}pp, Δmean={d['delta_mean_edges']:+.3f})"
            )

        del cache
        del walks
        gc.collect()


if __name__ == "__main__":
    main()
