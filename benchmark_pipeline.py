#!/usr/bin/env python3
"""Benchmark build vs load time and memory for the data pipeline.

Usage:
  python benchmark_pipeline.py [--config config.yaml] [--mode build|load|both] [--train-epoch]

Measures:
  - Data build path: wall time + peak RAM
  - Data load path:  wall time + peak RAM
  - 1 training epoch (optional): wall time + peak RAM
  - 1 inference pass (optional): wall time + peak RAM

Designed for bitcoin-alpha 50K walks / walk_length=80.
Run before and after pipeline changes to compare.
"""

import argparse
import gc
import os
import sys
import time
import tracemalloc

import torch
import numpy as np
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.utils.config import load_config, get_seed
from src.data.prepare_data import prepare_data
from src.data.dataset_cache import cache_exists


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rss_mb() -> float:
    """Current resident set size in MB (Linux /proc)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _measure(label: str, fn):
    """Run fn(), returning (result, wall_s, peak_rss_mb, tracemalloc_peak_mb)."""
    gc.collect()
    rss_before = _rss_mb()
    tracemalloc.start()

    t0 = time.perf_counter()
    result = fn()
    wall_s = time.perf_counter() - t0

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_after = _rss_mb()
    peak_tracemalloc_mb = peak_bytes / (1024 * 1024)
    delta_rss_mb = rss_after - rss_before

    print(f"  [{label}]")
    print(f"    wall time : {wall_s:.2f}s")
    print(f"    RSS delta : {delta_rss_mb:+.1f} MB  (before={rss_before:.0f} MB, after={rss_after:.0f} MB)")
    print(f"    alloc peak: {peak_tracemalloc_mb:.1f} MB  (tracemalloc, Python heap only)")

    return result, wall_s, delta_rss_mb, peak_tracemalloc_mb


def _one_epoch(data_module, device):
    """Consume one full pass of the training DataLoader (no gradient)."""
    loader = data_module["train"]
    n_batches = 0
    for batch in loader:
        # Just move to device to measure PCIe transfer time too
        if isinstance(batch, (list, tuple)):
            _ = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        n_batches += 1
    return n_batches


def _one_inference_pass(data_module, device):
    """Consume val + test DataLoaders (simulates run_posthoc inference scan)."""
    total = 0
    for split in ("val", "test"):
        for batch in data_module[split]:
            if isinstance(batch, (list, tuple)):
                _ = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            total += 1
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode", default="both", choices=["build", "load", "both"],
        help="Which path to benchmark"
    )
    parser.add_argument(
        "--train-epoch", action="store_true",
        help="Also benchmark one DataLoader pass over the training split"
    )
    parser.add_argument(
        "--inference-pass", action="store_true",
        help="Also benchmark one DataLoader pass over val+test splits"
    )
    parser.add_argument(
        "overrides", nargs=argparse.REMAINDER,
        help="OmegaConf dotlist overrides, e.g. dataset.num_walks=50000"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)

    device = torch.device(
        "cuda:0" if cfg.training.use_cuda and torch.cuda.is_available() else "cpu"
    )

    dataset_cache_path = os.path.join(cfg.dataset.data_dir, "dataset_cache.pt")
    results = {}

    print("=" * 60)
    print(f"Benchmark: {cfg.dataset.name}")
    print(f"  num_walks       : {cfg.dataset.num_walks}")
    print(f"  max_walk_length : {cfg.dataset.max_walk_length}")
    print(f"  cache path      : {dataset_cache_path}")
    print(f"  cache exists    : {cache_exists(dataset_cache_path)}")
    print(f"  device          : {device}")
    print("=" * 60)

    # ── Build path ──────────────────────────────────────────────────────────
    if args.mode in ("build", "both"):
        print("\n--- BUILD PATH ---")
        # Force build: override use_cache to False (keep save as-is from config)
        cfg_build = OmegaConf.merge(cfg, OmegaConf.create({"preprocess": {"use_cache": False}}))

        data_module, wall_s, rss_delta, alloc_peak = _measure(
            "build", lambda: prepare_data(cfg_build)
        )
        results["build_wall_s"] = wall_s
        results["build_rss_delta_mb"] = rss_delta
        results["build_alloc_peak_mb"] = alloc_peak

        if args.train_epoch:
            print("\n--- BUILD → TRAIN EPOCH ---")
            _, tw, tr, ta = _measure("train_epoch (build)", lambda: _one_epoch(data_module, device))
            results["train_epoch_wall_s_build"] = tw
            results["train_epoch_rss_delta_mb_build"] = tr

        if args.inference_pass:
            print("\n--- BUILD → INFERENCE PASS ---")
            _, iw, ir, ia = _measure("inference_pass (build)", lambda: _one_inference_pass(data_module, device))
            results["inference_wall_s_build"] = iw

        del data_module
        gc.collect()

    # ── Load path ───────────────────────────────────────────────────────────
    if args.mode in ("load", "both"):
        if not cache_exists(dataset_cache_path):
            print(f"\n⚠  Cache not found at {dataset_cache_path} — skipping load benchmark.")
            print("   Run with --mode=build first to create it, or use preprocess.save=true")
        else:
            cache_size_mb = os.path.getsize(dataset_cache_path) / (1024 * 1024)
            print(f"\n--- LOAD PATH (cache {cache_size_mb:.1f} MB) ---")
            cfg_load = OmegaConf.merge(cfg, OmegaConf.create({"preprocess": {"use_cache": True}}))

            data_module, wall_s, rss_delta, alloc_peak = _measure(
                "load", lambda: prepare_data(cfg_load)
            )
            results["load_wall_s"] = wall_s
            results["load_rss_delta_mb"] = rss_delta
            results["load_alloc_peak_mb"] = alloc_peak
            results["cache_size_mb"] = cache_size_mb

            if args.train_epoch:
                print("\n--- LOAD → TRAIN EPOCH ---")
                _, tw, tr, ta = _measure("train_epoch (load)", lambda: _one_epoch(data_module, device))
                results["train_epoch_wall_s_load"] = tw
                results["train_epoch_rss_delta_mb_load"] = tr

            if args.inference_pass:
                print("\n--- LOAD → INFERENCE PASS ---")
                _, iw, ir, ia = _measure("inference_pass (load)", lambda: _one_inference_pass(data_module, device))
                results["inference_wall_s_load"] = iw

            del data_module
            gc.collect()

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:<40s}: {v:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
