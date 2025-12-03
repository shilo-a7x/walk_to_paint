#!/usr/bin/env python3
"""
Validate an experiment configuration and dataset files before a full run.

Usage:
  python scripts/validate_experiment.py --config configs/slashdot090221.yaml [--check-edges] [--max-edges 1000000]

This script will:
 - load/merge the config (with optional dotlist overrides)
 - resolve the outputs directories (without creating them by default)
 - check dataset file existence and print a small sample
 - optionally compute edge counts, self-loops, and multi-edge statistics
 - report GPU availability and device memory
 - warn if any CLI/base config values fall outside OPTUNA_RANGES (if available)

"""
import argparse
import os
import gzip
from types import SimpleNamespace
import sys

sys.path.insert(0, os.getcwd())
from src.utils.config import load_config
from src.utils.paths import resolve_outputs_dirs
import importlib
import shutil
import copy
import time

try:
    from optuna_run import OPTUNA_RANGES
except Exception:
    OPTUNA_RANGES = None


def open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="ignore")
    return open(path, "r", errors="ignore")


def sample_file(path, n=20):
    lines = []
    try:
        with open_text(path) as f:
            for i, l in enumerate(f):
                lines.append(l.rstrip("\n"))
                if i + 1 >= n:
                    break
    except Exception as e:
        lines = [f"Could not read file: {e}"]
    return lines


def analyze_edges(path, max_lines=None):
    total = 0
    self_loops = 0
    uv = {}
    try:
        with open_text(path) as f:
            for i, line in enumerate(f):
                ln = line.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split()
                if len(parts) < 3:
                    parts = ln.split(",")
                    if len(parts) < 3:
                        continue
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                except Exception:
                    # skip header or malformed
                    continue
                total += 1
                if u == v:
                    self_loops += 1
                key = (u, v)
                uv[key] = uv.get(key, 0) + 1
                if max_lines and total >= max_lines:
                    break
    except Exception as e:
        return {"error": str(e)}
    multi_pairs = sum(1 for k, c in uv.items() if c > 1)
    extra = total - len(uv)
    max_mult = max(uv.values()) if uv else 0
    return {
        "edges_read": total,
        "self_loops": self_loops,
        "unique_pairs": len(uv),
        "multi_pairs": multi_pairs,
        "extra_edges_due_to_multigraph": extra,
        "max_multiplicity": max_mult,
    }


def gpu_info():
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            devs = []
            for i in range(n):
                prop = torch.cuda.get_device_properties(i)
                total_mem = prop.total_memory
                devs.append(
                    {"id": i, "name": prop.name, "total_memory": int(total_mem)}
                )
            return {"available": True, "devices": devs}
        else:
            return {"available": False}
    except Exception as e:
        return {"available": False, "error": str(e)}


def disk_space_info(path: str):
    try:
        du = shutil.disk_usage(path)
        return {"total": du.total, "used": du.used, "free": du.free}
    except Exception as e:
        return {"error": str(e)}


def run_micro_batch(cfg, num_walks=1000, batch_size=8):
    """Run a small data preparation and fetch one batch to validate shapes and memory usage.

    Returns a dict with status and sample shapes or an error.
    """
    out = {"status": "ok"}
    try:
        # Make a safe copy and override small values
        tcfg = copy.deepcopy(cfg)
        try:
            tcfg.dataset.num_walks = num_walks
        except Exception:
            pass
        try:
            tcfg.training.batch_size = batch_size
        except Exception:
            pass

        # Import prepare_data lazily to avoid heavy imports if not requested
        from src.data.prepare_data import prepare_data

        t0 = time.time()
        data_module = prepare_data(tcfg)
        t1 = time.time()
        out["prepare_time_s"] = t1 - t0

        # Attempt to get one batch from train loader
        train_loader = data_module["train"]
        it = iter(train_loader)
        batch = next(it)

        # give a simple summary of batch
        def summarize(x):
            try:
                import torch

                if isinstance(x, torch.Tensor):
                    return {"shape": tuple(x.shape), "dtype": str(x.dtype)}
                elif isinstance(x, (list, tuple)):
                    return [summarize(e) for e in x[:3]]
            except Exception:
                return str(type(x))
            return str(type(x))

        out["batch_summary"] = summarize(batch)
    except Exception as e:
        out["status"] = "error"
        out["error"] = str(e)
    return out


def check_ranges(cfg):
    issues = []
    if OPTUNA_RANGES is None:
        return issues
    # Check dataset.num_walks against OPTUNA_RANGES if present
    try:
        nw = getattr(cfg.dataset, "num_walks", None)
        if nw is not None and "dataset.num_walks" in OPTUNA_RANGES:
            lo, hi = OPTUNA_RANGES["dataset.num_walks"]
            if not (lo <= int(nw) <= hi):
                issues.append(f"dataset.num_walks={nw} outside OPTUNA_RANGES {lo}-{hi}")
    except Exception:
        pass
    return issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--check-edges", action="store_true")
    p.add_argument("--max-edges", type=int, default=None, help="Cap edges to analyze")
    p.add_argument(
        "--micro-batch",
        action="store_true",
        help="Run a small data preparation and fetch one batch",
    )
    p.add_argument(
        "--micro-num-walks",
        type=int,
        default=1000,
        help="Num walks to use for micro-batch",
    )
    p.add_argument(
        "--micro-batch-size",
        type=int,
        default=8,
        help="Batch size to use for micro-batch",
    )
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)

    print("Loaded config:")
    print(" dataset.name:", getattr(cfg.dataset, "name", None))
    print(" dataset.data_dir:", getattr(cfg.dataset, "data_dir", None))
    print(" dataset.edge_list_file:", getattr(cfg.dataset, "edge_list_file", None))
    print(
        " training.exp_name:", getattr(getattr(cfg, "training", {}), "exp_name", None)
    )
    print(
        " training.exp_note:", getattr(getattr(cfg, "training", {}), "exp_note", None)
    )

    resolved = resolve_outputs_dirs(cfg, make_dirs=False)
    print("\nResolved outputs (will be created under):")
    for k, v in resolved.items():
        print(f"  {k}: {v}")

    # dataset file
    data_path = os.path.join(cfg.dataset.data_dir, cfg.dataset.edge_list_file)
    print("\nDataset file path:", data_path)
    print("Exists:", os.path.exists(data_path))
    print("\nSample lines:")
    for l in sample_file(data_path, n=20):
        print("  ", l)

    if args.check_edges:
        print("\nAnalyzing edges (may take a while)")
        stats = analyze_edges(data_path, max_lines=args.max_edges)
        print("Edge stats:")
        for k, v in stats.items():
            print("  ", k, v)

    print("\nGPU info:")
    gi = gpu_info()
    print(gi)

    # Disk usage for the experiment base
    try:
        base = getattr(getattr(cfg, "paths", {}), "base_outputs_dir", None) or "outputs"
    except Exception:
        base = "outputs"
    ds = disk_space_info(base)
    print("\nDisk usage for base outputs dir:", base)
    print(ds)

    # ranges check
    issues = check_ranges(cfg)
    if issues:
        print("\nWarnings:")
        for it in issues:
            print("  ", it)
    else:
        print("\nNo obvious range issues detected.")

    # Optional micro-batch validation
    if args.micro_batch:
        print("\nRunning micro-batch test (small data gen + one batch)")
        mb = run_micro_batch(
            cfg, num_walks=args.micro_num_walks, batch_size=args.micro_batch_size
        )
        print("Micro-batch result:")
        for k, v in mb.items():
            print("  ", k, v)

    print("\nValidation finished.")


if __name__ == "__main__":
    main()
