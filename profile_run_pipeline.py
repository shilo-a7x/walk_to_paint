import argparse
import json
import os
import random
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback

from src.data.prepare_data import prepare_data
from src.model.lit_model import LitEdgeClassifier
from src.training import callbacks as training_callbacks
from src.training import train as train_module
from src.utils.config import get_seed, load_config
from src.utils.paths import resolve_outputs_dirs


HOOK_METHODS = [
    "setup",
    "teardown",
    "on_fit_start",
    "on_fit_end",
    "on_train_start",
    "on_train_end",
    "on_validation_start",
    "on_validation_end",
    "on_test_start",
    "on_test_end",
    "on_train_epoch_start",
    "on_train_epoch_end",
    "on_validation_epoch_start",
    "on_validation_epoch_end",
    "on_test_epoch_start",
    "on_test_epoch_end",
    "on_train_batch_start",
    "on_train_batch_end",
    "on_validation_batch_start",
    "on_validation_batch_end",
    "on_test_batch_start",
    "on_test_batch_end",
    "on_before_backward",
    "on_after_backward",
    "on_before_optimizer_step",
    "on_before_zero_grad",
]


class PipelineTimer:
    def __init__(self):
        self._starts = {}
        self.values = {}

    def start(self, name: str):
        self._starts[name] = time.perf_counter()

    def stop(self, name: str):
        start = self._starts.pop(name, None)
        if start is None:
            return
        self.values[name] = self.values.get(name, 0.0) + (time.perf_counter() - start)


class TrainingRuntimeProfiler(Callback):
    def __init__(self):
        super().__init__()
        self.train_batch_times = []
        self.train_wait_times = []
        self.val_batch_times = []
        self.val_wait_times = []
        self.test_batch_times = []
        self.test_wait_times = []

        self.train_to_val_transition_times = []
        self.val_to_train_transition_times = []
        self.val_to_test_transition_times = []

        self.epoch_train_times = []
        self.epoch_val_times = []

        self._train_batch_start = None
        self._val_batch_start = None
        self._test_batch_start = None

        self._prev_train_batch_end = None
        self._prev_val_batch_end = None
        self._prev_test_batch_end = None

        self._train_epoch_start = None
        self._val_epoch_start = None
        self._train_epoch_end = None
        self._val_epoch_end = None

    def on_train_epoch_start(self, trainer, pl_module):
        now = time.perf_counter()
        if self._val_epoch_end is not None:
            self.val_to_train_transition_times.append(now - self._val_epoch_end)
            self._val_epoch_end = None
        self._prev_train_batch_end = None
        self._train_epoch_start = now

    def on_train_epoch_end(self, trainer, pl_module):
        now = time.perf_counter()
        if self._train_epoch_start is not None:
            self.epoch_train_times.append(now - self._train_epoch_start)
        self._train_epoch_end = now

    def on_validation_epoch_start(self, trainer, pl_module):
        now = time.perf_counter()
        if self._train_epoch_end is not None:
            self.train_to_val_transition_times.append(now - self._train_epoch_end)
            self._train_epoch_end = None
        self._prev_val_batch_end = None
        self._val_epoch_start = now

    def on_validation_epoch_end(self, trainer, pl_module):
        now = time.perf_counter()
        if self._val_epoch_start is not None:
            self.epoch_val_times.append(now - self._val_epoch_start)
        self._val_epoch_end = now

    def on_test_start(self, trainer, pl_module):
        now = time.perf_counter()
        if self._val_epoch_end is not None:
            self.val_to_test_transition_times.append(now - self._val_epoch_end)
            self._val_epoch_end = None
        self._prev_test_batch_end = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        if self._prev_train_batch_end is not None:
            self.train_wait_times.append(now - self._prev_train_batch_end)
        self._train_batch_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.perf_counter()
        if self._train_batch_start is not None:
            self.train_batch_times.append(now - self._train_batch_start)
        self._prev_train_batch_end = now

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        now = time.perf_counter()
        if self._prev_val_batch_end is not None:
            self.val_wait_times.append(now - self._prev_val_batch_end)
        self._val_batch_start = now

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        now = time.perf_counter()
        if self._val_batch_start is not None:
            self.val_batch_times.append(now - self._val_batch_start)
        self._prev_val_batch_end = now

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        now = time.perf_counter()
        if self._prev_test_batch_end is not None:
            self.test_wait_times.append(now - self._prev_test_batch_end)
        self._test_batch_start = now

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        now = time.perf_counter()
        if self._test_batch_start is not None:
            self.test_batch_times.append(now - self._test_batch_start)
        self._prev_test_batch_end = now


def _summary_stats(values):
    if not values:
        return {"count": 0, "sum_s": 0.0, "mean_s": 0.0, "p50_s": 0.0, "p95_s": 0.0, "max_s": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "sum_s": float(arr.sum()),
        "mean_s": float(arr.mean()),
        "p50_s": float(np.percentile(arr, 50)),
        "p95_s": float(np.percentile(arr, 95)),
        "max_s": float(arr.max()),
    }


def _patch_method_timing(timing_store: Dict[str, list], cls: Any, method_name: str):
    if not hasattr(cls, method_name):
        return None

    original = getattr(cls, method_name)
    key = f"{cls.__name__}.{method_name}"

    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            timing_store[key].append(time.perf_counter() - start)

    setattr(cls, method_name, wrapped)
    return original


def _restore_method(cls: Any, method_name: str, original: Any):
    if original is not None:
        setattr(cls, method_name, original)


def _bind_timed_instance_method(instance: Any, method_name: str, timing_store: Dict[str, list]):
    if not hasattr(instance, method_name):
        return None

    original = getattr(instance, method_name)
    if not callable(original):
        return None

    key = f"{instance.__class__.__name__}.{method_name}"

    def wrapped(_self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            timing_store[key].append(time.perf_counter() - start)

    setattr(instance, method_name, types.MethodType(wrapped, instance))
    return original


def _install_callback_hook_timers(callbacks, timing_store: Dict[str, list]):
    restores = []
    for cb in callbacks:
        for hook in HOOK_METHODS:
            original = _bind_timed_instance_method(cb, hook, timing_store)
            if original is not None:
                restores.append((cb, hook, original))
    return restores


def _restore_instance_methods(restores):
    for instance, method_name, original in restores:
        setattr(instance, method_name, original)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile full run.py pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/profiling",
        help="Directory to write profile artifacts",
    )
    parser.add_argument(
        "--disable-custom-callbacks",
        action="store_true",
        help="Disable PerEpochPredictionSaver and PerEpochTestRunner to isolate standard training overhead",
    )
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Config overrides")
    return parser.parse_args()


def _print_top_level_breakdown(timings: Dict[str, float]):
    print("\n=== Top-level run.py phase timings ===")
    total = sum(timings.values())
    for name, value in sorted(timings.items(), key=lambda kv: kv[1], reverse=True):
        pct = (100.0 * value / total) if total > 0 else 0.0
        print(f"- {name:<28} {value:>10.3f}s  ({pct:>6.2f}%)")
    print(f"- {'TOTAL':<28} {total:>10.3f}s")


def _print_training_breakdown(runtime_profiler: TrainingRuntimeProfiler, callback_method_times: Dict[str, list]):
    print("\n=== Training runtime breakdown ===")

    train_batch = _summary_stats(runtime_profiler.train_batch_times)
    train_wait = _summary_stats(runtime_profiler.train_wait_times)
    val_batch = _summary_stats(runtime_profiler.val_batch_times)
    val_wait = _summary_stats(runtime_profiler.val_wait_times)
    test_batch = _summary_stats(runtime_profiler.test_batch_times)
    test_wait = _summary_stats(runtime_profiler.test_wait_times)
    train_to_val = _summary_stats(runtime_profiler.train_to_val_transition_times)
    val_to_train = _summary_stats(runtime_profiler.val_to_train_transition_times)
    val_to_test = _summary_stats(runtime_profiler.val_to_test_transition_times)

    print("Train batches:")
    print(f"  batch_time mean/p95: {train_batch['mean_s']:.4f}s / {train_batch['p95_s']:.4f}s")
    print(f"  data_wait  mean/p95: {train_wait['mean_s']:.4f}s / {train_wait['p95_s']:.4f}s")
    print(f"  samples: {train_batch['count']} batches")

    print("Validation batches:")
    print(f"  batch_time mean/p95: {val_batch['mean_s']:.4f}s / {val_batch['p95_s']:.4f}s")
    print(f"  data_wait  mean/p95: {val_wait['mean_s']:.4f}s / {val_wait['p95_s']:.4f}s")
    print(f"  samples: {val_batch['count']} batches")

    print("Test batches:")
    print(f"  batch_time mean/p95: {test_batch['mean_s']:.4f}s / {test_batch['p95_s']:.4f}s")
    print(f"  data_wait  mean/p95: {test_wait['mean_s']:.4f}s / {test_wait['p95_s']:.4f}s")
    print(f"  samples: {test_batch['count']} batches")

    print("Phase transitions:")
    print(f"  train->val mean/p95: {train_to_val['mean_s']:.4f}s / {train_to_val['p95_s']:.4f}s ({train_to_val['count']} transitions)")
    print(f"  val->train mean/p95: {val_to_train['mean_s']:.4f}s / {val_to_train['p95_s']:.4f}s ({val_to_train['count']} transitions)")
    print(f"  val->test  mean/p95: {val_to_test['mean_s']:.4f}s / {val_to_test['p95_s']:.4f}s ({val_to_test['count']} transitions)")

    if callback_method_times:
        print("\nCallback/hook overhead totals:")
        totals = {
            key: float(sum(values))
            for key, values in callback_method_times.items()
            if values
        }
        for key, total_s in sorted(totals.items(), key=lambda kv: kv[1], reverse=True):
            calls = len(callback_method_times[key])
            mean_s = total_s / calls if calls else 0.0
            print(f"  - {key:<42} total={total_s:>9.3f}s  calls={calls:>4d}  mean={mean_s:.4f}s")


def _top_n_detailed_timings(timing_store: Dict[str, list], top_n: int = 40):
    rows = []
    for key, values in timing_store.items():
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        rows.append(
            {
                "name": key,
                "calls": int(arr.size),
                "total_s": float(arr.sum()),
                "mean_s": float(arr.mean()),
                "p95_s": float(np.percentile(arr, 95)),
                "max_s": float(arr.max()),
            }
        )
    rows.sort(key=lambda x: x["total_s"], reverse=True)
    return rows[:top_n], rows


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_timer = PipelineTimer()
    callback_method_times = defaultdict(list)
    detailed_call_times = defaultdict(list)

    # Patch callback-heavy methods to capture overhead inside training.
    patches = []
    methods_to_patch = [
        (training_callbacks.PerEpochPredictionSaver, "_extract_predictions"),
        (training_callbacks.PerEpochPredictionSaver, "_save_predictions"),
        (training_callbacks.PerEpochPredictionSaver, "on_train_epoch_end"),
        (training_callbacks.PerEpochPredictionSaver, "on_validation_epoch_end"),
        (training_callbacks.PerEpochPredictionSaver, "on_test_epoch_end"),
        (training_callbacks.PerEpochTestRunner, "on_validation_epoch_end"),
        (LitEdgeClassifier, "on_train_epoch_end"),
        (LitEdgeClassifier, "on_validation_epoch_end"),
        (LitEdgeClassifier, "on_test_epoch_end"),
    ]
    for cls, method_name in methods_to_patch:
        original = _patch_method_timing(callback_method_times, cls, method_name)
        patches.append((cls, method_name, original))

    runtime_profiler = TrainingRuntimeProfiler()

    # Monkeypatch Trainer used inside src.training.train.train_model to inject runtime profiler callback
    # and detailed per-callback hook timing.
    original_trainer_ctor = train_module.Trainer

    def profiled_trainer_ctor(*t_args, **t_kwargs):
        callbacks = list(t_kwargs.get("callbacks", []))
        callbacks.append(runtime_profiler)
        hook_restores = _install_callback_hook_timers(callbacks, detailed_call_times)
        t_kwargs["callbacks"] = callbacks
        trainer = original_trainer_ctor(*t_args, **t_kwargs)
        trainer._profile_hook_restores = hook_restores
        return trainer

    train_module.Trainer = profiled_trainer_ctor

    try:
        pipeline_timer.start("load_config")
        cfg = load_config(args.config, overrides=args.overrides)
        pipeline_timer.stop("load_config")

        if args.disable_custom_callbacks:
            if not hasattr(cfg.training, "callbacks") or cfg.training.callbacks is None:
                cfg.training.callbacks = {}
            cfg.training.callbacks.enable_prediction_saver = False
            cfg.training.callbacks.enable_per_epoch_test_runner = False

        # Matches run.py behavior
        pipeline_timer.start("auto_exp_name")
        try:
            current_exp = getattr(cfg.training, "exp_name", None)
        except Exception:
            current_exp = None
        if not current_exp or current_exp in ("walk_to_paint_experiment", "experiment"):
            try:
                note = getattr(cfg.training, "exp_note", None)
            except Exception:
                note = None
            note_part = f"-{note}" if note else ""
            cfg.training.exp_name = f"{getattr(cfg.dataset, 'name', 'dataset')}-run{note_part}"
        pipeline_timer.stop("auto_exp_name")

        pipeline_timer.start("seed_setup")
        try:
            seed = get_seed(cfg)
            seed_everything(seed, workers=True)
            random.seed(seed)
            np.random.seed(seed)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        pipeline_timer.stop("seed_setup")

        pipeline_timer.start("resolve_outputs_dirs")
        resolved = resolve_outputs_dirs(cfg)
        pipeline_timer.stop("resolve_outputs_dirs")

        pipeline_timer.start("matmul_precision")
        try:
            if cfg.training.use_cuda and torch.cuda.is_available():
                precision = getattr(cfg, "float32_precision", "medium")
                torch.set_float32_matmul_precision(precision)
        except Exception:
            pass
        pipeline_timer.stop("matmul_precision")

        pipeline_timer.start("device_setup")
        if cfg.training.use_cuda and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        pipeline_timer.stop("device_setup")

        pipeline_timer.start("prepare_data")
        data_module = prepare_data(cfg)
        pipeline_timer.stop("prepare_data")

        pipeline_timer.start("train_model_total")
        train_module.train_model(cfg, data_module)
        pipeline_timer.stop("train_model_total")

        top_detailed, all_detailed = _top_n_detailed_timings(detailed_call_times)

        profile = {
            "config": {
                "config_path": args.config,
                "overrides": args.overrides,
                "dataset": getattr(cfg.dataset, "name", None),
                "exp_name": getattr(cfg.training, "exp_name", None),
                "outputs_exp_dir": resolved.get("exp_dir") if isinstance(resolved, dict) else None,
                "device": "cuda" if (cfg.training.use_cuda and torch.cuda.is_available()) else "cpu",
                "disable_custom_callbacks": args.disable_custom_callbacks,
            },
            "top_level_seconds": pipeline_timer.values,
            "train_runtime": {
                "train_batch": _summary_stats(runtime_profiler.train_batch_times),
                "train_wait": _summary_stats(runtime_profiler.train_wait_times),
                "val_batch": _summary_stats(runtime_profiler.val_batch_times),
                "val_wait": _summary_stats(runtime_profiler.val_wait_times),
                "test_batch": _summary_stats(runtime_profiler.test_batch_times),
                "test_wait": _summary_stats(runtime_profiler.test_wait_times),
                "train_to_val_transition": _summary_stats(runtime_profiler.train_to_val_transition_times),
                "val_to_train_transition": _summary_stats(runtime_profiler.val_to_train_transition_times),
                "val_to_test_transition": _summary_stats(runtime_profiler.val_to_test_transition_times),
                "train_epoch": _summary_stats(runtime_profiler.epoch_train_times),
                "val_epoch": _summary_stats(runtime_profiler.epoch_val_times),
            },
            "callback_method_seconds": {
                key: {
                    "calls": len(values),
                    "total_s": float(sum(values)),
                    "mean_s": float(np.mean(values)) if values else 0.0,
                    "p95_s": float(np.percentile(values, 95)) if values else 0.0,
                }
                for key, values in callback_method_times.items()
            },
            "detailed_function_seconds": all_detailed,
        }

        json_path = out_dir / "run_pipeline_profile.json"
        txt_path = out_dir / "run_pipeline_profile.txt"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("run.py pipeline profile\n")
            f.write("=" * 80 + "\n")
            total = sum(pipeline_timer.values.values())
            f.write("Top-level timings:\n")
            for name, value in sorted(pipeline_timer.values.items(), key=lambda kv: kv[1], reverse=True):
                pct = (100.0 * value / total) if total > 0 else 0.0
                f.write(f"- {name:<28} {value:>10.3f}s  ({pct:>6.2f}%)\n")
            f.write(f"- {'TOTAL':<28} {total:>10.3f}s\n\n")

            f.write("Training runtime:\n")
            f.write(f"- train_batch mean/p95: {_summary_stats(runtime_profiler.train_batch_times)['mean_s']:.4f}s / {_summary_stats(runtime_profiler.train_batch_times)['p95_s']:.4f}s\n")
            f.write(f"- train_wait  mean/p95: {_summary_stats(runtime_profiler.train_wait_times)['mean_s']:.4f}s / {_summary_stats(runtime_profiler.train_wait_times)['p95_s']:.4f}s\n")
            f.write(f"- val_batch   mean/p95: {_summary_stats(runtime_profiler.val_batch_times)['mean_s']:.4f}s / {_summary_stats(runtime_profiler.val_batch_times)['p95_s']:.4f}s\n")
            f.write(f"- val_wait    mean/p95: {_summary_stats(runtime_profiler.val_wait_times)['mean_s']:.4f}s / {_summary_stats(runtime_profiler.val_wait_times)['p95_s']:.4f}s\n")
            f.write(f"- test_batch  mean/p95: {_summary_stats(runtime_profiler.test_batch_times)['mean_s']:.4f}s / {_summary_stats(runtime_profiler.test_batch_times)['p95_s']:.4f}s\n")
            f.write(f"- test_wait   mean/p95: {_summary_stats(runtime_profiler.test_wait_times)['mean_s']:.4f}s / {_summary_stats(runtime_profiler.test_wait_times)['p95_s']:.4f}s\n\n")

            f.write("Phase transitions:\n")
            f.write(f"- train_to_val mean/p95: {_summary_stats(runtime_profiler.train_to_val_transition_times)['mean_s']:.4f}s / {_summary_stats(runtime_profiler.train_to_val_transition_times)['p95_s']:.4f}s\n")
            f.write(f"- val_to_train mean/p95: {_summary_stats(runtime_profiler.val_to_train_transition_times)['mean_s']:.4f}s / {_summary_stats(runtime_profiler.val_to_train_transition_times)['p95_s']:.4f}s\n")
            f.write(f"- val_to_test  mean/p95: {_summary_stats(runtime_profiler.val_to_test_transition_times)['mean_s']:.4f}s / {_summary_stats(runtime_profiler.val_to_test_transition_times)['p95_s']:.4f}s\n\n")

            f.write("Callback/hook overhead:\n")
            totals = {
                key: float(sum(values))
                for key, values in callback_method_times.items()
                if values
            }
            for key, total_s in sorted(totals.items(), key=lambda kv: kv[1], reverse=True):
                calls = len(callback_method_times[key])
                mean_s = total_s / calls if calls else 0.0
                f.write(f"- {key:<42} total={total_s:>9.3f}s  calls={calls:>4d}  mean={mean_s:.4f}s\n")

            f.write("\nTop detailed function timings (training stage):\n")
            for row in top_detailed:
                f.write(
                    f"- {row['name']:<42} total={row['total_s']:>9.3f}s "
                    f"calls={row['calls']:>6d} mean={row['mean_s']:.6f}s "
                    f"p95={row['p95_s']:.6f}s max={row['max_s']:.6f}s\n"
                )

        _print_top_level_breakdown(pipeline_timer.values)
        _print_training_breakdown(runtime_profiler, callback_method_times)

        print(f"\nProfile artifacts written:")
        print(f"- {json_path}")
        print(f"- {txt_path}")

    finally:
        train_module.Trainer = original_trainer_ctor
        for cls, method_name, original in patches:
            _restore_method(cls, method_name, original)


if __name__ == "__main__":
    main()
