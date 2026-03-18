#!/usr/bin/env python3
"""Run isolated incremental transformer experiments (baseline + anti-overfit ablations).

Protocol guarantees:
- Never writes into original `data/<dataset>` artifacts (.pt safety check before/after).
- Uses an isolated temporary `dataset.data_dir` that only contains copied raw edge file + cache.
- Reuses one shared cache for fast experiments (500K walks, max length 80 by default).
- Forces optional callbacks OFF for all runs.
- Uses CUDA device 1 by default.
- Writes each run to a separate output directory with deterministic naming.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
FIXED_NUM_WALKS = 500_000


@dataclass(frozen=True)
class ExperimentSpec:
    exp_id: str
    change: str
    overrides: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incremental transformer experiments with strict isolation"
    )
    parser.add_argument(
        "--dataset",
        default="wiki-rfa",
        help="Dataset key (must match configs/<dataset>.yaml)",
    )
    parser.add_argument("--seed", type=int, default=42, help="reproducibility.seed")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs (default: keep dataset YAML value)",
    )
    parser.add_argument("--device", type=int, default=1, help="CUDA device id")
    parser.add_argument(
        "--num-walks",
        type=int,
        default=None,
        help=f"Optional explicit value; must be {FIXED_NUM_WALKS} (enforced for speed/delta consistency)",
    )
    parser.add_argument(
        "--max-walk-length",
        type=int,
        default=None,
        help="Override dataset.max_walk_length (default: keep dataset YAML value)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training.batch_size (default: keep dataset YAML value)",
    )
    parser.add_argument(
        "--fast-baseline",
        action="store_true",
        help="Shortcut for fast iteration: max_walk_length=80, batch_size=1024 (num_walks is already fixed to 500000)",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/transformer_incremental",
        help="Root for suite outputs/results",
    )
    parser.add_argument(
        "--tmp-root",
        default="tmp/transformer_incremental",
        help="Root for isolated temporary dataset dirs",
    )
    parser.add_argument(
        "--run-ids",
        default="E0_BASELINE,E1_SMALLER_MODEL,E2_DROPOUT_UP,E3_WEIGHT_DECAY_UP,E4_EARLY_STOP_TIGHTER,E5_LR_DOWN",
        help="Comma-separated subset of experiment IDs to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs and exit without executing training",
    )
    return parser.parse_args()


def _load_dataset_cfg(dataset: str) -> Dict[str, object]:
    ds_cfg_path = ROOT / "configs" / f"{dataset}.yaml"
    if not ds_cfg_path.exists():
        raise FileNotFoundError(f"Missing dataset config: {ds_cfg_path}")

    cfg = OmegaConf.load(str(ds_cfg_path))
    data_dir = ROOT / str(cfg.dataset.data_dir)
    edge_list_file = str(cfg.dataset.edge_list_file)
    raw_src = data_dir / edge_list_file
    if not raw_src.exists():
        raise FileNotFoundError(f"Missing raw edge list file: {raw_src}")

    return {
        "data_dir": data_dir,
        "edge_list_file": edge_list_file,
        "raw_src": raw_src,
        "base_lr": float(cfg.training.lr),
        "base_epochs": int(cfg.training.epochs),
        "base_batch_size": int(cfg.training.batch_size),
        "base_num_walks": int(cfg.dataset.num_walks),
        "base_max_walk_length": int(cfg.dataset.max_walk_length),
        "base_weight_decay": float(cfg.training.weight_decay),
        "base_patience": int(cfg.training.early_stopping_patience),
        "base_dropout": float(cfg.model.dropout),
        "base_embedding_dim": int(cfg.model.embedding_dim),
        "base_hidden_dim": int(cfg.model.hidden_dim),
        "base_nhead": int(cfg.model.nhead),
        "base_nlayers": int(cfg.model.nlayers),
    }


def _snapshot_pt_artifacts(dataset_dir: Path) -> Dict[str, Tuple[int, float]]:
    if not dataset_dir.exists():
        return {}
    snapshot: Dict[str, Tuple[int, float]] = {}
    for path in sorted(dataset_dir.glob("*.pt")):
        stat = path.stat()
        snapshot[str(path)] = (int(stat.st_size), float(stat.st_mtime))
    return snapshot


def _compare_snapshots(
    before: Dict[str, Tuple[int, float]],
    after: Dict[str, Tuple[int, float]],
) -> Tuple[bool, List[str]]:
    changed: List[str] = []
    before_keys = set(before)
    after_keys = set(after)

    for path in sorted(after_keys - before_keys):
        changed.append(f"ADDED: {path}")
    for path in sorted(before_keys - after_keys):
        changed.append(f"REMOVED: {path}")

    for path in sorted(before_keys & after_keys):
        if before[path] != after[path]:
            changed.append(f"MODIFIED: {path}")

    return (len(changed) == 0, changed)


def _copy_raw_to_temp(raw_src: Path, tmp_data_dir: Path) -> None:
    tmp_data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_src, tmp_data_dir / raw_src.name)


def _extract_scalar_series(event_file: Path, tag: str) -> List[Tuple[int, float]]:
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    if tag not in scalar_tags:
        return []
    return [(int(e.step), float(e.value)) for e in accumulator.Scalars(tag)]


def _pick_best_epoch(val_auc_series: List[Tuple[int, float]]) -> Optional[int]:
    if not val_auc_series:
        return None
    return int(max(val_auc_series, key=lambda x: x[1])[0])


def _value_at_step(series: List[Tuple[int, float]], step: Optional[int]) -> Optional[float]:
    if not series:
        return None
    if step is None:
        return float(series[-1][1])
    mapping = {s: v for s, v in series}
    if step in mapping:
        return float(mapping[step])
    return float(series[-1][1])


def _safe_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a - b)


def _find_event_file_for_exp(log_dir: Path, dataset: str, exp_name: str) -> Optional[Path]:
    logger_root = log_dir / f"{dataset}-{exp_name}"
    if not logger_root.exists():
        return None
    event_files = sorted(
        logger_root.glob("**/events.out.tfevents.*"),
        key=lambda p: p.stat().st_mtime,
    )
    return event_files[-1] if event_files else None


def _round_or_blank(value: Optional[float], digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return ""
    return f"{value:.{digits}f}"


def _bool_verdict(
    row: Dict[str, object],
    baseline: Optional[Dict[str, object]],
) -> str:
    if int(row.get("return_code", 1)) != 0:
        return "Drop"
    if baseline is None:
        return "Baseline"

    val_auc = row.get("val_auc")
    test_auc = row.get("test_auc")
    gap_loss = row.get("gap_loss")
    gap_auc = row.get("gap_auc")

    base_val_auc = baseline.get("val_auc")
    base_test_auc = baseline.get("test_auc")
    base_gap_loss = baseline.get("gap_loss")
    base_gap_auc = baseline.get("gap_auc")

    if None in (val_auc, test_auc, gap_loss, gap_auc, base_val_auc, base_test_auc, base_gap_loss, base_gap_auc):
        return "Maybe"

    better_metrics = (float(val_auc) >= float(base_val_auc)) and (float(test_auc) >= float(base_test_auc))
    smaller_gaps = (float(gap_loss) <= float(base_gap_loss)) and (float(gap_auc) <= float(base_gap_auc))

    if better_metrics and smaller_gaps:
        return "Keep"
    if better_metrics or smaller_gaps:
        return "Maybe"
    return "Drop"


def _build_experiments(base_cfg: Dict[str, object]) -> List[ExperimentSpec]:
    base_wd = float(base_cfg["base_weight_decay"])
    base_lr = float(base_cfg["base_lr"])
    base_patience = int(base_cfg["base_patience"])

    return [
        ExperimentSpec(
            exp_id="E0_BASELINE",
            change="Baseline = current optimized dataset YAML values",
            overrides=[],
        ),
        ExperimentSpec(
            exp_id="E1_SMALLER_MODEL",
            change="Capacity down: embedding/hidden/layers reduced",
            overrides=[
                "model.embedding_dim=16",
                "model.hidden_dim=16",
                "model.nhead=4",
                "model.nlayers=2",
            ],
        ),
        ExperimentSpec(
            exp_id="E2_DROPOUT_UP",
            change="Regularization up: dropout=0.2",
            overrides=["model.dropout=0.2"],
        ),
        ExperimentSpec(
            exp_id="E3_WEIGHT_DECAY_UP",
            change="Regularization up: weight_decay x10",
            overrides=[f"training.weight_decay={max(base_wd * 10.0, 1e-6):.10f}"],
        ),
        ExperimentSpec(
            exp_id="E4_EARLY_STOP_TIGHTER",
            change="Training control: tighter early stopping",
            overrides=[f"training.early_stopping_patience={max(3, min(base_patience, 5))}"],
        ),
        ExperimentSpec(
            exp_id="E5_LR_DOWN",
            change="Training control: lr halved",
            overrides=[f"training.lr={max(base_lr * 0.5, 1e-6):.10f}"],
        ),
    ]


def _run_one(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    resolved: Dict[str, int],
    suite_dir: Path,
    tmp_data_dir: Path,
    edge_list_file: str,
) -> Dict[str, object]:
    runs_root = suite_dir / "runs"
    logs_root = runs_root / args.dataset / spec.exp_id / "logs"
    run_dir = suite_dir / "artifacts" / spec.exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_dir / "run.log"

    cmd = [
        sys.executable,
        "run.py",
        "--device",
        str(args.device),
        f"dataset.name={args.dataset}",
        f"dataset.data_dir={tmp_data_dir.as_posix()}",
        f"dataset.edge_list_file={edge_list_file}",
        f"dataset.num_walks={resolved['num_walks']}",
        f"dataset.max_walk_length={resolved['max_walk_length']}",
        f"training.batch_size={resolved['batch_size']}",
        f"training.epochs={resolved['epochs']}",
        f"reproducibility.seed={args.seed}",
        f"training.exp_name={spec.exp_id}",
        f"paths.base_outputs_dir={(suite_dir / 'runs').as_posix()}",
        "paths.use_dataset_outputs=true",
        "paths.append_timestamp=false",
        "preprocess.use_cache=true",
        "preprocess.save=true",
        "training.callbacks.enable_prediction_saver=false",
        "training.callbacks.enable_per_epoch_test_runner=false",
        *spec.overrides,
    ]

    started = time.time()
    with run_log.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    runtime_s = float(time.time() - started)

    event_file = _find_event_file_for_exp(logs_root, args.dataset, spec.exp_id)

    best_epoch: Optional[int] = None
    train_auc = val_auc = test_auc = None
    train_loss = val_loss = None

    if event_file is not None:
        val_auc_series = _extract_scalar_series(event_file, "val_auc_epoch")
        train_auc_series = _extract_scalar_series(event_file, "train_auc_epoch")
        test_auc_series = _extract_scalar_series(event_file, "test_auc_epoch")
        train_loss_series = _extract_scalar_series(event_file, "train_loss_epoch")
        val_loss_series = _extract_scalar_series(event_file, "val_loss_epoch")

        best_epoch = _pick_best_epoch(val_auc_series)
        train_auc = _value_at_step(train_auc_series, best_epoch)
        val_auc = _value_at_step(val_auc_series, best_epoch)
        test_auc = _value_at_step(test_auc_series, best_epoch)
        train_loss = _value_at_step(train_loss_series, best_epoch)
        val_loss = _value_at_step(val_loss_series, best_epoch)

    row: Dict[str, object] = {
        "exp_id": spec.exp_id,
        "change": spec.change,
        "seed": int(args.seed),
        "best_epoch": best_epoch,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "train_auc": train_auc,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "gap_loss": _safe_diff(val_loss, train_loss),
        "gap_auc": _safe_diff(train_auc, val_auc),
        "runtime_per_epoch_min": runtime_s / 60.0 / max(resolved["epochs"], 1),
        "runtime_total_min": runtime_s / 60.0,
        "return_code": int(proc.returncode),
        "status": "ok" if proc.returncode == 0 else "failed",
        "run_log": str(run_log),
        "event_file": str(event_file) if event_file else "",
        "tmp_data_dir": str(tmp_data_dir),
        "cache_path": str(tmp_data_dir / "dataset_cache.pt"),
        "num_walks": int(resolved["num_walks"]),
        "max_walk_length": int(resolved["max_walk_length"]),
        "batch_size": int(resolved["batch_size"]),
        "epochs": int(resolved["epochs"]),
    }
    return row


def _write_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    fields = [
        "exp_id",
        "change",
        "seed",
        "best_epoch",
        "val_auc",
        "test_auc",
        "train_auc",
        "train_loss",
        "val_loss",
        "gap_loss",
        "gap_auc",
        "runtime_per_epoch_min",
        "runtime_total_min",
        "verdict",
        "status",
        "return_code",
        "run_log",
        "event_file",
        "tmp_data_dir",
        "cache_path",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown_summary(
    rows: List[Dict[str, object]],
    out_md: Path,
    args: argparse.Namespace,
    suite_dir: Path,
    data_safety_ok: bool,
    data_safety_issues: List[str],
    resolved: Dict[str, int],
) -> None:
    lines: List[str] = []
    lines.append("# Transformer Incremental Experiments")
    lines.append("")
    lines.append("## Protocol")
    lines.append(f"- Dataset: `{args.dataset}`")
    lines.append(f"- Seed: `{args.seed}`")
    lines.append(f"- Device: `cuda:{args.device}`")
    lines.append(
        "- Effective budget: "
        f"`num_walks={resolved['num_walks']}`, "
        f"`max_walk_length={resolved['max_walk_length']}`, "
        f"`batch_size={resolved['batch_size']}`, "
        f"`epochs={resolved['epochs']}`"
    )
    lines.append(f"- Shared isolated tmp dir: `{(suite_dir / 'tmp_data').as_posix()}`")
    lines.append("- Optional callbacks forced OFF for all runs")
    lines.append("")
    lines.append("## Data Safety Check")
    lines.append(f"- Original `data/<dataset>` `.pt` artifacts unchanged: `{data_safety_ok}`")
    if data_safety_issues:
        lines.append("- Changes detected:")
        for issue in data_safety_issues:
            lines.append(f"  - {issue}")
    lines.append("")
    lines.append("## Results")
    lines.append("| Exp ID | Change | Seed | Best Epoch | Val AUC | Test AUC | Train Loss | Val Loss | Gap Loss | Gap AUC | Runtime/Epoch (min) | Verdict |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            "| {exp_id} | {change} | {seed} | {best_epoch} | {val_auc} | {test_auc} | {train_loss} | {val_loss} | {gap_loss} | {gap_auc} | {runtime} | {verdict} |".format(
                exp_id=row["exp_id"],
                change=row["change"],
                seed=row["seed"],
                best_epoch=("" if row.get("best_epoch") is None else row.get("best_epoch")),
                val_auc=_round_or_blank(row.get("val_auc")),
                test_auc=_round_or_blank(row.get("test_auc")),
                train_loss=_round_or_blank(row.get("train_loss")),
                val_loss=_round_or_blank(row.get("val_loss")),
                gap_loss=_round_or_blank(row.get("gap_loss")),
                gap_auc=_round_or_blank(row.get("gap_auc")),
                runtime=_round_or_blank(row.get("runtime_per_epoch_min"), digits=2),
                verdict=row.get("verdict", ""),
            )
        )

    lines.append("")
    lines.append("## Next Iteration Recommendation")
    keep_rows = [r for r in rows if r.get("verdict") == "Keep"]
    maybe_rows = [r for r in rows if r.get("verdict") == "Maybe"]
    if keep_rows:
        top = sorted(
            keep_rows,
            key=lambda r: (
                float(r.get("val_auc") or -1e9),
                -float(r.get("gap_auc") or 1e9),
                -float(r.get("gap_loss") or 1e9),
            ),
            reverse=True,
        )[0]
        lines.append(
            f"- Keep `{top['exp_id']}` as current best single-factor candidate; rerun with 2 additional seeds for stability."
        )
    elif maybe_rows:
        lines.append("- No clear winner; repeat top-2 Maybe runs with 2 additional seeds before combining changes.")
    else:
        lines.append("- No improvement found; revert to baseline and test a different regularization axis.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.num_walks is not None and args.num_walks <= 0:
        raise ValueError("--num-walks must be > 0")
    if args.max_walk_length is not None and args.max_walk_length <= 0:
        raise ValueError("--max-walk-length must be > 0")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    dataset_cfg = _load_dataset_cfg(args.dataset)
    resolved = {
        "num_walks": FIXED_NUM_WALKS,
        "max_walk_length": int(dataset_cfg["base_max_walk_length"]),
        "batch_size": int(dataset_cfg["base_batch_size"]),
        "epochs": int(dataset_cfg["base_epochs"]),
    }

    if args.fast_baseline:
        resolved["max_walk_length"] = 80
        resolved["batch_size"] = 1024

    if args.num_walks is not None:
        if int(args.num_walks) != FIXED_NUM_WALKS:
            raise ValueError(
                f"--num-walks is fixed to {FIXED_NUM_WALKS} for this experiment protocol."
            )
        resolved["num_walks"] = FIXED_NUM_WALKS
    if args.max_walk_length is not None:
        resolved["max_walk_length"] = int(args.max_walk_length)
    if args.batch_size is not None:
        resolved["batch_size"] = int(args.batch_size)
    if args.epochs is not None:
        resolved["epochs"] = int(args.epochs)

    experiments = _build_experiments(dataset_cfg)

    requested_ids = {s.strip() for s in args.run_ids.split(",") if s.strip()}
    selected = [exp for exp in experiments if exp.exp_id in requested_ids]
    if not selected:
        raise ValueError("No experiments selected. Check --run-ids.")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suite_tag = (
        f"{args.dataset}_seed{args.seed}"
        f"_nw{resolved['num_walks']}_mw{resolved['max_walk_length']}"
        f"_bs{resolved['batch_size']}_ep{resolved['epochs']}_{timestamp}"
    )

    suite_dir = (ROOT / args.output_root / suite_tag).resolve()
    tmp_data_dir = (ROOT / args.tmp_root / suite_tag / "tmp_data").resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)

    print(f"Suite dir: {suite_dir}")
    print(f"Temp data dir: {tmp_data_dir}")
    print(
        "Effective experiment values: "
        f"num_walks={resolved['num_walks']} max_walk_length={resolved['max_walk_length']} "
        f"batch_size={resolved['batch_size']} epochs={resolved['epochs']}"
    )
    print(f"Planned runs ({len(selected)}): {[s.exp_id for s in selected]}")

    if args.dry_run:
        return 0

    original_dataset_dir = Path(dataset_cfg["data_dir"])
    before_snapshot = _snapshot_pt_artifacts(original_dataset_dir)

    _copy_raw_to_temp(dataset_cfg["raw_src"], tmp_data_dir)

    rows: List[Dict[str, object]] = []
    baseline_row: Optional[Dict[str, object]] = None

    for spec in selected:
        print(f"\n=== Running {spec.exp_id}: {spec.change} ===")
        row = _run_one(
            spec=spec,
            args=args,
            resolved=resolved,
            suite_dir=suite_dir,
            tmp_data_dir=tmp_data_dir,
            edge_list_file=str(dataset_cfg["edge_list_file"]),
        )

        if spec.exp_id == "E0_BASELINE":
            baseline_row = row
        row["verdict"] = _bool_verdict(row, baseline_row if spec.exp_id != "E0_BASELINE" else None)

        rows.append(row)
        print(
            f"{spec.exp_id}: status={row['status']} val_auc={_round_or_blank(row.get('val_auc'))} "
            f"test_auc={_round_or_blank(row.get('test_auc'))} gap_loss={_round_or_blank(row.get('gap_loss'))} "
            f"gap_auc={_round_or_blank(row.get('gap_auc'))} verdict={row['verdict']}"
        )

    after_snapshot = _snapshot_pt_artifacts(original_dataset_dir)
    data_safety_ok, data_safety_issues = _compare_snapshots(before_snapshot, after_snapshot)

    csv_path = suite_dir / "results.csv"
    md_path = suite_dir / "RESULTS.md"
    _write_csv(rows, csv_path)
    _write_markdown_summary(
        rows,
        md_path,
        args,
        suite_dir,
        data_safety_ok,
        data_safety_issues,
        resolved,
    )

    print("\n=== Complete ===")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print(f"Data safety check (original data/*.pt unchanged): {data_safety_ok}")
    if data_safety_issues:
        for issue in data_safety_issues:
            print(f"  - {issue}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
