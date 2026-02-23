#!/usr/bin/env python3
"""Run controlled walk-budget sensitivity experiments by model quality.

Protocol highlights:
- Enforces `training.epochs<=5`
- Forces full preprocessing rebuild (`preprocess.use_cache=false`, `preprocess.save=false`)
- Uses isolated temp `dataset.data_dir` per run to avoid touching production artifacts
- Parses TensorBoard scalars to extract train/val/test AUC/loss
- Produces CSV + Markdown summary with recommendation shortlist
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    seed: int
    max_walk_length: int
    num_walks: int
    phase: str

    @property
    def tag(self) -> str:
        return (
            f"{self.dataset}_mw{self.max_walk_length}_nw{self.num_walks}_"
            f"seed{self.seed}_{self.phase}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk budget sensitivity by AUC + overfit (isolated temp data dirs)"
    )
    parser.add_argument(
        "--dataset",
        default="slashdot090221",
        help="Dataset key (must match configs/<dataset>.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Canonical reproducibility.seed override used for all runs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epoch cap (must be <=5)",
    )
    parser.add_argument(
        "--num-walks",
        default="1000000,2000000,3500000,5000000",
        help="Comma-separated num_walks candidates for phase-1 sweep",
    )
    parser.add_argument(
        "--max-walk-lengths",
        default="60,80",
        help="Comma-separated max_walk_length candidates",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="How many budgets to keep from phase-1 for phase-2",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/walk_budget_auc",
        help="Output root for logs/results",
    )
    parser.add_argument(
        "--tmp-root",
        default="tmp/walk_budget_auc",
        help="Root for isolated per-run temporary dataset dirs",
    )
    return parser.parse_args()


def _parse_int_list(csv_values: str) -> List[int]:
    values: List[int] = []
    for raw in csv_values.split(","):
        raw = raw.strip().replace("_", "")
        if not raw:
            continue
        values.append(int(raw))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _load_dataset_cfg(dataset: str) -> Dict[str, str]:
    cfg_path = ROOT / "configs" / f"{dataset}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing dataset config: {cfg_path}")
    cfg = OmegaConf.load(str(cfg_path))

    data_dir = Path(str(cfg.dataset.data_dir))
    edge_list_file = str(cfg.dataset.edge_list_file)
    raw_src = ROOT / data_dir / edge_list_file
    if not raw_src.exists():
        raise FileNotFoundError(f"Missing raw edge file: {raw_src}")
    return {
        "data_dir": str(ROOT / data_dir),
        "edge_list_file": edge_list_file,
        "raw_src": str(raw_src),
    }


def _snapshot_pt_artifacts(
    data_root: Path, dataset: str
) -> Dict[str, Dict[str, float]]:
    dataset_dir = data_root / dataset
    if not dataset_dir.exists():
        return {}

    snapshot: Dict[str, Dict[str, float]] = {}
    for path in sorted(dataset_dir.glob("*.pt")):
        stat = path.stat()
        snapshot[str(path)] = {
            "size": float(stat.st_size),
            "mtime": float(stat.st_mtime),
        }
    return snapshot


def _compare_snapshots(
    before: Dict[str, Dict[str, float]],
    after: Dict[str, Dict[str, float]],
) -> Tuple[bool, List[str]]:
    changed: List[str] = []
    before_keys = set(before.keys())
    after_keys = set(after.keys())

    for added in sorted(after_keys - before_keys):
        changed.append(f"ADDED: {added}")
    for removed in sorted(before_keys - after_keys):
        changed.append(f"REMOVED: {removed}")

    for common in sorted(before_keys & after_keys):
        b = before[common]
        a = after[common]
        if int(b["size"]) != int(a["size"]) or float(b["mtime"]) != float(a["mtime"]):
            changed.append(
                f"MODIFIED: {common} (size {int(b['size'])}->{int(a['size'])}, "
                f"mtime {b['mtime']:.3f}->{a['mtime']:.3f})"
            )

    return (len(changed) == 0, changed)


def _copy_raw_to_temp(raw_src: Path, tmp_data_dir: Path) -> Path:
    tmp_data_dir.mkdir(parents=True, exist_ok=True)
    dst = tmp_data_dir / raw_src.name
    shutil.copy2(raw_src, dst)
    return dst


def _find_latest_event_file(
    log_root: Path, exp_name: str, dataset: str
) -> Optional[Path]:
    logger_dir = log_root / f"{dataset}-{exp_name}"
    if not logger_dir.exists():
        return None

    event_files = sorted(
        logger_dir.glob("**/events.out.tfevents.*"),
        key=lambda p: p.stat().st_mtime,
    )
    return event_files[-1] if event_files else None


def _extract_scalar_series(event_file: Path, tag: str) -> List[Tuple[int, float]]:
    acc = EventAccumulator(str(event_file))
    acc.Reload()
    tags = set(acc.Tags().get("scalars", []))
    if tag not in tags:
        return []
    return [(int(e.step), float(e.value)) for e in acc.Scalars(tag)]


def _pick_best_epoch(val_auc_series: List[Tuple[int, float]]) -> int:
    if not val_auc_series:
        return -1
    best_step, _ = max(val_auc_series, key=lambda x: x[1])
    return int(best_step)


def _value_at_step(series: List[Tuple[int, float]], step: int) -> Optional[float]:
    if not series:
        return None
    mapping = {s: v for s, v in series}
    if step in mapping:
        return mapping[step]
    latest_step = max(mapping.keys())
    return mapping[latest_step]


def _read_metrics_from_events(event_file: Path) -> Dict[str, Optional[float]]:
    val_auc = _extract_scalar_series(event_file, "val_auc_epoch")
    train_auc = _extract_scalar_series(event_file, "train_auc_epoch")
    test_auc = _extract_scalar_series(event_file, "test_auc_epoch")

    train_loss = _extract_scalar_series(event_file, "train_loss")
    if not train_loss:
        train_loss = _extract_scalar_series(event_file, "train_loss_epoch")
    val_loss = _extract_scalar_series(event_file, "val_loss")
    if not val_loss:
        val_loss = _extract_scalar_series(event_file, "val_loss_epoch")

    best_epoch = _pick_best_epoch(val_auc)

    train_auc_val = _value_at_step(train_auc, best_epoch)
    val_auc_val = _value_at_step(val_auc, best_epoch)
    test_auc_val = _value_at_step(test_auc, best_epoch)
    train_loss_val = _value_at_step(train_loss, best_epoch)
    val_loss_val = _value_at_step(val_loss, best_epoch)

    return {
        "best_epoch": float(best_epoch) if best_epoch >= 0 else None,
        "train_auc": train_auc_val,
        "val_auc": val_auc_val,
        "test_auc": test_auc_val,
        "train_loss": train_loss_val,
        "val_loss": val_loss_val,
    }


def _safe_gap(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a - b)


def _float_or_nan(x: Optional[float]) -> float:
    return float("nan") if x is None else float(x)


def _score_tuple(row: Dict[str, object]) -> Tuple[float, float, float, int, float]:
    val_auc = _float_or_nan(row.get("val_auc"))
    test_auc = _float_or_nan(row.get("test_auc"))
    auc_gap = _float_or_nan(row.get("auc_gap"))
    num_walks = int(row.get("num_walks", 10**12))
    runtime = _float_or_nan(row.get("runtime_minutes"))

    if val_auc != val_auc:
        val_auc = -1e9
    if test_auc != test_auc:
        test_auc = -1e9
    if auc_gap != auc_gap:
        auc_gap = 1e9
    if runtime != runtime:
        runtime = 1e9

    return (val_auc, test_auc, -auc_gap, -num_walks, -runtime)


def _run_one(
    spec: RunSpec, paths: Dict[str, Path], raw_src: Path, edge_list_file: str
) -> Dict[str, object]:
    run_dir = paths["run_root"] / spec.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    temp_data_dir = paths["tmp_root"] / spec.dataset / spec.tag / "data"
    _copy_raw_to_temp(raw_src=raw_src, tmp_data_dir=temp_data_dir)

    exp_name = f"walkbudget-{spec.tag}"
    run_log = run_dir / "run.log"

    cmd = [
        sys.executable,
        "run.py",
        f"dataset.name={spec.dataset}",
        f"dataset.data_dir={temp_data_dir.as_posix()}",
        f"dataset.edge_list_file={edge_list_file}",
        f"dataset.max_walk_length={spec.max_walk_length}",
        f"dataset.num_walks={spec.num_walks}",
        f"reproducibility.seed={spec.seed}",
        "preprocess.use_cache=false",
        "preprocess.save=false",
        "training.callbacks.enable_prediction_saver=false",
        "training.callbacks.enable_per_epoch_test_runner=false",
        f"training.epochs=5",
        f"training.exp_name={exp_name}",
    ]

    started = time.time()
    with run_log.open("w", encoding="utf-8") as f:
        process = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    runtime_minutes = (time.time() - started) / 60.0

    event_file = _find_latest_event_file(paths["logs_root"], exp_name, spec.dataset)
    if event_file is None:
        metrics = {
            "best_epoch": None,
            "train_auc": None,
            "val_auc": None,
            "test_auc": None,
            "train_loss": None,
            "val_loss": None,
        }
    else:
        metrics = _read_metrics_from_events(event_file)

    train_auc = metrics["train_auc"]
    val_auc = metrics["val_auc"]
    test_auc = metrics["test_auc"]
    train_loss = metrics["train_loss"]
    val_loss = metrics["val_loss"]

    row: Dict[str, object] = {
        "dataset": spec.dataset,
        "seed": spec.seed,
        "num_walks": spec.num_walks,
        "max_walk_length": spec.max_walk_length,
        "phase": spec.phase,
        "best_epoch": (
            None if metrics["best_epoch"] is None else int(metrics["best_epoch"])
        ),
        "train_auc": train_auc,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "auc_gap": _safe_gap(train_auc, val_auc),
        "loss_gap": _safe_gap(val_loss, train_loss),
        "test_minus_val": _safe_gap(test_auc, val_auc),
        "runtime_minutes": runtime_minutes,
        "return_code": int(process.returncode),
        "status": "ok" if process.returncode == 0 else "failed",
        "run_log": str(run_log),
        "temp_data_dir": str(temp_data_dir),
        "event_file": str(event_file) if event_file else None,
    }
    return row


def _write_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    fieldnames = [
        "dataset",
        "seed",
        "num_walks",
        "max_walk_length",
        "phase",
        "best_epoch",
        "train_auc",
        "val_auc",
        "test_auc",
        "train_loss",
        "val_loss",
        "auc_gap",
        "loss_gap",
        "test_minus_val",
        "runtime_minutes",
        "return_code",
        "status",
        "run_log",
        "temp_data_dir",
        "event_file",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "NA"
    return f"{float(x):.{digits}f}"


def _pick_recommendations(
    ok_rows: List[Dict[str, object]],
) -> Dict[str, Optional[Dict[str, object]]]:
    if not ok_rows:
        return {"quality": None, "anti_overfit": None, "balance": None}

    quality = sorted(ok_rows, key=lambda r: _score_tuple(r), reverse=True)[0]

    anti_overfit = sorted(
        ok_rows,
        key=lambda r: (
            _float_or_nan(r.get("auc_gap")),
            -_float_or_nan(r.get("val_auc")),
            -_float_or_nan(r.get("test_auc")),
        ),
    )[0]

    balance = sorted(
        ok_rows,
        key=lambda r: (
            -_float_or_nan(r.get("val_auc")),
            _float_or_nan(r.get("auc_gap")),
            int(r.get("num_walks", 10**12)),
            _float_or_nan(r.get("runtime_minutes")),
        ),
    )[0]

    return {"quality": quality, "anti_overfit": anti_overfit, "balance": balance}


def _write_markdown(
    out_md: Path,
    rows: List[Dict[str, object]],
    recommendations: Dict[str, Optional[Dict[str, object]]],
    data_unchanged: bool,
    changed_files: List[str],
) -> None:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    sorted_rows = sorted(ok_rows, key=lambda r: _score_tuple(r), reverse=True)

    lines: List[str] = []
    lines.append("# Walk Budget AUC + Overfit Experiment Report")
    lines.append("")
    lines.append("## Results Table")
    lines.append("")
    lines.append(
        "| dataset | seed | num_walks | max_walk_length | phase | best_epoch | train_auc | val_auc | test_auc | train_loss | val_loss | auc_gap | loss_gap | runtime_minutes |"
    )
    lines.append(
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in sorted_rows:
        lines.append(
            "| "
            f"{r['dataset']} | {r['seed']} | {r['num_walks']} | {r['max_walk_length']} | {r['phase']} | "
            f"{r['best_epoch']} | {_fmt(r['train_auc'])} | {_fmt(r['val_auc'])} | {_fmt(r['test_auc'])} | "
            f"{_fmt(r['train_loss'])} | {_fmt(r['val_loss'])} | {_fmt(r['auc_gap'])} | {_fmt(r['loss_gap'])} | "
            f"{_fmt(r['runtime_minutes'], digits=2)} |"
        )

    lines.append("")
    lines.append("## Recommendations")
    lines.append("")

    def _rec_line(title: str, rec: Optional[Dict[str, object]]) -> None:
        if rec is None:
            lines.append(f"- **{title}**: NA")
            return
        lines.append(
            f"- **{title}**: num_walks={rec['num_walks']}, max_walk_length={rec['max_walk_length']}, "
            f"val_auc={_fmt(rec.get('val_auc'))}, test_auc={_fmt(rec.get('test_auc'))}, "
            f"auc_gap={_fmt(rec.get('auc_gap'))}, runtime={_fmt(rec.get('runtime_minutes'), digits=2)} min"
        )

    _rec_line("Winner (quality)", recommendations.get("quality"))
    _rec_line("Winner (anti-overfit)", recommendations.get("anti_overfit"))
    _rec_line("Winner (compute/quality balance)", recommendations.get("balance"))

    lines.append("")
    lines.append("## Data Artifact Integrity")
    lines.append("")
    lines.append(
        f"- Production `data/<dataset>/*.pt` unchanged: **{str(data_unchanged)}**"
    )
    if changed_files:
        lines.append("- Detected changes:")
        for item in changed_files:
            lines.append(f"  - {item}")
    else:
        lines.append(
            "- No modifications detected in tracked production `.pt` artifacts."
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.epochs > 5:
        raise ValueError("Hard constraint violated: training.epochs must be <= 5")

    dataset_cfg = _load_dataset_cfg(args.dataset)
    num_walks_candidates = _parse_int_list(args.num_walks)
    max_walk_lengths = _parse_int_list(args.max_walk_lengths)
    if 80 not in max_walk_lengths:
        raise ValueError(
            "Expected max_walk_length candidates to include 80 for staged phase-1."
        )

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = ROOT / args.output_root / args.dataset / ts
    logs_root = ROOT / "outputs"
    tmp_root = ROOT / args.tmp_root

    paths = {
        "run_root": run_root,
        "logs_root": logs_root,
        "tmp_root": tmp_root,
    }
    run_root.mkdir(parents=True, exist_ok=True)

    data_root = ROOT / "data"
    before_snapshot = _snapshot_pt_artifacts(data_root, args.dataset)

    rows: List[Dict[str, object]] = []
    seen = set()

    phase1_specs: List[RunSpec] = []
    for nw in num_walks_candidates:
        phase1_specs.append(
            RunSpec(
                dataset=args.dataset,
                seed=args.seed,
                max_walk_length=80,
                num_walks=nw,
                phase="phase1",
            )
        )

    for spec in phase1_specs:
        if spec.tag in seen:
            continue
        seen.add(spec.tag)
        print(
            f"[RUN] {spec.tag} | epochs=5 | cache=false | save=false | isolated_data_dir=true"
        )
        row = _run_one(
            spec=spec,
            paths=paths,
            raw_src=Path(dataset_cfg["raw_src"]),
            edge_list_file=dataset_cfg["edge_list_file"],
        )
        rows.append(row)
        print(
            f"      status={row['status']} val_auc={_fmt(row.get('val_auc'))} "
            f"test_auc={_fmt(row.get('test_auc'))} auc_gap={_fmt(row.get('auc_gap'))} "
            f"runtime={_fmt(row.get('runtime_minutes'), digits=2)}m"
        )

    ok_phase1 = [r for r in rows if r["phase"] == "phase1" and r["status"] == "ok"]
    ranked_phase1 = sorted(ok_phase1, key=lambda r: _score_tuple(r), reverse=True)
    top_k = ranked_phase1[: max(1, args.top_k)]
    top_num_walks = sorted({int(r["num_walks"]) for r in top_k})

    for nw in top_num_walks:
        for mw in max_walk_lengths:
            phase = "phase2"
            spec = RunSpec(
                dataset=args.dataset,
                seed=args.seed,
                max_walk_length=int(mw),
                num_walks=int(nw),
                phase=phase,
            )
            if spec.tag in seen:
                continue
            seen.add(spec.tag)
            print(
                f"[RUN] {spec.tag} | epochs=5 | cache=false | save=false | isolated_data_dir=true"
            )
            row = _run_one(
                spec=spec,
                paths=paths,
                raw_src=Path(dataset_cfg["raw_src"]),
                edge_list_file=dataset_cfg["edge_list_file"],
            )
            rows.append(row)
            print(
                f"      status={row['status']} val_auc={_fmt(row.get('val_auc'))} "
                f"test_auc={_fmt(row.get('test_auc'))} auc_gap={_fmt(row.get('auc_gap'))} "
                f"runtime={_fmt(row.get('runtime_minutes'), digits=2)}m"
            )

    after_snapshot = _snapshot_pt_artifacts(data_root, args.dataset)
    data_unchanged, changed_files = _compare_snapshots(before_snapshot, after_snapshot)

    out_csv = run_root / "walk_budget_auc_results.csv"
    out_md = run_root / "walk_budget_auc_report.md"
    _write_csv(rows, out_csv)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    recs = _pick_recommendations(ok_rows)
    _write_markdown(
        out_md=out_md,
        rows=rows,
        recommendations=recs,
        data_unchanged=data_unchanged,
        changed_files=changed_files,
    )

    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved report: {out_md}")
    print(f"Production .pt unchanged: {data_unchanged}")
    if changed_files:
        for item in changed_files:
            print(f"  - {item}")

    failed = [r for r in rows if r["status"] != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
