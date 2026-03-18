#!/usr/bin/env python3
"""Run controlled batch-size sensitivity experiments by model quality.

Protocol highlights:
- Fixed walk budget (`dataset.num_walks`) and walk length (`dataset.max_walk_length`)
- Sweeps `training.batch_size` values (small to large)
- Stops larger batch sizes for a dataset after first OOM
- Forces full preprocessing rebuild (`preprocess.use_cache=false`, `preprocess.save=false`)
- Uses isolated temp `dataset.data_dir` per run to protect production artifacts
- Parses TensorBoard scalars (across multiple event files) for train/val/test AUC/loss
- Produces per-dataset CSV + Markdown reports
"""

from __future__ import annotations

import argparse
import csv
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


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    seed: int
    batch_size: int
    num_walks: int
    max_walk_length: int

    @property
    def tag(self) -> str:
        return (
            f"{self.dataset}_bs{self.batch_size}_nw{self.num_walks}_"
            f"mw{self.max_walk_length}_seed{self.seed}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-size sensitivity by AUC + overfit (isolated temp data dirs)"
    )
    parser.add_argument(
        "--datasets",
        default="slashdot090221,epinions,wiki-rfa",
        help="Comma-separated datasets (must match configs/<dataset>.yaml)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device id passed to run.py --device",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-walks", type=int, default=1_000_000)
    parser.add_argument("--max-walk-length", type=int, default=80)
    parser.add_argument(
        "--batch-sizes",
        default="32,64,128,256,512,1024,2048,4096",
        help="Comma-separated batch sizes to sweep",
    )
    parser.add_argument("--output-root", default="outputs/batch_size_auc")
    parser.add_argument("--tmp-root", default="tmp/batch_size_auc")
    return parser.parse_args()


def _parse_int_list(csv_values: str) -> List[int]:
    out: List[int] = []
    for raw in csv_values.split(","):
        raw = raw.strip().replace("_", "")
        if not raw:
            continue
        out.append(int(raw))
    if not out:
        raise ValueError("Expected at least one integer value")
    return out


def _parse_str_list(csv_values: str) -> List[str]:
    out = [x.strip() for x in csv_values.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one dataset")
    return out


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
        "edge_list_file": edge_list_file,
        "raw_src": str(raw_src),
    }


def _snapshot_pt_artifacts(data_root: Path, dataset: str) -> Dict[str, Dict[str, float]]:
    dataset_dir = data_root / dataset
    if not dataset_dir.exists():
        return {}
    snap: Dict[str, Dict[str, float]] = {}
    for p in sorted(dataset_dir.glob("*.pt")):
        st = p.stat()
        snap[str(p)] = {"size": float(st.st_size), "mtime": float(st.st_mtime)}
    return snap


def _compare_snapshots(
    before: Dict[str, Dict[str, float]],
    after: Dict[str, Dict[str, float]],
) -> Tuple[bool, List[str]]:
    changed: List[str] = []
    bk = set(before.keys())
    ak = set(after.keys())

    for added in sorted(ak - bk):
        changed.append(f"ADDED: {added}")
    for removed in sorted(bk - ak):
        changed.append(f"REMOVED: {removed}")
    for common in sorted(bk & ak):
        b = before[common]
        a = after[common]
        if int(b["size"]) != int(a["size"]) or float(b["mtime"]) != float(a["mtime"]):
            changed.append(
                f"MODIFIED: {common} (size {int(b['size'])}->{int(a['size'])}, "
                f"mtime {b['mtime']:.3f}->{a['mtime']:.3f})"
            )

    return len(changed) == 0, changed


def _copy_raw_to_temp(raw_src: Path, tmp_data_dir: Path) -> Path:
    tmp_data_dir.mkdir(parents=True, exist_ok=True)
    dst = tmp_data_dir / raw_src.name
    shutil.copy2(raw_src, dst)
    return dst


def _find_event_files(dataset: str, exp_name: str) -> List[Path]:
    exp_dirs = sorted((ROOT / "outputs" / dataset).glob(f"{exp_name}_*"))
    if not exp_dirs:
        return []
    latest = exp_dirs[-1]
    return sorted(latest.glob("logs/**/events.out.tfevents.*"))


def _extract_scalar_series(event_files: List[Path], tag: str) -> List[Tuple[int, float]]:
    series: List[Tuple[int, float]] = []
    for event_file in event_files:
        try:
            acc = EventAccumulator(str(event_file))
            acc.Reload()
            tags = set(acc.Tags().get("scalars", []))
            if tag in tags:
                series.extend((int(e.step), float(e.value)) for e in acc.Scalars(tag))
        except Exception:
            continue

    dedup = {s: v for s, v in series}
    return sorted(dedup.items())


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
    lower = [s for s in mapping if s <= step]
    if lower:
        return mapping[max(lower)]
    return mapping[min(mapping.keys())]


def _read_metrics_from_events(event_files: List[Path]) -> Dict[str, Optional[float]]:
    val_auc = _extract_scalar_series(event_files, "val_auc_epoch")
    train_auc = _extract_scalar_series(event_files, "train_auc_epoch")
    test_auc = _extract_scalar_series(event_files, "test_auc_epoch")

    train_loss = _extract_scalar_series(event_files, "train_loss")
    if not train_loss:
        train_loss = _extract_scalar_series(event_files, "train_loss_epoch")
    val_loss = _extract_scalar_series(event_files, "val_loss")
    if not val_loss:
        val_loss = _extract_scalar_series(event_files, "val_loss_epoch")

    best_epoch = _pick_best_epoch(val_auc)

    return {
        "best_epoch": float(best_epoch) if best_epoch >= 0 else None,
        "train_auc": _value_at_step(train_auc, best_epoch),
        "val_auc": _value_at_step(val_auc, best_epoch),
        "test_auc": _value_at_step(test_auc, best_epoch),
        "train_loss": _value_at_step(train_loss, best_epoch),
        "val_loss": _value_at_step(val_loss, best_epoch),
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
    batch_size = int(row.get("batch_size", 10**12))
    runtime = _float_or_nan(row.get("runtime_minutes"))

    if val_auc != val_auc:
        val_auc = -1e9
    if test_auc != test_auc:
        test_auc = -1e9
    if auc_gap != auc_gap:
        auc_gap = 1e9
    if runtime != runtime:
        runtime = 1e9

    return (val_auc, test_auc, -auc_gap, -batch_size, -runtime)


def _run_one(
    spec: RunSpec,
    paths: Dict[str, Path],
    raw_src: Path,
    edge_list_file: str,
    device: int,
) -> Dict[str, object]:
    run_dir = paths["run_root"] / spec.dataset / spec.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    temp_data_dir = paths["tmp_root"] / spec.dataset / spec.tag / "data"
    _copy_raw_to_temp(raw_src=raw_src, tmp_data_dir=temp_data_dir)

    exp_name = f"batchsize-{spec.tag}"
    run_log = run_dir / "run.log"

    cmd = [
        sys.executable,
        "run.py",
        "--device",
        str(device),
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
        "training.epochs=5",
        f"training.batch_size={spec.batch_size}",
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

    log_text = run_log.read_text(encoding="utf-8", errors="ignore")
    oom = ("out of memory" in log_text.lower()) or ("cuda out of memory" in log_text.lower())

    event_files = _find_event_files(spec.dataset, exp_name)
    if not event_files:
        metrics = {
            "best_epoch": None,
            "train_auc": None,
            "val_auc": None,
            "test_auc": None,
            "train_loss": None,
            "val_loss": None,
        }
    else:
        metrics = _read_metrics_from_events(event_files)

    train_auc = metrics["train_auc"]
    val_auc = metrics["val_auc"]
    test_auc = metrics["test_auc"]
    train_loss = metrics["train_loss"]
    val_loss = metrics["val_loss"]

    status = "ok"
    if process.returncode != 0:
        status = "oom" if oom else "failed"

    return {
        "dataset": spec.dataset,
        "seed": spec.seed,
        "batch_size": spec.batch_size,
        "num_walks": spec.num_walks,
        "max_walk_length": spec.max_walk_length,
        "best_epoch": None if metrics["best_epoch"] is None else int(metrics["best_epoch"]),
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
        "status": status,
        "run_log": str(run_log),
        "temp_data_dir": str(temp_data_dir),
        "event_files": ";".join(str(p) for p in event_files),
    }


def _write_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    fieldnames = [
        "dataset",
        "seed",
        "batch_size",
        "num_walks",
        "max_walk_length",
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
        "event_files",
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


def _pick_recommendations(ok_rows: List[Dict[str, object]]) -> Dict[str, Optional[Dict[str, object]]]:
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
            int(r.get("batch_size", 10**12)),
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
    stopped_on_oom: bool,
) -> None:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    sorted_rows = sorted(ok_rows, key=lambda r: _score_tuple(r), reverse=True)

    lines: List[str] = []
    lines.append("# Batch Size AUC + Overfit Experiment Report")
    lines.append("")
    lines.append("## Results Table")
    lines.append("")
    lines.append(
        "| dataset | seed | batch_size | num_walks | max_walk_length | best_epoch | train_auc | val_auc | test_auc | train_loss | val_loss | auc_gap | loss_gap | runtime_minutes | status |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )

    for r in sorted(rows, key=lambda rr: int(rr["batch_size"])):
        lines.append(
            "| "
            f"{r['dataset']} | {r['seed']} | {r['batch_size']} | {r['num_walks']} | {r['max_walk_length']} | "
            f"{r.get('best_epoch')} | {_fmt(r.get('train_auc'))} | {_fmt(r.get('val_auc'))} | {_fmt(r.get('test_auc'))} | "
            f"{_fmt(r.get('train_loss'))} | {_fmt(r.get('val_loss'))} | {_fmt(r.get('auc_gap'))} | {_fmt(r.get('loss_gap'))} | "
            f"{_fmt(r.get('runtime_minutes'),2)} | {r['status']} |"
        )

    lines.append("")
    lines.append("## Recommendations")
    lines.append("")

    def _rec_line(title: str, rec: Optional[Dict[str, object]]) -> None:
        if rec is None:
            lines.append(f"- **{title}**: NA")
            return
        lines.append(
            f"- **{title}**: batch_size={rec['batch_size']}, val_auc={_fmt(rec.get('val_auc'))}, "
            f"test_auc={_fmt(rec.get('test_auc'))}, auc_gap={_fmt(rec.get('auc_gap'))}, "
            f"runtime={_fmt(rec.get('runtime_minutes'),2)} min"
        )

    _rec_line("Winner (quality)", recommendations.get("quality"))
    _rec_line("Winner (anti-overfit)", recommendations.get("anti_overfit"))
    _rec_line("Winner (compute/quality balance)", recommendations.get("balance"))

    lines.append("")
    lines.append("## Execution Notes")
    lines.append("")
    lines.append(f"- Sweep stopped on OOM: **{stopped_on_oom}**")

    lines.append("")
    lines.append("## Data Artifact Integrity")
    lines.append("")
    lines.append(f"- Production `data/<dataset>/*.pt` unchanged: **{data_unchanged}**")
    if changed_files:
        lines.append("- Detected changes:")
        for item in changed_files:
            lines.append(f"  - {item}")
    else:
        lines.append("- No modifications detected in tracked production `.pt` artifacts.")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.epochs > 5:
        raise ValueError("Hard constraint violated: training.epochs must be <= 5")

    datasets = _parse_str_list(args.datasets)
    batch_sizes = sorted(set(_parse_int_list(args.batch_sizes)))

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    root_out = ROOT / args.output_root / ts
    tmp_root = ROOT / args.tmp_root
    root_out.mkdir(parents=True, exist_ok=True)

    data_root = ROOT / "data"
    overall_failures = 0

    for dataset in datasets:
        cfg = _load_dataset_cfg(dataset)

        before_snapshot = _snapshot_pt_artifacts(data_root, dataset)
        rows: List[Dict[str, object]] = []
        stopped_on_oom = False

        print(f"\n[DATASET] {dataset}")
        for bs in batch_sizes:
            spec = RunSpec(
                dataset=dataset,
                seed=args.seed,
                batch_size=bs,
                num_walks=args.num_walks,
                max_walk_length=args.max_walk_length,
            )
            print(
                f"[RUN] {spec.tag} | epochs=5 | cache=false | save=false | batch_size={bs}"
            )
            row = _run_one(
                spec=spec,
                paths={"run_root": root_out, "tmp_root": tmp_root},
                raw_src=Path(cfg["raw_src"]),
                edge_list_file=cfg["edge_list_file"],
                device=args.device,
            )
            rows.append(row)
            print(
                f"      status={row['status']} val_auc={_fmt(row.get('val_auc'))} "
                f"test_auc={_fmt(row.get('test_auc'))} auc_gap={_fmt(row.get('auc_gap'))} "
                f"runtime={_fmt(row.get('runtime_minutes'),2)}m"
            )

            if row["status"] == "oom":
                print(f"      stopping larger batch sizes for {dataset} due to OOM")
                stopped_on_oom = True
                break

        after_snapshot = _snapshot_pt_artifacts(data_root, dataset)
        data_unchanged, changed_files = _compare_snapshots(before_snapshot, after_snapshot)

        dataset_dir = root_out / dataset
        out_csv = dataset_dir / "batch_size_auc_results.csv"
        out_md = dataset_dir / "batch_size_auc_report.md"

        _write_csv(rows, out_csv)
        recs = _pick_recommendations([r for r in rows if r["status"] == "ok"])
        _write_markdown(
            out_md=out_md,
            rows=rows,
            recommendations=recs,
            data_unchanged=data_unchanged,
            changed_files=changed_files,
            stopped_on_oom=stopped_on_oom,
        )

        print(f"Saved CSV: {out_csv}")
        print(f"Saved report: {out_md}")
        print(f"Production .pt unchanged: {data_unchanged}")

        failed = [r for r in rows if r["status"] == "failed"]
        overall_failures += len(failed)

    return 1 if overall_failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
