#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import random
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data.prepare_data import prepare_data
from src.model.lit_model import LitEdgeClassifier
from src.training.callbacks import PerEpochPredictionSaver
from src.utils.config import get_seed, load_config, validate_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run post-hoc predictions/analysis from a selected checkpoint"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Experiment directory (e.g., outputs/wiki-rfa/wiki-rfa-run_20260222-123456)",
    )
    parser.add_argument(
        "--checkpoint-choice",
        type=str,
        default="best",
        choices=["best", "last", "other"],
        help="Checkpoint selection strategy",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Checkpoint path when --checkpoint-choice=other",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to process",
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default="predictions,triplets,heatmaps,aggregator",
        help="Comma-separated artifacts: predictions,triplets,heatmaps,aggregator",
    )
    parser.add_argument(
        "--agg-models",
        type=str,
        default="logistic",
        help="Comma-separated aggregator models: logistic,xgboost,lgbm",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id for post-hoc artifact folder",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Optional OmegaConf dotlist overrides",
    )
    return parser.parse_args()


def _parse_epoch(path: Path):
    match = re.search(r"epoch[=\-_](\d+)", path.name)
    return int(match.group(1)) if match else None


def _parse_metric(path: Path, key: str):
    match = re.search(rf"{key}(?:_epoch)?[=\-_]([0-9]*\.?[0-9]+)", path.name)
    return float(match.group(1)) if match else None


def resolve_checkpoint(checkpoint_dir: Path, choice: str, checkpoint_path: str = None):
    if choice == "other":
        if not checkpoint_path:
            raise ValueError(
                "--checkpoint-path is required when --checkpoint-choice=other"
            )
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    all_ckpts = sorted(checkpoint_dir.glob("*.ckpt"))
    if not all_ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    if choice == "last":
        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            return last_ckpt

        epochs = [(p, _parse_epoch(p)) for p in all_ckpts]
        epochs = [(p, e) for p, e in epochs if e is not None]
        if epochs:
            return max(epochs, key=lambda x: x[1])[0]
        return max(all_ckpts, key=lambda p: p.stat().st_mtime)

    # best
    auc_scored = [(p, _parse_metric(p, "val_auc")) for p in all_ckpts]
    auc_scored = [(p, score) for p, score in auc_scored if score is not None]
    if auc_scored:
        return max(auc_scored, key=lambda x: x[1])[0]

    loss_scored = [(p, _parse_metric(p, "val_loss")) for p in all_ckpts]
    loss_scored = [(p, score) for p, score in loss_scored if score is not None]
    if loss_scored:
        return min(loss_scored, key=lambda x: x[1])[0]

    return max(all_ckpts, key=lambda p: p.stat().st_mtime)


def prediction_path(exp_dir: Path, dataset_name: str, epoch: int, split: str):
    return (
        exp_dir
        / "checkpoints"
        / f"{dataset_name}_predictions"
        / f"epoch_{epoch:03d}"
        / f"{split}_predictions.pkl"
    )


def load_prediction(exp_dir: Path, dataset_name: str, epoch: int, split: str):
    path = prediction_path(exp_dir, dataset_name, epoch, split)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_triplets_from_prediction(payload):
    preds = np.asarray(payload["predictions"]).astype(int)
    targets = np.asarray(payload["targets"]).astype(int)
    correct = (preds == targets).astype(int)

    triplets = {
        "edge_id": np.asarray(payload["edge_ids"]).astype(int).tolist(),
        "walk_id": np.asarray(payload["walk_ids"]).astype(int).tolist(),
        "position": np.asarray(payload["positions"]).astype(int).tolist(),
        "walk_len": np.asarray(payload["walk_lengths"]).astype(int).tolist(),
        "dist_from_start": np.asarray(payload["dist_from_start"]).astype(int).tolist(),
        "dist_from_end": np.asarray(payload["dist_from_end"]).astype(int).tolist(),
        "correct": correct.tolist(),
    }
    return triplets


def save_triplets_and_heatmaps(
    exp_dir: Path, dataset_name: str, run_id: str, epoch: int, splits
):
    from scripts.plot_triplet_heatmap import (
        build_heatmap,
        compute_heatmap_stats,
        plot_heatmap,
    )

    out_base = exp_dir / "posthoc" / run_id
    out_base.mkdir(parents=True, exist_ok=True)

    for split in splits:
        payload = load_prediction(exp_dir, dataset_name, epoch, split)
        triplets = build_triplets_from_prediction(payload)

        triplet_path = out_base / f"triplets_{split}.pkl"
        with open(triplet_path, "wb") as f:
            pickle.dump(triplets, f)

        heatmap, counts = build_heatmap(triplets)
        heatmap_png = out_base / f"heatmap_{split}.png"
        plot_heatmap(heatmap, str(heatmap_png))

        np.savez(
            str(out_base / f"heatmap_{split}_data.npz"), grid=heatmap, counts=counts
        )
        stats = compute_heatmap_stats(heatmap, counts, triplets)
        with open(out_base / f"heatmap_{split}_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"✓ Triplets + heatmap saved for {split}: {out_base}")


def _aggregate_edge_probs(edge_ids, probs, y):
    edge_to_probs = {}
    edge_to_labels = {}
    for eid, prob, label in zip(edge_ids, probs, y):
        edge_to_probs.setdefault(int(eid), []).append(float(prob))
        edge_to_labels.setdefault(int(eid), []).append(int(label))

    edge_probs_out = []
    edge_labels_out = []
    for eid, plist in edge_to_probs.items():
        edge_probs_out.append(float(np.mean(plist)))
        labels = np.asarray(edge_to_labels[eid], dtype=int)
        edge_labels_out.append(int(np.bincount(labels).argmax()))

    return np.asarray(edge_probs_out), np.asarray(edge_labels_out)


def _pred_prob(preds):
    probs = np.asarray(preds.get("probabilities", []))
    if probs.ndim == 2 and probs.shape[1] >= 2:
        return probs[:, 1].astype(float)
    return np.asarray(preds["predictions"]).astype(float)


def run_aggregator(
    exp_dir: Path, dataset_name: str, run_id: str, epoch: int, models, seed
):
    val_preds = load_prediction(exp_dir, dataset_name, epoch, "val")
    test_preds = load_prediction(exp_dir, dataset_name, epoch, "test")

    y_train = np.asarray(val_preds["targets"]).astype(int)
    y_test = np.asarray(test_preds["targets"]).astype(int)

    unique = np.unique(y_train)
    if len(unique) != 2:
        print(
            "⚠ Aggregator currently supports binary labels only. Skipping aggregator stage."
        )
        return

    x_train = np.stack(
        [
            np.asarray(val_preds["dist_from_start"]).astype(float),
            np.asarray(val_preds["walk_lengths"]).astype(float),
            _pred_prob(val_preds),
        ],
        axis=1,
    )
    x_test = np.stack(
        [
            np.asarray(test_preds["dist_from_start"]).astype(float),
            np.asarray(test_preds["walk_lengths"]).astype(float),
            _pred_prob(test_preds),
        ],
        axis=1,
    )

    edge_train = np.asarray(val_preds["edge_ids"]).astype(int)
    edge_test = np.asarray(test_preds["edge_ids"]).astype(int)

    out_dir = exp_dir / "posthoc" / run_id / "aggregator"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        model_name = model_name.strip().lower()
        if not model_name:
            continue

        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)

        if model_name == "logistic":
            model = LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=seed
            )
            model.fit(x_train_s, y_train)
            train_probs = model.predict_proba(x_train_s)[:, 1]
            test_probs = model.predict_proba(x_test_s)[:, 1]
            model_config = {
                "model_name": "logistic",
                "seed": int(seed),
                "feature_names": ["dist_from_start", "walk_length", "predicted_prob"],
                "scaler": "StandardScaler",
                "params": {
                    "max_iter": 1000,
                    "class_weight": "balanced",
                    "random_state": int(seed),
                },
            }
        elif model_name == "xgboost":
            try:
                import xgboost as xgb
            except Exception:
                print("⚠ xgboost is not installed. Skipping xgboost aggregator.")
                continue

            scale_pos_weight = float(
                np.sum(y_train == 0) / max(1, np.sum(y_train == 1))
            )
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=seed,
                verbosity=0,
            )
            model.fit(x_train_s, y_train)
            train_probs = model.predict_proba(x_train_s)[:, 1]
            test_probs = model.predict_proba(x_test_s)[:, 1]
            model_config = {
                "model_name": "xgboost",
                "seed": int(seed),
                "feature_names": ["dist_from_start", "walk_length", "predicted_prob"],
                "scaler": "StandardScaler",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "learning_rate": 0.1,
                    "scale_pos_weight": float(scale_pos_weight),
                    "random_state": int(seed),
                    "verbosity": 0,
                },
            }
        elif model_name == "lgbm":
            try:
                import lightgbm as lgb
            except Exception:
                print("⚠ lightgbm is not installed. Skipping lgbm aggregator.")
                continue

            model = lgb.LGBMClassifier(
                n_estimators=200,
                num_leaves=31,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=seed,
                verbose=-1,
            )
            model.fit(x_train_s, y_train)
            train_probs = model.predict_proba(x_train_s)[:, 1]
            test_probs = model.predict_proba(x_test_s)[:, 1]
            model_config = {
                "model_name": "lgbm",
                "seed": int(seed),
                "feature_names": ["dist_from_start", "walk_length", "predicted_prob"],
                "scaler": "StandardScaler",
                "params": {
                    "n_estimators": 200,
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "class_weight": "balanced",
                    "random_state": int(seed),
                    "verbose": -1,
                },
            }
        else:
            print(f"⚠ Unknown aggregator model '{model_name}', skipping")
            continue

        train_pred = (train_probs >= 0.5).astype(int)
        test_pred = (test_probs >= 0.5).astype(int)

        walk_train_auc = roc_auc_score(y_train, train_probs)
        walk_test_auc = roc_auc_score(y_test, test_probs)

        edge_train_probs, edge_train_labels = _aggregate_edge_probs(
            edge_train, train_probs, y_train
        )
        edge_test_probs, edge_test_labels = _aggregate_edge_probs(
            edge_test, test_probs, y_test
        )

        edge_train_auc = roc_auc_score(edge_train_labels, edge_train_probs)
        edge_test_auc = roc_auc_score(edge_test_labels, edge_test_probs)

        model_dir = out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump({"model": model, "scaler": scaler}, f)
        with open(model_dir / "model_config.json", "w") as f:
            json.dump(model_config, f, indent=2)

        with open(model_dir / "summary.txt", "w") as f:
            f.write("Post-hoc aggregator summary\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Run ID: {run_id}\n")
            f.write(f"Model: {model_name}\n\n")
            f.write("Walk-level metrics:\n")
            f.write(f"  Train AUC: {walk_train_auc:.4f}\n")
            f.write(f"  Test  AUC: {walk_test_auc:.4f}\n")
            f.write(f"  Train ACC: {accuracy_score(y_train, train_pred):.4f}\n")
            f.write(f"  Test  ACC: {accuracy_score(y_test, test_pred):.4f}\n")
            f.write(f"  Train F1: {f1_score(y_train, train_pred):.4f}\n")
            f.write(f"  Test  F1: {f1_score(y_test, test_pred):.4f}\n\n")
            f.write("Edge-level aggregated AUC:\n")
            f.write(
                f"  Train AUC: {edge_train_auc:.4f} ({len(edge_train_probs)} edges)\n"
            )
            f.write(
                f"  Test  AUC: {edge_test_auc:.4f} ({len(edge_test_probs)} edges)\n"
            )

        print(f"✓ Aggregator '{model_name}' complete: {model_dir}")


def main():
    args = parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    validate_config(cfg, context="posthoc")
    exp_dir = Path(args.exp_dir)
    checkpoint_dir = exp_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    ckpt_path = resolve_checkpoint(
        checkpoint_dir, args.checkpoint_choice, args.checkpoint_path
    )
    epoch = _parse_epoch(ckpt_path)
    if epoch is None:
        epoch = 0

    cfg.training.checkpoint_dir = str(checkpoint_dir)
    cfg.training.log_dir = str(exp_dir / "logs")

    try:
        seed = get_seed(cfg)
    except ValueError:
        seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        f"cuda:{args.device}"
        if cfg.training.use_cuda and torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using checkpoint: {ckpt_path}")
    print(f"Using epoch index: {epoch}")
    print(f"Using device: {device}")

    data_module = prepare_data(cfg)

    try:
        model = LitEdgeClassifier.load_from_checkpoint(str(ckpt_path), cfg=cfg)
    except Exception:
        model = LitEdgeClassifier.load_from_checkpoint(str(ckpt_path))
    model = model.to(device)
    model.eval()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    artifacts = {a.strip() for a in args.artifacts.split(",") if a.strip()}
    agg_models = [m.strip() for m in args.agg_models.split(",") if m.strip()]

    run_id = args.run_id or f"{ckpt_path.stem}_posthoc"

    if (
        "predictions" in artifacts
        or "triplets" in artifacts
        or "heatmaps" in artifacts
        or "aggregator" in artifacts
    ):
        saver = PerEpochPredictionSaver(cfg, data_module)
        fake_trainer = SimpleNamespace(current_epoch=epoch)

        for split in splits:
            if split not in data_module:
                print(f"⚠ Unknown split '{split}', skipping")
                continue
            print(f"Generating predictions for split={split}...")
            pred = saver._extract_predictions(
                fake_trainer, model, data_module[split], split
            )
            saver._save_predictions(pred, epoch, split)

    if "triplets" in artifacts or "heatmaps" in artifacts:
        required_splits = splits if "triplets" in artifacts else splits
        save_triplets_and_heatmaps(
            exp_dir, cfg.dataset.name, run_id, epoch, required_splits
        )

    if "aggregator" in artifacts:
        run_aggregator(exp_dir, cfg.dataset.name, run_id, epoch, agg_models, seed)

    print(f"\n✓ Post-hoc pipeline complete. Artifacts: {exp_dir / 'posthoc' / run_id}")


if __name__ == "__main__":
    main()
