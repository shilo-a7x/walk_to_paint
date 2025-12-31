#!/usr/bin/env python3
"""Dump config overrides for a specific Optuna trial.

Usage:
  python scripts/dump_trial_overrides.py --study outputs/wiki-rfa/.../optuna_study_x.pkl --trial 194

Outputs YAML (and optional dotlist) you can feed to run.py overrides.
"""
import argparse
import json
from pathlib import Path

import joblib
import optuna
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, help="Path to Optuna study .pkl")
    ap.add_argument("--trial", type=int, required=True, help="Trial number")
    ap.add_argument("--dotlist", action="store_true", help="Also print dotlist overrides")
    args = ap.parse_args()

    study_path = Path(args.study)
    if not study_path.exists():
        raise FileNotFoundError(study_path)

    study = joblib.load(study_path)
    is_max = study.direction == optuna.study.StudyDirection.MAXIMIZE

    matches = [t for t in study.trials if t.number == args.trial]
    if not matches:
        raise SystemExit(f"Trial {args.trial} not found in study")
    trial = matches[0]

    val = trial.value if trial.value is not None else None
    if val is not None and not is_max and val < 0:
        val_auc = -val
    else:
        val_auc = val

    params = trial.params
    cfg = {
        "dataset": {
            "max_walk_length": params.get("dataset.max_walk_length"),
            "num_walks": params.get("dataset.num_walks"),
        },
        "model": {
            "embedding_dim": params.get("model.embedding_dim"),
            "hidden_dim": params.get("model.hidden_dim"),
            "nhead": params.get("model.nhead"),
            "nlayers": params.get("model.nlayers"),
            "dropout": params.get("model.dropout"),
        },
        "training": {
            "batch_size": params.get("training.batch_size"),
            "epochs": params.get("training.epochs"),
            "lr": params.get("training.lr"),
            "weight_decay": params.get("training.weight_decay"),
            "gradient_clip_val": params.get("training.gradient_clip_val"),
            "early_stopping_patience": params.get("training.early_stopping_patience"),
        },
    }

    print(f"# Trial {args.trial} (AUC={val_auc})")
    print(yaml.safe_dump(cfg, sort_keys=False))

    if args.dotlist:
        dot_items = []
        for section, fields in cfg.items():
            for k, v in fields.items():
                if v is not None:
                    dot_items.append(f"{section}.{k}={v}")
        print("# Dotlist overrides:")
        print(" ".join(dot_items))


if __name__ == "__main__":
    main()
