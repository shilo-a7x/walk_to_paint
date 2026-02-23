#!/usr/bin/env python3
"""
Train an edge-level aggregator using per-walk triplets:
    (dist_from_start, walk_length, predicted_prob)

Model is trained on walk-level samples and aggregated to per-edge scores
by averaging predicted probabilities across walks of the same edge.

Splits:
    - Agg train: 100% of transformer val edges
    - Agg test: 100% of transformer test edges
"""

import argparse
import json
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb


def _aggregate_edge_probs(edge_ids, probs, y):
    edge_to_probs = {}
    edge_to_labels = {}
    for eid, p, label in zip(edge_ids, probs, y):
        edge_to_probs.setdefault(int(eid), []).append(float(p))
        edge_to_labels.setdefault(int(eid), []).append(int(label))

    edge_ids_out = []
    edge_probs_out = []
    edge_labels_out = []
    for eid, plist in edge_to_probs.items():
        edge_ids_out.append(eid)
        edge_probs_out.append(float(np.mean(plist)))
        labels = np.array(edge_to_labels[eid], dtype=int)
        edge_labels_out.append(int(np.bincount(labels).argmax()))

    return np.array(edge_ids_out), np.array(edge_probs_out), np.array(edge_labels_out)


def main():
    parser = argparse.ArgumentParser(
        description="Train edge-level aggregator with 2 features"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input features pickle from agg_build_features.py",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for model and summary"
    )
    parser.add_argument(
        "--model", choices=["logistic", "xgboost", "lgbm"], default="logistic"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    val_dict = data["val"]
    test_dict = data["test"]

    # Use val_dict as train, test_dict as test (no validation split)
    X_train = val_dict["X"]
    y_train = val_dict["y"]
    train_edge_ids = val_dict["edge_ids"]

    X_test = test_dict["X"]
    y_test = test_dict["y"]
    test_edge_ids = test_dict["edge_ids"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Compute class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"\nClass distribution (train):")
    for cls, cnt in class_dist.items():
        print(f"  Class {cls}: {cnt} ({100*cnt/len(y_train):.1f}%)")

    if args.model == "logistic":
        model = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=args.seed
        )
        model.fit(X_train_s, y_train)
        model_config = {
            "model_name": "logistic",
            "seed": args.seed,
            "feature_names": ["dist_from_start", "walk_length", "predicted_prob"],
            "scaler": "StandardScaler",
            "params": {
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": args.seed,
            },
        }
    elif args.model == "xgboost":
        # XGBoost with scale_pos_weight to handle class imbalance
        scale_pos_weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=args.seed,
            verbosity=1,
        )
        model.fit(X_train_s, y_train)
        model_config = {
            "model_name": "xgboost",
            "seed": args.seed,
            "feature_names": ["dist_from_start", "walk_length", "predicted_prob"],
            "scaler": "StandardScaler",
            "params": {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "scale_pos_weight": scale_pos_weight,
                "random_state": args.seed,
                "verbosity": 1,
            },
        }
    else:
        model = lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=args.seed,
            verbose=-1,
        )
        model.fit(X_train_s, y_train)
        model_config = {
            "model_name": "lgbm",
            "seed": args.seed,
            "feature_names": ["dist_from_start", "walk_length", "predicted_prob"],
            "scaler": "StandardScaler",
            "params": {
                "n_estimators": 200,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "class_weight": "balanced",
                "random_state": args.seed,
                "verbose": -1,
            },
        }

    # Evaluate on scaled sets
    train_probs = model.predict_proba(X_train_s)[:, 1]
    test_probs = model.predict_proba(X_test_s)[:, 1]

    train_preds = (train_probs >= 0.5).astype(int)
    test_preds = (test_probs >= 0.5).astype(int)

    metrics_train = {
        "acc": accuracy_score(y_train, train_preds),
        "f1": f1_score(y_train, train_preds),
        "auc": roc_auc_score(y_train, train_probs),
    }
    metrics_test = {
        "acc": accuracy_score(y_test, test_preds),
        "f1": f1_score(y_test, test_preds),
        "auc": roc_auc_score(y_test, test_probs),
    }

    # Aggregate to edge-level scores
    _, train_edge_probs, train_edge_labels = _aggregate_edge_probs(
        train_edge_ids, train_probs, y_train
    )
    _, test_edge_probs, test_edge_labels = _aggregate_edge_probs(
        test_edge_ids, test_probs, y_test
    )

    agg_train_auc = roc_auc_score(train_edge_labels, train_edge_probs)
    agg_test_auc = roc_auc_score(test_edge_labels, test_edge_probs)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "model.pkl"), "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Aggregator (walk-level triplets)\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Class weighting: balanced\n\n")
        f.write(f"Train samples: {len(y_train)} walks\n")
        f.write(f"Test samples: {len(y_test)} walks\n\n")
        f.write("Walk-level metrics:\n")
        f.write(f"  Train AUC: {metrics_train['auc']:.4f}\n")
        f.write(f"  Test  AUC: {metrics_test['auc']:.4f}\n\n")
        f.write("Edge-level aggregated AUC (mean prob per edge):\n")
        f.write(f"  Train AUC: {agg_train_auc:.4f} ({len(train_edge_probs)} edges)\n")
        f.write(f"  Test  AUC: {agg_test_auc:.4f} ({len(test_edge_probs)} edges)\n")

    print(f"Saved model: {args.output_dir}/model.pkl")
    print(f"Saved model config: {args.output_dir}/model_config.json")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
