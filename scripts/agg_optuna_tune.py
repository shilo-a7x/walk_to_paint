#!/usr/bin/env python3
"""
Optuna-based hyperparameter tuning for aggregator models.

Uses 80/20 split of transformer val set for agg train/val.
Tests on transformer test set.

Optimizes: Logistic Regression, XGBoost, LightGBM
"""

import argparse
import json
import os
import pickle

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb


def _split_edges(edges, train_ratio=0.8, seed=42):
    """Split edges for train/val."""
    rng = np.random.RandomState(seed)
    edges = list(edges)
    rng.shuffle(edges)
    split_idx = int(train_ratio * len(edges))
    return edges[:split_idx], edges[split_idx:]


def _filter_by_edges(edge_ids, X, y, allowed_edges):
    """Filter walks belonging to allowed edges."""
    mask = np.isin(edge_ids, list(allowed_edges))
    return X[mask], y[mask], edge_ids[mask]


def _aggregate_edge_probs(edge_ids, probs, y):
    """Aggregate walk-level probs to edge-level."""
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


class LogisticObjective:
    def __init__(self, X_train, y_train, X_val, y_val, val_edge_ids):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.val_edge_ids = val_edge_ids

    def __call__(self, trial):
        C = trial.suggest_float("C", 1e-4, 1e2, log=True)
        max_iter = trial.suggest_int("max_iter", 500, 2000)

        model = LogisticRegression(
            C=C, max_iter=max_iter, class_weight="balanced", random_state=42
        )
        model.fit(self.X_train, self.y_train)

        val_probs = model.predict_proba(self.X_val)[:, 1]
        _, val_edge_probs, val_edge_labels = _aggregate_edge_probs(
            self.val_edge_ids, val_probs, self.y_val
        )
        auc = roc_auc_score(val_edge_labels, val_edge_probs)
        return auc


class XGBObjective:
    def __init__(self, X_train, y_train, X_val, y_val, val_edge_ids):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.val_edge_ids = val_edge_ids
        self.scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

    def __call__(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=self.scale_pos_weight,
            random_state=42,
            verbosity=0,
        )
        model.fit(self.X_train, self.y_train)

        val_probs = model.predict_proba(self.X_val)[:, 1]
        _, val_edge_probs, val_edge_labels = _aggregate_edge_probs(
            self.val_edge_ids, val_probs, self.y_val
        )
        auc = roc_auc_score(val_edge_labels, val_edge_probs)
        return auc


class LGBMObjective:
    def __init__(self, X_train, y_train, X_val, y_val, val_edge_ids):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.val_edge_ids = val_edge_ids

    def __call__(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        num_leaves = trial.suggest_int("num_leaves", 20, 100)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        feature_fraction = trial.suggest_float("feature_fraction", 0.5, 1.0)
        bagging_fraction = trial.suggest_float("bagging_fraction", 0.5, 1.0)

        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        model.fit(self.X_train, self.y_train)

        val_probs = model.predict_proba(self.X_val)[:, 1]
        _, val_edge_probs, val_edge_labels = _aggregate_edge_probs(
            self.val_edge_ids, val_probs, self.y_val
        )
        auc = roc_auc_score(val_edge_labels, val_edge_probs)
        return auc


def main():
    parser = argparse.ArgumentParser(description="Tune aggregator models with Optuna")
    parser.add_argument(
        "--input",
        required=True,
        help="Input features pickle from agg_build_features.py",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for models and summary"
    )
    parser.add_argument(
        "--n-trials", type=int, default=30, help="Number of Optuna trials per model"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    val_dict = data["val"]
    test_dict = data["test"]

    # Split val edges 80/20 for agg train/val
    val_edge_ids = np.unique(val_dict["edge_ids"])
    train_edges, val_edges = _split_edges(val_edge_ids, train_ratio=0.8, seed=args.seed)

    X_train, y_train, train_edge_ids = _filter_by_edges(
        val_dict["edge_ids"], val_dict["X"], val_dict["y"], train_edges
    )
    X_val, y_val, val_edge_ids_filtered = _filter_by_edges(
        val_dict["edge_ids"], val_dict["X"], val_dict["y"], val_edges
    )
    X_test, y_test, test_edge_ids = _filter_by_edges(
        test_dict["edge_ids"],
        test_dict["X"],
        test_dict["y"],
        np.unique(test_dict["edge_ids"]),
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    os.makedirs(args.output_dir, exist_ok=True)

    # Tune each model
    models_config = {
        "logistic": LogisticObjective(
            X_train_s, y_train, X_val_s, y_val, val_edge_ids_filtered
        ),
        "xgboost": XGBObjective(
            X_train_s, y_train, X_val_s, y_val, val_edge_ids_filtered
        ),
        "lgbm": LGBMObjective(
            X_train_s, y_train, X_val_s, y_val, val_edge_ids_filtered
        ),
    }

    results = {}
    model_configs = {}

    for model_name, objective in models_config.items():
        print(f"\n{'='*70}")
        print(f"Tuning {model_name.upper()} (n_trials={args.n_trials})")
        print(f"{'='*70}")

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=args.seed),
        )
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_val_auc = study.best_value

        print(f"Best val AUC: {best_val_auc:.4f}")
        print(f"Best params: {best_params}")

        # Train final model on full train+val set with best params
        X_full_train_s = np.vstack([X_train_s, X_val_s])
        y_full_train = np.hstack([y_train, y_val])
        full_edge_ids = np.hstack([train_edge_ids, val_edge_ids_filtered])

        if model_name == "logistic":
            fixed_params = {
                "class_weight": "balanced",
                "random_state": int(args.seed),
            }
            model = LogisticRegression(
                C=best_params["C"],
                max_iter=best_params["max_iter"],
                class_weight=fixed_params["class_weight"],
                random_state=fixed_params["random_state"],
            )
            model.fit(X_full_train_s, y_full_train)
        elif model_name == "xgboost":
            scale_pos_weight = np.sum(y_full_train == 0) / np.sum(y_full_train == 1)
            fixed_params = {
                "scale_pos_weight": float(scale_pos_weight),
                "random_state": int(args.seed),
                "verbosity": 0,
            }
            model = xgb.XGBClassifier(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"],
                learning_rate=best_params["learning_rate"],
                subsample=best_params["subsample"],
                colsample_bytree=best_params["colsample_bytree"],
                scale_pos_weight=fixed_params["scale_pos_weight"],
                random_state=fixed_params["random_state"],
                verbosity=fixed_params["verbosity"],
            )
            model.fit(X_full_train_s, y_full_train)
        else:  # lgbm
            fixed_params = {
                "class_weight": "balanced",
                "random_state": int(args.seed),
                "verbose": -1,
            }
            model = lgb.LGBMClassifier(
                n_estimators=best_params["n_estimators"],
                num_leaves=best_params["num_leaves"],
                learning_rate=best_params["learning_rate"],
                feature_fraction=best_params["feature_fraction"],
                bagging_fraction=best_params["bagging_fraction"],
                class_weight=fixed_params["class_weight"],
                random_state=fixed_params["random_state"],
                verbose=fixed_params["verbose"],
            )
            model.fit(X_full_train_s, y_full_train)

        # Evaluate
        train_probs = model.predict_proba(X_full_train_s)[:, 1]
        test_probs = model.predict_proba(X_test_s)[:, 1]

        _, train_edge_probs, train_edge_labels = _aggregate_edge_probs(
            full_edge_ids, train_probs, y_full_train
        )
        _, test_edge_probs, test_edge_labels = _aggregate_edge_probs(
            test_edge_ids, test_probs, y_test
        )

        train_auc = roc_auc_score(train_edge_labels, train_edge_probs)
        test_auc = roc_auc_score(test_edge_labels, test_edge_probs)

        results[model_name] = {
            "val_auc": best_val_auc,
            "train_auc": train_auc,
            "test_auc": test_auc,
            "params": best_params,
        }

        full_config = {
            "model_name": model_name,
            "seed": int(args.seed),
            "feature_names": ["dist_from_start", "walk_length", "predicted_prob"],
            "scaler": "StandardScaler",
            "tuned_params": best_params,
            "fixed_params": fixed_params,
            "train_context": {
                "n_trials": int(args.n_trials),
                "train_ratio": 0.8,
                "validation_ratio": 0.2,
                "training_source": "transformer_val_split",
                "test_source": "transformer_test_split",
                "n_train_walk_samples": int(len(y_train)),
                "n_val_walk_samples": int(len(y_val)),
                "n_test_walk_samples": int(len(y_test)),
                "n_train_edges": int(len(np.unique(train_edge_ids))),
                "n_val_edges": int(len(np.unique(val_edge_ids_filtered))),
                "n_test_edges": int(len(np.unique(test_edge_ids))),
            },
            "best_metrics": {
                "edge_val_auc": float(best_val_auc),
                "edge_train_auc": float(train_auc),
                "edge_test_auc": float(test_auc),
            },
        }
        model_configs[model_name] = full_config

        print(f"Train AUC (on full train): {train_auc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")

        # Save model
        model_path = os.path.join(args.output_dir, f"{model_name}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler}, f)
        print(f"Saved: {model_path}")

        config_path = os.path.join(args.output_dir, f"{model_name}_config.json")
        with open(config_path, "w") as f:
            json.dump(full_config, f, indent=2)
        print(f"Saved: {config_path}")

    # Summary
    summary_path = os.path.join(args.output_dir, "optuna_summary.txt")
    with open(summary_path, "w") as f:
        f.write("AGGREGATOR OPTUNA TUNING RESULTS\n")
        f.write("=" * 70 + "\n\n")
        for model_name, res in results.items():
            f.write(f"{model_name.upper()}:\n")
            f.write(f"  Val AUC:   {res['val_auc']:.4f}\n")
            f.write(f"  Train AUC: {res['train_auc']:.4f}\n")
            f.write(f"  Test AUC:  {res['test_auc']:.4f}\n")
            f.write(f"  Params:    {res['params']}\n\n")

    json_summary_path = os.path.join(args.output_dir, "optuna_summary.json")
    with open(json_summary_path, "w") as f:
        json.dump(
            {
                "seed": int(args.seed),
                "n_trials": int(args.n_trials),
                "models": model_configs,
            },
            f,
            indent=2,
        )

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved summary JSON: {json_summary_path}")


if __name__ == "__main__":
    main()
