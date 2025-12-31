#!/usr/bin/env python3
"""
Utility to resume Optuna studies and load trained models with hyperparams.
Supports caching datasets and models for fast evaluation.
"""
import os
import json
from pathlib import Path
from optuna.storages import JournalStorage, JournalFileStorage
import optuna
import torch
from pytorch_lightning import Trainer
from src.utils.config import load_config
from src.model.lit_model import LitEdgeClassifier
from src.data.prepare_data import prepare_data


class OptunaBestModel:
    """Load best trial hyperparams and trained model."""

    def __init__(
        self,
        dataset_name,
        outputs_root="/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs",
    ):
        """
        Args:
            dataset_name: "wiki-rfa", "epinions", or "slashdot090221"
            outputs_root: path to outputs folder
        """
        self.dataset_name = dataset_name
        self.outputs_root = outputs_root
        self.best_exp = None
        self.study = None
        self.best_trial = None
        self.hyperparms = None
        self.model = None
        self.train_data = None
        self.val_data = None

    def find_best_study(self):
        """Find the best completed study for this dataset."""
        root = Path(self.outputs_root) / self.dataset_name
        if not root.exists():
            raise ValueError(f"No outputs found for {self.dataset_name} at {root}")

        best_score = -float("inf")
        best_study_path = None
        best_exp_name = None

        for exp_folder in sorted(root.glob(f"{self.dataset_name}-optuna_*")):
            journal_path = exp_folder / "optuna" / "optuna_study.log"
            if not journal_path.exists():
                continue

            try:
                storage = JournalStorage(JournalFileStorage(str(journal_path)))
                summaries = optuna.study.get_all_study_summaries(storage)
                if not summaries:
                    continue

                study = optuna.load_study(
                    study_name=summaries[0].study_name, storage=storage
                )

                # Check if any trials completed
                completed = [t for t in study.trials if t.state.name == "COMPLETE"]
                if not completed:
                    continue

                if study.best_value > best_score:
                    best_score = study.best_value
                    best_study_path = journal_path
                    best_exp_name = exp_folder.name
                    self.study = study

            except Exception as e:
                print(f"  Skipped {exp_folder.name}: {e}")
                continue

        if not best_study_path:
            raise ValueError(f"No valid studies found for {self.dataset_name}")

        self.best_exp = best_exp_name
        self.best_trial = self.study.best_trial
        print(f"✅ Found best study: {best_exp_name}")
        print(
            f"   Best trial: #{self.best_trial.number} with score {self.study.best_value:.6f}"
        )

        return self.best_trial

    def get_hyperparms(self):
        """Extract hyperparams from best trial."""
        if not self.best_trial:
            self.find_best_study()

        self.hyperparms = dict(self.best_trial.params)
        return self.hyperparms

    def get_checkpoint_path(self):
        """Return path to best trial checkpoint."""
        if not self.best_exp:
            self.find_best_study()

        root = Path(self.outputs_root) / self.dataset_name / self.best_exp
        ckpt_dir = root / "checkpoints"

        # Find checkpoint for this trial
        pattern = f"trial_{self.best_trial.number}-*.ckpt"
        matches = list(ckpt_dir.glob(pattern))

        if not matches:
            return None

        # Return newest checkpoint
        newest = max(matches, key=lambda p: p.stat().st_mtime)
        return str(newest)

    def load_model(self, device=2):
        """Load the best model from checkpoint."""
        ckpt_path = self.get_checkpoint_path()
        if not ckpt_path:
            print(f"⚠️  No checkpoint found for trial {self.best_trial.number}")
            return None

        # Build config with best hyperparams
        cfg = load_config(config_path="config.yaml")
        cfg.dataset.name = self.dataset_name

        # Override with best trial hyperparms
        for key, val in self.hyperparms.items():
            keys = key.split(".")
            node = cfg
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = val

        try:
            model = LitEdgeClassifier.load_from_checkpoint(ckpt_path, config=cfg)
            self.model = model
            print(f"✅ Loaded model from {ckpt_path}")
            return model
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None

    def prepare_data(self, use_cache=True, cache_dir=None):
        """Prepare train/val data with optional caching."""
        if not self.hyperparms:
            self.get_hyperparms()

        # Build config
        cfg = load_config(config_path="config.yaml")
        cfg.dataset.name = self.dataset_name
        for key, val in self.hyperparms.items():
            keys = key.split(".")
            node = cfg
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = val

        # Set cache directory if requested
        if use_cache and cache_dir is None:
            cache_dir = (
                Path(self.outputs_root) / self.dataset_name / "preprocessed_cache"
            )
            cache_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data (this will use cache if available)
        # You'll need to add caching logic to prepare_data
        train_loader, val_loader, test_loader = prepare_data(
            cfg, cache_dir=cache_dir if use_cache else None
        )

        self.train_data = train_loader
        self.val_data = val_loader

        return train_loader, val_loader, test_loader


def resume_optuna_study(
    dataset_name,
    n_more_trials=50,
    outputs_root="/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs",
):
    """
    Resume the best Optuna study for a dataset with more trials.

    Usage:
        resume_optuna_study("wiki-rfa", n_more_trials=100)
    """
    best_model = OptunaBestModel(dataset_name, outputs_root)
    best_model.find_best_study()

    # Get the study object
    study = best_model.study

    print(f"\n▶️  Resuming {study.study_name}")
    print(f"   Current trials: {len(study.trials)}")
    print(f"   Running {n_more_trials} more trials...")

    # Use the existing objective from optuna_run.py
    # For now, just show how to resume
    print(f"\nTo resume in optuna_run.py, use:")
    print(f"  python optuna_run.py --config=config.yaml --n-trials={n_more_trials} \\")
    print(f"    dataset.name={dataset_name}")

    return study


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python load_best_model.py <dataset_name> [--load-model] [--resume]")
        print("\nExamples:")
        print("  python load_best_model.py wiki-rfa")
        print("  python load_best_model.py epinions --load-model")
        print("  python load_best_model.py slashdot090221 --resume")
        sys.exit(1)

    dataset = sys.argv[1]
    load_model = "--load-model" in sys.argv
    resume = "--resume" in sys.argv

    try:
        best = OptunaBestModel(dataset)
        best.find_best_study()
        hyperparms = best.get_hyperparms()

        print(f"\n📊 Hyperparameters for {dataset}:")
        for k, v in sorted(hyperparms.items()):
            print(f"   {k}: {v}")

        if load_model:
            print(f"\n🔄 Loading model...")
            model = best.load_model()
            if model:
                print(f"   Model type: {type(model)}")
                print(f"   Model device: {next(model.parameters()).device}")

        if resume:
            print(f"\n▶️  Resuming study with 50 more trials...")
            resume_optuna_study(dataset, n_more_trials=50)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
