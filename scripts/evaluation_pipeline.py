#!/usr/bin/env python3
"""
Complete pipeline for evaluation with trained models and cached datasets.
Supports:
1. Loading best trained models with Optuna hyperparams
2. Using cached preprocessed datasets to avoid rebuilding
3. Aggregating edge predictions across multiple walks
"""
import os
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import Trainer
import optuna

from src.utils.config import load_config
from src.model.lit_model import LitEdgeClassifier
from src.data.prepare_data import prepare_data
from optuna.storages import JournalStorage, JournalFileStorage


class EvaluationPipeline:
    """Complete pipeline for model evaluation with score aggregation."""

    def __init__(
        self,
        dataset_name,
        outputs_root="/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs",
    ):
        self.dataset_name = dataset_name
        self.outputs_root = outputs_root
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.best_trial = None
        self.cfg = None

    def find_best_study(self):
        """Locate best study for dataset."""
        root = Path(self.outputs_root) / self.dataset_name
        best_score = -float("inf")
        best_study_path = None

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
                completed = [t for t in study.trials if t.state.name == "COMPLETE"]
                if not completed:
                    continue

                if study.best_value > best_score:
                    best_score = study.best_value
                    best_study_path = journal_path
                    self.best_trial = study.best_trial
                    self.best_exp = exp_folder.name
                    self.study = study
            except Exception as e:
                print(f"  Skipped {exp_folder.name}: {e}")
                continue

        if not best_study_path:
            raise ValueError(f"No valid studies found for {self.dataset_name}")

        print(f"✅ Found best study: {self.best_exp}")
        print(f"   Trial #{self.best_trial.number}, score: {self.study.best_value:.6f}")
        return self.best_trial

    def load_model(self, device=2):
        """Load best model checkpoint."""
        if not self.best_trial:
            self.find_best_study()

        # Find checkpoint
        root = Path(self.outputs_root) / self.dataset_name / self.best_exp
        ckpt_dir = root / "checkpoints"
        pattern = f"trial_{self.best_trial.number}-*.ckpt"
        matches = list(ckpt_dir.glob(pattern))

        if not matches:
            raise FileNotFoundError(
                f"No checkpoint found for trial {self.best_trial.number}"
            )

        ckpt_path = max(matches, key=lambda p: p.stat().st_mtime)

        # Build config with best hyperparams
        self.cfg = load_config(config_path="config.yaml")
        self.cfg.dataset.name = self.dataset_name

        for key, val in self.best_trial.params.items():
            keys = key.split(".")
            node = self.cfg
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = val

        # Set device
        self.cfg.device = device

        # Load model
        self.model = LitEdgeClassifier.load_from_checkpoint(
            str(ckpt_path), config=self.cfg
        )
        self.model.eval()
        self.model = self.model.to(device)

        print(f"✅ Loaded model from {ckpt_path}")
        return self.model

    def prepare_data(self, use_cache=True):
        """Prepare data with caching."""
        if not self.cfg:
            # Load config with best params if not already loaded
            self.cfg = load_config(config_path="config.yaml")
            self.cfg.dataset.name = self.dataset_name
            if self.best_trial:
                for key, val in self.best_trial.params.items():
                    keys = key.split(".")
                    node = self.cfg
                    for k in keys[:-1]:
                        node = node[k]
                    node[keys[-1]] = val

        # Enable caching
        self.cfg.preprocess.use_cache = use_cache

        print(f"🔄 Preparing data (cache: {use_cache})...")
        self.train_loader, self.val_loader, self.test_loader = prepare_data(self.cfg)
        print(f"✅ Data ready")

        return self.train_loader, self.val_loader, self.test_loader

    def evaluate_on_test_set(self):
        """Evaluate model on test set."""
        if not self.test_loader:
            self.prepare_data(use_cache=True)

        if not self.model:
            self.load_model()

        print(f"\n📊 Evaluating on test set...")

        all_preds = []
        all_labels = []
        all_losses = []

        with torch.no_grad():
            for batch in self.test_loader:
                # Move batch to device
                batch = {
                    k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                outputs = self.model(batch)

                # Get predictions
                logits = outputs.get("logits", None)
                if logits is not None:
                    preds = logits.argmax(dim=-1)
                    all_preds.append(preds.cpu())

                # Get labels
                labels = batch.get("labels", None)
                if labels is not None:
                    all_labels.append(labels.cpu())

        if all_preds and all_labels:
            preds = torch.cat(all_preds, dim=0)
            labels = torch.cat(all_labels, dim=0)

            accuracy = (preds == labels).float().mean().item()
            print(f"   Test Accuracy: {accuracy:.6f}")

            return {"accuracy": accuracy, "preds": preds, "labels": labels}
        else:
            print(f"   No test data available")
            return None


def aggregate_edge_scores_across_walks(predictions_per_walk, aggregation_method="mean"):
    """
    Aggregate edge predictions across multiple walks.

    Args:
        predictions_per_walk: dict of {edge_id: [scores_from_walk1, scores_from_walk2, ...]}
        aggregation_method: "mean", "max", "majority_vote", or "weighted_by_walk_length"

    Returns:
        dict of {edge_id: aggregated_score}
    """
    aggregated = {}

    for edge_id, scores in predictions_per_walk.items():
        if not scores:
            continue

        scores = np.array(scores)

        if aggregation_method == "mean":
            aggregated[edge_id] = scores.mean()
        elif aggregation_method == "max":
            aggregated[edge_id] = scores.max()
        elif aggregation_method == "majority_vote":
            # For multi-class, use mode
            unique, counts = np.unique(scores, return_counts=True)
            aggregated[edge_id] = unique[counts.argmax()]
        elif aggregation_method == "weighted_by_walk_length":
            # Weight by position in walk (edges near middle of walk might be more reliable)
            weights = np.linspace(0.5, 1.0, len(scores))
            weights /= weights.sum()
            aggregated[edge_id] = (scores * weights).sum()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    return aggregated


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python evaluation_pipeline.py <dataset_name> [--device 2]")
        print("\nExamples:")
        print("  python evaluation_pipeline.py wiki-rfa")
        print("  python evaluation_pipeline.py epinions --device 3")
        sys.exit(1)

    dataset = sys.argv[1]
    device = 2

    if "--device" in sys.argv:
        device = int(sys.argv[sys.argv.index("--device") + 1])

    try:
        pipe = EvaluationPipeline(dataset)
        pipe.find_best_study()
        pipe.load_model(device=device)
        pipe.prepare_data(use_cache=True)
        results = pipe.evaluate_on_test_set()

        if results:
            print(f"\n✅ Evaluation complete for {dataset}")
            print(f"   Results: {results}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
