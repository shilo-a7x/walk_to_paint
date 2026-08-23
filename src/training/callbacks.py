"""
Training callbacks for saving per-epoch predictions with walk metadata.

Saves predictions for train/val/test splits every epoch, including:
- Edge IDs (for aggregator)
- Walk context (position, length)
- Predictions with probabilities
- Distance metrics (for heatmaps)
"""

import os
import pickle
import torch
import numpy as np
from pytorch_lightning import Callback
from sklearn.metrics import roc_auc_score


class PerEpochPredictionSaver(Callback):
    """Save predictions from all splits (train/val/test) every epoch.

    Saves walk-level predictions with metadata needed for:
    - Heatmap analysis (distance from start/end)
    - Aggregator training (multiple predictions per edge)
    - Position-aware analysis
    """

    def __init__(self, cfg, data_module):
        """
        Args:
            cfg: OmegaConf config object
            data_module: Dict with 'train', 'val', 'test' DataLoaders
        """
        self.cfg = cfg
        self.data_module = data_module

        callback_cfg = getattr(cfg.training, "callbacks", None)
        self.save_every = int(getattr(callback_cfg, "save_every", 0))

        # Create output directory
        self.predictions_dir = os.path.join(
            cfg.training.checkpoint_dir, f"{cfg.dataset.name}_predictions"
        )
        os.makedirs(self.predictions_dir, exist_ok=True)

        print(f"✓ PerEpochPredictionSaver initialized")
        print(f"  Predictions will be saved to: {self.predictions_dir}")
        if self.save_every > 0:
            print(f"  save_every={self.save_every} (+ always last epoch)")

    def _extract_predictions(self, trainer, pl_module, dataloader, split_name):
        """Extract predictions from a dataloader with full walk metadata.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: LitEdgeClassifier model
            dataloader: DataLoader to process
            split_name: 'train', 'val', or 'test'

        Returns:
            dict with all prediction data including walk metadata
        """
        pl_module.eval()
        device = pl_module.device

        all_data = {
            "edge_ids": [],
            "walk_ids": [],
            "positions": [],
            "walk_lengths": [],
            "predictions": [],
            "probabilities": [],
            "targets": [],
            "correct": [],
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Unpack batch (always 4-tuple with metadata)
                input_ids, labels, attention_mask, metadata = batch

                input_ids = input_ids.to(device)
                labels = labels.to(device)
                attention_mask = attention_mask.to(device)

                # Reapply any token-masking ablation (mask_node_tokens/mask_edge_tokens)
                # the model was actually trained under -- this is a no-op unless one of
                # those cfg flags is set, matching LitEdgeClassifier._step exactly. Bug
                # fixed 2026-08-23: this call was previously missing, so ablated
                # checkpoints were trained blind to a token role but evaluated with it
                # fully visible -- a real train/eval mismatch, not a leak during
                # training. See aaai2027/PAPER_CLOSEOUT_LOG.md's 2026-08-23 entry.
                if hasattr(pl_module, "_maybe_apply_token_masking"):
                    positions_for_mask = metadata.get("positions")
                    if positions_for_mask is not None:
                        positions_for_mask = positions_for_mask.to(device)
                        node_mask_ablation = (positions_for_mask >= 0) & ((positions_for_mask % 2) == 0)
                        edge_mask_ablation = (positions_for_mask >= 0) & ((positions_for_mask % 2) == 1)
                        input_ids = pl_module._maybe_apply_token_masking(
                            input_ids, node_mask_ablation, edge_mask_ablation
                        )

                # Forward pass
                logits = pl_module.model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)

                # Flatten tensors
                preds_flat = preds.view(-1)
                probs_flat = probs.view(-1, probs.size(-1))
                targets_flat = labels.view(-1)

                # Filter out ignore_index on GPU
                ignore_idx = pl_module.cfg.model.ignore_index
                valid_mask = targets_flat != ignore_idx

                # Apply mask on GPU
                preds_valid = preds_flat[valid_mask]
                probs_valid = probs_flat[valid_mask]
                targets_valid = targets_flat[valid_mask]
                correct_valid = preds_valid == targets_valid

                # Transfer to CPU in one batch (optimized)
                preds_cpu = preds_valid.cpu().numpy()
                probs_cpu = probs_valid.cpu().numpy()
                targets_cpu = targets_valid.cpu().numpy()
                correct_cpu = correct_valid.cpu().numpy()

                # Store predictions
                all_data["predictions"].append(preds_cpu)
                all_data["probabilities"].append(probs_cpu)
                all_data["targets"].append(targets_cpu)
                all_data["correct"].append(correct_cpu)

                # Store metadata (transfer to CPU in batch)
                # Apply valid_mask on CPU after transfer for metadata
                # In ragged mode walk_ids/walk_lengths are [B]; expand to [B, S] first.
                _B, _S = labels.shape
                _walk_ids = metadata["walk_ids"]
                _walk_lengths = metadata["walk_lengths"]
                if _walk_ids.dim() == 1:
                    _walk_ids = _walk_ids.unsqueeze(1).expand(_B, _S)
                if _walk_lengths.dim() == 1:
                    _walk_lengths = _walk_lengths.unsqueeze(1).expand(_B, _S)
                edge_ids_cpu = metadata["edge_ids"].view(-1).cpu().numpy()
                walk_ids_cpu = _walk_ids.reshape(-1).cpu().numpy()
                positions_cpu = metadata["positions"].view(-1).cpu().numpy()
                walk_lengths_cpu = _walk_lengths.reshape(-1).cpu().numpy()
                valid_mask_cpu = valid_mask.cpu().numpy()

                # Convert sequence positions/lengths to EDGE counts
                # Walk structure: [node, edge, node, edge, ..., node]
                # Edge positions are at odd indices (1, 3, 5, ...), so
                # position_in_edges = (position - 1) // 2
                # total_edges = (walk_length - 1) // 2
                pos_seq = positions_cpu[valid_mask_cpu]
                len_seq = walk_lengths_cpu[valid_mask_cpu]
                pos_edges = (pos_seq - 1) // 2
                len_edges = (len_seq - 1) // 2

                all_data["edge_ids"].append(edge_ids_cpu[valid_mask_cpu])
                all_data["walk_ids"].append(walk_ids_cpu[valid_mask_cpu])
                all_data["positions"].append(pos_edges)
                all_data["walk_lengths"].append(len_edges)

        # Concatenate all batches
        result = {
            "epoch": trainer.current_epoch,
            "split": split_name,
            "edge_ids": np.concatenate(all_data["edge_ids"]),
            "walk_ids": np.concatenate(all_data["walk_ids"]),
            "positions": np.concatenate(all_data["positions"]),
            "walk_lengths": np.concatenate(all_data["walk_lengths"]),
            "predictions": np.concatenate(all_data["predictions"]),
            "probabilities": np.concatenate(all_data["probabilities"]),
            "targets": np.concatenate(all_data["targets"]),
            "correct": np.concatenate(all_data["correct"]),
        }

        # Compute distance metrics (for heatmaps) in EDGE counts
        result["dist_from_start"] = result["positions"]
        result["dist_from_end"] = result["walk_lengths"] - result["positions"] - 1

        # Compute AUC for this split
        try:
            # Check if we have both classes
            unique_classes = np.unique(result["targets"])
            n_classes = result["probabilities"].shape[1]

            if len(unique_classes) < 2:
                print(
                    f"  ⚠️  Cannot compute AUC for {split_name}: only {len(unique_classes)} class(es) present"
                )
                result["auc"] = None
            elif n_classes == 2:
                # Binary classification: use probabilities for positive class
                auc = roc_auc_score(result["targets"], result["probabilities"][:, 1])
                result["auc"] = float(auc)
            else:
                # Multi-class: use one-vs-rest strategy
                auc = roc_auc_score(
                    result["targets"],
                    result["probabilities"],
                    multi_class="ovr",
                    average="weighted",
                )
                result["auc"] = float(auc)
        except Exception as e:
            print(f"  ⚠️  AUC computation failed for {split_name}: {str(e)}")
            result["auc"] = None

        return result

    def _save_predictions(self, predictions_data, epoch, split_name):
        """Save predictions to disk."""
        epoch_dir = os.path.join(self.predictions_dir, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        filepath = os.path.join(epoch_dir, f"{split_name}_predictions.pkl")

        with open(filepath, "wb") as f:
            pickle.dump(predictions_data, f)

        n_occurrences = len(predictions_data["edge_ids"])
        n_unique_edges = len(np.unique(predictions_data["edge_ids"]))
        auc = predictions_data.get("auc", None)

        if auc is not None:
            print(
                f"  ✓ Saved {split_name} predictions: {n_occurrences} occurrences, "
                f"{n_unique_edges} unique edges, AUC={auc:.4f}"
            )
        else:
            print(
                f"  ✓ Saved {split_name} predictions: {n_occurrences} occurrences, "
                f"{n_unique_edges} unique edges, AUC=N/A"
            )

    def _should_save(self, trainer) -> bool:
        """Return True if this epoch should trigger a save.

        save_every=0  → every epoch (default, backward-compatible)
        save_every=N  → every Nth epoch (1-based) and always on the final epoch
        """
        if self.save_every <= 0:
            return True
        epoch = trainer.current_epoch  # 0-based
        is_last = (epoch + 1) >= trainer.max_epochs
        return is_last or ((epoch + 1) % self.save_every == 0)

    def on_train_epoch_end(self, trainer, pl_module):
        """Save train predictions after each eligible epoch."""
        if not self._should_save(trainer):
            return
        print(f"\nEpoch {trainer.current_epoch}: Saving train predictions...")
        predictions = self._extract_predictions(
            trainer, pl_module, self.data_module["train"], "train"
        )
        self._save_predictions(predictions, trainer.current_epoch, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Save val predictions after each eligible epoch."""
        if not self._should_save(trainer):
            return
        print(f"Epoch {trainer.current_epoch}: Saving val predictions...")
        predictions = self._extract_predictions(
            trainer, pl_module, self.data_module["val"], "val"
        )
        self._save_predictions(predictions, trainer.current_epoch, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        """Save test predictions after test run."""
        if not self._should_save(trainer):
            return
        print(f"Epoch {trainer.current_epoch}: Saving test predictions...")
        predictions = self._extract_predictions(
            trainer, pl_module, self.data_module["test"], "test"
        )
        self._save_predictions(predictions, trainer.current_epoch, "test")


class PerEpochTestRunner(Callback):
    """Run test loop and save predictions after each validation epoch (not just at the end)."""

    def __init__(self, data_module, prediction_saver=None):
        """
        Args:
            data_module: Dict with 'test' DataLoader
            prediction_saver: PerEpochPredictionSaver instance for saving predictions
        """
        self.test_dataloader = data_module["test"]
        self.prediction_saver = prediction_saver
        save_every = prediction_saver.save_every if prediction_saver is not None else 0
        freq = (
            "every epoch" if save_every <= 0 else f"every {save_every} epoch(s) + last"
        )
        print(f"✓ PerEpochTestRunner initialized - test will run {freq}")

    def _should_run(self, trainer) -> bool:
        """Mirror PerEpochPredictionSaver._should_save logic."""
        if self.prediction_saver is None:
            return True
        return self.prediction_saver._should_save(trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Run test loop after validation completes (only on eligible epochs)."""
        # Skip during sanity check
        if trainer.sanity_checking:
            return

        if not self._should_run(trainer):
            return

        # Manually run inference on test set instead of calling trainer.test()
        # to avoid logging issues during fit loop
        pl_module.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_dataloader):
                # Move batch to device (always 4-tuple with metadata)
                input_ids, labels, attention_mask, metadata = batch
                input_ids = input_ids.to(pl_module.device)
                labels = labels.to(pl_module.device)
                attention_mask = attention_mask.to(pl_module.device)
                batch = (input_ids, labels, attention_mask, metadata)

                # Call test_step
                pl_module.test_step(batch, batch_idx)

        # Call on_test_epoch_end to compute metrics, log to TensorBoard, and create plots
        # This ensures test gets same treatment as train/val (confusion matrix, ROC curves, etc.)
        pl_module.on_test_epoch_end()

        # Save test predictions if prediction saver provided
        if self.prediction_saver is not None:
            print(f"Epoch {trainer.current_epoch}: Saving test predictions...")
            predictions = self.prediction_saver._extract_predictions(
                trainer, pl_module, self.test_dataloader, "test"
            )
            self.prediction_saver._save_predictions(
                predictions, trainer.current_epoch, "test"
            )

        pl_module.train()
