import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from src.model.model import TransformerModel
from src.model.metrics_helper import MetricsManager, PlottingHelper


class LitEdgeClassifier(pl.LightningModule):
    def __init__(self, cfg=None):
        super().__init__()

        # Allow loading from checkpoint without passing cfg manually
        if cfg is None:
            if hasattr(self, "hparams") and "cfg" in self.hparams:
                cfg = OmegaConf.create(self.hparams["cfg"])
            else:
                raise ValueError("cfg is required when not loading from checkpoint")

        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        # Snapshot full resolved config so checkpoints are self-describing
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=True)})
        self.model = TransformerModel(cfg)
        self.cfg = cfg
        self.ignore_index = cfg.model.ignore_index
        self.num_classes = cfg.model.num_classes

        # Get class weights from config (computed during data preparation)
        if hasattr(cfg.model, "class_weights") and cfg.model.class_weights is not None:
            self.class_weights = torch.tensor(
                cfg.model.class_weights, dtype=torch.float32
            )
            print(f"✓ Using class weights from config: {self.class_weights.tolist()}")
        else:
            # Fallback for checkpoint loading or when weights not computed
            self.class_weights = torch.ones(self.num_classes, dtype=torch.float32)
            print("⚠️  Using uniform weights (no class_weights in config)")

        # Initialize metrics manager
        self.metrics_manager = MetricsManager(self.num_classes, self.ignore_index)

        # Initialize helper classes
        self.plotting_helper = PlottingHelper()

    def forward(self, input_ids):
        return self.model(input_ids)

    def _step(self, batch, stage: str):
        # Unpack batch (always 4-tuple with metadata)
        input_ids, labels, attention_mask, metadata = batch
        # metadata contains: edge_ids, walk_ids, positions, walk_lengths
        # Store for callback access if needed
        self._last_metadata = metadata

        # Check if all labels are ignore_index
        if torch.all(labels == self.cfg.model.ignore_index):
            # Skip the batch completely if all labels are ignore_index
            return None  # Returning None indicates that no loss/metrics are computed

        logits = self.model(input_ids, attention_mask=attention_mask)

        # Compute loss with MANDATORY class weighting (global train-only weights)
        # Same weights used for train, val, test - no data leakage
        weights = (
            self.class_weights.to(logits.device)
            if self.class_weights is not None
            else None
        )
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            weight=weights,
            ignore_index=self.ignore_index,
        )

        preds = logits.argmax(dim=-1).view(-1)
        targets = labels.view(-1)
        probs = torch.softmax(logits, dim=-1).view(-1, logits.size(-1))

        # Update metrics using the metrics manager
        self.metrics_manager.update_metrics(stage, preds, targets, probs)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.training.epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_train_epoch_end(self):
        self.log("step", self.current_epoch)
        results = self.metrics_manager.compute_and_reset_metrics("train")

        self.log("train_acc_epoch", results["accuracy"], prog_bar=True)
        self.log("train_f1_epoch", results["f1"], prog_bar=True)
        self.log("train_auc_epoch", results["auc"], prog_bar=True)

        # Log confusion matrix (only if logger is available)
        if self.logger is not None:
            self.logger.experiment.add_figure(
                "train_confusion_matrix",
                self.plotting_helper.plot_confusion_matrix(
                    results["confmat"], "Training", self.num_classes
                ),
                self.current_epoch,
            )

        # Log ROC curve
        targets_list, probs_list = self.metrics_manager.get_roc_data("train")
        if targets_list and probs_list:
            roc_fig = self.plotting_helper.plot_roc_curve(
                targets_list, probs_list, "Training", self.num_classes
            )
            if roc_fig and self.logger is not None:
                self.logger.experiment.add_figure(
                    "train_roc_curve",
                    roc_fig,
                    self.current_epoch,
                )

        # Reset ROC data
        self.metrics_manager.reset_roc_data("train")

    def on_validation_epoch_end(self):
        # Skip logging during sanity check
        if self.trainer.sanity_checking:
            return

        self.log("step", self.current_epoch)
        results = self.metrics_manager.compute_and_reset_metrics("val")

        self.log("val_acc_epoch", results["accuracy"], prog_bar=True)
        self.log("val_f1_epoch", results["f1"], prog_bar=True)
        self.log("val_auc_epoch", results["auc"], prog_bar=True)

        # Log confusion matrix (only if logger is available)
        if self.logger is not None:
            self.logger.experiment.add_figure(
                "val_confusion_matrix",
                self.plotting_helper.plot_confusion_matrix(
                    results["confmat"], "Validation", self.num_classes
                ),
                self.current_epoch,
            )

        # Log ROC curve
        targets_list, probs_list = self.metrics_manager.get_roc_data("val")
        if targets_list and probs_list:
            roc_fig = self.plotting_helper.plot_roc_curve(
                targets_list, probs_list, "Validation", self.num_classes
            )
            if roc_fig and self.logger is not None:
                self.logger.experiment.add_figure(
                    "val_roc_curve",
                    roc_fig,
                    self.current_epoch,
                )

        # Reset ROC data
        self.metrics_manager.reset_roc_data("val")

    def on_test_epoch_end(self):
        self.log("step", self.current_epoch)
        results = self.metrics_manager.compute_and_reset_metrics("test")

        self.log("test_acc_epoch", results["accuracy"], prog_bar=True)
        self.log("test_f1_epoch", results["f1"], prog_bar=True)
        self.log("test_auc_epoch", results["auc"], prog_bar=True)

        # Log confusion matrix (only if logger is available)
        if self.logger is not None:
            self.logger.experiment.add_figure(
                "test_confusion_matrix",
                self.plotting_helper.plot_confusion_matrix(
                    results["confmat"], "Test", self.num_classes
                ),
                self.current_epoch,
            )

        # Log ROC curve
        targets_list, probs_list = self.metrics_manager.get_roc_data("test")
        if targets_list and probs_list:
            roc_fig = self.plotting_helper.plot_roc_curve(
                targets_list, probs_list, "Test", self.num_classes
            )
            if roc_fig and self.logger is not None:
                self.logger.experiment.add_figure(
                    "test_roc_curve",
                    roc_fig,
                    self.current_epoch,
                )

        # Reset ROC data
        self.metrics_manager.reset_roc_data("test")
