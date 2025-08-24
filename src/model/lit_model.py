import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.model import TransformerModel
from src.model.metrics_helper import MetricsManager, PlottingHelper, WeightedLossHelper


class LitEdgeClassifier(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.model = TransformerModel(cfg)
        self.cfg = cfg
        self.ignore_index = cfg.model.ignore_index
        self.num_classes = cfg.model.num_classes

        # Option to enable/disable weighted loss
        self.use_weighted_loss = cfg.training.use_weighted_loss

        # Initialize metrics manager
        self.metrics_manager = MetricsManager(self.num_classes, self.ignore_index)

        # Initialize helper classes
        self.plotting_helper = PlottingHelper()
        self.weighted_loss_helper = WeightedLossHelper()

    def forward(self, input_ids):
        return self.model(input_ids)

    def _step(self, batch, stage: str):
        input_ids, labels, attention_mask = batch

        # Check if all labels are ignore_index
        if torch.all(labels == self.cfg.model.ignore_index):
            # Skip the batch completely if all labels are ignore_index
            return None  # Returning None indicates that no loss/metrics are computed

        logits = self.model(input_ids, attention_mask=attention_mask)

        # Compute loss with optional weighting
        if self.use_weighted_loss:
            # Compute class weights based on batch distribution
            class_weights = self.weighted_loss_helper.compute_class_weights(
                labels.view(-1), self.num_classes, self.ignore_index
            )
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                weight=class_weights,
                ignore_index=self.ignore_index,
            )

            # # Log the weights for monitoring
            # if stage == "train":
            #     for i, weight in enumerate(class_weights):
            #         self.log(
            #             f"class_{i}_weight",
            #             weight,
            #             on_step=False,
            #             on_epoch=True,
            #             prog_bar=False,
            #         )
        else:
            # Standard unweighted loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.ignore_index,
            )

        preds = logits.argmax(dim=-1).view(-1)
        targets = labels.view(-1)
        probs = torch.softmax(logits, dim=-1).view(-1, logits.size(-1))

        # Update metrics using the metrics manager
        self.metrics_manager.update_metrics(stage, preds, targets, probs)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
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
        results = self.metrics_manager.compute_and_reset_metrics("train")

        self.log(
            "train_acc_epoch",
            results["accuracy"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_f1_epoch", results["f1"], prog_bar=True, on_step=False, on_epoch=True
        )
        self.log(
            "train_auc_epoch",
            results["auc"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        # Log confusion matrix
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
            if roc_fig:
                self.logger.experiment.add_figure(
                    "train_roc_curve",
                    roc_fig,
                    self.current_epoch,
                )

        # Reset ROC data
        self.metrics_manager.reset_roc_data("train")

    def on_validation_epoch_end(self):
        results = self.metrics_manager.compute_and_reset_metrics("val")

        self.log(
            "val_acc_epoch",
            results["accuracy"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_f1_epoch", results["f1"], prog_bar=True, on_step=False, on_epoch=True
        )
        self.log(
            "val_auc_epoch", results["auc"], prog_bar=True, on_step=False, on_epoch=True
        )

        # Log confusion matrix
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
            if roc_fig:
                self.logger.experiment.add_figure(
                    "val_roc_curve",
                    roc_fig,
                    self.current_epoch,
                )

        # Reset ROC data
        self.metrics_manager.reset_roc_data("val")

    def on_test_epoch_end(self):
        results = self.metrics_manager.compute_and_reset_metrics("test")

        self.log(
            "test_acc_epoch",
            results["accuracy"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_f1_epoch", results["f1"], prog_bar=True, on_step=False, on_epoch=True
        )
        self.log(
            "test_auc_epoch",
            results["auc"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        # Log confusion matrix
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
            if roc_fig:
                self.logger.experiment.add_figure(
                    "test_roc_curve",
                    roc_fig,
                    self.current_epoch,
                )

        # Reset ROC data
        self.metrics_manager.reset_roc_data("test")
