import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassAUROC,
)


class MetricsManager(nn.Module):
    """Helper class to manage metrics for training, validation, and test phases."""

    def __init__(self, num_classes, ignore_index):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Initialize metrics for each stage
        self.metrics = self._init_metrics()

        # Storage for ROC curve data
        self.roc_data = {
            "train": {"targets": [], "probs": []},
            "val": {"targets": [], "probs": []},
            "test": {"targets": [], "probs": []},
        }

    def _create_stage_metrics(self):
        """Create metrics for a single stage."""
        return nn.ModuleDict(
            {
                "accuracy": MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="macro",
                    ignore_index=self.ignore_index,
                ),
                "f1": MulticlassF1Score(
                    num_classes=self.num_classes,
                    average="macro",
                    ignore_index=self.ignore_index,
                ),
                "confmat": MulticlassConfusionMatrix(
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index,
                ),
                "auc": MulticlassAUROC(
                    num_classes=self.num_classes,
                    average="macro",
                    ignore_index=self.ignore_index,
                ),
            }
        )

    def _init_metrics(self):
        """Initialize all metrics for train, val, and test."""
        # Use stage-specific attributes to avoid conflicts with nn.Module methods
        self.stage_train = self._create_stage_metrics()
        self.stage_val = self._create_stage_metrics()
        self.stage_test = self._create_stage_metrics()

        # Create a mapping for easier access
        return {
            "train": self.stage_train,
            "val": self.stage_val,
            "test": self.stage_test,
        }

    def update_metrics(self, stage, preds, targets, probs):
        """Update metrics for a given stage."""
        self.metrics[stage]["accuracy"].update(preds, targets)
        self.metrics[stage]["f1"].update(preds, targets)
        self.metrics[stage]["confmat"].update(preds, targets)
        self.metrics[stage]["auc"].update(probs, targets)

        # Store ROC data (filter out ignore_index)
        valid_mask = targets != self.ignore_index
        if valid_mask.any():
            self.roc_data[stage]["targets"].append(targets[valid_mask].detach().cpu())
            self.roc_data[stage]["probs"].append(probs[valid_mask].detach().cpu())

    def compute_and_reset_metrics(self, stage):
        """Compute metrics and reset for next epoch."""
        results = {}
        for metric_name, metric in self.metrics[stage].items():
            if metric_name == "confmat":
                results[metric_name] = metric.compute()
            else:
                results[metric_name] = metric.compute()
            metric.reset()

        return results

    def reset_roc_data(self, stage):
        """Reset ROC data for a stage."""
        self.roc_data[stage]["targets"] = []
        self.roc_data[stage]["probs"] = []

    def get_roc_data(self, stage):
        """Get ROC data for a stage."""
        return self.roc_data[stage]["targets"], self.roc_data[stage]["probs"]


class PlottingHelper:
    """Helper class for creating plots."""

    @staticmethod
    def plot_confusion_matrix(confmat, title, num_classes):
        """Plot confusion matrix for tensorboard logging."""
        fig, ax = plt.subplots(figsize=(8, 6))

        # Convert to numpy and normalize
        cm = confmat.cpu().numpy()
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, None]

        # Plot with matplotlib with fixed color scale for consistency across epochs
        im = ax.imshow(
            cm_norm,
            interpolation="nearest",
            cmap="Blues",
            vmin=0.0,  # Fixed minimum for consistent color scale
            vmax=1.0,  # Fixed maximum for consistent color scale
        )
        ax.figure.colorbar(im, ax=ax)

        # Add text annotations with better contrast logic
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = f"{cm[i, j]}\n({cm_norm[i, j]:.2f})"
                # Use white text for dark backgrounds (>0.5), black for light backgrounds
                text_color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=text_color,
                )

        ax.set_title(f"{title} Confusion Matrix")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_roc_curve(targets_list, probs_list, title, num_classes):
        """Plot ROC curves for each class."""
        if not targets_list or not probs_list:
            return None

        # Concatenate all batches and detach gradients
        all_targets = torch.cat(targets_list, dim=0).detach().cpu().numpy()
        all_probs = torch.cat(probs_list, dim=0).detach().cpu().numpy()

        # Convert to binary format for multi-class ROC
        y_bin = label_binarize(all_targets, classes=range(num_classes))

        # Handle case where not all classes are present
        if y_bin.shape[1] < num_classes:
            # Pad with zeros for missing classes
            y_full = np.zeros((len(all_targets), num_classes))
            present_classes = np.unique(all_targets)
            y_full[:, present_classes] = y_bin
            y_bin = y_full

        fig, ax = plt.subplots(figsize=(10, 8))

        # Compute ROC curve and AUC for each class
        colors = plt.cm.Set1(np.linspace(0, 1, num_classes))

        for i in range(num_classes):
            if np.sum(y_bin[:, i]) > 0:  # Only plot if class is present
                fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(
                    fpr,
                    tpr,
                    color=colors[i],
                    lw=2,
                    label=f"Class {i} (AUC = {roc_auc:.3f})",
                )

        # Plot random classifier line
        ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{title} ROC Curves")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class WeightedLossHelper:
    """Helper class for weighted loss computation."""

    @staticmethod
    def compute_class_weights(labels, num_classes, ignore_index):
        """Compute class weights based on inverse frequency in the batch."""
        # Filter out ignore_index
        valid_labels = labels[labels != ignore_index]

        if len(valid_labels) == 0:
            # No valid labels, return uniform weights
            return torch.ones(num_classes, device=labels.device)

        # Count occurrences of each class
        class_counts = torch.zeros(num_classes, device=labels.device)
        for i in range(num_classes):
            class_counts[i] = (valid_labels == i).sum().float()

        # Compute weights as inverse frequency
        # Add small epsilon to avoid division by zero
        total_samples = len(valid_labels)
        weights = torch.zeros(num_classes, device=labels.device)

        for i in range(num_classes):
            if class_counts[i] > 0:
                weights[i] = total_samples / (num_classes * class_counts[i])
            else:
                weights[i] = 1.0  # Default weight for unseen classes

        return weights
