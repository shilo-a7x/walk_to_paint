import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from src.model.model import TransformerModel
from src.model.metrics_helper import MetricsManager, PlottingHelper


SPLIT_TRAIN = 0
SPLIT_MASK = 1
SPLIT_VAL = 2
SPLIT_TEST = 3
SPLIT_BAD = -1


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

        self.dynamic_train_masking = bool(
            getattr(self.cfg.model, "dynamic_train_masking", False)
        )
        self.dynamic_train_mask_seed_offset = int(
            getattr(self.cfg.model, "dynamic_train_mask_seed_offset", 0)
        )
        self._epoch_target_edge_ids_cpu = None
        self._epoch_target_edge_ids_by_device = {}
        self._dynamic_pool_ready = False
        self._dynamic_pool_edge_ids_cpu = None
        self._dynamic_pool_edge_classes_cpu = None

        # Hard-node reweighting (E14)
        self.hardness_lambda = float(getattr(self.cfg.model, "hardness_lambda", 0.5))
        hardness_map_path = getattr(self.cfg.model, "hardness_map_path", None)
        if hardness_map_path:
            hmap = torch.load(
                str(hardness_map_path), map_location="cpu", weights_only=True
            )
            self.register_buffer("hardness_map_tensor", hmap.float())
            print(
                f"✓ Loaded hardness map from {hardness_map_path} (vocab_size={hmap.shape[0]})"
            )
        else:
            self.hardness_map_tensor = None

    def forward(self, input_ids):
        return self.model(input_ids)

    def _maybe_apply_node_replacement(self, input_ids, node_mask):
        """Apply node-token replacement regularization in training mode only."""
        mode = str(getattr(self.cfg.model, "node_context_mode", "none"))
        if mode != "replace":
            return input_ids

        replace_prob = float(getattr(self.cfg.model, "node_replace_prob", 0.0))
        if replace_prob <= 0.0:
            return input_ids

        unk_ratio = float(getattr(self.cfg.model, "node_replace_unk_ratio", 0.7))
        x = input_ids.clone()

        candidates = node_mask
        if not candidates.any():
            return x

        replace_mask = (
            torch.rand_like(candidates, dtype=torch.float) < replace_prob
        ) & candidates
        if not replace_mask.any():
            return x

        node_pool = x[candidates]
        if node_pool.numel() == 0:
            return x

        selected = replace_mask.nonzero(as_tuple=False)
        if selected.numel() == 0:
            return x

        use_unk = torch.rand(selected.size(0), device=x.device) < unk_ratio

        unk_id = int(getattr(self.cfg.model, "unk_id", 2))
        if use_unk.any():
            unk_positions = selected[use_unk]
            x[unk_positions[:, 0], unk_positions[:, 1]] = unk_id

        rand_count = int((~use_unk).sum().item())
        if rand_count > 0:
            rand_positions = selected[~use_unk]
            rand_idx = torch.randint(
                0, node_pool.numel(), (rand_count,), device=x.device
            )
            x[rand_positions[:, 0], rand_positions[:, 1]] = node_pool[rand_idx]

        return x

    def _build_dynamic_train_pool(self):
        if not self.dynamic_train_masking:
            return

        train_loader = self.trainer.train_dataloader
        train_dataset = getattr(train_loader, "dataset", None)
        if train_dataset is None:
            raise RuntimeError(
                "dynamic_train_masking requires accessible train dataset"
            )

        edge_ids = getattr(train_dataset, "edge_ids", None)
        edge_split_mask = getattr(train_dataset, "edge_split_mask", None)
        input_ids = getattr(train_dataset, "input_ids", None)
        id2class = getattr(train_dataset, "id2class", None)

        if (
            edge_ids is None
            or edge_split_mask is None
            or input_ids is None
            or id2class is None
        ):
            raise RuntimeError(
                "dynamic_train_masking requires edge_ids, edge_split_mask, input_ids and id2class metadata"
            )

        pool_mask = (
            (edge_split_mask == SPLIT_TRAIN) | (edge_split_mask == SPLIT_MASK)
        ) & (edge_ids >= 0)
        pool_edge_ids = edge_ids[pool_mask].long().cpu()
        if pool_edge_ids.numel() == 0:
            raise RuntimeError("dynamic_train_masking pool is empty")

        pool_edge_classes = id2class[input_ids[pool_mask].long()].long().cpu()
        valid = pool_edge_classes != self.ignore_index
        if not valid.any():
            raise RuntimeError("dynamic_train_masking pool has no valid classes")

        pool_edge_ids = pool_edge_ids[valid]
        pool_edge_classes = pool_edge_classes[valid]

        max_edge_id = int(pool_edge_ids.max().item())
        edge_class_map = torch.full(
            (max_edge_id + 1,), self.ignore_index, dtype=torch.long
        )
        edge_class_map[pool_edge_ids] = pool_edge_classes

        unique_edge_ids = torch.unique(pool_edge_ids)
        unique_edge_classes = edge_class_map[unique_edge_ids]

        self._dynamic_pool_edge_ids_cpu = unique_edge_ids.long()
        self._dynamic_pool_edge_classes_cpu = unique_edge_classes.long()
        self._dynamic_pool_ready = True

    def _sample_epoch_targets(self, epoch: int):
        if not self.dynamic_train_masking:
            return
        if not self._dynamic_pool_ready:
            self._build_dynamic_train_pool()

        target_ratio = float(self.cfg.dataset.mask_ratio) / float(
            self.cfg.dataset.train_ratio + self.cfg.dataset.mask_ratio
        )
        target_ratio = max(0.0, min(1.0, target_ratio))

        gen = torch.Generator(device="cpu")
        base_seed = int(self.cfg.reproducibility.seed)
        gen.manual_seed(base_seed + self.dynamic_train_mask_seed_offset + int(epoch))

        edge_ids = self._dynamic_pool_edge_ids_cpu
        edge_classes = self._dynamic_pool_edge_classes_cpu
        selected_chunks = []

        for class_id in range(self.num_classes):
            class_mask = edge_classes == class_id
            class_ids = edge_ids[class_mask]
            count = int(class_ids.numel())
            if count == 0:
                continue

            k = int(round(count * target_ratio))
            if target_ratio > 0.0 and k == 0:
                k = 1
            k = min(k, count)

            if k == count:
                selected_chunks.append(class_ids)
            elif k > 0:
                perm = torch.randperm(count, generator=gen)
                selected_chunks.append(class_ids[perm[:k]])

        if selected_chunks:
            selected = torch.unique(torch.cat(selected_chunks, dim=0))
        else:
            selected = torch.empty(0, dtype=torch.long)

        self._epoch_target_edge_ids_cpu = selected
        self._epoch_target_edge_ids_by_device = {}

    def _build_dynamic_targets_for_batch(self, input_ids, metadata):
        labels = torch.full_like(input_ids, self.ignore_index)
        edge_ids = metadata["edge_ids"]
        split_mask = metadata["edge_split_mask"]
        edge_classes = metadata["edge_classes"]

        target_ids = self._epoch_target_edge_ids_by_device.get(edge_ids.device)
        if target_ids is None:
            if self._epoch_target_edge_ids_cpu is None:
                target_ids = torch.empty(0, dtype=torch.long, device=edge_ids.device)
            else:
                target_ids = self._epoch_target_edge_ids_cpu.to(edge_ids.device)
            self._epoch_target_edge_ids_by_device[edge_ids.device] = target_ids

        in_pool = (split_mask == SPLIT_TRAIN) | (split_mask == SPLIT_MASK)
        valid_edge = edge_ids >= 0
        target_positions = in_pool & valid_edge & torch.isin(edge_ids, target_ids)

        labels[target_positions] = edge_classes[target_positions]
        dynamic_input_ids = input_ids.clone()
        mask_id = int(getattr(self.cfg.model, "mask_id", 1))
        dynamic_input_ids[target_positions] = mask_id

        return dynamic_input_ids, labels

    def on_train_start(self):
        if self.dynamic_train_masking:
            self._build_dynamic_train_pool()

    def on_train_epoch_start(self):
        if self.dynamic_train_masking:
            self._sample_epoch_targets(self.current_epoch)

    def _step(self, batch, stage: str):
        # Unpack batch (always 4-tuple with metadata)
        input_ids, labels, attention_mask, metadata = batch
        # metadata contains: edge_ids, walk_ids, positions, walk_lengths
        # Store for callback access if needed
        self._last_metadata = metadata

        model_input_ids = input_ids
        if stage == "train" and self.training and self.dynamic_train_masking:
            model_input_ids, labels = self._build_dynamic_targets_for_batch(
                input_ids,
                metadata,
            )

        # Check if all labels are ignore_index
        if torch.all(labels == self.cfg.model.ignore_index):
            # Skip the batch completely if all labels are ignore_index
            return None  # Returning None indicates that no loss/metrics are computed

        positions = metadata.get("positions")
        node_mask = None
        if positions is not None:
            node_mask = (positions >= 0) & ((positions % 2) == 0)

        if stage == "train" and self.training and node_mask is not None:
            model_input_ids = self._maybe_apply_node_replacement(
                model_input_ids, node_mask
            )

        logits = self.model(
            model_input_ids,
            attention_mask=attention_mask,
            node_mask=node_mask if stage == "train" else None,
        )

        # Compute loss with MANDATORY class weighting (global train-only weights)
        # Same weights used for train, val, test - no data leakage
        weights = (
            self.class_weights.to(logits.device)
            if self.class_weights is not None
            else None
        )
        if stage == "train" and self.hardness_map_tensor is not None:
            # Hard-node reweighting: weight each walk by hardness of adjacent nodes
            B, S = labels.shape
            loss_flat = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                weight=weights,
                ignore_index=self.ignore_index,
                reduction="none",
            )
            loss_2d = loss_flat.view(B, S)
            target_mask = labels != self.ignore_index  # [B, S]

            # Per-walk target position (argmax finds first True; OK for 1 mask/walk)
            target_pos = target_mask.long().argmax(dim=1)  # [B]
            seq_idx = torch.arange(B, device=logits.device)
            left_pos = (target_pos - 1).clamp(min=0)
            right_pos = (target_pos + 1).clamp(max=S - 1)

            # Use original input_ids for node lookup (before any replacement/masking)
            left_toks = input_ids[seq_idx, left_pos]
            right_toks = input_ids[seq_idx, right_pos]
            h_left = self.hardness_map_tensor[left_toks]  # [B]
            h_right = self.hardness_map_tensor[right_toks]  # [B]
            walk_weights = 1.0 + self.hardness_lambda * (h_left + h_right) / 2.0  # [B]

            target_float = target_mask.float()
            weighted = (loss_2d * walk_weights.unsqueeze(1) * target_float).sum()
            loss = weighted / target_float.sum().clamp(min=1.0)
        else:
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
