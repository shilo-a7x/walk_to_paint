import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.model.model import TransformerModel
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy


class LitEdgeClassifier(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.model = TransformerModel(cfg)
        self.cfg = cfg
        ignore_index = cfg.model.ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.num_classes = cfg.model.num_classes
        self.train_acc = MulticlassAccuracy(
            num_classes=self.num_classes,
            average="macro",
            ignore_index=ignore_index,
        )
        self.val_acc = MulticlassAccuracy(
            num_classes=self.num_classes,
            average="macro",
            ignore_index=ignore_index,
        )
        self.test_acc = MulticlassAccuracy(
            num_classes=self.num_classes,
            average="macro",
            ignore_index=ignore_index,
        )

        self.train_f1 = MulticlassF1Score(
            num_classes=self.num_classes,
            average="macro",
            ignore_index=ignore_index,
        )
        self.val_f1 = MulticlassF1Score(
            num_classes=self.num_classes,
            average="macro",
            ignore_index=ignore_index,
        )
        self.test_f1 = MulticlassF1Score(
            num_classes=self.num_classes,
            average="macro",
            ignore_index=ignore_index,
        )

    def forward(self, input_ids):
        return self.model(input_ids)

    def _step(self, batch, stage: str):
        input_ids, labels, attention_mask = batch

        logits = self.model(input_ids, attention_mask=attention_mask)

        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        preds = logits.argmax(dim=-1).view(-1)
        targets = labels.view(-1)

        if stage == "train":
            self.train_acc.update(preds, targets)
            self.train_f1.update(preds, targets)
        elif stage == "val":
            self.val_acc.update(preds, targets)
            self.val_f1.update(preds, targets)
        elif stage == "test":
            self.test_acc.update(preds, targets)
            self.test_f1.update(preds, targets)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )

    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", self.val_acc.compute(), prog_bar=True)
        self.log("val_f1_epoch", self.val_f1.compute(), prog_bar=True)
        self.val_acc.reset()
        self.val_f1.reset()

    def on_test_epoch_end(self):
        self.log("test_acc_epoch", self.test_acc.compute(), prog_bar=True)
        self.log("test_f1_epoch", self.test_f1.compute(), prog_bar=True)
        self.test_acc.reset()
        self.test_f1.reset()
