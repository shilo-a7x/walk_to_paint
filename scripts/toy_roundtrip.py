import os
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC


class ToyDataModule(pl.LightningDataModule):
    """Tiny separable dataset to make overfitting easy."""

    def __init__(self, seed: int = 123, n_train: int = 800, n_val: int = 200, n_test: int = 200, batch_size: int = 64):
        super().__init__()
        self.seed = seed
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.batch_size = batch_size
        self.train_set: Optional[TensorDataset] = None
        self.val_set: Optional[TensorDataset] = None
        self.test_set: Optional[TensorDataset] = None

    def prepare_data(self) -> None:
        # Nothing to download
        return

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_set is not None:
            return

        g = torch.Generator().manual_seed(self.seed)
        total = self.n_train + self.n_val + self.n_test
        x0 = torch.randn(total, 2, generator=g) - 2.0
        x1 = torch.randn(total, 2, generator=g) + 2.0
        x = torch.cat([x0, x1], dim=0)
        y = torch.cat([torch.zeros(total), torch.ones(total)])

        dataset = TensorDataset(x, y)
        lengths = [self.n_train, self.n_val, self.n_test, len(dataset) - (self.n_train + self.n_val + self.n_test)]
        train_set, val_set, test_set, _ = random_split(dataset, lengths, generator=g)
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)


class ToyClassifier(pl.LightningModule):
    def __init__(self, lr: float = 1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.acc = BinaryAccuracy()
        self.auroc = BinaryAUROC()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", self.acc(probs, y.int()), prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_auroc", self.auroc(probs, y.int()), prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def train_then_eval_roundtrip(output_dir: str = "checkpoints/toy_roundtrip") -> None:
    pl.seed_everything(42, workers=True)
    data = ToyDataModule()

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="toy-epoch={epoch:02d}-val_acc={val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=30,
        deterministic=True,
        logger=False,
        enable_model_summary=False,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
    )

    model = ToyClassifier()
    trainer.fit(model, datamodule=data)

    best_ckpt = checkpoint_callback.best_model_path
    if not best_ckpt:
        raise RuntimeError("No checkpoint saved during training.")

    test_metrics = trainer.test(ckpt_path="best", datamodule=data)

    # Fresh datamodule to mimic eval-only run
    data_eval = ToyDataModule()
    loaded = ToyClassifier.load_from_checkpoint(best_ckpt)
    eval_trainer = pl.Trainer(
        deterministic=True,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
    )
    eval_metrics = eval_trainer.test(loaded, datamodule=data_eval)

    print("\n=== Toy roundtrip results ===")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Train-run test metrics: {test_metrics}")
    print(f"Eval-only test metrics: {eval_metrics}")
    # Quick assertion to ensure parity
    train_acc = test_metrics[0].get("test_acc")
    eval_acc = eval_metrics[0].get("test_acc")
    if train_acc is not None and eval_acc is not None and abs(train_acc - eval_acc) > 1e-4:
        raise RuntimeError("Eval-only metrics diverged from train-run metrics.")


if __name__ == "__main__":
    os.makedirs("checkpoints/toy_roundtrip", exist_ok=True)
    train_then_eval_roundtrip()
