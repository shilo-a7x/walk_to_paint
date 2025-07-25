import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from src.model.lit_model import LitEdgeCompletion
import os


def train_model(cfg, data_module):
    vocab_size = cfg.get("vocab_size", 1000)  # overwritten later
    logger = TensorBoardLogger(
        save_dir=cfg.training.log_dir, name=cfg.training.exp_name
    )
    model = LitEdgeCompletion(cfg, vocab_size=vocab_size)

    checkpoint = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename=f"{cfg.dataset.name}-{cfg.training.exp_name}"
        + "-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        logger=logger,
        default_root_dir=cfg.training.checkpoint_dir,
        log_every_n_steps=5,
        accelerator=(
            "gpu" if cfg.training.use_cuda and torch.cuda.is_available() else "cpu"
        ),
        callbacks=[checkpoint],
    )

    trainer.fit(model, data_module["train"], data_module["val"])
    trainer.test(model, data_module["test"])
    return model
