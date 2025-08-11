import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from src.model.lit_model import LitEdgeClassifier


def train_model(cfg, data_module):
    logger = TensorBoardLogger(
        save_dir=cfg.training.log_dir, name=cfg.training.exp_name
    )
    model = LitEdgeClassifier(cfg)

    checkpoint = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename=f"{cfg.dataset.name}-{cfg.training.exp_name}"
        + "-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=-1,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=cfg.training.early_stopping_patience,
        verbose=True,
        mode="min",
    )

    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        logger=logger,
        log_every_n_steps=len(data_module["train"]) / 10,
        default_root_dir=cfg.training.checkpoint_dir,
        accelerator=(
            "gpu" if cfg.training.use_cuda and torch.cuda.is_available() else "cpu"
        ),
        callbacks=[checkpoint, early_stopping],
    )

    ckpt_path = cfg.training.resume_from_checkpoint or None
    if cfg.training.eval_only:
        # trainer.validate(model, data_module["val"], ckpt_path=ckpt_path)
        trainer.test(model, data_module["test"], ckpt_path=ckpt_path)
    else:
        trainer.fit(
            model, data_module["train"], data_module["val"], ckpt_path=ckpt_path
        )
        trainer.test(model, data_module["test"], ckpt_path=ckpt_path)
    return model
