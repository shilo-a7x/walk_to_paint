"""Edge-identity-token variant of src/training/train.py.

Copy of the production driver with exactly one substantive change: it builds an
EIDLitEdgeClassifier instead of a LitEdgeClassifier. Everything else -- the
TensorBoardLogger, ModelCheckpoint (monitor val_auc_epoch, save every improving
epoch + last), EarlyStopping (same monitor, same patience/min_delta semantics
read from cfg.training), gradient clipping, the fit -> test flow -- is the real
production Trainer setup, unmodified.

PerEpochPredictionSaver / PerEpochTestRunner (src/training/callbacks.py) are not
wired in here -- this pilot doesn't need per-epoch prediction artifacts, and
callbacks.py calls `pl_module.model(input_ids, attention_mask=...)` directly
without a sign_ids argument, so it would need its own small EID-aware variant
before it could be reused; not needed for this experiment's scope. Both callback
flags default to False in config.yaml already, so this is a no-op unless someone
explicitly turns them on for an EID run -- don't.
"""

import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from experiments.edge_identity_tokens.eid_src.model.lit_model import EIDLitEdgeClassifier


def train_eid_model(cfg, data_module):
    logger = TensorBoardLogger(
        save_dir=cfg.training.log_dir,
        name=f"{cfg.dataset.name}-{cfg.training.exp_name}",
    )

    ckpt_path = cfg.training.resume_from_checkpoint or None
    model = EIDLitEdgeClassifier(cfg)
    if ckpt_path:
        print(f"Loading model from checkpoint: {ckpt_path}")
        model = EIDLitEdgeClassifier.load_from_checkpoint(ckpt_path, cfg=cfg)
        print("Model loaded from checkpoint.")
    else:
        print("No checkpoint path provided, training from scratch.")

    checkpoint = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename=f"{cfg.dataset.name}-{cfg.training.exp_name}"
        + "-{epoch:02d}-{val_auc_epoch:.4f}",
        monitor="val_auc_epoch",
        mode="max",
        save_top_k=-1,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_auc_epoch",
        patience=cfg.training.early_stopping_patience,
        min_delta=float(getattr(cfg.training, "early_stopping_min_delta", 0.0)),
        verbose=True,
        mode="max",
    )

    callbacks = [checkpoint, early_stopping]

    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        logger=logger,
        log_every_n_steps=max(1, len(data_module["train"]) // 10),
        default_root_dir=cfg.training.checkpoint_dir,
        accelerator=(
            "gpu" if cfg.training.use_cuda and torch.cuda.is_available() else "cpu"
        ),
        callbacks=callbacks,
        gradient_clip_val=cfg.training.gradient_clip_val,
    )

    if cfg.training.eval_only:
        trainer.validate(model, data_module["val"], ckpt_path=ckpt_path)
        trainer.test(model, data_module["test"], ckpt_path=ckpt_path)
    else:
        trainer.fit(model, data_module["train"], data_module["val"], ckpt_path=ckpt_path)
        trainer.test(model, data_module["test"])

    return model
