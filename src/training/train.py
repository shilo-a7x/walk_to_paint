import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from src.model.lit_model import LitEdgeClassifier
from src.training.callbacks import PerEpochPredictionSaver, PerEpochTestRunner


def train_model(cfg, data_module):
    logger = TensorBoardLogger(
        save_dir=cfg.training.log_dir,
        name=f"{cfg.dataset.name}-{cfg.training.exp_name}",
    )

    ckpt_path = cfg.training.resume_from_checkpoint or None
    model = LitEdgeClassifier(cfg)
    if ckpt_path:
        print(f"Loading model from checkpoint: {ckpt_path}")
        print("Checkpoint exists:", os.path.exists(ckpt_path))
        print("Checkpoint size (bytes):", os.path.getsize(ckpt_path))
        model = LitEdgeClassifier.load_from_checkpoint(ckpt_path, cfg=cfg)
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

    # TODO: add non-zero min_delta
    early_stopping = EarlyStopping(
        monitor="val_auc_epoch",
        patience=cfg.training.early_stopping_patience,
        verbose=True,
        mode="max",
    )

    callback_cfg = getattr(cfg.training, "callbacks", None)
    enable_prediction_saver = bool(
        getattr(callback_cfg, "enable_prediction_saver", False)
    )
    enable_per_epoch_test_runner = bool(
        getattr(callback_cfg, "enable_per_epoch_test_runner", False)
    )

    callbacks = [checkpoint, early_stopping]

    prediction_saver = None
    if enable_prediction_saver:
        prediction_saver = PerEpochPredictionSaver(cfg, data_module)
        callbacks.append(prediction_saver)

    if enable_per_epoch_test_runner:
        test_runner = PerEpochTestRunner(data_module, prediction_saver)
        callbacks.append(test_runner)

    print(
        "Custom callbacks: "
        f"prediction_saver={enable_prediction_saver}, "
        f"per_epoch_test_runner={enable_per_epoch_test_runner}"
    )

    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        logger=logger,
        log_every_n_steps=len(data_module["train"]) // 10,
        # log_every_n_steps=None,
        default_root_dir=cfg.training.checkpoint_dir,
        accelerator=(
            "gpu" if cfg.training.use_cuda and torch.cuda.is_available() else "cpu"
        ),
        callbacks=callbacks,
        gradient_clip_val=cfg.training.gradient_clip_val,
    )

    if cfg.training.eval_only:
        # 🔧 Ensure a concrete model is provided for validate/test
        trainer.validate(model, data_module["val"], ckpt_path=ckpt_path)
        trainer.test(model, data_module["test"], ckpt_path=ckpt_path)
    else:
        trainer.fit(
            model, data_module["train"], data_module["val"], ckpt_path=ckpt_path
        )
        # Final test run after training completes
        trainer.test(model, data_module["test"])

    return model
