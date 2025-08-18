import os
import copy
import argparse
from omegaconf import OmegaConf
import torch
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Reuse your project modules
from src.data.prepare_data import prepare_data
from src.model.lit_model import LitEdgeClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    p.add_argument("--device", type=int, default=0, help="CUDA device id")
    p.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
    p.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="OmegaConf dotlist overrides, e.g. training.epochs=30 model.num_layers=4",
    )
    return p.parse_args()


def build_trainer(cfg, val_loader=None, trial=None, enable_pruning=True):
    logger = TensorBoardLogger(
        save_dir=cfg.training.log_dir,
        name=f"{cfg.dataset.name}-{cfg.training.exp_name}",
    )

    # Save only the best model according to val_loss
    checkpoint = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename=f"{cfg.dataset.name}-{cfg.training.exp_name}"
        + "-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=cfg.training.early_stopping_patience,
        verbose=True,
        mode="min",
    )

    callbacks = [checkpoint, early_stopping]
    if enable_pruning and trial is not None and val_loader is not None:
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_loss"))

    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        logger=logger,
        log_every_n_steps=1,
        default_root_dir=cfg.training.checkpoint_dir,
        accelerator=(
            "gpu" if cfg.training.use_cuda and torch.cuda.is_available() else "cpu"
        ),
        callbacks=callbacks,
        gradient_clip_val=cfg.training.gradient_clip_val,
        enable_progress_bar=True,
    )
    return trainer, checkpoint


def objective_factory(base_cfg, device, enable_pruning=True):
    """
    Returns an Optuna objective that:
      1) clones and tweaks the config per-trial,
      2) trains,
      3) returns the best val_loss from the best checkpoint.
    """

    def objective(trial: optuna.trial.Trial):
        # ---- Clone and set per-trial hyperparams ----
        cfg = copy.deepcopy(base_cfg)

        # Unique experiment name per trial
        cfg.training.exp_name = f"{base_cfg.training.exp_name}-optuna-t{trial.number}"

        # Sample hyperparameters from ranges
        cfg.training.lr = trial.suggest_loguniform("training.lr", 1e-5, 1e-2)
        cfg.training.weight_decay = trial.suggest_loguniform(
            "training.weight_decay", 1e-8, 1e-2
        )
        cfg.training.batch_size = trial.suggest_int(
            "training.batch_size", 16, 128, log=True
        )
        cfg.training.gradient_clip_val = trial.suggest_float(
            "training.gradient_clip_val", 0.0, 1.0
        )

        # ---- Model Hyperparameters ----
        cfg.model.num_layers = trial.suggest_int("model.num_layers", 1, 4)
        cfg.model.hidden_dim = trial.suggest_int("model.hidden_dim", 4, 128, log=True)
        cfg.model.dropout = trial.suggest_float("model.dropout", 0.0, 0.5)
        cfg.model.emb_dim = trial.suggest_int(
            "model.emb_dim", 4, 128, step=4
        )  # Embedding dimension
        cfg.model.n_heads = trial.suggest_int(
            "model.n_heads", 1, 8
        )  # Number of attention heads

        # Ensure emb_dim is divisible by n_heads
        while cfg.model.emb_dim % cfg.model.n_heads != 0:
            cfg.model.emb_dim += 4  # Increment until divisible by n_heads

        # Optional: shorter epochs for hyperparameter search; keep at least 10
        cfg.training.epochs = max(10, min(base_cfg.training.epochs, 50))

        # Ensure CUDA device selection mirrors your run.py behavior
        if cfg.training.use_cuda and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # ---- Data & Model ----
        data_module = prepare_data(cfg)
        model = LitEdgeClassifier(cfg)

        # ---- Trainer with pruning (optional) ----
        val_loader = data_module["val"]
        trainer, checkpoint = build_trainer(cfg, val_loader, trial, enable_pruning)

        # ---- Fit ----
        trainer.fit(model, data_module["train"], val_loader)

        # ---- Evaluate best checkpoint on the validation set to get the score ----
        best_ckpt_path = checkpoint.best_model_path or None

        # If best_ckpt_path is None (edge case), validate current weights
        val_metrics = trainer.validate(model, val_loader, ckpt_path=best_ckpt_path)
        # `val_metrics` is a list of dicts (one per dataloader); take the first
        score = float(val_metrics[0]["val_loss"])

        # Report back to Optuna
        return score

    return objective


def main():
    args = parse_args()

    # ---- Load config and apply CLI overrides (same pattern as your run.py) ----
    base_cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_dotlist(args.overrides)
    base_cfg = OmegaConf.merge(base_cfg, cli_cfg)

    # ---- Build study ----
    study = optuna.create_study(
        direction="minimize",
    )

    # ---- Optimize ----
    objective = objective_factory(base_cfg, args.device, enable_pruning=True)
    study.optimize(objective, n_trials=args.n_trials)

    # ---- Print / save results ----
    print("Best trial:")
    print(f"  number={study.best_trial.number}")
    print(f"  value (val_loss)={study.best_value:.6f}")
    print("  params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    # Optionally persist best params to a YAML for later reuse
    out_yaml = f"best_params_optuna.yaml"
    best_cfg = {
        "training": {
            "lr": study.best_trial.params.get("training.lr"),
            "weight_decay": study.best_trial.params.get("training.weight_decay"),
            "batch_size": study.best_trial.params.get("training.batch_size"),
            "gradient_clip_val": study.best_trial.params.get(
                "training.gradient_clip_val"
            ),
        },
        "model": {
            "num_layers": study.best_trial.params.get("model.num_layers"),
            "hidden_dim": study.best_trial.params.get("model.hidden_dim"),
            "dropout": study.best_trial.params.get("model.dropout"),
            "emb_dim": study.best_trial.params.get("model.emb_dim"),
            "n_heads": study.best_trial.params.get("model.n_heads"),
        },
    }
    with open(out_yaml, "w") as f:
        import yaml

        yaml.safe_dump(best_cfg, f)
    print(f"Wrote best params to {out_yaml}")


if __name__ == "__main__":
    main()
