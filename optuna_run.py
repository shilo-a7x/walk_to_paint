import os
import copy
import argparse
import shutil
import glob
from pathlib import Path
from omegaconf import OmegaConf
import torch
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from optuna.integration import PyTorchLightningPruningCallback

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Reuse your project modules
from src.data.prepare_data import prepare_data
from src.model.lit_model import LitEdgeClassifier


def cleanup_old_checkpoints_and_logs(checkpoint_dir, log_dir, keep_top_n=10):
    """
    Keep only the top N checkpoints based on trial number and clean up old logs.
    """
    print(f"🧹 Cleaning up old checkpoints and logs (keeping top {keep_top_n})...")

    # Get all trial checkpoints
    checkpoint_pattern = os.path.join(checkpoint_dir, "trial_*.ckpt")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if len(checkpoint_files) > keep_top_n:
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)

        # Remove old checkpoints
        for old_checkpoint in checkpoint_files[keep_top_n:]:
            try:
                os.remove(old_checkpoint)
                print(f"  Removed old checkpoint: {os.path.basename(old_checkpoint)}")
            except OSError:
                pass

    # Clean up old tensorboard logs (keep only recent trial logs)
    if os.path.exists(log_dir):
        trial_log_dirs = glob.glob(os.path.join(log_dir, "*", "trial_*"))
        if len(trial_log_dirs) > keep_top_n:
            trial_log_dirs.sort(key=os.path.getmtime, reverse=True)
            for old_log_dir in trial_log_dirs[keep_top_n:]:
                try:
                    shutil.rmtree(old_log_dir)
                    print(f"  Removed old log dir: {os.path.basename(old_log_dir)}")
                except OSError:
                    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    p.add_argument("--device", type=int, default=2, help="CUDA device id")
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
        version=f"trial_{trial.number}" if trial else None,
    )

    # Save only the best model according to val_auc (maximize)
    checkpoint = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename=(
            f"trial_{trial.number}-" + "{epoch:02d}-{val_auc_epoch:.4f}"
            if trial
            else f"{cfg.dataset.name}-{cfg.training.exp_name}"
            + "-{epoch:02d}-{val_auc_epoch:.4f}"
        ),
        monitor="val_auc_epoch",
        mode="max",  # Maximize AUC
        save_top_k=1,  # Only save best checkpoint per trial
        save_last=False,
    )

    early_stopping = EarlyStopping(
        monitor="val_auc_epoch",
        patience=cfg.training.early_stopping_patience,
        verbose=True,
        mode="max",  # Maximize AUC
    )

    callbacks = [checkpoint, early_stopping]
    if enable_pruning and trial is not None and val_loader is not None:
        callbacks.append(
            PyTorchLightningPruningCallback(trial, monitor="val_auc_epoch")
        )

    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        logger=logger,
        log_every_n_steps=50,  # Reduce logging frequency
        default_root_dir=cfg.training.checkpoint_dir,
        accelerator=(
            "gpu" if cfg.training.use_cuda and torch.cuda.is_available() else "cpu"
        ),
        callbacks=callbacks,
        gradient_clip_val=cfg.training.gradient_clip_val,
        enable_progress_bar=False,  # Disable for cleaner output during optimization
        enable_model_summary=False,  # Reduce log clutter
    )
    return trainer, checkpoint


def objective_factory(base_cfg, device, enable_pruning=True):
    """
    Returns an Optuna objective that:
      1) clones and tweaks the config per-trial,
      2) trains,
      3) returns the best val_auc from the best checkpoint.
    """

    def objective(trial: optuna.trial.Trial):
        # ---- Clone and set per-trial hyperparams ----
        cfg = copy.deepcopy(base_cfg)

        # Unique experiment name per trial
        cfg.training.exp_name = f"{base_cfg.training.exp_name}-optuna-t{trial.number}"

        # ===== DATA GENERATION HYPERPARAMETERS =====
        # Walk generation parameters (most impactful for graph learning)
        cfg.dataset.max_walk_length = trial.suggest_int(
            "dataset.max_walk_length", 10, 100
        )
        cfg.dataset.num_walks = trial.suggest_int(
            "dataset.num_walks", 100000, 5000000, log=True
        )

        # ===== TRAINING HYPERPARAMETERS =====
        cfg.training.lr = trial.suggest_loguniform("training.lr", 1e-5, 1e-1)
        cfg.training.weight_decay = trial.suggest_loguniform(
            "training.weight_decay", 1e-8, 1e-1
        )
        cfg.training.batch_size = trial.suggest_categorical(
            "training.batch_size", [32, 64, 128, 256, 512]
        )
        cfg.training.gradient_clip_val = trial.suggest_float(
            "training.gradient_clip_val", 0.1, 2.0
        )

        # Early stopping patience (adaptive based on epochs)
        cfg.training.early_stopping_patience = trial.suggest_int(
            "training.early_stopping_patience", 5, 20
        )

        # ===== MODEL ARCHITECTURE HYPERPARAMETERS =====
        # Sample nhead and embedding_dim separately with validation
        # This is the correct approach for categorical parameters

        max_attempts = 10  # Prevent infinite loops
        for attempt in range(max_attempts):
            cfg.model.nhead = trial.suggest_categorical("model.nhead", [1, 2, 4, 8, 16])
            cfg.model.embedding_dim = trial.suggest_categorical(
                "model.embedding_dim", [4, 8, 16, 32, 64, 128]
            )

            # Check if embedding_dim is divisible by nhead
            if cfg.model.embedding_dim % cfg.model.nhead == 0:
                break

            # If not valid, prune this trial and let Optuna try again
            if attempt == max_attempts - 1:
                raise optuna.TrialPruned(
                    f"Could not find valid nhead/embedding_dim combination after {max_attempts} attempts"
                )

        # No need for user attributes - values are directly stored in trial.params

        cfg.model.hidden_dim = trial.suggest_categorical(
            "model.hidden_dim", [4, 8, 16, 32, 64, 128, 256]
        )
        cfg.model.nlayers = trial.suggest_int("model.nlayers", 1, 6)

        # Regularization
        cfg.model.dropout = trial.suggest_float("model.dropout", 0.0, 0.7)

        # ===== EPOCHS ADAPTATION =====
        # Shorter epochs for hyperparameter search but ensure minimum learning time
        base_epochs = base_cfg.training.epochs
        cfg.training.epochs = trial.suggest_int(
            "training.epochs", min(15, base_epochs // 4), min(100, base_epochs)
        )

        # Ensure CUDA device selection
        if cfg.training.use_cuda and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        print(f"\n🔬 Trial {trial.number} hyperparameters:")
        print(
            f"  Walk length: {cfg.dataset.max_walk_length}, Num walks: {cfg.dataset.num_walks:,}"
        )
        print(f"  LR: {cfg.training.lr:.2e}, Batch size: {cfg.training.batch_size}")
        print(
            f"  Model: emb_dim={cfg.model.embedding_dim}, hidden_dim={cfg.model.hidden_dim}"
        )
        print(f"  Transformer: {cfg.model.nlayers} layers, {cfg.model.nhead} heads")
        print(
            f"  Epochs: {cfg.training.epochs}, Weighted loss: {cfg.training.use_weighted_loss}"
        )

        # ---- Data & Model ----
        try:
            data_module = prepare_data(cfg)
            model = LitEdgeClassifier(cfg)

            # ---- Trainer with pruning ----
            val_loader = data_module["val"]
            trainer, checkpoint = build_trainer(cfg, val_loader, trial, enable_pruning)

            # ---- Fit ----
            trainer.fit(model, data_module["train"], val_loader)

            # ---- Evaluate best checkpoint on validation set ----
            best_ckpt_path = checkpoint.best_model_path

            if best_ckpt_path and os.path.exists(best_ckpt_path):
                val_metrics = trainer.validate(
                    model, val_loader, ckpt_path=best_ckpt_path
                )
            else:
                # Fallback to current model if no checkpoint
                val_metrics = trainer.validate(model, val_loader)

            # Extract val_auc_epoch (maximize this!)
            val_auc = float(val_metrics[0]["val_auc_epoch"])

            print(f"✅ Trial {trial.number} completed - Val AUC: {val_auc:.4f}")

            # Return negative AUC because Optuna minimizes by default
            return -val_auc

        except Exception as e:
            print(f"❌ Trial {trial.number} failed: {str(e)}")
            # Return a bad score for failed trials
            return 0.0  # This will be interpreted as -0.0 AUC (very bad)

    return objective


def main():

    args = parse_args()

    # ---- Load config and apply CLI overrides ----
    base_cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_dotlist(args.overrides)
    base_cfg = OmegaConf.merge(base_cfg, cli_cfg)

    print(f"🎯 Starting Optuna hyperparameter optimization with {args.n_trials} trials")
    print(f"📊 Objective: Maximize validation AUC")
    print(f"🖥️  Device: {args.device}")
    print(f"💾 Keeping top 10 checkpoints/logs to save space")

    # ---- Create study with Optuna journal file storage ----
    storage = JournalStorage(JournalFileStorage("optuna_study.log"))
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1,
        ),
        study_name=f"walk_to_paint_study_{base_cfg.training.exp_name}",
        storage=storage,
        load_if_exists=True,
    )

    # ---- Add callback to cleanup after every 5 trials ----
    def cleanup_callback(study, trial):
        if trial.number > 0 and trial.number % 5 == 0:
            cleanup_old_checkpoints_and_logs(
                base_cfg.training.checkpoint_dir,
                base_cfg.training.log_dir,
                keep_top_n=10,
            )

    # ---- Optimize ----
    objective = objective_factory(base_cfg, args.device, enable_pruning=True)
    study.optimize(objective, n_trials=args.n_trials, callbacks=[cleanup_callback])

    # ---- Final cleanup ----
    cleanup_old_checkpoints_and_logs(
        base_cfg.training.checkpoint_dir, base_cfg.training.log_dir, keep_top_n=10
    )

    # ---- Print results ----
    print("\n" + "=" * 60)
    print("🏆 OPTUNA OPTIMIZATION COMPLETED!")
    print("=" * 60)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best validation AUC: {-study.best_value:.6f}")  # Convert back from negative
    print("\n📋 Best hyperparameters:")

    # Group parameters by category for better readability
    params = study.best_trial.params

    print("\n🗃️  Data Generation:")
    for key in sorted(params.keys()):
        if key.startswith("dataset."):
            print(f"  {key}: {params[key]}")

    print("\n🏋️  Training:")
    for key in sorted(params.keys()):
        if key.startswith("training."):
            print(f"  {key}: {params[key]}")

    print("\n🧠 Model Architecture:")
    for key in sorted(params.keys()):
        if key.startswith("model."):
            print(f"  {key}: {params[key]}")

    # ---- Save best config ----
    out_yaml = f"best_params_optuna_{base_cfg.training.exp_name}.yaml"
    best_cfg = {
        "dataset": {
            "max_walk_length": params.get("dataset.max_walk_length"),
            "num_walks": params.get("dataset.num_walks"),
        },
        "training": {
            "lr": params.get("training.lr"),
            "weight_decay": params.get("training.weight_decay"),
            "batch_size": params.get("training.batch_size"),
            "gradient_clip_val": params.get("training.gradient_clip_val"),
            "early_stopping_patience": params.get("training.early_stopping_patience"),
            "epochs": params.get("training.epochs"),
        },
        "model": {
            "embedding_dim": params.get("model.embedding_dim"),
            "hidden_dim": params.get("model.hidden_dim"),
            "nhead": params.get("model.nhead"),
            "nlayers": params.get("model.nlayers"),
            "dropout": params.get("model.dropout"),
        },
    }

    with open(out_yaml, "w") as f:
        import yaml

        yaml.safe_dump(best_cfg, f, default_flow_style=False)

    print(f"\n💾 Saved best configuration to: {out_yaml}")
    print(f"🎯 Best validation AUC achieved: {-study.best_value:.6f}")

    # ---- Save study for future analysis ----
    import joblib

    study_path = f"optuna_study_{base_cfg.training.exp_name}.pkl"
    joblib.dump(study, study_path)
    print(f"📊 Saved complete study to: {study_path}")

    print("\n🚀 Use the best configuration to train your final model!")
    print("=" * 60)


if __name__ == "__main__":
    main()
