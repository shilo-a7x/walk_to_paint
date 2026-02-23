import os
import copy
import argparse
import shutil
import glob
import time
from pathlib import Path
from omegaconf import OmegaConf
import torch
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from optuna.integration import PyTorchLightningPruningCallback

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

# Reuse your project modules
from src.data.prepare_data import prepare_data
from src.model.lit_model import LitEdgeClassifier
from src.utils.paths import resolve_outputs_dirs
from src.utils.config import load_config, get_seed, validate_config
import random
import numpy as np
import shutil
import glob
import importlib

# Centralized Optuna suggestion ranges (easy to tweak for experiments)
# =============================================================================
# ANTI-OVERFITTING FOCUSED HYPERPARAMETER RANGES
# =============================================================================
# Walk parameters are now FIXED (not tuned) to leverage dataset caching:
#   - max_walk_length = 80 (FIXED)
#   - num_walks = 5,000,000 (FIXED)
#   - batch_size = 1024 (FIXED)
# All trials share the same cached dataset for 10-20x speedup!
#
# Hyperparameter ranges focus on REGULARIZATION to combat overfitting:
#   - Slower learning rates (max 1e-3 instead of 1e-1)
#   - Stronger weight decay (min 1e-5 instead of 1e-8)
#   - Mandatory dropout (min 20% instead of 0%)
#   - Smaller model capacity (fewer dims, fewer layers)
#   - More aggressive early stopping (patience 3-10 vs 5-20)
# =============================================================================
OPTUNA_RANGES = {
    # REMOVED: Walk parameters are now FIXED (see objective function)
    # REMOVED: batch_size is now FIXED at 1024
    # Training - Anti-overfitting focus
    "training.lr": (1e-5, 1e-3),  # ← Slower learning (was 1e-1)
    "training.weight_decay": (1e-5, 1e-1),  # ← Stronger L2 regularization (was 1e-8)
    "training.gradient_clip_val": (
        0.5,
        2.0,
    ),  # ← Higher min for more clipping (was 0.1)
    "training.early_stopping_patience": (3, 10),  # ← Stop faster (was 5-20)
    "training.epochs": (15, 50),  # ← Simple fixed range for hyperparameter search
    # Model - Smaller capacity to reduce overfitting
    "model.nhead": [
        2,
        4,
        8,
    ],  # even heads only (avoids nested-tensor warning for odd heads)
    "model.embedding_dim": [4, 8, 16, 32, 64],  # ← Removed large (128)
    "model.hidden_dim": [4, 8, 16, 32, 64, 128],  # ← Removed tiny (4,8) and large (256)
    "model.nlayers": (1, 4),  # ← Max 4 layers (was 6)
    "model.dropout": (0.2, 0.7),  # ← Min 20% dropout (was 0%)
}


def cleanup_old_checkpoints_and_logs(
    checkpoint_dir, log_dir, study=None, keep_top_n=10
):
    """
    Keep only the top N checkpoints based on trial value (AUC), not date.
    """
    print(f"🧹 Cleaning up old checkpoints and logs (keeping top {keep_top_n})...")

    # Get all trial checkpoints
    checkpoint_pattern = os.path.join(checkpoint_dir, "trial_*.ckpt")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if len(checkpoint_files) > keep_top_n:
        # If study provided, sort by trial value; otherwise fall back to modification time
        if study is not None:
            # Extract trial numbers from checkpoint filenames
            trial_to_checkpoint = {}
            for ckpt_file in checkpoint_files:
                # Extract trial number from filename like "trial_123-*.ckpt"
                basename = os.path.basename(ckpt_file)
                try:
                    trial_num = int(basename.split("_")[1].split("-")[0])
                    trial_to_checkpoint[trial_num] = ckpt_file
                except (IndexError, ValueError):
                    continue

            # Get completed trials (with values) and sort by value descending
            completed_trials = [t for t in study.trials if t.value is not None]
            sorted_trials = sorted(
                completed_trials, key=lambda t: t.value, reverse=True
            )

            # Keep checkpoints for top trials
            top_trial_numbers = {t.number for t in sorted_trials[:keep_top_n]}
            checkpoints_to_remove = [
                ckpt
                for trial_num, ckpt in trial_to_checkpoint.items()
                if trial_num not in top_trial_numbers
            ]
        else:
            # Fallback: sort by modification time if no study provided
            checkpoint_files.sort(key=os.path.getmtime, reverse=True)
            checkpoints_to_remove = checkpoint_files[keep_top_n:]

        # Remove old checkpoints
        for old_checkpoint in checkpoints_to_remove:
            try:
                os.remove(old_checkpoint)
                print(f"  Removed old checkpoint: {os.path.basename(old_checkpoint)}")
            except OSError:
                pass

    # Clean up old tensorboard logs (keep only top-value trial logs)
    if os.path.exists(log_dir):
        trial_log_dirs = glob.glob(os.path.join(log_dir, "*", "trial_*"))
        if len(trial_log_dirs) > keep_top_n:
            if study is not None:
                # Extract trial numbers from log dir names
                log_dir_to_trial = {}
                for log_dir_path in trial_log_dirs:
                    dirname = os.path.basename(log_dir_path)
                    try:
                        trial_num = int(dirname.split("_")[1])
                        log_dir_to_trial[trial_num] = log_dir_path
                    except (IndexError, ValueError):
                        continue

                # Keep logs for top trials
                completed_trials = [t for t in study.trials if t.value is not None]
                sorted_trials = sorted(
                    completed_trials, key=lambda t: t.value, reverse=True
                )
                top_trial_numbers = {t.number for t in sorted_trials[:keep_top_n]}

                log_dirs_to_remove = [
                    log_dir_path
                    for trial_num, log_dir_path in log_dir_to_trial.items()
                    if trial_num not in top_trial_numbers
                ]
            else:
                # Fallback: sort by modification time
                trial_log_dirs.sort(key=os.path.getmtime, reverse=True)
                log_dirs_to_remove = trial_log_dirs[keep_top_n:]

            for old_log_dir in log_dirs_to_remove:
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

    early_stopping_min_delta = float(
        getattr(cfg.training, "early_stopping_min_delta", 0.005)
    )
    early_stopping = EarlyStopping(
        monitor="val_auc_epoch",
        patience=cfg.training.early_stopping_patience,
        verbose=True,
        mode="max",  # Maximize AUC
        min_delta=early_stopping_min_delta,
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


def objective_factory(
    base_cfg,
    device,
    enable_pruning=True,
    shared_data_module=None,
    shared_post_prepare_cfg=None,
):
    """
    Returns an Optuna objective that:
      1) clones and tweaks the config per-trial,
      2) trains,
      3) returns the best val_auc from the best checkpoint.

    Args:
        shared_data_module: Pre-loaded data module shared across all trials (loaded once before study)
                           Avoids reloading the .pt cache file for every trial (~242s saved per trial).
        shared_post_prepare_cfg: Config after prepare_data() was called (includes pad_id and other modifications)
                                Each trial deep-copies this and applies hyperparameters on top.
    """

    def objective(trial: optuna.trial.Trial):
        # ---- Clone and set per-trial hyperparams ----
        # Use the post-prepare config if available (has pad_id and other prepare_data modifications)
        # Otherwise fall back to base_cfg
        base_for_trial = (
            shared_post_prepare_cfg if shared_post_prepare_cfg is not None else base_cfg
        )
        cfg = copy.deepcopy(base_for_trial)

        # Unique experiment name per trial
        cfg.training.exp_name = f"{base_cfg.training.exp_name}-optuna-t{trial.number}"

        # ===== FIXED PARAMETERS (Not Hyperparameters) =====
        # Walk parameters are FIXED to enable dataset caching across all trials
        # This provides 10-20x speedup: first trial builds cache (~40s), subsequent trials load cache (~3-5s)
        cfg.dataset.max_walk_length = 80  # FIXED
        cfg.dataset.num_walks = 5000000  # FIXED (5M walks)
        cfg.training.batch_size = 1024  # FIXED
        cfg.training.num_workers = (
            0  # Optuna stability: avoid DataLoader worker teardown noise
        )
        cfg.training.persistent_workers = False

        # Enable dataset caching (critical for performance with fixed walks)
        cfg.preprocess.use_cache = True
        cfg.preprocess.save = True

        # ===== TRAINING HYPERPARAMETERS =====
        lr_lo, lr_hi = OPTUNA_RANGES.get("training.lr", (1e-5, 1e-3))
        cfg.training.lr = trial.suggest_float("training.lr", lr_lo, lr_hi, log=True)
        wd_lo, wd_hi = OPTUNA_RANGES.get("training.weight_decay", (1e-5, 1e-1))
        cfg.training.weight_decay = trial.suggest_float(
            "training.weight_decay", wd_lo, wd_hi, log=True
        )
        gc_lo, gc_hi = OPTUNA_RANGES.get("training.gradient_clip_val", (0.5, 2.0))
        cfg.training.gradient_clip_val = trial.suggest_float(
            "training.gradient_clip_val", gc_lo, gc_hi
        )

        # Early stopping patience (adaptive based on epochs)
        ep_lo, ep_hi = OPTUNA_RANGES.get("training.early_stopping_patience", (3, 10))
        cfg.training.early_stopping_patience = trial.suggest_int(
            "training.early_stopping_patience", ep_lo, ep_hi
        )

        # ===== MODEL ARCHITECTURE HYPERPARAMETERS =====
        # Sample nhead and embedding_dim separately with validation
        # This is the correct approach for categorical parameters

        max_attempts = 10  # Prevent infinite loops
        for attempt in range(max_attempts):
            cfg.model.nhead = trial.suggest_categorical(
                "model.nhead", OPTUNA_RANGES.get("model.nhead", [2, 4, 8])
            )
            cfg.model.embedding_dim = trial.suggest_categorical(
                "model.embedding_dim",
                OPTUNA_RANGES.get("model.embedding_dim", [16, 32, 64]),
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
            "model.hidden_dim",
            OPTUNA_RANGES.get("model.hidden_dim", [16, 32, 64, 128]),
        )
        nl_lo, nl_hi = OPTUNA_RANGES.get("model.nlayers", (1, 4))
        cfg.model.nlayers = trial.suggest_int("model.nlayers", nl_lo, nl_hi)

        # Regularization
        dr_lo, dr_hi = OPTUNA_RANGES.get("model.dropout", (0.2, 0.7))
        cfg.model.dropout = trial.suggest_float("model.dropout", dr_lo, dr_hi)

        # ===== EPOCHS (from OPTUNA_RANGES) =====
        e_lo, e_hi = OPTUNA_RANGES.get("training.epochs", (15, 50))
        cfg.training.epochs = trial.suggest_int("training.epochs", int(e_lo), int(e_hi))

        # ===== UNIFIED SEEDING STRATEGY =====
        # Use get_seed(cfg) to get canonical seed from config.reproducibility.seed
        # This ensures all trials use the same seed from config (not hardcoded values)
        seed = get_seed(cfg)
        seed_everything(seed, workers=True)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Record the seed used for reproducibility tracking
        try:
            trial.set_user_attr("seed", seed)
        except Exception:
            pass

        # Ensure CUDA device selection
        if cfg.training.use_cuda and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        print(f"\n🔬 Trial {trial.number} hyperparameters:")
        print(f"  Seed (from config): {seed}")
        print(
            f"  Walk length: {cfg.dataset.max_walk_length} (FIXED), Num walks: {cfg.dataset.num_walks:,} (FIXED)"
        )
        print(
            f"  LR: {cfg.training.lr:.2e}, Weight decay: {cfg.training.weight_decay:.2e}"
        )
        print(
            f"  Batch size: {cfg.training.batch_size} (FIXED), Gradient clip: {cfg.training.gradient_clip_val:.2f}"
        )
        print(
            f"  Model: emb_dim={cfg.model.embedding_dim}, hidden_dim={cfg.model.hidden_dim}"
        )
        print(f"  Transformer: {cfg.model.nlayers} layers, {cfg.model.nhead} heads")
        print(f"  Dropout: {cfg.model.dropout:.2f}, Epochs: {cfg.training.epochs}")
        print(f"  Early stopping patience: {cfg.training.early_stopping_patience}")
        use_weighted = getattr(cfg.training, "use_weighted_loss", True)  # Default True
        print(
            f"  Weighted loss: {use_weighted}, Cache enabled: {cfg.preprocess.use_cache}"
        )

        # ---- Data & Model ----
        try:
            # Use shared pre-loaded data module (loaded once before study started)
            # This avoids reloading the .pt cache file for every trial (~242s saved per trial)
            if shared_data_module is not None:
                data_module = shared_data_module
                print(f"  📦 Dataset cache: REUSED (pre-loaded, 0.0s)")
            else:
                # Fallback: load data if no shared module provided (first trial or single trial mode)
                cache_start_time = time.time()
                data_module = prepare_data(cfg)
                cache_load_time = time.time() - cache_start_time
                cache_status = (
                    "HIT (loaded from cache)"
                    if cache_load_time < 10
                    else "MISS (built new cache)"
                )
                print(f"  📦 Dataset cache: {cache_status} ({cache_load_time:.1f}s)")

            model = LitEdgeClassifier(cfg)

            # ---- Trainer with pruning ----
            val_loader = data_module["val"]
            trainer, checkpoint = build_trainer(cfg, val_loader, trial, enable_pruning)

            # ---- Fit ----
            trainer.fit(model, data_module["train"], val_loader)

            # ---- Get best validation metrics from checkpoint ----
            best_ckpt_path = checkpoint.best_model_path

            # IMPORTANT: evaluate with a fresh trainer WITHOUT pruning callbacks
            # to avoid duplicate Optuna step reports on post-fit validation.
            eval_trainer = Trainer(
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                accelerator=(
                    "gpu"
                    if cfg.training.use_cuda and torch.cuda.is_available()
                    else "cpu"
                ),
                devices=(
                    1 if (cfg.training.use_cuda and torch.cuda.is_available()) else None
                ),
            )

            if best_ckpt_path and os.path.exists(best_ckpt_path):
                val_metrics = eval_trainer.validate(
                    model, val_loader, ckpt_path=best_ckpt_path
                )
            else:
                # Fallback to current model if no checkpoint
                val_metrics = eval_trainer.validate(model, val_loader)

            # Extract metrics
            val_auc = float(val_metrics[0]["val_auc_epoch"])
            val_loss = float(val_metrics[0].get("val_loss", 0.0))

            # Custom objective: alpha*val_auc - beta*val_loss
            # This balances AUC improvement with loss degradation
            alpha = 1.0  # Weight for AUC (maximize)
            beta = 0.1  # Weight for loss (minimize)
            score = alpha * val_auc - beta * val_loss

            print(f"✅ Trial {trial.number} completed:")
            print(f"   Val AUC: {val_auc:.4f}, Val Loss: {val_loss:.4f}")
            print(f"   Score (α*AUC - β*Loss): {score:.4f}")

            return score

        except optuna.TrialPruned as e:
            print(f"⚠️ Trial {trial.number} pruned: {str(e)}")
            raise
        except Exception as e:
            print(f"❌ Trial {trial.number} failed: {str(e)}")
            # Return a bad score for failed trials
            return 0.0  # This will be interpreted as 0.0 AUC (very bad)

    return objective


def main():

    args = parse_args()

    # ---- Load config and apply CLI overrides ----
    # Load and merge config (supports `configs/<dataset>.yaml` and CLI dotlist overrides)
    base_cfg = load_config(args.config, overrides=args.overrides)
    validate_config(base_cfg, context="optuna")

    # ---- Auto-generate an informative exp_name when not explicitly set ----
    try:
        current_exp = getattr(base_cfg.training, "exp_name", None)
    except Exception:
        current_exp = None

    # If exp_name is unset or left as a placeholder, generate a more informative one
    if not current_exp or current_exp in ("walk_to_paint_experiment", "experiment"):
        # Allow an optional short note in config: training.exp_note
        try:
            note = getattr(base_cfg.training, "exp_note", None)
        except Exception:
            note = None
        note_part = f"-{note}" if note else ""
        auto_name = f"{getattr(base_cfg.dataset, 'name', 'dataset')}-optuna{note_part}"
        base_cfg.training.exp_name = auto_name
        print(f"Auto-generated exp_name: {base_cfg.training.exp_name}")

    # Global seed (single seed for all components)
    seed = get_seed(base_cfg)
    seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"✅ Using global seed from config.reproducibility.seed: {seed}")

    # Resolve outputs dirs early so Optuna and Trainer write into namespaced locations
    resolved = resolve_outputs_dirs(base_cfg)
    print(f"Outputs -> exp_dir: {resolved['exp_dir']}")

    # Prefer medium float32 matmul precision on capable GPUs to speed matmuls
    try:
        if base_cfg.training.use_cuda and torch.cuda.is_available():
            torch.set_float32_matmul_precision("medium")
            print(
                "Set torch.float32 matmul precision to 'medium' (Tensor Cores enabled)"
            )
    except Exception:
        pass

    print(f"🎯 Starting Optuna hyperparameter optimization with {args.n_trials} trials")
    print(f"📊 Objective: Maximize validation AUC")
    print(f"🎯 Focus: Anti-overfitting regularization (fixed walks, caching enabled)")
    print(f"🖥️  Device: {args.device}")
    print(f"💾 Keeping top 10 checkpoints/logs to save space")

    # ---- Create study with Optuna journal file storage (per-experiment) ----
    # Ensure optuna directory exists first
    optuna_dir = resolved.get("optuna_dir", ".")
    os.makedirs(optuna_dir, exist_ok=True)

    optuna_log = os.path.join(optuna_dir, "optuna_study.log")
    storage = JournalStorage(JournalFileStorage(optuna_log))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1,
        ),
        study_name=f"walk_to_paint_study_{base_cfg.training.exp_name}",
        storage=storage,
        load_if_exists=True,
    )

    # ---- Optionally enqueue an initial trial using config-provided params ----
    # If `optuna.initial_params` is present in the base config, enqueue it
    # so the study will run that parameter set as a trial (useful as a seed).
    try:
        ip = OmegaConf.to_container(base_cfg).get("optuna", {}).get("initial_params")
    except Exception:
        ip = None

    if ip:
        # Build initial trial parameter mapping for suggestion keys used in objective()
        def _pick(path, fallback):
            # Prefer CLI/loaded base_cfg value if present, otherwise fallback to initial_params
            try:
                val = OmegaConf.select(base_cfg, path)
                if val is not None:
                    return val
            except Exception:
                pass
            return fallback

        initial_trial_params = {}
        # REMOVED: Walk parameters are now FIXED (not in OPTUNA_RANGES)
        # REMOVED: batch_size is now FIXED at 1024

        # Training hyperparameters only
        t_ip = ip.get("training", {})
        initial_trial_params["training.lr"] = float(
            _pick("training.lr", t_ip.get("lr"))
        )
        initial_trial_params["training.weight_decay"] = float(
            _pick("training.weight_decay", t_ip.get("weight_decay"))
        )
        initial_trial_params["training.gradient_clip_val"] = float(
            _pick("training.gradient_clip_val", t_ip.get("gradient_clip_val"))
        )
        initial_trial_params["training.early_stopping_patience"] = int(
            _pick(
                "training.early_stopping_patience", t_ip.get("early_stopping_patience")
            )
        )
        initial_trial_params["training.epochs"] = int(
            _pick("training.epochs", t_ip.get("epochs"))
        )

        # model
        m_ip = ip.get("model", {})
        initial_trial_params["model.embedding_dim"] = int(
            _pick("model.embedding_dim", m_ip.get("embedding_dim"))
        )
        initial_trial_params["model.hidden_dim"] = int(
            _pick("model.hidden_dim", m_ip.get("hidden_dim"))
        )
        initial_trial_params["model.nhead"] = int(
            _pick("model.nhead", m_ip.get("nhead"))
        )
        initial_trial_params["model.nlayers"] = int(
            _pick("model.nlayers", m_ip.get("nlayers"))
        )
        initial_trial_params["model.dropout"] = float(
            _pick("model.dropout", m_ip.get("dropout"))
        )

        # Clamp all initial params to OPTUNA_RANGES to avoid out-of-range errors
        lr_lo, lr_hi = OPTUNA_RANGES.get("training.lr", (1e-5, 1e-3))
        initial_trial_params["training.lr"] = max(
            lr_lo, min(lr_hi, initial_trial_params["training.lr"])
        )

        wd_lo, wd_hi = OPTUNA_RANGES.get("training.weight_decay", (1e-5, 1e-1))
        initial_trial_params["training.weight_decay"] = max(
            wd_lo, min(wd_hi, initial_trial_params["training.weight_decay"])
        )

        gc_lo, gc_hi = OPTUNA_RANGES.get("training.gradient_clip_val", (0.5, 2.0))
        initial_trial_params["training.gradient_clip_val"] = max(
            gc_lo, min(gc_hi, initial_trial_params["training.gradient_clip_val"])
        )

        ep_lo, ep_hi = OPTUNA_RANGES.get("training.early_stopping_patience", (3, 10))
        initial_trial_params["training.early_stopping_patience"] = max(
            ep_lo, min(ep_hi, initial_trial_params["training.early_stopping_patience"])
        )

        nl_lo, nl_hi = OPTUNA_RANGES.get("model.nlayers", (1, 4))
        initial_trial_params["model.nlayers"] = max(
            nl_lo, min(nl_hi, initial_trial_params["model.nlayers"])
        )

        dr_lo, dr_hi = OPTUNA_RANGES.get("model.dropout", (0.2, 0.7))
        initial_trial_params["model.dropout"] = max(
            dr_lo, min(dr_hi, initial_trial_params["model.dropout"])
        )

        # Clamp categorical parameters to valid choices
        valid_nhead = OPTUNA_RANGES.get("model.nhead", [2, 4, 8])
        if initial_trial_params["model.nhead"] not in valid_nhead:
            initial_trial_params["model.nhead"] = valid_nhead[0]

        valid_emb_dim = OPTUNA_RANGES.get("model.embedding_dim", [16, 32, 64])
        if initial_trial_params["model.embedding_dim"] not in valid_emb_dim:
            initial_trial_params["model.embedding_dim"] = valid_emb_dim[0]

        valid_hid_dim = OPTUNA_RANGES.get("model.hidden_dim", [16, 32, 64, 128])
        if initial_trial_params["model.hidden_dim"] not in valid_hid_dim:
            initial_trial_params["model.hidden_dim"] = valid_hid_dim[0]

        try:
            # Validate that initial params are within OPTUNA_RANGES before enqueuing
            lr_lo, lr_hi = OPTUNA_RANGES.get("training.lr", (1e-5, 1e-3))
            if not (lr_lo <= initial_trial_params["training.lr"] <= lr_hi):
                print(
                    f"  ⚠️  Initial LR {initial_trial_params['training.lr']:.2e} out of range [{lr_lo:.2e}, {lr_hi:.2e}] - skipping enqueue"
                )
                raise ValueError("Initial LR out of range")

            ep_lo, ep_hi = OPTUNA_RANGES.get(
                "training.early_stopping_patience", (3, 10)
            )
            if not (
                ep_lo
                <= initial_trial_params["training.early_stopping_patience"]
                <= ep_hi
            ):
                print(
                    f"  ⚠️  Initial patience {initial_trial_params['training.early_stopping_patience']} out of range [{ep_lo}, {ep_hi}] - skipping enqueue"
                )
                raise ValueError("Initial patience out of range")

            nl_lo, nl_hi = OPTUNA_RANGES.get("model.nlayers", (1, 4))
            if not (nl_lo <= initial_trial_params["model.nlayers"] <= nl_hi):
                print(
                    f"  ⚠️  Initial nlayers {initial_trial_params['model.nlayers']} out of range [{nl_lo}, {nl_hi}] - skipping enqueue"
                )
                raise ValueError("Initial nlayers out of range")

            dr_lo, dr_hi = OPTUNA_RANGES.get("model.dropout", (0.2, 0.7))
            if not (dr_lo <= initial_trial_params["model.dropout"] <= dr_hi):
                print(
                    f"  ⚠️  Initial dropout {initial_trial_params['model.dropout']:.4f} out of range [{dr_lo}, {dr_hi}] - skipping enqueue"
                )
                raise ValueError("Initial dropout out of range")

            # All checks passed, enqueue the trial
            study.enqueue_trial(initial_trial_params)
            print("🔁 Enqueued initial seed trial from config.optuna.initial_params")
        except ValueError:
            print(
                "⚠️  Skipping initial params - they are outside the new anti-overfitting ranges"
            )
        except Exception as e:
            print("⚠️ Could not enqueue seed trial:", e)

    # ---- Add callback to cleanup after every 5 trials ----
    def cleanup_callback(study, trial):
        if trial.number > 0 and trial.number % 5 == 0:
            cleanup_old_checkpoints_and_logs(
                base_cfg.training.checkpoint_dir,
                base_cfg.training.log_dir,
                study=study,
                keep_top_n=10,
            )

    # Study-level early stop if plateau with min_delta
    es_patience = int(
        getattr(getattr(base_cfg, "optuna", {}), "study_early_stop_patience", 25)
    )
    es_min_delta = float(
        getattr(getattr(base_cfg, "optuna", {}), "study_early_stop_min_delta", 0.005)
    )
    best_seen = {
        "value": None,
        "since": 0,
    }

    def early_stop_callback(study, trial):
        nonlocal best_seen
        if study.best_value is None:
            return
        if (
            best_seen["value"] is None
            or study.best_value > best_seen["value"] + es_min_delta
        ):
            best_seen["value"] = study.best_value
            best_seen["since"] = 0
        else:
            best_seen["since"] += 1
            if best_seen["since"] >= es_patience:
                print(
                    f"⏹️  Study early-stopping: no improvement > {es_min_delta} for {es_patience} trials. Stopping study."
                )
                study.stop()

    # ---- Pre-load data module once to avoid reloading .pt cache for every trial ----
    # This is critical: torch.load() on a 46GB .pt file takes ~242s per trial if not cached
    # Loading once here and reusing across all trials saves ~7,260s (2+ hours) for 30 trials
    # ALSO: Capture the config AFTER prepare_data() since it modifies cfg (e.g., sets pad_id)
    print("\n🔄 Pre-loading dataset (shared across all trials)...")
    data_preload_start = time.time()
    shared_data_module = None
    shared_post_prepare_cfg = None
    try:
        # Set cache config for pre-loading
        base_cfg_preload = copy.deepcopy(base_cfg)
        base_cfg_preload.preprocess.use_cache = True
        base_cfg_preload.preprocess.save = True
        base_cfg_preload.dataset.max_walk_length = 80
        base_cfg_preload.dataset.num_walks = 5000000
        base_cfg_preload.training.num_workers = 0
        base_cfg_preload.training.persistent_workers = False

        shared_data_module = prepare_data(base_cfg_preload)
        # IMPORTANT: capture the updated config after prepare_data() modifies it
        # (e.g., sets pad_id from cache metadata). Each trial will use this as base.
        shared_post_prepare_cfg = base_cfg_preload
        data_preload_time = time.time() - data_preload_start
        print(
            f"✅ Data pre-loaded in {data_preload_time:.1f}s (will be reused for all {args.n_trials} trials)"
        )
    except Exception as e:
        print(f"⚠️  Could not pre-load data: {e}. Trials will load data individually.")
        shared_data_module = None
        shared_post_prepare_cfg = None

    # ---- Optimize ----
    objective = objective_factory(
        base_cfg,
        args.device,
        enable_pruning=True,
        shared_data_module=shared_data_module,
        shared_post_prepare_cfg=shared_post_prepare_cfg,
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        callbacks=[cleanup_callback, early_stop_callback],
    )

    # ---- Final cleanup ----
    cleanup_old_checkpoints_and_logs(
        base_cfg.training.checkpoint_dir,
        base_cfg.training.log_dir,
        study=study,
        keep_top_n=10,
    )

    # ---- Print results ----
    print(f"\n" + "=" * 60)
    print("🏆 OPTUNA OPTIMIZATION COMPLETED!")
    print("=" * 60)

    # Check if we have any completed trials
    if study.best_trial is None or study.best_value is None:
        print("❌ No trials completed successfully!")
        print("   Check logs for errors.")
        return

    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best validation AUC: {study.best_value:.6f}")
    print(f"\n💡 Note: Walk parameters were FIXED (not tuned):")
    print(f"   max_walk_length = 80, num_walks = 5,000,000, batch_size = 1024")
    print("\n📋 Best hyperparameters:")

    # Group parameters by category for better readability
    params = study.best_trial.params

    print("\n🏋️  Training:")
    for key in sorted(params.keys()):
        if key.startswith("training."):
            print(f"  {key}: {params[key]}")

    print("\n🧠 Model Architecture:")
    for key in sorted(params.keys()):
        if key.startswith("model."):
            print(f"  {key}: {params[key]}")

    # ---- Save best config ----
    # Save best config inside the experiment optuna directory
    out_yaml = os.path.join(
        resolved.get("optuna_dir", "."),
        f"best_params_optuna_{base_cfg.training.exp_name}.yaml",
    )
    best_cfg = {
        # FIXED parameters (not tuned, but included for completeness)
        "dataset": {
            "max_walk_length": 80,  # FIXED
            "num_walks": 5000000,  # FIXED
        },
        "training": {
            "lr": params.get("training.lr"),
            "weight_decay": params.get("training.weight_decay"),
            "batch_size": 1024,  # FIXED
            "gradient_clip_val": params.get("training.gradient_clip_val"),
            "early_stopping_patience": params.get("training.early_stopping_patience"),
            "epochs": params.get("training.epochs"),
            # include the seed used for the best trial if available
            "seed": getattr(study.best_trial, "user_attrs", {}).get("seed", None),
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
    print(f"🎯 Best validation AUC achieved: {study.best_value:.6f}")

    # ---- Save study for future analysis ----
    import joblib

    study_path = os.path.join(
        resolved.get("optuna_dir", "."),
        f"optuna_study_{base_cfg.training.exp_name}.pkl",
    )
    joblib.dump(study, study_path)
    print(f"📊 Saved complete study to: {study_path}")

    # ---- Optional: Plot top trials and copy their checkpoints into the optuna folder ----
    # Disabled by default to avoid optional heavy dependencies (e.g., tensorflow in plotting stack).
    enable_top_trial_plots = bool(
        getattr(getattr(base_cfg, "optuna", {}), "enable_top_trial_plots", False)
    )
    if enable_top_trial_plots:
        try:
            import plot_metrics

            top_k = min(3, len(study.trials))
            # Filter out trials with None values (pruned trials) before sorting
            completed_trials = [t for t in study.trials if t.value is not None]
            if len(completed_trials) == 0:
                print("⚠️ No completed trials to plot")
                return
            sorted_trials = sorted(
                completed_trials, key=lambda t: t.value, reverse=True
            )
            top_trials = sorted_trials[:top_k]

            top_ckpt_dir = os.path.join(
                resolved.get("optuna_dir", "."), "top_checkpoints"
            )
            os.makedirs(top_ckpt_dir, exist_ok=True)

            top_trials_info = []
            for t in top_trials:
                tr_num = t.number
                # Reconstruct the per-trial log directory used by the trainer
                trial_log_dir = os.path.join(
                    resolved.get("log_dir", "."),
                    f"{base_cfg.dataset.name}-{base_cfg.training.exp_name}-optuna-t{tr_num}",
                    f"trial_{tr_num}",
                )
                plot_metrics.plot_from_logdir(
                    trial_log_dir,
                    resolved.get("plots_dir", "."),
                    prefix=f"trial_{tr_num}_",
                )

                # copy checkpoint for this trial (pick newest matching file)
                ckpt_pattern = os.path.join(
                    resolved.get("checkpoint_dir", "."), f"trial_{tr_num}-*.ckpt"
                )
                matches = glob.glob(ckpt_pattern)
                if matches:
                    newest = max(matches, key=os.path.getmtime)
                    dst = os.path.join(top_ckpt_dir, f"trial_{tr_num}.ckpt")
                    shutil.copy2(newest, dst)
                    print(f"Copied checkpoint for trial {tr_num} -> {dst}")
                    ckpt_path = dst
                else:
                    print(
                        f"No checkpoint found for trial {tr_num} (pattern={ckpt_pattern})"
                    )
                    ckpt_path = None

                # collect metadata
                # include recorded seed (if present) for reproducibility
                try:
                    seed_for_trial = t.user_attrs.get("seed", None)
                except Exception:
                    seed_for_trial = None

                top_trials_info.append(
                    {
                        "trial_number": tr_num,
                        "value": t.value,
                        "params": t.params,
                        "seed": seed_for_trial,
                        "checkpoint": ckpt_path,
                    }
                )

            # Save top trials metadata
            try:
                import json

                meta_path = os.path.join(
                    resolved.get("optuna_dir", "."), "top_trials.json"
                )
                with open(meta_path, "w") as mf:
                    json.dump(top_trials_info, mf, indent=2)
                print(f"Saved top trials metadata to: {meta_path}")
            except Exception as e:
                print("Could not save top trials metadata:", e)
        except Exception as e:
            print("Could not generate top-trial plots/checkpoints:", e)
    else:
        print(
            "ℹ️ Skipping optional top-trial plots/checkpoints (optuna.enable_top_trial_plots=False)"
        )

    print("\n🚀 Use the best configuration to train your final model!")
    print("=" * 60)


if __name__ == "__main__":
    main()
