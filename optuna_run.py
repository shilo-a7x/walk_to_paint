import os
import copy
import argparse
import shutil
import glob
import time
import traceback
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

# =============================================================================
# Fine-tuning-scale hyperparameter search space (rewritten 2026-07-22)
# =============================================================================
# Rewritten against the current CSR-cache / edge_cover-sampler / D-R-L pipeline
# (the pre-2026-07 version predated all three and hard-coded a stale 5M-walk
# k_cover-era budget -- see CLAUDE.md "OPEN WORKSTREAMS" / plan-stats-rigor.md).
#
# Design choices, see CLAUDE.md's cost/performance-tradeoff conventions and the
# 2026-07-22 planning discussion:
#   - Walk sampling (dataset.walk_strategy/num_walks/max_walk_length) is NOT
#     searched and NOT overridden here -- it comes straight from
#     configs/<dataset>.yaml, already the product of dedicated E25/E26 budget
#     sweeps. Re-exposing it to Optuna would re-litigate a settled question.
#   - model.local_attention_window is FIXED (not searched) at the production
#     LocalAttn4 value (4) via --local-attention-window, since configs/*.yaml
#     still bake in `null` and the LocalAttn4-vs-full-attention question is
#     independently settled (CLAUDE.md "Attention variant: full vs. local").
#   - model.hardness_* is never touched -- H is scrapped everywhere as of
#     2026-07-19 (CLAUDE.md "Hardness reweighting (H): scrapped everywhere").
#   - training.epochs is FIXED at the loaded config's value, not searched: it
#     sets the CosineAnnealingLR horizon, and early stopping already controls
#     effective training length -- searching both was redundant in the old
#     script.
#   - Ordered/dimension-like params (head_dim, hidden_dim, nlayers) use
#     suggest_int with a step instead of suggest_categorical: with a small
#     trial budget (~20-30), Optuna's TPE sampler models ordered int/float
#     params with a continuous KDE and can exploit "bigger tends to help"
#     trends from a handful of trials, whereas suggest_categorical treats each
#     choice as an independent, unordered arm and gets no benefit from
#     ordering. suggest_categorical is kept only for genuinely nominal choices
#     (model.nhead -- 3 nested-tensor-friendly even values, not a quantity
#     worth interpolating between).
#   - embedding_dim/nhead used to be sampled as two independent categoricals
#     with a reject-retry loop (max 10 attempts, TrialPruned on failure) to
#     satisfy embedding_dim % nhead == 0 (validate_config, src/utils/config.py).
#     Reparameterized here as nhead * head_dim -- always valid by
#     construction, zero wasted trials.
#   - training.lr's old range (1e-5, 1e-3) no longer contains bitcoin-alpha's
#     current production lr (0.0016) -- widened to (1e-5, 5e-3).
#   - Added model.node_replace_prob / node_replace_unk_ratio (the "R" node
#     token replacement regularizer, CLAUDE.md "Training feature flags") --
#     previously entirely absent from the search space despite being a live
#     production knob. node_context_mode itself stays fixed at "replace" (the
#     production default) -- only its two probability knobs are tuned.
OPTUNA_RANGES = {
    "training.lr": {"type": "float", "low": 1e-5, "high": 5e-3, "log": True},
    "training.weight_decay": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
    "training.gradient_clip_val": {"type": "float", "low": 0.3, "high": 2.0},
    "training.early_stopping_patience": {"type": "int", "low": 5, "high": 15},
    "model.nhead": {"type": "categorical", "choices": [2, 4, 8]},
    # embedding_dim = nhead * head_dim (computed per-trial, always valid).
    "model.head_dim": {"type": "int", "low": 8, "high": 32, "step": 8},
    "model.hidden_dim": {"type": "int", "low": 16, "high": 256, "step": 16},
    "model.nlayers": {"type": "int", "low": 1, "high": 6},
    "model.dropout": {"type": "float", "low": 0.1, "high": 0.6},
    "model.node_replace_prob": {"type": "float", "low": 0.0, "high": 0.5},
    "model.node_replace_unk_ratio": {"type": "float", "low": 0.3, "high": 1.0},
}


def _suggest(trial, name):
    """Sample `name` from OPTUNA_RANGES using the right Optuna distribution."""
    spec = OPTUNA_RANGES[name]
    if spec["type"] == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if spec["type"] == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if spec["type"] == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown OPTUNA_RANGES type for {name}: {spec['type']}")


def _clamp_to_range(name, value):
    """Clamp/snap a config-provided initial value into OPTUNA_RANGES[name]'s bounds."""
    spec = OPTUNA_RANGES[name]
    if spec["type"] == "categorical":
        choices = spec["choices"]
        if value in choices:
            return value
        return min(choices, key=lambda c: abs(c - value))
    lo, hi = spec["low"], spec["high"]
    value = max(lo, min(hi, value))
    if spec["type"] == "int":
        step = spec.get("step", 1)
        n_steps = round((value - lo) / step)
        value = int(max(lo, min(hi, lo + n_steps * step)))
    return value


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
    p.add_argument(
        "--n-trials", type=int, default=25, help="Number of Optuna trials (fine-tuning scale)"
    )
    p.add_argument(
        "--local-attention-window",
        type=str,
        default="4",
        help="Fixed model.local_attention_window for every trial (not searched). "
        "'4' = LocalAttn4, the settled production default (CLAUDE.md). "
        "'null' = full attention.",
    )
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
    local_attention_window,
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
        local_attention_window: fixed (not searched) model.local_attention_window
                                 applied to every trial -- see OPTUNA_RANGES docstring.
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

        # ===== FIXED PARAMETERS (not part of this search, see OPTUNA_RANGES docstring) =====
        # dataset.walk_strategy/num_walks/max_walk_length: untouched, inherited from the
        # loaded config (configs/<dataset>.yaml) -- already the product of E25/E26 sweeps.
        # training.batch_size/epochs: untouched, inherited from the loaded config.
        cfg.training.num_workers = 0  # Optuna stability: avoid DataLoader worker teardown noise
        cfg.training.persistent_workers = False
        cfg.preprocess.use_cache = True
        cfg.preprocess.save = True
        cfg.model.local_attention_window = local_attention_window

        # ===== TRAINING HYPERPARAMETERS =====
        cfg.training.lr = _suggest(trial, "training.lr")
        cfg.training.weight_decay = _suggest(trial, "training.weight_decay")
        cfg.training.gradient_clip_val = _suggest(trial, "training.gradient_clip_val")
        cfg.training.early_stopping_patience = _suggest(
            trial, "training.early_stopping_patience"
        )

        # ===== MODEL ARCHITECTURE HYPERPARAMETERS =====
        # embedding_dim = nhead * head_dim: always satisfies validate_config's
        # embedding_dim % nhead == 0 constraint by construction, no reject-retry needed.
        cfg.model.nhead = _suggest(trial, "model.nhead")
        head_dim = _suggest(trial, "model.head_dim")
        cfg.model.embedding_dim = cfg.model.nhead * head_dim

        cfg.model.hidden_dim = _suggest(trial, "model.hidden_dim")
        cfg.model.nlayers = _suggest(trial, "model.nlayers")
        cfg.model.dropout = _suggest(trial, "model.dropout")
        cfg.model.node_replace_prob = _suggest(trial, "model.node_replace_prob")
        cfg.model.node_replace_unk_ratio = _suggest(trial, "model.node_replace_unk_ratio")

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
            f"  Walk strategy: {cfg.dataset.walk_strategy} (FIXED, from config), "
            f"num_walks: {cfg.dataset.num_walks:,} (FIXED, from config)"
        )
        print(
            f"  LR: {cfg.training.lr:.2e}, Weight decay: {cfg.training.weight_decay:.2e}"
        )
        print(
            f"  Batch size: {cfg.training.batch_size} (FIXED), Gradient clip: {cfg.training.gradient_clip_val:.2f}"
        )
        print(
            f"  Model: nhead={cfg.model.nhead}, head_dim={head_dim} -> emb_dim={cfg.model.embedding_dim}, "
            f"hidden_dim={cfg.model.hidden_dim}"
        )
        print(
            f"  Transformer: {cfg.model.nlayers} layers, local_attention_window={cfg.model.local_attention_window} (FIXED)"
        )
        print(
            f"  Dropout: {cfg.model.dropout:.2f}, Epochs: {cfg.training.epochs} (FIXED)"
        )
        print(
            f"  Early stopping patience: {cfg.training.early_stopping_patience}"
        )
        print(
            f"  Node replace prob: {cfg.model.node_replace_prob:.2f}, "
            f"unk_ratio: {cfg.model.node_replace_unk_ratio:.2f}"
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

            # Extract metrics -- objective is raw val AUC (single source of truth,
            # matches how every other result in this project is reported; the old
            # composite alpha*AUC - beta*loss score was mislabeled as "AUC" in logs
            # and never justified, so it's dropped here).
            val_auc = float(val_metrics[0]["val_auc_epoch"])
            val_loss = float(val_metrics[0].get("val_loss", 0.0))

            print(f"✅ Trial {trial.number} completed:")
            print(f"   Val AUC: {val_auc:.4f}, Val Loss: {val_loss:.4f}")

            return val_auc

        except optuna.TrialPruned:
            raise
        except Exception as e:
            # Surface real failures as PRUNED trials (visibly distinct from a bad-but-
            # valid AUC) instead of silently returning 0.0 -- returning a fake 0.0 AUC
            # for e.g. an OOM or a config bug would poison TPE with a fabricated data
            # point and mask the underlying failure.
            print(f"❌ Trial {trial.number} failed: {e}")
            traceback.print_exc()
            raise optuna.TrialPruned(f"Trial {trial.number} failed with exception: {e}")

    return objective


def main():

    args = parse_args()

    local_attention_window_raw = args.local_attention_window.strip().lower()
    local_attention_window = (
        None if local_attention_window_raw in ("null", "none") else int(local_attention_window_raw)
    )

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

    print(f"🎯 Starting Optuna hyperparameter fine-tuning with {args.n_trials} trials")
    print(f"📊 Objective: raw validation AUC (val_auc_epoch)")
    print(
        f"🎯 Fixed (not searched): walk_strategy={base_cfg.dataset.walk_strategy}, "
        f"num_walks={base_cfg.dataset.num_walks:,}, epochs={base_cfg.training.epochs}, "
        f"local_attention_window={local_attention_window}"
    )
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

        t_ip = ip.get("training", {})
        m_ip = ip.get("model", {})

        initial_embedding_dim = _pick("model.embedding_dim", m_ip.get("embedding_dim"))
        initial_nhead = _pick("model.nhead", m_ip.get("nhead"))
        initial_head_dim = (
            int(initial_embedding_dim) // int(initial_nhead)
            if (initial_embedding_dim is not None and initial_nhead)
            else None
        )

        # Only the keys that are actually part of OPTUNA_RANGES today -- config
        # sections may still contain now-unsearched leftovers (e.g. dataset.num_walks,
        # training.epochs); those are simply not picked up here.
        candidate_params = {
            "training.lr": _pick("training.lr", t_ip.get("lr")),
            "training.weight_decay": _pick("training.weight_decay", t_ip.get("weight_decay")),
            "training.gradient_clip_val": _pick(
                "training.gradient_clip_val", t_ip.get("gradient_clip_val")
            ),
            "training.early_stopping_patience": _pick(
                "training.early_stopping_patience", t_ip.get("early_stopping_patience")
            ),
            "model.nhead": initial_nhead,
            "model.head_dim": initial_head_dim,
            "model.hidden_dim": _pick("model.hidden_dim", m_ip.get("hidden_dim")),
            "model.nlayers": _pick("model.nlayers", m_ip.get("nlayers")),
            "model.dropout": _pick("model.dropout", m_ip.get("dropout")),
            "model.node_replace_prob": _pick(
                "model.node_replace_prob", m_ip.get("node_replace_prob", 0.2)
            ),
            "model.node_replace_unk_ratio": _pick(
                "model.node_replace_unk_ratio", m_ip.get("node_replace_unk_ratio", 0.7)
            ),
        }

        initial_trial_params = {}
        for name, raw_value in candidate_params.items():
            if raw_value is None:
                continue
            try:
                spec = OPTUNA_RANGES[name]
                value = float(raw_value) if spec["type"] == "float" else int(raw_value)
                initial_trial_params[name] = _clamp_to_range(name, value)
            except Exception as e:
                print(f"  ⚠️  Could not prepare initial value for {name}: {e}")

        if initial_trial_params:
            try:
                study.enqueue_trial(initial_trial_params)
                print(
                    f"🔁 Enqueued initial seed trial from config.optuna.initial_params: "
                    f"{initial_trial_params}"
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
    # This is critical: torch.load() on a large .pt file takes tens-to-hundreds of
    # seconds per trial if not cached. Loading once here and reusing across all trials
    # saves that cost for every subsequent trial.
    # ALSO: Capture the config AFTER prepare_data() since it modifies cfg (e.g., sets pad_id).
    # Walk params (strategy/num_walks/max_walk_length) are NOT forced here -- they come
    # from the loaded dataset config, so the shared cache matches each dataset's actual
    # production walk budget instead of a hard-coded one.
    print("\n🔄 Pre-loading dataset (shared across all trials)...")
    data_preload_start = time.time()
    shared_data_module = None
    shared_post_prepare_cfg = None
    try:
        base_cfg_preload = copy.deepcopy(base_cfg)
        base_cfg_preload.preprocess.use_cache = True
        base_cfg_preload.preprocess.save = True
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
        local_attention_window,
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
    print(f"\n💡 Note: not searched, fixed at this run's config value:")
    print(
        f"   walk_strategy={base_cfg.dataset.walk_strategy}, num_walks={base_cfg.dataset.num_walks:,}, "
        f"batch_size={base_cfg.training.batch_size}, epochs={base_cfg.training.epochs}, "
        f"local_attention_window={local_attention_window}"
    )
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
    # Save best config inside the experiment optuna directory.
    # dataset.* is deliberately NOT included -- nothing there was searched; merge this
    # file's training/model sections on top of the same configs/<dataset>.yaml used for
    # this run to reproduce.
    out_yaml = os.path.join(
        resolved.get("optuna_dir", "."),
        f"best_params_optuna_{base_cfg.training.exp_name}.yaml",
    )
    best_nhead = params.get("model.nhead")
    best_head_dim = params.get("model.head_dim")
    best_embedding_dim = (
        best_nhead * best_head_dim if (best_nhead and best_head_dim) else None
    )
    best_cfg = {
        "training": {
            "lr": params.get("training.lr"),
            "weight_decay": params.get("training.weight_decay"),
            "gradient_clip_val": params.get("training.gradient_clip_val"),
            "early_stopping_patience": params.get("training.early_stopping_patience"),
            # FIXED (not searched) at this run's config value:
            "batch_size": base_cfg.training.batch_size,
            "epochs": base_cfg.training.epochs,
            # include the seed used for the best trial if available
            "seed": getattr(study.best_trial, "user_attrs", {}).get("seed", None),
        },
        "model": {
            "embedding_dim": best_embedding_dim,
            "hidden_dim": params.get("model.hidden_dim"),
            "nhead": best_nhead,
            "nlayers": params.get("model.nlayers"),
            "dropout": params.get("model.dropout"),
            "node_replace_prob": params.get("model.node_replace_prob"),
            "node_replace_unk_ratio": params.get("model.node_replace_unk_ratio"),
            # FIXED (not searched) at this run's value:
            "local_attention_window": local_attention_window,
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
