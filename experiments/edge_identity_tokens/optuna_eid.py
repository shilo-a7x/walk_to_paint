"""Small Optuna search over the edge-identity-token (EID) model's own knobs.

Adapted from ../../optuna_run.py's in-process PL-Trainer + shared-data-preload
pattern, but trimmed to the axes specific to this experiment -- production's
architecture search (nhead/hidden_dim/nlayers) is NOT re-run here, see "FIXED"
below. Reuses the real eid_src/ classes (EIDLitEdgeClassifier,
EdgeIdentityTransformerModel via it, prepare_eid_data), same discipline as
run_eid.py: no hand-rolled training loop.

Goal (per user, 2026-08-24): find out whether a genuinely good EID model exists
at all -- so far the best point found by hand (rank=4, replace=0.9, concat)
is 0.8512 test AUC on bitcoin-alpha vs. production's 0.9188 -- by searching the
EID-specific capacity/regularization knobs jointly instead of one at a time.

===== FIXED (not searched) =====
  - dataset.* (walk sampler/budget): from configs/<dataset>.yaml, already
    E25/E26-swept, not re-litigated here.
  - model.local_attention_window=4 (LocalAttn4): settled production default.
  - model.dynamic_train_masking=False, model.scramble_edge_signs=False:
    unsupported by EIDLitEdgeClassifier (see its module docstring); enforced
    the same way run_eid.py does.
  - model.edge_sign_combine: forced to "concat" whenever edge_embed_rank>0
    (the only add-vs-concat comparison run so far, rank=8/replace=0.9,
    favored concat by ~0.7pp: 0.8453 vs 0.8381 -- one data point, not a
    proof, but not worth spending trial budget on with a small study).
    Irrelevant/unused when edge_embed_rank=0 (model.py's unified-table branch
    ignores this flag entirely).
  - model.nhead/hidden_dim/nlayers: kept at the dataset's already-tuned
    production values (configs/<dataset>.yaml) -- retuning transformer
    depth/width is a separate question from "does edge identity help at all,"
    and production's own per-dataset search already picked reasonable values.
  - training.batch_size, training.epochs: from config.

===== SEARCHED (11 dimensions) =====
  - model.edge_embed_rank (int, 0-32): 0 = disabled, the original unified
    full-width table (still viable in principle now that edge_replace_prob,
    unk_ratio, and node_replace_* are ALL being tuned jointly here for the
    first time -- earlier full-table runs (EID_REG1-5, up to 0.8320 at
    replace=0.9) never combined high replace with a tuned unk_ratio or R
    regularizer, so "simple table always loses to LoRA" isn't actually
    established, just under-tested). >0 = ALBERT-style low-rank factorization
    per this file's model.py. sign_embed_dim/node_embed_dim/edge_embed_
    weight_decay below are still sampled every trial for TPE consistency but
    only applied to cfg (and only affect the objective) when rank>0.
  - model.node_replace_prob (float, 0.0-0.5) and model.node_replace_unk_ratio
    (float, 0.3-1.0): the vertex-side "R" regularizer -- reused from
    optuna_run.py's production range, but never actually retuned for EID
    specifically (production's 0.2/0.7 defaults were just inherited
    unchanged in every run so far) despite EID's much larger total vocab
    potentially changing the right operating point.
  - model.sign_embed_dim (int, 4-32 step 4): width of the sign channel in
    concat mode.
  - model.node_embed_dim (categorical, incl. 0=tied-to-content_dim): the new
    lever added this session -- decouples vertex-embedding width from
    whatever's left of content_dim after the edge/sign split, via a
    node_proj up-projection (model.py).
  - head_dim (int, 8-32 step 8) -> model.embedding_dim = nhead(fixed)*head_dim:
    same reparameterization trick as optuna_run.py, always satisfies
    validate_config's embedding_dim % nhead == 0 by construction.
  - model.edge_replace_prob (float, 0.0-0.95): identity-corruption
    regularizer: monotonically improving from 0.2->0.9 in the manual sweep,
    upper end not yet found.
  - model.edge_replace_unk_ratio (float, 0.0-1.0): UNK-vs-random-other-edge
    replacement mix -- never swept before this script (always left at its
    0.7 default).
  - model.edge_embed_weight_decay (conditional float, log 1e-4-1.0, or
    unset/0): direct L2 penalty on just the edge table, a capacity-control
    mechanism distinct from edge_replace_prob's corruption-based one.
  - training.lr (float, log 1e-5-5e-3): reused from optuna_run.py's range,
    EID's much larger vocab/embedding table may want a different point.
  - model.dropout (float, 0.1-0.6): reused from optuna_run.py's range.

===== Pruning =====
Two layers, both active by default:
  1. optuna.pruners.MedianPruner -- relative, standard: prunes a trial if its
     intermediate val_auc is worse than the median of other trials at the
     same epoch (needs n_startup_trials completed first before it starts
     comparing).
  2. AbsoluteFloorPruning (this file) -- absolute, aggressive, per user
     request: after --prune-warmup-epochs epochs, kills a trial outright if
     val_auc_epoch is still below --prune-floor-frac * --vanilla-auc (default
     0.75 * 0.9188 = 0.689 on bitcoin-alpha). Catches the specific known
     failure mode (rank=0/insufficient replace regularization collapsing to
     ~0.499 chance, per EID_REAL_v1) well before MedianPruner would notice on
     its own, and well before such a trial would otherwise run to
     early-stopping completion -- these configs are never going to be
     competitive, no reason to pay their full training cost. The floor is
     relative to a known real ceiling (vanilla PEWTER's own production AUC on
     this dataset), not an arbitrary constant, so it stays meaningful if this
     script is later pointed at a different dataset (pass --vanilla-auc to
     match).

===== Multi-GPU =====
Trials are independent (each trains its own model from scratch), so scaling to
all 4 GPUs is just launching this script 4 times, one per --device, all
pointed at the SAME --study-name / same dataset -- they share one JournalStorage
file (outputs/.../optuna/optuna_eid_study.log) and one Optuna study, so TPE
sees every trial's result regardless of which GPU ran it, not 4 independent
searches:

  for d in 0 1 2 3; do
    nohup .venv/bin/python experiments/edge_identity_tokens/optuna_eid.py \
        dataset.name=bitcoin-alpha training.exp_name=EID_OPTUNA_v1 --device $d \
        --n-trials 20 --total-trials 80 \
        > experiments/edge_identity_tokens/optuna_eid_gpu${d}.log 2>&1 &
  done

===== Resume =====
JournalStorage persists to disk and `load_if_exists=True` is always on, so
just rerunning the exact same command (same training.exp_name -> same
study_name) picks up every already-completed trial's history and continues
TPE from there -- no special resume flag needed for that part. --total-trials
(optional, shared target across all worker processes) makes relaunching
idempotent: each process checks how many trials already exist in the shared
study and only runs enough of its own --n-trials budget to top up to that
target, so re-running the same launch command after an interruption (or after
adding more GPUs mid-study) does the right thing instead of overshooting.
Omit --total-trials to just add --n-trials more trials unconditionally.

Usage (single GPU):
  .venv/bin/python experiments/edge_identity_tokens/optuna_eid.py \
      dataset.name=bitcoin-alpha training.exp_name=EID_OPTUNA_v1 --device 0 \
      --n-trials 30
"""

import argparse
import copy
import os
import random
import time
import traceback

import numpy as np
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from optuna_integration import PyTorchLightningPruningCallback
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from src.utils.config import load_config, get_seed, validate_config
from src.utils.paths import resolve_outputs_dirs

from experiments.edge_identity_tokens.run_eid import ensure_eid_cache, EID_CACHE_PATH
from experiments.edge_identity_tokens.eid_src.data.prepare_eid_data import prepare_eid_data
from experiments.edge_identity_tokens.eid_src.model.lit_model import EIDLitEdgeClassifier


OPTUNA_RANGES = {
    "model.edge_embed_rank": {"type": "int", "low": 0, "high": 32},
    "model.sign_embed_dim": {"type": "int", "low": 4, "high": 32, "step": 4},
    "model.node_embed_dim": {"type": "categorical", "choices": [0, 16, 32, 48, 64, 96]},
    "model.head_dim": {"type": "int", "low": 8, "high": 32, "step": 8},
    "model.edge_replace_prob": {"type": "float", "low": 0.0, "high": 0.95},
    "model.edge_replace_unk_ratio": {"type": "float", "low": 0.0, "high": 1.0},
    "model.edge_embed_weight_decay": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
    "model.node_replace_prob": {"type": "float", "low": 0.0, "high": 0.5},
    "model.node_replace_unk_ratio": {"type": "float", "low": 0.3, "high": 1.0},
    "training.lr": {"type": "float", "low": 1e-5, "high": 5e-3, "log": True},
    "model.dropout": {"type": "float", "low": 0.1, "high": 0.6},
}


def _suggest(trial, name):
    spec = OPTUNA_RANGES[name]
    if spec["type"] == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if spec["type"] == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if spec["type"] == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown OPTUNA_RANGES type for {name}: {spec['type']}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--n-trials", type=int, default=30,
                    help="Trials this process will run (its share of the budget).")
    p.add_argument("--total-trials", type=int, default=None,
                    help="Optional shared target across all worker processes on this "
                         "study -- caps this process's --n-trials so relaunching the "
                         "same command after an interruption tops up rather than "
                         "overshoots. Omit to just run --n-trials unconditionally.")
    p.add_argument("--study-name", type=str, default=None,
                    help="Optuna study name; defaults to training.exp_name. Use the "
                         "same value across all --device launches to share one study.")
    p.add_argument("--vanilla-auc", type=float, default=0.9188,
                    help="Reference production PEWTER test AUC on this dataset -- "
                         "AbsoluteFloorPruning's floor is a fraction of this.")
    p.add_argument("--prune-floor-frac", type=float, default=0.75,
                    help="Prune a trial if val_auc_epoch stays below "
                         "prune_floor_frac * vanilla_auc past the warmup epochs.")
    p.add_argument("--prune-warmup-epochs", type=int, default=3,
                    help="Epochs to let a trial run before the absolute floor applies.")
    # parse_known_args (not nargs=REMAINDER): REMAINDER swallows every later token,
    # flags included, the instant it hits the first dotlist override -- bit run_eid.py
    # this same session (see CLAUDE.md's GPU-pinning gotcha) when --device came after
    # a dotlist arg on the command line and silently stuck at its default. This parses
    # correctly regardless of where the dotlist overrides fall relative to the flags.
    args, overrides = p.parse_known_args()
    args.overrides = overrides
    return args


class AbsoluteFloorPruning(Callback):
    """Aggressive absolute-threshold pruning, complementary to Optuna's own
    (relative, trial-vs-trial) MedianPruner -- see this file's module
    docstring, "Pruning" section, for the rationale."""

    def __init__(self, trial, floor, warmup_epochs):
        self.trial = trial
        self.floor = floor
        self.warmup_epochs = warmup_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.current_epoch < self.warmup_epochs:
            return
        val_auc = trainer.callback_metrics.get("val_auc_epoch")
        if val_auc is None:
            return
        val_auc = float(val_auc)
        if val_auc < self.floor:
            print(f"Trial {self.trial.number}: AbsoluteFloorPruning at epoch "
                  f"{trainer.current_epoch}, val_auc={val_auc:.4f} < floor={self.floor:.4f}")
            raise optuna.TrialPruned(
                f"val_auc={val_auc:.4f} below absolute floor {self.floor:.4f} "
                f"at epoch {trainer.current_epoch}"
            )


def build_trainer(cfg, val_loader, trial, floor_pruning_kwargs=None, enable_pruning=True):
    logger = TensorBoardLogger(
        save_dir=cfg.training.log_dir,
        name=f"{cfg.dataset.name}-{cfg.training.exp_name}",
        version=f"trial_{trial.number}",
    )
    checkpoint = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename=f"trial_{trial.number}-" + "{epoch:02d}-{val_auc_epoch:.4f}",
        monitor="val_auc_epoch",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    early_stopping = EarlyStopping(
        monitor="val_auc_epoch",
        patience=cfg.training.early_stopping_patience,
        mode="max",
        min_delta=float(getattr(cfg.training, "early_stopping_min_delta", 0.001)),
    )
    callbacks = [checkpoint, early_stopping]
    if enable_pruning:
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_auc_epoch"))
    if floor_pruning_kwargs is not None:
        callbacks.append(AbsoluteFloorPruning(trial, **floor_pruning_kwargs))

    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        logger=logger,
        log_every_n_steps=50,
        default_root_dir=cfg.training.checkpoint_dir,
        accelerator="gpu" if cfg.training.use_cuda and torch.cuda.is_available() else "cpu",
        callbacks=callbacks,
        gradient_clip_val=getattr(cfg.training, "gradient_clip_val", 1.0),
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    return trainer, checkpoint


def objective_factory(base_cfg, device, shared_post_prepare_cfg, floor_pruning_kwargs=None):
    def objective(trial: optuna.trial.Trial):
        cfg = copy.deepcopy(shared_post_prepare_cfg)
        cfg.training.exp_name = f"{base_cfg.training.exp_name}-t{trial.number}"

        # ===== FIXED (see module docstring) =====
        cfg.training.num_workers = 0
        cfg.training.persistent_workers = False
        cfg.model.local_attention_window = 4
        cfg.model.dynamic_train_masking = False
        cfg.model.scramble_edge_signs = False

        # ===== SEARCHED =====
        # Sample every dimension every trial (TPE consistency) even though
        # sign_embed_dim/node_embed_dim/edge_embed_weight_decay/edge_sign_combine
        # only affect the objective when edge_embed_rank > 0 -- see module docstring.
        edge_embed_rank = _suggest(trial, "model.edge_embed_rank")
        sign_embed_dim = _suggest(trial, "model.sign_embed_dim")
        node_embed_dim = _suggest(trial, "model.node_embed_dim")
        edge_embed_weight_decay = _suggest(trial, "model.edge_embed_weight_decay")

        cfg.model.edge_embed_rank = edge_embed_rank
        if edge_embed_rank > 0:
            cfg.model.edge_sign_combine = "concat"
            cfg.model.sign_embed_dim = sign_embed_dim
            cfg.model.node_embed_dim = node_embed_dim
            cfg.model.edge_embed_weight_decay = edge_embed_weight_decay

        head_dim = _suggest(trial, "model.head_dim")
        cfg.model.embedding_dim = cfg.model.nhead * head_dim
        cfg.model.edge_replace_prob = _suggest(trial, "model.edge_replace_prob")
        cfg.model.edge_replace_unk_ratio = _suggest(trial, "model.edge_replace_unk_ratio")
        cfg.model.node_replace_prob = _suggest(trial, "model.node_replace_prob")
        cfg.model.node_replace_unk_ratio = _suggest(trial, "model.node_replace_unk_ratio")
        cfg.training.lr = _suggest(trial, "training.lr")
        cfg.model.dropout = _suggest(trial, "model.dropout")

        seed = get_seed(cfg)
        seed_everything(seed, workers=True)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        if cfg.training.use_cuda and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

        print(f"\nTrial {trial.number}: edge_embed_rank={edge_embed_rank} "
              f"({'factorized' if edge_embed_rank > 0 else 'simple/unified table'}), "
              f"sign_embed_dim={sign_embed_dim}, node_embed_dim={node_embed_dim}, "
              f"edge_embed_weight_decay={edge_embed_weight_decay:.2e} [rank>0 only], "
              f"embedding_dim={cfg.model.embedding_dim} (head_dim={head_dim}), "
              f"edge_replace_prob={cfg.model.edge_replace_prob:.2f}, "
              f"edge_replace_unk_ratio={cfg.model.edge_replace_unk_ratio:.2f}, "
              f"node_replace_prob={cfg.model.node_replace_prob:.2f}, "
              f"node_replace_unk_ratio={cfg.model.node_replace_unk_ratio:.2f}, "
              f"lr={cfg.training.lr:.2e}, dropout={cfg.model.dropout:.2f}")

        try:
            model = EIDLitEdgeClassifier(cfg)
            val_loader = shared_val_loader
            trainer, checkpoint = build_trainer(cfg, val_loader, trial, floor_pruning_kwargs)
            trainer.fit(model, shared_train_loader, val_loader)

            best_ckpt_path = checkpoint.best_model_path
            eval_trainer = Trainer(
                logger=False, enable_checkpointing=False, enable_progress_bar=False,
                enable_model_summary=False,
                accelerator="gpu" if cfg.training.use_cuda and torch.cuda.is_available() else "cpu",
                devices=1 if (cfg.training.use_cuda and torch.cuda.is_available()) else None,
            )
            if best_ckpt_path and os.path.exists(best_ckpt_path):
                val_metrics = eval_trainer.validate(model, val_loader, ckpt_path=best_ckpt_path)
            else:
                val_metrics = eval_trainer.validate(model, val_loader)

            val_auc = float(val_metrics[0]["val_auc_epoch"])
            print(f"Trial {trial.number} done: val_auc={val_auc:.4f}")
            return val_auc
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            traceback.print_exc()
            raise optuna.TrialPruned(f"Trial {trial.number} failed: {e}")

    return objective


def main():
    args = parse_args()
    base_cfg = load_config(args.config, overrides=args.overrides)
    validate_config(base_cfg, context="optuna")

    if not getattr(base_cfg.training, "exp_name", None) or base_cfg.training.exp_name in (
        "walk_to_paint_experiment", "experiment",
    ):
        base_cfg.training.exp_name = f"{base_cfg.dataset.name}-EID_OPTUNA"

    seed = get_seed(base_cfg)
    seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Force a stable (non-timestamped) output dir: resolve_outputs_dirs defaults to
    # appending a per-call timestamp, which would give each --device process launched
    # for this same study its own exp_dir/optuna_dir -- silently defeating the shared
    # JournalStorage file multi-GPU relies on (they'd each write their own separate,
    # unsynchronized study instead of collaborating on one). Every process sharing a
    # training.exp_name must resolve to the identical directory.
    if "paths" not in base_cfg:
        base_cfg.paths = {}
    base_cfg.paths.append_timestamp = False
    resolved = resolve_outputs_dirs(base_cfg)
    print(f"Outputs -> exp_dir: {resolved['exp_dir']}")

    if base_cfg.training.use_cuda and torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    print(f"Starting EID Optuna search with {args.n_trials} trials on device {args.device}")

    eid_cache_path = EID_CACHE_PATH.format(dataset=base_cfg.dataset.name)
    ensure_eid_cache(base_cfg, eid_cache_path)

    print("\nPre-loading EID dataset cache (shared across all trials)...")
    t0 = time.time()
    base_cfg_preload = copy.deepcopy(base_cfg)
    base_cfg_preload.model.dynamic_train_masking = False
    base_cfg_preload.model.scramble_edge_signs = False
    base_cfg_preload.training.num_workers = 0
    base_cfg_preload.training.persistent_workers = False
    data_module = prepare_eid_data(base_cfg_preload, eid_cache_path)
    global shared_train_loader, shared_val_loader
    shared_train_loader = data_module["train"]
    shared_val_loader = data_module["val"]
    print(f"Data pre-loaded in {time.time() - t0:.1f}s (reused for all {args.n_trials} trials)")

    optuna_dir = resolved.get("optuna_dir", ".")
    os.makedirs(optuna_dir, exist_ok=True)
    storage = JournalStorage(JournalFileStorage(os.path.join(optuna_dir, "optuna_eid_study.log")))

    study_name = args.study_name or f"eid_optuna_{base_cfg.training.exp_name}"
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    n_existing = len(study.get_trials(deepcopy=False))
    print(f"Study '{study_name}': {n_existing} trial(s) already recorded "
          f"(storage: {os.path.join(optuna_dir, 'optuna_eid_study.log')})")

    n_trials_this_run = args.n_trials
    if args.total_trials is not None:
        remaining = max(0, args.total_trials - n_existing)
        n_trials_this_run = min(args.n_trials, remaining)
        print(f"--total-trials {args.total_trials} set: {remaining} remaining, "
              f"this process will run {n_trials_this_run} of its {args.n_trials}-trial budget")
        if n_trials_this_run == 0:
            print("Target already reached, nothing to do.")
            return

    floor = args.vanilla_auc * args.prune_floor_frac
    floor_pruning_kwargs = {"floor": floor, "warmup_epochs": args.prune_warmup_epochs}
    print(f"AbsoluteFloorPruning: floor={floor:.4f} "
          f"({args.prune_floor_frac:.0%} of vanilla_auc={args.vanilla_auc:.4f}), "
          f"warmup={args.prune_warmup_epochs} epochs")

    objective = objective_factory(base_cfg, args.device, base_cfg_preload, floor_pruning_kwargs)
    study.optimize(objective, n_trials=n_trials_this_run)

    print("\n" + "=" * 60)
    print("EID OPTUNA SEARCH COMPLETE")
    print("=" * 60)
    if study.best_trial is None:
        print("No trials completed successfully.")
        return
    print(f"Best val_auc: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
