import argparse
import json
import os
import pickle
import random
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

from src.data.prepare_data import prepare_data
from src.model.lit_model import LitEdgeClassifier
from src.training.callbacks import PerEpochPredictionSaver
from src.utils.config import get_seed, load_config, validate_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run post-hoc predictions/analysis from a selected checkpoint"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Experiment directory (e.g., outputs/wiki-rfa/wiki-rfa-run_20260222-123456)",
    )
    parser.add_argument(
        "--checkpoint-choice",
        type=str,
        default="best",
        choices=["best", "last", "other"],
        help="Checkpoint selection strategy",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Checkpoint path when --checkpoint-choice=other",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="val,test",
        help="Comma-separated splits to process (aggregation fits on val, reports on test)",
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default="predictions,aggregator",
        help="Comma-separated artifacts: predictions,aggregator",
    )
    parser.add_argument(
        "--agg-models",
        type=str,
        default="func_logit_power",
        help="Comma-separated aggregator model names (see _func_registry in this "
             "file for the available func_<name> choices)",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id for post-hoc artifact folder",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Optional OmegaConf dotlist overrides",
    )
    return parser.parse_args()


def _parse_epoch(path: Path):
    match = re.search(r"epoch[=\-_](\d+)", path.name)
    return int(match.group(1)) if match else None


def _parse_metric(path: Path, key: str):
    match = re.search(rf"{key}(?:_epoch)?[=\-_]([0-9]*\.?[0-9]+)", path.name)
    return float(match.group(1)) if match else None


def resolve_checkpoint(checkpoint_dir: Path, choice: str, checkpoint_path: str = None):
    if choice == "other":
        if not checkpoint_path:
            raise ValueError(
                "--checkpoint-path is required when --checkpoint-choice=other"
            )
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    all_ckpts = sorted(checkpoint_dir.glob("*.ckpt"))
    if not all_ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    if choice == "last":
        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            return last_ckpt

        epochs = [(p, _parse_epoch(p)) for p in all_ckpts]
        epochs = [(p, e) for p, e in epochs if e is not None]
        if epochs:
            return max(epochs, key=lambda x: x[1])[0]
        return max(all_ckpts, key=lambda p: p.stat().st_mtime)

    # best
    auc_scored = [(p, _parse_metric(p, "val_auc")) for p in all_ckpts]
    auc_scored = [(p, score) for p, score in auc_scored if score is not None]
    if auc_scored:
        return max(auc_scored, key=lambda x: x[1])[0]

    loss_scored = [(p, _parse_metric(p, "val_loss")) for p in all_ckpts]
    loss_scored = [(p, score) for p, score in loss_scored if score is not None]
    if loss_scored:
        return min(loss_scored, key=lambda x: x[1])[0]

    return max(all_ckpts, key=lambda p: p.stat().st_mtime)


def prediction_path(exp_dir: Path, dataset_name: str, epoch: int, split: str):
    return (
        exp_dir
        / "checkpoints"
        / f"{dataset_name}_predictions"
        / f"epoch_{epoch:03d}"
        / f"{split}_predictions.pkl"
    )


def load_prediction(exp_dir: Path, dataset_name: str, epoch: int, split: str):
    path = prediction_path(exp_dir, dataset_name, epoch, split)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _pred_prob(preds):
    """Predicted probability of the positive class, from softmax output."""
    return np.asarray(preds["probabilities"])[:, 1].astype(float)


# ── Functional-aggregator helpers ─────────────────────────────────────────────

_FUNC_EPS = 1e-9


def _wfn_edge_scores(w_fn, theta, q, ds, de, lens, rp, edge_ids, y_occur):
    """
    Compute per-edge weighted-mean score for a parametric weight function.
    w_fn(theta, q, ds, de, lens, rp) -> array of weights (length N_occurrences).
    Returns (edge_scores, edge_labels) both sorted by ascending edge_id.
    """
    w = np.maximum(w_fn(theta, q, ds, de, lens, rp), _FUNC_EPS)
    sort_idx = np.argsort(edge_ids, kind="stable")
    eids_s   = edge_ids[sort_idx]
    w_s      = w[sort_idx]
    q_s      = q[sort_idx]
    y_s      = y_occur[sort_idx]

    unique_eids, inv_idx, cnts = np.unique(eids_s, return_inverse=True, return_counts=True)
    n_e   = len(unique_eids)
    wsum  = np.bincount(inv_idx, weights=w_s,       minlength=n_e)
    wqsum = np.bincount(inv_idx, weights=w_s * q_s, minlength=n_e)
    scores = wqsum / np.maximum(wsum, _FUNC_EPS)
    # All occurrences of an edge share the same label — take first occurrence per edge
    first_occ = np.concatenate([[0], np.cumsum(cnts)[:-1]])
    labels = y_s[first_occ].astype(int)
    return scores, labels


def _wfn_neg_auc_fast(theta, w_fn, q_s, ds_s, de_s, len_s, rp_s,
                      inv_idx, n_edges, labels):
    """
    Fast minimize objective using pre-sorted arrays and bincount.
    q_s / ds_s / ... are already sorted by edge_id (precomputed once).
    inv_idx maps each walk to its edge group index.
    Skips argsort + np.unique per call — only weight computation + bincount.
    """
    try:
        w     = np.maximum(w_fn(theta, q_s, ds_s, de_s, len_s, rp_s), _FUNC_EPS)
        wsum  = np.bincount(inv_idx, weights=w,       minlength=n_edges)
        wqsum = np.bincount(inv_idx, weights=w * q_s, minlength=n_edges)
        scores = wqsum / np.maximum(wsum, _FUNC_EPS)
        if len(np.unique(labels)) < 2:
            return 0.0
        return -float(roc_auc_score(labels, scores))
    except Exception:
        return 0.0


def _func_registry():
    """
    Registry of parametric weight functions for functional aggregators. Each entry:
    model_name -> (w_fn(theta,q,ds,de,l,rp), theta0, description). w_fn must return
    a positive array of the same length as q.
    """
    E = _FUNC_EPS
    reg = {}

    def r(name, fn, t0, desc):
        reg[name] = (fn, list(t0), desc)

    # ── Confidence-only ─────────────────────────────────────────────
    r("func_conf_power",   lambda t,q,ds,de,l,rp: np.power(np.maximum(q, E), t[0]),
      [1.],    "q^b")
    r("func_conf_exp",     lambda t,q,ds,de,l,rp: np.exp(np.clip(t[0]*q, -30, 30)),
      [2.],    "exp(b*q)")
    r("func_conf_cert",    lambda t,q,ds,de,l,rp: np.power(np.abs(q - .5) + E, t[0]),
      [1.],    "|q-0.5|^b")
    r("func_conf_logit",   lambda t,q,ds,de,l,rp: 1./(1.+np.exp(np.clip(-t[0]*(q-.5), -30, 30))),
      [3.],    "sig(b*(q-0.5))")

    # ── No-param baseline ──────────────────────────────────────────
    r("func_uniform", lambda t,q,ds,de,l,rp: np.ones_like(q),  [], "uniform mean")

    # ── log-probability weight variants ──────────────────────────────
    r("func_logq_power",
      lambda t,q,ds,de,l,rp: np.power(-np.log(np.maximum(q, E)) + E, t[0]),
      [1.],   "(-log(q))^b")
    r("func_logit_power",
      lambda t,q,ds,de,l,rp: np.power(np.abs(np.log(np.maximum(q, E) / np.maximum(1. - q, E))) + E, t[0]),
      [1.],   "|logit(q)|^b")

    # ── Bernoulli-entropy confidence: H(q) = -[q*log(q) + (1-q)*log(1-q)],
    # symmetric under q<->1-q by construction ───────────────────────────────
    LN2 = float(np.log(2.))

    def _bern_ent(q):
        q = np.clip(q, E, 1. - E)
        return -(q * np.log(q) + (1. - q) * np.log(1. - q))

    r("func_entropy_power",
      lambda t, q, ds, de, l, rp: np.power(np.maximum(LN2 - _bern_ent(q), E), t[0]),
      [1.], "(ln2-H(q))^b")
    r("func_entropy_exp",
      lambda t, q, ds, de, l, rp: np.exp(np.clip(-t[0] * _bern_ent(q), -30, 30)),
      [2.], "exp(-a*H(q))")

    # ── Variance-based weighting: Var(Bernoulli(q)) = q(1-q) ─────────────────
    r("func_fisher_power",
      lambda t, q, ds, de, l, rp: np.power(np.maximum(q * (1. - q), E), -t[0]),
      [1.], "(q(1-q))^{-b}  inverse-variance weighting")
    # max(q,1-q) = confidence in the predicted class.
    r("func_maxprob_power",
      lambda t, q, ds, de, l, rp: np.power(np.maximum(np.maximum(q, 1. - q), E), t[0]),
      [1.], "max(q,1-q)^b")

    return reg


def run_aggregator(
    exp_dir: Path, dataset_name: str, run_id: str, epoch: int, models, seed, device=None
):
    if device is None:
        device = torch.device("cpu")
    val_preds = load_prediction(exp_dir, dataset_name, epoch, "val")
    test_preds = load_prediction(exp_dir, dataset_name, epoch, "test")

    y_train = np.asarray(val_preds["targets"]).astype(int)
    y_test = np.asarray(test_preds["targets"]).astype(int)

    unique = np.unique(y_train)
    if len(unique) != 2:
        print(
            "\u26a0 Aggregator currently supports binary labels only. Skipping aggregator stage."
        )
        return

    edge_train = np.asarray(val_preds["edge_ids"]).astype(int)
    edge_test = np.asarray(test_preds["edge_ids"]).astype(int)

    out_dir = exp_dir / "posthoc" / run_id / "aggregator"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Pre-compute arrays shared across all func_* aggregators --
    func_registry = _func_registry()
    q_tr_f   = _pred_prob(val_preds).astype(np.float64)
    q_te_f   = _pred_prob(test_preds).astype(np.float64)
    ds_tr_f  = np.asarray(val_preds["dist_from_start"],  dtype=np.float64)
    ds_te_f  = np.asarray(test_preds["dist_from_start"], dtype=np.float64)
    len_tr_f = np.maximum(np.asarray(val_preds["walk_lengths"],  dtype=np.float64), 1.)
    len_te_f = np.maximum(np.asarray(test_preds["walk_lengths"], dtype=np.float64), 1.)
    de_tr_f  = np.maximum(len_tr_f - ds_tr_f - 1., 0.)
    de_te_f  = np.maximum(len_te_f - ds_te_f - 1., 0.)
    rp_tr_f  = ds_tr_f / len_tr_f
    rp_te_f  = ds_te_f / len_te_f
    # Pre-sort by edge_id once; precompute grouping for fast per-call bincount
    _fsort   = np.argsort(edge_train, kind="stable")
    _, _finv, _fcnts = np.unique(
        edge_train[_fsort], return_inverse=True, return_counts=True)
    _fn_e    = int(_finv.max()) + 1
    _ffo     = np.concatenate([[0], np.cumsum(_fcnts)[:-1]])
    _flabels = y_train[_fsort][_ffo].astype(int)
    _fq_s    = q_tr_f[_fsort];  _fds_s  = ds_tr_f[_fsort]
    _fde_s   = de_tr_f[_fsort]; _flen_s = len_tr_f[_fsort]
    _frp_s   = rp_tr_f[_fsort]

    for model_name in models:
        model_name = model_name.strip().lower()
        if not model_name:
            continue
        if model_name not in func_registry:
            print(f"\u26a0 Unknown aggregator '{model_name}', skipping "
                  f"(must be one of {sorted(func_registry)})")
            continue

        # w_j = f(theta; q_j, ds_j, de_j, len_j, rel_pos_j)
        # theta* = argmax edge-level val AUC via Nelder-Mead. No neural net.
        w_fn, theta0, desc = func_registry[model_name]

        if len(theta0) == 0:
            res_theta = np.array([])
            n_iters   = 0
        else:
            theta0_arr = np.asarray(theta0, dtype=np.float64)
            res = minimize(
                _wfn_neg_auc_fast,
                theta0_arr,
                args=(w_fn, _fq_s, _fds_s, _fde_s, _flen_s, _frp_s,
                      _finv, _fn_e, _flabels),
                method="Nelder-Mead",
                options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4},
            )
            res_theta = res.x
            n_iters   = int(res.nit)

        tr_sc, tr_lb = _wfn_edge_scores(
            w_fn, res_theta, q_tr_f, ds_tr_f, de_tr_f, len_tr_f, rp_tr_f,
            edge_train, y_train)
        te_sc, te_lb = _wfn_edge_scores(
            w_fn, res_theta, q_te_f, ds_te_f, de_te_f, len_te_f, rp_te_f,
            edge_test, y_test)
        edge_tr_auc = roc_auc_score(tr_lb, tr_sc)
        edge_te_auc = roc_auc_score(te_lb, te_sc)

        model_dir = out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "model_config.json", "w") as f:
            json.dump({
                "model_name": model_name,
                "description": desc,
                "theta_star": [float(v) for v in res_theta] if len(res_theta) > 0 else [],
                "n_iters": n_iters,
                "n_params": len(theta0),
                "seed": int(seed),
            }, f, indent=2)
        with open(model_dir / "summary.txt", "w") as f:
            f.write("Post-hoc aggregator summary\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Run ID: {run_id}\n")
            f.write(f"Model: {model_name}  ({desc})\n")
            f.write(f"theta* = {[float(v) for v in res_theta]}\n\n")
            f.write("Edge-level aggregated AUC:\n")
            f.write(f"  Train AUC: {edge_tr_auc:.4f} ({len(tr_sc)} edges)\n")
            f.write(f"  Test  AUC: {edge_te_auc:.4f} ({len(te_sc)} edges)\n")
        print(f"\u2713 Aggregator '{model_name}' ({desc})")
        print(f"  theta*={np.round(res_theta, 4).tolist()}  n_iters={n_iters}")
        print(f"  Edge agg_tr AUC={edge_tr_auc:.4f}  Edge test AUC={edge_te_auc:.4f}")


def main():
    args = parse_args()

    exp_dir = Path(args.exp_dir)
    checkpoint_dir = exp_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    ckpt_path = resolve_checkpoint(
        checkpoint_dir, args.checkpoint_choice, args.checkpoint_path
    )
    epoch = _parse_epoch(ckpt_path)
    if epoch is None:
        epoch = 0

    # Recover the exact cfg used at training time from the checkpoint itself.
    # LitEdgeClassifier.save_hyperparameters() (src/model/lit_model.py) saves the
    # full resolved cfg into every checkpoint -- this is the single source of
    # truth for anything that affects the walk cache or model architecture
    # (dataset.*, model.*). Positional `overrides` are merged on top only for
    # deliberate, explicit changes; they are not required to reproduce the
    # training-time setup, so forgetting one doesn't silently desync the
    # data/model from the checkpoint.
    raw_ckpt = torch.load(str(ckpt_path), map_location="cpu")
    saved_cfg = (raw_ckpt.get("hyper_parameters") or {}).get("cfg")
    if saved_cfg is None:
        print(
            "⚠️  Checkpoint has no saved cfg (pre-save_hyperparameters checkpoint?) "
            "-- falling back to config.yaml + CLI overrides. Every non-default "
            "dataset.*/model.* override used at training time MUST be repeated "
            "manually here or results will silently mismatch the training run."
        )
        cfg = load_config(args.config, overrides=args.overrides)
    else:
        cfg = OmegaConf.merge(
            OmegaConf.create(saved_cfg), OmegaConf.from_dotlist(args.overrides or [])
        )
    validate_config(cfg, context="posthoc")

    cfg.training.checkpoint_dir = str(checkpoint_dir)
    cfg.training.log_dir = str(exp_dir / "logs")

    try:
        seed = get_seed(cfg)
    except ValueError:
        seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if cfg.training.use_cuda and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        print(f"Using CUDA device {args.device}")
        device = torch.device("cuda:0")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    print(f"Using checkpoint: {ckpt_path}")
    print(f"Using epoch index: {epoch}")
    print(f"Using device: {device}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    artifacts = {a.strip() for a in args.artifacts.split(",") if a.strip()}
    agg_models = [m.strip() for m in args.agg_models.split(",") if m.strip()]

    run_id = args.run_id or f"{ckpt_path.stem}_posthoc"

    if "predictions" in artifacts:
        data_module = prepare_data(cfg)

        try:
            model = LitEdgeClassifier.load_from_checkpoint(str(ckpt_path), cfg=cfg)
        except Exception:
            model = LitEdgeClassifier.load_from_checkpoint(str(ckpt_path))
        model = model.to(device)
        model.eval()

        saver = PerEpochPredictionSaver(cfg, data_module)
        fake_trainer = SimpleNamespace(current_epoch=epoch)

        for split in splits:
            if split not in data_module:
                print(f"⚠ Unknown split '{split}', skipping")
                continue
            print(f"Generating predictions for split={split}...")
            pred = saver._extract_predictions(
                fake_trainer, model, data_module[split], split
            )
            saver._save_predictions(pred, epoch, split)

    if "aggregator" in artifacts:
        run_aggregator(exp_dir, cfg.dataset.name, run_id, epoch, agg_models, seed, device)

    print(f"\n✓ Post-hoc pipeline complete. Artifacts: {exp_dir / 'posthoc' / run_id}")


if __name__ == "__main__":
    main()
