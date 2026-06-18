#!/usr/bin/env python3
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

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
        default="train,val,test",
        help="Comma-separated splits to process",
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default="predictions,triplets,heatmaps,aggregator",
        help="Comma-separated artifacts: predictions,triplets,heatmaps,aggregator",
    )
    parser.add_argument(
        "--agg-models",
        type=str,
        default="lgbm",
        help="Comma-separated aggregator models: logistic,xgboost,lgbm",
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


def build_triplets_from_prediction(payload):
    preds = np.asarray(payload["predictions"]).astype(int)
    targets = np.asarray(payload["targets"]).astype(int)
    correct = (preds == targets).astype(int)

    triplets = {
        "edge_id": np.asarray(payload["edge_ids"]).astype(int).tolist(),
        "walk_id": np.asarray(payload["walk_ids"]).astype(int).tolist(),
        "position": np.asarray(payload["positions"]).astype(int).tolist(),
        "walk_len": np.asarray(payload["walk_lengths"]).astype(int).tolist(),
        "dist_from_start": np.asarray(payload["dist_from_start"]).astype(int).tolist(),
        "dist_from_end": np.asarray(payload["dist_from_end"]).astype(int).tolist(),
        "correct": correct.tolist(),
    }
    return triplets


def save_triplets_and_heatmaps(
    exp_dir: Path, dataset_name: str, run_id: str, epoch: int, splits
):
    from scripts.plot_triplet_heatmap import (
        build_heatmap,
        compute_heatmap_stats,
        plot_heatmap,
    )

    out_base = exp_dir / "posthoc" / run_id
    out_base.mkdir(parents=True, exist_ok=True)

    for split in splits:
        payload = load_prediction(exp_dir, dataset_name, epoch, split)
        triplets = build_triplets_from_prediction(payload)

        triplet_path = out_base / f"triplets_{split}.pkl"
        with open(triplet_path, "wb") as f:
            pickle.dump(triplets, f)

        heatmap, counts = build_heatmap(triplets)
        heatmap_png = out_base / f"heatmap_{split}.png"
        plot_heatmap(heatmap, str(heatmap_png))

        np.savez(
            str(out_base / f"heatmap_{split}_data.npz"), grid=heatmap, counts=counts
        )
        stats = compute_heatmap_stats(heatmap, counts, triplets)
        with open(out_base / f"heatmap_{split}_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"✓ Triplets + heatmap saved for {split}: {out_base}")


def _aggregate_edge_probs(edge_ids, probs, y):
    edge_to_probs = {}
    edge_to_labels = {}
    for eid, prob, label in zip(edge_ids, probs, y):
        edge_to_probs.setdefault(int(eid), []).append(float(prob))
        edge_to_labels.setdefault(int(eid), []).append(int(label))

    edge_probs_out = []
    edge_labels_out = []
    for eid, plist in edge_to_probs.items():
        edge_probs_out.append(float(np.mean(plist)))
        labels = np.asarray(edge_to_labels[eid], dtype=int)
        edge_labels_out.append(int(np.bincount(labels).argmax()))

    return np.asarray(edge_probs_out), np.asarray(edge_labels_out)


def _pred_prob(preds):
    probs = np.asarray(preds.get("probabilities", []))
    if probs.ndim == 2 and probs.shape[1] >= 2:
        return probs[:, 1].astype(float)
    return np.asarray(preds["predictions"]).astype(float)


def _make_attention_features(preds):
    """[pred_prob, rel_pos, norm_len] — all normalised to [0, 1], no scaling needed."""
    s = _pred_prob(preds).astype(np.float32)
    p = np.asarray(preds["dist_from_start"], dtype=np.float32)
    l = np.asarray(preds["walk_lengths"], dtype=np.float32)
    l_safe = np.maximum(l, 1.0)
    max_l = float(l_safe.max()) if len(l_safe) > 0 else 1.0
    rel_pos = p / l_safe
    norm_len = l_safe / max(max_l, 1.0)
    return np.stack([s, rel_pos, norm_len], axis=1)  # (N, 3)


def _build_edge_bags(edge_ids, feats, labels):
    """Group occurrence rows into per-edge bags: {edge_id: {feats: ndarray, label: int}}."""
    bags = {}
    for eid, f, y in zip(edge_ids, feats, labels):
        eid = int(eid)
        if eid not in bags:
            bags[eid] = {"feats": [], "label": int(y)}
        bags[eid]["feats"].append(f)
    for bag in bags.values():
        bag["feats"] = np.array(bag["feats"], dtype=np.float32)
    return bags


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


def _wfn_neg_auc(theta, w_fn, q, ds, de, lens, rp, edge_ids, y_occur):
    """Scipy minimize objective: −AUC(edge_labels, edge_weighted_scores)."""
    try:
        scores, labels = _wfn_edge_scores(
            w_fn, theta, q, ds, de, lens, rp, edge_ids, y_occur)
        if len(np.unique(labels)) < 2:
            return 0.0
        return -float(roc_auc_score(labels, scores))
    except Exception:
        return 0.0


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
    Registry of parametric weight functions for functional aggregators.
    Each entry: model_name -> (w_fn(theta,q,ds,de,l,rp), theta0, description)
    w_fn must return a positive array of the same length as q.
    """
    E = _FUNC_EPS
    reg = {}

    def r(name, fn, t0, desc):
        reg[name] = (fn, list(t0), desc)

    # ── Group 1: Length-only ─────────────────────────────────────────────────
    r("func_len_power",    lambda t,q,ds,de,l,rp: np.power(l + E, -t[0]),
      [0.5],   "len^{-a}")
    r("func_len_exp",      lambda t,q,ds,de,l,rp: np.exp(np.clip(-t[0]*l, -30, 30)),
      [0.01],  "exp(-a*len)")
    r("func_len_harmonic", lambda t,q,ds,de,l,rp: 1. / (1. + np.abs(t[0]) * l),
      [0.02],  "1/(1+a*len)")
    r("func_de_power",     lambda t,q,ds,de,l,rp: np.power(de + 1., -t[0]),
      [0.5],   "(de+1)^{-a}")

    # ── Group 2: Position-only ───────────────────────────────────────────────
    r("func_pos_exprel",   lambda t,q,ds,de,l,rp: np.exp(np.clip(-t[0]*rp, -30, 30)),
      [1.],    "exp(-a*rel_pos)")
    r("func_pos_power",    lambda t,q,ds,de,l,rp: np.power(np.maximum(1. - rp, E), t[0]),
      [1.],    "(1-rel_pos)^a")
    r("func_pos_expds",    lambda t,q,ds,de,l,rp: np.exp(np.clip(-t[0]*ds, -30, 30)),
      [0.05],  "exp(-a*ds_raw)")
    r("func_pos_expde",    lambda t,q,ds,de,l,rp: np.exp(np.clip(-t[0]*de, -30, 30)),
      [0.05],  "exp(-a*de_raw)")
    r("func_pos_middle",   lambda t,q,ds,de,l,rp: np.exp(np.clip(-t[0]*np.abs(rp - .5), -30, 30)),
      [1.],    "exp(-a*|rp-0.5|)")

    # ── Group 3: Confidence-only ─────────────────────────────────────────────
    r("func_conf_power",   lambda t,q,ds,de,l,rp: np.power(np.maximum(q, E), t[0]),
      [1.],    "q^b")
    r("func_conf_exp",     lambda t,q,ds,de,l,rp: np.exp(np.clip(t[0]*q, -30, 30)),
      [2.],    "exp(b*q)")
    r("func_conf_cert",    lambda t,q,ds,de,l,rp: np.power(np.abs(q - .5) + E, t[0]),
      [1.],    "|q-0.5|^b")
    r("func_conf_logit",   lambda t,q,ds,de,l,rp: 1./(1.+np.exp(np.clip(-t[0]*(q-.5), -30, 30))),
      [3.],    "sig(b*(q-0.5))")

    # ── Group 4: 2-param combinations ───────────────────────────────────────
    r("func_len_conf",
      lambda t,q,ds,de,l,rp: np.power(l+E,-t[0]) * np.power(np.maximum(q,E), t[1]),
      [.5, 1.], "len^{-a}*q^b")
    r("func_len_pos",
      lambda t,q,ds,de,l,rp: np.power(l+E,-t[0]) * np.exp(np.clip(-t[1]*rp,-30,30)),
      [.5, 1.], "len^{-a}*exp(-g*rp)")
    r("func_pos_conf",
      lambda t,q,ds,de,l,rp: np.exp(np.clip(-t[0]*rp,-30,30)) * np.power(np.maximum(q,E), t[1]),
      [1., 1.], "exp(-a*rp)*q^b")
    r("func_ds_de",
      lambda t,q,ds,de,l,rp: np.power(ds+1.,-t[0]) * np.power(de+1.,-t[1]),
      [.5, .5], "(ds+1)^{-a}*(de+1)^{-b}")
    r("func_len_cert",
      lambda t,q,ds,de,l,rp: np.power(l+E,-t[0]) * np.power(np.abs(q-.5)+E, t[1]),
      [.5, 1.], "len^{-a}*|q-0.5|^b")

    # ── Group 5: Linear-in-log-space (professor's suggestion) ────────────────
    r("func_log3",
      lambda t,q,ds,de,l,rp: np.exp(np.clip(
          t[0]*q + t[1]*rp + t[2]*np.log(l+E), -30, 30)),
      [0., 0., -.5], "exp(b1*q + b2*rp + b3*log_len)")
    r("func_log4",
      lambda t,q,ds,de,l,rp: np.exp(np.clip(
          t[0] + t[1]*q + t[2]*rp + t[3]*np.log(l+E), -30, 30)),
      [0., 0., 0., -.5], "exp(b0 + b1*q + b2*rp + b3*log_len)")
    r("func_log5",
      lambda t,q,ds,de,l,rp: np.exp(np.clip(
          t[0]*q + t[1]*np.log(ds+1.) + t[2]*np.log(de+1.)
          + t[3]*np.log(l+E) + t[4]*q*np.log(l+E), -30, 30)),
      [0., 0., 0., -.5, 0.],
      "exp(b1*q + b2*log_ds + b3*log_de + b4*log_len + b5*q*log_len)")
    r("func_log6",
      lambda t,q,ds,de,l,rp: np.exp(np.clip(
          t[0]*q + t[1]*rp + t[2]*np.log(l+E)
          + t[3]*q*rp + t[4]*q*np.log(l+E) + t[5]*rp*np.log(l+E), -30, 30)),
      [0., 0., -.5, 0., 0., 0.],
      "exp(b1..b6 with pairwise q/rp/log_len interactions)")

    # ── Group 6: Geometric / dual-endpoint ───────────────────────────────────
    r("func_dist_prod",
      lambda t,q,ds,de,l,rp: np.power((ds+1.)*(de+1.) + E, -t[0]),
      [.5],  "((ds+1)(de+1))^{-a}")
    r("func_dist_min",
      lambda t,q,ds,de,l,rp: np.power(np.minimum(ds, de) + 1., -t[0]),
      [.5],  "(min(ds,de)+1)^{-a}")
    r("func_centrality",
      lambda t,q,ds,de,l,rp: np.exp(np.clip(-t[0]*(rp - .5)**2, -30, 30)),
      [2.],  "Gaussian exp(-a*(rp-0.5)^2)")
    r("func_rp_sym",
      lambda t,q,ds,de,l,rp: np.power(rp*(1. - rp) + E, t[0]),
      [.5],  "(rp*(1-rp))^a  mid-peak")

    # ── Group 7: No-param baselines ──────────────────────────────────────────
    r("func_uniform", lambda t,q,ds,de,l,rp: np.ones_like(q),  [], "uniform mean")
    r("func_inv_len", lambda t,q,ds,de,l,rp: 1. / l,           [], "1/len (fixed)")

    # ── Group 8: 1-q (trust-confidence) variants ─────────────────────────────
    # For datasets where class-0 is the "graph positive" (e.g. bitcoin-alpha:
    # class 0 = trust), (1-q) = P(trust) is the natural confidence signal.
    r("func_invq_power",
      lambda t,q,ds,de,l,rp: np.power(np.maximum(1. - q, E), t[0]),
      [1.],   "(1-q)^b")
    r("func_invq_exp",
      lambda t,q,ds,de,l,rp: np.exp(np.clip(t[0] * (1. - q), -30, 30)),
      [2.],   "exp(b*(1-q))")
    r("func_len_invq",
      lambda t,q,ds,de,l,rp: np.power(l + E, -t[0]) * np.power(np.maximum(1. - q, E), t[1]),
      [.5, 1.], "len^{-a}*(1-q)^b")
    r("func_pos_invq",
      lambda t,q,ds,de,l,rp: np.exp(np.clip(-t[0] * rp, -30, 30)) * np.power(np.maximum(1. - q, E), t[1]),
      [1., 1.], "exp(-a*rp)*(1-q)^b")

    # ── Group 9: log-probability weight variants ──────────────────────────────
    # (-log q) = information content of the distrust prediction.
    # (-log(1-q)) = information content of the trust prediction.
    # These weight walks by how "surprising" (confident) the model is.
    r("func_logq_power",
      lambda t,q,ds,de,l,rp: np.power(-np.log(np.maximum(q, E)) + E, t[0]),
      [1.],   "(-log(q))^b")
    r("func_log1mq_power",
      lambda t,q,ds,de,l,rp: np.power(-np.log(np.maximum(1. - q, E)) + E, t[0]),
      [1.],   "(-log(1-q))^b")
    r("func_len_logq",
      lambda t,q,ds,de,l,rp: np.power(l + E, -t[0]) * np.power(-np.log(np.maximum(q, E)) + E, t[1]),
      [.5, 1.], "len^{-a}*(-log(q))^b")
    r("func_logit_power",
      lambda t,q,ds,de,l,rp: np.power(np.abs(np.log(np.maximum(q, E) / np.maximum(1. - q, E))) + E, t[0]),
      [1.],   "|logit(q)|^b")

    # ── Group 10: log-of-q in multi-feature combinations ─────────────────────
    # Natural extensions of func_log3/4 replacing raw q with log(q) or logit(q)
    # as the confidence signal; and two-param |logit(q)| × position/length forms.
    r("func_logq3",
      lambda t,q,ds,de,l,rp: np.exp(np.clip(
          t[0]*np.log(np.maximum(q, E)) + t[1]*rp + t[2]*np.log(l+E), -30, 30)),
      [0., 0., -.5], "exp(b1*log(q) + b2*rp + b3*log_len)  i.e. q^b1*exp(b2*rp)*len^b3")
    r("func_logit3",
      lambda t,q,ds,de,l,rp: np.exp(np.clip(
          t[0]*np.log(np.maximum(q, E) / np.maximum(1. - q, E)) + t[1]*rp + t[2]*np.log(l+E), -30, 30)),
      [0., 0., -.5], "exp(b1*logit(q) + b2*rp + b3*log_len)  signed confidence")
    r("func_log_both",
      lambda t,q,ds,de,l,rp: np.exp(np.clip(
          t[0]*np.log(np.maximum(q, E)) + t[1]*np.log(np.maximum(1. - q, E)) + t[2]*np.log(l+E), -30, 30)),
      [0., 0., -.5], "exp(b1*log(q)+b2*log(1-q)+b3*log_len)  =q^b1*(1-q)^b2*len^b3")
    r("func_logit_len",
      lambda t,q,ds,de,l,rp: np.power(l + E, -t[0]) * np.power(
          np.abs(np.log(np.maximum(q, E) / np.maximum(1. - q, E))) + E, t[1]),
      [.5, 1.], "len^{-a}*|logit(q)|^b")
    r("func_logit_pos",
      lambda t,q,ds,de,l,rp: np.exp(np.clip(-t[0]*rp, -30, 30)) * np.power(
          np.abs(np.log(np.maximum(q, E) / np.maximum(1. - q, E))) + E, t[1]),
      [1., 1.], "exp(-a*rp)*|logit(q)|^b")

    return reg


import math as _math


# ── MIL gate architectures ────────────────────────────────────────────────────

class _AttentionGate(torch.nn.Module):
    """Basic Ilse & Tomczak (2018): tanh(Wx) → softmax → weighted avg of pred_prob."""

    def __init__(self, input_dim=3, hidden_dim=16):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def bag_score(self, padded, mask):
        logits = self(padded).masked_fill(~mask, float("-inf"))
        return (torch.softmax(logits, dim=1) * padded[:, :, 0]).sum(dim=1)


class _GatedAttentionGate(torch.nn.Module):
    """Gated variant (Ilse & Tomczak 2018): tanh(Wx) ⊙ sigmoid(Ux) → softmax → weighted avg."""

    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        self.V = torch.nn.Linear(input_dim, hidden_dim)
        self.U = torch.nn.Linear(input_dim, hidden_dim)
        self.w = torch.nn.Linear(hidden_dim, 1)

    def bag_score(self, padded, mask):
        t = torch.tanh(self.V(padded))     # (B, L, H)
        s = torch.sigmoid(self.U(padded))  # (B, L, H)
        logits = self.w(t * s).squeeze(-1) # (B, L)
        logits = logits.masked_fill(~mask, float("-inf"))
        return (torch.softmax(logits, dim=1) * padded[:, :, 0]).sum(dim=1)


class _SelfAttentionAgg(torch.nn.Module):
    """1-layer MHSA over occurrences → masked mean pool → sigmoid. O(L²) per bag.
    Large bags (L > max_len) are truncated to max_len to avoid OOM."""

    def __init__(self, input_dim=3, d_model=32, n_heads=4, dropout=0.1, max_len=2048):
        super().__init__()
        self.max_len = max_len
        self.proj  = torch.nn.Linear(input_dim, d_model)
        self.norm0 = torch.nn.LayerNorm(d_model)
        self.attn  = torch.nn.MultiheadAttention(d_model, n_heads,
                                                  batch_first=True, dropout=dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.ff    = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 2), torch.nn.GELU(),
            torch.nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.head  = torch.nn.Sequential(torch.nn.Linear(d_model, 1), torch.nn.Sigmoid())

    def bag_score(self, padded, mask):
        if padded.shape[1] > self.max_len:
            padded = padded[:, :self.max_len]
            mask   = mask[:,   :self.max_len]
        x = self.norm0(self.proj(padded))                        # (B, L, d)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=~mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        mf = mask.unsqueeze(-1).float()
        pooled = (x * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)  # (B, d)
        return self.head(pooled).squeeze(-1)                         # (B,)


class _PMAAgg(torch.nn.Module):
    """Set Transformer PMA(1) (Lee et al. 2019): one seed cross-attends to instances. O(L)."""

    def __init__(self, input_dim=3, d_model=32, n_heads=4):
        super().__init__()
        self.proj  = torch.nn.Linear(input_dim, d_model)
        self.norm0 = torch.nn.LayerNorm(d_model)
        self.seed  = torch.nn.Parameter(torch.randn(1, 1, d_model))
        self.cross = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.ff    = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 2), torch.nn.GELU(),
            torch.nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.head  = torch.nn.Sequential(torch.nn.Linear(d_model, 1), torch.nn.Sigmoid())

    def bag_score(self, padded, mask):
        B  = padded.shape[0]
        kv = self.norm0(self.proj(padded))           # (B, L, d)
        q  = self.seed.expand(B, -1, -1)             # (B, 1, d)
        out, _ = self.cross(q, kv, kv, key_padding_mask=~mask)
        out = self.norm1(q + out)                    # (B, 1, d)
        out = self.norm2(out + self.ff(out))
        return self.head(out.squeeze(1)).squeeze(-1) # (B,)


# ── MIL shared infrastructure ─────────────────────────────────────────────────

class _BagDataset(torch.utils.data.Dataset):
    def __init__(self, bags):
        self.eids    = list(bags.keys())
        self.bags    = bags
        self.lengths = [len(bags[e]["feats"]) for e in self.eids]

    def __len__(self):
        return len(self.eids)

    def __getitem__(self, idx):
        e = self.eids[idx]
        b = self.bags[e]
        return torch.tensor(b["feats"]), float(b["label"])


class _BagBucketSampler(torch.utils.data.Sampler):
    """Groups bags by size bucket; pads each mini-batch only to its local max."""

    def __init__(self, lengths, batch_size, bucket_width, shuffle, seed):
        self.bs, self.shuffle, self.seed = batch_size, shuffle, seed
        self._iter = 0
        buckets: dict = {}
        for i, l in enumerate(lengths):
            bid = int(l) // max(1, int(bucket_width))
            buckets.setdefault(bid, []).append(i)
        self._buckets = list(buckets.values())
        self._n = sum(_math.ceil(len(b) / batch_size) for b in self._buckets)

    def __len__(self):
        return self._n

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._iter)
        self._iter += 1
        batches = []
        for b in self._buckets:
            idxs = list(b)
            if self.shuffle:
                rng.shuffle(idxs)
            for s in range(0, len(idxs), self.bs):
                batches.append(idxs[s: s + self.bs])
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def _bag_collate(batch):
    feats_list, labels = zip(*batch)
    max_sz = max(f.shape[0] for f in feats_list)
    n = len(feats_list)
    padded = torch.zeros(n, max_sz, 3)
    mask   = torch.zeros(n, max_sz, dtype=torch.bool)
    for i, f in enumerate(feats_list):
        padded[i, :f.shape[0]] = f
        mask[i,   :f.shape[0]] = True
    return padded, mask, torch.tensor(labels, dtype=torch.float32)


def _bag_stats(bags):
    """Print bag-size distribution; return a data-driven bucket_width."""
    sizes = [len(b["feats"]) for b in bags.values()]
    percs = [0, 25, 50, 75, 90, 95, 99, 100]
    vals  = np.percentile(sizes, percs)
    print("  Bag-size distribution:")
    print("    " + "  ".join(f"p{p}={int(v)}" for p, v in zip(percs, vals)))
    iqr = float(vals[3] - vals[1])   # p75 - p25
    bw  = max(1, int(iqr / 4)) if iqr > 0 else max(1, int(vals[2] / 4))
    print(f"  → bucket_width={bw}  (IQR={iqr:.0f}, n={len(sizes)} bags)")
    return bw


def _eval_mil(gate, bags, device, batch_size=32):
    """Evaluate any MIL gate; returns (scores_np, labels_np)."""
    ds = _BagDataset(bags)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, collate_fn=_bag_collate,
        pin_memory=True, num_workers=0, shuffle=False,
    )
    gate.eval()
    all_sc, all_lb = [], []
    with torch.no_grad():
        for padded, mask, labels_b in dl:
            padded = padded.to(device, non_blocking=True)
            mask   = mask.to(device,   non_blocking=True)
            all_sc.append(gate.bag_score(padded, mask).cpu())
            all_lb.append(labels_b)
    torch.cuda.empty_cache()
    return torch.cat(all_sc).numpy(), torch.cat(all_lb).numpy().astype(int)


def _run_mil_experiment(
    gate, bags_train, bags_test, device, seed, pos_w,
    n_epochs=150, lr=1e-3, wd=1e-4, batch_size=128, bucket_width=None,
):
    """Train any MIL gate with bucket batching; print per-10-epoch loss + AUC curve."""
    import torch.nn.functional as F

    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(seed)

    bw = bucket_width if bucket_width is not None else _bag_stats(bags_train)

    train_ds = _BagDataset(bags_train)
    sampler  = _BagBucketSampler(
        train_ds.lengths, batch_size=batch_size,
        bucket_width=bw, shuffle=True, seed=seed,
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_sampler=sampler, collate_fn=_bag_collate,
        pin_memory=True, num_workers=0,
    )
    opt   = torch.optim.Adam(gate.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)

    n_pos = sum(1 for b in bags_train.values() if b["label"] == 1)
    n_neg = len(bags_train) - n_pos
    print(f"  {n_pos} pos / {n_neg} neg  |  {len(train_dl)} batches/epoch  |  {n_epochs} epochs")

    for ep in range(n_epochs):
        gate.train()
        ep_loss = 0.0
        for padded, mask, labels_b in train_dl:
            padded   = padded.to(device,   non_blocking=True)
            mask     = mask.to(device,     non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)
            wts_b    = torch.tensor(
                [pos_w if l.item() == 1.0 else 1.0 for l in labels_b], device=device
            )
            scores = gate.bag_score(padded, mask)
            loss   = F.binary_cross_entropy(
                scores.clamp(1e-6, 1 - 1e-6), labels_b, weight=wts_b
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()

        if (ep + 1) % 10 == 0:
            val_sc, val_lb = _eval_mil(gate, bags_train, device)
            tst_sc, tst_lb = _eval_mil(gate, bags_test,  device)
            val_auc  = roc_auc_score(val_lb, val_sc)
            test_auc = roc_auc_score(tst_lb, tst_sc)
            print(
                f"    ep {ep+1:3d}  loss={ep_loss/len(train_dl):.4f}"
                f"  agg_tr_AUC={val_auc:.4f}  test_AUC={test_auc:.4f}"
            )

    # Free optimizer + scheduler before final eval to reclaim GPU memory
    import gc
    del opt, sched
    gc.collect()
    torch.cuda.empty_cache()

    val_sc,  val_lb  = _eval_mil(gate, bags_train, device)
    tst_sc,  tst_lb  = _eval_mil(gate, bags_test,  device)
    return gate, roc_auc_score(val_lb, val_sc), roc_auc_score(tst_lb, tst_sc), val_sc, val_lb, tst_sc, tst_lb


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
            "⚠ Aggregator currently supports binary labels only. Skipping aggregator stage."
        )
        return

    x_train = np.stack(
        [
            np.asarray(val_preds["dist_from_start"]).astype(float),
            np.asarray(val_preds["walk_lengths"]).astype(float),
            _pred_prob(val_preds),
        ],
        axis=1,
    )
    x_test = np.stack(
        [
            np.asarray(test_preds["dist_from_start"]).astype(float),
            np.asarray(test_preds["walk_lengths"]).astype(float),
            _pred_prob(test_preds),
        ],
        axis=1,
    )

    edge_train = np.asarray(val_preds["edge_ids"]).astype(int)
    edge_test = np.asarray(test_preds["edge_ids"]).astype(int)

    out_dir = exp_dir / "posthoc" / run_id / "aggregator"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Pre-compute arrays for func_* aggregators once (avoid repeat per model) ──
    _func_has_any = any(m.strip().lower().startswith("func_") for m in models)
    if _func_has_any:
        _FN_REG_CACHE = _func_registry()
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

    # ── Pre-compute lgbm scores for lgbm_func_* aggregators (trained once) ──────
    _lgbm_func_has_any = any(m.strip().lower().startswith("lgbm_func_") for m in models)
    _lgbm_f_model = None
    _lf_scaler    = None
    q_lgbm_tr = q_lgbm_te = None
    if _lgbm_func_has_any:
        if not _func_has_any:
            _FN_REG_CACHE = _func_registry()
        try:
            import lightgbm as _lgb_f
            _lf_scaler = StandardScaler()
            _lf_x_tr = _lf_scaler.fit_transform(x_train)
            _lf_x_te = _lf_scaler.transform(x_test)
            _lgbm_f_model = _lgb_f.LGBMClassifier(
                n_estimators=200, num_leaves=31, learning_rate=0.05,
                class_weight="balanced", random_state=seed, verbose=-1,
            )
            _lgbm_f_model.fit(_lf_x_tr, y_train)
            q_lgbm_tr = _lgbm_f_model.predict_proba(_lf_x_tr)[:, 1].astype(np.float64)
            q_lgbm_te = _lgbm_f_model.predict_proba(_lf_x_te)[:, 1].astype(np.float64)
            # Ensure geometry / sort arrays exist (only computed if _func_has_any was set)
            if not _func_has_any:
                ds_tr_f  = np.asarray(val_preds["dist_from_start"],  dtype=np.float64)
                ds_te_f  = np.asarray(test_preds["dist_from_start"], dtype=np.float64)
                len_tr_f = np.maximum(np.asarray(val_preds["walk_lengths"],  dtype=np.float64), 1.)
                len_te_f = np.maximum(np.asarray(test_preds["walk_lengths"], dtype=np.float64), 1.)
                de_tr_f  = np.maximum(len_tr_f - ds_tr_f - 1., 0.)
                de_te_f  = np.maximum(len_te_f - ds_te_f - 1., 0.)
                rp_tr_f  = ds_tr_f / len_tr_f
                rp_te_f  = ds_te_f / len_te_f
                _fsort   = np.argsort(edge_train, kind="stable")
                _, _finv, _fcnts = np.unique(
                    edge_train[_fsort], return_inverse=True, return_counts=True)
                _fn_e    = int(_finv.max()) + 1
                _ffo     = np.concatenate([[0], np.cumsum(_fcnts)[:-1]])
                _flabels = y_train[_fsort][_ffo].astype(int)
                _fds_s   = ds_tr_f[_fsort]
                _fde_s   = de_tr_f[_fsort]
                _flen_s  = len_tr_f[_fsort]
                _frp_s   = rp_tr_f[_fsort]
            _lfq_s = q_lgbm_tr[_fsort]
            print(f"  [lgbm_func_*] lgbm scored {len(q_lgbm_tr)} walk occurrences "
                  f"(walk AUC={roc_auc_score(y_train, q_lgbm_tr):.4f})")
        except Exception as _lfe:
            print(f"⚠ lgbm_func_* pre-compute failed ({_lfe}); skipping all lgbm_func_ models")
            _lgbm_func_has_any = False

    for model_name in models:
        model_name = model_name.strip().lower()
        if not model_name:
            continue

        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)

        if model_name == "logistic":
            model = LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=seed
            )
            model.fit(x_train_s, y_train)
            train_probs = model.predict_proba(x_train_s)[:, 1]
            test_probs = model.predict_proba(x_test_s)[:, 1]
            model_config = {
                "model_name": "logistic",
                "seed": int(seed),
                "feature_names": ["dist_from_start", "walk_length", "predicted_prob"],
                "scaler": "StandardScaler",
                "params": {
                    "max_iter": 1000,
                    "class_weight": "balanced",
                    "random_state": int(seed),
                },
            }
        elif model_name == "xgboost":
            try:
                import xgboost as xgb
            except Exception:
                print("⚠ xgboost is not installed. Skipping xgboost aggregator.")
                continue

            scale_pos_weight = float(
                np.sum(y_train == 0) / max(1, np.sum(y_train == 1))
            )
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=seed,
                verbosity=0,
            )
            model.fit(x_train_s, y_train)
            train_probs = model.predict_proba(x_train_s)[:, 1]
            test_probs = model.predict_proba(x_test_s)[:, 1]
            model_config = {
                "model_name": "xgboost",
                "seed": int(seed),
                "feature_names": ["dist_from_start", "walk_length", "predicted_prob"],
                "scaler": "StandardScaler",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "learning_rate": 0.1,
                    "scale_pos_weight": float(scale_pos_weight),
                    "random_state": int(seed),
                    "verbosity": 0,
                },
            }
        elif model_name == "lgbm":
            try:
                import lightgbm as lgb
            except Exception:
                print("⚠ lightgbm is not installed. Skipping lgbm aggregator.")
                continue

            model = lgb.LGBMClassifier(
                n_estimators=200,
                num_leaves=31,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=seed,
                verbose=-1,
            )
            model.fit(x_train_s, y_train)
            train_probs = model.predict_proba(x_train_s)[:, 1]
            test_probs = model.predict_proba(x_test_s)[:, 1]
            model_config = {
                "model_name": "lgbm",
                "seed": int(seed),
                "feature_names": ["dist_from_start", "walk_length", "predicted_prob"],
                "scaler": "StandardScaler",
                "params": {
                    "n_estimators": 200,
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "class_weight": "balanced",
                    "random_state": int(seed),
                    "verbose": -1,
                },
            }
        elif model_name == "certainty":
            # Certainty gate: phi_j = |s_j - 0.5|, no training required.
            s_val_c  = _pred_prob(val_preds).astype(np.float32)
            s_test_c = _pred_prob(test_preds).astype(np.float32)

            def _cert_pool(eids, scores, targets):
                edge_to_s, edge_to_y = {}, {}
                for eid_c, sj, yj in zip(eids, scores, targets):
                    eid_c = int(eid_c)
                    edge_to_s.setdefault(eid_c, []).append(float(sj))
                    edge_to_y.setdefault(eid_c, []).append(int(yj))
                probs_out, labels_out = [], []
                for eid_c, sl in edge_to_s.items():
                    arr = np.asarray(sl, dtype=np.float32)
                    phi = np.abs(arr - 0.5)
                    denom = float(phi.sum())
                    score_c = float((phi * arr).sum() / denom) if denom > 1e-9 else float(arr.mean())
                    probs_out.append(score_c)
                    yl = np.asarray(edge_to_y[eid_c], dtype=int)
                    labels_out.append(int(np.bincount(yl).argmax()))
                return np.asarray(probs_out, dtype=np.float32), np.asarray(labels_out, dtype=int)

            ep_cert_tr, el_cert_tr = _cert_pool(edge_train, s_val_c,  y_train)
            ep_cert_te, el_cert_te = _cert_pool(edge_test,  s_test_c, y_test)
            edge_tr_auc_c = roc_auc_score(el_cert_tr, ep_cert_tr)
            edge_te_auc_c = roc_auc_score(el_cert_te, ep_cert_te)

            model_dir = out_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(model_dir / "model_config.json", "w") as f:
                json.dump({"model_name": "certainty", "formula": "phi_j = |s_j - 0.5|, weighted mean"}, f, indent=2)
            with open(model_dir / "summary.txt", "w") as f:
                f.write("Post-hoc aggregator summary\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Run ID: {run_id}\n")
                f.write("Model: certainty (phi_j = |s_j - 0.5|, no training)\n\n")
                f.write("Edge-level aggregated AUC:\n")
                f.write(f"  Train AUC: {edge_tr_auc_c:.4f} ({len(ep_cert_tr)} edges)\n")
                f.write(f"  Test  AUC: {edge_te_auc_c:.4f} ({len(ep_cert_te)} edges)\n")
            print(f"✓ Aggregator 'certainty' complete: {model_dir}")
            print(f"  Edge train AUC={edge_tr_auc_c:.4f}  Edge test AUC={edge_te_auc_c:.4f}")
            continue

        elif model_name in ("attention", "self_attention", "set_attention"):
            feats_val_m  = _make_attention_features(val_preds)
            feats_test_m = _make_attention_features(test_preds)
            bags_train_m = _build_edge_bags(edge_train, feats_val_m,  y_train)
            bags_test_m  = _build_edge_bags(edge_test,  feats_test_m, y_test)
            n_pos_m = sum(1 for b in bags_train_m.values() if b["label"] == 1)
            n_neg_m = len(bags_train_m) - n_pos_m
            pos_w_m = float(n_neg_m) / max(n_pos_m, 1)

            if model_name == "attention":
                gate_m = _GatedAttentionGate(hidden_dim=32).to(device)
                arch   = "GatedAttention(3→32, tanh⊙sigmoid→softmax pool)"
                bs_m   = 128
            elif model_name == "self_attention":
                gate_m = _SelfAttentionAgg(d_model=32, n_heads=4).to(device)
                arch   = "SelfAttention(3→32, 1L MHSA, mean pool)"
                bs_m   = 32   # O(L²) attention — keep batches small
            else:
                gate_m = _PMAAgg(d_model=32, n_heads=4).to(device)
                arch   = "SetTransformerPMA(3→32, cross-attn seed)"
                bs_m   = 64   # O(L) cross-attn — moderate

            torch.cuda.empty_cache()  # flush previous model's cache
            print(f"  [{model_name}] {arch}")
            gate_m, edge_tr_auc_m, edge_te_auc_m, tr_sc, tr_lb, te_sc, te_lb = \
                _run_mil_experiment(
                    gate_m, bags_train_m, bags_test_m, device, seed, pos_w_m,
                    n_epochs=150, lr=1e-3, wd=1e-4, batch_size=bs_m,
                )

            model_dir = out_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(gate_m.cpu().state_dict(), model_dir / "gate.pt")
            del gate_m
            torch.cuda.empty_cache()
            with open(model_dir / "model_config.json", "w") as f:
                json.dump({
                    "model_name": model_name, "architecture": arch,
                    "n_epochs": 150, "lr": 1e-3, "weight_decay": 1e-4,
                    "pos_weight": pos_w_m, "seed": int(seed),
                }, f, indent=2)
            with open(model_dir / "summary.txt", "w") as f:
                f.write("Post-hoc aggregator summary\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Run ID: {run_id}\n")
                f.write(f"Model: {model_name} ({arch})\n\n")
                f.write("Edge-level aggregated AUC:\n")
                f.write(f"  Train AUC: {edge_tr_auc_m:.4f} ({len(tr_sc)} edges)\n")
                f.write(f"  Test  AUC: {edge_te_auc_m:.4f} ({len(te_sc)} edges)\n")
            print(f"✓ Aggregator '{model_name}' complete: {model_dir}")
            print(f"  Edge agg_tr AUC={edge_tr_auc_m:.4f}  Edge test AUC={edge_te_auc_m:.4f}")
            continue

        elif model_name in ("lgbm_max", "lgbm_lse"):
            # lgbm per-occurrence scores pooled by max or learnable LogSumExp (β).
            # lgbm_max: score_i = max_j q_j       → hard-picks the most confident walk
            # lgbm_lse: score_i = LSE_β(q_j)      → smooth interpolation max↔mean
            #           β=0 ≡ mean, β→∞ ≡ max. β is a trained scalar (≥0).
            try:
                import lightgbm as lgb
            except Exception:
                print(f"⚠ lightgbm not installed. Skipping {model_name}.")
                continue

            scale_pos_pool = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
            lgbm_pool = lgb.LGBMClassifier(
                n_estimators=200, num_leaves=31, learning_rate=0.05,
                scale_pos_weight=scale_pos_pool, random_state=seed, verbose=-1,
            )
            lgbm_pool.fit(x_train_s, y_train)
            q_tr_pool = lgbm_pool.predict_proba(x_train_s)[:, 1].astype(np.float32)
            q_te_pool = lgbm_pool.predict_proba(x_test_s)[:, 1].astype(np.float32)

            def _pool_edge(q_arr, edge_ids_arr, y_arr, beta=None):
                """Group q by edge, apply max or LSE-β pool → (edge_scores, edge_labels)."""
                from collections import defaultdict
                buckets = defaultdict(list)
                labels_map = {}
                for eid, qv, lv in zip(edge_ids_arr, q_arr, y_arr):
                    buckets[int(eid)].append(float(qv))
                    labels_map[int(eid)] = int(lv)
                scores, labels = [], []
                for eid, qlist in buckets.items():
                    q_e = np.asarray(qlist, dtype=np.float64)
                    if beta is None:
                        s = float(q_e.max())
                    else:
                        # numerically stable LSE: (1/β)*log(mean(exp(β*(q-max)))) + max
                        b = float(beta)
                        q_shift = q_e - q_e.max()
                        s = float(q_e.max()) + (1.0 / (b + 1e-8)) * float(
                            np.log(np.mean(np.exp(b * q_shift) + 1e-12))
                        )
                    scores.append(s)
                    labels.append(labels_map[eid])
                return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=int)

            if model_name == "lgbm_max":
                tr_sc_pool, tr_lb_pool = _pool_edge(q_tr_pool, edge_train, y_train)
                te_sc_pool, te_lb_pool = _pool_edge(q_te_pool, edge_test,  y_test)
                arch_pool = "lgbm(3 feats)→q_j, max pool per edge"
                beta_saved = None
            else:
                # learn β via AUC-maximising grid search (β is a single scalar)
                from collections import defaultdict as _dd
                tr_edge_bags, te_edge_bags = _dd(list), _dd(list)
                tr_edge_labels, te_edge_labels = {}, {}
                for eid, qv, lv in zip(edge_train, q_tr_pool, y_train):
                    tr_edge_bags[int(eid)].append(float(qv))
                    tr_edge_labels[int(eid)] = int(lv)
                for eid, qv, lv in zip(edge_test, q_te_pool, y_test):
                    te_edge_bags[int(eid)].append(float(qv))
                    te_edge_labels[int(eid)] = int(lv)

                tr_eids = sorted(tr_edge_bags)
                tr_y_edge_arr = np.asarray([tr_edge_labels[e] for e in tr_eids], dtype=np.float32)

                # Pad edge bags to matrix (n_edges × max_bag_len); mask=True where padded
                max_bag_len = max(len(tr_edge_bags[e]) for e in tr_eids)
                Q_mat = np.zeros((len(tr_eids), max_bag_len), dtype=np.float64)
                mask  = np.zeros((len(tr_eids), max_bag_len), dtype=bool)
                for i, e in enumerate(tr_eids):
                    n = len(tr_edge_bags[e])
                    Q_mat[i, :n] = tr_edge_bags[e]
                    mask[i, :n]  = True
                n_per_edge = mask.sum(axis=1).astype(np.float64)

                def _lse_scores_np(beta_val, Q, msk, n_e):
                    """Vectorized LSE pool: (n_edges,) → edge scores."""
                    b = float(max(beta_val, 1e-6))
                    q_max = np.where(msk, Q, -np.inf).max(axis=1, keepdims=True)
                    shifted = np.where(msk, Q - q_max, 0.0)
                    sumexp = np.where(msk, np.exp(b * shifted), 0.0).sum(axis=1)
                    return q_max.ravel() + (1.0 / b) * np.log(sumexp / n_e + 1e-15)

                # Grid search over β ∈ [0.01 … 200] — find β maximising train AUC
                beta_grid = np.concatenate([
                    np.linspace(0.01, 1.0, 40),
                    np.linspace(1.0, 10.0, 30),
                    np.linspace(10.0, 200.0, 20),
                ])
                best_beta, best_auc_lse = 1.0, 0.0
                for _b in beta_grid:
                    sc_b = _lse_scores_np(_b, Q_mat, mask, n_per_edge)
                    a = roc_auc_score(tr_y_edge_arr, sc_b)
                    if a > best_auc_lse:
                        best_auc_lse, best_beta = a, float(_b)

                beta_saved = best_beta
                print(f"  [lgbm_lse] learned β={beta_saved:.3f}  (train AUC at β={beta_saved:.3f}: {best_auc_lse:.4f})")
                tr_sc_pool, tr_lb_pool = _pool_edge(q_tr_pool, edge_train, y_train, beta=beta_saved)
                te_sc_pool, te_lb_pool = _pool_edge(q_te_pool, edge_test,  y_test,  beta=beta_saved)
                arch_pool = f"lgbm(3 feats)→q_j, LSE(β={beta_saved:.3f}) pool per edge"

            edge_tr_auc_pool = roc_auc_score(tr_lb_pool, tr_sc_pool)
            edge_te_auc_pool = roc_auc_score(te_lb_pool, te_sc_pool)

            model_dir = out_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(model_dir / "lgbm.pkl", "wb") as f:
                pickle.dump({"model": lgbm_pool, "scaler": scaler, "beta": beta_saved}, f)
            with open(model_dir / "model_config.json", "w") as f:
                json.dump({
                    "model_name": model_name, "architecture": arch_pool,
                    "beta": beta_saved, "seed": int(seed),
                    "lgbm": {"n_estimators": 200, "num_leaves": 31, "lr": 0.05},
                }, f, indent=2)
            with open(model_dir / "summary.txt", "w") as f:
                f.write("Post-hoc aggregator summary\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Run ID: {run_id}\n")
                f.write(f"Model: {model_name} ({arch_pool})\n\n")
                f.write("Edge-level aggregated AUC:\n")
                f.write(f"  Train AUC: {edge_tr_auc_pool:.4f} ({len(tr_sc_pool)} edges)\n")
                f.write(f"  Test  AUC: {edge_te_auc_pool:.4f} ({len(te_sc_pool)} edges)\n")
            print(f"✓ Aggregator '{model_name}' complete: {model_dir}")
            print(f"  Edge agg_tr AUC={edge_tr_auc_pool:.4f}  Edge test AUC={edge_te_auc_pool:.4f}")
            continue

        elif model_name == "lgbm_stats":
            # Two-level lgbm:
            #   1. lgbm1 scores per occurrence q_j = f(dist, len, pred_prob)
            #   2. For each edge, compute statistics over {q_j} → edge-level feature vector
            #   3. lgbm2 classifies edge directly from those statistics
            # This keeps everything tabular — no neural components.
            try:
                import lightgbm as lgb
            except Exception:
                print("⚠ lightgbm not installed. Skipping lgbm_stats.")
                continue

            scale_pos_s1 = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
            lgbm_s1 = lgb.LGBMClassifier(
                n_estimators=200, num_leaves=31, learning_rate=0.05,
                scale_pos_weight=scale_pos_s1, random_state=seed, verbose=-1,
            )
            lgbm_s1.fit(x_train_s, y_train)
            q_tr_s = lgbm_s1.predict_proba(x_train_s)[:, 1].astype(np.float32)
            q_te_s = lgbm_s1.predict_proba(x_test_s)[:, 1].astype(np.float32)

            # Also need rel_pos for position-weighted features
            p_tr_s = np.asarray(val_preds["dist_from_start"], dtype=np.float32)
            l_tr_s = np.maximum(np.asarray(val_preds["walk_lengths"], dtype=np.float32), 1.0)
            p_te_s = np.asarray(test_preds["dist_from_start"], dtype=np.float32)
            l_te_s = np.maximum(np.asarray(test_preds["walk_lengths"], dtype=np.float32), 1.0)
            rp_tr_s = p_tr_s / l_tr_s   # rel_pos in [0,1]
            rp_te_s = p_te_s / l_te_s

            def _edge_stat_feats(q_arr, rp_arr, edge_ids_arr, y_arr, k_list=(5, 10)):
                """Compute per-edge statistics from per-occurrence q_j and rel_pos."""
                from collections import defaultdict
                buckets_q  = defaultdict(list)
                buckets_rp = defaultdict(list)
                labels_map = {}
                for eid, qv, rpv, lv in zip(edge_ids_arr, q_arr, rp_arr, y_arr):
                    buckets_q[int(eid)].append(float(qv))
                    buckets_rp[int(eid)].append(float(rpv))
                    labels_map[int(eid)] = int(lv)

                rows, lbls = [], []
                for eid in sorted(buckets_q):
                    q_e  = np.asarray(buckets_q[eid],  dtype=np.float64)
                    rp_e = np.asarray(buckets_rp[eid], dtype=np.float64)
                    q_sort = np.sort(q_e)[::-1]   # descending
                    n = len(q_e)
                    feats = [
                        float(q_e.max()),                        # max score
                        float(q_e.mean()),                       # mean score
                        float(q_e.std()) if n > 1 else 0.0,     # std score
                        float(np.median(q_e)),                   # median score
                        float(q_e.min()),                        # min score
                    ]
                    for k in k_list:
                        top_k = q_sort[:k]
                        feats.append(float(top_k.mean()) if len(top_k) > 0 else float(q_e.mean()))
                    # fraction of occurrences with q > 0.5
                    feats.append(float(np.mean(q_e > 0.5)))
                    # q-weighted mean rel_pos
                    w_sum = q_e.sum()
                    feats.append(float((q_e * rp_e).sum() / (w_sum + 1e-8)))
                    # min rel_pos (earliest occurrence)
                    feats.append(float(rp_e.min()))
                    # log(count)
                    feats.append(float(np.log1p(n)))
                    rows.append(feats)
                    lbls.append(labels_map[eid])
                return np.asarray(rows, dtype=np.float32), np.asarray(lbls, dtype=int)

            X_tr_edge, y_tr_edge = _edge_stat_feats(q_tr_s, rp_tr_s, edge_train, y_train)
            X_te_edge, y_te_edge = _edge_stat_feats(q_te_s, rp_te_s, edge_test,  y_test)

            scale_pos_s2 = float(np.sum(y_tr_edge == 0) / max(1, np.sum(y_tr_edge == 1)))
            feat_names = [
                "q_max", "q_mean", "q_std", "q_median", "q_min",
                "q_top5_mean", "q_top10_mean",
                "frac_pos", "q_wtd_relpos", "min_relpos", "log_count"
            ]
            lgbm_s2 = lgb.LGBMClassifier(
                n_estimators=300, num_leaves=31, learning_rate=0.03,
                scale_pos_weight=scale_pos_s2, random_state=seed, verbose=-1,
                min_child_samples=5,
            )
            lgbm_s2.fit(X_tr_edge, y_tr_edge, feature_name=feat_names)
            tr_sc_s = lgbm_s2.predict_proba(X_tr_edge)[:, 1]
            te_sc_s = lgbm_s2.predict_proba(X_te_edge)[:, 1]
            edge_tr_auc_s = roc_auc_score(y_tr_edge, tr_sc_s)
            edge_te_auc_s = roc_auc_score(y_te_edge, te_sc_s)

            arch_s = "lgbm1(3 occ-feats)→q_j, edge-stats(11 feats)→lgbm2"
            model_dir = out_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(model_dir / "lgbm.pkl", "wb") as f:
                pickle.dump({"lgbm1": lgbm_s1, "lgbm2": lgbm_s2, "scaler": scaler}, f)
            with open(model_dir / "model_config.json", "w") as f:
                json.dump({
                    "model_name": "lgbm_stats", "architecture": arch_s,
                    "seed": int(seed),
                    "lgbm1": {"n_estimators": 200, "num_leaves": 31, "lr": 0.05},
                    "lgbm2": {"n_estimators": 300, "num_leaves": 31, "lr": 0.03},
                    "edge_features": feat_names,
                }, f, indent=2)
            with open(model_dir / "summary.txt", "w") as f:
                f.write("Post-hoc aggregator summary\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Run ID: {run_id}\n")
                f.write(f"Model: lgbm_stats ({arch_s})\n\n")
                f.write("Edge-level aggregated AUC:\n")
                f.write(f"  Train AUC: {edge_tr_auc_s:.4f} ({len(tr_sc_s)} edges)\n")
                f.write(f"  Test  AUC: {edge_te_auc_s:.4f} ({len(te_sc_s)} edges)\n")
            print(f"✓ Aggregator 'lgbm_stats' complete: {model_dir}")
            print(f"  Edge agg_tr AUC={edge_tr_auc_s:.4f}  Edge test AUC={edge_te_auc_s:.4f}")
            continue

        elif model_name == "lgbm_attention":
            # Two-stage cascade:
            #   1. lgbm refines raw (dist, len, pred_prob) → per-occurrence score q_j
            #   2. gated attention weights q_j values within each edge bag → edge score
            # lgbm handles feature interaction; attention handles occurrence weighting.
            try:
                import lightgbm as lgb
            except Exception:
                print("⚠ lightgbm is not installed. Skipping lgbm_attention aggregator.")
                continue

            scale_pos_la = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
            lgbm_la = lgb.LGBMClassifier(
                n_estimators=200, num_leaves=31, learning_rate=0.05,
                scale_pos_weight=scale_pos_la, random_state=seed, verbose=-1,
            )
            lgbm_la.fit(x_train_s, y_train)
            q_tr = lgbm_la.predict_proba(x_train_s)[:, 1].astype(np.float32)
            q_te = lgbm_la.predict_proba(x_test_s)[:, 1].astype(np.float32)

            # Build attention features with lgbm-refined score as primary signal
            p_tr_la = np.asarray(val_preds["dist_from_start"],  dtype=np.float32)
            l_tr_la = np.maximum(np.asarray(val_preds["walk_lengths"], dtype=np.float32), 1.0)
            p_te_la = np.asarray(test_preds["dist_from_start"], dtype=np.float32)
            l_te_la = np.maximum(np.asarray(test_preds["walk_lengths"], dtype=np.float32), 1.0)
            max_l_tr_la = float(l_tr_la.max())
            max_l_te_la = float(l_te_la.max())

            feats_tr_la = np.stack(
                [q_tr, p_tr_la / l_tr_la, l_tr_la / max(max_l_tr_la, 1.)], axis=1
            )
            feats_te_la = np.stack(
                [q_te, p_te_la / l_te_la, l_te_la / max(max_l_te_la, 1.)], axis=1
            )
            bags_tr_la = _build_edge_bags(edge_train, feats_tr_la, y_train)
            bags_te_la = _build_edge_bags(edge_test,  feats_te_la, y_test)

            n_pos_la = sum(1 for b in bags_tr_la.values() if b["label"] == 1)
            n_neg_la = len(bags_tr_la) - n_pos_la
            pos_w_la = float(n_neg_la) / max(n_pos_la, 1)
            arch_la  = "lgbm(3 feats)→q_j, GatedAttention weights q_j per edge"

            print(f"  [lgbm_attention] {arch_la}")
            gate_la = _GatedAttentionGate(hidden_dim=32).to(device)
            torch.cuda.empty_cache()
            gate_la, edge_tr_auc_la, edge_te_auc_la, tr_sc_la, tr_lb_la, te_sc_la, te_lb_la = \
                _run_mil_experiment(
                    gate_la, bags_tr_la, bags_te_la, device, seed, pos_w_la,
                    n_epochs=150, lr=1e-3, wd=1e-4, batch_size=128,
                )

            model_dir = out_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(gate_la.cpu().state_dict(), model_dir / "gate.pt")
            with open(model_dir / "lgbm.pkl", "wb") as f:
                pickle.dump({"model": lgbm_la, "scaler": scaler}, f)
            del gate_la
            torch.cuda.empty_cache()
            with open(model_dir / "model_config.json", "w") as f:
                json.dump({
                    "model_name": "lgbm_attention", "architecture": arch_la,
                    "n_epochs": 150, "lr": 1e-3, "weight_decay": 1e-4,
                    "pos_weight": pos_w_la, "seed": int(seed),
                    "lgbm": {"n_estimators": 200, "num_leaves": 31, "lr": 0.05},
                }, f, indent=2)
            with open(model_dir / "summary.txt", "w") as f:
                f.write("Post-hoc aggregator summary\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Run ID: {run_id}\n")
                f.write(f"Model: lgbm_attention ({arch_la})\n\n")
                f.write("Edge-level aggregated AUC:\n")
                f.write(f"  Train AUC: {edge_tr_auc_la:.4f} ({len(tr_sc_la)} edges)\n")
                f.write(f"  Test  AUC: {edge_te_auc_la:.4f} ({len(te_sc_la)} edges)\n")
            print(f"✓ Aggregator 'lgbm_attention' complete: {model_dir}")
            print(f"  Edge agg_tr AUC={edge_tr_auc_la:.4f}  Edge test AUC={edge_te_auc_la:.4f}")
            continue

        elif model_name in ("lgbm_self_attention", "lgbm_set_attention"):
            # Same two-stage cascade as lgbm_attention but with a transformer-style
            # aggregator in stage 2: self-attention (MHSA) or PMA (set transformer seed).
            try:
                import lightgbm as lgb
            except Exception:
                print(f"⚠ lightgbm is not installed. Skipping {model_name} aggregator.")
                continue

            scale_pos_lx = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
            lgbm_lx = lgb.LGBMClassifier(
                n_estimators=200, num_leaves=31, learning_rate=0.05,
                scale_pos_weight=scale_pos_lx, random_state=seed, verbose=-1,
            )
            lgbm_lx.fit(x_train_s, y_train)
            q_tr_lx = lgbm_lx.predict_proba(x_train_s)[:, 1].astype(np.float32)
            q_te_lx = lgbm_lx.predict_proba(x_test_s)[:, 1].astype(np.float32)

            p_tr_lx = np.asarray(val_preds["dist_from_start"],  dtype=np.float32)
            l_tr_lx = np.maximum(np.asarray(val_preds["walk_lengths"], dtype=np.float32), 1.0)
            p_te_lx = np.asarray(test_preds["dist_from_start"], dtype=np.float32)
            l_te_lx = np.maximum(np.asarray(test_preds["walk_lengths"], dtype=np.float32), 1.0)

            feats_tr_lx = np.stack(
                [q_tr_lx, p_tr_lx / l_tr_lx, l_tr_lx / max(float(l_tr_lx.max()), 1.)], axis=1
            )
            feats_te_lx = np.stack(
                [q_te_lx, p_te_lx / l_te_lx, l_te_lx / max(float(l_te_lx.max()), 1.)], axis=1
            )
            bags_tr_lx = _build_edge_bags(edge_train, feats_tr_lx, y_train)
            bags_te_lx = _build_edge_bags(edge_test,  feats_te_lx, y_test)

            n_pos_lx = sum(1 for b in bags_tr_lx.values() if b["label"] == 1)
            n_neg_lx = len(bags_tr_lx) - n_pos_lx
            pos_w_lx  = float(n_neg_lx) / max(n_pos_lx, 1)

            if model_name == "lgbm_self_attention":
                gate_lx  = _SelfAttentionAgg(d_model=32, n_heads=4).to(device)
                bs_lx    = 32
                arch_lx  = "lgbm(3 feats)→q_j, SelfAttention(d=32,h=4) mean pool"
            else:
                gate_lx  = _PMAAgg(d_model=32, n_heads=4).to(device)
                bs_lx    = 64
                arch_lx  = "lgbm(3 feats)→q_j, PMA(d=32,h=4) seed pool"

            print(f"  [{model_name}] {arch_lx}")
            torch.cuda.empty_cache()
            gate_lx, edge_tr_auc_lx, edge_te_auc_lx, tr_sc_lx, tr_lb_lx, te_sc_lx, te_lb_lx = \
                _run_mil_experiment(
                    gate_lx, bags_tr_lx, bags_te_lx, device, seed, pos_w_lx,
                    n_epochs=150, lr=1e-3, wd=1e-4, batch_size=bs_lx,
                )

            model_dir = out_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(gate_lx.cpu().state_dict(), model_dir / "gate.pt")
            with open(model_dir / "lgbm.pkl", "wb") as f:
                pickle.dump({"model": lgbm_lx, "scaler": scaler}, f)
            del gate_lx
            torch.cuda.empty_cache()
            with open(model_dir / "model_config.json", "w") as f:
                json.dump({
                    "model_name": model_name, "architecture": arch_lx,
                    "n_epochs": 150, "lr": 1e-3, "weight_decay": 1e-4,
                    "pos_weight": pos_w_lx, "seed": int(seed),
                    "lgbm": {"n_estimators": 200, "num_leaves": 31, "lr": 0.05},
                }, f, indent=2)
            with open(model_dir / "summary.txt", "w") as f:
                f.write("Post-hoc aggregator summary\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Run ID: {run_id}\n")
                f.write(f"Model: {model_name} ({arch_lx})\n\n")
                f.write("Edge-level aggregated AUC:\n")
                f.write(f"  Train AUC: {edge_tr_auc_lx:.4f} ({len(tr_sc_lx)} edges)\n")
                f.write(f"  Test  AUC: {edge_te_auc_lx:.4f} ({len(te_sc_lx)} edges)\n")
            print(f"✓ Aggregator '{model_name}' complete: {model_dir}")
            print(f"  Edge agg_tr AUC={edge_tr_auc_lx:.4f}  Edge test AUC={edge_te_auc_lx:.4f}")
            continue

        elif model_name.startswith("lgbm_func_"):
            # ── lgbm-scored q_j fed into any func_ weight form ────────────────
            # Stage 1: lgbm(dist, len, pred_prob) → q_j^lgbm  (trained once above)
            # Stage 2: w_j = f(theta; q_j^lgbm, ...)  θ* via Nelder-Mead on val AUC
            if not _lgbm_func_has_any or q_lgbm_tr is None:
                print(f"⚠ lgbm_func_* unavailable; skipping {model_name}")
                continue
            fn_key = "func_" + model_name[len("lgbm_func_"):]
            if fn_key not in _FN_REG_CACHE:
                print(f"⚠ Unknown functional form '{fn_key}' for '{model_name}', skipping")
                continue

            from scipy.optimize import minimize as _sp_minimize
            import matplotlib as _mpl_lf
            _mpl_lf.use("Agg")
            import matplotlib.pyplot as _plt_lf

            w_fn_lf, theta0_lf, desc_lf = _FN_REG_CACHE[fn_key]

            if len(theta0_lf) == 0:
                res_theta_lf = np.array([])
                n_iters_lf   = 0
            else:
                theta0_arr_lf = np.asarray(theta0_lf, dtype=np.float64)
                res_lf = _sp_minimize(
                    _wfn_neg_auc_fast,
                    theta0_arr_lf,
                    args=(w_fn_lf, _lfq_s, _fds_s, _fde_s, _flen_s, _frp_s,
                          _finv, _fn_e, _flabels),
                    method="Nelder-Mead",
                    options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4},
                )
                res_theta_lf = res_lf.x
                n_iters_lf   = int(res_lf.nit)

            tr_sc_lf, tr_lb_lf = _wfn_edge_scores(
                w_fn_lf, res_theta_lf, q_lgbm_tr,
                ds_tr_f, de_tr_f, len_tr_f, rp_tr_f, edge_train, y_train)
            te_sc_lf, te_lb_lf = _wfn_edge_scores(
                w_fn_lf, res_theta_lf, q_lgbm_te,
                ds_te_f, de_te_f, len_te_f, rp_te_f, edge_test, y_test)
            edge_tr_auc_lf = roc_auc_score(tr_lb_lf, tr_sc_lf)
            edge_te_auc_lf = roc_auc_score(te_lb_lf, te_sc_lf)

            model_dir = out_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(model_dir / "lgbm.pkl", "wb") as f:
                pickle.dump({"model": _lgbm_f_model, "scaler": _lf_scaler}, f)
            with open(model_dir / "model_config.json", "w") as f:
                json.dump({
                    "model_name": model_name,
                    "func_key": fn_key,
                    "description": f"lgbm\u2192q_j then {desc_lf}",
                    "theta_star": [float(v) for v in res_theta_lf] if len(res_theta_lf) > 0 else [],
                    "n_iters": n_iters_lf,
                    "n_params": len(theta0_lf),
                    "seed": int(seed),
                    "lgbm": {"n_estimators": 200, "num_leaves": 31,
                             "learning_rate": 0.05, "class_weight": "balanced"},
                }, f, indent=2)
            with open(model_dir / "summary.txt", "w") as f:
                f.write("Post-hoc aggregator summary\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Run ID: {run_id}\n")
                f.write(f"Model: {model_name}  (lgbm\u2192q_j then {desc_lf})\n")
                f.write(f"theta* = {[float(v) for v in res_theta_lf]}\n\n")
                f.write("Edge-level aggregated AUC:\n")
                f.write(f"  Train AUC: {edge_tr_auc_lf:.4f} ({len(tr_sc_lf)} edges)\n")
                f.write(f"  Test  AUC: {edge_te_auc_lf:.4f} ({len(te_sc_lf)} edges)\n")
            print(f"\u2713 Aggregator '{model_name}' (lgbm\u2192q_j then {desc_lf})")
            print(f"  theta*={np.round(res_theta_lf, 4).tolist()}  n_iters={n_iters_lf}")
            print(f"  Edge agg_tr AUC={edge_tr_auc_lf:.4f}  Edge test AUC={edge_te_auc_lf:.4f}")
            continue

        elif model_name.startswith("func_"):
            # ── Parametric functional weight aggregators ──────────────────────
            # w_j = f(theta; q_j, ds_j, de_j, len_j, rel_pos_j)
            # theta* = argmax edge-level val AUC via Nelder-Mead. No neural net.
            from scipy.optimize import minimize as _sp_minimize
            import matplotlib as _mpl_f
            _mpl_f.use("Agg")
            import matplotlib.pyplot as _plt_f

            if model_name not in _FN_REG_CACHE:
                print(f"⚠ Unknown functional aggregator '{model_name}', skipping")
                continue

            w_fn_f, theta0_f, desc_f = _FN_REG_CACHE[model_name]

            if len(theta0_f) == 0:
                res_theta = np.array([])
                n_iters   = 0
            else:
                # Optimise on full val data via pre-sorted arrays + bincount
                theta0_arr = np.asarray(theta0_f, dtype=np.float64)
                res = _sp_minimize(
                    _wfn_neg_auc_fast,
                    theta0_arr,
                    args=(w_fn_f, _fq_s, _fds_s, _fde_s, _flen_s, _frp_s,
                          _finv, _fn_e, _flabels),
                    method="Nelder-Mead",
                    options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4},
                )
                res_theta = res.x
                n_iters   = int(res.nit)

            tr_sc_f, tr_lb_f = _wfn_edge_scores(
                w_fn_f, res_theta, q_tr_f, ds_tr_f, de_tr_f, len_tr_f, rp_tr_f,
                edge_train, y_train)
            te_sc_f, te_lb_f = _wfn_edge_scores(
                w_fn_f, res_theta, q_te_f, ds_te_f, de_te_f, len_te_f, rp_te_f,
                edge_test, y_test)
            edge_tr_auc_f = roc_auc_score(tr_lb_f, tr_sc_f)
            edge_te_auc_f = roc_auc_score(te_lb_f, te_sc_f)

            # ── Weight surface visualisation ───────────────────────────────────
            if len(res_theta) > 0:
                try:
                    rp_g  = np.linspace(0, 1, 50)
                    ll_g  = np.linspace(0, np.log2(float(len_tr_f.max()) + 1), 50)
                    RG, LG = np.meshgrid(rp_g, ll_g)
                    L_g   = np.power(2., LG)
                    DS_g  = RG * L_g
                    DE_g  = np.maximum(L_g - DS_g - 1., 0.)
                    Q_g   = np.full_like(RG, 0.5)
                    W_g   = np.log(np.maximum(
                        w_fn_f(res_theta, Q_g.ravel(), DS_g.ravel(),
                               DE_g.ravel(), L_g.ravel(), RG.ravel()),
                        _FUNC_EPS)).reshape(RG.shape)

                    fig_f, axes_f = _plt_f.subplots(1, 2, figsize=(12, 4))
                    fig_f.suptitle(
                        f"{model_name}: {desc_f}\n"
                        f"theta*={np.round(res_theta, 3).tolist()}  "
                        f"val={edge_tr_auc_f:.4f}  test={edge_te_auc_f:.4f}",
                        fontsize=9)
                    im_f = axes_f[0].imshow(
                        W_g, origin="lower", aspect="auto",
                        extent=[0, 1, 0, ll_g[-1]], cmap="RdBu_r")
                    _plt_f.colorbar(im_f, ax=axes_f[0])
                    axes_f[0].set_xlabel("rel_pos (dist_start/walk_len)")
                    axes_f[0].set_ylabel("log2(walk_length)")
                    axes_f[0].set_title("log w(rel_pos, log2_len) at q=0.5")
                    med_ll  = float(np.log2(np.median(len_tr_f) + 1))
                    med_idx = int(np.argmin(np.abs(ll_g - med_ll)))
                    mid_idx = len(rp_g) // 2
                    axes_f[1].plot(rp_g, W_g[med_idx, :], color="steelblue",
                                   label=f"len\u2248{np.median(len_tr_f):.0f} (median)")
                    axes_f[1].plot(ll_g, W_g[:, mid_idx], color="tomato", ls="--",
                                   label="rp=0.5 vs log2_len")
                    axes_f[1].axhline(0, color="black", lw=0.7)
                    axes_f[1].set_xlabel("rel_pos (blue) / log2_len (red dashed)")
                    axes_f[1].set_ylabel("log w")
                    axes_f[1].legend(fontsize=8)
                    axes_f[1].set_title("1D slices through weight surface")
                    _plt_f.tight_layout()
                    _surf_dir = out_dir / model_name
                    _surf_dir.mkdir(parents=True, exist_ok=True)
                    _plt_f.savefig(_surf_dir / "weight_surface.png",
                                   dpi=120, bbox_inches="tight")
                    _plt_f.close()
                except Exception as _pe:
                    print(f"  Weight surface plot failed: {_pe}")

            model_dir = out_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(model_dir / "model_config.json", "w") as f:
                json.dump({
                    "model_name": model_name,
                    "description": desc_f,
                    "theta_star": [float(v) for v in res_theta] if len(res_theta) > 0 else [],
                    "n_iters": n_iters,
                    "n_params": len(theta0_f),
                    "seed": int(seed),
                }, f, indent=2)
            with open(model_dir / "summary.txt", "w") as f:
                f.write("Post-hoc aggregator summary\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Run ID: {run_id}\n")
                f.write(f"Model: {model_name}  ({desc_f})\n")
                f.write(f"theta* = {[float(v) for v in res_theta]}\n\n")
                f.write("Edge-level aggregated AUC:\n")
                f.write(f"  Train AUC: {edge_tr_auc_f:.4f} ({len(tr_sc_f)} edges)\n")
                f.write(f"  Test  AUC: {edge_te_auc_f:.4f} ({len(te_sc_f)} edges)\n")
            print(f"\u2713 Aggregator '{model_name}' ({desc_f})")
            print(f"  theta*={np.round(res_theta, 4).tolist()}  n_iters={n_iters}")
            print(f"  Edge agg_tr AUC={edge_tr_auc_f:.4f}  Edge test AUC={edge_te_auc_f:.4f}")
            continue

        else:
            print(f"⚠ Unknown aggregator model '{model_name}', skipping")
            continue

        train_pred = (train_probs >= 0.5).astype(int)
        test_pred = (test_probs >= 0.5).astype(int)

        walk_train_auc = roc_auc_score(y_train, train_probs)
        walk_test_auc = roc_auc_score(y_test, test_probs)

        edge_train_probs, edge_train_labels = _aggregate_edge_probs(
            edge_train, train_probs, y_train
        )
        edge_test_probs, edge_test_labels = _aggregate_edge_probs(
            edge_test, test_probs, y_test
        )

        edge_train_auc = roc_auc_score(edge_train_labels, edge_train_probs)
        edge_test_auc = roc_auc_score(edge_test_labels, edge_test_probs)

        model_dir = out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump({"model": model, "scaler": scaler}, f)
        with open(model_dir / "model_config.json", "w") as f:
            json.dump(model_config, f, indent=2)

        with open(model_dir / "summary.txt", "w") as f:
            f.write("Post-hoc aggregator summary\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Run ID: {run_id}\n")
            f.write(f"Model: {model_name}\n\n")
            f.write("Walk-level metrics:\n")
            f.write(f"  Train AUC: {walk_train_auc:.4f}\n")
            f.write(f"  Test  AUC: {walk_test_auc:.4f}\n")
            f.write(f"  Train ACC: {accuracy_score(y_train, train_pred):.4f}\n")
            f.write(f"  Test  ACC: {accuracy_score(y_test, test_pred):.4f}\n")
            f.write(f"  Train F1: {f1_score(y_train, train_pred):.4f}\n")
            f.write(f"  Test  F1: {f1_score(y_test, test_pred):.4f}\n\n")
            f.write("Edge-level aggregated AUC:\n")
            f.write(
                f"  Train AUC: {edge_train_auc:.4f} ({len(edge_train_probs)} edges)\n"
            )
            f.write(
                f"  Test  AUC: {edge_test_auc:.4f} ({len(edge_test_probs)} edges)\n"
            )

        print(f"✓ Aggregator '{model_name}' complete: {model_dir}")


def main():
    args = parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    validate_config(cfg, context="posthoc")
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

    if (
        "predictions" in artifacts
        or "triplets" in artifacts
        or "heatmaps" in artifacts
    ):
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

    if "triplets" in artifacts or "heatmaps" in artifacts:
        required_splits = splits if "triplets" in artifacts else splits
        save_triplets_and_heatmaps(
            exp_dir, cfg.dataset.name, run_id, epoch, required_splits
        )

    if "aggregator" in artifacts:
        run_aggregator(exp_dir, cfg.dataset.name, run_id, epoch, agg_models, seed, device)

    print(f"\n✓ Post-hoc pipeline complete. Artifacts: {exp_dir / 'posthoc' / run_id}")


if __name__ == "__main__":
    main()
