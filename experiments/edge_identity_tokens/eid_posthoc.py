"""EID equivalent of run_posthoc.py's predictions+aggregator path.

Every EID test AUC reported so far (v1/v2 Optuna, eval_eid_checkpoint.py, the budget
sweep) came from PyTorch Lightning's own Trainer.test() -- a WALK-LEVEL metric, scoring
each individual walk occurrence directly. Every production number actually in the paper
(e.g. bitcoin-alpha local attention 0.9188 at seed 42, or the 10-seed 0.9134+-.0173) is
EDGE-LEVEL: pooling an edge's many walk occurrences via the confidence-weighted vote
(func_logit_power) that run_posthoc.py::run_aggregator implements. Comparing an EID
walk-level number against a production edge-level number is not apples-to-apples.

This script produces the missing edge-level number for an EID checkpoint. It reuses
run_posthoc.py's aggregation core (run_aggregator, load_prediction, prediction_path,
resolve_checkpoint) completely unmodified -- that logic doesn't touch the model, only
the saved prediction pickles, so it works for any checkpoint whose predictions are saved
in the same schema/path convention. What it does NOT reuse is
PerEpochPredictionSaver._extract_predictions (src/training/callbacks.py), since that
calls `pl_module.model(input_ids, attention_mask=...)` -- EdgeIdentityTransformerModel's
forward signature requires a second positional arg, sign_ids, that production's model
doesn't have at all. _extract_eid_predictions below is a minimal EID-aware copy of that
one method (threads sign_ids, reapplies mask_node_tokens/mask_edge_tokens the same way),
writing to the exact same pickle schema/path so run_aggregator needs no changes.

Usage:
  .venv/bin/python experiments/edge_identity_tokens/eid_posthoc.py \
      --exp-dir outputs/bitcoin-alpha/EID_BUDGET_DYN_120930_<timestamp> \
      --device 0 --run-id posthoc
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from run_posthoc import (
    resolve_checkpoint,
    _parse_epoch,
    prediction_path,
    run_aggregator,
)
from src.utils.config import validate_config, get_seed

from experiments.edge_identity_tokens.eid_src.model.lit_model import EIDLitEdgeClassifier
from experiments.edge_identity_tokens.eid_src.data.prepare_eid_data import prepare_eid_data
from experiments.edge_identity_tokens.run_eid import EID_CACHE_PATH, ensure_eid_cache


def _extract_eid_predictions(model, dataloader, split_name, epoch, device):
    """EID-aware copy of PerEpochPredictionSaver._extract_predictions -- same output
    schema, same AUC computation, only the forward call differs (sign_ids threaded in,
    matching EIDLitEdgeClassifier._step)."""
    model.eval()
    all_data = {
        "edge_ids": [], "walk_ids": [], "positions": [], "walk_lengths": [],
        "predictions": [], "probabilities": [], "targets": [], "correct": [],
    }
    ignore_idx = model.cfg.model.ignore_index

    with torch.no_grad():
        for batch in dataloader:
            input_ids, labels, attention_mask, metadata = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            sign_ids = metadata["sign_ids"].to(device)

            # Reapply every ablation flag the SAME way EIDLitEdgeClassifier._step does at
            # train/eval time -- this used to call only _maybe_apply_token_masking (identity-
            # only, mask_node_tokens/mask_edge_tokens), silently skipping mask_edge_tokens's
            # sign-channel companion and all three EID-native scramble/context ablations.
            # Checkpoints were always trained correctly (_step applies everything below); this
            # bug only affected edge-level posthoc evaluation -- the exact same bug category
            # (and same fix pattern: posthoc-only rerun, no retraining) as production's real
            # abl:maskedge bug, see PAPER_CLOSEOUT_LOG.md 2026-08-23. Confirmed the walk-level
            # Trainer.test() numbers (test_auc_epoch) were NOT affected, since those go through
            # _step directly.
            positions_for_mask = metadata.get("positions")
            if positions_for_mask is not None and hasattr(model, "_maybe_apply_token_masking"):
                positions_for_mask = positions_for_mask.to(device)
                node_mask = (positions_for_mask >= 0) & ((positions_for_mask % 2) == 0)
                edge_mask = (positions_for_mask >= 0) & ((positions_for_mask % 2) == 1)
                input_ids = model._maybe_apply_token_masking(input_ids, node_mask, edge_mask)
                if hasattr(model, "_maybe_apply_edge_sign_masking"):
                    sign_ids = model._maybe_apply_edge_sign_masking(sign_ids, edge_mask)
                if hasattr(model, "_maybe_apply_eid_sign_scramble"):
                    sign_ids = model._maybe_apply_eid_sign_scramble(
                        sign_ids, edge_mask, metadata["edge_ids"].to(device), metadata["edge_classes"].to(device),
                        dataset=dataloader.dataset,
                    )
                if hasattr(model, "_maybe_apply_eid_identity_scramble"):
                    input_ids = model._maybe_apply_eid_identity_scramble(
                        input_ids, edge_mask, metadata["edge_ids"].to(device),
                        dataset=dataloader.dataset,
                    )
                if hasattr(model, "_maybe_apply_target_identity_masking"):
                    input_ids = model._maybe_apply_target_identity_masking(input_ids, edge_mask, labels)
                if hasattr(model, "_maybe_apply_context_edge_masking"):
                    input_ids, sign_ids = model._maybe_apply_context_edge_masking(
                        input_ids, sign_ids, edge_mask, labels
                    )

            logits = model.model(input_ids, sign_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            preds_flat = preds.view(-1)
            probs_flat = probs.view(-1, probs.size(-1))
            targets_flat = labels.view(-1)
            valid_mask = targets_flat != ignore_idx

            preds_valid = preds_flat[valid_mask].cpu().numpy()
            probs_valid = probs_flat[valid_mask].cpu().numpy()
            targets_valid = targets_flat[valid_mask].cpu().numpy()
            correct_valid = preds_valid == targets_valid

            all_data["predictions"].append(preds_valid)
            all_data["probabilities"].append(probs_valid)
            all_data["targets"].append(targets_valid)
            all_data["correct"].append(correct_valid)

            B, S = labels.shape
            walk_ids = metadata["walk_ids"]
            walk_lengths = metadata["walk_lengths"]
            if walk_ids.dim() == 1:
                walk_ids = walk_ids.unsqueeze(1).expand(B, S)
            if walk_lengths.dim() == 1:
                walk_lengths = walk_lengths.unsqueeze(1).expand(B, S)
            edge_ids_cpu = metadata["edge_ids"].view(-1).cpu().numpy()
            walk_ids_cpu = walk_ids.reshape(-1).cpu().numpy()
            positions_cpu = metadata["positions"].view(-1).cpu().numpy()
            walk_lengths_cpu = walk_lengths.reshape(-1).cpu().numpy()
            valid_mask_cpu = valid_mask.cpu().numpy()

            pos_seq = positions_cpu[valid_mask_cpu]
            len_seq = walk_lengths_cpu[valid_mask_cpu]
            pos_edges = (pos_seq - 1) // 2
            len_edges = (len_seq - 1) // 2

            all_data["edge_ids"].append(edge_ids_cpu[valid_mask_cpu])
            all_data["walk_ids"].append(walk_ids_cpu[valid_mask_cpu])
            all_data["positions"].append(pos_edges)
            all_data["walk_lengths"].append(len_edges)

    result = {
        "epoch": epoch,
        "split": split_name,
        "edge_ids": np.concatenate(all_data["edge_ids"]),
        "walk_ids": np.concatenate(all_data["walk_ids"]),
        "positions": np.concatenate(all_data["positions"]),
        "walk_lengths": np.concatenate(all_data["walk_lengths"]),
        "predictions": np.concatenate(all_data["predictions"]),
        "probabilities": np.concatenate(all_data["probabilities"]),
        "targets": np.concatenate(all_data["targets"]),
        "correct": np.concatenate(all_data["correct"]),
    }
    result["dist_from_start"] = result["positions"]
    result["dist_from_end"] = result["walk_lengths"] - result["positions"] - 1

    unique_classes = np.unique(result["targets"])
    if len(unique_classes) < 2:
        result["auc"] = None
    else:
        result["auc"] = float(roc_auc_score(result["targets"], result["probabilities"][:, 1]))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=str, required=True)
    ap.add_argument("--checkpoint-choice", type=str, default="best", choices=["best", "last", "other"])
    ap.add_argument("--checkpoint-path", type=str, default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--splits", type=str, default="val,test")
    ap.add_argument("--agg-models", type=str, default="func_logit_power")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    checkpoint_dir = exp_dir / "checkpoints"
    ckpt_path = resolve_checkpoint(checkpoint_dir, args.checkpoint_choice, args.checkpoint_path)
    epoch = _parse_epoch(ckpt_path) or 0
    run_id = args.run_id or f"{ckpt_path.stem}_eid_posthoc"

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        device = torch.device("cuda:0")

    raw_ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    saved_cfg = (raw_ckpt.get("hyper_parameters") or {}).get("cfg")
    if saved_cfg is None:
        raise RuntimeError("Checkpoint has no saved cfg -- cannot safely reproduce training-time setup.")
    cfg = OmegaConf.create(saved_cfg)
    validate_config(cfg, context="posthoc")
    cfg.training.checkpoint_dir = str(checkpoint_dir)
    seed = get_seed(cfg)

    print(f"Checkpoint: {ckpt_path}")
    print(f"dataset={cfg.dataset.name} num_walks={cfg.dataset.num_walks} seed={seed} "
          f"edge_embed_rank={cfg.model.edge_embed_rank} local_attention_window={cfg.model.local_attention_window}")

    eid_cache_path = EID_CACHE_PATH.format(dataset=cfg.dataset.name, num_walks=int(cfg.dataset.num_walks))
    ensure_eid_cache(cfg, eid_cache_path)
    data_module = prepare_eid_data(cfg, eid_cache_path)

    model = EIDLitEdgeClassifier.load_from_checkpoint(str(ckpt_path), cfg=cfg, map_location="cpu")
    model = model.to(device)
    model.eval()

    predictions_dir = checkpoint_dir / f"{cfg.dataset.name}_predictions" / f"epoch_{epoch:03d}"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        print(f"Extracting {split} predictions (edge-level, walk-occurrence granularity)...")
        pred = _extract_eid_predictions(model, data_module[split], split, epoch, device)
        n_occ = len(pred["edge_ids"])
        n_edges = len(np.unique(pred["edge_ids"]))
        auc_str = f"{pred['auc']:.4f}" if pred["auc"] is not None else "N/A"
        print(f"  {split}: {n_occ} occurrences, {n_edges} unique edges, walk-level AUC={auc_str}")
        out_path = prediction_path(exp_dir, cfg.dataset.name, epoch, split)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(pred, f)

    agg_models = [m.strip() for m in args.agg_models.split(",") if m.strip()]
    run_aggregator(exp_dir, cfg.dataset.name, run_id, epoch, agg_models, seed, device)

    for agg_model in agg_models:
        summary_path = exp_dir / "posthoc" / run_id / "aggregator" / agg_model / "summary.txt"
        if summary_path.exists():
            print(f"\n=== {summary_path} ===")
            print(summary_path.read_text())


if __name__ == "__main__":
    main()
