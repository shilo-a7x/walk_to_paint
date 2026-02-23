#!/usr/bin/env python3
"""
Extract per-edge score distributions from a trained checkpoint.

For each edge in transformer's val/test splits, collect all predictions across walks
and compute aggregated statistics (percentiles, mean, std, count).

Aggregator will use:
  - Agg train: 80% of transformer val edges
  - Agg val: 20% of transformer val edges
  - Agg test: 100% of transformer test edges

Output: A pickle file with structure:
{
    'agg_train': {(u, v, label): {'features': array, 'ground_truth': label}, ...},
    'agg_val': {(u, v, label): {'features': array, 'ground_truth': label}, ...},
    'agg_test': {(u, v, label): {'features': array, 'ground_truth': label}, ...},
    'feature_names': ['p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'mean', 'std', 'count']
}

USAGE:
  For new checkpoints (with embedded config):
    python scripts/extract_edge_scores.py \\
      --checkpoint <path.ckpt> \\
      --output <output.pkl>

  For old checkpoints (without embedded config):
    python scripts/extract_edge_scores.py \\
      --checkpoint <path.ckpt> \\
      --config config.yaml \\
      --output <output.pkl> \\
      [dataset.name=wiki-rfa ...]

Key behavior:
  - Checkpoint config is automatically extracted and used
  - Data is prepared with the exact config used during training
  - All params (ignore_index, pad_id, vocab_size, etc.) are aligned
  - Seed is read from checkpoint's config for reproducibility
"""


import os
import sys
import random
import argparse
import pickle
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config, get_seed
from src.data.prepare_data import prepare_data
from src.data.dataset_cache import load_dataset_cache, tokenizer_from_cache
from src.model.lit_model import LitEdgeClassifier


def compute_edge_features(scores):
    """
    Compute aggregated features from a list of probability scores.

    Args:
        scores: List of scalar probabilities (for binary) or probability vectors

    Returns:
        Feature vector: [p5, p10, p25, p50, p75, p90, p95, mean, std, count]
    """
    if len(scores) == 0:
        return np.zeros(10)

    scores = np.array(scores)
    percentiles = np.percentile(scores, [5, 10, 25, 50, 75, 90, 95])
    mean = np.mean(scores)
    std = np.std(scores)
    count = len(scores)

    return np.concatenate([percentiles, [mean, std, count]])


def build_node_id_map(tokenizer, device):
    """Build a tensor mapping token_id -> node_id (or -1 if not a node token)."""
    node_id_map = torch.full((tokenizer.vocab_size,), -1, dtype=torch.long)
    for tok_id, tok_str in tokenizer.id2token.items():
        node_id = tokenizer.parse_node(tok_str)
        if node_id is not None:
            node_id_map[int(tok_id)] = int(node_id)
    return node_id_map.to(device)


def extract_edge_scores_from_dataloader(
    model,
    dataloader,
    tokenizer,
    device,
    stage_name,
    cfg,
    max_batches=None,
    node_id_map=None,
):
    """
    Extract edge scores from a single dataloader (train/val/test).

    Returns:
        edge_scores: dict mapping (u, v, label) -> list of probability scores
    """
    model.eval()
    edge_scores = defaultdict(list)
    ignore_index = cfg.model.ignore_index
    num_classes = int(cfg.model.num_classes)
    if node_id_map is None:
        node_id_map = build_node_id_map(tokenizer, device)

    print(f"\nExtracting scores from {stage_name} split...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc=f"Processing {stage_name}")
        ):
            # Handle both 3-tuple (old) and 4-tuple (new with metadata)
            if len(batch) == 4:
                input_ids, labels, attention_mask, _ = batch
            else:
                input_ids, labels, attention_mask = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass
            logits = model.model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)

            valid_mask = labels != ignore_index
            valid_idx = valid_mask.nonzero(as_tuple=False)
            if valid_idx.numel() == 0:
                if max_batches is not None and (batch_idx + 1) >= max_batches:
                    break
                continue

            b_idx = valid_idx[:, 0]
            i_idx = valid_idx[:, 1]

            within_bounds = (i_idx > 0) & (i_idx < input_ids.size(1) - 1)
            if within_bounds.any():
                b_idx = b_idx[within_bounds]
                i_idx = i_idx[within_bounds]
            else:
                if max_batches is not None and (batch_idx + 1) >= max_batches:
                    break
                continue

            u_ids = node_id_map[input_ids[b_idx, i_idx - 1]]
            v_ids = node_id_map[input_ids[b_idx, i_idx + 1]]
            valid_uv = (u_ids >= 0) & (v_ids >= 0)
            if not valid_uv.any():
                if max_batches is not None and (batch_idx + 1) >= max_batches:
                    break
                continue

            b_idx = b_idx[valid_uv]
            i_idx = i_idx[valid_uv]
            u_ids = u_ids[valid_uv]
            v_ids = v_ids[valid_uv]

            label_class = labels[b_idx, i_idx]
            if num_classes == 2:
                score = probs[b_idx, i_idx, 1]
            else:
                score = probs[b_idx, i_idx, label_class]

            for u, v, y, s in zip(
                u_ids.tolist(),
                v_ids.tolist(),
                label_class.tolist(),
                score.tolist(),
            ):
                edge_scores[(u, v, int(y))].append(float(s))

            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

    print(f"Collected scores for {len(edge_scores)} unique edges in {stage_name}")
    return edge_scores


def extract_edge_occurrence_predictions(
    model,
    dataloader,
    tokenizer,
    device,
    stage_name,
    cfg,
    max_batches=None,
    node_id_map=None,
):
    """
    Extract per-edge-occurrence predictions (score + labels + walk metadata).

    Returns:
        payload: dict of lists containing per-occurrence fields
    """
    model.eval()
    ignore_index = cfg.model.ignore_index
    pad_id = int(cfg.model.pad_id)
    num_classes = int(cfg.model.num_classes)
    if node_id_map is None:
        node_id_map = build_node_id_map(tokenizer, device)

    payload = {
        "edge_id": [],
        "score": [],
        "pred_label": [],
        "true_label": [],
        "walk_id": [],
        "position": [],
        "walk_len": [],
        "dist_from_start": [],
        "dist_from_end": [],
        "split": stage_name,
        "dataset": cfg.dataset.name,
    }

    print(f"\nExtracting per-occurrence predictions from {stage_name} split...")

    total_seen = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc=f"Processing {stage_name} occurrences")
        ):
            input_ids, labels, attention_mask = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            logits = model.model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)

            batch_size, _ = input_ids.shape

            valid_mask = labels != ignore_index
            valid_idx = valid_mask.nonzero(as_tuple=False)
            if valid_idx.numel() == 0:
                total_seen += batch_size
                if max_batches is not None and (batch_idx + 1) >= max_batches:
                    break
                continue

            b_idx = valid_idx[:, 0]
            i_idx = valid_idx[:, 1]

            within_bounds = (i_idx > 0) & (i_idx < input_ids.size(1) - 1)
            if within_bounds.any():
                b_idx = b_idx[within_bounds]
                i_idx = i_idx[within_bounds]
            else:
                total_seen += batch_size
                if max_batches is not None and (batch_idx + 1) >= max_batches:
                    break
                continue

            u_ids = node_id_map[input_ids[b_idx, i_idx - 1]]
            v_ids = node_id_map[input_ids[b_idx, i_idx + 1]]
            valid_uv = (u_ids >= 0) & (v_ids >= 0)
            if not valid_uv.any():
                total_seen += batch_size
                if max_batches is not None and (batch_idx + 1) >= max_batches:
                    break
                continue

            b_idx = b_idx[valid_uv]
            i_idx = i_idx[valid_uv]
            u_ids = u_ids[valid_uv]
            v_ids = v_ids[valid_uv]

            true_label = labels[b_idx, i_idx]
            pred_label = torch.argmax(probs[b_idx, i_idx], dim=-1)
            if num_classes == 2:
                score = probs[b_idx, i_idx, 1]
            else:
                score = probs[b_idx, i_idx, pred_label]

            walk_len_tokens = (input_ids != pad_id).sum(dim=1).long()
            walk_len_edges = torch.clamp((walk_len_tokens - 1) // 2, min=0)
            edge_pos = (i_idx // 2).long()
            dist_from_start = edge_pos
            dist_from_end = walk_len_edges[b_idx] - 1 - edge_pos
            walk_ids = (b_idx + total_seen).long()

            payload["edge_id"].extend(list(zip(u_ids.tolist(), v_ids.tolist())))
            payload["score"].extend([float(s) for s in score.tolist()])
            payload["pred_label"].extend([int(p) for p in pred_label.tolist()])
            payload["true_label"].extend([int(t) for t in true_label.tolist()])
            payload["walk_id"].extend([int(w) for w in walk_ids.tolist()])
            payload["position"].extend([int(p) for p in edge_pos.tolist()])
            payload["walk_len"].extend([int(l) for l in walk_len_edges[b_idx].tolist()])
            payload["dist_from_start"].extend(
                [int(d) for d in dist_from_start.tolist()]
            )
            payload["dist_from_end"].extend([int(d) for d in dist_from_end.tolist()])

            total_seen += batch_size

            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

    print(f"Collected {len(payload['edge_id'])} edge occurrences in {stage_name} split")
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Extract edge score features from checkpoint"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output pickle file path"
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="[OPTIONAL] Path to config file (only needed for old checkpoints without embedded cfg)",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-occurrence predictions for val/test splits",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default="outputs/predictions",
        help="Base directory for per-occurrence prediction outputs",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier for prediction output filenames",
    )
    parser.add_argument(
        "--preds-only",
        action="store_true",
        help="Skip aggregated features and save predictions only",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit number of dataloader batches per split (smoke test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override dataloader batch size for extraction",
    )
    parser.add_argument(
        "overrides", nargs="*", help="[OPTIONAL] Config overrides in dotlist format"
    )

    args = parser.parse_args()

    # Set device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Try to load model and extract embedded config
    print(f"\nLoading model from checkpoint: {args.checkpoint}")
    model_has_embedded_cfg = False
    try:
        # Try loading without cfg (works for new checkpoints with embedded cfg)
        model = LitEdgeClassifier.load_from_checkpoint(args.checkpoint)
        print("✅ Checkpoint has embedded config (new format)")
        model_has_embedded_cfg = True

        # Extract config from model's hyperparameters
        if hasattr(model, "hparams") and "cfg" in model.hparams:
            cfg = OmegaConf.create(model.hparams["cfg"])
            print("✅ Extracted config from checkpoint")
        else:
            raise ValueError("Checkpoint missing embedded 'cfg' in hparams")

    except (TypeError, ValueError) as e:
        # Fall back to CLI config for old checkpoints
        if args.config is None:
            raise ValueError(
                f"Checkpoint missing embedded config and no --config provided.\n"
                f"For old checkpoints, provide: --config <config_file> [overrides...]\n"
                f"Error: {e}"
            )

        print(f"⚠️  Checkpoint missing embedded config, using CLI config: {args.config}")
        overrides = args.overrides if args.overrides else []
        cfg = load_config(args.config, overrides=overrides)
        model = None  # Will create after prepare_data() populates cfg.model

    # Apply CLI overrides (dotlist) to cfg for both new/old checkpoints
    if args.overrides:
        try:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
        except Exception:
            pass

    if args.batch_size is not None:
        cfg.training.batch_size = int(args.batch_size)

    # Set seeds for reproducibility from config
    try:
        seed = get_seed(cfg)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"✅ Reproducibility enabled: seed={seed}")
    except ValueError as e:
        print(f"❌ ERROR: {e}")
        print("Cannot proceed without a valid seed.")
        sys.exit(1)

    # Load tokenizer from dataset cache (only supported format)
    dataset_cache_path = os.path.join(cfg.dataset.data_dir, "dataset_cache.pt")
    print(f"Loading tokenizer from dataset cache: {dataset_cache_path}...")
    cache_data = load_dataset_cache(dataset_cache_path)
    tokenizer = tokenizer_from_cache(cache_data)

    print(
        f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, num_classes={tokenizer.num_edge_tokens}"
    )

    # Prepare data with checkpoint's config (ensures alignment)
    # IMPORTANT: Must be called BEFORE loading old checkpoint to populate cfg.model.*
    print("\nPreparing data with checkpoint's config...")
    data_module = prepare_data(cfg)

    # NOW load the checkpoint (has all cfg.model.* values from prepare_data)
    if not model_has_embedded_cfg:
        print("\nLoading checkpoint with populated config...")
        model = LitEdgeClassifier.load_from_checkpoint(args.checkpoint, cfg=cfg)
        print("✅ Old checkpoint loaded with prepared config")

    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    node_id_map = build_node_id_map(tokenizer, device)

    if args.preds_only and not args.save_predictions:
        print("⚠️  --preds-only set; enabling --save-predictions")
        args.save_predictions = True

    # Extract scores from transformer val and test (only these have labels)
    if not args.preds_only:
        print("\nExtracting from transformer val and test splits...")
        val_scores = extract_edge_scores_from_dataloader(
            model,
            data_module["val"],
            tokenizer,
            device,
            "transformer_val",
            cfg,
            max_batches=args.max_batches,
            node_id_map=node_id_map,
        )
        test_scores = extract_edge_scores_from_dataloader(
            model,
            data_module["test"],
            tokenizer,
            device,
            "transformer_test",
            cfg,
            max_batches=args.max_batches,
            node_id_map=node_id_map,
        )

    # Optional: Save per-occurrence predictions for val/test splits
    if args.save_predictions:
        run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(args.predictions_dir, cfg.dataset.name, "raw_scores")
        os.makedirs(base_dir, exist_ok=True)

        val_payload = extract_edge_occurrence_predictions(
            model,
            data_module["val"],
            tokenizer,
            device,
            "val",
            cfg,
            max_batches=args.max_batches,
            node_id_map=node_id_map,
        )
        val_path = os.path.join(base_dir, f"{run_id}_val.pkl")
        with open(val_path, "wb") as f:
            pickle.dump(val_payload, f)
        print(f"Saved val predictions to: {val_path}")

        test_payload = extract_edge_occurrence_predictions(
            model,
            data_module["test"],
            tokenizer,
            device,
            "test",
            cfg,
            max_batches=args.max_batches,
            node_id_map=node_id_map,
        )
        test_path = os.path.join(base_dir, f"{run_id}_test.pkl")
        with open(test_path, "wb") as f:
            pickle.dump(test_payload, f)
        print(f"Saved test predictions to: {test_path}")

    if args.preds_only:
        return

    # Compute features for each edge
    print("\nComputing aggregated features...")

    feature_names = [
        "p5",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "mean",
        "std",
        "count",
    ]

    def build_feature_dict(edge_scores_dict):
        """Convert raw scores to feature dict"""
        result = {}
        for edge_key, scores in tqdm(
            edge_scores_dict.items(), desc="Computing features"
        ):
            features = compute_edge_features(scores)
            result[edge_key] = {
                "features": features,
                "ground_truth": edge_key[2],  # label is third element
            }
        return result

    val_features = build_feature_dict(val_scores)
    test_features = build_feature_dict(test_scores)

    # Split transformer val into agg_train (80%) and agg_val (20%)
    print("\nSplitting transformer val into aggregator train (80%) and val (20%)...")
    val_edges = list(val_features.keys())
    np.random.seed(42)
    np.random.shuffle(val_edges)
    split_idx = int(0.8 * len(val_edges))

    agg_train_edges = val_edges[:split_idx]
    agg_val_edges = val_edges[split_idx:]

    agg_train_features = {e: val_features[e] for e in agg_train_edges}
    agg_val_features = {e: val_features[e] for e in agg_val_edges}
    agg_test_features = test_features

    # Save to pickle
    output_data = {
        "agg_train": agg_train_features,
        "agg_val": agg_val_features,
        "agg_test": agg_test_features,
        "feature_names": feature_names,
        "config": {
            "dataset": cfg.dataset.name,
            "num_classes": cfg.model.num_classes,
            "checkpoint": args.checkpoint,
        },
    }
    print(f"\nSaving features to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(output_data, f)

    print("\nSummary:")
    print(f"  Agg train edges: {len(agg_train_features)}")
    print(f"  Agg val edges: {len(agg_val_features)}")
    print(f"  Agg test edges: {len(agg_test_features)}")
    print(f"  Feature dimension: {len(feature_names)}")
    print(f"\nFeatures saved to: {args.output}")


if __name__ == "__main__":
    main()
