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
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config, get_seed
from src.data.prepare_data import prepare_data
from src.data.tokenizer import Tokenizer
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


def extract_edge_scores_from_dataloader(
    model, dataloader, tokenizer, device, stage_name, cfg
):
    """
    Extract edge scores from a single dataloader (train/val/test).

    Returns:
        edge_scores: dict mapping (u, v, label) -> list of probability scores
    """
    model.eval()
    edge_scores = defaultdict(list)
    ignore_index = cfg.model.ignore_index

    print(f"\nExtracting scores from {stage_name} split...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Processing {stage_name}"):
            input_ids, labels, attention_mask = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass
            logits = model.model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)

            # Find valid target positions
            valid_mask = labels != ignore_index

            # Process each valid position
            batch_size, seq_len = input_ids.shape
            for b in range(batch_size):
                for i in range(1, seq_len - 1):  # Skip boundaries
                    if not valid_mask[b, i]:
                        continue

                    # Extract edge identity
                    try:
                        u_token = tokenizer.id2token.get(
                            input_ids[b, i - 1].item(), None
                        )
                        v_token = tokenizer.id2token.get(
                            input_ids[b, i + 1].item(), None
                        )

                        if u_token is None or v_token is None:
                            continue

                        u = tokenizer.parse_node(u_token)
                        v = tokenizer.parse_node(v_token)
                        label_class = labels[b, i].item()

                        if u is None or v is None:
                            continue

                        edge_key = (u, v, label_class)

                        # Extract probability score
                        if cfg.model.num_classes == 2:
                            # Binary: use positive class probability
                            score = probs[b, i, 1].item()
                        else:
                            # Multi-class: use true class probability
                            score = probs[b, i, label_class].item()

                        edge_scores[edge_key].append(score)

                    except Exception as e:
                        # Skip problematic edges
                        continue

    print(f"Collected scores for {len(edge_scores)} unique edges in {stage_name}")
    return edge_scores


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
        "overrides", nargs="*", help="[OPTIONAL] Config overrides in dotlist format"
    )

    args = parser.parse_args()

    # Set device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Try to load model and extract embedded config
    print(f"\nLoading model from checkpoint: {args.checkpoint}")
    try:
        # Try loading without cfg (works for new checkpoints with embedded cfg)
        model = LitEdgeClassifier.load_from_checkpoint(args.checkpoint)
        print("✅ Checkpoint has embedded config (new format)")

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

        # Reload model with explicit cfg for old checkpoints
        model = LitEdgeClassifier.load_from_checkpoint(args.checkpoint, cfg=cfg)

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

    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Load tokenizer
    tokenizer_path = os.path.join(cfg.dataset.data_dir, cfg.dataset.tokenizer_file)
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.load(tokenizer_path)
    print(
        f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, num_classes={tokenizer.num_edge_tokens}"
    )

    # Prepare data with checkpoint's config (ensures alignment)
    print("\nPreparing data with checkpoint's config...")
    data_module = prepare_data(cfg)

    # Extract scores from transformer val and test (only these have labels)
    print("\nExtracting from transformer val and test splits...")
    val_scores = extract_edge_scores_from_dataloader(
        model, data_module["val"], tokenizer, device, "transformer_val", cfg
    )
    test_scores = extract_edge_scores_from_dataloader(
        model, data_module["test"], tokenizer, device, "transformer_test", cfg
    )

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
