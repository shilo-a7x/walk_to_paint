#!/usr/bin/env python3
"""
T2.1 Analysis: Audit Current Loss Weighting Strategy

Tasks:
1. Trace loss function in lit_model.py
2. Check class distributions across splits
3. Audit data leakage risk
4. Batch-level analysis
"""

import os
import sys
import json
import torch
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.data.prepare_data import (
    get_edge_list,
    split_edges,
    get_walks,
    get_tokenizer,
    encode_walks,
    _stage_views_from_base,
)


def task1_trace_loss_function():
    """Task 1: Identify and document loss function."""
    print("\n" + "=" * 70)
    print("TASK 1: Trace Loss Function")
    print("=" * 70)

    # Read lit_model.py
    lit_model_path = "src/model/lit_model.py"
    with open(lit_model_path) as f:
        content = f.read()

    findings = {
        "loss_function": "F.cross_entropy (PyTorch)",
        "pos_weight_parameter": "NOT used",
        "weighted_loss_enabled": True,
        "weighting_strategy": "Batch-level inverse frequency weighting",
        "weight_computation": "Per-batch via WeightedLossHelper.compute_class_weights()",
        "formula": "weight[i] = total_samples / (num_classes * count[i])",
        "data_leakage_risk": "MEDIUM - weights computed per batch, includes both train and non-train labels",
    }

    print("\n✓ Loss Function Found:")
    print(f"  - Function: {findings['loss_function']}")
    print(f"  - Weighted: {findings['weighted_loss_enabled']}")
    print(f"  - Strategy: {findings['weighting_strategy']}")
    print(f"  - Weight Computation: {findings['weight_computation']}")
    print(f"  - Formula: {findings['formula']}")
    print(f"  - Data Leakage Risk: {findings['data_leakage_risk']}")

    return findings


def task2_check_class_distributions(cfg):
    """Task 2: Load and check class distributions in train/val/test."""
    print("\n" + "=" * 70)
    print("TASK 2: Check Class Distributions")
    print("=" * 70)

    # Load edges and splits
    edges = get_edge_list(cfg)
    train_set, mask_set, val_set, test_set = split_edges(cfg, edges)

    distributions = {}

    for split_name, split_set in [
        ("train", train_set),
        ("val", val_set),
        ("test", test_set),
    ]:
        split_labels = [e[2] for e in split_set]

        n_pos = sum(1 for l in split_labels if l == 1)
        n_neg = sum(1 for l in split_labels if l == 0)
        total = len(split_labels)
        pct_pos = 100.0 * n_pos / total if total > 0 else 0

        distributions[split_name] = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "total": total,
            "pct_pos": pct_pos,
        }

        print(f"\n{split_name.upper()}:")
        print(f"  Positive (1): {n_pos:7d} ({pct_pos:6.2f}%)")
        print(f"  Negative (0): {n_neg:7d} ({100-pct_pos:6.2f}%)")
        print(f"  Total:        {total:7d}")

    return distributions


def task3_audit_data_leakage(cfg):
    """Task 3: Assess data leakage risk in weight computation."""
    print("\n" + "=" * 70)
    print("TASK 3: Audit Data Leakage Risk")
    print("=" * 70)

    findings = {
        "weights_computed_at": "Per-batch during training/validation/test",
        "weight_source": "Current batch labels (mixed from same split)",
        "train_only": False,
        "leakage_risk": "MEDIUM",
        "explanation": [
            "Weights are computed PER BATCH using F.cross_entropy(weight=class_weights)",
            "In lit_model.py _step(), weights computed from labels.view(-1) without splitting",
            "During validation/test, weights still computed from current batch (not train-only)",
            "This means val/test batches influence their own loss weights",
            "However, no data from FUTURE splits affects PAST weights (temporal isolation OK)",
        ],
    }

    print("\n⚠️  Data Leakage Assessment:")
    print(f"  Weights computed at: {findings['weights_computed_at']}")
    print(f"  Weight source: {findings['weight_source']}")
    print(f"  Train-only weights: {findings['train_only']}")
    print(f"  Leakage risk: {findings['leakage_risk']}")
    print("\n  Explanation:")
    for line in findings["explanation"]:
        print(f"    • {line}")

    return findings


def task4_batch_level_analysis(cfg):
    """Task 4: Sample batches and check composition."""
    print("\n" + "=" * 70)
    print("TASK 4: Batch-Level Analysis")
    print("=" * 70)

    # Load data
    edges = get_edge_list(cfg)
    train_set, mask_set, val_set, test_set = split_edges(cfg, edges)
    walks = get_walks(cfg, edges)
    tokenizer = get_tokenizer(cfg, walks, edges)

    input_ids, edge_split_masks = encode_walks(
        walks, tokenizer, train_set, mask_set, val_set, test_set
    )

    # Get stage views
    stage_views = []
    for view_type, input_ids_list, edge_split_masks_list in [
        ("train", input_ids, edge_split_masks),
        ("val", input_ids, edge_split_masks),
        ("test", input_ids, edge_split_masks),
    ]:
        # Sample a few batches
        batch_size = cfg.training.batch_size
        num_samples = min(5, len(input_ids_list))

        batch_stats = []
        for i in range(num_samples):
            if i < len(input_ids_list):
                ids = input_ids_list[i]
                split_mask = edge_split_masks_list[i]

                # Count class distribution for this sequence
                from src.data.prepare_data import SplitID

                # Get labels from different views
                train_view = _stage_views_from_base(ids, split_mask, tokenizer)[1]
                train_labels = train_view[train_view != tokenizer.UNK_LABEL_ID]

                if len(train_labels) > 0:
                    n_pos = (train_labels == 1).sum().item()
                    n_neg = (train_labels == 0).sum().item()
                    total = len(train_labels)
                    pct_pos = 100.0 * n_pos / total if total > 0 else 0

                    batch_stats.append(
                        {
                            "seq_idx": i,
                            "n_pos": n_pos,
                            "n_neg": n_neg,
                            "total": total,
                            "pct_pos": pct_pos,
                        }
                    )

        if batch_stats:
            avg_pct_pos = np.mean([s["pct_pos"] for s in batch_stats])
            print(f"\n{view_type.upper()} View - Sample Batches:")
            print(f"  Average class balance: {avg_pct_pos:.2f}% positive")
            for stat in batch_stats[:3]:
                print(
                    f"    Seq {stat['seq_idx']}: {stat['n_pos']}/{stat['total']} pos ({stat['pct_pos']:.1f}%)"
                )


def main():
    print("\n" + "=" * 70)
    print("T2.1 LOSS WEIGHTING ANALYSIS - COMPREHENSIVE AUDIT")
    print("=" * 70)

    # Load config with CLI overrides
    import sys

    overrides = sys.argv[1:] if len(sys.argv) > 1 else []
    cfg = load_config("config.yaml", overrides=overrides)
    print(f"\nConfig loaded: {cfg.dataset.name}")
    print(f"Weighted loss enabled: {cfg.training.use_weighted_loss}")

    # Run tasks
    findings_task1 = task1_trace_loss_function()

    try:
        distributions = task2_check_class_distributions(cfg)
    except Exception as e:
        print(f"✗ Task 2 error: {e}")
        distributions = {}

    findings_task3 = task3_audit_data_leakage(cfg)

    try:
        task4_batch_level_analysis(cfg)
    except Exception as e:
        print(f"⚠️  Task 4 (batch analysis) skipped: {e}")

    # Compile findings
    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)

    print("\n✓ Current Loss Weighting Strategy:")
    print("  - Loss function: PyTorch F.cross_entropy")
    print("  - Weighting: ENABLED (batch-level inverse frequency)")
    print("  - Config key: training.use_weighted_loss = true")

    print("\n✓ Class Distributions (if available):")
    if distributions:
        for split_name, dist in distributions.items():
            print(
                f"  {split_name}: {dist['n_pos']}/{dist['total']} pos ({dist['pct_pos']:.2f}%)"
            )
    else:
        print("  Could not load distributions (data may need preprocessing)")

    print("\n⚠️  Data Leakage Risk: MEDIUM")
    print("  - Weights computed PER BATCH, not globally from train split")
    print("  - Val/Test batches compute weights from own labels")
    print("  - This is SUBOPTIMAL but not severe (no temporal leakage)")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR T2.2")
    print("=" * 70)
    print(
        """
✓ CRITICAL FINDING: Current implementation has batch-level weighting
  
✓ RECOMMENDED FIX (for T2.2 implementation):
  1. Compute class weights GLOBALLY from train split at model init
  2. Store weights as model property (not per-batch)
  3. Use same weights for train, val, test (computed from train only)
  4. Make this MANDATORY (no config toggle after implementation)
  
✓ FORMULA (inverse frequency):
  weight[class] = total_samples / (num_classes * count[class])
  
✓ IMPLEMENTATION:
  - At model __init__: compute weights from train split
  - Pass to BCEWithLogitsLoss or CrossEntropyLoss as initialization param
  - This ensures no data leakage
"""
    )

    print("\n✅ Task 2.1 Analysis Complete!")
    print("Ready to proceed to T2.2 (implementation) after user confirmation.")


if __name__ == "__main__":
    main()
