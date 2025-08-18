#!/usr/bin/env python3
"""
Focused analysis of data issues found during verification.
"""

import torch
import yaml
import json
import os
from collections import Counter
from omegaconf import OmegaConf


def analyze_data_issues():
    """Analyze specific data issues identified."""
    print("🚨 DATA ISSUE ANALYSIS")
    print("=" * 60)

    # Load config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)

    # Load meta
    with open("data/chess/meta.json", "r") as f:
        meta = json.load(f)

    print(f"\n1️⃣ WALK LENGTH ISSUE")
    print("-" * 30)
    print(f"   Configured max_walk_length: {cfg.dataset.max_walk_length}")
    print(f"   Actual max walk length found: 41")
    print(f"   ❌ Walks exceed configured maximum!")
    print(f"   💡 This could cause memory issues and unexpected behavior")

    print(f"\n2️⃣ DATA SPLIT OVERLAP ISSUE")
    print("-" * 30)

    # Load splits to analyze overlap
    with open("data/chess/splits.json", "r") as f:
        splits = json.load(f)

    train_set = {tuple(edge) for edge in splits["train"]}
    mask_set = {tuple(edge) for edge in splits["mask"]}
    val_set = {tuple(edge) for edge in splits["val"]}
    test_set = {tuple(edge) for edge in splits["test"]}

    overlaps = {
        "train_mask": len(train_set & mask_set),
        "train_val": len(train_set & val_set),
        "train_test": len(train_set & test_set),
        "mask_val": len(mask_set & val_set),
        "mask_test": len(mask_set & test_set),
        "val_test": len(val_set & test_set),
    }

    print(f"   Split overlaps detected:")
    for split_pair, count in overlaps.items():
        if count > 0:
            print(f"     {split_pair}: {count} overlapping edges ❌")

    print(f"   💡 This causes data leakage between train/val/test sets!")

    print(f"\n3️⃣ SPARSE LABELING ISSUE")
    print("-" * 30)

    # Load a sample of encoded data to check labeling
    encoded_data = torch.load("data/chess/encoded.pt")
    train_pack, val_pack, test_pack = encoded_data

    train_input, train_labels, train_mask = train_pack

    # Analyze label sparsity
    total_positions = train_labels.numel()
    valid_labels = (train_labels != meta["ignore_index"]).sum().item()
    sparsity = 100 * (1 - valid_labels / total_positions)

    print(f"   Total positions: {total_positions:,}")
    print(f"   Valid labels: {valid_labels:,}")
    print(f"   Sparsity: {sparsity:.1f}% positions have ignore_index")
    print(
        f"   💡 Very sparse labeling - only {100-sparsity:.1f}% positions have targets"
    )

    # Check label distribution in valid positions
    valid_label_positions = train_labels[train_labels != meta["ignore_index"]]
    if len(valid_label_positions) > 0:
        label_counts = torch.bincount(
            valid_label_positions, minlength=meta["num_classes"]
        )
        print(f"   Valid label distribution: {label_counts.tolist()}")

        # Check if distribution is balanced
        min_count = label_counts.min().item()
        max_count = label_counts.max().item()
        imbalance_ratio = max_count / max(min_count, 1)
        print(f"   Class imbalance ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 2:
            print(f"   ⚠️ Significant class imbalance detected!")

    print(f"\n4️⃣ ATTENTION MASK ANALYSIS")
    print("-" * 30)

    # Check attention patterns
    attended_positions = train_mask.sum().item()
    attention_rate = 100 * attended_positions / total_positions

    print(f"   Attended positions: {attended_positions:,} / {total_positions:,}")
    print(f"   Attention rate: {attention_rate:.1f}%")

    # Check if masking aligns with labeling
    attended_but_ignored = (
        ((train_mask == 1) & (train_labels == meta["ignore_index"])).sum().item()
    )
    print(f"   Attended but ignored positions: {attended_but_ignored:,}")
    print(f"   💡 These are node positions that are attended but not labeled")

    print(f"\n5️⃣ SEQUENCE LENGTH ANALYSIS")
    print("-" * 30)

    # Analyze actual sequence lengths
    seq_lengths = train_mask.sum(dim=1)  # Length of each sequence
    print(f"   Sequence length statistics:")
    print(f"     Min: {seq_lengths.min().item()}")
    print(f"     Max: {seq_lengths.max().item()}")
    print(f"     Mean: {seq_lengths.float().mean().item():.1f}")
    print(f"     Median: {seq_lengths.median().item()}")

    # Check if we're hitting the configured max length often
    max_configured = 2 * cfg.dataset.max_walk_length + 1  # From model config
    long_sequences = (seq_lengths >= max_configured).sum().item()
    print(f"   Sequences at/above model max ({max_configured}): {long_sequences:,}")

    print(f"\n6️⃣ TOKEN DISTRIBUTION ANALYSIS")
    print("-" * 30)

    # Analyze token usage in input
    unique_tokens, token_counts = torch.unique(train_input, return_counts=True)

    # Separate by token type
    pad_count = (
        token_counts[unique_tokens == meta["pad_id"]].item()
        if meta["pad_id"] in unique_tokens
        else 0
    )
    mask_count = (
        token_counts[unique_tokens == 1].item() if 1 in unique_tokens else 0
    )  # MASK token

    print(f"   PAD token usage: {pad_count:,} ({100*pad_count/total_positions:.1f}%)")
    print(
        f"   MASK token usage: {mask_count:,} ({100*mask_count/total_positions:.1f}%)"
    )

    # Count node vs edge tokens (excluding special tokens)
    special_tokens = {0, 1, 2}  # PAD, MASK, UNK
    content_tokens = unique_tokens[
        ~torch.isin(unique_tokens, torch.tensor(list(special_tokens)))
    ]
    content_counts = token_counts[
        ~torch.isin(unique_tokens, torch.tensor(list(special_tokens)))
    ]

    if len(content_tokens) > 0:
        total_content = content_counts.sum().item()
        print(
            f"   Content tokens: {len(content_tokens)} unique types, {total_content:,} total usage"
        )


def suggest_fixes():
    """Suggest potential fixes for identified issues."""
    print(f"\n💡 SUGGESTED FIXES")
    print("=" * 60)

    print(f"\n1️⃣ Fix Walk Length Issue:")
    print(f"   - Update config max_walk_length to 41 (or higher)")
    print(f"   - Or modify walk sampling to respect the configured limit")
    print(f"   - Consider if 41-token sequences are necessary or can be truncated")

    print(f"\n2️⃣ Fix Data Splits:")
    print(f"   - Regenerate splits with proper random seed control")
    print(f"   - Ensure no overlap by using set operations during splitting")
    print(f"   - Consider stratified splitting to maintain label distribution")

    print(f"\n3️⃣ Address Sparse Labeling:")
    print(f"   - This might be intentional (only edge positions are labeled)")
    print(f"   - Verify that the training loss properly handles ignore_index")
    print(f"   - Consider if node positions should also have labels")

    print(f"\n4️⃣ Optimize Memory Usage:")
    print(f"   - With 35% padding, consider dynamic batching")
    print(f"   - Implement sequence packing to reduce memory waste")
    print(f"   - Use smaller batch sizes if memory is limited")

    print(f"\n5️⃣ Validate Model Configuration:")
    print(f"   - Ensure model's max_length matches data max_length")
    print(f"   - Verify positional encoding covers full sequence length")
    print(f"   - Check if attention patterns make sense for the task")


def main():
    """Run focused analysis of data issues."""
    analyze_data_issues()
    suggest_fixes()

    print(f"\n" + "=" * 60)
    print("🔍 FOCUSED ANALYSIS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
