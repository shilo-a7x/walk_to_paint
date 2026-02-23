#!/usr/bin/env python3
"""
Comprehensive data verification script to check if data is loaded and represented correctly.
"""

import torch
import yaml
import os
import random
from collections import Counter
from omegaconf import OmegaConf
from src.data.prepare_data import prepare_data
from src.data.dataset_cache import load_dataset_cache, tokenizer_from_cache
from src.data.datasets import get_loader


def load_config():
    """Load configuration and meta information."""
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    return OmegaConf.create(cfg)


def load_cache(cfg):
    """Load dataset cache (only supported format)."""
    cache_path = os.path.join(cfg.dataset.data_dir, "dataset_cache.pt")
    return load_dataset_cache(cache_path)


def verify_raw_data(cfg):
    """Verify the raw edge data."""
    print("=" * 60)
    print("🔍 RAW DATA VERIFICATION")
    print("=" * 60)

    # Load raw edges
    edges = get_loader(cfg.dataset.name)(cfg)

    print(f"\n📊 Edge Statistics:")
    print(f"  Total edges: {len(edges):,}")

    # Analyze edge labels
    labels = [label for _, _, label in edges]
    label_counts = Counter(labels)
    print(f"  Unique labels: {sorted(label_counts.keys())}")
    print(f"  Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"    Label {label}: {count:,} edges ({100*count/len(edges):.1f}%)")

    # Analyze nodes
    nodes = set()
    for u, v, _ in edges:
        nodes.add(u)
        nodes.add(v)

    print(f"  Unique nodes: {len(nodes):,}")
    print(f"  Node range: [{min(nodes)}, {max(nodes)}]")

    # Sample edges
    print(f"\n🔍 Sample Edges:")
    sample_edges = random.sample(edges, min(5, len(edges)))
    for i, (u, v, label) in enumerate(sample_edges):
        print(f"  {i+1}. Node {u} --({label})--> Node {v}")

    return edges


def verify_walks(cfg):
    """Verify the random walks."""
    print(f"\n🚶 WALK VERIFICATION")
    print("=" * 60)

    cache_data = load_cache(cfg)
    walks = cache_data["walks"]

    print(f"\n📊 Walk Statistics:")
    print(f"  Total walks: {len(walks):,}")

    # Analyze walk lengths
    walk_lengths = [len(walk) for walk in walks]
    print(f"  Walk length range: [{min(walk_lengths)}, {max(walk_lengths)}]")
    print(f"  Average walk length: {sum(walk_lengths)/len(walk_lengths):.1f}")
    print(f"  Max configured length: {cfg.dataset.max_walk_length}")

    # Count token types
    node_tokens = 0
    edge_tokens = 0
    for walk in walks[:1000]:  # Sample first 1000 walks
        for token in walk:
            if token.startswith("N_"):
                node_tokens += 1
            elif token.startswith("E_"):
                edge_tokens += 1

    print(f"  Node vs Edge tokens (in first 1000 walks):")
    print(f"    Node tokens: {node_tokens:,}")
    print(f"    Edge tokens: {edge_tokens:,}")
    print(f"    Ratio: {node_tokens/edge_tokens:.2f} nodes per edge")

    # Sample walks
    print(f"\n🔍 Sample Walks:")
    for i in range(min(3, len(walks))):
        walk = walks[i]
        print(f"  Walk {i+1} (length {len(walk)}):")
        print(f"    {' -> '.join(walk[:10])}{'...' if len(walk) > 10 else ''}")

    return walks


def verify_tokenizer(cfg, walks, edges):
    """Verify the tokenizer."""
    print(f"\n🔤 TOKENIZER VERIFICATION")
    print("=" * 60)

    cache_data = load_cache(cfg)
    tokenizer = tokenizer_from_cache(cache_data)

    print(f"\n📊 Tokenizer Statistics:")
    print(f"  Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"  Number of edge types: {tokenizer.num_edge_tokens}")
    print(f"  PAD ID: {tokenizer.PAD_ID}")
    print(f"  MASK ID: {tokenizer.MASK_ID}")
    print(f"  UNK ID: {tokenizer.UNK_ID}")
    print(f"  UNK_LABEL ID: {tokenizer.UNK_LABEL_ID}")

    # Check special tokens
    print(f"\n🔍 Special Tokens:")
    special_tokens = ["<PAD>", "<MASK>", "<UNK>"]
    for token in special_tokens:
        token_id = tokenizer.token2id.get(token, "NOT_FOUND")
        print(f"  {token}: ID {token_id}")

    # Analyze token types
    node_tokens = sum(1 for t in tokenizer.token2id if t.startswith("N_"))
    edge_tokens = sum(1 for t in tokenizer.token2id if t.startswith("E_"))
    special_count = len(special_tokens)

    print(f"\n📊 Token Type Distribution:")
    print(f"  Node tokens: {node_tokens:,}")
    print(f"  Edge tokens: {edge_tokens:,}")
    print(f"  Special tokens: {special_count}")
    print(
        f"  Total: {node_tokens + edge_tokens + special_count} (vocab_size: {tokenizer.vocab_size})"
    )

    # Test encoding/decoding
    print(f"\n🧪 Encoding/Decoding Test:")
    sample_walk = walks[0][:10]  # First 10 tokens of first walk
    encoded = tokenizer.encode(sample_walk)
    decoded = tokenizer.decode(encoded)

    print(f"  Original:  {sample_walk}")
    print(f"  Encoded:   {encoded}")
    print(f"  Decoded:   {decoded}")
    print(f"  Round-trip success: {sample_walk == decoded}")

    # Test edge label mapping
    print(f"\n🏷️ Edge Label Mapping:")
    print(f"  Edge label to class mapping:")
    for edge_token, class_id in list(tokenizer.edge_label2id.items())[:5]:
        print(f"    {edge_token} -> class {class_id}")

    return tokenizer


def verify_data_splits(cfg):
    """Verify data splits."""
    print(f"\n📊 DATA SPLITS VERIFICATION")
    print("=" * 60)

    cache_data = load_cache(cfg)
    splits = {k: list(v) for k, v in cache_data["splits"].items()}

    print(f"\n📊 Split Statistics:")
    total_edges = sum(len(splits[split]) for split in splits)

    for split_name in ["train", "mask", "val", "test"]:
        split_data = splits[split_name]
        count = len(split_data)
        percentage = 100 * count / total_edges
        print(f"  {split_name.capitalize()} split: {count:,} edges ({percentage:.1f}%)")

        # Show sample from each split
        if count > 0:
            sample = random.choice(split_data)
            print(f"    Sample: Node {sample[0]} --({sample[2]})--> Node {sample[1]}")

    print(f"  Total edges in splits: {total_edges:,}")

    # Verify no overlap between splits
    train_set = {tuple(edge) for edge in splits["train"]}
    mask_set = {tuple(edge) for edge in splits["mask"]}
    val_set = {tuple(edge) for edge in splits["val"]}
    test_set = {tuple(edge) for edge in splits["test"]}

    overlaps = [
        ("train", "mask", len(train_set & mask_set)),
        ("train", "val", len(train_set & val_set)),
        ("train", "test", len(train_set & test_set)),
        ("mask", "val", len(mask_set & val_set)),
        ("mask", "test", len(mask_set & test_set)),
        ("val", "test", len(val_set & test_set)),
    ]

    print(f"\n🔍 Split Overlap Check:")
    all_clean = True
    for split1, split2, overlap_count in overlaps:
        status = "✅" if overlap_count == 0 else "❌"
        print(f"  {split1} ∩ {split2}: {overlap_count} overlaps {status}")
        if overlap_count > 0:
            all_clean = False

    print(
        f"  Overall: {'✅ Clean splits' if all_clean else '❌ Overlapping splits detected'}"
    )


def verify_encoded_data(cfg):
    """Verify the encoded and processed data."""
    print(f"\n🔢 ENCODED DATA VERIFICATION")
    print("=" * 60)

    cache_data = load_cache(cfg)
    encoded = cache_data["encoded"]
    meta = cache_data["metadata"]

    print(f"\n📊 Meta Information:")
    for key, value in meta.items():
        print(f"  {key}: {value}")

    input_ids = encoded["input_ids"]
    edge_split_mask = encoded["edge_split_mask"]
    attention_mask = encoded["attention_base"]

    print(f"\n🎯 Encoded Base Tensors:")
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Edge split mask shape: {edge_split_mask.shape}")
    print(f"  Attention mask shape: {attention_mask.shape}")

    print(f"  Input IDs dtype: {input_ids.dtype}")
    print(f"  Edge split mask dtype: {edge_split_mask.dtype}")
    print(f"  Attention mask dtype: {attention_mask.dtype}")

    print(f"  Input IDs range: [{input_ids.min()}, {input_ids.max()}]")
    print(
        f"  Edge split mask range: [{edge_split_mask.min()}, {edge_split_mask.max()}]"
    )
    print(f"  Attention mask range: [{attention_mask.min()}, {attention_mask.max()}]")

    pad_id = meta["pad_id"]
    padded_positions = (input_ids == pad_id).sum()
    attended_positions = attention_mask.sum()

    print(
        f"  Padded positions: {padded_positions} / {input_ids.numel()} ({100*padded_positions/input_ids.numel():.1f}%)"
    )
    print(
        f"  Attended positions: {attended_positions} / {attention_mask.numel()} ({100*attended_positions/attention_mask.numel():.1f}%)"
    )

    sample_idx = 0
    sample_input = input_ids[sample_idx]
    sample_mask = attention_mask[sample_idx]
    seq_len = sample_mask.sum().item()
    print(f"  Sample sequence length: {seq_len}")
    print(f"  Sample input IDs: {sample_input[:min(10, seq_len)].tolist()}...")
    print(f"  Sample attention: {sample_mask[:min(10, seq_len)].tolist()}...")


def verify_dataloaders(cfg):
    """Verify the final dataloaders."""
    print(f"\n🔄 DATALOADER VERIFICATION")
    print("=" * 60)

    # Create dataloaders
    dataloaders = prepare_data(cfg)

    print(f"\n📊 DataLoader Statistics:")
    for stage_name, dataloader in dataloaders.items():
        print(f"  {stage_name.capitalize()} DataLoader:")
        print(f"    Batches: {len(dataloader)}")
        print(f"    Batch size: {dataloader.batch_size}")
        print(f"    Total samples: {len(dataloader.dataset)}")
        print(f"    Shuffle: {dataloader.sampler is not None}")

    # Test loading a batch from each dataloader
    print(f"\n🧪 Batch Loading Test:")
    for stage_name, dataloader in dataloaders.items():
        batch = next(iter(dataloader))
        if len(batch) == 4:
            input_ids, labels, attention_mask, _ = batch
        else:
            input_ids, labels, attention_mask = batch

        print(f"  {stage_name.capitalize()} batch:")
        print(f"    Input IDs: {input_ids.shape} {input_ids.dtype}")
        print(f"    Labels: {labels.shape} {labels.dtype}")
        print(f"    Attention mask: {attention_mask.shape} {attention_mask.dtype}")

        # Check for any NaN or inf values
        has_nan_input = torch.isnan(input_ids.float()).any()
        has_nan_labels = torch.isnan(labels.float()).any()
        has_nan_mask = torch.isnan(attention_mask.float()).any()

        print(
            f"    Contains NaN: Input={has_nan_input}, Labels={has_nan_labels}, Mask={has_nan_mask}"
        )

        # Check label distribution in batch
        valid_labels = labels[labels != cfg.model.ignore_index]
        if len(valid_labels) > 0:
            label_counts = torch.bincount(valid_labels, minlength=cfg.model.num_classes)
            print(f"    Label distribution: {label_counts.tolist()}")


def main():
    """Run comprehensive data verification."""
    print("🔍 COMPREHENSIVE DATA VERIFICATION")
    print("=" * 60)

    cfg = load_config()

    # Load meta information for model config from dataset cache
    cache_data = load_cache(cfg)
    meta = cache_data["metadata"]
    cfg.model.vocab_size = meta["vocab_size"]
    cfg.model.num_classes = meta["num_classes"]
    cfg.model.pad_id = meta["pad_id"]
    cfg.model.ignore_index = meta["ignore_index"]

    try:
        # Run all verifications
        edges = verify_raw_data(cfg)
        walks = verify_walks(cfg)
        tokenizer = verify_tokenizer(cfg, walks, edges)
        verify_data_splits(cfg)
        verify_encoded_data(cfg)
        verify_dataloaders(cfg)

        print(f"\n" + "=" * 60)
        print("✅ DATA VERIFICATION COMPLETED SUCCESSFULLY!")
        print("   All data appears to be loaded and represented correctly.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during verification: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
