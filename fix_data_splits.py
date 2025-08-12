#!/usr/bin/env python3
"""
Fix data splits to eliminate overlap between train/val/test sets.
"""

import json
import random
import yaml
from collections import Counter
from omegaconf import OmegaConf
from src.data.datasets import get_loader

def fix_data_splits():
    """Fix the data splits to eliminate overlaps."""
    print("🔧 FIXING DATA SPLITS")
    print("=" * 50)
    
    # Load config
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)
    
    # Load raw edges
    print("Loading raw edges...")
    edges = get_loader(cfg.dataset.name)(cfg)
    print(f"Loaded {len(edges):,} edges")
    
    # Sort edges for extra consistency (by node1, node2, label)
    edges_sorted = sorted(edges, key=lambda x: (x[0], x[1], x[2]))
    
    # Convert to tuples and deduplicate deterministically
    def deterministic_dedup(edges):
        """Deterministic deduplication that preserves order."""
        seen = set()
        unique_edges = []
        for edge in edges:
            edge_tuple = tuple(edge)
            if edge_tuple not in seen:
                seen.add(edge_tuple)
                unique_edges.append(edge_tuple)
        return unique_edges
    
    edge_tuples = [tuple(edge) for edge in edges_sorted]
    unique_edges = deterministic_dedup(edges_sorted)
    
    print(f"Unique edges: {len(unique_edges):,}")
    if len(unique_edges) != len(edge_tuples):
        print(f"⚠️ Found {len(edge_tuples) - len(unique_edges)} duplicate edges")
        print(f"✅ Using deterministic deduplication (preserves order)")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Shuffle edges
    shuffled_edges = unique_edges.copy()
    random.shuffle(shuffled_edges)
    
    # Calculate split sizes
    n_total = len(shuffled_edges)
    n_train = int(cfg.dataset.train_ratio * n_total)
    n_mask = int(cfg.dataset.mask_ratio * n_total)
    n_val = int(cfg.dataset.val_ratio * n_total)
    n_test = n_total - n_train - n_mask - n_val
    
    print(f"\nSplit sizes:")
    print(f"  Train: {n_train:,} ({cfg.dataset.train_ratio*100:.1f}%)")
    print(f"  Mask: {n_mask:,} ({cfg.dataset.mask_ratio*100:.1f}%)")
    print(f"  Val: {n_val:,} ({cfg.dataset.val_ratio*100:.1f}%)")
    print(f"  Test: {n_test:,} ({(n_test/n_total)*100:.1f}%)")
    print(f"  Total: {n_train + n_mask + n_val + n_test:,}")
    
    if n_test < 0:
        raise ValueError("Ratios sum to > 1. Adjust train/mask/val ratios.")
    
    # Create non-overlapping splits
    train_edges = shuffled_edges[:n_train]
    mask_edges = shuffled_edges[n_train:n_train + n_mask]
    val_edges = shuffled_edges[n_train + n_mask:n_train + n_mask + n_val]
    test_edges = shuffled_edges[n_train + n_mask + n_val:]
    
    # Convert back to lists for JSON serialization
    splits = {
        "train": [list(edge) for edge in train_edges],
        "mask": [list(edge) for edge in mask_edges],
        "val": [list(edge) for edge in val_edges],
        "test": [list(edge) for edge in test_edges]
    }
    
    # Verify no overlaps
    print(f"\n✅ Verifying no overlaps...")
    train_set = set(train_edges)
    mask_set = set(mask_edges)
    val_set = set(val_edges)
    test_set = set(test_edges)
    
    overlaps = [
        ("train", "mask", len(train_set & mask_set)),
        ("train", "val", len(train_set & val_set)),
        ("train", "test", len(train_set & test_set)),
        ("mask", "val", len(mask_set & val_set)),
        ("mask", "test", len(mask_set & test_set)),
        ("val", "test", len(val_set & test_set)),
    ]
    
    all_clean = True
    for split1, split2, overlap_count in overlaps:
        status = "✅" if overlap_count == 0 else "❌"
        print(f"  {split1} ∩ {split2}: {overlap_count} overlaps {status}")
        if overlap_count > 0:
            all_clean = False
    
    if not all_clean:
        raise ValueError("Still have overlaps after fixing!")
    
    # Check label distribution across splits
    print(f"\n📊 Label distribution across splits:")
    for split_name, split_edges in splits.items():
        labels = [edge[2] for edge in split_edges]
        label_counts = Counter(labels)
        print(f"  {split_name.capitalize()}:")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            pct = 100 * count / len(split_edges)
            print(f"    Label {label}: {count:,} ({pct:.1f}%)")
    
    # Just show the fix without saving
    print(f"\n� Fixed splits ready (not saved yet)")
    print(f"   To apply: replace data/{cfg.dataset.name}/splits.json with fixed version")
    
    return splits

def main():
    """Main function to fix data splits."""
    try:
        splits = fix_data_splits()
        
        print(f"\n" + "=" * 50)
        print("✅ SPLITS ANALYSIS COMPLETED!")
        print("   Showed how to fix overlaps (no files changed)")
        print("=" * 50)
        
        print(f"\n💡 To actually apply the fix:")
        print(f"   1. Uncomment the saving code in fix_data_splits.py")
        print(f"   2. Run the script again to save fixed splits")
        print(f"   3. Delete encoded.pt to force regeneration with clean splits")
        
    except Exception as e:
        print(f"\n❌ Error fixing splits: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
