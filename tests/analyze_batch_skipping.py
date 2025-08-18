#!/usr/bin/env python3
"""
Analyze toy dataset to check how many batches would be skipped.
"""

import torch
import yaml
import json
from omegaconf import OmegaConf
from src.data.prepare_data import prepare_data

def analyze_batch_skipping():
    """Analyze how many batches would be skipped in toy dataset."""
    print("🔍 ANALYZING BATCH SKIPPING IN TOY DATASET")
    print("=" * 60)
    
    # Load config
    with open('config_toy.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)
    
    # Load meta info
    with open('data/toy/meta.json', 'r') as f:
        meta = json.load(f)
    
    cfg.model.vocab_size = meta['vocab_size']
    cfg.model.num_classes = meta['num_classes']
    cfg.model.pad_id = meta['pad_id']
    cfg.model.ignore_index = meta['ignore_index']
    
    print(f"Dataset config:")
    print(f"  Ignore index: {cfg.model.ignore_index}")
    print(f"  Batch size: {cfg.training.batch_size}")
    print(f"  Max walk length: {cfg.dataset.max_walk_length}")
    print(f"  Number of walks: {cfg.dataset.num_walks}")
    
    # Load data
    print(f"\n📊 Loading toy dataset...")
    dataloaders = prepare_data(cfg)
    
    # Analyze each split
    for split_name, dataloader in dataloaders.items():
        print(f"\n🎯 Analyzing {split_name.upper()} split:")
        print(f"  Total batches: {len(dataloader)}")
        
        valid_batches = 0
        skipped_batches = 0
        total_valid_labels = 0
        total_positions = 0
        
        valid_labels_per_batch = []
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids, labels, attention_mask = batch
            
            # Count valid labels in this batch
            labels_flat = labels.view(-1)
            valid_mask = labels_flat != cfg.model.ignore_index
            num_valid = valid_mask.sum().item()
            total_positions += labels_flat.numel()
            
            if num_valid == 0:
                skipped_batches += 1
            else:
                valid_batches += 1
                total_valid_labels += num_valid
                valid_labels_per_batch.append(num_valid)
        
        # Calculate statistics
        skip_rate = 100 * skipped_batches / len(dataloader)
        valid_rate = 100 * total_valid_labels / total_positions
        
        print(f"  Valid batches: {valid_batches}")
        print(f"  Skipped batches: {skipped_batches}")
        print(f"  Skip rate: {skip_rate:.1f}%")
        print(f"  Total valid labels: {total_valid_labels:,} / {total_positions:,}")
        print(f"  Valid label rate: {valid_rate:.1f}%")
        
        if valid_labels_per_batch:
            avg_valid = sum(valid_labels_per_batch) / len(valid_labels_per_batch)
            min_valid = min(valid_labels_per_batch)
            max_valid = max(valid_labels_per_batch)
            print(f"  Valid labels per batch: min={min_valid}, max={max_valid}, avg={avg_valid:.1f}")
        
        # Show sample batches
        if split_name == 'train':
            print(f"\n  📋 Sample batch analysis:")
            sample_batch = next(iter(dataloader))
            input_ids, labels, attention_mask = sample_batch
            
            print(f"    Batch shape: {labels.shape}")
            print(f"    Sample labels: {labels[0]}")
            valid_positions = (labels[0] != cfg.model.ignore_index).nonzero().flatten()
            print(f"    Valid positions in first sequence: {valid_positions.tolist()}")
            
            # Check sequence lengths
            seq_lengths = attention_mask.sum(dim=1)
            print(f"    Sequence lengths: {seq_lengths.tolist()}")

def analyze_data_sparsity():
    """Analyze the underlying data sparsity."""
    print(f"\n🔬 ANALYZING DATA SPARSITY")
    print("=" * 60)
    
    # Load encoded data directly
    encoded_data = torch.load('data/toy/encoded.pt')
    train_pack, val_pack, test_pack = encoded_data
    
    # Load meta
    with open('data/toy/meta.json', 'r') as f:
        meta = json.load(f)
    
    splits = [
        ("Train", train_pack),
        ("Val", val_pack),
        ("Test", test_pack)
    ]
    
    for split_name, (input_ids, labels, attention_mask) in splits:
        print(f"\n{split_name} Split Analysis:")
        
        total_positions = labels.numel()
        attended_positions = attention_mask.sum().item()
        valid_labels = (labels != meta['ignore_index']).sum().item()
        padded_positions = (input_ids == meta['pad_id']).sum().item()
        
        print(f"  Total positions: {total_positions:,}")
        print(f"  Attended positions: {attended_positions:,} ({100*attended_positions/total_positions:.1f}%)")
        print(f"  Valid labels: {valid_labels:,} ({100*valid_labels/total_positions:.1f}%)")
        print(f"  Padded positions: {padded_positions:,} ({100*padded_positions/total_positions:.1f}%)")
        
        # Analyze per-sequence sparsity
        num_sequences = labels.shape[0]
        sequences_with_labels = 0
        labels_per_sequence = []
        
        for seq_idx in range(num_sequences):
            seq_labels = labels[seq_idx]
            seq_valid = (seq_labels != meta['ignore_index']).sum().item()
            if seq_valid > 0:
                sequences_with_labels += 1
                labels_per_sequence.append(seq_valid)
        
        print(f"  Sequences with valid labels: {sequences_with_labels:,} / {num_sequences:,} ({100*sequences_with_labels/num_sequences:.1f}%)")
        
        if labels_per_sequence:
            avg_labels = sum(labels_per_sequence) / len(labels_per_sequence)
            min_labels = min(labels_per_sequence)
            max_labels = max(labels_per_sequence)
            print(f"  Labels per valid sequence: min={min_labels}, max={max_labels}, avg={avg_labels:.1f}")

def suggest_improvements():
    """Suggest improvements to reduce batch skipping."""
    print(f"\n💡 SUGGESTIONS TO REDUCE BATCH SKIPPING")
    print("=" * 60)
    
    print(f"1️⃣ Increase number of walks:")
    print(f"   - Current: 1500 walks")
    print(f"   - Try: 5000+ walks for better coverage")
    
    print(f"\n2️⃣ Increase walk length:")
    print(f"   - Current: max_walk_length = 3 (7 tokens)")
    print(f"   - Try: max_walk_length = 5 (11 tokens)")
    
    print(f"\n3️⃣ Larger batch size:")
    print(f"   - Current: batch_size = 2")
    print(f"   - Try: batch_size = 8 (more likely to have valid labels)")
    
    print(f"\n4️⃣ Use dynamic batching:")
    print(f"   - Group sequences with similar valid label counts")
    print(f"   - Ensure each batch has at least some valid labels")
    
    print(f"\n5️⃣ Create more balanced toy dataset:")
    print(f"   - Ensure more edges in the toy graph")
    print(f"   - Use longer random walks")

def main():
    """Run complete batch skipping analysis."""
    analyze_batch_skipping()
    analyze_data_sparsity()
    suggest_improvements()
    
    print(f"\n" + "=" * 60)
    print("🔍 BATCH SKIPPING ANALYSIS COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
