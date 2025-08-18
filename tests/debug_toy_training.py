#!/usr/bin/env python3
"""
Debug toy dataset to understand the NaN issue.
"""

import torch
import yaml
import json
from omegaconf import OmegaConf
from src.data.prepare_data import prepare_data
from src.model.model import TransformerModel

def debug_toy_training():
    """Debug what's causing NaN in toy training."""
    print("🔍 DEBUGGING TOY DATASET TRAINING")
    print("=" * 50)
    
    # Load config
    with open('config_toy.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)
    
    # Load data
    print("📊 Loading toy data...")
    dataloaders = prepare_data(cfg)
    
    # Get a sample batch
    train_loader = dataloaders['train']
    batch = next(iter(train_loader))
    input_ids, labels, attention_mask = batch
    
    print(f"Batch info:")
    print(f"  Input IDs: {input_ids.shape} {input_ids.dtype}")
    print(f"  Labels: {labels.shape} {labels.dtype}")
    print(f"  Attention: {attention_mask.shape} {attention_mask.dtype}")
    
    # Check for extreme values
    print(f"\nValue ranges:")
    print(f"  Input IDs: [{input_ids.min()}, {input_ids.max()}]")
    print(f"  Labels: [{labels.min()}, {labels.max()}]")
    print(f"  Attention: [{attention_mask.min()}, {attention_mask.max()}]")
    
    # Check label distribution
    valid_labels = labels[labels != cfg.model.ignore_index]
    if len(valid_labels) > 0:
        label_counts = torch.bincount(valid_labels, minlength=cfg.model.num_classes)
        print(f"  Valid labels in batch: {len(valid_labels)} / {labels.numel()}")
        print(f"  Label distribution: {label_counts.tolist()}")
    else:
        print(f"  ⚠️ No valid labels in this batch!")
    
    # Create model and test forward pass
    print(f"\n🤖 Testing model forward pass...")
    model = TransformerModel(cfg)
    model.eval()
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
        print(f"  Logits mean: {logits.mean():.4f}")
        print(f"  Logits std: {logits.std():.4f}")
        
        # Check for NaN/inf
        has_nan = torch.isnan(logits).any()
        has_inf = torch.isinf(logits).any()
        print(f"  Contains NaN: {has_nan}")
        print(f"  Contains Inf: {has_inf}")
        
        # Test loss computation
        print(f"\n💢 Testing loss computation...")
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=cfg.model.ignore_index)
        
        # Reshape for loss
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        
        print(f"  Flattened logits: {logits_flat.shape}")
        print(f"  Flattened labels: {labels_flat.shape}")
        
        # Compute loss
        loss = loss_fn(logits_flat, labels_flat)
        print(f"  Loss value: {loss.item()}")
        print(f"  Loss is NaN: {torch.isnan(loss)}")
        print(f"  Loss is finite: {torch.isfinite(loss)}")
        
        # Check valid positions for loss
        valid_mask = labels_flat != cfg.model.ignore_index
        valid_count = valid_mask.sum().item()
        print(f"  Valid positions for loss: {valid_count} / {len(labels_flat)}")
        
        if valid_count > 0:
            valid_logits = logits_flat[valid_mask]
            valid_labels = labels_flat[valid_mask]
            print(f"  Valid logits range: [{valid_logits.min():.4f}, {valid_logits.max():.4f}]")
            print(f"  Valid labels range: [{valid_labels.min()}, {valid_labels.max()}]")
            
            # Test loss on valid positions only
            valid_loss = loss_fn(valid_logits, valid_labels)
            print(f"  Loss on valid only: {valid_loss.item()}")

def check_toy_data_stats():
    """Check toy dataset statistics."""
    print(f"\n📈 TOY DATASET STATISTICS")
    print("=" * 50)
    
    # Load meta
    with open('data/toy/meta.json', 'r') as f:
        meta = json.load(f)
    
    print(f"Meta info: {meta}")
    
    # Load encoded data
    encoded_data = torch.load('data/toy/encoded.pt')
    train_pack, val_pack, test_pack = encoded_data
    
    train_input, train_labels, train_mask = train_pack
    
    print(f"\nData shapes:")
    print(f"  Train input: {train_input.shape}")
    print(f"  Train labels: {train_labels.shape}")
    print(f"  Train mask: {train_mask.shape}")
    
    # Check sparsity
    total_positions = train_labels.numel()
    valid_labels = (train_labels != meta['ignore_index']).sum().item()
    attention_positions = train_mask.sum().item()
    
    print(f"\nSparsity analysis:")
    print(f"  Total positions: {total_positions}")
    print(f"  Valid labels: {valid_labels} ({100*valid_labels/total_positions:.1f}%)")
    print(f"  Attended positions: {attention_positions} ({100*attention_positions/total_positions:.1f}%)")
    
    # Check sequence lengths
    seq_lengths = train_mask.sum(dim=1)
    print(f"\nSequence lengths:")
    print(f"  Min: {seq_lengths.min().item()}")
    print(f"  Max: {seq_lengths.max().item()}")
    print(f"  Mean: {seq_lengths.float().mean().item():.1f}")

def main():
    """Run toy dataset debugging."""
    check_toy_data_stats()
    debug_toy_training()
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Use config_toy_stable.yaml with lower LR")
    print(f"   2. Try: python run.py --config config_toy_stable.yaml")
    print(f"   3. Monitor for gradient explosion during training")

if __name__ == "__main__":
    main()
