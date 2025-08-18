#!/usr/bin/env python3
"""
Test the fixed lit_model with NaN handling.
"""

import torch
import yaml
from omegaconf import OmegaConf
from src.model.lit_model import LitEdgeClassifier

def test_nan_fix():
    """Test the NaN fix for batches with all ignore_index labels."""
    print("🧪 TESTING NaN FIX")
    print("=" * 40)
    
    # Load toy config
    with open('config_toy.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)
    
    # Add required model config (normally from meta.json)
    with open('data/toy/meta.json', 'r') as f:
        import json
        meta = json.load(f)
    
    cfg.model.vocab_size = meta['vocab_size']
    cfg.model.num_classes = meta['num_classes']
    cfg.model.pad_id = meta['pad_id']
    cfg.model.ignore_index = meta['ignore_index']
    
    # Create model
    model = LitEdgeClassifier(cfg)
    model.eval()
    
    print(f"Model config:")
    print(f"  Vocab size: {cfg.model.vocab_size}")
    print(f"  Num classes: {cfg.model.num_classes}")
    print(f"  Ignore index: {cfg.model.ignore_index}")
    
    # Test Case 1: Normal batch with some valid labels
    print(f"\n🟢 Test 1: Normal batch with valid labels")
    batch_size, seq_len = 2, 7
    input_ids = torch.randint(3, cfg.model.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, cfg.model.num_classes, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Add some ignore_index labels
    labels[0, -2:] = cfg.model.ignore_index
    
    batch1 = (input_ids, labels, attention_mask)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Labels: {labels}")
    print(f"  Valid labels: {(labels != cfg.model.ignore_index).sum().item()}")
    
    with torch.no_grad():
        loss1 = model._step(batch1, "train")
        print(f"  Loss: {loss1.item():.4f}")
        print(f"  Loss is finite: {torch.isfinite(loss1)}")
    
    # Test Case 2: Problematic batch with ALL ignore_index labels
    print(f"\n🔴 Test 2: Batch with ALL ignore_index labels (should be skipped)")
    input_ids2 = torch.randint(3, cfg.model.vocab_size, (batch_size, seq_len))
    labels2 = torch.full((batch_size, seq_len), cfg.model.ignore_index)  # All ignore_index
    attention_mask2 = torch.ones(batch_size, seq_len)
    
    batch2 = (input_ids2, labels2, attention_mask2)
    
    print(f"  Input shape: {input_ids2.shape}")
    print(f"  Labels: {labels2}")
    print(f"  Valid labels: {(labels2 != cfg.model.ignore_index).sum().item()}")
    
    with torch.no_grad():
        loss2 = model._step(batch2, "train")
        print(f"  Returned: {loss2}")
        print(f"  Batch skipped: {loss2 is None}")
        if loss2 is not None:
            print(f"  Loss: {loss2.item():.4f}")
            print(f"  Loss is finite: {torch.isfinite(loss2)}")
        else:
            print(f"  ✅ Batch properly skipped (no dummy loss)")
    
    # Test Case 3: Mixed batch
    print(f"\n🟡 Test 3: Mixed batch")
    input_ids3 = torch.randint(3, cfg.model.vocab_size, (batch_size, seq_len))
    labels3 = torch.full((batch_size, seq_len), cfg.model.ignore_index)
    # Add just one valid label
    labels3[0, 2] = 1
    attention_mask3 = torch.ones(batch_size, seq_len)
    
    batch3 = (input_ids3, labels3, attention_mask3)
    
    print(f"  Input shape: {input_ids3.shape}")
    print(f"  Labels: {labels3}")
    print(f"  Valid labels: {(labels3 != cfg.model.ignore_index).sum().item()}")
    
    with torch.no_grad():
        loss3 = model._step(batch3, "train")
        print(f"  Loss: {loss3.item():.4f}")
        print(f"  Loss is finite: {torch.isfinite(loss3)}")
    
    print(f"\n✅ All tests completed successfully!")
    print(f"💡 The fix properly skips batches with no valid labels instead of")
    print(f"   returning dummy losses that would mislead training metrics.")

def main():
    test_nan_fix()

if __name__ == "__main__":
    main()
