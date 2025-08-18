#!/usr/bin/env python3
"""
Fix the critical bug in _stage_views_from_base where target positions are not attended.

BUG: Target edges are not included in allowed_edges, so they get masked out in attention.
FIX: Target edges should ALWAYS be attended, even if they're not in the allowed splits.
"""

import torch
import os
import sys
sys.path.append('/home/dsi/shilo_avital/yolo_lab/walk_to_paint/src')

from src.data.prepare_data import SplitID
from src.data.tokenizer import Tokenizer

def fixed_stage_views_from_base(
    input_ids: torch.Tensor, edge_split_mask: torch.Tensor, tokenizer: Tokenizer
):
    """
    FIXED VERSION: Build per-stage (input_ids, labels, attention_mask) with those rules:
      Train: allowed = TRAIN; target = MASK
      Val:   allowed = TRAIN|MASK; target = VAL  
      Test:  allowed = TRAIN|MASK|VAL; target = TEST
    
    KEY FIX: Target edges are ALWAYS attended, even if not in allowed splits.
    """
    ignore_index = tokenizer.UNK_LABEL_ID
    pad_id = int(tokenizer.PAD_ID)
    mask_id = int(tokenizer.MASK_ID)

    # base attention (nodes + all edges visible initially)
    base_attn = (input_ids != pad_id).long()
    is_edge_pos = edge_split_mask != SplitID.BAD
    
    # map token_id -> class_id (for labels)
    id2class = torch.full((tokenizer.vocab_size,), ignore_index, dtype=torch.long)
    for class_id, edge_tok in tokenizer.id2edge_label.items():
        tok_id = tokenizer.token2id.get(edge_tok, None)
        if tok_id is not None:
            id2class[tok_id] = int(class_id)

    def build_for_stage(allowed_splits, target_split: int):
        # attention: start from base and zero-out disallowed edges
        attn = base_attn.clone()
        
        # allowed edges from splits
        allowed_edges = torch.zeros_like(edge_split_mask, dtype=torch.bool)
        for s in allowed_splits:
            allowed_edges |= edge_split_mask == s
            
        # target edges (these should ALWAYS be attended!)
        target_edges = edge_split_mask == target_split
        
        # FIX: Include target edges in allowed edges for attention
        allowed_edges_with_targets = allowed_edges | target_edges
        
        # disallowed edges are those that are edge positions but not allowed AND not targets
        disallowed_edges = is_edge_pos & (~allowed_edges_with_targets)
        attn[disallowed_edges] = 0

        # input ids: mask only target edges; pad disallowed edges  
        x = input_ids.clone()
        
        # labels from original token ids (before overwrite)
        labels = torch.full_like(input_ids, ignore_index)
        if target_edges.any():
            labels[target_edges] = id2class[x[target_edges]]
            
        # replace target edges with MASK token id
        x[target_edges] = mask_id
        
        # hide disallowed edges completely (but NOT targets!)
        x[disallowed_edges] = pad_id

        return x, labels, attn

    # Stage definitions
    train_allowed = [SplitID.TRAIN]
    val_allowed = [SplitID.TRAIN, SplitID.MASK]
    test_allowed = [SplitID.TRAIN, SplitID.MASK, SplitID.VAL]

    train_x, train_y, train_attn = build_for_stage(train_allowed, SplitID.MASK)
    val_x, val_y, val_attn = build_for_stage(val_allowed, SplitID.VAL)
    test_x, test_y, test_attn = build_for_stage(test_allowed, SplitID.TEST)

    return (
        (train_x, train_y, train_attn),
        (val_x, val_y, val_attn),
        (test_x, test_y, test_attn),
    )


def test_fix():
    """Test the fix with toy dataset to verify targets are now attended."""
    print("🔧 TESTING ATTENTION BUG FIX")
    print("=" * 60)
    
    # Load toy data
    toy_dir = "/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/toy"
    
    # Load encoded data
    encoded_path = os.path.join(toy_dir, "encoded.pt")
    if not os.path.exists(encoded_path):
        print(f"❌ Encoded data not found at {encoded_path}")
        return
        
    data = torch.load(encoded_path)
    
    # Load tokenizer
    tokenizer = Tokenizer.load(os.path.join(toy_dir, "tokenizer.json"))
    
    # Data is tuple: (train_pack, val_pack, test_pack)
    train_pack, val_pack, test_pack = data
    train_x, train_y, train_attn = train_pack
    
    print(f"Data shapes: train_x={train_x.shape}")
    
    # We need to reconstruct the original input_ids and edge_split_mask
    # Load the base data before stage views were applied
    base_data_path = os.path.join(toy_dir, "base_data.pt")
    if os.path.exists(base_data_path):
        base_data = torch.load(base_data_path)
        orig_input_ids = base_data['input_ids']
        orig_edge_mask = base_data['edge_split_mask']
    else:
        print(f"❌ Base data not found. Using train data as approximation.")
        # Use train data but need to reverse the masking
        orig_input_ids = train_x.clone()
        # This is tricky without the original data
        print("⚠️ Cannot perfectly reconstruct original data without base_data.pt")
        return
    
    # Test first sequence with FIXED function
    seq_idx = 0
    print(f"\n🧪 Testing sequence {seq_idx} with FIXED attention logic:")
    
    # Get original padded data for one sequence
    orig_input_ids_seq = orig_input_ids[seq_idx:seq_idx+1]  # [1, seq_len]
    orig_edge_mask_seq = orig_edge_mask[seq_idx:seq_idx+1]  # [1, seq_len]
    
    # Apply FIXED stage views
    (train_x_fixed, train_y_fixed, train_attn_fixed), (val_x_fixed, val_y_fixed, val_attn_fixed), (test_x_fixed, test_y_fixed, test_attn_fixed) = \
        fixed_stage_views_from_base(orig_input_ids_seq, orig_edge_mask_seq, tokenizer)
    
    def analyze_stage(stage_name, input_ids, labels, attention):
        print(f"\n📊 {stage_name} Stage (FIXED):")
        seq = input_ids[0].tolist()
        lab = labels[0].tolist() 
        att = attention[0].tolist()
        
        print(f"  Input IDs:  {seq}")
        print(f"  Labels:     {lab}")
        print(f"  Attention:  {att}")
        
        # Find targets (non-ignore positions)
        target_positions = [i for i, l in enumerate(lab) if l != -1]
        targets_attended = [att[i] for i in target_positions]
        
        print(f"  Target positions: {target_positions}")
        print(f"  Targets attended: {targets_attended}")
        
        if target_positions:
            all_attended = all(targets_attended)
            print(f"  All targets attended: {all_attended}")
            if all_attended:
                print(f"  ✅ ALL TARGETS ARE ATTENDED!")
            else:
                unattended = [pos for pos, att in zip(target_positions, targets_attended) if not att]
                print(f"  ❌ Unattended target positions: {unattended}")
        else:
            print(f"  No targets in this sequence")
    
    analyze_stage("TRAIN", train_x_fixed, train_y_fixed, train_attn_fixed)
    analyze_stage("VAL", val_x_fixed, val_y_fixed, val_attn_fixed)
    analyze_stage("TEST", test_x_fixed, test_y_fixed, test_attn_fixed)


if __name__ == "__main__":
    test_fix()
