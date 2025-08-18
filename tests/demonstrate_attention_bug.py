#!/usr/bin/env python3
"""
Demonstrate the critical attention bug and show the fix.
"""

import torch
import sys
sys.path.append('/home/dsi/shilo_avital/yolo_lab/walk_to_paint/src')

# Mock the needed constants and tokenizer
class SplitID:
    BAD = 0
    TRAIN = 1
    MASK = 2
    VAL = 3
    TEST = 4

class MockTokenizer:
    def __init__(self):
        self.UNK_LABEL_ID = -1
        self.PAD_ID = 0
        self.MASK_ID = 10
        self.vocab_size = 16
        self.id2edge_label = {1: "edge1", 2: "edge2"}
        self.token2id = {"edge1": 5, "edge2": 6}

def original_buggy_stage_views(input_ids, edge_split_mask, tokenizer):
    """Original buggy version where targets are not attended."""
    ignore_index = tokenizer.UNK_LABEL_ID
    pad_id = int(tokenizer.PAD_ID)
    mask_id = int(tokenizer.MASK_ID)

    base_attn = (input_ids != pad_id).long()
    is_edge_pos = edge_split_mask != SplitID.BAD
    
    id2class = torch.full((tokenizer.vocab_size,), ignore_index, dtype=torch.long)
    for class_id, edge_tok in tokenizer.id2edge_label.items():
        tok_id = tokenizer.token2id.get(edge_tok, None)
        if tok_id is not None:
            id2class[tok_id] = int(class_id)

    def build_for_stage(allowed_splits, target_split: int):
        attn = base_attn.clone()
        allowed_edges = torch.zeros_like(edge_split_mask, dtype=torch.bool)
        for s in allowed_splits:
            allowed_edges |= edge_split_mask == s
            
        # BUG: disallowed_edges excludes target edges from attention!
        disallowed_edges = is_edge_pos & (~allowed_edges)
        attn[disallowed_edges] = 0

        x = input_ids.clone()
        target_mask = edge_split_mask == target_split
        labels = torch.full_like(input_ids, ignore_index)
        if target_mask.any():
            labels[target_mask] = id2class[x[target_mask]]
        x[target_mask] = mask_id
        x[disallowed_edges] = pad_id

        return x, labels, attn

    train_allowed = [SplitID.TRAIN]
    train_x, train_y, train_attn = build_for_stage(train_allowed, SplitID.MASK)
    return train_x, train_y, train_attn

def fixed_stage_views(input_ids, edge_split_mask, tokenizer):
    """Fixed version where targets are always attended."""
    ignore_index = tokenizer.UNK_LABEL_ID
    pad_id = int(tokenizer.PAD_ID)
    mask_id = int(tokenizer.MASK_ID)

    base_attn = (input_ids != pad_id).long()
    is_edge_pos = edge_split_mask != SplitID.BAD
    
    id2class = torch.full((tokenizer.vocab_size,), ignore_index, dtype=torch.long)
    for class_id, edge_tok in tokenizer.id2edge_label.items():
        tok_id = tokenizer.token2id.get(edge_tok, None)
        if tok_id is not None:
            id2class[tok_id] = int(class_id)

    def build_for_stage(allowed_splits, target_split: int):
        attn = base_attn.clone()
        allowed_edges = torch.zeros_like(edge_split_mask, dtype=torch.bool)
        for s in allowed_splits:
            allowed_edges |= edge_split_mask == s
            
        target_edges = edge_split_mask == target_split
        
        # FIX: Include target edges in allowed for attention
        allowed_edges_with_targets = allowed_edges | target_edges
        
        disallowed_edges = is_edge_pos & (~allowed_edges_with_targets)
        attn[disallowed_edges] = 0

        x = input_ids.clone()
        labels = torch.full_like(input_ids, ignore_index)
        if target_edges.any():
            labels[target_edges] = id2class[x[target_edges]]
        x[target_edges] = mask_id
        x[disallowed_edges] = pad_id

        return x, labels, attn

    train_allowed = [SplitID.TRAIN]
    train_x, train_y, train_attn = build_for_stage(train_allowed, SplitID.MASK)
    return train_x, train_y, train_attn

def demonstrate_bug():
    print("🐛 DEMONSTRATING THE ATTENTION BUG")
    print("=" * 60)
    
    # Create a test case
    tokenizer = MockTokenizer()
    
    # Example: [node, edge, node, target_edge, node]
    input_ids = torch.tensor([[3, 5, 4, 6, 7]])  # 5,6 are edge tokens
    edge_split_mask = torch.tensor([[0, 1, 0, 2, 0]])  # pos 1=TRAIN edge, pos 3=MASK edge (target)
    
    print("Test case:")
    print(f"  Input IDs:        {input_ids[0].tolist()}")
    print(f"  Edge split mask:  {edge_split_mask[0].tolist()}")
    print(f"  Split meanings:   [BAD, TRAIN, BAD, MASK, BAD]")
    print(f"  Target position:  3 (MASK split, should be labeled & attended)")
    
    print(f"\n🔴 ORIGINAL BUGGY VERSION:")
    train_x_bug, train_y_bug, train_attn_bug = original_buggy_stage_views(input_ids, edge_split_mask, tokenizer)
    
    print(f"  Input:     {train_x_bug[0].tolist()}")
    print(f"  Labels:    {train_y_bug[0].tolist()}")
    print(f"  Attention: {train_attn_bug[0].tolist()}")
    
    target_pos = 3
    target_label = train_y_bug[0][target_pos].item()
    target_attended = train_attn_bug[0][target_pos].item()
    
    print(f"  Target at pos {target_pos}: label={target_label}, attended={target_attended}")
    if target_label != -1 and target_attended == 0:
        print(f"  ❌ BUG: Target has label but is NOT attended!")
    
    print(f"\n🟢 FIXED VERSION:")
    train_x_fix, train_y_fix, train_attn_fix = fixed_stage_views(input_ids, edge_split_mask, tokenizer)
    
    print(f"  Input:     {train_x_fix[0].tolist()}")
    print(f"  Labels:    {train_y_fix[0].tolist()}")
    print(f"  Attention: {train_attn_fix[0].tolist()}")
    
    target_label_fix = train_y_fix[0][target_pos].item()
    target_attended_fix = train_attn_fix[0][target_pos].item()
    
    print(f"  Target at pos {target_pos}: label={target_label_fix}, attended={target_attended_fix}")
    if target_label_fix != -1 and target_attended_fix == 1:
        print(f"  ✅ FIXED: Target has label AND is attended!")
        
    print(f"\n💡 THE BUG EXPLANATION:")
    print(f"   In train stage: allowed_splits = [TRAIN]")
    print(f"   Target split = MASK")
    print(f"   Problem: MASK edges are targets but not in allowed_splits")
    print(f"   So they get masked out in attention (disallowed_edges)")
    print(f"   Fix: Include target edges in attention regardless of allowed_splits")

if __name__ == "__main__":
    demonstrate_bug()
