#!/usr/bin/env python3
"""
Debug the _stage_views function to understand attention mask and target relationships.
"""

import torch
import yaml
import json
from omegaconf import OmegaConf

def debug_stage_views():
    """Debug how attention masks and targets work in _stage_views_from_base."""
    print("🔍 DEBUGGING STAGE VIEWS - ATTENTION & TARGETS")
    print("=" * 60)
    
    # Load toy dataset to get real data
    encoded_data = torch.load('data/toy/encoded.pt')
    train_pack, val_pack, test_pack = encoded_data
    
    with open('data/toy/meta.json', 'r') as f:
        meta = json.load(f)
    
    print(f"Meta info: {meta}")
    
    # Get first few samples for analysis
    train_input, train_labels, train_attn = train_pack
    val_input, val_labels, val_attn = val_pack
    test_input, test_labels, test_attn = test_pack
    
    print(f"\nData shapes:")
    print(f"  Train: input={train_input.shape}, labels={train_labels.shape}, attn={train_attn.shape}")
    
    # Analyze first sequence across all stages
    seq_idx = 0
    print(f"\n🔍 ANALYZING SEQUENCE {seq_idx} ACROSS STAGES:")
    print("=" * 50)
    
    stages = [
        ("TRAIN", train_input[seq_idx], train_labels[seq_idx], train_attn[seq_idx]),
        ("VAL", val_input[seq_idx], val_labels[seq_idx], val_attn[seq_idx]),
        ("TEST", test_input[seq_idx], test_labels[seq_idx], test_attn[seq_idx])
    ]
    
    for stage_name, input_seq, label_seq, attn_seq in stages:
        print(f"\n📊 {stage_name} Stage:")
        print(f"  Input IDs:  {input_seq.tolist()}")
        print(f"  Labels:     {label_seq.tolist()}")
        print(f"  Attention:  {attn_seq.tolist()}")
        
        # Find target positions (where labels != ignore_index)
        target_positions = (label_seq != meta['ignore_index']).nonzero().flatten()
        print(f"  Target positions: {target_positions.tolist()}")
        
        # Check if targets are attended
        if len(target_positions) > 0:
            target_attended = attn_seq[target_positions]
            print(f"  Targets attended: {target_attended.tolist()}")
            all_targets_attended = target_attended.all().item()
            print(f"  All targets attended: {all_targets_attended}")
            
            if not all_targets_attended:
                print(f"  ⚠️ Some targets are NOT attended!")
                unattended_targets = target_positions[target_attended == 0]
                print(f"  Unattended target positions: {unattended_targets.tolist()}")
        else:
            print(f"  No targets in this sequence")
        
        # Find attended positions
        attended_positions = (attn_seq == 1).nonzero().flatten()
        print(f"  Attended positions: {attended_positions.tolist()}")
        
        # Find MASK tokens
        mask_positions = (input_seq == meta['pad_id']).nonzero().flatten()  # Actually looking for MASK, but let's check pad too
        print(f"  Padded positions: {mask_positions.tolist()}")
        
        # Check for MASK token (ID=1)
        actual_mask_positions = (input_seq == 1).nonzero().flatten()
        print(f"  MASK token positions: {actual_mask_positions.tolist()}")

def analyze_attention_logic():
    """Analyze the logic of attention mask construction."""
    print(f"\n🧠 ATTENTION LOGIC ANALYSIS")
    print("=" * 60)
    
    print(f"According to _stage_views_from_base function:")
    print(f"")
    print(f"1️⃣ Base Attention:")
    print(f"   base_attn = (input_ids != pad_id)")
    print(f"   → All non-padded positions are initially attended")
    print(f"")
    print(f"2️⃣ Edge Filtering per Stage:")
    print(f"   TRAIN: allowed_edges = edges in TRAIN split")
    print(f"   VAL:   allowed_edges = edges in TRAIN + MASK splits") 
    print(f"   TEST:  allowed_edges = edges in TRAIN + MASK + VAL splits")
    print(f"")
    print(f"3️⃣ Disallowed Edge Masking:")
    print(f"   disallowed_edges = is_edge_pos & (~allowed_edges)")
    print(f"   attn[disallowed_edges] = 0")
    print(f"   → Hide edges not allowed in current stage")
    print(f"")
    print(f"4️⃣ Target Masking:")
    print(f"   TRAIN: target = MASK split edges")
    print(f"   VAL:   target = VAL split edges")
    print(f"   TEST:  target = TEST split edges")
    print(f"   → Targets get labels, input masked with MASK token")

def check_specific_case():
    """Check a specific case where target might not be attended."""
    print(f"\n🎯 CHECKING SPECIFIC ATTENTION-TARGET RELATIONSHIP")
    print("=" * 60)
    
    # Load encoded data
    encoded_data = torch.load('data/toy/encoded.pt')
    train_pack, val_pack, test_pack = encoded_data
    
    with open('data/toy/meta.json', 'r') as f:
        meta = json.load(f)
    
    # Check train stage
    train_input, train_labels, train_attn = train_pack
    
    # Find sequences with targets
    sequences_with_targets = []
    for seq_idx in range(min(10, len(train_labels))):
        target_positions = (train_labels[seq_idx] != meta['ignore_index']).nonzero().flatten()
        if len(target_positions) > 0:
            sequences_with_targets.append((seq_idx, target_positions))
    
    print(f"Found {len(sequences_with_targets)} sequences with targets (first 10 checked)")
    
    # Analyze each sequence with targets
    for seq_idx, target_positions in sequences_with_targets[:3]:  # Check first 3
        print(f"\nSequence {seq_idx}:")
        input_seq = train_input[seq_idx]
        label_seq = train_labels[seq_idx]
        attn_seq = train_attn[seq_idx]
        
        print(f"  Targets at positions: {target_positions.tolist()}")
        
        for pos in target_positions:
            pos_item = pos.item()
            input_val = input_seq[pos_item].item()
            label_val = label_seq[pos_item].item()
            attn_val = attn_seq[pos_item].item()
            
            print(f"    Position {pos_item}: input={input_val}, label={label_val}, attended={attn_val}")
            
            if attn_val == 0:
                print(f"    ⚠️ TARGET NOT ATTENDED at position {pos_item}!")
            else:
                print(f"    ✅ Target properly attended at position {pos_item}")

def main():
    """Run complete stage views debugging."""
    debug_stage_views()
    analyze_attention_logic()
    check_specific_case()
    
    print(f"\n" + "=" * 60)
    print("🔍 STAGE VIEWS DEBUGGING COMPLETED")
    print("=" * 60)
    
    print(f"\n💡 KEY QUESTIONS TO VERIFY:")
    print(f"   1. Are ALL target positions attended? (Should be YES)")
    print(f"   2. Are target positions masked with MASK token in input?")
    print(f"   3. Are disallowed edges properly hidden in attention?")
    print(f"   4. Do attention masks make sense for each stage?")

if __name__ == "__main__":
    main()
