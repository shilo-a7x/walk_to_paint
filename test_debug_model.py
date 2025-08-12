#!/usr/bin/env python3
"""
Quick test script to demonstrate the debug model functionality.
"""

import torch
import yaml
import json
from omegaconf import OmegaConf
from src.model.model_debug import TransformerModelDebug


def main():
    # Load config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    with open("data/chess/meta.json", "r") as f:
        meta = json.load(f)

    cfg["model"]["vocab_size"] = meta["vocab_size"]
    cfg["model"]["num_classes"] = meta["num_classes"]
    cfg["model"]["pad_id"] = meta["pad_id"]
    cfg["model"]["ignore_index"] = meta["ignore_index"]
    cfg = OmegaConf.create(cfg)

    # Create debug model
    model = TransformerModelDebug(cfg, debug=True)
    model.eval()

    print("🔍 Testing model with debug output:")
    print("=" * 50)

    # Create sample input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(1, cfg.model.vocab_size, (batch_size, seq_len))

    # Add some padding
    input_ids[0, 7:] = cfg.model.pad_id
    input_ids[1, 9:] = cfg.model.pad_id

    attention_mask = (input_ids != cfg.model.pad_id).long()

    print(f"Input shape: {input_ids.shape}")
    print(f"Sample input_ids: {input_ids[0].tolist()}")
    print(f"Sample attention_mask: {attention_mask[0].tolist()}")
    print("\nForward pass with debug output:")
    print("-" * 30)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    print("-" * 30)
    print(f"✅ Final output shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {cfg.model.num_classes})")


if __name__ == "__main__":
    main()
