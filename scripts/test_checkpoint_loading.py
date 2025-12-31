#!/usr/bin/env python3
"""Test checkpoint loading with trial hyperparameters.

Tests that:
1. Old checkpoints (no saved hyperparameters) can be loaded by explicitly passing cfg
2. Model architecture matches the weights in the checkpoint
3. Data pipeline uses the correct preprocessing hyperparameters
"""
import sys
import torch
from omegaconf import OmegaConf
from src.utils.config import load_config
from src.model.lit_model import LitEdgeClassifier
from src.data.prepare_data import prepare_data

def test_checkpoint_loading(ckpt_path: str, trial_overrides_dotlist: str):
    """
    Load checkpoint with trial-specific overrides and verify architecture.
    
    Args:
        ckpt_path: Path to checkpoint file
        trial_overrides_dotlist: Space-separated dotlist overrides (e.g., 'model.hidden_dim=256 ...')
    """
    print(f"Testing checkpoint: {ckpt_path}")
    print(f"With overrides: {trial_overrides_dotlist[:100]}...")
    
    # Parse overrides
    overrides = trial_overrides_dotlist.split() if trial_overrides_dotlist else []
    
    # Load config with trial hyperparameters
    cfg = load_config("config.yaml", overrides=["dataset.name=wiki-rfa"] + overrides)
    
    # Prepare data to populate runtime fields like pad_id, vocab_size
    print("\n📦 Preparing data (to set pad_id, vocab_size, etc.)...")
    data_module = prepare_data(cfg)
    print(f"  pad_id: {cfg.model.pad_id}")
    print(f"  vocab_size: {cfg.model.vocab_size}")
    
    print("\n📋 Merged config:")
    print(f"  dataset.max_walk_length: {cfg.dataset.max_walk_length}")
    print(f"  dataset.num_walks: {cfg.dataset.num_walks}")
    print(f"  model.embedding_dim: {cfg.model.embedding_dim}")
    print(f"  model.hidden_dim: {cfg.model.hidden_dim}")
    print(f"  model.nhead: {cfg.model.nhead}")
    print(f"  model.nlayers: {cfg.model.nlayers}")
    print(f"  model.dropout: {cfg.model.dropout}")
    
    # Try loading checkpoint with explicit cfg
    print("\n🔄 Loading checkpoint with explicit cfg...")
    try:
        model = LitEdgeClassifier.load_from_checkpoint(ckpt_path, cfg=cfg)
        print("✅ Successfully loaded checkpoint with explicit cfg")
        
        # Check model architecture
        print("\n🏗️  Model architecture:")
        print(f"  Embedding dim: {model.model.embed.weight.shape[1]}")
        print(f"  Hidden dim: {model.model.transformer.layers[0].linear1.weight.shape[0]}")
        print(f"  Num layers: {len(model.model.transformer.layers)}")
        print(f"  Num heads: {model.model.transformer.layers[0].self_attn.num_heads}")
        
        # Verify architecture matches config
        assert model.model.embed.weight.shape[1] == cfg.model.embedding_dim, \
            f"Embedding dim mismatch: {model.model.embed.weight.shape[1]} != {cfg.model.embedding_dim}"
        assert model.model.transformer.layers[0].linear1.weight.shape[0] == cfg.model.hidden_dim, \
            f"Hidden dim mismatch"
        assert len(model.model.transformer.layers) == cfg.model.nlayers, \
            f"Num layers mismatch"
        assert model.model.transformer.layers[0].self_attn.num_heads == cfg.model.nhead, \
            f"Num heads mismatch"
        
        print("\n✅ Architecture verification passed!")
        
        # Check if checkpoint has saved hyperparameters
        print("\n📦 Checkpoint hyperparameters:")
        if hasattr(model, 'hparams') and model.hparams:
            print(f"  Keys: {list(model.hparams.keys())}")
            if 'cfg' in model.hparams:
                print(f"  cfg present: Yes (contains {len(model.hparams['cfg'])} top-level keys)")
            else:
                print(f"  cfg present: No (old checkpoint)")
        else:
            print("  No hyperparameters saved (old checkpoint)")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_checkpoint_loading.py <checkpoint_path> [dotlist_overrides]")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    overrides = sys.argv[2] if len(sys.argv) > 2 else ""
    
    success = test_checkpoint_loading(ckpt_path, overrides)
    sys.exit(0 if success else 1)
