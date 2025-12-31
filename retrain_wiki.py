#!/usr/bin/env python3
"""
Retrain wiki-rfa model with best Optuna hyperparameters (trial #194).
Verifies training metrics and checkpoint saving work correctly.
"""
import sys
sys.path.insert(0, '.')

from src.utils.config import load_config
from src.data.prepare_data import prepare_data
from src.training.train import train_model

def main():
    print("=" * 80)
    print("WIKI-RFA RETRAINING: Using Best Optuna Hyperparameters")
    print("=" * 80)
    
    # Load config with wiki-rfa dataset (auto-merges configs/wiki-rfa.yaml)
    cfg = load_config("config.yaml", overrides=["dataset.name=wiki-rfa"])
    
    # Set experiment name for this retrain
    cfg.training.exp_name = "wiki-rfa-retrain-trial194-hyperparams"
    cfg.training.eval_only = False
    cfg.training.resume_from_checkpoint = None
    
    print(f"\n📋 Configuration:")
    print(f"   Dataset: {cfg.dataset.name}")
    print(f"   Data dir: {cfg.dataset.data_dir}")
    print(f"   Max walk length: {cfg.dataset.max_walk_length}")
    print(f"   Num walks: {cfg.dataset.num_walks}")
    print(f"\n   Model:")
    print(f"     embedding_dim: {cfg.model.embedding_dim}")
    print(f"     hidden_dim: {cfg.model.hidden_dim}")
    print(f"     nhead: {cfg.model.nhead}")
    print(f"     nlayers: {cfg.model.nlayers}")
    print(f"     dropout: {cfg.model.dropout}")
    print(f"\n   Training:")
    print(f"     batch_size: {cfg.training.batch_size}")
    print(f"     epochs: {cfg.training.epochs}")
    print(f"     lr: {cfg.training.lr}")
    print(f"     weight_decay: {cfg.training.weight_decay}")
    print(f"     early_stopping_patience: {cfg.training.early_stopping_patience}")
    print(f"     gradient_clip_val: {cfg.training.gradient_clip_val}")
    
    # Prepare data (uses cache if available)
    print(f"\n🔄 Preparing data...")
    data_module = prepare_data(cfg)
    print(f"✅ Data ready")
    
    # Train model
    print(f"\n🚀 Starting training...")
    print(f"   Checkpoints will be saved to: {cfg.training.checkpoint_dir}")
    print(f"   Logs will be saved to: {cfg.training.log_dir}")
    
    model = train_model(cfg, data_module)
    
    print(f"\n✅ Training complete!")
    print(f"\nTo evaluate the best checkpoint:")
    print(f"  python run.py --config=config.yaml --device 0 \\")
    print(f"    dataset.name=wiki-rfa \\")
    print(f"    training.eval_only=true \\")
    print(f"    training.resume_from_checkpoint=<checkpoint_path>")
    
    return model


if __name__ == "__main__":
    try:
        model = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
