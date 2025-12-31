#!/usr/bin/env python3
"""
Demonstrate loading a checkpoint and evaluating it to reproduce training metrics.

This shows step-by-step:
1. Loading checkpoint with trial hyperparameters
2. Preparing data with same preprocessing
3. Running evaluation
4. Comparing to training-time metrics
"""
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from src.utils.config import load_config
from src.model.lit_model import LitEdgeClassifier
from src.data.prepare_data import prepare_data


def eval_checkpoint_demo(ckpt_path: str, trial_overrides_dotlist: str):
    """
    Load checkpoint with trial hyperparameters and evaluate.
    """
    print("=" * 80)
    print("🔬 CHECKPOINT EVALUATION DEMO")
    print("=" * 80)
    print(f"\n📂 Checkpoint: {ckpt_path}")
    
    # Extract expected AUC from checkpoint filename
    ckpt_name = Path(ckpt_path).name
    expected_auc = None
    if "val_auc_epoch=" in ckpt_name:
        try:
            expected_auc = float(ckpt_name.split("val_auc_epoch=")[1].split(".ckpt")[0])
            print(f"📊 Expected val AUC (from training): {expected_auc:.4f}")
        except:
            pass
    
    print("\n" + "=" * 80)
    print("STEP 1: Load Configuration with Trial Hyperparameters")
    print("=" * 80)
    
    # Parse overrides
    overrides = trial_overrides_dotlist.split() if trial_overrides_dotlist else []
    
    # Load config with trial hyperparameters
    cfg = load_config("config.yaml", overrides=["dataset.name=wiki-rfa"] + overrides)
    
    print("\n✅ Configuration loaded:")
    print(f"   Dataset:")
    print(f"     - max_walk_length: {cfg.dataset.max_walk_length}")
    print(f"     - num_walks: {cfg.dataset.num_walks}")
    print(f"   Model:")
    print(f"     - embedding_dim: {cfg.model.embedding_dim}")
    print(f"     - hidden_dim: {cfg.model.hidden_dim}")
    print(f"     - nhead: {cfg.model.nhead}")
    print(f"     - nlayers: {cfg.model.nlayers}")
    print(f"     - dropout: {cfg.model.dropout}")
    print(f"   Training:")
    print(f"     - batch_size: {cfg.training.batch_size}")
    print(f"     - lr: {cfg.training.lr}")
    print(f"     - weight_decay: {cfg.training.weight_decay}")
    
    print("\n" + "=" * 80)
    print("STEP 2: Prepare Data (same preprocessing as training)")
    print("=" * 80)
    
    # Set seed for reproducibility
    seed_everything(42, workers=True)
    
    # Prepare data - this will load cached data with the same preprocessing
    data_module = prepare_data(cfg)
    
    print(f"\n✅ Data prepared:")
    print(f"   - vocab_size: {cfg.model.vocab_size}")
    print(f"   - pad_id: {cfg.model.pad_id}")
    print(f"   - Train batches: {len(data_module['train'])}")
    print(f"   - Val batches: {len(data_module['val'])}")
    print(f"   - Test batches: {len(data_module['test'])}")
    
    print("\n" + "=" * 80)
    print("STEP 3: Load Checkpoint")
    print("=" * 80)
    
    # Load model from checkpoint
    model = LitEdgeClassifier.load_from_checkpoint(ckpt_path, cfg=cfg)
    
    print(f"\n✅ Checkpoint loaded successfully!")
    print(f"   Model architecture:")
    print(f"     - Embedding dim: {model.model.embed.weight.shape[1]}")
    print(f"     - Hidden dim: {model.model.transformer.layers[0].linear1.weight.shape[0]}")
    print(f"     - Num layers: {len(model.model.transformer.layers)}")
    print(f"     - Num heads: {model.model.transformer.layers[0].self_attn.num_heads}")
    
    # Verify architecture matches config
    assert model.model.embed.weight.shape[1] == cfg.model.embedding_dim, "Embedding dim mismatch!"
    assert model.model.transformer.layers[0].linear1.weight.shape[0] == cfg.model.hidden_dim, "Hidden dim mismatch!"
    assert len(model.model.transformer.layers) == cfg.model.nlayers, "Num layers mismatch!"
    assert model.model.transformer.layers[0].self_attn.num_heads == cfg.model.nhead, "Num heads mismatch!"
    
    print(f"   ✓ Architecture verified (matches config)")
    
    print("\n" + "=" * 80)
    print("STEP 4: Run Evaluation")
    print("=" * 80)
    
    # Create trainer for evaluation
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    
    print("\n🔄 Running validation...")
    val_results = trainer.validate(model, data_module["val"])
    
    print("\n🔄 Running test...")
    test_results = trainer.test(model, data_module["test"])
    
    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    
    if val_results:
        val_metrics = val_results[0]
        print("\n✅ Validation Metrics:")
        print(f"   - Loss: {val_metrics.get('val_loss', 'N/A'):.4f}")
        print(f"   - Accuracy: {val_metrics.get('val_acc_epoch', 'N/A'):.4f}")
        print(f"   - F1: {val_metrics.get('val_f1_epoch', 'N/A'):.4f}")
        print(f"   - AUC: {val_metrics.get('val_auc_epoch', 'N/A'):.4f}")
        
        if expected_auc is not None:
            actual_auc = val_metrics.get('val_auc_epoch', 0)
            diff = abs(actual_auc - expected_auc)
            print(f"\n   📈 Comparison:")
            print(f"      Training-time val AUC: {expected_auc:.4f}")
            print(f"      Eval-only val AUC:     {actual_auc:.4f}")
            print(f"      Difference:            {diff:.6f}")
            
            if diff < 0.001:
                print(f"      ✅ EXCELLENT MATCH! (diff < 0.001)")
            elif diff < 0.01:
                print(f"      ✅ GOOD MATCH (diff < 0.01)")
            else:
                print(f"      ⚠️  Larger difference - may need investigation")
    
    if test_results:
        test_metrics = test_results[0]
        print("\n✅ Test Metrics:")
        print(f"   - Loss: {test_metrics.get('test_loss', 'N/A'):.4f}")
        print(f"   - Accuracy: {test_metrics.get('test_acc_epoch', 'N/A'):.4f}")
        print(f"   - F1: {test_metrics.get('test_f1_epoch', 'N/A'):.4f}")
        print(f"   - AUC: {test_metrics.get('test_auc_epoch', 'N/A'):.4f}")
    
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE")
    print("=" * 80)
    print("\n💡 Key Takeaways:")
    print("   1. Checkpoint loaded successfully with trial hyperparameters")
    print("   2. Model architecture matches the saved weights")
    print("   3. Evaluation runs without errors")
    if expected_auc and val_results:
        actual_auc = val_results[0].get('val_auc_epoch', 0)
        if abs(actual_auc - expected_auc) < 0.01:
            print("   4. ✅ Metrics closely match training-time values!")
        else:
            print("   4. ⚠️  Metrics differ - may be due to:")
            print("      - Different data splits/preprocessing")
            print("      - Different random seed")
            print("      - Cached data from different hyperparameters")
    
    return val_results, test_results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python eval_checkpoint_demo.py <checkpoint_path> '<dotlist_overrides>'")
        print("\nExample:")
        print("  python eval_checkpoint_demo.py \\")
        print("    outputs/wiki-rfa/.../trial_194-epoch=07-val_auc_epoch=0.7429.ckpt \\")
        print("    'dataset.max_walk_length=86 dataset.num_walks=589237 model.hidden_dim=256 ...'")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    overrides = sys.argv[2]
    
    eval_checkpoint_demo(ckpt_path, overrides)
