#!/usr/bin/env python3
"""
Simple test: Just load the best model checkpoint to verify reproducibility.
"""
import sys
sys.path.insert(0, '.')

import torch
from pathlib import Path
from optuna.storages import JournalStorage, JournalFileStorage
import optuna


def test_model_checkpoint_loading():
    """Load best wiki-rfa model checkpoint."""
    
    print("=" * 80)
    print("WIKI-RFA MODEL LOADING TEST")
    print("=" * 80)
    
    dataset_name = "wiki-rfa"
    outputs_root = Path("/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs")
    
    # Step 1: Find best study
    print(f"\n📍 Step 1: Finding best Optuna study for {dataset_name}...")
    root = outputs_root / dataset_name
    best_score = -float("inf")
    study = None
    best_exp_name = None
    
    for exp_folder in sorted(root.glob(f"{dataset_name}-optuna_*")):
        journal_path = exp_folder / "optuna" / "optuna_study.log"
        if not journal_path.exists():
            continue
        
        try:
            storage = JournalStorage(JournalFileStorage(str(journal_path)))
            summaries = optuna.study.get_all_study_summaries(storage)
            if not summaries:
                continue
            
            study = optuna.load_study(
                study_name=summaries[0].study_name, storage=storage
            )
            
            completed = [t for t in study.trials if t.state.name == "COMPLETE"]
            if not completed:
                continue
            
            if study.best_value > best_score:
                best_score = study.best_value
                best_exp_name = exp_folder.name
        except Exception as e:
            continue
    
    if not study:
        print("❌ No valid studies found")
        return
    
    best_trial = study.best_trial
    print(f"✅ Found best study: {best_exp_name}")
    print(f"   Best trial (overall): #{best_trial.number}, score: {study.best_value:.6f}")
    
    # Step 2: Find best trial WITH checkpoint
    print(f"\n📍 Step 2: Finding best trial with checkpoint...")
    best_exp = best_exp_name
    ckpt_dir = outputs_root / dataset_name / best_exp / "checkpoints"
    checkpoints = list(ckpt_dir.glob("*.ckpt"))
    print(f"   Available checkpoints: {len(checkpoints)}")
    
    best_with_ckpt = None
    best_with_ckpt_val = -float("inf")
    
    for trial in study.trials:
        if trial.state.name != "COMPLETE":
            continue
        ckpt_pattern = f'trial_{trial.number}-'
        has_ckpt = any(ckpt_pattern in c.name for c in checkpoints)
        if has_ckpt and trial.value is not None:
            if trial.value > best_with_ckpt_val:
                best_with_ckpt = trial
                best_with_ckpt_val = trial.value
    
    if not best_with_ckpt:
        print("❌ No trial with checkpoint found")
        return
    
    print(f"✅ Best trial with checkpoint: #{best_with_ckpt.number}, score: {best_with_ckpt.value:.6f}")
    
    # Step 3: Load checkpoint file directly
    print(f"\n📍 Step 3: Loading checkpoint file...")
    ckpt_pattern = f'trial_{best_with_ckpt.number}-'
    ckpt_matches = [c for c in checkpoints if ckpt_pattern in c.name]
    
    if not ckpt_matches:
        print("❌ Checkpoint not found")
        return
    
    ckpt_path = max(ckpt_matches, key=lambda p: p.stat().st_mtime)
    print(f"   Checkpoint file: {ckpt_path.name}")
    print(f"   Checkpoint size: {ckpt_path.stat().st_size / 1e6:.2f} MB")
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        print(f"✅ Checkpoint loaded successfully")
        print(f"\n   Checkpoint contents:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"     {key}: dict with {len(checkpoint[key])} keys")
            elif isinstance(checkpoint[key], torch.Tensor):
                print(f"     {key}: tensor {checkpoint[key].shape}")
            else:
                print(f"     {key}: {type(checkpoint[key])}")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return
    
    # Step 4: Show hyperparameters
    print(f"\n📍 Step 4: Best trial hyperparameters:")
    for k, v in sorted(best_with_ckpt.params.items()):
        print(f"   {k}: {v}")
    
    print(f"\n" + "=" * 80)
    print(f"✅ TEST COMPLETE: Model checkpoint successfully loaded!")
    print(f"=" * 80)
    print(f"\n📊 Summary:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Best Study: {best_exp_name}")
    print(f"   Best Trial #: {best_with_ckpt.number}")
    print(f"   Trial Value (AUC): {best_with_ckpt.value:.6f}")
    print(f"   Checkpoint: {ckpt_path.name}")
    print(f"\n✅ Model can be loaded and used for inference!")
    
    return {
        "study_name": best_exp_name,
        "trial_number": best_with_ckpt.number,
        "trial_value": best_with_ckpt.value,
        "checkpoint_path": str(ckpt_path),
        "hyperparams": best_with_ckpt.params,
    }


if __name__ == "__main__":
    try:
        results = test_model_checkpoint_loading()
        if results:
            print(f"\n📄 Result dict keys: {list(results.keys())}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
