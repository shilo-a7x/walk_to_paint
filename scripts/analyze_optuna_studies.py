#!/usr/bin/env python3
"""
Analyze all Optuna studies across datasets.
Prints trial counts, best values, and study resumability.
"""
import os
import glob
from pathlib import Path
from optuna.storages import JournalStorage, JournalFileStorage
import optuna

def analyze_study(journal_path):
    """Load and summarize a single study."""
    try:
        storage = JournalStorage(JournalFileStorage(journal_path))
        summaries = optuna.study.get_all_study_summaries(storage)
        if not summaries:
            return None
        
        summary = summaries[0]
        study = optuna.load_study(study_name=summary.study_name, storage=storage)
        
        return {
            "study_name": summary.study_name,
            "num_trials": len(study.trials),
            "best_value": study.best_value,
            "best_trial": study.best_trial.number,
            "direction": study.direction.name,
            "completed_trials": sum(1 for t in study.trials if t.state.name == "COMPLETE"),
            "failed_trials": sum(1 for t in study.trials if t.state.name == "FAIL"),
            "pruned_trials": sum(1 for t in study.trials if t.state.name == "PRUNED"),
        }
    except Exception as e:
        print(f"  ERROR loading {journal_path}: {e}")
        return None

def main():
    root = Path("/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs")
    
    # Group by dataset
    datasets = {}
    for journal_file in sorted(root.glob("*/*/optuna/optuna_study.log")):
        # Extract dataset name
        parts = journal_file.parts
        dataset_name = parts[-3]  # e.g., "wiki-rfa", "epinions", "slashdot090221"
        exp_folder = parts[-4]
        
        if dataset_name not in datasets:
            datasets[dataset_name] = []
        
        datasets[dataset_name].append((exp_folder, str(journal_file)))
    
    # Print results
    for dataset_name in sorted(datasets.keys()):
        print(f"\n{'='*70}")
        print(f"📊 DATASET: {dataset_name}")
        print(f"{'='*70}")
        
        for exp_folder, journal_path in datasets[dataset_name]:
            print(f"\n  📁 {exp_folder}")
            info = analyze_study(journal_path)
            
            if info:
                print(f"     Study: {info['study_name']}")
                print(f"     Trials: {info['num_trials']} total")
                print(f"       ✅ Completed: {info['completed_trials']}")
                print(f"       ❌ Failed: {info['failed_trials']}")
                print(f"       ⏸️  Pruned: {info['pruned_trials']}")
                print(f"     Best: Trial #{info['best_trial']} with {info['direction']} = {info['best_value']:.6f}")
                print(f"     Path: {journal_path}")
            else:
                print(f"     ⚠️  Could not load study")

if __name__ == "__main__":
    main()
