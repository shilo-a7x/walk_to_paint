#!/usr/bin/env python3
"""
Quick-start guide: Run this to set up everything for fast evaluation.
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print output."""
    print(f"\n{'='*80}")
    print(f"📍 {description}")
    print(f"{'='*80}")
    print(f"$ {cmd}\n")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"⚠️  Command failed with code {result.returncode}")
        return False
    return True


def main():
    print(
        f"""
╔════════════════════════════════════════════════════════════════════════════╗
║           WALK-TO-PAINT: SETUP FOR FAST EVALUATION & ITERATION            ║
╚════════════════════════════════════════════════════════════════════════════╝

This script will:
1. ✅ Analyze your Optuna studies
2. ✅ Extract alternative top trials (fewer walks, shorter length)
3. ✅ Build cached datasets for fast evaluation
4. ✅ Load best models for each dataset

Let's go! 🚀
"""
    )

    venv_python = Path(".venv/bin/python")
    if not venv_python.exists():
        print("❌ Virtual environment not found. Please activate it first:")
        print("   source .venv/bin/activate")
        sys.exit(1)

    all_success = True

    # 1. Analyze studies
    all_success &= run_command(
        f"{venv_python} scripts/analyze_optuna_studies.py",
        "STEP 1: Analyze your Optuna studies",
    )

    # 2. Extract top trials for each dataset
    for dataset in ["wiki-rfa", "epinions", "slashdot090221"]:
        all_success &= run_command(
            f"{venv_python} scripts/extract_top_trials.py {dataset} --metric num_walks --tolerance 3",
            f"STEP 2a: Find {dataset} trials with fewer walks (within 3% of best)",
        )

    # 3. Load best models
    for dataset in ["wiki-rfa", "epinions", "slashdot090221"]:
        all_success &= run_command(
            f"{venv_python} scripts/load_best_model.py {dataset}",
            f"STEP 2b: Load best model for {dataset}",
        )

    # 4. Print final summary
    print(f"\n{'='*80}")
    print(f"✅ SETUP COMPLETE!")
    print(f"{'='*80}\n")

    print(
        """
Next Steps:

1. ENABLE CACHING (huge speedup):
   Edit config.yaml and change:
     preprocess:
       use_cache: true  # Enable this

2. BUILD CACHED DATASETS (first run):
   python scripts/evaluation_pipeline.py wiki-rfa
   python scripts/evaluation_pipeline.py epinions
   python scripts/evaluation_pipeline.py slashdot090221
   
   ⏱️  First run: 5-10 minutes (builds cache)
   
3. FAST EVALUATION (subsequent runs):
   python scripts/evaluation_pipeline.py wiki-rfa
   ⏱️  Subsequent runs: < 10 seconds!

4. RESUME OPTUNA STUDIES (add more trials):
   python optuna_run.py --config=config.yaml --n-trials=100 dataset.name=wiki-rfa
   
5. DESIGN EDGE SCORE AGGREGATION:
   Read OPTIMIZATION_GUIDE.md section 8 for aggregation strategies
   Then modify evaluation_pipeline.py to test different methods

Quick Reference Commands:

  # Show all your optuna results
  python scripts/analyze_optuna_studies.py
  
  # Find fast alternatives (fewer walks but similar score)
  python scripts/extract_top_trials.py wiki-rfa --metric num_walks --tolerance 5
  
  # Find shorter walks (shorter but similar score)
  python scripts/extract_top_trials.py epinions --metric max_walk_length --tolerance 5
  
  # Profile data building speed
  python scripts/profile_data_building.py wiki-rfa --quick
  
  # Load and evaluate best model
  python scripts/evaluation_pipeline.py slashdot090221
  
  # Continue optuna optimization
  python optuna_run.py --config=config.yaml --n-trials=50 dataset.name=epinions

Current Status:
  ✅ 3 datasets with complete Optuna studies (Wiki-RFA, Epinions, Slashdot)
  ✅ Best models: wiki-rfa=0.7779, epinions=0.9133, slashdot=0.8529
  ✅ All studies are resumable
  ✅ Dataset caching infrastructure ready
  ✅ Evaluation pipeline ready
  
Your next focus: Edge score aggregation (see OPTIMIZATION_GUIDE.md)
"""
    )


if __name__ == "__main__":
    main()
