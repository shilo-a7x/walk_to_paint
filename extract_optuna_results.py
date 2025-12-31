#!/usr/bin/env python3
"""
Extract and analyze results from existing Optuna studies.
Handles both old format (nhead_emb_combo) and new format (separate nhead/embedding_dim).
"""

import argparse
import glob
import joblib
import optuna
import pandas as pd
from pathlib import Path


def decode_nhead_emb_combo(combo_idx):
    """
    Decode the old combo index format back to nhead and embedding_dim.
    This matches the exact logic from the original optuna_run.py
    """
    nhead_options = [1, 2, 4, 8, 16]
    embedding_dim_options = [4, 8, 16, 32, 64, 128]

    # Recreate valid combinations (same order as original)
    valid_combinations = []
    for nhead in nhead_options:
        for emb_dim in embedding_dim_options:
            if emb_dim % nhead == 0:
                valid_combinations.append((nhead, emb_dim))

    if 0 <= combo_idx < len(valid_combinations):
        return valid_combinations[combo_idx]
    else:
        return None, None


def extract_trial_data(study, top_n=10):
    """Extract trial data with proper nhead/embedding_dim decoding.
    
    Respects study.direction: if MAXIMIZE, sort descending; if MINIMIZE, sort ascending.
    For MINIMIZE studies that stored negative AUC, negate the value back to positive.
    """
    import optuna

    # Get completed trials
    completed_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
    
    # Sort by value, respecting direction
    # Direction 2 = MAXIMIZE (higher is better)
    # Direction 1 = MINIMIZE (lower is better)
    is_maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE
    completed_trials.sort(
        key=lambda t: t.value if t.value is not None else (float("-inf") if is_maximize else float("inf")),
        reverse=is_maximize  # Reverse for maximize, normal for minimize
    )

    trials_data = []

    for i, trial in enumerate(completed_trials[:top_n]):
        # For MINIMIZE studies, the value is typically -AUC; for MAXIMIZE, it's +AUC
        # Check if we need to negate (heuristic: if study stored negative values for MINIMIZE)
        val_auc = trial.value if trial.value is not None else None
        if val_auc is not None and not is_maximize and val_auc < 0:
            val_auc = -val_auc  # Convert back from negative
        
        data = {
            "rank": i + 1,
            "trial_number": trial.number,
            "val_auc": val_auc,
        }

        # Extract parameters
        params = trial.params

        # Handle nhead and embedding_dim (multiple formats for backward compatibility)
        if "model.nhead" in params and "model.embedding_dim" in params:
            # Direct categorical format (cleanest - current approach)
            data["model.nhead"] = params["model.nhead"]
            data["model.embedding_dim"] = params["model.embedding_dim"]
        elif (
            "actual_nhead" in trial.user_attrs
            and "actual_embedding_dim" in trial.user_attrs
        ):
            # User attributes format (if present)
            data["model.nhead"] = trial.user_attrs["actual_nhead"]
            data["model.embedding_dim"] = trial.user_attrs["actual_embedding_dim"]
        elif "model.nhead_emb_combo" in params:
            # Old combo format - decode the index
            combo_idx = params["model.nhead_emb_combo"]
            nhead, embedding_dim = decode_nhead_emb_combo(combo_idx)
            data["model.nhead"] = nhead
            data["model.embedding_dim"] = embedding_dim
        else:
            # Fallback
            data["model.nhead"] = params.get("model.nhead", "Unknown")
            data["model.embedding_dim"] = params.get("model.embedding_dim", "Unknown")

        # Extract other parameters
        for key, value in params.items():
            if key != "model.nhead_emb_combo":  # Skip the old combo key
                data[key] = value

        trials_data.append(data)

    return trials_data


def print_trial_summary(trials_data, top_n=10):
    """Print a nicely formatted summary of top trials."""

    print(f"\n🏆 TOP {len(trials_data)} TRIALS SUMMARY")
    print("=" * 80)

    for data in trials_data:
        print(f"\n📊 Rank #{data['rank']} - Trial #{data['trial_number']}")
        print(f"   Val AUC: {data['val_auc']:.6f}")

        print("   📈 Data Generation:")
        print(f"      Walk length: {data.get('dataset.max_walk_length', 'N/A')}")
        print(f"      Num walks: {data.get('dataset.num_walks', 'N/A'):,}")

        print("   🏋️  Training:")
        print(f"      LR: {data.get('training.lr', 'N/A'):.2e}")
        print(f"      Batch size: {data.get('training.batch_size', 'N/A')}")
        print(f"      Weight decay: {data.get('training.weight_decay', 'N/A'):.2e}")
        print(f"      Epochs: {data.get('training.epochs', 'N/A')}")

        print("   🧠 Model Architecture:")
        print(f"      Embedding dim: {data.get('model.embedding_dim', 'N/A')}")
        print(f"      Hidden dim: {data.get('model.hidden_dim', 'N/A')}")
        print(f"      Num heads: {data.get('model.nhead', 'N/A')}")
        print(f"      Num layers: {data.get('model.nlayers', 'N/A')}")
        print(f"      Dropout: {data.get('model.dropout', 'N/A'):.3f}")


def save_results_csv(trials_data, output_file):
    """Save results to CSV for further analysis."""
    df = pd.DataFrame(trials_data)
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved detailed results to: {output_file}")


def generate_best_config_yaml(best_trial_data, output_file):
    """Generate a config YAML from the best trial."""

    config_content = f"""# Best Optuna configuration - Trial #{best_trial_data['trial_number']} (AUC: {best_trial_data['val_auc']:.6f})

dataset:
  max_walk_length: {best_trial_data.get('dataset.max_walk_length', 50)}
  num_walks: {best_trial_data.get('dataset.num_walks', 1000000)}

training:
  lr: {best_trial_data.get('training.lr', 0.001)}
  weight_decay: {best_trial_data.get('training.weight_decay', 1e-5)}
  batch_size: {best_trial_data.get('training.batch_size', 64)}
  gradient_clip_val: {best_trial_data.get('training.gradient_clip_val', 1.0)}
  early_stopping_patience: {best_trial_data.get('training.early_stopping_patience', 10)}
  epochs: {best_trial_data.get('training.epochs', 50)}

model:
  embedding_dim: {best_trial_data.get('model.embedding_dim', 64)}
  hidden_dim: {best_trial_data.get('model.hidden_dim', 64)}
  nhead: {best_trial_data.get('model.nhead', 4)}
  nlayers: {best_trial_data.get('model.nlayers', 2)}
  dropout: {best_trial_data.get('model.dropout', 0.1)}
"""

    with open(output_file, "w") as f:
        f.write(config_content)

    print(f"📋 Saved best config to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract results from Optuna studies")
    parser.add_argument(
        "--study-path", type=str, help="Path to specific study .pkl file"
    )
    parser.add_argument(
        "--top-n", type=int, default=10, help="Number of top trials to analyze"
    )
    parser.add_argument(
        "--output-csv", type=str, default="optuna_results.csv", help="Output CSV file"
    )
    parser.add_argument(
        "--output-yaml",
        type=str,
        default="best_optuna_config.yaml",
        help="Output YAML config",
    )

    args = parser.parse_args()

    # Find study files
    if args.study_path:
        study_files = [args.study_path]
    else:
        # Look for all study files in current directory
        study_files = glob.glob("optuna_study_*.pkl")

    if not study_files:
        print("❌ No Optuna study files found!")
        print("   Expected files: optuna_study_*.pkl")
        print("   Or specify with --study-path")
        return

    print(f"📊 Found {len(study_files)} study file(s)")

    all_trials_data = []

    for study_file in study_files:
        print(f"\n🔍 Analyzing: {study_file}")

        try:
            study = joblib.load(study_file)

            print(f"   Total trials: {len(study.trials)}")
            completed_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
            print(f"   Completed trials: {len(completed_trials)}")

            if completed_trials:
                is_maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE
                best_trial = (
                    max(completed_trials, key=lambda t: t.value)
                    if is_maximize
                    else min(
                        completed_trials,
                        key=lambda t: t.value if t.value is not None else float("inf"),
                    )
                )

                best_value = best_trial.value if best_trial.value is not None else 0
                if not is_maximize and best_value < 0:
                    best_auc = -best_value
                else:
                    best_auc = best_value

                print(
                    f"   Best AUC: {best_auc:.6f} (Trial #{best_trial.number})"
                )

            # Extract trial data
            trials_data = extract_trial_data(study, args.top_n)
            all_trials_data.extend(trials_data)

        except Exception as e:
            print(f"   ❌ Error loading {study_file}: {e}")

    if not all_trials_data:
        print("❌ No trial data extracted!")
        return

    # Sort all trials by AUC (best first)
    all_trials_data.sort(
        key=lambda x: x["val_auc"] if x["val_auc"] is not None else -1, reverse=True
    )

    # Update ranks
    for i, data in enumerate(all_trials_data[: args.top_n]):
        data["rank"] = i + 1

    # Print summary
    print_trial_summary(all_trials_data[: args.top_n], args.top_n)

    # Save results
    save_results_csv(all_trials_data[: args.top_n], args.output_csv)

    # Generate best config
    if all_trials_data:
        generate_best_config_yaml(all_trials_data[0], args.output_yaml)

    print(f"\n✅ Analysis complete! Check {args.output_csv} and {args.output_yaml}")


if __name__ == "__main__":
    main()
