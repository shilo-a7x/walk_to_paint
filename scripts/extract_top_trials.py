#!/usr/bin/env python3
"""
Extract and rank top trials by alternative criteria.
Useful for finding trials with fewer walks or shorter walks but similar scores.
"""
import json
from pathlib import Path
from optuna.storages import JournalStorage, JournalFileStorage
import optuna
import pandas as pd


def get_alternative_top_trials(
    dataset_name,
    outputs_root="/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs",
    metric="num_walks",
    tolerance_pct=5.0,
    top_n=10,
):
    """
    Find top-N trials with alternative optimization criteria.

    Args:
        dataset_name: "wiki-rfa", "epinions", or "slashdot090221"
        metric: "num_walks" or "max_walk_length"
                (minimize this metric while keeping score within tolerance_pct of best)
        tolerance_pct: allow trials within this % of best score
        top_n: return top N alternatives
    """
    root = Path(outputs_root) / dataset_name
    best_score = -float("inf")
    best_study = None
    best_exp = None

    # Find best study
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
                best_study = study
                best_exp = exp_folder.name
        except:
            continue

    if not best_study:
        print(f"❌ No valid study found for {dataset_name}")
        return None

    print(f"🎯 Best score: {best_score:.6f} (from {best_exp})")

    # Extract trial data
    trials_data = []
    for t in best_study.trials:
        if t.state.name != "COMPLETE" or t.value is None:
            continue

        trial_dict = {
            "trial_id": t.number,
            "value": t.value,
            "num_walks": t.params.get("dataset.num_walks", None),
            "max_walk_length": t.params.get("dataset.max_walk_length", None),
        }

        # Copy other params
        for k, v in t.params.items():
            if "dataset" not in k:
                trial_dict[k] = v

        trials_data.append(trial_dict)

    # Build dataframe
    df = pd.DataFrame(trials_data)
    df = df.sort_values("value", ascending=False).reset_index(drop=True)

    # Filter by tolerance
    threshold_score = best_score * (1 - tolerance_pct / 100.0)
    df_valid = df[df["value"] >= threshold_score].copy()

    print(
        f"\n📊 {len(df_valid)} trials within {tolerance_pct}% of best score (threshold: {threshold_score:.6f})"
    )

    # Rank by metric
    if metric == "num_walks":
        df_valid = df_valid.sort_values("num_walks").reset_index(drop=True)
        print(
            f"\n🏆 Top {top_n} by FEWEST WALKS (while keeping score > {threshold_score:.6f}):"
        )
    elif metric == "max_walk_length":
        df_valid = df_valid.sort_values("max_walk_length").reset_index(drop=True)
        print(
            f"\n🏆 Top {top_n} by SHORTEST WALKS (while keeping score > {threshold_score:.6f}):"
        )
    else:
        print(f"Unknown metric: {metric}")
        return None

    # Display
    for idx, row in df_valid.head(top_n).iterrows():
        print(f"\n  Trial #{row['trial_id']}: score={row['value']:.6f}")
        print(
            f"    num_walks: {row['num_walks']:,.0f}"
            if pd.notna(row["num_walks"])
            else ""
        )
        print(
            f"    max_walk_length: {row['max_walk_length']}"
            if pd.notna(row["max_walk_length"])
            else ""
        )
        print(f"    batch_size: {row.get('training.batch_size', 'N/A')}")
        print(f"    lr: {row.get('training.lr', 'N/A')}")

    return df_valid.head(top_n)


def save_trial_details(
    dataset_name,
    trial_id,
    outputs_root="/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs",
):
    """Save all hyperparams for a specific trial to JSON for reproducibility."""
    root = Path(outputs_root) / dataset_name

    # Find study with this trial
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

            for t in study.trials:
                if t.number == trial_id and t.state.name == "COMPLETE":
                    # Found it
                    trial_dict = {
                        "trial_id": trial_id,
                        "dataset": dataset_name,
                        "score": t.value,
                        "params": t.params,
                        "study_name": study.study_name,
                        "exp_folder": exp_folder.name,
                    }

                    # Save to file
                    output_file = (
                        exp_folder / "optuna" / f"trial_{trial_id}_params.json"
                    )
                    with open(output_file, "w") as f:
                        json.dump(trial_dict, f, indent=2)

                    print(f"✅ Saved trial {trial_id} params to {output_file}")
                    return trial_dict
        except:
            continue

    print(f"❌ Trial {trial_id} not found in {dataset_name}")
    return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print(
            "  python extract_top_trials.py <dataset_name> [--metric num_walks|max_walk_length] [--tolerance 5]"
        )
        print("\nExamples:")
        print("  python extract_top_trials.py wiki-rfa")
        print(
            "  python extract_top_trials.py epinions --metric num_walks --tolerance 2"
        )
        print("  python extract_top_trials.py slashdot090221 --metric max_walk_length")
        sys.exit(1)

    dataset = sys.argv[1]
    metric = "num_walks"
    tolerance = 5.0

    # Parse args
    for i, arg in enumerate(sys.argv[2:]):
        if arg == "--metric" and i + 3 < len(sys.argv):
            metric = sys.argv[i + 3]
        elif arg == "--tolerance" and i + 3 < len(sys.argv):
            tolerance = float(sys.argv[i + 3])

    try:
        result = get_alternative_top_trials(
            dataset, metric=metric, tolerance_pct=tolerance, top_n=10
        )
        if result is not None:
            print(f"\n💾 Full results saved above")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
