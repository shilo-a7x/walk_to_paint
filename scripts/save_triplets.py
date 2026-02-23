#!/usr/bin/env python3
"""
Save per-occurrence triplets for aggregation analysis.

Triplet fields per occurrence:
  (dist_from_start, dist_from_end, correct_flag)

Input:
  outputs/predictions/<dataset>/raw_scores/<run_id>_<split>.pkl

Output:
  outputs/aggregation/<dataset>/strategy_mean/triplets_<split>.pkl
"""

import argparse
import os
import pickle


def load_predictions(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_triplets(payload):
    pred_label = payload.get("pred_label", [])
    true_label = payload.get("true_label", [])

    if len(pred_label) != len(true_label):
        raise ValueError(
            f"pred_label and true_label length mismatch: {len(pred_label)} vs {len(true_label)}"
        )

    correct = [int(p == t) for p, t in zip(pred_label, true_label)]

    triplets = {
        "edge_id": payload.get("edge_id", []),
        "walk_id": payload.get("walk_id", []),
        "position": payload.get("position", []),
        "walk_len": payload.get("walk_len", []),
        "dist_from_start": payload.get("dist_from_start", []),
        "dist_from_end": payload.get("dist_from_end", []),
        "correct": correct,
    }

    lengths = [len(v) for v in triplets.values()]
    if len(set(lengths)) != 1:
        raise ValueError(f"Triplet field length mismatch: {lengths}")

    return triplets


def save_triplets(triplets, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(triplets, f)


def main():
    parser = argparse.ArgumentParser(
        description="Save per-occurrence triplets from prediction outputs"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--run-id", type=str, required=True, help="Run identifier")
    parser.add_argument(
        "--splits",
        type=str,
        default="val,test",
        help="Comma-separated splits to process (val,test)",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default="outputs/predictions",
        help="Base directory for prediction inputs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/aggregation",
        help="Base directory for triplet outputs",
    )

    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise ValueError("No splits provided")

    for split in splits:
        input_path = os.path.join(
            args.predictions_dir,
            args.dataset,
            "raw_scores",
            f"{args.run_id}_{split}.pkl",
        )
        # Include run_id in output path to avoid overwriting
        output_path = os.path.join(
            args.output_dir,
            args.dataset,
            "strategy_mean",
            args.run_id,
            f"triplets_{split}.pkl",
        )

        payload = load_predictions(input_path)
        triplets = build_triplets(payload)
        save_triplets(triplets, output_path)

        print(
            f"Saved {len(triplets['correct'])} triplets for {split} to: {output_path}"
        )


if __name__ == "__main__":
    main()
