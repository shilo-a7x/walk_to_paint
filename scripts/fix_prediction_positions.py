#!/usr/bin/env python3
"""
Fix saved prediction PKLs by converting positions/walk_lengths to EDGE counts
and recomputing dist_from_start/dist_from_end accordingly.

This updates files in-place (with optional .bak backups).
"""

import argparse
import os
import pickle
from pathlib import Path


def convert_arrays(data):
    """Convert sequence-based positions/lengths to edge-based counts."""
    if "positions" not in data or "walk_lengths" not in data:
        return False

    pos_seq = data["positions"]
    len_seq = data["walk_lengths"]

    # Edge positions are at odd indices in the sequence
    # position_in_edges = (position - 1) // 2
    # total_edges = (walk_length - 1) // 2
    pos_edges = (pos_seq - 1) // 2
    len_edges = (len_seq - 1) // 2

    data["positions"] = pos_edges
    data["walk_lengths"] = len_edges
    data["dist_from_start"] = pos_edges
    data["dist_from_end"] = len_edges - pos_edges - 1

    return True


def process_file(path: Path, backup: bool) -> bool:
    """Update a single PKL file in place."""
    with path.open("rb") as f:
        data = pickle.load(f)

    changed = convert_arrays(data)
    if not changed:
        return False

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            with backup_path.open("wb") as f:
                pickle.dump(data, f)

    with path.open("wb") as f:
        pickle.dump(data, f)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fix prediction PKL positions to edge-based distances"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="outputs",
        help="Root directory to search for *_predictions PKLs (default: outputs)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .bak backups before overwriting files",
    )
    args = parser.parse_args()

    root = Path(args.root)
    pkl_files = list(root.glob("**/*_predictions/epoch_*/**/*_predictions.pkl"))

    if not pkl_files:
        print("No prediction PKLs found.")
        return

    updated = 0
    for pkl in pkl_files:
        try:
            if process_file(pkl, args.backup):
                updated += 1
                print(f"Updated: {pkl}")
        except Exception as e:
            print(f"Failed: {pkl} ({e})")

    print(f"\nDone. Updated {updated} file(s).")


if __name__ == "__main__":
    main()
