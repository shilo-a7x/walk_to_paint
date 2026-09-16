"""Thesis prep (plan-eid-multiseed-thesis.md Phase 3a, THESIS_ASSET_TRIAGE.md's "Table 1"
row): EID's headline per-dataset AUC, real 10-seed mean+-std -- the same convention
production's own "Current SOTA" table in CLAUDE.md uses (real split-to-split std, not the
Hanley-McNeil closed-form SE, which CLAUDE.md notes understates true uncertainty).

Source: EID's `noablation` condition, seeds 42 (backfilled where possible) + 43-51, via
`run_eid_multiseed.eid_seed_auc`-equivalent lookup (imported from eid_significance.py, which
already implements this exact resolution logic). No new posthoc runs -- reads only
already-written summary.txt files. Datasets/seeds with missing cells are reported, not
silently dropped or averaged over a partial seed set.

Usage:
  .venv/bin/python experiments/edge_identity_tokens/eid_result1_table.py
"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import experiments.edge_identity_tokens.eid_significance as sig  # noqa: E402

DATASETS = sig.DATASETS
SEEDS = sig.SEEDS

OUT_CSV = "experiments/edge_identity_tokens/thesis_figure_data/eid_result1_table.csv"


def main():
    rows = []
    for ds in DATASETS:
        vals = [sig.eid_seed_auc(ds, "noablation", s) for s in SEEDS]
        missing = [s for s, v in zip(SEEDS, vals) if v is None]
        if missing:
            print(f"{ds:16s} -- INCOMPLETE, missing seeds {missing} ({len(SEEDS) - len(missing)}/{len(SEEDS)} done)")
            rows.append({"dataset": ds, "n_seeds": len(SEEDS) - len(missing),
                         "auc_mean": "", "auc_std": "", "complete": False})
            continue
        arr = np.array(vals)
        print(f"{ds:16s} AUC = {arr.mean():.4f} +/- {arr.std():.4f}  (n={len(arr)} seeds)")
        rows.append({"dataset": ds, "n_seeds": len(arr),
                     "auc_mean": f"{arr.mean():.4f}", "auc_std": f"{arr.std():.4f}", "complete": True})

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "n_seeds", "auc_mean", "auc_std", "complete"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV}")
    n_complete = sum(1 for r in rows if r["complete"])
    print(f"{n_complete}/{len(DATASETS)} datasets complete")


if __name__ == "__main__":
    main()
