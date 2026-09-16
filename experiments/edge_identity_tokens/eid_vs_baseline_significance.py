"""Thesis prep (Phase 3a): EID vs. best non-Pewter baseline, one-sided paired Wilcoxon over
the 10 shared seeds -- the EID-as-production analogue of
`scripts/paper_figures/table1_paired_significance.py`. Reuses that script's baseline-side
lookup (SNEA/SiGAT per-seed `score.csv`, `tst_auc` column, unaffected by the Pewter->EID
swap) and `eid_significance.py`'s EID-side lookup (`eid_seed_auc`).

Note (2026-09-15): this compares EID against the same best-baseline picks the current
`aaai2027/` paper uses (SNEA on bitcoin-alpha/bitcoin-otc, SiGAT elsewhere) -- per the plan,
EID doesn't need to "beat production" as a thesis finding, but reproducing Table 1's own
"beats every baseline" significance claim under the new architecture is still part of the 3a
reproduction (same headline table, new model instantiating it).

Usage:
  .venv/bin/python experiments/edge_identity_tokens/eid_vs_baseline_significance.py
"""
import csv
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import experiments.edge_identity_tokens.eid_significance as sig  # noqa: E402

DATASETS = sig.DATASETS
SEEDS = sig.SEEDS

# Same picks as table1_paired_significance.py's BEST_BASELINE -- read directly off Table 1's
# current numbers, not re-derived here (unaffected by the Pewter->EID swap).
BEST_BASELINE = {
    "bitcoin-alpha": "SNEA", "bitcoin-otc": "SNEA",
    "epinions": "SiGAT", "wiki-elec": "SiGAT", "wiki-rfa": "SiGAT", "slashdot090221": "SiGAT",
}
BASELINE_SCORE_ROOT = {
    "SiGAT": "baselines/SGA/results_our_splits_canonical",
    "SNEA": "baselines/CopulaLSP/results_our_splits_canonical",
}

OUT_CSV = "experiments/edge_identity_tokens/thesis_figure_data/eid_vs_baseline_significance.csv"


def baseline_seed_auc(model, ds, seed):
    path = os.path.join(BASELINE_SCORE_ROOT[model], ds, model, f"seed{seed}", "score.csv")
    if not os.path.exists(path):
        return None
    row = next(csv.DictReader(open(path)))
    return float(row["tst_auc"])


def main():
    rows_out = []
    for ds in DATASETS:
        baseline = BEST_BASELINE[ds]
        eid_vals = [sig.eid_seed_auc(ds, "noablation", s) for s in SEEDS]
        if any(v is None for v in eid_vals):
            missing = [s for s, v in zip(SEEDS, eid_vals) if v is None]
            print(f"{ds}: EID MISSING seeds {missing}")
            continue
        baseline_vals = [baseline_seed_auc(baseline, ds, s) for s in SEEDS]
        if any(v is None for v in baseline_vals):
            missing = [s for s, v in zip(SEEDS, baseline_vals) if v is None]
            print(f"{ds}: {baseline} MISSING seeds {missing}")
            continue

        eid_arr = np.array(eid_vals)
        baseline_arr = np.array(baseline_vals)
        diff = eid_arr - baseline_arr
        stat, p = wilcoxon(diff, alternative="greater")
        print(f"{ds}: EID mean={eid_arr.mean():.4f}  {baseline} mean={baseline_arr.mean():.4f}  "
              f"median diff={np.median(diff):+.4f}  wins={int((diff > 0).sum())}/{len(diff)}  "
              f"one-sided Wilcoxon p={p:.4g}")
        rows_out.append({
            "dataset": ds, "eid_mean": eid_arr.mean(), "baseline": baseline,
            "baseline_mean": baseline_arr.mean(), "median_diff": np.median(diff),
            "n_wins": int((diff > 0).sum()), "p_value": p,
        })

    if not rows_out:
        print("\nNo complete datasets -- nothing to summarize.")
        return

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "eid_mean", "baseline", "baseline_mean",
                                           "median_diff", "n_wins", "p_value"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nsaved {OUT_CSV}")
    n_sig = sum(1 for r in rows_out if r["p_value"] < 0.05)
    print(f"significant (p<0.05) on {n_sig}/{len(rows_out)} datasets")


if __name__ == "__main__":
    main()
