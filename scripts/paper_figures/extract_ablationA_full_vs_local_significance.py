"""Ablation A (proximal attention) significance test -- STATISTICAL_TESTS_AUDIT.md item #8.
Replaces the informal "within reported standard errors" eyeball claim with a real paired
test: full vs. local Pewter AUC, same 10 seeds, two-sided (no directional prior -- the
paragraph's own claim is "costs nothing either way", not "one beats the other").
"""
import csv
import glob
import os
import re
import sys

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
SEEDS = [42] + list(range(43, 52))
SEED42_TAG = {"full": "E31_PY314_MIGRATION", "local": "E32_PY314_LOCALATTN4"}

OUT_CSV = "aaai2027/figure_data/ablationA_full_vs_local_significance.csv"
_AUC_RE = re.compile(r"Test\s+AUC:\s+([0-9.]+)\s+\((\d+)\s+edges\)")


def pewter_seed_auc(ds, variant, seed):
    if seed == 42:
        run_dirs = [f"outputs/{ds}/{SEED42_TAG[variant]}_*"]
    else:
        run_dirs = [f"outputs/{ds}/MULTISEED_s{seed}_{variant}_*"]
    matches = sorted(glob.glob(run_dirs[0]))
    for run_dir in reversed(matches):
        summaries = glob.glob(os.path.join(run_dir, "posthoc", "*", "aggregator",
                                            "func_logit_power", "summary.txt"))
        for path in sorted(summaries):
            m = _AUC_RE.search(open(path).read())
            if m:
                return float(m.group(1))
    return None


def main():
    rows_out = []
    for ds in DATASETS:
        full_vals = [pewter_seed_auc(ds, "full", s) for s in SEEDS]
        local_vals = [pewter_seed_auc(ds, "local", s) for s in SEEDS]
        if any(v is None for v in full_vals) or any(v is None for v in local_vals):
            print(f"{ds}: MISSING seeds")
            continue
        full_arr = np.array(full_vals)
        local_arr = np.array(local_vals)
        diff = local_arr - full_arr
        stat, p = wilcoxon(diff, alternative="two-sided")
        print(f"{ds}: full={full_arr.mean():.4f} local={local_arr.mean():.4f} "
              f"median diff(local-full)={np.median(diff):+.4f} "
              f"wins(local>full)={int((diff > 0).sum())}/10 "
              f"two-sided Wilcoxon p={p:.4g}")
        rows_out.append({
            "dataset": ds, "full_mean": full_arr.mean(), "local_mean": local_arr.mean(),
            "median_diff_local_minus_full": np.median(diff),
            "n_local_wins": int((diff > 0).sum()), "p_value": p,
        })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "full_mean", "local_mean",
                                           "median_diff_local_minus_full", "n_local_wins", "p_value"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nsaved {OUT_CSV}")
    n_sig = sum(1 for r in rows_out if r["p_value"] < 0.05)
    print(f"significant (p<0.05) on {n_sig}/{len(rows_out)} datasets")


if __name__ == "__main__":
    main()
