"""Figure 3 (delta heatmap) general-difference test -- companion to
table1_paired_significance.py, but always against SiGAT specifically (not each dataset's
best baseline), since Figure 3 itself is a Pewter-vs-SiGAT comparison throughout. Answers
"is Pewter significantly better than SiGAT overall" as a per-dataset paired test across the
10 shared seeds, before the entropy-bin regression (delta_heatmap_entropy_regression.py)
addresses whether that gap grows with entropy.
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
SIGAT_SCORE_ROOT = "baselines/SGA/results_our_splits_canonical"

OUT_CSV = "aaai2027/figure_data/figure3_pewter_vs_sigat_significance.csv"

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


def sigat_seed_auc(ds, seed):
    path = os.path.join(SIGAT_SCORE_ROOT, ds, "SiGAT", f"seed{seed}", "score.csv")
    if not os.path.exists(path):
        return None
    row = next(csv.DictReader(open(path)))
    return float(row["tst_auc"])


def main():
    rows_out = []
    for ds in DATASETS:
        pewter_by_variant = {}
        for variant in ["full", "local"]:
            vals = [pewter_seed_auc(ds, variant, s) for s in SEEDS]
            if any(v is None for v in vals):
                missing = [s for s, v in zip(SEEDS, vals) if v is None]
                print(f"{ds} pewter/{variant}: MISSING seeds {missing}")
                continue
            pewter_by_variant[variant] = np.array(vals)

        sigat_vals = [sigat_seed_auc(ds, s) for s in SEEDS]
        if any(v is None for v in sigat_vals):
            missing = [s for s, v in zip(SEEDS, sigat_vals) if v is None]
            print(f"{ds} SiGAT: MISSING seeds {missing}")
            continue
        sigat_arr = np.array(sigat_vals)

        # local attention is what Figure 3 itself compares against SiGAT
        variant = "local"
        pewter_arr = pewter_by_variant[variant]

        diff = pewter_arr - sigat_arr
        stat, p = wilcoxon(diff, alternative="greater")
        print(f"\n{ds}: Pewter({variant}) mean={pewter_arr.mean():.4f}  "
              f"SiGAT mean={sigat_arr.mean():.4f}  "
              f"median diff={np.median(diff):+.4f}  wins={int((diff > 0).sum())}/10  "
              f"one-sided Wilcoxon p={p:.4g}")
        rows_out.append({
            "dataset": ds, "pewter_variant": variant,
            "pewter_mean": pewter_arr.mean(), "sigat_mean": sigat_arr.mean(),
            "median_diff": np.median(diff), "n_wins": int((diff > 0).sum()), "p_value": p,
        })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "pewter_variant", "pewter_mean",
                                           "sigat_mean", "median_diff", "n_wins", "p_value"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nsaved {OUT_CSV}")
    n_sig = sum(1 for r in rows_out if r["p_value"] < 0.05)
    print(f"\nsignificant (p<0.05) on {n_sig}/{len(rows_out)} datasets")


if __name__ == "__main__":
    main()
