"""Table 1 paired significance test -- STATISTICAL_TESTS_AUDIT.md item #1 ("Pewter beats
the best baseline on all six datasets"). Currently backed only by two point estimates
(mean+-std over 10 splits each) with no formal paired comparison -- doesn't exploit that
both models are scored on the SAME 10 splits (a paired design, much more powerful than
treating the two sets of 10 numbers as independent samples).

Best non-Pewter baseline per dataset, read directly off Table 1's current numbers:
    bitcoin-alpha: SNEA (0.871)      bitcoin-otc: SNEA (0.890)
    epinions: SiGAT (0.909)          wiki-elec: SiGAT (0.888)
    wiki-rfa: SiGAT (0.878)          slashdot090221: SiGAT (0.859)
Both SNEA and SiGAT have confirmed real 10-seed campaigns (audit item #1) -- GS-GNN/SGCN's
uncertain per-seed status never needs to be resolved, since neither is the runner-up on
any dataset.

Per-seed source data:
  Pewter:  outputs/<ds>/<run>/posthoc/*/aggregator/func_logit_power/summary.txt
           seed 42 pinned to E31_PY314_MIGRATION (full) / E32_PY314_LOCALATTN4 (local) --
           same tags extract_result1_walk_aucs.py already uses for this exact reason (the
           multiseed campaign's own docstring, run_multiseed_pewter.py, confirms seed 42
           was "already trained, reused/backfilled" from these two runs, not from the
           older pre-migration E25/E26 checkpoints attention_directionality.py points to
           for a different figure). Seeds 43-51: globs MULTISEED_s<seed>_<variant>_*,
           latest timestamped dir with a valid summary (same pattern
           extract_multiseed_entropy_heatmaps.py's walk_raw_seed() uses for reruns).
  SNEA / SiGAT: baselines/{CopulaLSP,SGA}/results_our_splits_canonical/<ds>/<model>/
           seed<N>/score.csv, `tst_auc` column.

For each dataset: pick Pewter's Table-1-winning variant (whichever of full/local has the
higher 10-seed mean -- printed as a consistency check against Table 1's own bold marks),
then run a one-sided paired Wilcoxon signed-rank test (Pewter AUC > baseline AUC) across
the 10 shared seeds.
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
BEST_BASELINE = {
    "bitcoin-alpha": "SNEA", "bitcoin-otc": "SNEA",
    "epinions": "SiGAT", "wiki-elec": "SiGAT", "wiki-rfa": "SiGAT", "slashdot090221": "SiGAT",
}
BASELINE_SCORE_ROOT = {
    "SiGAT": "baselines/SGA/results_our_splits_canonical",
    "SNEA": "baselines/CopulaLSP/results_our_splits_canonical",
}

OUT_CSV = "aaai2027/figure_data/table1_paired_significance.csv"

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
        pewter_by_variant = {}
        for variant in ["full", "local"]:
            vals = [pewter_seed_auc(ds, variant, s) for s in SEEDS]
            if any(v is None for v in vals):
                missing = [s for s, v in zip(SEEDS, vals) if v is None]
                print(f"{ds} pewter/{variant}: MISSING seeds {missing}")
                continue
            pewter_by_variant[variant] = np.array(vals)

        baseline_vals = [baseline_seed_auc(baseline, ds, s) for s in SEEDS]
        if any(v is None for v in baseline_vals):
            missing = [s for s, v in zip(SEEDS, baseline_vals) if v is None]
            print(f"{ds} {baseline}: MISSING seeds {missing}")
            continue
        baseline_arr = np.array(baseline_vals)

        winning_variant = max(pewter_by_variant, key=lambda v: pewter_by_variant[v].mean())
        pewter_arr = pewter_by_variant[winning_variant]

        diff = pewter_arr - baseline_arr
        stat, p = wilcoxon(diff, alternative="greater")
        print(f"\n{ds}: Pewter({winning_variant}) mean={pewter_arr.mean():.4f}  "
              f"{baseline} mean={baseline_arr.mean():.4f}  "
              f"median diff={np.median(diff):+.4f}  wins={int((diff > 0).sum())}/10  "
              f"one-sided Wilcoxon p={p:.4g}")
        rows_out.append({
            "dataset": ds, "pewter_variant": winning_variant,
            "pewter_mean": pewter_arr.mean(), "baseline": baseline, "baseline_mean": baseline_arr.mean(),
            "median_diff": np.median(diff), "n_wins": int((diff > 0).sum()), "p_value": p,
        })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "pewter_variant", "pewter_mean", "baseline",
                                           "baseline_mean", "median_diff", "n_wins", "p_value"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nsaved {OUT_CSV}")
    n_sig = sum(1 for r in rows_out if r["p_value"] < 0.05)
    print(f"\nsignificant (p<0.05) on {n_sig}/{len(rows_out)} datasets")


if __name__ == "__main__":
    main()
