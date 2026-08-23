"""Ablations abl:masknode / abl:maskedge -- corrected numbers, post posthoc-eval bug fix.

Bug (2026-08-23, see aaai2027/PAPER_CLOSEOUT_LOG.md): PerEpochPredictionSaver._extract_predictions
(src/training/callbacks.py) never reapplied model.mask_node_tokens/mask_edge_tokens before the
forward pass at posthoc-eval time, so these two ablations were trained correctly-ablated but
evaluated with the masked token type fully visible. Fixed, then all 120 checkpoints (6 datasets
x 10 seeds x 2 ablations) were re-evaluated (posthoc only, no retraining) via
scripts/rerun_ablation_posthoc_fix.py, run_id="ablation_agg". This script reads the corrected
per-seed AUCs and recomputes the paragraph's delta/significance numbers, same pattern as
extract_ablationA_full_vs_local_significance.py.
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
DISPLAY = {
    "bitcoin-alpha": "Bitcoin-alpha", "bitcoin-otc": "Bitcoin-otc", "epinions": "Epinions",
    "wiki-elec": "Wiki-elec", "wiki-rfa": "Wiki-RfA", "slashdot090221": "Slashdot",
}
SEEDS = list(range(42, 52))
SEED42_LOCAL_TAG = "E32_PY314_LOCALATTN4"
ABLATIONS = ["MASKNODE", "MASKEDGE"]

OUT_CSV = "aaai2027/figure_data/ablation_masknode_maskedge_significance.csv"
_AUC_RE = re.compile(r"Test\s+AUC:\s+([0-9.]+)\s+\((\d+)\s+edges\)")


def _read_auc(summary_path):
    m = _AUC_RE.search(open(summary_path).read())
    return float(m.group(1)) if m else None


def local_seed_auc(ds, seed):
    if seed == 42:
        pattern = f"outputs/{ds}/{SEED42_LOCAL_TAG}_*"
    else:
        pattern = f"outputs/{ds}/MULTISEED_s{seed}_local_*"
    for run_dir in reversed(sorted(glob.glob(pattern))):
        for path in sorted(glob.glob(os.path.join(
                run_dir, "posthoc", "*", "aggregator", "func_logit_power", "summary.txt"))):
            auc = _read_auc(path)
            if auc is not None:
                return auc
    return None


def ablation_seed_auc(ds, tag, seed):
    pattern = f"outputs/{ds}/ABLATION_{tag}_s{seed}_*"
    for run_dir in reversed(sorted(glob.glob(pattern))):
        path = os.path.join(run_dir, "posthoc", "ablation_agg", "aggregator",
                             "func_logit_power", "summary.txt")
        if os.path.isfile(path):
            return _read_auc(path)
    return None


def main():
    rows_out = []
    print(f"{'dataset':16s} {'local':>8s} {'masknode':>10s}  {'delta(pp)':>10s} {'p(one-sided)':>13s}   "
          f"{'maskedge':>10s}  {'delta(pp)':>10s} {'p(one-sided)':>13s}")
    for ds in DATASETS:
        local_vals = np.array([local_seed_auc(ds, s) for s in SEEDS])
        masknode_vals = np.array([ablation_seed_auc(ds, "MASKNODE", s) for s in SEEDS])
        maskedge_vals = np.array([ablation_seed_auc(ds, "MASKEDGE", s) for s in SEEDS])
        assert not any(v is None for v in local_vals), f"{ds}: missing local seeds"
        assert not any(v is None for v in masknode_vals), f"{ds}: missing masknode seeds"
        assert not any(v is None for v in maskedge_vals), f"{ds}: missing maskedge seeds"

        d_node = masknode_vals - local_vals
        d_edge = maskedge_vals - local_vals
        _, p_node = wilcoxon(d_node, alternative="less")
        _, p_edge = wilcoxon(d_edge, alternative="less")

        print(f"{DISPLAY[ds]:16s} {local_vals.mean():8.4f} {masknode_vals.mean():10.4f}  "
              f"{100 * d_node.mean():+10.2f} {p_node:13.4g}   "
              f"{maskedge_vals.mean():10.4f}  {100 * d_edge.mean():+10.2f} {p_edge:13.4g}")

        rows_out.append({
            "dataset": ds, "display_name": DISPLAY[ds],
            "local_mean": local_vals.mean(), "local_std": local_vals.std(ddof=1),
            "masknode_mean": masknode_vals.mean(), "masknode_std": masknode_vals.std(ddof=1),
            "masknode_delta_pp": 100 * d_node.mean(), "masknode_p_one_sided": p_node,
            "maskedge_mean": maskedge_vals.mean(), "maskedge_std": maskedge_vals.std(ddof=1),
            "maskedge_delta_pp": 100 * d_edge.mean(), "maskedge_p_one_sided": p_edge,
        })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
