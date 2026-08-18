"""Paired significance test for Panel D (4-way sign-agreement AUC), per the professor's
C-ter ask ("please have statistical tests for D and e"). Panel E already has one (the
>=8/10-seed BH-FDR robustness convention); this is Panel D's.

Claim Panel D visualizes: AUC is higher when a target edge's neighbor edges agree with its
own sign ("same") than when they disagree ("diff") -- shown separately for the target's
in-edge neighbors and out-edge neighbors. Test: per dataset, one-sided paired Wilcoxon
signed-rank test across the 10 seeds (AUC_same - AUC_diff > 0), for in and out separately --
same test family already used for Table 1 (table1_paired_significance.py) and the
entropy-asymmetry claim in 6.1(C), so this matches the paper's existing convention rather
than introducing a new one.

Reuses the already-cached raw SiGAT (u,v,y,p) predictions and bucketing logic from
extract_empconf_panelD_signagreement_auc.py -- no refit, no new predictions, just a
different reduction of data that already exists on disk.
"""
import csv
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baselines"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from src.utils.config import load_config  # noqa: E402
from src.data.prepare_data import get_edge_list  # noqa: E402
from scripts.paper_figures.extract_multiseed_entropy_heatmaps import SEEDS  # noqa: E402
from scripts.paper_figures.extract_empconf_panelD_signagreement_auc import (  # noqa: E402
    DATASETS, build_adjacency, bucket_predictions, safe_auc, sigat_raw_seed_cached,
)

OUT_CSV = "aaai2027/figure_data/panelD_paired_significance.csv"

_cfg_cache = {}


def load_cfg(ds):
    if ds not in _cfg_cache:
        _cfg_cache[ds] = load_config(overrides=[f"dataset.name={ds}"])
    return _cfg_cache[ds]


def main():
    rows_out = []
    for ds in DATASETS:
        edges = get_edge_list(load_cfg(ds))
        in_edges_of, out_edges_of = build_adjacency(edges)

        per_seed_auc = {b: {} for b in ("in_same", "in_diff", "out_same", "out_diff")}
        for seed in SEEDS:
            rec = sigat_raw_seed_cached(ds, seed)
            if rec is None or not len(rec["u"]):
                continue
            buckets = bucket_predictions(rec, in_edges_of, out_edges_of)
            for b, (y, p) in buckets.items():
                a = safe_auc(y, p)
                if not np.isnan(a):
                    per_seed_auc[b][seed] = a

        for direction, same_k, diff_k in (("in", "in_same", "in_diff"), ("out", "out_same", "out_diff")):
            shared_seeds = sorted(set(per_seed_auc[same_k]) & set(per_seed_auc[diff_k]))
            same_vals = np.array([per_seed_auc[same_k][s] for s in shared_seeds])
            diff_vals = np.array([per_seed_auc[diff_k][s] for s in shared_seeds])
            n = len(shared_seeds)
            if n >= 2 and not np.allclose(same_vals, diff_vals):
                stat, p_value = wilcoxon(same_vals, diff_vals, alternative="greater")
            else:
                p_value = float("nan")
            mean_delta = float(np.mean(same_vals - diff_vals)) if n else float("nan")
            rows_out.append({
                "dataset": ds, "direction": direction, "n_seeds": n,
                "mean_auc_same": float(np.mean(same_vals)) if n else "",
                "mean_auc_diff": float(np.mean(diff_vals)) if n else "",
                "mean_delta": mean_delta, "p_value": p_value,
            })
            print(f"{ds:<15} {direction:<4} same={np.mean(same_vals):.4f} diff={np.mean(diff_vals):.4f} "
                  f"delta={mean_delta:+.4f} p={p_value:.4g} (n={n})")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "direction", "n_seeds", "mean_auc_same",
                                           "mean_auc_diff", "mean_delta", "p_value"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
