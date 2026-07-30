"""Extract step for Empirical Confirmation Panel A -- entropy asymmetry
(H_out vs. H_in per node), six datasets. Fills the previously-deferred Panel A
slot in the 3-panel Empirical Confirmation figure (checklist #12) with a real
boxplot instead of text-only prose.

Reuses the exact same per-node entropy computation already used for the
paper's Panel A paragraph numbers (build_sign_dicts/entropy_lookup from
lead4_entropy_heterogeneity.py, same pairing rule as
lead4c_directionality_answers.py::claim1_for_dataset -- a node counts only if
it has both an out-degree and an in-degree, i.e. appears as both a source and
a target somewhere in the dataset), so the boxplot is consistent with the
prose numbers already cited in the text (see LEAD4C_DIRECTIONALITY_ANSWERS.md
and outputs/lead4c_entropy_logit_regression/directionality_answers/claim1_table.csv).

Adds a paired t-test (mean H_out - H_in, per dataset) alongside the existing
Wilcoxon signed-rank result -- the t-test is what the plot caption reports,
the Wilcoxon/rank-biserial numbers already used in the paper prose are left
untouched (they're the more appropriate test given how non-normal/zero-heavy
these entropy distributions are; the t-test is included because it's the
more familiar readout for a boxplot caption, not a replacement).

Output: aaai2027/figure_data/empconf_panelA_entropy_boxplot.csv, one row per
(dataset, node, direction) -- long format, sufficient for a real boxplot (not
just quartiles) and for recomputing any summary stat downstream.
Also writes aaai2027/figure_data/empconf_panelA_entropy_ttest.csv, one row
per dataset with the paired t-test (and Wilcoxon, for cross-reference).
"""
import csv
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.node_mi_structural_embedding import DATASET_CONFIGS, load_edges_canonical
from scripts.lead4_entropy_heterogeneity import build_sign_dicts, entropy_lookup, _ds_key

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
OUT_LONG_CSV = "aaai2027/figure_data/empconf_panelA_entropy_boxplot.csv"
OUT_STATS_CSV = "aaai2027/figure_data/empconf_panelA_entropy_ttest.csv"


def per_dataset(ds_name):
    edges = load_edges_canonical(DATASET_CONFIGS[_ds_key(ds_name)]["ds_name"])
    sign_dicts = build_sign_dicts(edges)
    ent = entropy_lookup(sign_dicts)
    out_e, in_e = ent["out"], ent["in"]
    nodes = [n for n in out_e if n in in_e]
    h_out = np.array([out_e[n] for n in nodes])
    h_in = np.array([in_e[n] for n in nodes])
    return nodes, h_out, h_in


def main():
    long_rows = []
    stat_rows = []
    for ds in DATASETS:
        nodes, h_out, h_in = per_dataset(ds)
        diff = h_out - h_in

        t_stat, t_p = stats.ttest_rel(h_out, h_in)
        tied = diff == 0
        d_nontied = diff[~tied]
        # one-sided, in whichever direction the data actually points -- matches
        # lead4c_directionality_answers.py::claim1_for_dataset's convention exactly
        # (that script tests "less" for the confirm direction, "greater" for reversed),
        # so this reproduces claim1_table.csv's p_confirm_direction/p_reversed_direction
        # numbers already cited in the paper's prose, rather than a different two-sided p.
        if len(d_nontied):
            alt = "less" if h_out.mean() < h_in.mean() else "greater"
            w_stat, w_p = stats.wilcoxon(d_nontied, alternative=alt)
        else:
            w_stat, w_p = float("nan"), float("nan")

        print(f"{ds}: n={len(nodes)} mean_H_out={h_out.mean():.4f} mean_H_in={h_in.mean():.4f} "
              f"paired-t={t_stat:.3f} p={t_p:.3e}")

        for n, ho, hi in zip(nodes, h_out, h_in):
            long_rows.append({"dataset": ds, "node": n, "direction": "out", "entropy": ho})
            long_rows.append({"dataset": ds, "node": n, "direction": "in", "entropy": hi})

        stat_rows.append({
            "dataset": ds, "n_nodes": len(nodes),
            "mean_H_out": float(h_out.mean()), "mean_H_in": float(h_in.mean()),
            "median_H_out": float(np.median(h_out)), "median_H_in": float(np.median(h_in)),
            "t_stat": float(t_stat), "t_pvalue": float(t_p),
            "wilcoxon_stat": float(w_stat), "wilcoxon_pvalue": float(w_p),
        })

    os.makedirs(os.path.dirname(OUT_LONG_CSV), exist_ok=True)
    with open(OUT_LONG_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "node", "direction", "entropy"])
        w.writeheader()
        w.writerows(long_rows)
    print(f"wrote {len(long_rows)} rows to {OUT_LONG_CSV}")

    with open(OUT_STATS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "n_nodes", "mean_H_out", "mean_H_in",
                                           "median_H_out", "median_H_in",
                                           "t_stat", "t_pvalue", "wilcoxon_stat", "wilcoxon_pvalue"])
        w.writeheader()
        w.writerows(stat_rows)
    print(f"wrote {len(stat_rows)} rows to {OUT_STATS_CSV}")


if __name__ == "__main__":
    main()
