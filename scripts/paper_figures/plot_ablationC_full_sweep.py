"""Plot for Ablation C -- uniform mean vs. production choice (func_logit_power)
vs. best-of-full-sweep (all ~39 func_ forms, no lgbm/attention), all 6 datasets.

Reads aaai2027/figure_data/ablationC_summary3.csv (built by
extract_ablationC_full_sweep.py from the real run_posthoc.py sweep). SE via
Hanley-McNeil on real per-dataset test counts.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanley_mcneil import auc_se

IN_CSV = "aaai2027/figure_data/ablationC_summary3.csv"
COUNTS_CSV = "aaai2027/figure_data/test_set_counts.csv"
OUT_PNG = "aaai2027/figures/ablationC_full_sweep.png"

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
DISPLAY_LABEL = {"slashdot090221": "slashdot"}
COLORS = ["#eb6834", "#2a78d6", "#1baf7a"]
LABELS = ["Uniform mean", "func_logit_power (production)", "Best of 39 func_ forms"]


def load():
    rows = {r["dataset"]: r for r in csv.DictReader(open(IN_CSV))}
    counts = {r["dataset"]: r for r in csv.DictReader(open(COUNTS_CSV))}
    series = [[], [], []]
    se_series = [[], [], []]
    best_models = []
    for ds in DATASET_ORDER:
        r = rows[ds]
        n_pos, n_neg = int(counts[ds]["n_pos"]), int(counts[ds]["n_neg"])
        vals = [float(r["uniform_auc"]), float(r["logit_power_auc"]), float(r["best_auc"])]
        for i, v in enumerate(vals):
            series[i].append(v)
            se_series[i].append(auc_se(v, n_pos, n_neg))
        best_models.append(r["best_model"])
    return series, se_series, best_models


def main():
    series, se_series, best_models = load()
    x = np.arange(len(DATASET_ORDER))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.0, 3.6))

    for i, (vals, ses, color, label) in enumerate(zip(series, se_series, COLORS, LABELS)):
        offset = (i - 1) * w
        ax.bar(x + offset, vals, width=w, yerr=ses, capsize=2.5, color=color, label=label)

    for i, ds in enumerate(DATASET_ORDER):
        ax.text(x[i] + w, series[2][i] + se_series[2][i] + 0.002, best_models[i],
                ha="center", va="bottom", fontsize=5.5, rotation=90, color="#3a3a38")

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABEL.get(d, d) for d in DATASET_ORDER], fontsize=8, rotation=15)
    ax.set_ylabel("test AUC")
    ax.set_ylim(0.85, 1.0)
    ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
