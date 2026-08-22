"""Forest plot summarizing delta_heatmap_entropy_regression.py's per-dataset results --
NOT a paper figure, a reviewing aid so the user can show their professor the "gain
concentrates where the bound bites" test result while deciding how (or whether) to state
it in the paper. Pure rendering: reads
aaai2027/figure_data/delta_heatmap_entropy_regression.csv.

One row per (dataset, axis): dot = regression coefficient (how much the Pewter-minus-
SiGAT AUC delta changes per unit of that entropy axis), whisker = 95% CI (coefficient +-
1.96 x cluster-robust SE, cluster=seed), filled marker = significant (p<0.05), open
marker = not. A coefficient/CI entirely to the right of zero means that axis's entropy
predicts a GROWING Pewter advantage (the paper's current claim); entirely to the left
means the opposite.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN_CSV = "aaai2027/figure_data/delta_heatmap_entropy_regression.csv"
OUT_PNG = "aaai2027/figures/delta_heatmap_entropy_regression_forest.png"

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
DATASET_DISPLAY = {
    "bitcoin-alpha": "Bitcoin-alpha", "bitcoin-otc": "Bitcoin-otc", "epinions": "Epinions",
    "slashdot090221": "Slashdot", "wiki-elec": "Wiki-elec", "wiki-rfa": "Wiki-RfA",
}
AXES = [("src_coef", "src_se", "src_p", "source entropy"), ("tgt_coef", "tgt_se", "tgt_p", "target entropy")]
AXIS_COLOR = {"source entropy": "#c0392b", "target entropy": "#8e44ad"}


def main():
    rows = {r["dataset"]: r for r in csv.DictReader(open(IN_CSV))}

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    y_labels = []
    y = 0
    yticks = []
    for ds in DATASET_ORDER:
        r = rows[ds]
        for coef_k, se_k, p_k, axis_label in AXES:
            coef = float(r[coef_k])
            se = float(r[se_k])
            p = float(r[p_k])
            ci = 1.96 * se
            sig = p < 0.05
            color = AXIS_COLOR[axis_label]
            ax.errorbar(coef, y, xerr=ci, fmt="o", color=color, capsize=3,
                        markerfacecolor=color if sig else "white", markeredgecolor=color,
                        markersize=7, linewidth=1.3, zorder=3)
            y_labels.append(f"{DATASET_DISPLAY[ds]} — {axis_label}")
            yticks.append(y)
            y -= 1
        y -= 0.4  # gap between datasets

    ax.axvline(0, color="black", linewidth=0.9, zorder=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("regression coefficient on Pewter-minus-SiGAT AUC delta\n"
                   "(per unit of entropy, 95% CI; filled = p<0.05)", fontsize=9.5)
    ax.set_title("Does the Pewter advantage grow with entropy? (per dataset, per axis)", fontsize=11)

    handles = [
        plt.Line2D([0], [0], marker="o", color=AXIS_COLOR["source entropy"], markerfacecolor=AXIS_COLOR["source entropy"],
                   linestyle="none", markersize=7, label="source entropy (filled=sig.)"),
        plt.Line2D([0], [0], marker="o", color=AXIS_COLOR["target entropy"], markerfacecolor=AXIS_COLOR["target entropy"],
                   linestyle="none", markersize=7, label="target entropy (filled=sig.)"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
