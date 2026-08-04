"""Plot for Result 2 -- dedicated 4-row kernel-smoothed AUC-vs-entropy
heatmap: Pewter (full attn), Pewter (LocalAttn4), SiGAT, GINEConv.

Pure rendering: reads aaai2027/figure_data/result2_entropy_heatmap.csv (built
by extract_result2_entropy_heatmap.py). Edit THIS file freely for
colormap/scale/style changes -- no recomputation needed.

Layout: 4 rows (models) x 6 columns (datasets).
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_CSV = "aaai2027/figure_data/result2_entropy_heatmap.csv"
OUT_PNG = "aaai2027/figures/result2_entropy_heatmap.png"

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
MODELS = ["Pewter (full attn)", "Pewter (LocalAttn4)", "SiGAT", "GINEConv"]
VMIN, VMAX = 0.5, 1.0
DISPLAY_LABEL = {"slashdot090221": "slashdot", "SiGAT": "SiGAT"}


def load(path):
    grid_vals = sorted({float(row["src_ent"]) for row in csv.DictReader(open(path))})
    n = len(grid_vals)
    idx = {v: i for i, v in enumerate(grid_vals)}
    data = {(ds, m): np.full((n, n), np.nan) for ds in DATASETS for m in MODELS}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], row["model"])
            if key not in data:
                continue
            i = idx[float(row["src_ent"])]
            j = idx[float(row["tgt_ent"])]
            auc = row["auc"]
            data[key][j, i] = float(auc) if auc != "" and auc.lower() != "nan" else np.nan
    return data, grid_vals


def main():
    data, grid_vals = load(IN_CSV)
    n_rows, n_cols = len(MODELS), len(DATASETS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.9 * n_cols, 1.9 * n_rows), sharex=True, sharey=True)

    im = None
    for r, model in enumerate(MODELS):
        for c, ds in enumerate(DATASETS):
            ax = axes[r, c]
            grid = data[(ds, model)]
            im = ax.imshow(grid, origin="lower", extent=[0, 1, 0, 1], aspect="auto",
                            cmap="RdYlGn", vmin=VMIN, vmax=VMAX)
            if r == 0:
                ax.set_title(DISPLAY_LABEL.get(ds, ds), fontsize=8)
            if c == 0:
                ax.set_ylabel(DISPLAY_LABEL.get(model, model), fontsize=8)
            ax.tick_params(labelsize=6)
            if r == n_rows - 1:
                ax.set_xlabel("src", fontsize=7)

    fig.suptitle("Test AUC vs. source out-entropy (rater consistency) and target in-entropy (reputation contestedness)",
                  fontsize=9)
    fig.text(0.005, 0.5, "target\nin-ent.", va="center", rotation="vertical", fontsize=7)
    fig.tight_layout(rect=(0.02, 0, 0.93, 0.95))
    cbar_ax = fig.add_axes((0.94, 0.15, 0.015, 0.7))
    fig.colorbar(im, cax=cbar_ax, label="AUC")

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
