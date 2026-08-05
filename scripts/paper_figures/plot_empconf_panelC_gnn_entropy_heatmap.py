"""Plot step for Empirical Confirmation Panel C -- GNN accuracy vs. entropy,
DISCRETE 4x4 binned heatmap. GNN-only (see extract script docstring for why
-- the walk-vs-GNN comparison lives in its own figure for Result 2 instead).

Pure rendering: reads aaai2027/figure_data/empconf_panelC_gnn_entropy_heatmap.csv
(built by extract_empconf_panelC_gnn_entropy_heatmap.py). Edit THIS file freely
for colormap/scale/style/size changes -- no recomputation needed.

Layout: SiGAT only (per 2026-08-04 call -- GINEConv dropped from this figure,
kept in the appendix/baseline table instead; the extract CSV still has both
models' cells for that reuse), each cell gets an AUC value annotated directly
on it (readable at 4x4 resolution), masked (insufficient-n) cells shown as
hatched grey rather than left blank.

2026-08-05 (user call): re-lined from a single 1x6 row of datasets to a 2x3
grid, and native figsize shrunk to match this panel's actual single-column
display width (~3.3in) -- a 1x6 row at that width would give each of the 6
heatmaps under 0.6in, illegible; 2x3 roughly doubles each cell's width.
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_CSV = "aaai2027/figure_data/empconf_panelC_gnn_entropy_heatmap.csv"
OUT_PNG = "aaai2027/figures/empconf_panelC_gnn_entropy_heatmap.png"

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
MODELS = ["SiGAT"]  # GINEConv dropped from this figure 2026-08-04, kept in appendix/table only
VMIN, VMAX = 0.5, 1.0
N_BINS = 4
DISPLAY_LABEL = {"slashdot090221": "slashdot", "SiGAT": "SiGAT"}


def load(path):
    data = {(ds, m): np.full((N_BINS, N_BINS), np.nan) for ds in DATASETS for m in MODELS}
    ncounts = {(ds, m): np.zeros((N_BINS, N_BINS), dtype=int) for ds in DATASETS for m in MODELS}
    edges = None
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], row["model"])
            if key not in data:
                continue
            lo_s, hi_s = float(row["src_bin_lo"]), float(row["src_bin_hi"])
            lo_t = float(row["tgt_bin_lo"])
            i = round(lo_s * N_BINS)
            j = round(lo_t * N_BINS)
            auc = row["auc"]
            data[key][j, i] = float(auc) if auc != "" and auc.lower() != "nan" else np.nan
            ncounts[key][j, i] = int(row["n"])
            if edges is None:
                edges = np.linspace(0.0, 1.0, N_BINS + 1)
    return data, ncounts, edges


def main():
    data, ncounts, edges = load(IN_CSV)
    model = MODELS[0]
    n_cols = 3
    n_rows = -(-len(DATASETS) // n_cols)  # ceil
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3, 1.15 * n_rows + 0.35),
                              sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#d9d9d0")

    im = None
    for c, ds in enumerate(DATASETS):
        ax = axes[c]
        grid = data[(ds, model)]
        masked = np.ma.masked_invalid(grid)
        im = ax.imshow(masked, origin="lower", extent=[0, 1, 0, 1], aspect="auto",
                        cmap=cmap, vmin=VMIN, vmax=VMAX)
        for i in range(N_BINS):
            for j in range(N_BINS):
                val = grid[j, i]
                x0, x1 = edges[i], edges[i + 1]
                y0, y1 = edges[j], edges[j + 1]
                xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
                if np.isnan(val):
                    ax.text(xc, yc, "n/a", ha="center", va="center", fontsize=4.5, color="#777")
                else:
                    txt_color = "black" if 0.62 < val < 0.92 else "white"
                    ax.text(xc, yc, f"{val:.2f}", ha="center", va="center", fontsize=5,
                             color=txt_color, fontweight="medium")
        for e in edges:
            ax.axvline(e, color="white", linewidth=0.5)
            ax.axhline(e, color="white", linewidth=0.5)
        ax.set_title(DISPLAY_LABEL.get(ds, ds), fontsize=6.5)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=5)

    fig.suptitle(f"{model} test AUC vs. source out-entropy and target in-entropy, 4×4 bins",
                  fontsize=7)
    fig.text(0.5, 0.005, "src out-ent.", ha="center", fontsize=6)
    fig.text(0.005, 0.5, "target in-ent.", va="center", rotation="vertical", fontsize=6)
    fig.tight_layout(rect=(0.02, 0.02, 0.90, 0.92))
    cbar_ax = fig.add_axes((0.915, 0.15, 0.02, 0.7))
    cb = fig.colorbar(im, cax=cbar_ax, label="AUC")
    cb.ax.tick_params(labelsize=5)
    cb.set_label("AUC", fontsize=6)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
