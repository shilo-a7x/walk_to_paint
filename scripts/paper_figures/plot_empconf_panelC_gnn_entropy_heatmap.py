"""Plot step for Empirical Confirmation Panel C -- GNN accuracy vs. entropy,
DISCRETE 4x4 binned heatmap. GNN-only (see extract script docstring for why
-- the walk-vs-GNN comparison lives in its own figure for Result 2 instead).

Pure rendering: reads aaai2027/figure_data/empconf_panelC_gnn_entropy_heatmap.csv
(built by extract_empconf_panelC_gnn_entropy_heatmap.py). Edit THIS file freely
for colormap/scale/style/size changes -- no recomputation needed.

Layout: 1 row (SiGAT only, per 2026-08-04 call -- GINEConv dropped from this
figure, kept in the appendix/baseline table instead; the extract CSV still
has both models' cells for that reuse) x 6 columns (datasets), enlarged
(2026-07-28, per your call) vs. the previous kernel-smoothed version -- each
cell now also gets an AUC value annotated directly on it (readable at 4x4
resolution, wasn't at 25x25), and masked (insufficient-n) cells are shown as
hatched grey rather than left blank.
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
DISPLAY_LABEL = {
    "bitcoin-alpha": "Bitcoin-alpha", "bitcoin-otc": "Bitcoin-otc", "epinions": "Epinions",
    "slashdot090221": "Slashdot", "wiki-elec": "Wiki-elec", "wiki-rfa": "Wiki-RfA",
    "SiGAT": "SiGAT",
}


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
    n_rows, n_cols = len(MODELS), len(DATASETS)
    # bigger panel per user request: was figsize=(1.9*n_cols, 2.0*n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.3 * n_rows), sharex=True, sharey=True,
                              squeeze=False)

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#d9d9d0")

    im = None
    for r, model in enumerate(MODELS):
        for c, ds in enumerate(DATASETS):
            ax = axes[r, c]
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
                        ax.text(xc, yc, "n/a", ha="center", va="center", fontsize=11, color="#777")
                    else:
                        txt_color = "black" if 0.62 < val < 0.92 else "white"
                        ax.text(xc, yc, f"{val:.2f}", ha="center", va="center", fontsize=13,
                                 color=txt_color, fontweight="medium")
            for e in edges:
                ax.axvline(e, color="white", linewidth=0.6)
                ax.axhline(e, color="white", linewidth=0.6)
            if r == 0:
                ax.set_title(DISPLAY_LABEL.get(ds, ds), fontsize=15)
            if c == 0:
                ax.set_ylabel(DISPLAY_LABEL.get(model, model), fontsize=15)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.tick_params(labelsize=11)
            if r == n_rows - 1:
                ax.set_xlabel("src out-ent.", fontsize=13)

    fig.suptitle("SiGAT: AUC vs. source/target entropy", fontsize=17)
    fig.text(0.005, 0.5, "target in-ent.", va="center", rotation="vertical", fontsize=14)
    fig.tight_layout(rect=(0.02, 0, 0.93, 0.95))
    cbar_ax = fig.add_axes((0.945, 0.15, 0.013, 0.7))
    fig.colorbar(im, cax=cbar_ax, label="AUC")

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
