"""Plot step for the two 10-split entropy heatmaps agreed with the user (2026-08-10):
1. SiGAT entropy-vs-AUC heatmap, mean AUC over 10 splits (Option 2 method, see
   extract_multiseed_entropy_heatmaps.py / CLAUDE.md) -- same visual style as the
   original single-split Panel C (plot_empconf_panelC_gnn_entropy_heatmap.py):
   RdYlGn colormap, vmin/vmax=0.5/1.0, light-grey "n/a" for masked cells, white
   gridlines -- not a different color scheme just because the data source changed.
2. PEWTER(local)-vs-SiGAT delta heatmap -- diverging RdBu colormap (NOT reversed):
   blue = PEWTER higher, red = SiGAT higher (per the user's explicit color choice,
   2026-08-10 -- the first version used RdBu_r, which put PEWTER-higher on red;
   caught and flipped).

Same 4x4 fixed-entropy-value bin layout as the existing single-split Panel C --
one subplot per dataset, 2x3 grid. Pure rendering: reads the two CSVs written by
extract_multiseed_entropy_heatmaps.py.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_SIGAT_CSV = os.path.join(ROOT, "aaai2027", "figure_data", "empconf_panelC_sigat_10split.csv")
IN_DELTA_CSV = os.path.join(ROOT, "aaai2027", "figure_data", "pewter_sigat_delta_heatmap.csv")
OUT_SIGAT_PNG = os.path.join(ROOT, "aaai2027", "figures", "empconf_panelC_sigat_10split.png")
OUT_DELTA_PNG = os.path.join(ROOT, "aaai2027", "figures", "pewter_sigat_delta_heatmap.png")

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
DISPLAY_LABEL = {"slashdot090221": "slashdot"}
N_BINS = 4


def _grid_from_rows(rows, ds, value_key):
    grid = np.full((N_BINS, N_BINS), np.nan)
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    for r in rows:
        if r["dataset"] != ds:
            continue
        i = int(round(float(r["src_bin_lo"]) / 0.25))
        j = int(round(float(r["tgt_bin_lo"]) / 0.25))
        v = r[value_key]
        grid[i, j] = float(v) if v not in ("", "nan") else np.nan
    return grid, edges


def _draw_grid(ax, grid, edges, cmap, vmin, vmax, fmt="{:.2f}", text_thresh=None):
    """grid[i, j]: i = src bin, j = tgt bin (matches _grid_from_rows). imshow wants
    row=y (tgt), col=x (src), so transpose -- same convention as the original
    plot_empconf_panelC_gnn_entropy_heatmap.py (grid[j, i] indexing there)."""
    masked = np.ma.masked_invalid(grid.T)
    im = ax.imshow(masked, origin="lower", extent=[0, 1, 0, 1], cmap=cmap,
                    vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(N_BINS):
        for j in range(N_BINS):
            v = grid[i, j]
            cx, cy = (edges[i] + edges[i + 1]) / 2, (edges[j] + edges[j + 1]) / 2
            if np.isnan(v):
                ax.text(cx, cy, "n/a", ha="center", va="center", fontsize=6.5, color="#777")
            else:
                if text_thresh is not None:
                    lo, hi = text_thresh
                    color = "black" if lo < v < hi else "white"
                else:
                    color = "black"
                ax.text(cx, cy, fmt.format(v), ha="center", va="center", fontsize=7, color=color)
    for e in edges:
        ax.axvline(e, color="white", linewidth=0.5)
        ax.axhline(e, color="white", linewidth=0.5)
    ax.set_xticks(edges); ax.set_yticks(edges)
    ax.set_xticklabels([f"{e:.2f}" for e in edges], fontsize=6)
    ax.set_yticklabels([f"{e:.2f}" for e in edges], fontsize=6)
    return im


def plot_sigat_mean():
    rows = list(csv.DictReader(open(IN_SIGAT_CSV)))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#d9d9d0")

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
    im = None
    for ax, ds in zip(axes.flat, DATASET_ORDER):
        grid, edges = _grid_from_rows(rows, ds, "mean_auc")
        im = _draw_grid(ax, grid, edges, cmap=cmap, vmin=0.5, vmax=1.0, text_thresh=(0.62, 0.92))
        ax.set_title(DISPLAY_LABEL.get(ds, ds), fontsize=10)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$H_{out}(u)$ (src)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$H_{in}(v)$ (tgt)", fontsize=8)
    fig.suptitle("SiGAT: mean AUC by source/target entropy", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.7, label="mean AUC")
    fig.savefig(OUT_SIGAT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_SIGAT_PNG}")


def plot_delta():
    rows = list(csv.DictReader(open(IN_DELTA_CSV)))
    grids = {}
    max_abs = 0.0
    for ds in DATASET_ORDER:
        grid, edges = _grid_from_rows(rows, ds, "delta")
        grids[ds] = (grid, edges)
        if np.any(np.isfinite(grid)):
            max_abs = max(max_abs, np.nanmax(np.abs(grid)))
    vmax = max(max_abs, 0.01)

    cmap = plt.get_cmap("RdBu").copy()  # NOT reversed: low=red, high=blue
    cmap.set_bad(color="#d9d9d0")

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
    im = None
    for ax, ds in zip(axes.flat, DATASET_ORDER):
        grid, edges = grids[ds]
        im = _draw_grid(ax, grid, edges, cmap=cmap, vmin=-vmax, vmax=vmax, fmt="{:+.2f}")
        ax.set_title(DISPLAY_LABEL.get(ds, ds), fontsize=10)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$H_{out}(u)$ (src)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$H_{in}(v)$ (tgt)", fontsize=8)
    fig.suptitle("PEWTER (local) $-$ SiGAT: mean AUC delta by source/target entropy", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.7, label="AUC delta (blue = PEWTER higher, red = SiGAT higher)")
    fig.savefig(OUT_DELTA_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DELTA_PNG}")


def main():
    os.makedirs(os.path.dirname(OUT_SIGAT_PNG), exist_ok=True)
    plot_sigat_mean()
    plot_delta()


if __name__ == "__main__":
    main()
