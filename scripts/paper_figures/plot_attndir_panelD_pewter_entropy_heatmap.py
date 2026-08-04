"""Plot step for the Attention Directionality figure, Panel D -- binned entropy-vs-AUC
heatmap for PEWTER itself. Pure rendering: reads
aaai2027/figure_data/attndir_panelD_pewter_entropy_heatmap.csv (built by
extract_attndir_panelD_pewter_entropy_heatmap.py). Edit THIS file freely for
colormap/scale/style changes -- no recomputation needed.

Layout: 1 row x 6 columns, matching Empirical Confirmation Panel C's now-single-row
(SiGAT-only) layout, for direct visual comparison between the two figures. We say
"PEWTER" rather than "LocalAttn4" -- this panel doesn't compare against full
attention, so the full/local distinction isn't the point here.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_CSV = "aaai2027/figure_data/attndir_panelD_pewter_entropy_heatmap.csv"
OUT_PNG = "aaai2027/figures/attndir_panelD_pewter_entropy_heatmap.png"

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
VMIN, VMAX = 0.5, 1.0
N_BINS = 4
DISPLAY_LABEL = {"slashdot090221": "slashdot"}


def load(path):
    data = {ds: np.full((N_BINS, N_BINS), np.nan) for ds in DATASETS}
    edges = None
    with open(path) as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            if ds not in data:
                continue
            lo_s, lo_t = float(row["src_bin_lo"]), float(row["tgt_bin_lo"])
            i = round(lo_s * N_BINS)
            j = round(lo_t * N_BINS)
            auc = row["auc"]
            data[ds][j, i] = float(auc) if auc != "" and auc.lower() != "nan" else np.nan
            if edges is None:
                edges = np.linspace(0.0, 1.0, N_BINS + 1)
    return data, edges


def main():
    data, edges = load(IN_CSV)
    n_cols = len(DATASETS)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.1 * n_cols, 3.3), sharex=True, sharey=True,
                              squeeze=False)
    axes = axes[0]

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#d9d9d0")

    im = None
    for c, ds in enumerate(DATASETS):
        ax = axes[c]
        grid = data[ds]
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
                    ax.text(xc, yc, "n/a", ha="center", va="center", fontsize=7, color="#777")
                else:
                    txt_color = "black" if 0.62 < val < 0.92 else "white"
                    ax.text(xc, yc, f"{val:.2f}", ha="center", va="center", fontsize=8.5,
                             color=txt_color, fontweight="medium")
        for e in edges:
            ax.axvline(e, color="white", linewidth=0.6)
            ax.axhline(e, color="white", linewidth=0.6)
        ax.set_title(DISPLAY_LABEL.get(ds, ds), fontsize=10)
        if c == 0:
            ax.set_ylabel("PEWTER", fontsize=11)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(labelsize=7)
        ax.set_xlabel("src out-ent.", fontsize=8)

    fig.suptitle("PEWTER test AUC vs. source out-entropy (rater consistency) and target in-entropy "
                 "(reputation contestedness), 4×4 bins",
                 fontsize=11)
    fig.text(0.005, 0.5, "target in-ent.", va="center", rotation="vertical", fontsize=9)
    fig.tight_layout(rect=(0.02, 0, 0.93, 0.90))
    cbar_ax = fig.add_axes((0.945, 0.15, 0.013, 0.7))
    fig.colorbar(im, cax=cbar_ax, label="AUC")

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
