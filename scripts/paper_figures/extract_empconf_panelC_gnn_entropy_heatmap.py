"""Extract step for Empirical Confirmation Panel C -- GNN accuracy vs.
entropy, DISCRETE 4x4 binned heatmap. GNN-only (2026-07-27 call: Panel C
stays the original 2-row GNN comparison; the walk-vs-GNN comparison lives in
its own separate figure for Result 2 -- see extract_result2_entropy_heatmap.py).

Source: outputs/lead4_entropy_heterogeneity/computed_data.pkl, variant
"out_in" (src_ent = H(out-signs of u), tgt_ent = H(in-signs of v)) -- the two
atomic entropy axes Lead4c's regression found to actually matter (src_out,
tgt_in; the other two, tgt_out/src_in, were near-null). No new predictions --
this reuses the already-computed per-edge (src_ent, tgt_ent, y, p) records.

Models: SiGAT (raw, not SGA-augmented) and GINEConv only -- no walk column.

Binning (reverted 2026-07-28 from the Gaussian-kernel-smoothed 25x25 grid):
plain 4x4 equal-width bins over [0,1]x[0,1] (edges at 0, .25, .5, .75, 1.0),
AUC computed directly within each cell via roc_auc_score (no kernel
weighting). Cells below MIN_CELL_N real edges, or with only one class
present, are left NaN (masked grey in the plot) rather than extrapolated --
with only 16 cells per panel there's enough real mass per cell on all 6
datasets that grey cells are rare, unlike the finer discrete grids tried
earlier in this project's history (hence the original move to kernel
smoothing) -- 4x4 is coarse enough to sidestep that problem directly.
"""
import csv
import os
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score

DATA_PATH = "outputs/lead4_entropy_heterogeneity/computed_data.pkl"
OUT_CSV = "aaai2027/figure_data/empconf_panelC_gnn_entropy_heatmap.csv"
VARIANT = "out_in"
DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
MODELS = ["SiGAT", "GINEConv"]

N_BINS = 4            # N_BINS x N_BINS discrete bins over [0,1]^2
MIN_CELL_N = 30        # mask cells below this many real edges


def binned_auc_grid(src_ent, tgt_ent, y, p, n_bins):
    """Returns (auc_grid, n_grid), each shape (n_bins, n_bins), row=src bin, col=tgt bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    src_bin = np.clip(np.digitize(src_ent, edges[1:-1]), 0, n_bins - 1)
    tgt_bin = np.clip(np.digitize(tgt_ent, edges[1:-1]), 0, n_bins - 1)
    y_bin = (np.asarray(y) > 0).astype(int)

    auc_grid = np.full((n_bins, n_bins), np.nan)
    n_grid = np.zeros((n_bins, n_bins), dtype=int)
    for i in range(n_bins):
        for j in range(n_bins):
            mask = (src_bin == i) & (tgt_bin == j)
            n = int(mask.sum())
            n_grid[i, j] = n
            if n < MIN_CELL_N or len(set(y_bin[mask])) < 2:
                continue
            auc_grid[i, j] = roc_auc_score(y_bin[mask], p[mask])
    return auc_grid, n_grid, edges


def main():
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    rows = []
    for ds in DATASETS:
        for model in MODELS:
            rec = data[ds][VARIANT][model]
            src_ent = np.asarray(rec["src_ent"])
            tgt_ent = np.asarray(rec["tgt_ent"])
            p = np.asarray(rec["p"])
            auc_grid, n_grid, edges = binned_auc_grid(src_ent, tgt_ent, rec["y"], p, N_BINS)
            print(f"{ds} / {model}: n={len(rec['y'])}, "
                  f"valid cells={np.isfinite(auc_grid).sum()}/{N_BINS*N_BINS}, "
                  f"cell n range=[{n_grid.min()},{n_grid.max()}]")
            for i in range(N_BINS):
                for j in range(N_BINS):
                    rows.append({
                        "dataset": ds, "model": model,
                        "src_bin_lo": edges[i], "src_bin_hi": edges[i + 1],
                        "tgt_bin_lo": edges[j], "tgt_bin_hi": edges[j + 1],
                        "auc": auc_grid[i, j], "n": n_grid[i, j],
                    })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "src_bin_lo", "src_bin_hi",
                                           "tgt_bin_lo", "tgt_bin_hi", "auc", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
