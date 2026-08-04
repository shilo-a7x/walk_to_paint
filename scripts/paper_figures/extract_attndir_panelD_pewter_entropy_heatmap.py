"""Extract step for the Attention Directionality figure, Panel D -- binned entropy-
vs-AUC heatmap for PEWTER itself (the local-attention walk model), directly
comparable to Empirical Confirmation Panel C's GNN version (SiGAT only, as of
2026-08-04).

Source: outputs/lead4_entropy_heterogeneity/computed_data.pkl, variant "out_in"
(src_ent = H(out-signs of u), tgt_ent = H(in-signs of v)), model "walk_localattn4" --
same canonical shared-edge predictions used throughout Lead 4/4b/4c and Empirical
Confirmation Panel C (see that extract script's docstring for the full provenance).
No new predictions -- this reuses the already-computed per-edge (src_ent, tgt_ent, y,
p) records, restricted to one model instead of two.

Binning: identical 4x4 equal-width scheme to Empirical Confirmation Panel C (edges at
0, .25, .5, .75, 1.0), AUC via roc_auc_score per cell, cells below MIN_CELL_N or with
only one class present left NaN (masked grey).
"""
import csv
import os
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score

DATA_PATH = "outputs/lead4_entropy_heterogeneity/computed_data.pkl"
OUT_CSV = "aaai2027/figure_data/attndir_panelD_pewter_entropy_heatmap.csv"
VARIANT = "out_in"
DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
MODEL = "walk_localattn4"

N_BINS = 4
MIN_CELL_N = 30


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
        rec = data[ds][VARIANT][MODEL]
        src_ent = np.asarray(rec["src_ent"])
        tgt_ent = np.asarray(rec["tgt_ent"])
        y = np.asarray(rec["y"])
        p = np.asarray(rec["p"])
        auc_grid, n_grid, edges = binned_auc_grid(src_ent, tgt_ent, y, p, N_BINS)
        for i in range(N_BINS):
            for j in range(N_BINS):
                rows.append({
                    "dataset": ds, "model": MODEL,
                    "src_bin_lo": edges[i], "src_bin_hi": edges[i + 1],
                    "tgt_bin_lo": edges[j], "tgt_bin_hi": edges[j + 1],
                    "auc": auc_grid[i, j] if not np.isnan(auc_grid[i, j]) else "",
                    "n": n_grid[i, j],
                })
        print(f"{ds:16s} n_edges={len(y):,}  mean AUC={np.nanmean(auc_grid):.4f}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "src_bin_lo", "src_bin_hi",
                                           "tgt_bin_lo", "tgt_bin_hi", "auc", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
