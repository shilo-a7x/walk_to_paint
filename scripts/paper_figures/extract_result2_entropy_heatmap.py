"""Extract step for Result 2 -- dedicated 4-row kernel-smoothed AUC-vs-entropy
heatmap: Pewter (full attn), Pewter (LocalAttn4), SiGAT, GINEConv. Separate
figure from Empirical Confirmation Panel C (which stays GNN-only, 2 rows) --
this duplicates SiGAT/GINEConv on purpose (your call, 2026-07-27), so Result 2
stands alone as a complete walk-vs-GNN comparison without depending on Panel C.

Walk sources: aaai2027/figure_data/result2_walk_entropy_fresh.csv, built by
extract_result2_walk_entropy_fresh.py from the CURRENT production checkpoints
for both attention variants (E25/E26 full, E27 LocalAttn4) -- verified to
reproduce CLAUDE.md's exact reported test AUC on all 6 datasets for both
variants at extraction time. NOT computed_data.pkl's own walk entries (those
are stale E14-era checkpoints, see extract_walk_entropy_fresh.py's docstring).

GNN sources (SiGAT, GINEConv): outputs/lead4_entropy_heterogeneity/
computed_data.pkl, variant "out_in" -- same source as Panel C, unchanged.

Smoothing: same Gaussian-kernel-weighted AUC over a fine grid as Panel C
(see that script's docstring for the full method) -- not discrete bins.
"""
import csv
import os
import pickle
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

GNN_DATA_PATH = "outputs/lead4_entropy_heterogeneity/computed_data.pkl"
WALK_CSV = "aaai2027/figure_data/result2_walk_entropy_fresh.csv"
OUT_CSV = "aaai2027/figure_data/result2_entropy_heatmap.csv"
VARIANT = "out_in"
DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
MODELS = ["Pewter (full attn)", "Pewter (LocalAttn4)", "SiGAT", "GINEConv"]

GRID_N = 25
BANDWIDTH = 0.15
MIN_EFFECTIVE_N = 15


def load_walk_records(path):
    recs = defaultdict(lambda: defaultdict(lambda: {"src_ent": [], "tgt_ent": [], "y": [], "p": []}))
    with open(path) as f:
        for row in csv.DictReader(f):
            r = recs[row["dataset"]][row["model"]]
            r["src_ent"].append(float(row["src_ent"]))
            r["tgt_ent"].append(float(row["tgt_ent"]))
            r["y"].append(int(row["y"]))
            r["p"].append(float(row["p"]))
    return {ds: {m: {k: np.array(v) for k, v in rec.items()} for m, rec in models.items()}
            for ds, models in recs.items()}


def gaussian_weights(src_ent, tgt_ent, s0, t0, h):
    d2 = ((src_ent - s0) / h) ** 2 + ((tgt_ent - t0) / h) ** 2
    return np.exp(-0.5 * d2)


def weighted_auc_grid(src_ent, tgt_ent, y, p, grid, h):
    n = len(grid)
    auc_grid = np.full((n, n), np.nan)
    neff_grid = np.zeros((n, n))
    y_bin = (np.asarray(y) > 0).astype(int)
    has_both_classes = len(set(y_bin)) > 1
    for i, s0 in enumerate(grid):
        for j, t0 in enumerate(grid):
            w = gaussian_weights(src_ent, tgt_ent, s0, t0, h)
            sw = w.sum()
            sw2 = (w ** 2).sum()
            n_eff = (sw * sw / sw2) if sw2 > 0 else 0.0
            neff_grid[i, j] = n_eff
            if n_eff < MIN_EFFECTIVE_N or not has_both_classes:
                continue
            try:
                auc_grid[i, j] = roc_auc_score(y_bin, p, sample_weight=w)
            except ValueError:
                pass
    return auc_grid, neff_grid


def main():
    with open(GNN_DATA_PATH, "rb") as f:
        gnn_data = pickle.load(f)
    walk_data = load_walk_records(WALK_CSV)

    grid = np.linspace(0.0, 1.0, GRID_N)
    rows = []
    for ds in DATASETS:
        for model in MODELS:
            rec = walk_data[ds][model] if model.startswith("Pewter") else gnn_data[ds][VARIANT][model]
            auc_grid, neff_grid = weighted_auc_grid(
                rec["src_ent"], rec["tgt_ent"], rec["y"], rec["p"], grid, BANDWIDTH)
            print(f"{ds} / {model}: n={len(rec['y'])}, "
                  f"valid grid cells={np.isfinite(auc_grid).sum()}/{GRID_N*GRID_N}")
            for i, s0 in enumerate(grid):
                for j, t0 in enumerate(grid):
                    rows.append({
                        "dataset": ds, "model": model,
                        "src_ent": s0, "tgt_ent": t0,
                        "auc": auc_grid[i, j], "n_eff": neff_grid[i, j],
                    })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "src_ent", "tgt_ent", "auc", "n_eff"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
