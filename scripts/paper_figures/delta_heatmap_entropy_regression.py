"""Figure 3 delta heatmap ("gain concentrates where the bound bites") -- statistical test
for STATISTICAL_TESTS_AUDIT.md item #15 / the professor's Section 6.2 ask ("a statistical
claim on the difference in general and the contribution of the entropy to the
difference"). Currently the claim is read directly off the heatmap's visual pattern with
no formal statistic.

Design, per the audit doc's recommendation: a 2-variable regression of the Pewter-minus-
SiGAT AUC delta on the two entropy axes (source-entropy bin midpoint, target-entropy bin
midpoint), analogous in spirit to Panel C's two-way ANOVA (item #4) but continuous, since
what's wanted here is a coefficient (sign + magnitude + significance) per entropy axis,
not just an F-test -- "contribution of entropy to the difference" reads as a slope
question. One observation per (seed, cell): delta = pewter_auc(seed,cell) -
sigat_auc(seed,cell), same MIN_CELL_N_DELTA=50 threshold and "out_in" entropy axes
(src=H_out(u), tgt=H_in(v)) as the existing pewter_sigat_delta_heatmap.png. Cluster-robust
SE (cluster=seed), since the 16 cells within one seed share the same two trained models
and aren't independent draws.

Reuses SiGAT's raw predictions already cached for Panel D's rebuild
(outputs/cache/sigat_raw_predictions/, no refit) and PEWTER(local)'s walk_raw_seed()
(cheap CPU aggregation over already-saved test_predictions.pkl, same cost profile as the
K-ablation) -- both from extract_multiseed_entropy_heatmaps.py, imported directly rather
than reimplemented.
"""
import csv
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baselines"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

sys.path.insert(0, os.path.join(ROOT, "scripts", "paper_figures"))
from extract_multiseed_entropy_heatmaps import (  # noqa: E402
    binned_auc_grid, walk_raw_seed, MIN_CELL_N_DELTA, N_BINS, SEEDS,
)
from scripts.lead4_entropy_heterogeneity import build_sign_dicts, collect_model_records  # noqa: E402
from scripts.node_mi_structural_embedding import DATASET_CONFIGS, load_edges_canonical  # noqa: E402

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
RAW_CACHE_DIR = os.path.join(ROOT, "outputs", "cache", "sigat_raw_predictions")
OUT_CSV = "aaai2027/figure_data/delta_heatmap_entropy_regression.csv"


def _ds_key(ds_name):
    return "slashdot" if ds_name == "slashdot090221" else ds_name


def sigat_raw_seed_from_cache(ds, seed):
    """Reuses the raw SiGAT predictions already cached for Panel D's rebuild -- no refit."""
    path = os.path.join(RAW_CACHE_DIR, f"{ds}__seed{seed}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def cluster_robust_ols(y, X, cluster):
    """OLS with one-way cluster-robust SE. X: (n, k) design matrix incl. intercept col."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta

    clusters = np.unique(cluster)
    meat = np.zeros((k, k))
    for c in clusters:
        mask = cluster == c
        Xg = X[mask]
        ug = resid[mask]
        score = Xg.T @ ug
        meat += np.outer(score, score)
    n_clusters = len(clusters)
    dof_correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    cov = dof_correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n_clusters - 1))
    return beta, se, t, p


def main():
    rows_out = []
    edges_bins = np.linspace(0.0, 1.0, N_BINS + 1)
    mids = (edges_bins[:-1] + edges_bins[1:]) / 2

    for ds in DATASETS:
        edges_list = load_edges_canonical(DATASET_CONFIGS[_ds_key(ds)]["ds_name"])
        sign_dicts = build_sign_dicts(edges_list)

        long_rows = []
        for seed in SEEDS:
            sigat_rec = sigat_raw_seed_from_cache(ds, seed)
            walk_rec = walk_raw_seed(ds, seed, "local")
            if sigat_rec is None or walk_rec is None or not len(sigat_rec["u"]) or not len(walk_rec["u"]):
                continue
            sigat_entry = collect_model_records(
                list(zip(sigat_rec["u"], sigat_rec["v"], sigat_rec["y"], sigat_rec["p"])),
                sign_dicts, "out_in", min_total=1)
            walk_entry = collect_model_records(
                list(zip(walk_rec["u"], walk_rec["v"], walk_rec["y"], walk_rec["p"])),
                sign_dicts, "out_in", min_total=1)
            if sigat_entry is None or walk_entry is None:
                continue
            sigat_grid, _, _ = binned_auc_grid(
                sigat_entry["src_ent"], sigat_entry["tgt_ent"], sigat_entry["y"], sigat_entry["p"],
                N_BINS, MIN_CELL_N_DELTA)
            walk_grid, _, _ = binned_auc_grid(
                walk_entry["src_ent"], walk_entry["tgt_ent"], walk_entry["y"], walk_entry["p"],
                N_BINS, MIN_CELL_N_DELTA)
            for i in range(N_BINS):
                for j in range(N_BINS):
                    if np.isfinite(sigat_grid[i, j]) and np.isfinite(walk_grid[i, j]):
                        long_rows.append({
                            "seed": seed, "src_bin": i, "tgt_bin": j,
                            "src_mid": mids[i], "tgt_mid": mids[j],
                            "delta": walk_grid[i, j] - sigat_grid[i, j],
                        })

        df = pd.DataFrame(long_rows)
        if df.empty:
            print(f"{ds}: no usable (seed, cell) pairs, skipping")
            continue
        n_seeds_used = df["seed"].nunique()
        print(f"\n{ds}: {len(df)} (seed, cell) observations across {n_seeds_used} seeds")

        y = df["delta"].to_numpy()
        X = np.column_stack([np.ones(len(df)), df["src_mid"].to_numpy(), df["tgt_mid"].to_numpy()])
        cluster = df["seed"].to_numpy()
        beta, se, t, p = cluster_robust_ols(y, X, cluster)

        print(f"  intercept:  {beta[0]:+.4f} (se={se[0]:.4f}, p={p[0]:.4g})")
        print(f"  src_mid:    {beta[1]:+.4f} (se={se[1]:.4f}, p={p[1]:.4g})")
        print(f"  tgt_mid:    {beta[2]:+.4f} (se={se[2]:.4f}, p={p[2]:.4g})")
        rows_out.append({
            "dataset": ds, "n_obs": len(df), "n_seeds": n_seeds_used,
            "intercept": beta[0], "intercept_se": se[0], "intercept_p": p[0],
            "src_coef": beta[1], "src_se": se[1], "src_p": p[1],
            "tgt_coef": beta[2], "tgt_se": se[2], "tgt_p": p[2],
        })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "n_obs", "n_seeds", "intercept", "intercept_se",
                                           "intercept_p", "src_coef", "src_se", "src_p",
                                           "tgt_coef", "tgt_se", "tgt_p"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
