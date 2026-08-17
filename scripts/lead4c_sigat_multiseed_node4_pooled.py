"""
SiGAT 4-term node-entropy regression, POOLED across all 6 datasets (shared slope,
one intercept per dataset via full dummy encoding), refit independently on EACH of
the 10 seeds (42-51), then aggregated as mean +- std across seeds per term.

Companion to lead4c_sigat_multiseed_node4.py (which fits a SEPARATE slope per
dataset). This instead asks: if you force one shared src_out/src_in/tgt_out/tgt_in
slope across all 6 datasets (with per-dataset fixed effects on the intercept only),
how stable is that pooled slope across 10 independently-trained SiGAT models /
independently-drawn test splits?

Same pooling convention used throughout this project's Lead4c work: ALL dataset
dummy columns, add_const=False, so each dummy's own coefficient IS that dataset's
intercept directly (no reference-dataset offset); node ids are offset per dataset
before stacking so a raw id collision across datasets can't merge two different
nodes into one cluster.

Usage:
    .venv/bin/python scripts/lead4c_sigat_multiseed_node4_pooled.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scripts.paper_figures.extract_multiseed_entropy_heatmaps import sigat_raw_seed, SEEDS
from scripts.node_mi_structural_embedding import DATASET_CONFIGS, load_edges_canonical
from scripts.lead4_entropy_heterogeneity import build_sign_dicts, entropy_lookup, _ds_key
from scripts.lead4c_entropy_logit_regression import _fit_one, _zscore_and_mask, _bh_fdr, NODE4_TERMS

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
OUT_DIR = os.path.join(ROOT, "outputs", "lead4c_sigat_multiseed_node4_pooled")


def node4_columns_for_edges(u, v, ent):
    return {
        "src_out": np.array([ent["out"].get(int(n), np.nan) for n in u]),
        "src_in": np.array([ent["in"].get(int(n), np.nan) for n in u]),
        "tgt_out": np.array([ent["out"].get(int(n), np.nan) for n in v]),
        "tgt_in": np.array([ent["in"].get(int(n), np.nan) for n in v]),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # entropy bases + per-seed per-dataset edge tables (u, v, correct, features),
    # built once, reused for both the raw and z-scored pooled fit per seed.
    ent_by_ds = {}
    for ds in DATASETS:
        print(f"[{ds}] building entropy base...")
        edges = load_edges_canonical(DATASET_CONFIGS[_ds_key(ds)]["ds_name"])
        sign_dicts = build_sign_dicts(edges)
        ent_by_ds[ds] = entropy_lookup(sign_dicts)

    per_seed_rows = []
    for seed in SEEDS:
        print(f"seed{seed}: fitting pooled model across all 6 datasets...")
        feats = {k: [] for k in NODE4_TERMS}
        ys, us, vs, dummies = [], [], [], []
        offset_u = 0
        ds_present = []
        for ds in DATASETS:
            rec = sigat_raw_seed(ds, seed)
            if rec is None:
                print(f"  {ds}: missing, skipped")
                continue
            u = np.asarray(rec["u"]); v = np.asarray(rec["v"])
            y = np.asarray(rec["y"]); p = np.asarray(rec["p"])
            correct = ((p >= 0.5).astype(int) == y).astype(int)
            cols = node4_columns_for_edges(u, v, ent_by_ds[ds])
            n = len(correct)
            for k in NODE4_TERMS:
                feats[k].append(cols[k])
            ys.append(correct)
            us.append(u + offset_u); vs.append(v + offset_u)
            offset_u += int(max(u.max(), v.max())) + 1
            dummies.append(np.full(n, ds))
            ds_present.append(ds)

        y_all = np.concatenate(ys); u_all = np.concatenate(us); v_all = np.concatenate(vs)
        dummy_all = np.concatenate(dummies)

        # raw scale
        X_raw = {k: np.concatenate(feats[k]) for k in NODE4_TERMS}
        for d in ds_present:
            X_raw[f"ds_{d}"] = (dummy_all == d).astype(np.float64)
        rows = _fit_one(y_all, X_raw, u_all, v_all, f"POOLED/seed{seed}/node4/shared-slope", add_const=False)
        if rows is not None:
            for row in rows:
                row.update(seed=seed, scale="raw")
                per_seed_rows.append(row)

        # z-scored (standardize the 4 entropy columns on the pooled rows, dummies untouched)
        X_raw2 = {k: np.concatenate(feats[k]) for k in NODE4_TERMS}
        Xz, y_m, u_m, v_m, dummy_m = _zscore_and_mask(X_raw2, NODE4_TERMS, y_all, u_all, v_all, dummy_all)
        for d in ds_present:
            Xz[f"ds_{d}"] = (dummy_m == d).astype(np.float64)
        rows_z = _fit_one(y_m, Xz, u_m, v_m, f"POOLED/seed{seed}/node4_zscored/shared-slope", add_const=False)
        if rows_z is not None:
            for row in rows_z:
                row.update(seed=seed, scale="zscored")
                per_seed_rows.append(row)

        print(f"  n_total={len(y_all)}, base_rate={y_all.mean():.3f}, datasets={len(ds_present)}")

    df = pd.DataFrame.from_records(per_seed_rows)
    # BH-FDR within each (seed, scale) fit's 4 shared-slope terms (dataset
    # intercepts excluded from the correction, matching run_node4_fits' convention).
    df["p_fdr"] = np.nan
    m = df["term"].isin(NODE4_TERMS)
    for (seed, scale), idx in df[m].groupby(["seed", "scale"]).groups.items():
        df.loc[idx, "p_fdr"] = _bh_fdr(df.loc[idx, "p"].values)

    per_seed_path = os.path.join(OUT_DIR, "per_seed_pooled_fits.csv")
    df.to_csv(per_seed_path, index=False)
    df.to_pickle(os.path.join(OUT_DIR, "per_seed_pooled_fits.pkl"))
    print(f"\nwrote {per_seed_path} ({len(df)} rows)")

    # ── aggregate: mean +- std across the 10 seeds, per (scale, term) ──
    agg_rows = []
    for (scale, term), g in df.groupby(["scale", "term"]):
        n_seeds = len(g)
        is_slope = term in NODE4_TERMS
        n_sig = int((g["p_fdr"] < 0.05).sum()) if is_slope else None
        agg_rows.append(dict(
            scale=scale, term=term, is_dataset_intercept=not is_slope and term.startswith("ds_"),
            mean_beta=g["beta"].mean(), std_beta=g["beta"].std(ddof=1),
            min_beta=g["beta"].min(), max_beta=g["beta"].max(),
            n_seeds=n_seeds, n_seeds_significant=n_sig,
            frac_seeds_significant=(n_sig / n_seeds) if (is_slope and n_seeds) else np.nan,
        ))
    agg = pd.DataFrame.from_records(agg_rows)
    agg_path = os.path.join(OUT_DIR, "aggregated_summary.csv")
    agg.to_csv(agg_path, index=False)
    agg.to_pickle(os.path.join(OUT_DIR, "aggregated_summary.pkl"))
    print(f"wrote {agg_path} ({len(agg)} rows)")


if __name__ == "__main__":
    main()
