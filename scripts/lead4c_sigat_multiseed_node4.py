"""
SiGAT 4-term node-entropy regression, refit independently on EACH of the 10 seeds
(42-51), then aggregated as mean +- std across seeds per (dataset, term).

Rationale (2026-07-28 session): the single-seed (seed 42) per-dataset fits found
patterns (src_out-vs-tgt_in dominance flipping with graph size; src_in null on wiki
only) that could just be seed-42 noise on the smaller datasets. This checks that
directly by refitting on all 10 already-available SiGAT seeds, following the same
"fit each split independently, then mean+-std" convention already adopted for the
paper's entropy-vs-AUC heatmaps (CLAUDE.md "Entropy-heatmap multi-split methodology
-- decided 2026-08-10: Option 2", not pooling raw predictions across seeds).

Entropy features (src_out, src_in, tgt_out, tgt_in) are computed ONCE per dataset --
they only depend on the fixed graph (all edges), not on the split -- so only the
per-seed (y, p) SiGAT predictions differ across the 10 fits. Per-seed extraction
reuses `sigat_raw_seed` from scripts/paper_figures/extract_multiseed_entropy_heatmaps.py
unchanged (fits a fresh LogisticRegression on that seed's own train-split embeddings,
mirrors baselines/postprocess_canonical.py::sigat_raw parameterized over seed).

Uses each seed's own FULL SiGAT test split (not restricted to the walk-model-shared
edge set the earlier seed-42-only package used) -- this package has no walk-model
dependency, so there's no reason to throw away edges outside walk coverage.

Usage:
    .venv/bin/python scripts/lead4c_sigat_multiseed_node4.py
"""
import os
import pickle
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
OUT_DIR = os.path.join(ROOT, "outputs", "lead4c_sigat_multiseed_node4")


def node4_columns_for_edges(u, v, ent):
    return {
        "src_out": np.array([ent["out"].get(int(n), np.nan) for n in u]),
        "src_in": np.array([ent["in"].get(int(n), np.nan) for n in u]),
        "tgt_out": np.array([ent["out"].get(int(n), np.nan) for n in v]),
        "tgt_in": np.array([ent["in"].get(int(n), np.nan) for n in v]),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    per_seed_rows = []

    for ds in DATASETS:
        print(f"[{ds}] building entropy base (once, shared by all 10 seeds)...")
        edges = load_edges_canonical(DATASET_CONFIGS[_ds_key(ds)]["ds_name"])
        sign_dicts = build_sign_dicts(edges)
        ent = entropy_lookup(sign_dicts)

        for seed in SEEDS:
            rec = sigat_raw_seed(ds, seed)
            if rec is None:
                print(f"  seed{seed}: missing, skipped")
                continue
            u = np.asarray(rec["u"]); v = np.asarray(rec["v"])
            y = np.asarray(rec["y"]); p = np.asarray(rec["p"])
            correct = ((p >= 0.5).astype(int) == y).astype(int)
            cols = node4_columns_for_edges(u, v, ent)

            # raw scale
            rows = _fit_one(correct, dict(cols), u, v, f"{ds}/seed{seed}/node4")
            if rows is not None:
                for row in rows:
                    row.update(dataset=ds, seed=seed, scale="raw")
                    per_seed_rows.append(row)

            # z-scored
            Xz, y_m, u_m, v_m, _ = _zscore_and_mask(dict(cols), NODE4_TERMS, correct, u, v)
            rows_z = _fit_one(y_m, Xz, u_m, v_m, f"{ds}/seed{seed}/node4_zscored")
            if rows_z is not None:
                for row in rows_z:
                    row.update(dataset=ds, seed=seed, scale="zscored")
                    per_seed_rows.append(row)

            print(f"  seed{seed}: n={len(u)}, base_rate={correct.mean():.3f}")

    df = pd.DataFrame.from_records(per_seed_rows)
    # BH-FDR correction within each (dataset, seed, scale) fit's 4 slope terms
    # (const excluded) -- _fit_one only returns the raw p-value; this mirrors what
    # run_node4_fits does for a single pooled/dataset fit, applied per seed here.
    df["p_fdr"] = np.nan
    m = df["term"].isin(NODE4_TERMS)
    for (ds, seed, scale), idx in df[m].groupby(["dataset", "seed", "scale"]).groups.items():
        df.loc[idx, "p_fdr"] = _bh_fdr(df.loc[idx, "p"].values)
    per_seed_path = os.path.join(OUT_DIR, "per_seed_fits.csv")
    df.to_csv(per_seed_path, index=False)
    df.to_pickle(os.path.join(OUT_DIR, "per_seed_fits.pkl"))
    print(f"\nwrote {per_seed_path} ({len(df)} rows, {df['seed'].nunique()} seeds x "
          f"{df['dataset'].nunique()} datasets x {df['term'].nunique()} terms x 2 scales)")

    # ── aggregate: mean +- std across the 10 seeds, per (dataset, scale, term) ──
    agg_rows = []
    for (ds, scale, term), g in df.groupby(["dataset", "scale", "term"]):
        n_seeds = len(g)
        n_sig = int((g["p_fdr"] < 0.05).sum())
        agg_rows.append(dict(
            dataset=ds, scale=scale, term=term,
            mean_beta=g["beta"].mean(), std_beta=g["beta"].std(ddof=1),
            min_beta=g["beta"].min(), max_beta=g["beta"].max(),
            n_seeds=n_seeds, n_seeds_significant=n_sig,
            frac_seeds_significant=n_sig / n_seeds if n_seeds else np.nan,
            mean_n_edges=g["n"].mean(),
        ))
    agg = pd.DataFrame.from_records(agg_rows)
    agg_path = os.path.join(OUT_DIR, "aggregated_summary.csv")
    agg.to_csv(agg_path, index=False)
    agg.to_pickle(os.path.join(OUT_DIR, "aggregated_summary.pkl"))
    print(f"wrote {agg_path} ({len(agg)} rows)")


if __name__ == "__main__":
    main()
