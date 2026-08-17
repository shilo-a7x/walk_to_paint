"""
Standalone regression library for the src_out / tgt_in entropy-asymmetry analysis.

This is a self-contained EXTRACT of the fitting code in the parent project's
`scripts/lead4c_entropy_logit_regression.py` -- same logic, byte-identical formulas,
but with every dependency on the parent repo (dataset loaders, config system,
plotting) stripped out. It only needs `joined_table.pkl` (shipped alongside this
file in `../data/`) plus numpy/pandas/scipy/statsmodels.

What it does NOT include: the code that builds `joined_table.pkl` from the raw
graphs and trained-model predictions (that requires the full research repo, the
original dataset files, and the trained checkpoints). `joined_table.pkl` already
contains every per-edge feature these functions read, so none of that is needed
to reproduce or extend the regression itself -- see README.md.

No changes to any formula/logic vs. the parent script -- this file is a faithful
subset, not a reimplementation.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sstats

MODELS = ["walk_full", "walk_localattn4", "GINEConv", "SiGAT"]
WALK_MODELS = ("walk_full", "walk_localattn4")

ATOMIC_TERMS = ["src_out", "src_in", "tgt_out", "tgt_in", "twohop_in", "twohop_out"]
ATOMIC_LABELS = {
    "src_out":    "src u: out-edges\n(how u rates others)",
    "src_in":     "src u: in-edges\n(how others rate u)",
    "tgt_out":    "tgt v: out-edges\n(how v rates others)",
    "tgt_in":     "tgt v: in-edges\n(how others rate v)",
    "twohop_in":  "2-hop into u\n(s->t->u consistency)",
    "twohop_out": "2-hop from v\n(v->m->k consistency)",
}

SRCTGT_TERMS = ["src_out", "tgt_in"]
SRCTGT_LABELS = {k: ATOMIC_LABELS[k] for k in SRCTGT_TERMS}

NODE4_TERMS = ["src_out", "src_in", "tgt_out", "tgt_in"]
NODE4_LABELS = {k: ATOMIC_LABELS[k] for k in NODE4_TERMS}


# ── feature builders (read pre-computed columns out of joined_table.pkl) ────

def _entropy_from_totals(total, consistent):
    total = np.asarray(total, dtype=np.float64)
    consistent = np.asarray(consistent, dtype=np.float64)
    avail = ~np.isnan(total) & (total > 0)
    p = np.full_like(total, np.nan)
    p[avail] = consistent[avail] / total[avail]
    ent = np.full_like(total, np.nan)
    ent[avail] = 0.0
    interior = avail & (p > 0) & (p < 1)
    ent[interior] = -(p[interior] * np.log2(p[interior]) +
                       (1 - p[interior]) * np.log2(1 - p[interior]))
    return ent


def twohop_entropy_column(table, variant):
    """variant: 'in' (2-hop path ending at source u) or 'out' (2-hop path
    starting at target v). 'inout' (pooled) is not needed by atomic/srctgt2."""
    if variant == "out":
        return _entropy_from_totals(table["th_out_total_v"], table["th_out_consistent_v"])
    if variant == "in":
        return _entropy_from_totals(table["th_in_total_u"], table["th_in_consistent_u"])
    raise ValueError(variant)


def atomic_feature_columns(table):
    """The 6 atomic directional entropy features for one joined per-edge table."""
    return {
        "src_out": table["ent_out_u"],
        "src_in": table["ent_in_u"],
        "tgt_out": table["ent_out_v"],
        "tgt_in": table["ent_in_v"],
        "twohop_in": twohop_entropy_column(table, "in"),
        "twohop_out": twohop_entropy_column(table, "out"),
    }


def srctgt_feature_columns(table):
    """The minimal 2-term model: src_out + tgt_in only, no 2-hop/path terms."""
    cols = atomic_feature_columns(table)
    return {k: cols[k] for k in SRCTGT_TERMS}


def node4_feature_columns(table):
    """All 4 node-level directional entropies, no 2-hop/path terms."""
    cols = atomic_feature_columns(table)
    return {k: cols[k] for k in NODE4_TERMS}


# ── two-way cluster-robust covariance (Cameron-Gelbach-Miller 2011) ──────────

def cluster_robust_2way(result, groups_u, groups_v):
    """V_2way = V_u + V_v - V_{(u,v)}. Falls back to one-way (u) if the
    intersection covariance is not PSD-safe to subtract (rare, tiny clusters)."""
    from statsmodels.stats.sandwich_covariance import cov_cluster
    pair = groups_u.astype(np.int64) * (groups_v.max() + 1) + groups_v.astype(np.int64)
    try:
        v_u = cov_cluster(result, groups_u)
        v_v = cov_cluster(result, groups_v)
        v_uv = cov_cluster(result, pair)
        v2 = v_u + v_v - v_uv
        if np.all(np.diag(v2) > 0):
            return v2, "cluster_2way(u,v)"
    except Exception as e:
        warnings.warn(f"2-way cluster covariance failed ({e}); falling back to 1-way(u)")
    v_u = cov_cluster(result, groups_u)
    return v_u, "cluster_1way(u)"


def _bh_fdr(pvals):
    p = np.asarray(pvals, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def _fit_one(y, X_named, cluster_u, cluster_v, label, add_const=True):
    """X_named: dict{name: array}. Returns a row of results per term, plus one
    row for the intercept (term='const') if add_const, or None if degenerate.

    add_const=False is used for pooled fits that already include a FULL set of
    dataset-indicator columns (one per dataset, no reference level dropped) --
    each dummy's own coefficient IS that dataset's intercept directly."""
    names = list(X_named.keys())
    X = np.column_stack([X_named[n] for n in names])
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y_, cu, cv = X[mask], y[mask], cluster_u[mask], cluster_v[mask]
    n = len(y_)
    if n < 50 or len(np.unique(y_)) < 2:
        return None
    sds = X.std(axis=0)
    Xd = sm.add_constant(X, has_constant="add") if add_const else X
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Logit(y_, Xd).fit(disp=0, method="newton", maxiter=100)
    except Exception:
        try:
            res = sm.Logit(y_, Xd).fit(disp=0, method="lbfgs", maxiter=200)
        except Exception as e:
            warnings.warn(f"{label}: fit failed ({e})")
            return None
    cov, cov_label = cluster_robust_2way(res, cu, cv)
    se_robust = np.sqrt(np.maximum(np.diag(cov), 0))
    se_naive = res.bse
    z = res.params / np.where(se_robust > 0, se_robust, np.nan)
    pvals = 2 * (1 - sstats.norm.cdf(np.abs(z)))

    pseudo_r2 = 1 - res.llf / res.llnull if res.llnull != 0 else np.nan
    offset = 1 if add_const else 0

    rows = []
    if add_const:
        rows.append(dict(
            term="const", n=n, beta=res.params[0], beta_std=np.nan,
            se_naive=se_naive[0], se_robust=se_robust[0],
            z=z[0], p=pvals[0], odds_ratio=np.exp(res.params[0]),
            cov_method=cov_label, pseudo_r2=pseudo_r2, base_rate=y_.mean(),
        ))
    for i, name in enumerate(names):
        rows.append(dict(
            term=name, n=n, beta=res.params[i + offset], beta_std=res.params[i + offset] * sds[i],
            se_naive=se_naive[i + offset], se_robust=se_robust[i + offset],
            z=z[i + offset], p=pvals[i + offset], odds_ratio=np.exp(res.params[i + offset]),
            cov_method=cov_label, pseudo_r2=pseudo_r2, base_rate=y_.mean(),
        ))
    return rows


def _zscore_and_mask(X, keys, y, u, v, dummy=None):
    """Standardize each array in `keys` (mean 0, sd 1) BEFORE fitting -- a real
    preprocessing step, not a post-hoc rescaling of an already-fitted beta.
    Mean/SD computed on the same rows _fit_one will actually fit on (finite
    across all keys AND y jointly)."""
    Xmat = np.column_stack([np.asarray(X[k], dtype=np.float64) for k in keys])
    mask = np.isfinite(Xmat).all(axis=1) & np.isfinite(y)
    Xmat = Xmat[mask]
    means = Xmat.mean(axis=0); sds = Xmat.std(axis=0)
    sds[sds == 0] = 1.0
    Xz = (Xmat - means) / sds
    out = {k: Xz[:, i] for i, k in enumerate(keys)}
    dummy_masked = dummy[mask] if dummy is not None else None
    return out, y[mask], u[mask], v[mask], dummy_masked


# ── fit runners: per-(dataset,model) fits + one pooled shared-slope fit ──────
# Pooled fit: ALL dataset dummies, add_const=False, so each dummy IS that
# dataset's intercept directly (no reference-category offset). "Shared slope"
# means one src_out/tgt_in coefficient per model across all pooled datasets;
# only the intercept varies per dataset.

def _run_family(joined, datasets, terms, feature_fn, spec_name, zscored):
    records = []
    for ds_name in datasets:
        for model in MODELS:
            table = joined.get(ds_name, {}).get(model)
            if table is None:
                continue
            if zscored:
                X, y_m, u_m, v_m, _ = _zscore_and_mask(
                    feature_fn(table), terms, table["correct"], table["u"], table["v"])
                rows = _fit_one(y_m, X, u_m, v_m, f"{ds_name}/{model}/{spec_name}")
            else:
                X = feature_fn(table)
                rows = _fit_one(table["correct"], X, table["u"], table["v"], f"{ds_name}/{model}/{spec_name}")
            if rows is None:
                continue
            for row in rows:
                row.update(dataset=ds_name, model=model, spec=spec_name,
                           pooled=False, pooled_spec=None)
                records.append(row)

    all_ds_for_model = {}
    for ds_name in datasets:
        for model in MODELS:
            table = joined.get(ds_name, {}).get(model)
            if table is None:
                continue
            all_ds_for_model.setdefault(model, []).append((ds_name, table))

    for model, parts in all_ds_for_model.items():
        if len(parts) < 2:
            continue
        ds_names = [d for d, _ in parts]
        feats = {k: [] for k in terms}
        ys, us, vs, dummies = [], [], [], []
        offset_u = 0
        for ds_name, table in parts:
            cols = feature_fn(table)
            n = len(table["correct"])
            for k in terms:
                feats[k].append(cols[k])
            ys.append(table["correct"])
            us.append(table["u"] + offset_u); vs.append(table["v"] + offset_u)
            offset_u += int(max(table["u"].max(), table["v"].max())) + 1
            dummies.append(np.full(n, ds_name))
        X_raw = {k: np.concatenate(feats[k]) for k in terms}
        y_all = np.concatenate(ys); u_all = np.concatenate(us); v_all = np.concatenate(vs)
        dummy_all = np.concatenate(dummies)
        if zscored:
            X, y_all, u_all, v_all, dummy_all = _zscore_and_mask(
                X_raw, terms, y_all, u_all, v_all, dummy_all)
        else:
            X = X_raw
        for d in ds_names:
            X[f"ds_{d}"] = (dummy_all == d).astype(np.float64)
        rows = _fit_one(y_all, X, u_all, v_all, f"POOLED/{model}/{spec_name}/shared-slope", add_const=False)
        if rows is not None:
            for row in rows:
                row.update(dataset="POOLED", model=model, spec=spec_name,
                           pooled=True, pooled_spec="shared_slope")
                records.append(row)

    df = pd.DataFrame.from_records(records)
    if len(df):
        df["p_fdr"] = np.nan
        m = df["term"].isin(terms) & ~df["pooled"]
        df.loc[m, "p_fdr"] = _bh_fdr(df.loc[m, "p"].values)
        mp = df["term"].isin(terms) & df["pooled"] & (df["pooled_spec"] == "shared_slope")
        if mp.any():
            df.loc[mp, "p_fdr"] = _bh_fdr(df.loc[mp, "p"].values)
    return df


def run_atomic_fits(joined, datasets):
    """Full 6-term atomic model: correct ~ src_out+src_in+tgt_out+tgt_in+twohop_in+twohop_out."""
    return _run_family(joined, datasets, ATOMIC_TERMS, atomic_feature_columns, "atomic", zscored=False)


def run_atomic_fits_zscored(joined, datasets):
    """Same as run_atomic_fits, features z-scored before fitting."""
    return _run_family(joined, datasets, ATOMIC_TERMS, atomic_feature_columns, "atomic_zscored", zscored=True)


def run_srctgt_fits(joined, datasets):
    """Minimal 2-term model: correct ~ src_out + tgt_in. No 2-hop/path terms."""
    return _run_family(joined, datasets, SRCTGT_TERMS, srctgt_feature_columns, "srctgt2", zscored=False)


def run_srctgt_fits_zscored(joined, datasets):
    """Same as run_srctgt_fits, features z-scored before fitting."""
    return _run_family(joined, datasets, SRCTGT_TERMS, srctgt_feature_columns, "srctgt2_zscored", zscored=True)


def run_node4_fits(joined, datasets):
    """4-term model: correct ~ src_out + src_in + tgt_out + tgt_in. No 2-hop/path terms."""
    return _run_family(joined, datasets, NODE4_TERMS, node4_feature_columns, "node4", zscored=False)


def run_node4_fits_zscored(joined, datasets):
    """Same as run_node4_fits, features z-scored before fitting."""
    return _run_family(joined, datasets, NODE4_TERMS, node4_feature_columns, "node4_zscored", zscored=True)
