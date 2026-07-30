"""
Lead 4c -- joint logistic regression of per-edge correctness on sign-entropy.

Lead 4 / 4b bin entropy into buckets and compare per-bucket AUC. This script
instead fits, per (dataset|pooled, model, node-entropy variant, two-hop
variant), a single logistic regression of per-edge correctness on THREE
continuous entropy terms at once:

    logit(P(correct_i)) = b0 + b_src*H_src(u_i) + b_tgt*H_tgt(v_i)
                              + b_2hop*H_2hop(edge_i)            [+ dataset FE]

H_src/H_tgt come from Lead 4's 4 node-entropy variants (out_out, in_in,
out_in, inout_inout) -- kept as TWO separate terms (not collapsed to one
"node entropy" feature), because collapsing would average away exactly the
u/v asymmetry that made in_in strong and out_in weak in Lead 4. H_2hop comes
from Lead 4b's 3 two-hop path-consistency variants (out, in, inout). This
gives a direct, signed, p-valued test of "does entropy predict correctness,
and does the sign differ between the walk model and the GNNs" instead of a
qualitative bucket-AUC-drop comparison -- same hypothesis, sharper tool.

Ideal-world prediction: GNN b_src/b_tgt/b_2hop < 0 (more heterogeneity -> more
wrong), walk model's b's ~0 or > 0 (robust / unaffected).

Same-edge ground truth, same source as Lead 4/4b: every model's predictions
come from predictions_raw_canonical.pkl restricted to the shared edge set
(load_shared_predictions). Entropy/2-hop bases are built ONCE per dataset by
reusing Lead 4's sign-entropy lookup and Lead 4b's two-hop-count machinery
(no BFS/adjacency recomputation, no recompute across models or variants --
that's the only expensive part and it's shared).

Clustering caveat (handled): edges sharing an endpoint are not independent
draws -- H_src(u) repeats across every out-edge of u, H_tgt(v) across every
in-edge of v. Naive (i.i.d.) SEs are anti-conservative. We report two-way
cluster-robust SEs (Cameron-Gelbach-Miller 2011: V = V_u + V_v - V_{u,v})
alongside the naive ones.

Stages (compute/fit/plot separable, same convention as lead4*):

    # build the joined per-edge feature table (u,v,y,p,correct + all entropy
    # bases), once per dataset, cached to disk
    python scripts/lead4c_entropy_logit_regression.py --mode compute --datasets all

    # fit all (dataset|pooled) x model x variant-combo regressions, save the
    # flat results table
    python scripts/lead4c_entropy_logit_regression.py --mode fit

    # regenerate plots/report from the saved fit table, no recomputation
    python scripts/lead4c_entropy_logit_regression.py --mode plot

    # default: all three
    python scripts/lead4c_entropy_logit_regression.py --mode all --datasets all
"""
import argparse
import itertools
import os
import pickle
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.node_mi_structural_embedding import DATASET_CONFIGS, load_edges_canonical
from scripts import lead4_entropy_heterogeneity as L4
from scripts import lead4_twohop_path_consistency as L4b
from scripts.lead4_entropy_heterogeneity import (
    DATASETS, MODELS, WALK_MODELS, _ds_key, load_shared_predictions,
    CANON_PREDICTIONS_DEFAULT,
)

OUT_DIR_DEFAULT = os.path.join(ROOT, "outputs", "lead4c_entropy_logit_regression")
COMPUTE_FILE = "joined_table.pkl"
FIT_FILE = "fit_results.pkl"
FIT_CSV = "fit_results.csv"

NODE_VARIANTS = {  # name -> (src direction, tgt direction), matches L4.VARIANTS
    "out_out":     ("out", "out"),
    "in_in":       ("in", "in"),
    "out_in":      ("out", "in"),
    "inout_inout": ("inout", "inout"),
}
TWOHOP_VARIANTS = ["out", "in", "inout"]  # matches L4b.VARIANTS

MODEL_COLOR = {
    "walk_full": "#08519c", "walk_localattn4": "#6baed6",
    "GINEConv": "#a63603", "SiGAT": "#fd8d3c",
}


# ── Compute stage: one joined per-edge feature table per (dataset, model) ────

def _node_lookup(d, nodes):
    return np.array([d.get(int(n), np.nan) for n in nodes], dtype=np.float64)


def _count_lookup(d, nodes):
    n = len(nodes)
    total = np.full(n, np.nan); consistent = np.full(n, np.nan)
    for i, node in enumerate(nodes):
        c = d.get(int(node))
        if c is not None:
            total[i], consistent[i] = c
    return total, consistent


def compute_joined_table(predictions, datasets):
    """predictions: shared-edge {ds: {model: {u,v,y,p}}} (from
    load_shared_predictions). Returns {ds: {model: {col: array} | None}}."""
    out = {}
    for ds_name in datasets:
        print(f"[{ds_name}] building entropy/2-hop bases (once, shared by all models)...")
        edges = load_edges_canonical(DATASET_CONFIGS[_ds_key(ds_name)]["ds_name"])
        sign_dicts = L4.build_sign_dicts(edges)
        ent = L4.entropy_lookup(sign_dicts)            # {"out":{n:H}, "in":{...}, "inout":{...}}
        # per-node sign COUNTS over all incident (inout) edges, so the compact
        # composite can pool counts across both endpoints (count-level pooling,
        # the 'inout' semantics) rather than averaging entropies.
        inout_signs = sign_dicts[2]
        node_tot = {n: len(s) for n, s in inout_signs.items()}
        node_pos = {n: int(np.sum(np.asarray(s) > 0)) for n, s in inout_signs.items()}
        adj_out = L4b.build_adj_out(edges)
        adj_in = L4b.build_adj_in(edges)
        twohop_out = L4b.twohop_counts(adj_out)        # forward 2-hop, keyed by node (anchors "v")
        twohop_in = L4b.twohop_counts(adj_in)           # backward 2-hop, keyed by node (anchors "u")
        # raw out/in degree per node (professor follow-up, 2026-07-05: degree
        # covariates alongside the atomic entropy terms) -- additive only, does
        # not touch any existing key other consumers of this table rely on.
        outdeg, indeg = {}, {}
        for uu, vv, _s in edges:
            outdeg[uu] = outdeg.get(uu, 0) + 1
            indeg[vv] = indeg.get(vv, 0) + 1

        out[ds_name] = {}
        for model in MODELS:
            r = predictions.get(ds_name, {}).get(model)
            if r is None or not len(r["u"]):
                out[ds_name][model] = None
                print(f"  {model}: no data")
                continue
            u = np.asarray(r["u"]); v = np.asarray(r["v"])
            y = np.asarray(r["y"]); p = np.asarray(r["p"])
            correct = ((p >= 0.5).astype(int) == y).astype(int)

            th_out_total_v, th_out_consistent_v = _count_lookup(twohop_out, v)
            th_in_total_u, th_in_consistent_u = _count_lookup(twohop_in, u)
            # pooled node counts over BOTH endpoints' incident edges (u and v)
            tot_u = _node_lookup(node_tot, u); pos_u = _node_lookup(node_pos, u)
            tot_v = _node_lookup(node_tot, v); pos_v = _node_lookup(node_pos, v)
            node_pool_total = np.nan_to_num(tot_u, nan=0.0) + np.nan_to_num(tot_v, nan=0.0)
            node_pool_pos = np.nan_to_num(pos_u, nan=0.0) + np.nan_to_num(pos_v, nan=0.0)

            out[ds_name][model] = dict(
                u=u, v=v, y=y, p=p, correct=correct,
                ent_out_u=_node_lookup(ent["out"], u), ent_in_u=_node_lookup(ent["in"], u),
                ent_inout_u=_node_lookup(ent["inout"], u),
                ent_out_v=_node_lookup(ent["out"], v), ent_in_v=_node_lookup(ent["in"], v),
                ent_inout_v=_node_lookup(ent["inout"], v),
                th_out_total_v=th_out_total_v, th_out_consistent_v=th_out_consistent_v,
                th_in_total_u=th_in_total_u, th_in_consistent_u=th_in_consistent_u,
                node_pool_total=node_pool_total, node_pool_pos=node_pool_pos,
                deg_out_u=_node_lookup(outdeg, u), deg_in_v=_node_lookup(indeg, v),
            )
            print(f"  {model}: n={len(u)}")
    return out


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"saved -> {path}")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Vectorized entropy from raw 2-hop counts (so pooling happens on counts,
#    not on entropies -- matches Lead 4b's _entropy_pooled semantics) ────────

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
    if variant == "out":
        return _entropy_from_totals(table["th_out_total_v"], table["th_out_consistent_v"])
    if variant == "in":
        return _entropy_from_totals(table["th_in_total_u"], table["th_in_consistent_u"])
    if variant == "inout":
        t1, c1 = table["th_out_total_v"], table["th_out_consistent_v"]
        t2, c2 = table["th_in_total_u"], table["th_in_consistent_u"]
        t1f = np.nan_to_num(t1, nan=0.0); c1f = np.nan_to_num(c1, nan=0.0)
        t2f = np.nan_to_num(t2, nan=0.0); c2f = np.nan_to_num(c2, nan=0.0)
        total = t1f + t2f; consistent = c1f + c2f
        avail = ~(np.isnan(t1) & np.isnan(t2))
        total_or_nan = np.where(avail, total, np.nan)
        return _entropy_from_totals(total_or_nan, consistent)
    raise ValueError(variant)


def node_entropy_columns(table, node_variant):
    src_dir, tgt_dir = NODE_VARIANTS[node_variant]
    return table[f"ent_{src_dir}_u"], table[f"ent_{tgt_dir}_v"]


# ── Atomic-direction decomposition (Phase-1 headline) ────────────────────────
# The 12 (node-variant x two-hop-variant) combos are overlapping pairings of a
# small set of ATOMIC directional entropies. Entering all of them in ONE model
# dissolves the "which combo / which is the null control" arbitrariness: each
# direction gets one partial coefficient, and the irrelevant directions simply
# come out ~0 (an empirical output, not an assumed placebo). No direction is
# privileged a priori -- "relevant/placebo" are post-hoc labels only.

ATOMIC_TERMS = ["src_out", "src_in", "tgt_out", "tgt_in", "twohop_in", "twohop_out"]
ATOMIC_LABELS = {  # human-readable, for axes/legends
    "src_out":    "src u: out-edges\n(how u rates others)",
    "src_in":     "src u: in-edges\n(how others rate u)",
    "tgt_out":    "tgt v: out-edges\n(how v rates others)",
    "tgt_in":     "tgt v: in-edges\n(how others rate v)",
    "twohop_in":  "2-hop into u\n(s->t->u consistency)",
    "twohop_out": "2-hop from v\n(v->m->k consistency)",
}


def atomic_feature_columns(table):
    """The 6 atomic directional entropy features for one joined per-edge table.
    `inout`/pooled variants are omitted as redundant (out+in already span them)."""
    return {
        "src_out": table["ent_out_u"],
        "src_in": table["ent_in_u"],
        "tgt_out": table["ent_out_v"],
        "tgt_in": table["ent_in_v"],
        "twohop_in": twohop_entropy_column(table, "in"),    # anchored at source u
        "twohop_out": twohop_entropy_column(table, "out"),  # anchored at target v
    }


# ── Degree-augmented atomic model (professor follow-up, 2026-07-05) ──────────
# Extends the 6-term atomic model with 2 more covariates -- log out-degree(u),
# log in-degree(v) -- to test directly whether "in-degree distribution is
# broader, so paths learn more from incoming than outgoing" shows up as a
# model-specific coefficient, the same way the entropy asymmetry did. Kept as
# fully separate terms/columns/fit functions/spec tags below so the original
# 6-term ATOMIC_TERMS model and its fits are completely untouched.

DEGREE_TERMS = ["log_outdeg_u", "log_indeg_v"]
DEGREE_LABELS = {
    "log_outdeg_u": "log out-degree(u)",
    "log_indeg_v": "log in-degree(v)",
}
ATOMIC_DEGREE_TERMS = ATOMIC_TERMS + DEGREE_TERMS


def degree_feature_columns(table):
    return {
        "log_outdeg_u": np.log(table["deg_out_u"]),
        "log_indeg_v": np.log(table["deg_in_v"]),
    }


def atomic_degree_feature_columns(table):
    cols = dict(atomic_feature_columns(table))
    cols.update(degree_feature_columns(table))
    return cols


# ── Compact composite: ONE node-entropy number + ONE path-entropy number ─────
# Collapses the 6 atoms into two predictors WITHOUT averaging variants:
#   b_node = entropy of the sign-COUNTS pooled across both endpoints' incident
#            edges (count-level pooling, the 'inout' semantics)
#   b_path = entropy of the 2-hop consistency counts pooled over both anchors
# Caveat: pooling cancels the within-node direction asymmetry (src_out vs
# tgt_in run opposite ways), so this is a COMPANION to the atomic forest, not a
# replacement -- read the two together.

COMPOSITE_TERMS = ["b_node", "b_path"]
COMPOSITE_LABELS = {
    "b_node": "node entropy\n(both endpoints' incident signs, pooled)",
    "b_path": "path entropy\n(2-hop sign-consistency, pooled)",
}


def composite_feature_columns(table):
    return {
        "b_node": _entropy_from_totals(table["node_pool_total"], table["node_pool_pos"]),
        "b_path": twohop_entropy_column(table, "inout"),
    }


# ── Two-way cluster-robust covariance (Cameron-Gelbach-Miller 2011) ──────────

def cluster_robust_2way(result, groups_u, groups_v):
    """V_2way = V_u + V_v - V_{(u,v)}. Falls back to one-way (u) if the
    intersection covariance is not PSD-safe to subtract (rare, tiny clusters)."""
    from statsmodels.stats.sandwich_covariance import cov_cluster
    exog = result.model.exog
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


# ── Fit stage ─────────────────────────────────────────────────────────────────

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
    row for the intercept (term="const") if add_const, or None if degenerate
    (too few rows, no variance, or separation).

    add_const=False is for pooled fits that already include a FULL set of
    dataset-indicator columns (one per dataset, no reference level dropped) --
    fitting a separate global constant alongside all-of-them would be
    redundant (perfectly collinear: the dummies already sum to 1 everywhere).
    Each dummy's own coefficient is then directly that dataset's intercept,
    not an offset from some arbitrarily-chosen reference dataset."""
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
    from scipy import stats as sstats
    pvals = 2 * (1 - sstats.norm.cdf(np.abs(z)))

    # Standardized beta = raw beta * sd(x): rescaling one regressor column by a
    # constant only rescales that column's own coefficient by the inverse
    # factor (the other columns and the fit are unaffected), so this is exact
    # -- no second fit on z-scored features needed.
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


def run_all_fits(joined, datasets):
    records = []
    for ds_name in datasets:
        for model in MODELS:
            table = joined.get(ds_name, {}).get(model)
            if table is None:
                continue
            y = table["correct"]
            for node_var, twohop_var in itertools.product(NODE_VARIANTS, TWOHOP_VARIANTS):
                src, tgt = node_entropy_columns(table, node_var)
                h2h = twohop_entropy_column(table, twohop_var)
                X = {"b_src": src, "b_tgt": tgt, "b_2hop": h2h}
                label = f"{ds_name}/{model}/{node_var}+{twohop_var}"
                rows = _fit_one(y, X, table["u"], table["v"], label)
                if rows is None:
                    continue
                for row in rows:
                    row.update(dataset=ds_name, model=model, spec="marginal3",
                               node_variant=node_var, twohop_variant=twohop_var, pooled=False)
                    records.append(row)
            print(f"  fit {ds_name}/{model}: {len(NODE_VARIANTS) * len(TWOHOP_VARIANTS)} combos done")

    # pooled (all datasets), per model: dataset one-hot as intercept shift (shared slope),
    # and a second spec with entropy x dataset-dummy interactions (varying slope, robustness)
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
        for node_var, twohop_var in itertools.product(NODE_VARIANTS, TWOHOP_VARIANTS):
            srcs, tgts, h2hs, ys, us, vs, dummies = [], [], [], [], [], [], []
            offset_u = 0
            for ds_name, table in parts:
                src, tgt = node_entropy_columns(table, node_var)
                h2h = twohop_entropy_column(table, twohop_var)
                n = len(table["correct"])
                srcs.append(src); tgts.append(tgt); h2hs.append(h2h); ys.append(table["correct"])
                us.append(table["u"] + offset_u); vs.append(table["v"] + offset_u)
                offset_u += int(max(table["u"].max(), table["v"].max())) + 1
                dummies.append(np.full(n, ds_name))
            src_all = np.concatenate(srcs); tgt_all = np.concatenate(tgts)
            h2h_all = np.concatenate(h2hs); y_all = np.concatenate(ys)
            u_all = np.concatenate(us); v_all = np.concatenate(vs)
            dummy_all = np.concatenate(dummies)
            # ALL dataset dummies, no global constant: each dummy's own
            # coefficient IS that dataset's intercept directly (no arbitrary
            # reference dataset to mentally subtract back out).
            X_shared = {"b_src": src_all, "b_tgt": tgt_all, "b_2hop": h2h_all}
            for d in ds_names:
                X_shared[f"ds_{d}"] = (dummy_all == d).astype(np.float64)
            label = f"POOLED/{model}/{node_var}+{twohop_var}/shared-slope"
            rows = _fit_one(y_all, X_shared, u_all, v_all, label, add_const=False)
            if rows is not None:
                for row in rows:
                    row.update(dataset="POOLED", model=model, spec="marginal3", node_variant=node_var,
                               twohop_variant=twohop_var, pooled=True, pooled_spec="shared_slope")
                    records.append(row)

            # interacted: b_src/b_tgt/b_2hop are ds_names[0]'s slope, the
            # *_x_<other> terms are how much each other dataset's slope
            # differs from it -- a reference is intrinsic to "difference from"
            # by definition, unlike the intercept above.
            X_inter = dict(X_shared)
            for other in ds_names[1:]:
                m = (dummy_all == other).astype(np.float64)
                X_inter[f"b_src_x_{other}"] = src_all * m
                X_inter[f"b_tgt_x_{other}"] = tgt_all * m
                X_inter[f"b_2hop_x_{other}"] = h2h_all * m
            label = f"POOLED/{model}/{node_var}+{twohop_var}/interacted"
            rows = _fit_one(y_all, X_inter, u_all, v_all, label, add_const=False)
            if rows is not None:
                for row in rows:
                    row.update(dataset="POOLED", model=model, spec="marginal3", node_variant=node_var,
                               twohop_variant=twohop_var, pooled=True, pooled_spec="interacted")
                    records.append(row)
        print(f"  fit POOLED/{model}: shared + interacted, {len(NODE_VARIANTS) * len(TWOHOP_VARIANTS)} combos")

    df = pd.DataFrame.from_records(records)
    if len(df):
        if "pooled_spec" not in df.columns:
            df["pooled_spec"] = None
        df["p_fdr"] = np.nan
        mask = df["term"].isin(["b_src", "b_tgt", "b_2hop"]) & ~df["pooled"]
        df.loc[mask, "p_fdr"] = _bh_fdr(df.loc[mask, "p"].values)
        mask_pool = df["term"].isin(["b_src", "b_tgt", "b_2hop"]) & df["pooled"] & (df["pooled_spec"] == "shared_slope")
        if mask_pool.any():
            df.loc[mask_pool, "p_fdr"] = _bh_fdr(df.loc[mask_pool, "p"].values)
    return df


def _zscore_and_mask(X, keys, y, u, v, dummy=None):
    """Standardize each array in `keys` (mean 0, sd 1) -- an actual
    preprocessing step run BEFORE the regression, not a post-hoc conversion of
    an already-fitted beta. `X` must contain only the `keys` columns (dataset
    dummies, which are never standardized, are added by the caller AFTER this
    returns).

    Mean/SD are computed on the SAME rows `_fit_one` will actually fit on --
    i.e. rows finite across ALL of `keys` AND `y` jointly (one NaN in any
    entropy column drops that row from the fit; z-scoring on the full
    unmasked column would use a different, larger sample than the regression
    itself sees, which would not match a literal "z-score, then fit"
    workflow). Returns (X_zscored, y, u, v, dummy) with that mask applied to
    everything consistently, ready to hand straight to _fit_one."""
    Xmat = np.column_stack([np.asarray(X[k], dtype=np.float64) for k in keys])
    mask = np.isfinite(Xmat).all(axis=1) & np.isfinite(y)
    Xmat = Xmat[mask]
    means = Xmat.mean(axis=0); sds = Xmat.std(axis=0)
    sds[sds == 0] = 1.0
    Xz = (Xmat - means) / sds
    out = {k: Xz[:, i] for i, k in enumerate(keys)}
    dummy_masked = dummy[mask] if dummy is not None else None
    return out, y[mask], u[mask], v[mask], dummy_masked


def run_atomic_fits(joined, datasets):
    """Phase-1 headline: ONE logistic regression per (dataset, model) and pooled,
    with all 6 atomic directional entropies entered simultaneously. Each term is
    a partial coefficient; no combo selection, no designated null. Rows tagged
    spec="atomic". FDR is applied within the atomic family separately from the
    marginal3 sweep (term names differ, so the two never mix)."""
    records = []
    for ds_name in datasets:
        for model in MODELS:
            table = joined.get(ds_name, {}).get(model)
            if table is None:
                continue
            X = atomic_feature_columns(table)
            rows = _fit_one(table["correct"], X, table["u"], table["v"],
                            f"{ds_name}/{model}/atomic")
            if rows is None:
                print(f"  atomic {ds_name}/{model}: degenerate, skipped")
                continue
            for row in rows:
                row.update(dataset=ds_name, model=model, spec="atomic",
                           node_variant=None, twohop_variant=None, pooled=False, pooled_spec=None)
                records.append(row)
            print(f"  atomic {ds_name}/{model}: n={rows[0]['n']}")

    # pooled across datasets, shared slope with dataset fixed effects on the intercept
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
        feats = {k: [] for k in ATOMIC_TERMS}
        ys, us, vs, dummies = [], [], [], []
        offset_u = 0
        for ds_name, table in parts:
            cols = atomic_feature_columns(table)
            n = len(table["correct"])
            for k in ATOMIC_TERMS:
                feats[k].append(cols[k])
            ys.append(table["correct"])
            us.append(table["u"] + offset_u); vs.append(table["v"] + offset_u)
            offset_u += int(max(table["u"].max(), table["v"].max())) + 1
            dummies.append(np.full(n, ds_name))
        X = {k: np.concatenate(feats[k]) for k in ATOMIC_TERMS}
        y_all = np.concatenate(ys); u_all = np.concatenate(us); v_all = np.concatenate(vs)
        dummy_all = np.concatenate(dummies)
        # ALL dataset dummies, no global constant -- each one IS that
        # dataset's intercept directly, no reference-dataset bookkeeping.
        for d in ds_names:
            X[f"ds_{d}"] = (dummy_all == d).astype(np.float64)
        rows = _fit_one(y_all, X, u_all, v_all, f"POOLED/{model}/atomic/shared-slope", add_const=False)
        if rows is not None:
            for row in rows:
                row.update(dataset="POOLED", model=model, spec="atomic", node_variant=None,
                           twohop_variant=None, pooled=True, pooled_spec="shared_slope")
                records.append(row)
        print(f"  atomic POOLED/{model}: shared-slope done")

    df = pd.DataFrame.from_records(records)
    if len(df):
        df["p_fdr"] = np.nan
        m = df["term"].isin(ATOMIC_TERMS) & ~df["pooled"]
        df.loc[m, "p_fdr"] = _bh_fdr(df.loc[m, "p"].values)
        mp = df["term"].isin(ATOMIC_TERMS) & df["pooled"] & (df["pooled_spec"] == "shared_slope")
        if mp.any():
            df.loc[mp, "p_fdr"] = _bh_fdr(df.loc[mp, "p"].values)
    return df


def run_atomic_fits_zscored(joined, datasets):
    """Same atomic model as run_atomic_fits, but the 6 entropy columns are
    LITERALLY standardized (mean 0, sd 1) before the regression is fit --
    a genuine separate regression run on standardized inputs, not a post-hoc
    rescaling of an already-fitted raw beta. Dataset dummy columns are left
    as 0/1 (standard practice -- not standardized). Rows tagged
    spec="atomic_zscored"; CI/p/p_fdr here are this fit's own, not derived."""
    records = []
    for ds_name in datasets:
        for model in MODELS:
            table = joined.get(ds_name, {}).get(model)
            if table is None:
                continue
            X, y_m, u_m, v_m, _ = _zscore_and_mask(
                atomic_feature_columns(table), ATOMIC_TERMS, table["correct"], table["u"], table["v"])
            rows = _fit_one(y_m, X, u_m, v_m, f"{ds_name}/{model}/atomic_zscored")
            if rows is None:
                continue
            for row in rows:
                row.update(dataset=ds_name, model=model, spec="atomic_zscored",
                           node_variant=None, twohop_variant=None, pooled=False, pooled_spec=None)
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
        feats = {k: [] for k in ATOMIC_TERMS}
        ys, us, vs, dummies = [], [], [], []
        offset_u = 0
        for ds_name, table in parts:
            cols = atomic_feature_columns(table)
            n = len(table["correct"])
            for k in ATOMIC_TERMS:
                feats[k].append(cols[k])
            ys.append(table["correct"])
            us.append(table["u"] + offset_u); vs.append(table["v"] + offset_u)
            offset_u += int(max(table["u"].max(), table["v"].max())) + 1
            dummies.append(np.full(n, ds_name))
        X_raw = {k: np.concatenate(feats[k]) for k in ATOMIC_TERMS}
        y_all = np.concatenate(ys); u_all = np.concatenate(us); v_all = np.concatenate(vs)
        dummy_all = np.concatenate(dummies)
        # z-score on the POOLED column, before fitting -- same masked rows
        # used for the fit itself, then dummies (never standardized) added
        X, y_all, u_all, v_all, dummy_all = _zscore_and_mask(
            X_raw, ATOMIC_TERMS, y_all, u_all, v_all, dummy_all)
        for d in ds_names:
            X[f"ds_{d}"] = (dummy_all == d).astype(np.float64)
        rows = _fit_one(y_all, X, u_all, v_all,
                        f"POOLED/{model}/atomic_zscored/shared-slope", add_const=False)
        if rows is not None:
            for row in rows:
                row.update(dataset="POOLED", model=model, spec="atomic_zscored", node_variant=None,
                           twohop_variant=None, pooled=True, pooled_spec="shared_slope")
                records.append(row)
        print(f"  atomic_zscored POOLED/{model}: shared-slope done")

    df = pd.DataFrame.from_records(records)
    if len(df):
        df["p_fdr"] = np.nan
        m = df["term"].isin(ATOMIC_TERMS) & ~df["pooled"]
        df.loc[m, "p_fdr"] = _bh_fdr(df.loc[m, "p"].values)
        mp = df["term"].isin(ATOMIC_TERMS) & df["pooled"] & (df["pooled_spec"] == "shared_slope")
        if mp.any():
            df.loc[mp, "p_fdr"] = _bh_fdr(df.loc[mp, "p"].values)
    return df


def run_atomic_degree_fits(joined, datasets):
    """Professor follow-up (2026-07-05): the 6-term atomic model PLUS 2 degree
    covariates (log out-degree(u), log in-degree(v)), entered together in one
    fit. A fully separate code path from run_atomic_fits -- reuses the same
    machinery (_fit_one, cluster-robust SE, all-dummy pooling) but on
    ATOMIC_DEGREE_TERMS/atomic_degree_feature_columns, so the original 6-term
    atomic fits above are untouched. Rows tagged spec="atomic_degree"."""
    records = []
    for ds_name in datasets:
        for model in MODELS:
            table = joined.get(ds_name, {}).get(model)
            if table is None:
                continue
            X = atomic_degree_feature_columns(table)
            rows = _fit_one(table["correct"], X, table["u"], table["v"],
                            f"{ds_name}/{model}/atomic_degree")
            if rows is None:
                print(f"  atomic_degree {ds_name}/{model}: degenerate, skipped")
                continue
            for row in rows:
                row.update(dataset=ds_name, model=model, spec="atomic_degree",
                           node_variant=None, twohop_variant=None, pooled=False, pooled_spec=None)
                records.append(row)
            print(f"  atomic_degree {ds_name}/{model}: n={rows[0]['n']}")

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
        feats = {k: [] for k in ATOMIC_DEGREE_TERMS}
        ys, us, vs, dummies = [], [], [], []
        offset_u = 0
        for ds_name, table in parts:
            cols = atomic_degree_feature_columns(table)
            n = len(table["correct"])
            for k in ATOMIC_DEGREE_TERMS:
                feats[k].append(cols[k])
            ys.append(table["correct"])
            us.append(table["u"] + offset_u); vs.append(table["v"] + offset_u)
            offset_u += int(max(table["u"].max(), table["v"].max())) + 1
            dummies.append(np.full(n, ds_name))
        X = {k: np.concatenate(feats[k]) for k in ATOMIC_DEGREE_TERMS}
        y_all = np.concatenate(ys); u_all = np.concatenate(us); v_all = np.concatenate(vs)
        dummy_all = np.concatenate(dummies)
        for d in ds_names:
            X[f"ds_{d}"] = (dummy_all == d).astype(np.float64)
        rows = _fit_one(y_all, X, u_all, v_all, f"POOLED/{model}/atomic_degree/shared-slope", add_const=False)
        if rows is not None:
            for row in rows:
                row.update(dataset="POOLED", model=model, spec="atomic_degree", node_variant=None,
                           twohop_variant=None, pooled=True, pooled_spec="shared_slope")
                records.append(row)
        print(f"  atomic_degree POOLED/{model}: shared-slope done")

    df = pd.DataFrame.from_records(records)
    if len(df):
        df["p_fdr"] = np.nan
        m = df["term"].isin(ATOMIC_DEGREE_TERMS) & ~df["pooled"]
        df.loc[m, "p_fdr"] = _bh_fdr(df.loc[m, "p"].values)
        mp = df["term"].isin(ATOMIC_DEGREE_TERMS) & df["pooled"] & (df["pooled_spec"] == "shared_slope")
        if mp.any():
            df.loc[mp, "p_fdr"] = _bh_fdr(df.loc[mp, "p"].values)
    return df


def run_atomic_degree_fits_zscored(joined, datasets):
    """Same 8-term model as run_atomic_degree_fits, but all 8 columns
    (6 entropy + 2 log-degree) are literally standardized before fitting, same
    pattern as run_atomic_fits_zscored. Rows tagged spec="atomic_degree_zscored"."""
    records = []
    for ds_name in datasets:
        for model in MODELS:
            table = joined.get(ds_name, {}).get(model)
            if table is None:
                continue
            X, y_m, u_m, v_m, _ = _zscore_and_mask(
                atomic_degree_feature_columns(table), ATOMIC_DEGREE_TERMS,
                table["correct"], table["u"], table["v"])
            rows = _fit_one(y_m, X, u_m, v_m, f"{ds_name}/{model}/atomic_degree_zscored")
            if rows is None:
                continue
            for row in rows:
                row.update(dataset=ds_name, model=model, spec="atomic_degree_zscored",
                           node_variant=None, twohop_variant=None, pooled=False, pooled_spec=None)
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
        feats = {k: [] for k in ATOMIC_DEGREE_TERMS}
        ys, us, vs, dummies = [], [], [], []
        offset_u = 0
        for ds_name, table in parts:
            cols = atomic_degree_feature_columns(table)
            n = len(table["correct"])
            for k in ATOMIC_DEGREE_TERMS:
                feats[k].append(cols[k])
            ys.append(table["correct"])
            us.append(table["u"] + offset_u); vs.append(table["v"] + offset_u)
            offset_u += int(max(table["u"].max(), table["v"].max())) + 1
            dummies.append(np.full(n, ds_name))
        X_raw = {k: np.concatenate(feats[k]) for k in ATOMIC_DEGREE_TERMS}
        y_all = np.concatenate(ys); u_all = np.concatenate(us); v_all = np.concatenate(vs)
        dummy_all = np.concatenate(dummies)
        X, y_all, u_all, v_all, dummy_all = _zscore_and_mask(
            X_raw, ATOMIC_DEGREE_TERMS, y_all, u_all, v_all, dummy_all)
        for d in ds_names:
            X[f"ds_{d}"] = (dummy_all == d).astype(np.float64)
        rows = _fit_one(y_all, X, u_all, v_all,
                        f"POOLED/{model}/atomic_degree_zscored/shared-slope", add_const=False)
        if rows is not None:
            for row in rows:
                row.update(dataset="POOLED", model=model, spec="atomic_degree_zscored", node_variant=None,
                           twohop_variant=None, pooled=True, pooled_spec="shared_slope")
                records.append(row)
        print(f"  atomic_degree_zscored POOLED/{model}: shared-slope done")

    df = pd.DataFrame.from_records(records)
    if len(df):
        df["p_fdr"] = np.nan
        m = df["term"].isin(ATOMIC_DEGREE_TERMS) & ~df["pooled"]
        df.loc[m, "p_fdr"] = _bh_fdr(df.loc[m, "p"].values)
        mp = df["term"].isin(ATOMIC_DEGREE_TERMS) & df["pooled"] & (df["pooled_spec"] == "shared_slope")
        if mp.any():
            df.loc[mp, "p_fdr"] = _bh_fdr(df.loc[mp, "p"].values)
    return df


# ── Minimal 2-term model: src_out + tgt_in only, no 2-hop/path terms ─────────
# User request (2026-07-28): a stripped-down companion to the 6-term atomic model
# that drops both 2-hop/path terms entirely and keeps only the two node-entropy
# directions that actually carry the story (src_out drives the walk-model
# asymmetry, tgt_in drives the GNN asymmetry; src_in/tgt_out/twohop_in/twohop_out
# were all ~null or negligible in the 6-term fit). Fully separate code path --
# reuses _fit_one/cluster-robust SE/all-dummy pooling but on its own term list and
# spec tag, so run_atomic_fits and everything else above is untouched.

SRCTGT_TERMS = ["src_out", "tgt_in"]
SRCTGT_LABELS = {k: ATOMIC_LABELS[k] for k in SRCTGT_TERMS}


def srctgt_feature_columns(table):
    cols = atomic_feature_columns(table)
    return {k: cols[k] for k in SRCTGT_TERMS}


def run_srctgt_fits(joined, datasets):
    """Minimal model: correct ~ src_out + tgt_in (+ dataset FE, pooled). No
    2-hop terms, no src_in/tgt_out. Rows tagged spec='srctgt2'."""
    records = []
    for ds_name in datasets:
        for model in MODELS:
            table = joined.get(ds_name, {}).get(model)
            if table is None:
                continue
            X = srctgt_feature_columns(table)
            rows = _fit_one(table["correct"], X, table["u"], table["v"],
                            f"{ds_name}/{model}/srctgt2")
            if rows is None:
                print(f"  srctgt2 {ds_name}/{model}: degenerate, skipped")
                continue
            for row in rows:
                row.update(dataset=ds_name, model=model, spec="srctgt2",
                           node_variant=None, twohop_variant=None, pooled=False, pooled_spec=None)
                records.append(row)
            print(f"  srctgt2 {ds_name}/{model}: n={rows[0]['n']}")

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
        feats = {k: [] for k in SRCTGT_TERMS}
        ys, us, vs, dummies = [], [], [], []
        offset_u = 0
        for ds_name, table in parts:
            cols = srctgt_feature_columns(table)
            n = len(table["correct"])
            for k in SRCTGT_TERMS:
                feats[k].append(cols[k])
            ys.append(table["correct"])
            us.append(table["u"] + offset_u); vs.append(table["v"] + offset_u)
            offset_u += int(max(table["u"].max(), table["v"].max())) + 1
            dummies.append(np.full(n, ds_name))
        X = {k: np.concatenate(feats[k]) for k in SRCTGT_TERMS}
        y_all = np.concatenate(ys); u_all = np.concatenate(us); v_all = np.concatenate(vs)
        dummy_all = np.concatenate(dummies)
        for d in ds_names:
            X[f"ds_{d}"] = (dummy_all == d).astype(np.float64)
        rows = _fit_one(y_all, X, u_all, v_all, f"POOLED/{model}/srctgt2/shared-slope", add_const=False)
        if rows is not None:
            for row in rows:
                row.update(dataset="POOLED", model=model, spec="srctgt2", node_variant=None,
                           twohop_variant=None, pooled=True, pooled_spec="shared_slope")
                records.append(row)
        print(f"  srctgt2 POOLED/{model}: shared-slope done")

    df = pd.DataFrame.from_records(records)
    if len(df):
        df["p_fdr"] = np.nan
        m = df["term"].isin(SRCTGT_TERMS) & ~df["pooled"]
        df.loc[m, "p_fdr"] = _bh_fdr(df.loc[m, "p"].values)
        mp = df["term"].isin(SRCTGT_TERMS) & df["pooled"] & (df["pooled_spec"] == "shared_slope")
        if mp.any():
            df.loc[mp, "p_fdr"] = _bh_fdr(df.loc[mp, "p"].values)
    return df


def run_srctgt_fits_zscored(joined, datasets):
    """Same 2-term model as run_srctgt_fits, but src_out/tgt_in are literally
    standardized before fitting. Rows tagged spec='srctgt2_zscored'."""
    records = []
    for ds_name in datasets:
        for model in MODELS:
            table = joined.get(ds_name, {}).get(model)
            if table is None:
                continue
            X, y_m, u_m, v_m, _ = _zscore_and_mask(
                srctgt_feature_columns(table), SRCTGT_TERMS, table["correct"], table["u"], table["v"])
            rows = _fit_one(y_m, X, u_m, v_m, f"{ds_name}/{model}/srctgt2_zscored")
            if rows is None:
                continue
            for row in rows:
                row.update(dataset=ds_name, model=model, spec="srctgt2_zscored",
                           node_variant=None, twohop_variant=None, pooled=False, pooled_spec=None)
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
        feats = {k: [] for k in SRCTGT_TERMS}
        ys, us, vs, dummies = [], [], [], []
        offset_u = 0
        for ds_name, table in parts:
            cols = srctgt_feature_columns(table)
            n = len(table["correct"])
            for k in SRCTGT_TERMS:
                feats[k].append(cols[k])
            ys.append(table["correct"])
            us.append(table["u"] + offset_u); vs.append(table["v"] + offset_u)
            offset_u += int(max(table["u"].max(), table["v"].max())) + 1
            dummies.append(np.full(n, ds_name))
        X_raw = {k: np.concatenate(feats[k]) for k in SRCTGT_TERMS}
        y_all = np.concatenate(ys); u_all = np.concatenate(us); v_all = np.concatenate(vs)
        dummy_all = np.concatenate(dummies)
        X, y_all, u_all, v_all, dummy_all = _zscore_and_mask(
            X_raw, SRCTGT_TERMS, y_all, u_all, v_all, dummy_all)
        for d in ds_names:
            X[f"ds_{d}"] = (dummy_all == d).astype(np.float64)
        rows = _fit_one(y_all, X, u_all, v_all,
                        f"POOLED/{model}/srctgt2_zscored/shared-slope", add_const=False)
        if rows is not None:
            for row in rows:
                row.update(dataset="POOLED", model=model, spec="srctgt2_zscored", node_variant=None,
                           twohop_variant=None, pooled=True, pooled_spec="shared_slope")
                records.append(row)
        print(f"  srctgt2_zscored POOLED/{model}: shared-slope done")

    df = pd.DataFrame.from_records(records)
    if len(df):
        df["p_fdr"] = np.nan
        m = df["term"].isin(SRCTGT_TERMS) & ~df["pooled"]
        df.loc[m, "p_fdr"] = _bh_fdr(df.loc[m, "p"].values)
        mp = df["term"].isin(SRCTGT_TERMS) & df["pooled"] & (df["pooled_spec"] == "shared_slope")
        if mp.any():
            df.loc[mp, "p_fdr"] = _bh_fdr(df.loc[mp, "p"].values)
    return df


def run_composite_fits(joined, datasets):
    """Compact companion: ONE node-entropy term + ONE path-entropy term, per
    (dataset, model) and pooled. Rows tagged spec="composite". FDR within the
    composite family (term names b_node/b_path are distinct from the others)."""
    records = []
    for ds_name in datasets:
        for model in MODELS:
            table = joined.get(ds_name, {}).get(model)
            if table is None:
                continue
            rows = _fit_one(table["correct"], composite_feature_columns(table),
                            table["u"], table["v"], f"{ds_name}/{model}/composite")
            if rows is None:
                continue
            for row in rows:
                row.update(dataset=ds_name, model=model, spec="composite",
                           node_variant=None, twohop_variant=None, pooled=False, pooled_spec=None)
                records.append(row)

    all_ds_for_model = {}
    for ds_name in datasets:
        for model in MODELS:
            if joined.get(ds_name, {}).get(model) is not None:
                all_ds_for_model.setdefault(model, []).append((ds_name, joined[ds_name][model]))

    for model, parts in all_ds_for_model.items():
        if len(parts) < 2:
            continue
        ds_names = [d for d, _ in parts]
        feats = {k: [] for k in COMPOSITE_TERMS}
        ys, us, vs, dummies = [], [], [], []
        offset_u = 0
        for ds_name, table in parts:
            cols = composite_feature_columns(table)
            n = len(table["correct"])
            for k in COMPOSITE_TERMS:
                feats[k].append(cols[k])
            ys.append(table["correct"])
            us.append(table["u"] + offset_u); vs.append(table["v"] + offset_u)
            offset_u += int(max(table["u"].max(), table["v"].max())) + 1
            dummies.append(np.full(n, ds_name))
        X = {k: np.concatenate(feats[k]) for k in COMPOSITE_TERMS}
        y_all = np.concatenate(ys); u_all = np.concatenate(us); v_all = np.concatenate(vs)
        dummy_all = np.concatenate(dummies)
        # ALL dataset dummies, no global constant -- each one IS that
        # dataset's intercept directly, no reference-dataset bookkeeping.
        for d in ds_names:
            X[f"ds_{d}"] = (dummy_all == d).astype(np.float64)
        rows = _fit_one(y_all, X, u_all, v_all, f"POOLED/{model}/composite/shared-slope", add_const=False)
        if rows is not None:
            for row in rows:
                row.update(dataset="POOLED", model=model, spec="composite", node_variant=None,
                           twohop_variant=None, pooled=True, pooled_spec="shared_slope")
                records.append(row)
        print(f"  composite POOLED/{model}: shared-slope done")

    df = pd.DataFrame.from_records(records)
    if len(df):
        df["p_fdr"] = np.nan
        m = df["term"].isin(COMPOSITE_TERMS) & ~df["pooled"]
        df.loc[m, "p_fdr"] = _bh_fdr(df.loc[m, "p"].values)
        mp = df["term"].isin(COMPOSITE_TERMS) & df["pooled"] & (df["pooled_spec"] == "shared_slope")
        if mp.any():
            df.loc[mp, "p_fdr"] = _bh_fdr(df.loc[mp, "p"].values)
    return df


def run_composite_fits_zscored(joined, datasets):
    """Same compact model as run_composite_fits, but the 2 entropy columns are
    LITERALLY standardized before fitting (see run_atomic_fits_zscored for the
    rationale). Rows tagged spec="composite_zscored"."""
    records = []
    for ds_name in datasets:
        for model in MODELS:
            table = joined.get(ds_name, {}).get(model)
            if table is None:
                continue
            X, y_m, u_m, v_m, _ = _zscore_and_mask(
                composite_feature_columns(table), COMPOSITE_TERMS, table["correct"], table["u"], table["v"])
            rows = _fit_one(y_m, X, u_m, v_m, f"{ds_name}/{model}/composite_zscored")
            if rows is None:
                continue
            for row in rows:
                row.update(dataset=ds_name, model=model, spec="composite_zscored",
                           node_variant=None, twohop_variant=None, pooled=False, pooled_spec=None)
                records.append(row)

    all_ds_for_model = {}
    for ds_name in datasets:
        for model in MODELS:
            if joined.get(ds_name, {}).get(model) is not None:
                all_ds_for_model.setdefault(model, []).append((ds_name, joined[ds_name][model]))

    for model, parts in all_ds_for_model.items():
        if len(parts) < 2:
            continue
        ds_names = [d for d, _ in parts]
        feats = {k: [] for k in COMPOSITE_TERMS}
        ys, us, vs, dummies = [], [], [], []
        offset_u = 0
        for ds_name, table in parts:
            cols = composite_feature_columns(table)
            n = len(table["correct"])
            for k in COMPOSITE_TERMS:
                feats[k].append(cols[k])
            ys.append(table["correct"])
            us.append(table["u"] + offset_u); vs.append(table["v"] + offset_u)
            offset_u += int(max(table["u"].max(), table["v"].max())) + 1
            dummies.append(np.full(n, ds_name))
        X_raw = {k: np.concatenate(feats[k]) for k in COMPOSITE_TERMS}
        y_all = np.concatenate(ys); u_all = np.concatenate(us); v_all = np.concatenate(vs)
        dummy_all = np.concatenate(dummies)
        X, y_all, u_all, v_all, dummy_all = _zscore_and_mask(
            X_raw, COMPOSITE_TERMS, y_all, u_all, v_all, dummy_all)
        for d in ds_names:
            X[f"ds_{d}"] = (dummy_all == d).astype(np.float64)
        rows = _fit_one(y_all, X, u_all, v_all,
                        f"POOLED/{model}/composite_zscored/shared-slope", add_const=False)
        if rows is not None:
            for row in rows:
                row.update(dataset="POOLED", model=model, spec="composite_zscored", node_variant=None,
                           twohop_variant=None, pooled=True, pooled_spec="shared_slope")
                records.append(row)
        print(f"  composite_zscored POOLED/{model}: shared-slope done")

    df = pd.DataFrame.from_records(records)
    if len(df):
        df["p_fdr"] = np.nan
        m = df["term"].isin(COMPOSITE_TERMS) & ~df["pooled"]
        df.loc[m, "p_fdr"] = _bh_fdr(df.loc[m, "p"].values)
        mp = df["term"].isin(COMPOSITE_TERMS) & df["pooled"] & (df["pooled_spec"] == "shared_slope")
        if mp.any():
            df.loc[mp, "p_fdr"] = _bh_fdr(df.loc[mp, "p"].values)
    return df


# ── Plot stage (atomic-direction headline; 12-combo sweep -> appendix) ───────
#
# Design principles (per user): src and tgt treated symmetrically; the src/tgt
# asymmetry must EMERGE, never be assumed; every figure carries a real legend,
# axis labels/units, a zero/reference line and a stated significance encoding;
# no a-priori "null control" -- relevance is read off the fitted betas.

GNN_MODELS = [m for m in MODELS if m not in WALK_MODELS]
SIG_NOTE = "filled marker / star = BH-FDR p<0.05, hollow = n.s."


def _atomic_pooled(df, model):
    """Pooled shared-slope atomic rows for one model, indexed by term."""
    sub = df[(df["spec"] == "atomic") & df["pooled"] &
             (df["pooled_spec"] == "shared_slope") & (df["model"] == model)]
    return sub.set_index("term")


def _term_feature(table, term):
    return atomic_feature_columns(table)[term]


# ── P1.0  Neutral structure explainer ───────────────────────────────────────

def plot_atomic_explainer(out_dir):
    """Schematic of edge u->v + the 6 atomic directional entropies, and a table
    showing the 12 legacy combos as overlapping pairings of those 6 atoms.
    Deliberately NEUTRAL: no relevant/placebo labels (that is an output)."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.2),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})

    # left: schematic
    axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.axis("off")
    ux, vx, ny = 0.30, 0.70, 0.55
    for x, name in [(ux, "u (source)"), (vx, "v (target)")]:
        axL.add_patch(plt.Circle((x, ny), 0.055, fc="#dddddd", ec="black", zorder=3))
        axL.text(x, ny, name.split()[0], ha="center", va="center", fontweight="bold", zorder=4)
        axL.text(x, ny - 0.10, name.split(maxsplit=1)[1], ha="center", va="center", fontsize=8)
    axL.annotate("", xy=(vx - 0.06, ny), xytext=(ux + 0.06, ny),
                 arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#c0392b"))
    axL.text((ux + vx) / 2, ny + 0.035, "edge being predicted (u→v)",
             ha="center", fontsize=9, color="#c0392b")
    axL.text(ux, ny + 0.16, "src_out: u's out-edges\n(how u rates others)",
             ha="center", va="bottom", fontsize=8.5)
    axL.text(ux, ny - 0.20, "src_in: u's in-edges\n(how others rate u)",
             ha="center", va="top", fontsize=8.5)
    axL.text(vx, ny + 0.16, "tgt_out: v's out-edges\n(how v rates others)",
             ha="center", va="bottom", fontsize=8.5)
    axL.text(vx, ny - 0.20, "tgt_in: v's in-edges\n(how others rate v)",
             ha="center", va="top", fontsize=8.5)
    axL.text(ux, 0.06, "twohop_in: s→t→u\npath-sign consistency", ha="center", fontsize=8,
             color="#2c3e50")
    axL.text(vx, 0.06, "twohop_out: v→m→k\npath-sign consistency", ha="center", fontsize=8,
             color="#2c3e50")
    axL.set_title("The 6 atomic directional signals around edge u→v\n"
                  "(each is the binary sign-entropy of that edge set; 0 = uniform, 1 = 50/50)",
                  fontsize=10)

    # right: combo -> atom mapping table
    axR.axis("off")
    rows = [
        ["node-variant", "H_src(u) uses", "H_tgt(v) uses"],
        ["out_out", "src_out", "tgt_out"],
        ["in_in", "src_in", "tgt_in"],
        ["out_in", "src_out", "tgt_in"],
        ["inout_inout", "src_out + src_in", "tgt_out + tgt_in"],
        ["", "", ""],
        ["two-hop variant", "uses", ""],
        ["out", "twohop_out", ""],
        ["in", "twohop_in", ""],
        ["inout", "twohop_out + twohop_in", ""],
    ]
    tbl = axR.table(cellText=rows, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r in (0, 6):
            cell.set_facecolor("#34495e"); cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#cccccc")
    axR.set_title("The 4 node-variants × 3 two-hop variants = 12 combos are just\n"
                  "overlapping pairings of the 6 atoms (e.g. in_in / out_in / inout_inout\n"
                  "all re-use tgt_in). The atomic model enters all 6 at once.", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "atomic_explainer.png")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}")


# ── P1.1.1  THE story figure: atomic forest (both src & tgt, symmetric) ──────

def plot_atomic_forest(df, out_dir):
    """Pooled shared-slope beta (log-odds of correct per +1 bit of entropy) for
    each of the 6 atomic terms, all 4 models, 95% cluster-robust CI. Faint
    per-dataset ticks behind each pooled point. The src/tgt asymmetry and which
    directions matter read straight off this one figure."""
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    n_models = len(MODELS)
    group_h = n_models + 1.3
    yticks, ylabels = [], []
    for ti, term in enumerate(ATOMIC_TERMS):
        base = (len(ATOMIC_TERMS) - 1 - ti) * group_h  # first term on top
        yticks.append(base + (n_models - 1) / 2.0)
        ylabels.append(ATOMIC_LABELS[term])
        for mi, model in enumerate(MODELS):
            yp = base + (n_models - 1 - mi)
            pooled = _atomic_pooled(df, model)
            # faint per-dataset ticks
            perds = df[(df["spec"] == "atomic") & (~df["pooled"]) &
                       (df["model"] == model) & (df["term"] == term)]
            if len(perds):
                ax.scatter(perds["beta"], np.full(len(perds), yp),
                           s=10, color=MODEL_COLOR.get(model, "gray"), alpha=0.25, zorder=1)
            if term not in pooled.index:
                continue
            r = pooled.loc[term]
            ci = 1.96 * r["se_robust"]
            sig = pd.notna(r["p_fdr"]) and r["p_fdr"] < 0.05
            ax.errorbar(r["beta"], yp, xerr=ci, fmt="o", capsize=3, markersize=7,
                        color=MODEL_COLOR.get(model, "gray"),
                        markerfacecolor=MODEL_COLOR.get(model, "gray") if sig else "white",
                        markeredgewidth=1.4, zorder=3)
    ax.axvline(0, color="black", lw=1.0, ls="--")
    for ti in range(1, len(ATOMIC_TERMS)):
        ax.axhline(ti * group_h - 0.65, color="#eeeeee", lw=6, zorder=0)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("pooled β  (log-odds of a CORRECT prediction per +1 bit of entropy)\n"
                  "negative = model does worse where that signal is more heterogeneous")
    handles = [plt.Line2D([0], [0], marker="o", color=c, linestyle="", markersize=8, label=m)
               for m, c in MODEL_COLOR.items()]
    handles.append(plt.Line2D([0], [0], marker="o", color="black", linestyle="",
                              markerfacecolor="white", markersize=8, label="hollow = n.s. (BH-FDR)"))
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.95)
    # how-to-read box
    ax.text(0.985, 0.985,
            "HOW TO READ\n"
            "• left of 0 = model does worse when that signal is heterogeneous\n"
            "• filled = significant (BH-FDR p<0.05), hollow = not\n"
            "• within a row: red (GNN) LEFT of blue (walk) → GNNs hurt more;\n"
            "  red RIGHT of blue → the walk is hurt more",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round", fc="#fffbe6", ec="#bbbbbb"))
    # bottom line
    ax.text(0.5, -0.17,
            "BOTTOM LINE: the walk's edge is direction-specific — GNNs are hurt more only by "
            "tgt_in (contested target reputation);\non src_out (inconsistent rater) the WALK is hurt "
            "more; path/2-hop entropy barely matters for anyone.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#444444")
    ax.set_title("Atomic decomposition — which directional signal degrades each model?\n"
                 "all 6 atoms entered in ONE regression (pooled, dataset fixed effects); "
                 "faint dots = per-dataset estimates", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "atomic_forest.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}")


# ── Compact companion: ONE node-β + ONE path-β per model ────────────────────

def plot_composite_forest(df, out_dir):
    """The compact 2-number view: pooled b_node and b_path per model, 95% CI,
    per-dataset ticks. Read ALONGSIDE the atomic forest (pooling cancels the
    within-node src_out/tgt_in asymmetry)."""
    def _pooled(model):
        s = df[(df["spec"] == "composite") & df["pooled"] &
               (df["pooled_spec"] == "shared_slope") & (df["model"] == model)]
        return s.set_index("term")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    n_models = len(MODELS)
    group_h = n_models + 1.3
    yticks, ylabels = [], []
    for ti, term in enumerate(COMPOSITE_TERMS):
        base = (len(COMPOSITE_TERMS) - 1 - ti) * group_h
        yticks.append(base + (n_models - 1) / 2.0)
        ylabels.append(COMPOSITE_LABELS[term])
        for mi, model in enumerate(MODELS):
            yp = base + (n_models - 1 - mi)
            perds = df[(df["spec"] == "composite") & (~df["pooled"]) &
                       (df["model"] == model) & (df["term"] == term)]
            if len(perds):
                ax.scatter(perds["beta"], np.full(len(perds), yp), s=10,
                           color=MODEL_COLOR.get(model, "gray"), alpha=0.25, zorder=1)
            pooled = _pooled(model)
            if term not in pooled.index:
                continue
            r = pooled.loc[term]
            sig = pd.notna(r["p_fdr"]) and r["p_fdr"] < 0.05
            ax.errorbar(r["beta"], yp, xerr=1.96 * r["se_robust"], fmt="o", capsize=3,
                        markersize=7, color=MODEL_COLOR.get(model, "gray"),
                        markerfacecolor=MODEL_COLOR.get(model, "gray") if sig else "white",
                        markeredgewidth=1.4, zorder=3)
    ax.axvline(0, color="black", lw=1.0, ls="--")
    ax.axhline(group_h - 0.65, color="#eeeeee", lw=6, zorder=0)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("pooled β (log-odds of CORRECT per +1 bit of entropy);  "
                  "filled = BH-FDR p<0.05, hollow = n.s.")
    handles = [plt.Line2D([0], [0], marker="o", color=c, linestyle="", markersize=8, label=m)
               for m, c in MODEL_COLOR.items()]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.95)
    ax.set_title("Compact view — node-entropy vs path-entropy, ONE β each\n"
                 "(count-pooled, NOT a variant average; companion to the atomic forest)",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "composite_forest.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}")


# ── P1.1.2  Beta-behaviour heatmap (replaces the 3 win-matrices) ─────────────

def plot_atomic_heatmap(df, out_dir):
    """6 atomic terms x 4 models, cell = standardized beta (pooled shared-slope),
    diverging colormap centered at 0, star = BH-FDR sig."""
    mat = np.full((len(ATOMIC_TERMS), len(MODELS)), np.nan)
    sig = np.zeros_like(mat, dtype=bool)
    for j, model in enumerate(MODELS):
        pooled = _atomic_pooled(df, model)
        for i, term in enumerate(ATOMIC_TERMS):
            if term in pooled.index:
                mat[i, j] = pooled.loc[term, "beta_std"]
                pf = pooled.loc[term, "p_fdr"]
                sig[i, j] = pd.notna(pf) and pf < 0.05
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    im = ax.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, ("*" if sig[i, j] else "") + f"{mat[i, j]:+.2f}",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold" if sig[i, j] else "normal")
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(ATOMIC_TERMS)))
    ax.set_yticklabels([t.replace("\n", " ") for t in
                        [f"{k} — {ATOMIC_LABELS[k].splitlines()[1]}" for k in ATOMIC_TERMS]],
                       fontsize=8)
    ax.set_title("Standardized β per atomic direction (pooled shared-slope)\n"
                 "* = BH-FDR p<0.05;  blue = better-where-heterogeneous, red = worse", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="standardized β (log-odds per SD of entropy)")
    fig.tight_layout()
    path = os.path.join(out_dir, "atomic_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}")


# ── P1.1.3  Binned accuracy vs entropy (replaces PDP; model-free) ────────────

def plot_atomic_binned_accuracy(joined, datasets, out_dir):
    """For each atomic term: empirical accuracy vs entropy bucket, walk group vs
    GNN group, pooled across datasets. Quantile bins; per-bucket n annotated."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    axes = axes.ravel()
    grp_color = {"walk": "#08519c", "GNN": "#a63603"}
    for ai, term in enumerate(ATOMIC_TERMS):
        ax = axes[ai]
        feats = {"walk": ([], []), "GNN": ([], [])}  # (x, correct)
        for ds in datasets:
            for model in MODELS:
                table = joined.get(ds, {}).get(model)
                if table is None:
                    continue
                grp = "walk" if model in WALK_MODELS else "GNN"
                x = _term_feature(table, term); y = table["correct"]
                m = np.isfinite(x) & np.isfinite(y)
                feats[grp][0].append(x[m]); feats[grp][1].append(y[m])
        # common quantile edges from the pooled (both groups) feature
        allx = np.concatenate([np.concatenate(feats[g][0]) for g in feats if feats[g][0]])
        if not len(allx):
            ax.set_visible(False); continue
        edges = np.unique(np.quantile(allx, np.linspace(0, 1, 6)))
        if len(edges) < 3:
            edges = np.linspace(allx.min(), allx.max() + 1e-9, 4)
        centers = 0.5 * (edges[:-1] + edges[1:])
        for grp in ("walk", "GNN"):
            if not feats[grp][0]:
                continue
            xg = np.concatenate(feats[grp][0]); yg = np.concatenate(feats[grp][1])
            idx = np.clip(np.digitize(xg, edges[1:-1]), 0, len(centers) - 1)
            acc, ns = [], []
            for b in range(len(centers)):
                sel = idx == b
                acc.append(yg[sel].mean() if sel.any() else np.nan)
                ns.append(int(sel.sum()))
            ax.plot(centers, acc, "-o", color=grp_color[grp], label=grp, markersize=5)
            if grp == "GNN":
                for c, n in zip(centers, ns):
                    ax.annotate(f"n={n}", (c, 0.0), textcoords="offset points",
                                xytext=(0, 2), ha="center", fontsize=6, color="#777777",
                                annotation_clip=False)
        ax.set_title(ATOMIC_LABELS[term].replace("\n", "  "), fontsize=9)
        ax.set_xlabel("entropy (bits)"); ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("accuracy (P(correct))"); axes[3].set_ylabel("accuracy (P(correct))")
    handles = [plt.Line2D([0], [0], marker="o", color=c, label=g) for g, c in grp_color.items()]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Empirical accuracy vs entropy, by atomic direction — walk vs GNN "
                 "(pooled across datasets & within-group models)", y=1.06, fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "atomic_binned_accuracy.png")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}")


# ── P1.1.4  Pooling explainer: shared vs interacted (per-dataset) slope ──────

def plot_atomic_pooling_caterpillar(df, out_dir):
    """Shared vs interacted slope, for ALL 6 atomic terms (small multiples). In
    each panel: per-dataset βs (dots+CI) = the 'interacted'/separate-slope
    estimates; the vertical line = that model's single pooled 'shared-slope' β.
    Two representative models overlaid (GINEConv, walk_full)."""
    models_show = ["GINEConv", "walk_full"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey=True)
    axes = axes.ravel()
    for ai, term in enumerate(ATOMIC_TERMS):
        ax = axes[ai]
        ds_all = [d for d in DATASETS if d in set(
            df[(df["spec"] == "atomic") & (~df["pooled"]) & (df["term"] == term)]["dataset"])]
        for mi, model in enumerate(models_show):
            perds = df[(df["spec"] == "atomic") & (~df["pooled"]) &
                       (df["model"] == model) & (df["term"] == term)]
            yoff = (mi - 0.5) * 0.22
            for i, ds in enumerate(ds_all):
                row = perds[perds["dataset"] == ds]
                if not len(row):
                    continue
                r = row.iloc[0]
                ax.errorbar(r["beta"], i + yoff, xerr=1.96 * r["se_robust"], fmt="o",
                            color=MODEL_COLOR.get(model, "gray"), capsize=2, markersize=5)
            pooled = _atomic_pooled(df, model)
            if term in pooled.index:
                ax.axvline(pooled.loc[term, "beta"], color=MODEL_COLOR.get(model, "gray"),
                           lw=2, ls="-", alpha=0.9)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_yticks(range(len(ds_all)))
        ax.set_yticklabels(ds_all, fontsize=7)
        ax.set_title(ATOMIC_LABELS[term].replace("\n", "  "), fontsize=9)
        ax.set_xlabel("β (log-odds / bit)", fontsize=8)
    handles = [plt.Line2D([0], [0], marker="o", color=MODEL_COLOR[m], linestyle="", label=m)
               for m in models_show]
    handles.append(plt.Line2D([0], [0], color="gray", lw=2, label="vertical line = pooled shared slope"))
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Shared vs interacted slope, ALL atomic terms — dots = each dataset's OWN slope "
                 "(interacted); vertical line = ONE slope for all datasets (shared, dataset FE only)",
                 y=1.06, fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "atomic_pooling_caterpillar.png")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}")


# ── P1.1.5  Appendix: the 12-combo sweep, explained by the atoms ─────────────

def _combo_gap_table(df, term):
    """Per combo, pooled-shared-slope GNN-mean minus walk-mean beta for `term`."""
    sub = df[(df["spec"] == "marginal3") & df["pooled"] &
             (df["pooled_spec"] == "shared_slope") & (df["term"] == term)].copy()
    if not len(sub):
        return pd.DataFrame()
    sub["combo"] = sub["node_variant"] + "+" + sub["twohop_variant"]
    walk = sub[sub["model"].isin(WALK_MODELS)].groupby("combo")["beta"].mean()
    gnn = sub[~sub["model"].isin(WALK_MODELS)].groupby("combo")["beta"].mean()
    out = pd.concat([walk.rename("walk"), gnn.rename("gnn")], axis=1).reset_index()
    out["gap"] = out["gnn"] - out["walk"]
    out["node_variant"] = out["combo"].str.split("+").str[0]
    out["src_dir"] = out["node_variant"].map(lambda nv: NODE_VARIANTS[nv][0])
    out["tgt_dir"] = out["node_variant"].map(lambda nv: NODE_VARIANTS[nv][1])
    out["src_has_out"] = out["src_dir"].isin(["out", "inout"])
    out["tgt_has_in"] = out["tgt_dir"].isin(["in", "inout"])
    return out


def _appendix_row(axL, axR, gt, term, group_col, glabels, gcolors, xlabel):
    g = gt.sort_values("gap")
    colors = [gcolors[0] if h else gcolors[1] for h in g[group_col]]
    axL.barh(range(len(g)), g["gap"], color=colors)
    axL.set_yticks(range(len(g))); axL.set_yticklabels(g["combo"], fontsize=8)
    axL.axvline(0, color="black", lw=0.8)
    axL.set_xlabel(xlabel)
    axL.set_title(f"All 12 combos, ranked by the {term} gap", fontsize=10)
    handles = [plt.Line2D([0], [0], marker="s", color=gcolors[0], linestyle="", label=glabels[0]),
               plt.Line2D([0], [0], marker="s", color=gcolors[1], linestyle="", label=glabels[1])]
    axL.legend(handles=handles, fontsize=8, loc="lower left")
    for grp, x in [(True, 0), (False, 1)]:
        vals = gt[gt[group_col] == grp]["gap"].values
        axR.scatter(np.full(len(vals), x) + np.random.uniform(-0.05, 0.05, len(vals)),
                    vals, color=gcolors[0] if grp else gcolors[1], s=40, zorder=3)
        if len(vals):
            axR.hlines(vals.mean(), x - 0.18, x + 0.18, color="black", lw=2, zorder=4)
    axR.axhline(0, color="black", lw=0.8, ls="--")
    axR.set_xticks([0, 1]); axR.set_xticklabels([glabels[0], glabels[1]], fontsize=8)
    axR.set_ylabel(f"GNN − walk β ({term})")
    axR.set_title("Grouped by direction\n(black bar = group mean)", fontsize=10)


def plot_appendix_combos(df, out_dir):
    """Honest secondary view of the legacy 12 combos, for BOTH b_src and b_tgt:
    ranked by the GNN−walk gap and grouped by the relevant direction. Shows the
    combo sweep is just the atoms re-packaged — the tgt gap concentrates where
    the target term uses v's IN-edges; the src side is shown symmetrically."""
    gt = _combo_gap_table(df, "b_tgt")
    gs = _combo_gap_table(df, "b_src")
    if not len(gt) or not len(gs):
        print("  appendix combos: no marginal3 pooled rows"); return
    fig, axes = plt.subplots(2, 2, figsize=(14, 11),
                             gridspec_kw={"width_ratios": [1.3, 1.0]})
    _appendix_row(axes[0, 0], axes[0, 1], gt, "b_tgt", "tgt_has_in",
                  ["target uses v's IN-edges", "target = v's OUT only"],
                  ["#1a9850", "#999999"],
                  "GNN − walk pooled β for b_tgt  (more negative = GNN hurt more)")
    _appendix_row(axes[1, 0], axes[1, 1], gs, "b_src", "src_has_out",
                  ["source uses u's OUT-edges", "source = u's IN only"],
                  ["#6a51a3", "#999999"],
                  "GNN − walk pooled β for b_src  (positive = walk hurt more)")
    fig.suptitle("APPENDIX — the legacy 12-combo sweep is just the atoms re-packaged: the b_tgt gap "
                 "concentrates in tgt_in combos (top), the b_src side shown symmetrically (bottom)",
                 y=1.0, fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "appendix_combo_ranking.png")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}")


# ── P1.4  Model-free correlation / MI (first non-regression test) ────────────

def _mi_binned(x, y, nbins=8):
    """Mutual information (bits) between quantile-binned x and binary y."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 50 or len(np.unique(y)) < 2:
        return np.nan
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    if len(edges) < 3:
        return 0.0
    xb = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    mi = 0.0
    n = len(x)
    for xv in np.unique(xb):
        px = np.mean(xb == xv)
        for yv in (0, 1):
            pxy = np.mean((xb == xv) & (y == yv))
            py = np.mean(y == yv)
            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * np.log2(pxy / (px * py))
    return float(mi)


def compute_correlations(joined, datasets):
    """Spearman rho and binned MI between each atomic entropy term and per-edge
    correctness, per model x (dataset + POOLED). Cross-method check that the
    atomic betas are not a logistic functional-form artifact."""
    from scipy.stats import spearmanr
    records = []

    def _one(ds_label, model, term, x, y):
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if len(x) < 50 or len(np.unique(y)) < 2 or np.std(x) == 0:
            return
        rho, p = spearmanr(x, y)
        records.append(dict(dataset=ds_label, model=model, term=term,
                            spearman_rho=rho, spearman_p=p,
                            mi_bits=_mi_binned(x, y), n=len(x)))

    for ds in datasets:
        for model in MODELS:
            table = joined.get(ds, {}).get(model)
            if table is None:
                continue
            for term in ATOMIC_TERMS:
                _one(ds, model, term, _term_feature(table, term), table["correct"])
    # pooled across datasets per model
    for model in MODELS:
        for term in ATOMIC_TERMS:
            xs, ys = [], []
            for ds in datasets:
                table = joined.get(ds, {}).get(model)
                if table is None:
                    continue
                xs.append(_term_feature(table, term)); ys.append(table["correct"])
            if xs:
                _one("POOLED", model, term, np.concatenate(xs), np.concatenate(ys))
    return pd.DataFrame.from_records(records)


def plot_correlation_heatmap(corr_df, out_dir):
    """6 atomic terms x 4 models, cell = pooled Spearman rho(entropy, correct),
    star = p<0.05. Should track the atomic betas in sign."""
    pooled = corr_df[corr_df["dataset"] == "POOLED"]
    mat = np.full((len(ATOMIC_TERMS), len(MODELS)), np.nan)
    sig = np.zeros_like(mat, dtype=bool)
    for j, model in enumerate(MODELS):
        for i, term in enumerate(ATOMIC_TERMS):
            cell = pooled[(pooled["model"] == model) & (pooled["term"] == term)]
            if len(cell):
                mat[i, j] = cell["spearman_rho"].iloc[0]
                sig[i, j] = cell["spearman_p"].iloc[0] < 0.05
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 0.1
    im = ax.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, ("*" if sig[i, j] else "") + f"{mat[i, j]:+.3f}",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold" if sig[i, j] else "normal")
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(ATOMIC_TERMS)))
    ax.set_yticklabels([f"{k} — {ATOMIC_LABELS[k].splitlines()[1]}" for k in ATOMIC_TERMS],
                       fontsize=8)
    ax.set_title("Model-free check: pooled Spearman ρ(entropy, correct)\n"
                 "* = p<0.05;  should match the atomic-β signs (not a logit artifact)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    fig.tight_layout()
    path = os.path.join(out_dir, "correlation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}")


# ── Report ───────────────────────────────────────────────────────────────────

def _atomic_scorecard(df):
    """Per atomic term, cross-dataset consistency: fraction of datasets where the
    mean GNN beta is MORE negative than the mean walk beta, with a sign-test p."""
    from scipy.stats import binomtest
    sub = df[(df["spec"] == "atomic") & (~df["pooled"])]
    if not len(sub):
        return pd.DataFrame()
    walk = sub[sub["model"].isin(WALK_MODELS)].groupby(["dataset", "term"])["beta"].mean()
    gnn = sub[~sub["model"].isin(WALK_MODELS)].groupby(["dataset", "term"])["beta"].mean()
    cmp = pd.concat([walk.rename("walk"), gnn.rename("gnn")], axis=1).reset_index()
    cmp["gnn_more_negative"] = cmp["gnn"] < cmp["walk"]
    by_term = cmp.groupby("term")["gnn_more_negative"].agg(["mean", "count"])
    by_term["p_sign_test"] = [
        binomtest(int(round(r["mean"] * r["count"])), int(r["count"]), 0.5).pvalue
        for _, r in by_term.iterrows()]
    return by_term.reindex(ATOMIC_TERMS)


def write_report(df, corr_df, out_dir):
    L = []
    L.append("# Lead 4c — atomic-direction entropy regression\n")
    L.append("Per-edge `correct` regressed on the **6 atomic directional sign-entropies** at once:\n")
    L.append("```\nlogit(correct) ~ src_out + src_in + tgt_out + tgt_in + twohop_in + twohop_out  "
             "[+ dataset FE when pooled]\n```\n")
    L.append("Each β is a partial coefficient (effect of that direction holding the others "
             "fixed). Two-way cluster-robust SE on (u,v). `*` = BH-FDR p<0.05 within the atomic "
             "family. No direction is privileged a priori — which ones matter is read off the "
             "fitted βs below. Entropy is in bits (0 = uniform signs, 1 = 50/50).\n")
    L.append("**Common-support caveat:** an edge is dropped if any of the 6 entropies is undefined "
             "(node lacks in- or out-edges), so the atomic-model n is ≤ the per-combo n.\n")

    # 0. TL;DR bottom line (computed from the pooled atomic + composite betas)
    pa0 = df[(df["spec"] == "atomic") & df["pooled"] & (df["pooled_spec"] == "shared_slope")]
    pc0 = df[(df["spec"] == "composite") & df["pooled"] & (df["pooled_spec"] == "shared_slope")]

    def _gap(frame, term):
        w = frame[(frame["term"] == term) & frame["model"].isin(WALK_MODELS)]["beta"].mean()
        g = frame[(frame["term"] == term) & ~frame["model"].isin(WALK_MODELS)]["beta"].mean()
        return w, g, g - w

    L.append("\n## TL;DR — the bottom line\n")
    if len(pa0):
        wt, gt, dt = _gap(pa0, "tgt_in")
        ws, gs, ds_ = _gap(pa0, "src_out")
        L.append(f"- **The walk model's advantage is direction-specific, not blanket robustness.** "
                 f"GNNs are hurt *more* than the walk only on **tgt_in** (contested target reputation): "
                 f"walk β≈{wt:+.2f} vs GNN β≈{gt:+.2f} (gap {dt:+.2f}).")
        L.append(f"- On **src_out** (an inconsistent *rater* as source) — the largest entropy effect of "
                 f"all — the **walk is hurt more**: walk β≈{ws:+.2f} vs GNN β≈{gs:+.2f} (gap {ds_:+.2f}).")
        L.append("- **tgt_out** and **src_in** are near-null; **path/2-hop** entropy barely moves any "
                 "model. So the effect is a specific src/tgt **asymmetry**, not 'entropy hurts GNNs'.")
    if len(pc0):
        wn, gn, dn = _gap(pc0, "b_node")
        wp, gp, dp = _gap(pc0, "b_path")
        L.append(f"- **Compact 2-number companion** (count-pooled, §1b): node-entropy β walk≈{wn:+.2f} / "
                 f"GNN≈{gn:+.2f}; path-entropy β walk≈{wp:+.2f} / GNN≈{gp:+.2f}. NOTE the node number "
                 "pools src_out (walk-worse) with tgt_in (GNN-worse), so it *cancels* the asymmetry — "
                 "read it together with the atomic forest, never alone.")

    # 1. headline pooled table
    L.append("\n## 1. Headline — pooled (shared slope, dataset fixed effects)\n")
    L.append("| term | model | β | 95% CI (robust) | p (FDR) | OR | n |")
    L.append("|---|---|---:|---|---:|---:|---:|")
    pa = df[(df["spec"] == "atomic") & df["pooled"] & (df["pooled_spec"] == "shared_slope")]
    for term in ATOMIC_TERMS:
        for model in MODELS:
            r = pa[(pa["term"] == term) & (pa["model"] == model)]
            if not len(r):
                continue
            r = r.iloc[0]; ci = 1.96 * r["se_robust"]
            star = "*" if (pd.notna(r["p_fdr"]) and r["p_fdr"] < 0.05) else ""
            L.append(f"| {term} | {model} | {r['beta']:+.3f}{star} | "
                     f"[{r['beta'] - ci:+.3f}, {r['beta'] + ci:+.3f}] | "
                     f"{r['p_fdr']:.2e} | {r['odds_ratio']:.3f} | {int(r['n'])} |")

    # 1b. compact composite (one node-beta + one path-beta)
    L.append("\n## 1b. Compact — one node-β + one path-β (count-pooled)\n")
    L.append("`b_node` = entropy of sign-counts pooled across BOTH endpoints' incident edges; "
             "`b_path` = 2-hop consistency entropy pooled over both anchors. NOT a variant average. "
             "Companion to §1 — pooling cancels the src_out/tgt_in asymmetry, so read with the forest.\n")
    L.append("| term | model | β | 95% CI (robust) | p (FDR) | OR | n |")
    L.append("|---|---|---:|---|---:|---:|---:|")
    pcq = df[(df["spec"] == "composite") & df["pooled"] & (df["pooled_spec"] == "shared_slope")]
    for term in COMPOSITE_TERMS:
        for model in MODELS:
            r = pcq[(pcq["term"] == term) & (pcq["model"] == model)]
            if not len(r):
                continue
            r = r.iloc[0]; ci = 1.96 * r["se_robust"]
            star = "*" if (pd.notna(r["p_fdr"]) and r["p_fdr"] < 0.05) else ""
            L.append(f"| {term} | {model} | {r['beta']:+.3f}{star} | "
                     f"[{r['beta'] - ci:+.3f}, {r['beta'] + ci:+.3f}] | "
                     f"{r['p_fdr']:.2e} | {r['odds_ratio']:.3f} | {int(r['n'])} |")

    # 1c. per-dataset intercepts (every dataset reported directly -- no
    # arbitrary reference dataset hidden behind a single "const" + offsets)
    L.append("\n## 1c. Per-dataset intercepts (atomic + compact models)\n")
    L.append("Each dataset gets its own intercept in the pooled fit (the entropy slopes in §1/§1b "
             "are still SHARED across datasets). `baseline_accuracy` = sigmoid(intercept) = predicted "
             "accuracy when every entropy term is 0 (fully homogeneous neighborhood).\n")
    for spec_name, terms_lbl, table_lbl in [("atomic", "atomic", "Atomic"),
                                             ("composite", "composite", "Compact")]:
        sub = df[(df["spec"] == spec_name) & df["pooled"] & (df["pooled_spec"] == "shared_slope") &
                 df["term"].str.startswith("ds_", na=False)]
        if not len(sub):
            continue
        L.append(f"\n**{table_lbl} model:**\n")
        L.append("| model | dataset | intercept | 95% CI | baseline accuracy | n |")
        L.append("|---|---|---:|---|---:|---:|")
        for model in MODELS:
            for ds in DATASETS:
                r = sub[(sub["model"] == model) & (sub["term"] == f"ds_{ds}")]
                if not len(r):
                    continue
                r = r.iloc[0]; ci = 1.96 * r["se_robust"]
                acc = 1 / (1 + np.exp(-r["beta"]))
                L.append(f"| {model} | {ds} | {r['beta']:+.3f} | "
                         f"[{r['beta'] - ci:+.3f}, {r['beta'] + ci:+.3f}] | {acc:.3f} | {int(r['n'])} |")

    # 1d. literally z-scored inputs (a genuine separate regression, not beta*sd)
    L.append("\n## 1d. Same models, entropy inputs literally z-scored before fitting\n")
    L.append("Each entropy column standardized to mean 0 / sd 1 BEFORE the regression is run "
             "(`spec` suffix `_zscored`) -- a real separate fit with its own CI/p, not beta×SD of "
             "the raw fit (the two agree to ~1e-3, confirming no bug, but these are the genuine "
             "z-scored numbers). **This does not and cannot change the walk-vs-GNN comparison for "
             "a given term**: all 4 models share the same entropy columns, so the same SD is "
             "applied to all of them -- it only matters for comparing different terms to each "
             "other within one model.\n")
    for spec_name, terms, table_lbl in [("atomic_zscored", ATOMIC_TERMS, "Atomic"),
                                         ("composite_zscored", COMPOSITE_TERMS, "Compact")]:
        sub = df[(df["spec"] == spec_name) & df["pooled"] & (df["pooled_spec"] == "shared_slope")]
        if not len(sub):
            continue
        L.append(f"\n**{table_lbl} model (z-scored):**\n")
        L.append("| term | model | β (z-scored) | 95% CI | p (FDR) | n |")
        L.append("|---|---|---:|---|---:|---:|")
        for term in terms:
            for model in MODELS:
                r = sub[(sub["term"] == term) & (sub["model"] == model)]
                if not len(r):
                    continue
                r = r.iloc[0]; ci = 1.96 * r["se_robust"]
                star = "*" if (pd.notna(r["p_fdr"]) and r["p_fdr"] < 0.05) else ""
                L.append(f"| {term} | {model} | {r['beta']:+.3f}{star} | "
                         f"[{r['beta'] - ci:+.3f}, {r['beta'] + ci:+.3f}] | {r['p_fdr']:.2e} | {int(r['n'])} |")

    # 2. what the data support (computed, not asserted)
    L.append("\n## 2. What the data support (read off section 1, not assumed)\n")
    walk_pa = pa[pa["model"].isin(WALK_MODELS)].groupby("term")["beta"].mean()
    gnn_pa = pa[~pa["model"].isin(WALK_MODELS)].groupby("term")["beta"].mean()
    for term in ATOMIC_TERMS:
        if term not in walk_pa.index or term not in gnn_pa.index:
            continue
        gap = gnn_pa[term] - walk_pa[term]
        sig_models = pa[(pa["term"] == term) & pa["p_fdr"].notna() &
                        (pa["p_fdr"] < 0.05)]["model"].tolist()
        verdict = ("GNN more-negative than walk" if gap < 0 else "walk more-negative than GNN")
        L.append(f"- **{term}**: walk β≈{walk_pa[term]:+.2f}, GNN β≈{gnn_pa[term]:+.2f} "
                 f"(GNN−walk gap {gap:+.2f}; {verdict}). FDR-sig models: "
                 f"{', '.join(sig_models) if sig_models else 'none'}.")
    L.append("\nThe src/tgt asymmetry (if any) is whatever the above shows — compare the src_* rows "
             "to the tgt_* rows directly.\n")

    # 3. cross-dataset consistency scorecard
    sc = _atomic_scorecard(df)
    if len(sc):
        L.append("\n## 3. Cross-dataset consistency (per-dataset atomic fits)\n")
        L.append("Fraction of the 6 datasets where mean GNN β is more negative than mean walk β, "
                 "with a sign-test p.\n")
        L.append("| term | GNN-more-negative share | n datasets | sign-test p |")
        L.append("|---|---:|---:|---:|")
        for term in ATOMIC_TERMS:
            if term in sc.index and pd.notna(sc.loc[term, "mean"]):
                r = sc.loc[term]
                L.append(f"| {term} | {r['mean']:.0%} | {int(r['count'])} | {r['p_sign_test']:.4f} |")

    # 4. pooling explanation
    L.append("\n## 4. Pooling — shared vs interacted slope (plain language)\n")
    L.append("- **Shared slope** (what section 1 reports): ONE β per term for all datasets; the "
             "datasets are allowed only to shift the intercept (dataset fixed effects). Answers "
             "*\"is the effect the same everywhere?\"*\n")
    L.append("- **Interacted slope**: each dataset gets its OWN β per term (equivalently, the "
             "per-dataset fits). Answers *\"or does the effect vary by dataset?\"* The "
             "`atomic_pooling_caterpillar.png` figure overlays the two: dots = per-dataset "
             "(interacted) βs, red line = the single shared-slope β.\n")

    # 5. correlation cross-check
    if corr_df is not None and len(corr_df):
        L.append("\n## 5. Model-free cross-check — Spearman ρ & MI (pooled)\n")
        L.append("| term | model | Spearman ρ | p | MI (bits) | n |")
        L.append("|---|---|---:|---:|---:|---:|")
        pc = corr_df[corr_df["dataset"] == "POOLED"]
        for term in ATOMIC_TERMS:
            for model in MODELS:
                r = pc[(pc["term"] == term) & (pc["model"] == model)]
                if not len(r):
                    continue
                r = r.iloc[0]
                star = "*" if r["spearman_p"] < 0.05 else ""
                L.append(f"| {term} | {model} | {r['spearman_rho']:+.3f}{star} | "
                         f"{r['spearman_p']:.2e} | {r['mi_bits']:.4f} | {int(r['n'])} |")
        L.append("\nρ signs should match the atomic-β signs — confirmation the effect is not a "
                 "logistic functional-form artifact.\n")

    # 6. appendix
    L.append("\n## 6. Appendix — the legacy 12-combo sweep\n")
    L.append("The 4 node-variants × 3 two-hop variants are overlapping pairings of the 6 atoms "
             "(see `atomic_explainer.png`). `appendix_combo_ranking.png` shows the b_tgt GNN−walk "
             "gap across all 12 combos, grouped by whether the target term uses v's IN-edges; "
             "combos sharing `tgt_in` cluster together, i.e. the atomic model already explains the "
             "combo sweep. The full per-combo numbers remain in `fit_results.csv` "
             "(`spec == 'marginal3'`).\n")

    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["compute", "fit", "plot", "all"], default="all")
    ap.add_argument("--datasets", nargs="+", default=["all"])
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--predictions", default=CANON_PREDICTIONS_DEFAULT)
    args = ap.parse_args()
    datasets = DATASETS if args.datasets == ["all"] else args.datasets
    os.makedirs(args.out_dir, exist_ok=True)
    compute_path = os.path.join(args.out_dir, COMPUTE_FILE)
    fit_path = os.path.join(args.out_dir, FIT_FILE)
    fit_csv = os.path.join(args.out_dir, FIT_CSV)
    corr_path = os.path.join(args.out_dir, "correlations.csv")

    if args.mode in ("compute", "all"):
        preds = load_shared_predictions(args.predictions, datasets)
        joined = compute_joined_table(preds, datasets)
        save_pickle(joined, compute_path)

    if args.mode in ("fit", "all"):
        joined = load_pickle(compute_path)
        df_m = run_all_fits(joined, datasets)              # legacy 12-combo sweep (spec="marginal3")
        df_a = run_atomic_fits(joined, datasets)           # atomic decomposition (spec="atomic")
        df_c = run_composite_fits(joined, datasets)        # compact node/path (spec="composite")
        df_az = run_atomic_fits_zscored(joined, datasets)  # atomic, literally z-scored inputs
        df_cz = run_composite_fits_zscored(joined, datasets)  # compact, literally z-scored inputs
        df_ad = run_atomic_degree_fits(joined, datasets)   # atomic + 2 degree covariates (spec="atomic_degree")
        df_adz = run_atomic_degree_fits_zscored(joined, datasets)  # same, literally z-scored inputs
        df_st = run_srctgt_fits(joined, datasets)          # minimal src_out+tgt_in only (spec="srctgt2")
        df_stz = run_srctgt_fits_zscored(joined, datasets)  # same, literally z-scored inputs
        df = pd.concat([df_m, df_a, df_c, df_az, df_cz, df_ad, df_adz, df_st, df_stz], ignore_index=True)
        save_pickle(df, fit_path)
        df.to_csv(fit_csv, index=False)
        print(f"wrote {fit_csv} ({len(df)} rows: {len(df_m)} marginal3 + {len(df_a)} atomic + "
              f"{len(df_c)} composite + {len(df_az)} atomic_zscored + {len(df_cz)} composite_zscored + "
              f"{len(df_ad)} atomic_degree + {len(df_adz)} atomic_degree_zscored + "
              f"{len(df_st)} srctgt2 + {len(df_stz)} srctgt2_zscored)")
        corr = compute_correlations(joined, datasets)
        corr.to_csv(corr_path, index=False)
        print(f"wrote {corr_path} ({len(corr)} rows)")

    if args.mode in ("plot", "all"):
        df = load_pickle(fit_path)
        joined = load_pickle(compute_path)
        corr = pd.read_csv(corr_path) if os.path.exists(corr_path) else \
            compute_correlations(joined, datasets)
        # P1.0 explainer
        plot_atomic_explainer(args.out_dir)
        # P1.1 atomic headline figures
        plot_atomic_forest(df, args.out_dir)
        plot_composite_forest(df, args.out_dir)
        plot_atomic_heatmap(df, args.out_dir)
        plot_atomic_binned_accuracy(joined, datasets, args.out_dir)
        plot_atomic_pooling_caterpillar(df, args.out_dir)
        # P1.1.5 appendix + P1.4 correlations
        plot_appendix_combos(df, args.out_dir)
        plot_correlation_heatmap(corr, args.out_dir)
        write_report(df, corr, args.out_dir)


if __name__ == "__main__":
    main()
