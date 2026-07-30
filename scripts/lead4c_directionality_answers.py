"""
Lead 4c follow-up -- answers the request list in LEAD4C_DIRECTIONALITY_EXPERIMENTS_NEEDED.md.

Runs items #1-4 from that doc (item #5, multi-seed, stays deferred per the doc's own
priority call). Nothing here touches the paper or re-litigates the architecture-specific
src_out/tgt_in split -- this is only about strengthening Claim 1 ("source-side entropy is
lower than target-side") and its correlation with the walk-vs-GNN AUC gap.

Outputs (outputs/lead4c_entropy_logit_regression/directionality_answers/):
  claim1_table.csv          -- per-dataset H_out/H_in Wilcoxon table + rank-biserial effect size
  auc_gap_bootstrap.csv     -- per-dataset walk-vs-GNN AUC gap (shared edges) + bootstrap CI
  regression_results.json   -- item #3: continuous effect-size regression, point + MC-propagated CI
  meta_analysis.json        -- item #4: DerSimonian-Laird random-effects pooled Claim 1 estimate

Usage:
    .venv/bin/python scripts/lead4c_directionality_answers.py [--n-boot 2000] [--seed 42]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.node_mi_structural_embedding import DATASET_CONFIGS, load_edges_canonical
from scripts.lead4_entropy_heterogeneity import (
    DATASETS, MODELS, WALK_MODELS, _ds_key, load_shared_predictions,
    build_sign_dicts, entropy_lookup, CANON_PREDICTIONS_DEFAULT,
)

GNN_MODELS = tuple(m for m in MODELS if m not in WALK_MODELS)  # ("GINEConv", "SiGAT")
OUT_DIR = os.path.join(ROOT, "outputs", "lead4c_entropy_logit_regression", "directionality_answers")


# ── Fast rank-based AUC (Mann-Whitney U form) — cheap enough for repeated bootstrap ──

def fast_auc(y, p):
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = stats.rankdata(p, method="average")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ── Item: Claim 1 per-dataset H_out vs H_in table (+ rank-biserial effect size) ──

def claim1_for_dataset(ds_name):
    edges = load_edges_canonical(DATASET_CONFIGS[_ds_key(ds_name)]["ds_name"])
    sign_dicts = build_sign_dicts(edges)
    ent = entropy_lookup(sign_dicts)
    out_e, in_e = ent["out"], ent["in"]
    paired = [(out_e[n], in_e[n]) for n in out_e if n in in_e]
    h_out = np.array([a for a, b in paired])
    h_in = np.array([b for a, b in paired])
    diff = h_out - h_in  # negative => confirms professor's direction (H_out < H_in)

    n_paired = len(diff)
    tied = diff == 0
    n_tied = int(tied.sum())
    d = diff[~tied]
    n_nontied = len(d)

    frac_confirm = float((d < 0).mean()) if n_nontied else float("nan")

    # rank-biserial correlation matching the Wilcoxon signed-rank statistic:
    # rank |d|, split rank-sum by sign, r = (R_confirm - R_disconfirm) / total_rank_sum.
    # positive r => supports professor's direction (H_out < H_in).
    if n_nontied:
        ranks = stats.rankdata(np.abs(d))
        r_confirm = ranks[d < 0].sum()
        r_disconfirm = ranks[d > 0].sum()
        total = ranks.sum()
        rank_biserial = float((r_confirm - r_disconfirm) / total)
        # one-sided Wilcoxon, professor's direction (H_out < H_in)
        _, p_confirm = stats.wilcoxon(d, alternative="less")
        # explicit reversed direction (H_out > H_in)
        _, p_reversed = stats.wilcoxon(d, alternative="greater")
    else:
        rank_biserial = float("nan")
        p_confirm = p_reversed = float("nan")

    return {
        "dataset": ds_name,
        "n_paired_nodes": n_paired,
        "mean_H_out": float(h_out.mean()),
        "mean_H_in": float(h_in.mean()),
        "median_H_out": float(np.median(h_out)),
        "median_H_in": float(np.median(h_in)),
        "tie_rate": n_tied / n_paired if n_paired else float("nan"),
        "n_nontied": n_nontied,
        "frac_H_out_lt_H_in_nontied": frac_confirm,
        "rank_biserial_r": rank_biserial,  # + = confirms, - = reversed
        "p_confirm_direction": float(p_confirm),
        "p_reversed_direction": float(p_reversed),
    }


# ── Item #2: bootstrap CI on shared-edge walk-vs-GNN AUC gap ──

def bootstrap_gap_for_dataset(ds_name, shared_preds_ds, n_boot, rng):
    """shared_preds_ds: {model: {u,v,y,p}} already restricted to the shared edge set."""
    y_by_model = {m: np.asarray(shared_preds_ds[m]["y"]) for m in MODELS}
    p_by_model = {m: np.asarray(shared_preds_ds[m]["p"]) for m in MODELS}
    n = len(y_by_model[MODELS[0]])

    point_auc = {m: fast_auc(y_by_model[m], p_by_model[m]) for m in MODELS}
    point_walk_best = max(point_auc[m] for m in WALK_MODELS)
    point_gnn_best = max(point_auc[m] for m in GNN_MODELS)
    point_gap = point_walk_best - point_gnn_best

    boot_gap = np.empty(n_boot)
    boot_walk = np.empty(n_boot)
    boot_gnn = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        walk_aucs = [fast_auc(y_by_model[m][idx], p_by_model[m][idx]) for m in WALK_MODELS]
        gnn_aucs = [fast_auc(y_by_model[m][idx], p_by_model[m][idx]) for m in GNN_MODELS]
        wb, gb = max(walk_aucs), max(gnn_aucs)
        boot_walk[b] = wb
        boot_gnn[b] = gb
        boot_gap[b] = wb - gb

    return {
        "dataset": ds_name,
        "n_shared_edges": n,
        "walk_best_model": WALK_MODELS[int(np.argmax([point_auc[m] for m in WALK_MODELS]))],
        "gnn_best_model": GNN_MODELS[int(np.argmax([point_auc[m] for m in GNN_MODELS]))],
        "point_walk_auc": point_walk_best,
        "point_gnn_auc": point_gnn_best,
        "point_gap": point_gap,
        "boot_gap_mean": float(boot_gap.mean()),
        "boot_gap_se": float(boot_gap.std(ddof=1)),
        "boot_gap_ci_lo": float(np.percentile(boot_gap, 2.5)),
        "boot_gap_ci_hi": float(np.percentile(boot_gap, 97.5)),
        "_boot_gap_samples": boot_gap,  # kept in-process for item #3 MC propagation, not serialized
    }


# ── Item #3: continuous effect-size regression (with MC-propagated CI) ──

def ols_1d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    xbar, ybar = x.mean(), y.mean()
    sxx = ((x - xbar) ** 2).sum()
    sxy = ((x - xbar) * (y - ybar)).sum()
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    resid = y - (intercept + slope * x)
    dof = n - 2
    s2 = (resid ** 2).sum() / dof if dof > 0 else float("nan")
    se_slope = np.sqrt(s2 / sxx) if dof > 0 else float("nan")
    t_stat = slope / se_slope if dof > 0 else float("nan")
    p_val = 2 * stats.t.sf(abs(t_stat), dof) if dof > 0 else float("nan")
    ss_tot = ((y - ybar) ** 2).sum()
    ss_res = (resid ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n": n, "slope": float(slope), "intercept": float(intercept),
        "se_slope": float(se_slope), "t": float(t_stat), "p": float(p_val),
        "r2": float(r2),
    }


def item3_regression(claim1_rows, gap_rows, n_mc, rng):
    ds_order = [r["dataset"] for r in claim1_rows]
    x = np.array([r["rank_biserial_r"] for r in claim1_rows])  # +1 confirm .. -1 reversed
    y_point = np.array([next(g["point_gap"] for g in gap_rows if g["dataset"] == d) for d in ds_order])
    boot_samples = {g["dataset"]: g["_boot_gap_samples"] for g in gap_rows}

    point_fit = ols_1d(x, y_point)
    pearson_r, pearson_p = stats.pearsonr(x, y_point)
    spearman_r, spearman_p = stats.spearmanr(x, y_point)

    # Monte-Carlo propagation: redraw one bootstrap gap sample per dataset per MC
    # iteration, refit slope each time -> slope distribution reflecting BOTH the
    # n=6 cross-dataset spread AND each dataset's own AUC-gap estimation noise.
    n_boot_each = len(next(iter(boot_samples.values())))
    mc_slopes = np.empty(n_mc)
    mc_pearson = np.empty(n_mc)
    for i in range(n_mc):
        y_draw = np.array([boot_samples[d][rng.integers(0, n_boot_each)] for d in ds_order])
        mc_slopes[i] = ols_1d(x, y_draw)["slope"]
        mc_pearson[i] = np.corrcoef(x, y_draw)[0, 1]

    return {
        "x_definition": "rank_biserial_r of Claim 1 (H_out<H_in), +1=fully confirms .. -1=fully reversed",
        "y_definition": "point-estimate shared-edge AUC gap, best(walk_full,walk_localattn4) - best(GINEConv,SiGAT)",
        "dataset_order": ds_order,
        "x": x.tolist(), "y_point": y_point.tolist(),
        "ols_point_estimate": point_fit,
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "mc_slope_mean": float(mc_slopes.mean()),
        "mc_slope_ci_lo": float(np.percentile(mc_slopes, 2.5)),
        "mc_slope_ci_hi": float(np.percentile(mc_slopes, 97.5)),
        "mc_pearson_r_mean": float(mc_pearson.mean()),
        "mc_pearson_r_ci_lo": float(np.percentile(mc_pearson, 2.5)),
        "mc_pearson_r_ci_hi": float(np.percentile(mc_pearson, 97.5)),
        "n_mc": n_mc,
    }


# ── Item #4: DerSimonian-Laird random-effects meta-analysis of Claim 1 ──

def meta_analysis_claim1(claim1_rows):
    logit_y, var_y, labels, ni = [], [], [], []
    for r in claim1_rows:
        p = r["frac_H_out_lt_H_in_nontied"]
        n = r["n_nontied"]
        # avoid degenerate logit/variance at the boundary
        p_c = min(max(p, 1.0 / (2 * n)), 1 - 1.0 / (2 * n))
        yi = float(np.log(p_c / (1 - p_c)))
        vi = float(1.0 / (n * p_c * (1 - p_c)))
        logit_y.append(yi)
        var_y.append(vi)
        labels.append(r["dataset"])
        ni.append(n)
    logit_y = np.array(logit_y)
    var_y = np.array(var_y)
    w_fixed = 1.0 / var_y
    y_fixed = float((w_fixed * logit_y).sum() / w_fixed.sum())

    k = len(logit_y)
    Q = float((w_fixed * (logit_y - y_fixed) ** 2).sum())
    df = k - 1
    c = w_fixed.sum() - (w_fixed ** 2).sum() / w_fixed.sum()
    tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    Q_p = float(stats.chi2.sf(Q, df))

    w_re = 1.0 / (var_y + tau2)
    y_re = float((w_re * logit_y).sum() / w_re.sum())
    var_re = float(1.0 / w_re.sum())
    se_re = np.sqrt(var_re)
    ci_lo_logit, ci_hi_logit = y_re - 1.96 * se_re, y_re + 1.96 * se_re

    def inv_logit(z):
        return float(1 / (1 + np.exp(-z)))

    return {
        "k_datasets": k,
        "per_dataset": {labels[i]: {"logit": float(logit_y[i]), "var": float(var_y[i]), "n_nontied": ni[i]}
                         for i in range(k)},
        "fixed_effect_logit": y_fixed,
        "fixed_effect_prop": inv_logit(y_fixed),
        "Q": Q, "df": df, "Q_p": Q_p, "tau2": tau2, "I2_pct": I2,
        "random_effect_logit": y_re,
        "random_effect_prop": inv_logit(y_re),
        "random_effect_prop_ci_lo": inv_logit(ci_lo_logit),
        "random_effect_prop_ci_hi": inv_logit(ci_hi_logit),
        "null_prop": 0.5,
        "excludes_null": not (inv_logit(ci_lo_logit) <= 0.5 <= inv_logit(ci_hi_logit)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-mc", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=== Claim 1 per-dataset (H_out vs H_in) ===")
    claim1_rows = [claim1_for_dataset(ds) for ds in DATASETS]
    for r in claim1_rows:
        print(f"{r['dataset']:<16} frac_confirm={r['frac_H_out_lt_H_in_nontied']:.3f}  "
              f"rank_biserial_r={r['rank_biserial_r']:+.3f}  "
              f"p_confirm={r['p_confirm_direction']:.2e}  p_reversed={r['p_reversed_direction']:.2e}")

    print("\n=== Loading shared-edge predictions + bootstrapping AUC gap ===")
    shared_preds = load_shared_predictions(CANON_PREDICTIONS_DEFAULT)
    gap_rows = []
    for ds in DATASETS:
        print(f"  bootstrapping {ds} (n_boot={args.n_boot}) ...")
        g = bootstrap_gap_for_dataset(ds, shared_preds[ds], args.n_boot, rng)
        gap_rows.append(g)
        print(f"    {ds:<16} point_gap={g['point_gap']:+.4f}  "
              f"boot CI=[{g['boot_gap_ci_lo']:+.4f}, {g['boot_gap_ci_hi']:+.4f}]  "
              f"(walk={g['walk_best_model']}, gnn={g['gnn_best_model']})")

    print("\n=== Item 3: continuous regression (gap ~ rank_biserial_r) ===")
    reg = item3_regression(claim1_rows, gap_rows, args.n_mc, rng)
    print(json.dumps({k: v for k, v in reg.items() if not isinstance(v, list)}, indent=2))

    print("\n=== Item 4: DerSimonian-Laird random-effects meta-analysis of Claim 1 (all 6) ===")
    meta_all6 = meta_analysis_claim1(claim1_rows)
    print(json.dumps(meta_all6, indent=2))

    print("\n=== Item 4b: same, restricted to the 4 datasets where Claim 1 held (excl. wiki-elec/wiki-rfa) ===")
    confirmed_ds = {"bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221"}
    meta_confirmed4 = meta_analysis_claim1([r for r in claim1_rows if r["dataset"] in confirmed_ds])
    print(json.dumps(meta_confirmed4, indent=2))

    # ── persist ──
    import csv
    with open(os.path.join(OUT_DIR, "claim1_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(claim1_rows[0].keys()))
        w.writeheader()
        w.writerows(claim1_rows)

    with open(os.path.join(OUT_DIR, "auc_gap_bootstrap.csv"), "w", newline="") as f:
        fieldnames = [k for k in gap_rows[0].keys() if not k.startswith("_")]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for g in gap_rows:
            w.writerow({k: g[k] for k in fieldnames})

    with open(os.path.join(OUT_DIR, "regression_results.json"), "w") as f:
        json.dump(reg, f, indent=2)

    with open(os.path.join(OUT_DIR, "meta_analysis.json"), "w") as f:
        json.dump({"all_6": meta_all6, "confirmed_4_excl_wiki": meta_confirmed4}, f, indent=2)

    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
