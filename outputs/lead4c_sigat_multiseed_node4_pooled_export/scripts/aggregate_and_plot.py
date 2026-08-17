"""
Standalone reproduction: aggregates `../data/per_seed_pooled_fits.csv` (200 rows: 10
seeds x 2 scales x (4 shared slopes + 6 dataset intercepts), SiGAT 4-term
node-entropy regression POOLED across all 6 datasets, fit independently per seed)
into per-term mean +- SD across seeds, and renders the forest-plot figure. Only
needs numpy/pandas/matplotlib.

What this does NOT reproduce: `per_seed_pooled_fits.csv` itself, which requires
refitting a fresh LogisticRegression per seed on that seed's SiGAT node embeddings
(large files, one set per seed per dataset) plus the parent repository's
graph-loading pipeline -- out of scope for a compact package. See README.md
section 5 for exactly what `per_seed_pooled_fits.csv` contains and how it was built.

Usage:
    python aggregate_and_plot.py --per-seed ../data/per_seed_pooled_fits.csv --out-dir ../results
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

NODE4_TERMS = ["src_out", "src_in", "tgt_out", "tgt_in"]
TERM_ORDER = ["src_out", "tgt_in", "src_in", "tgt_out"]
TERM_LABEL = {
    "src_out": "src_out — how consistently u rates others",
    "tgt_in": "tgt_in — how contested v's reputation is",
    "src_in": "src_in — how consistently u is rated",
    "tgt_out": "tgt_out — how consistently v rates others",
}
SLOPE_COLOR = "#2a78d6"
ROBUST_THRESHOLD = 8

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "wiki-elec", "wiki-rfa", "epinions", "slashdot090221"]
DATASET_COLOR = {
    "bitcoin-alpha": "#2a78d6", "bitcoin-otc": "#eb6834", "wiki-elec": "#1baf7a",
    "wiki-rfa": "#eda100", "epinions": "#e87ba4", "slashdot090221": "#008300",
}


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def aggregate(df):
    if "p_fdr" not in df.columns or df["p_fdr"].isna().all():
        df = df.copy()
        df["p_fdr"] = np.nan
        m = df["term"].isin(NODE4_TERMS)
        for (seed, scale), idx in df[m].groupby(["seed", "scale"]).groups.items():
            df.loc[idx, "p_fdr"] = bh_fdr(df.loc[idx, "p"].values)

    rows = []
    for (scale, term), g in df.groupby(["scale", "term"]):
        n_seeds = len(g)
        is_slope = term in NODE4_TERMS
        n_sig = int((g["p_fdr"] < 0.05).sum()) if is_slope else None
        rows.append(dict(
            scale=scale, term=term, is_dataset_intercept=not is_slope and term.startswith("ds_"),
            mean_beta=g["beta"].mean(), std_beta=g["beta"].std(ddof=1),
            min_beta=g["beta"].min(), max_beta=g["beta"].max(),
            n_seeds=n_seeds, n_seeds_significant=n_sig,
            frac_seeds_significant=(n_sig / n_seeds) if (is_slope and n_seeds) else np.nan,
        ))
    return pd.DataFrame.from_records(rows)


def plot(agg, out_png):
    z = agg[agg["scale"] == "zscored"]
    slopes = z[~z["is_dataset_intercept"]].set_index("term")
    intercepts = z[z["is_dataset_intercept"]].copy()
    intercepts["dataset"] = intercepts["term"].str.replace("ds_", "", regex=False)
    intercepts = intercepts.set_index("dataset")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})

    ax1.axvline(0, color="#8a8a86", linewidth=1, zorder=1)
    for i, term in enumerate(TERM_ORDER):
        row = slopes.loc[term]
        robust = row["n_seeds_significant"] >= ROBUST_THRESHOLD
        ax1.errorbar(
            row["mean_beta"], i, xerr=row["std_beta"], fmt="o", color=SLOPE_COLOR,
            markerfacecolor=SLOPE_COLOR if robust else "white",
            markeredgecolor=SLOPE_COLOR, markeredgewidth=1.8,
            markersize=11, elinewidth=2.4, capsize=4, capthick=2.4, zorder=3,
        )
    ax1.set_yticks(range(len(TERM_ORDER)))
    ax1.set_yticklabels([TERM_LABEL[t] for t in TERM_ORDER], fontsize=10)
    ax1.set_ylim(-0.6, len(TERM_ORDER) - 0.4)
    ax1.invert_yaxis()
    ax1.set_xlabel("mean z-scored coefficient, pooled\n(shared slope across all 6 datasets)", fontsize=9)
    ax1.set_title("Shared slopes (all 6 datasets pooled)", fontsize=10.5, pad=8)
    ax1.grid(axis="x", color="#e8e7e2", linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax1.spines[s].set_color("#c9c8c2")
    ax1.tick_params(colors="#52514e", labelsize=9)

    for i, ds in enumerate(DATASET_ORDER):
        row = intercepts.loc[ds]
        color = DATASET_COLOR[ds]
        ax2.errorbar(
            row["mean_beta"], i, xerr=row["std_beta"], fmt="o", color=color,
            markerfacecolor=color, markeredgecolor=color, markeredgewidth=1.6,
            markersize=9, elinewidth=2, capsize=3, capthick=2, zorder=3,
        )
    ax2.set_yticks(range(len(DATASET_ORDER)))
    ax2.set_yticklabels(DATASET_ORDER, fontsize=9.5)
    ax2.set_ylim(-0.6, len(DATASET_ORDER) - 0.4)
    ax2.invert_yaxis()
    ax2.set_xlabel("mean intercept (log-odds of correct at zero entropy)", fontsize=9)
    ax2.set_title("Per-dataset intercepts (fixed effects)", fontsize=10.5, pad=8)
    ax2.grid(axis="x", color="#e8e7e2", linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax2.spines[s].set_color("#c9c8c2")
    ax2.tick_params(colors="#52514e", labelsize=9)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SLOPE_COLOR,
              markeredgecolor=SLOPE_COLOR, markersize=10, markeredgewidth=1.8,
              label=f"significant in ≥{ROBUST_THRESHOLD}/10 seeds (p_fdr<0.05)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
              markeredgecolor=SLOPE_COLOR, markersize=10, markeredgewidth=1.8,
              label=f"significant in <{ROBUST_THRESHOLD}/10 seeds"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False,
              fontsize=9.5, bbox_to_anchor=(0.28, -0.06))

    fig.tight_layout(rect=[0, 0.09, 1, 0.78])
    fig.suptitle(
        "SiGAT: node-entropy effect, POOLED across all 6 datasets\n"
        "one shared slope per term, mean ± SD across 10 independent seeds (42–51)",
        fontsize=12.5, y=0.985,
    )
    fig.text(0.5, 0.815,
             "Left: the 4 shared slopes (the pooled headline). Right: how much the baseline\n"
             "(zero-entropy) accuracy differs by dataset — context, not the main result.",
             ha="center", fontsize=9, color="#52514e")
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-seed", default="../data/per_seed_pooled_fits.csv")
    ap.add_argument("--out-dir", default="../results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.per_seed)
    agg = aggregate(df)
    csv_path = os.path.join(args.out_dir, "aggregated_summary.csv")
    agg.to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({len(agg)} rows)")

    png_path = os.path.join(args.out_dir, "sigat_multiseed_node4_pooled_forest.png")
    plot(agg, png_path)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
