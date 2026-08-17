"""
Standalone reproduction: aggregates `../data/per_seed_fits.csv` (600 rows: 10 seeds
x 6 datasets x 5 terms x 2 scales, SiGAT 4-term node-entropy regression fit
independently per dataset per seed) into per-(dataset, term) mean +- SD across
seeds, and renders the forest-plot figure. Only needs numpy/pandas/matplotlib.

What this does NOT reproduce: `per_seed_fits.csv` itself, which requires refitting
a fresh LogisticRegression per seed on that seed's SiGAT node embeddings (large
files) plus the parent repository's graph-loading pipeline -- out of scope for a
compact package. See README.md section 5 for exactly what `per_seed_fits.csv`
contains and how it was built.

Usage:
    python aggregate_and_plot.py --per-seed ../data/per_seed_fits.csv --out-dir ../results
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "wiki-elec", "wiki-rfa", "epinions", "slashdot090221"]
GROUP_SPLIT_AFTER = "wiki-rfa"
DATASET_COLOR = {
    "bitcoin-alpha": "#2a78d6", "bitcoin-otc": "#eb6834", "wiki-elec": "#1baf7a",
    "wiki-rfa": "#eda100", "epinions": "#e87ba4", "slashdot090221": "#008300",
}
TERM_LABEL = {
    "src_out": "src_out\n(how consistently u rates others)",
    "tgt_in": "tgt_in\n(how contested v's reputation is)",
    "src_in": "src_in\n(how consistently u is rated)",
    "tgt_out": "tgt_out\n(how consistently v rates others)",
}
PANEL_LAYOUT = [["src_out", "tgt_in"], ["src_in", "tgt_out"]]
ROBUST_THRESHOLD = 8
NODE4_TERMS = ["src_out", "src_in", "tgt_out", "tgt_in"]


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
        for (ds, seed, scale), idx in df[m].groupby(["dataset", "seed", "scale"]).groups.items():
            df.loc[idx, "p_fdr"] = bh_fdr(df.loc[idx, "p"].values)

    rows = []
    for (ds, scale, term), g in df.groupby(["dataset", "scale", "term"]):
        n_seeds = len(g)
        n_sig = int((g["p_fdr"] < 0.05).sum())
        rows.append(dict(
            dataset=ds, scale=scale, term=term,
            mean_beta=g["beta"].mean(), std_beta=g["beta"].std(ddof=1),
            min_beta=g["beta"].min(), max_beta=g["beta"].max(),
            n_seeds=n_seeds, n_seeds_significant=n_sig,
            frac_seeds_significant=n_sig / n_seeds if n_seeds else np.nan,
            mean_n_edges=g["n"].mean(),
        ))
    return pd.DataFrame.from_records(rows)


def plot(agg, out_png):
    z = agg[agg["scale"] == "zscored"].set_index(["dataset", "term"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=False)
    y_pos = {ds: i for i, ds in enumerate(DATASET_ORDER)}
    n_ds = len(DATASET_ORDER)

    for r, row_terms in enumerate(PANEL_LAYOUT):
        for c, term in enumerate(row_terms):
            ax = axes[r][c]
            ax.axvline(0, color="#8a8a86", linewidth=1, zorder=1)
            if GROUP_SPLIT_AFTER in y_pos:
                split_y = y_pos[GROUP_SPLIT_AFTER] + 0.5
                ax.axhline(split_y, color="#c9c8c2", linewidth=1, linestyle=(0, (3, 2)), zorder=1)
            for ds in DATASET_ORDER:
                row = z.loc[(ds, term)]
                y = y_pos[ds]
                color = DATASET_COLOR[ds]
                robust = row["n_seeds_significant"] >= ROBUST_THRESHOLD
                ax.errorbar(
                    row["mean_beta"], y, xerr=row["std_beta"], fmt="o", color=color,
                    markerfacecolor=color if robust else "white",
                    markeredgecolor=color, markeredgewidth=1.6,
                    markersize=9, elinewidth=2, capsize=3, capthick=2, zorder=3,
                )
            ax.set_yticks(range(n_ds))
            ax.set_yticklabels(DATASET_ORDER, fontsize=9.5)
            ax.set_ylim(-0.7, n_ds - 0.3)
            ax.invert_yaxis()
            ax.set_title(TERM_LABEL[term], fontsize=10.5, pad=8)
            ax.set_xlabel("mean z-scored coefficient (log-odds of correct per 1-SD entropy)", fontsize=8.5)
            ax.grid(axis="x", color="#e8e7e2", linewidth=0.8, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#c9c8c2")
            ax.spines["bottom"].set_color("#c9c8c2")
            ax.tick_params(colors="#52514e", labelsize=9)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#52514e",
              markeredgecolor="#52514e", markersize=9, markeredgewidth=1.6,
              label=f"significant in ≥{ROBUST_THRESHOLD}/10 seeds (p_fdr<0.05)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
              markeredgecolor="#52514e", markersize=9, markeredgewidth=1.6,
              label=f"significant in <{ROBUST_THRESHOLD}/10 seeds"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False,
              fontsize=9.5, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 0.86])
    fig.suptitle(
        "SiGAT: node-entropy effect on prediction accuracy, per dataset\n"
        "mean ± SD across 10 independent seeds (42–51), 4-term model fit separately per dataset",
        fontsize=12.5, y=0.995,
    )
    fig.text(0.5, 0.905,
             "Top row: the two terms that matter. Bottom row: the two that mostly don't. "
             "Dashed line separates the 4 smaller graphs from the 2 largest/densest.",
             ha="center", fontsize=9, color="#52514e")
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-seed", default="../data/per_seed_fits.csv")
    ap.add_argument("--out-dir", default="../results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.per_seed)
    agg = aggregate(df)
    csv_path = os.path.join(args.out_dir, "aggregated_summary.csv")
    agg.to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({len(agg)} rows)")

    png_path = os.path.join(args.out_dir, "sigat_multiseed_node4_forest.png")
    plot(agg, png_path)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
