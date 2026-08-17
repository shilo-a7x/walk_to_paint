"""
Plot: SiGAT 4-term node-entropy regression, POOLED across all 6 datasets (shared
slope + per-dataset intercepts), mean +- SD across 10 seeds (42-51).
Reads outputs/lead4c_sigat_multiseed_node4_pooled/aggregated_summary.csv (built by
scripts/lead4c_sigat_multiseed_node4_pooled.py) -- pure rendering, no recomputation.

Usage:
    .venv/bin/python scripts/plot_sigat_multiseed_node4_pooled.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV = os.path.join(ROOT, "outputs", "lead4c_sigat_multiseed_node4_pooled", "aggregated_summary.csv")
OUT_PNG = os.path.join(ROOT, "outputs", "lead4c_sigat_multiseed_node4_pooled",
                       "sigat_multiseed_node4_pooled_forest.png")

TERM_ORDER = ["src_out", "tgt_in", "src_in", "tgt_out"]
TERM_LABEL = {
    "src_out": "src_out — how consistently u rates others",
    "tgt_in": "tgt_in — how contested v's reputation is",
    "src_in": "src_in — how consistently u is rated",
    "tgt_out": "tgt_out — how consistently v rates others",
}
SLOPE_COLOR = "#2a78d6"  # single series -> one accent hue (slot 1)
ROBUST_THRESHOLD = 8

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "wiki-elec", "wiki-rfa", "epinions", "slashdot090221"]
DATASET_COLOR = {  # same mapping as the per-dataset package, for visual continuity
    "bitcoin-alpha": "#2a78d6", "bitcoin-otc": "#eb6834", "wiki-elec": "#1baf7a",
    "wiki-rfa": "#eda100", "epinions": "#e87ba4", "slashdot090221": "#008300",
}


def main():
    agg = pd.read_csv(IN_CSV)
    z = agg[agg["scale"] == "zscored"]
    slopes = z[~z["is_dataset_intercept"]].set_index("term")
    intercepts = z[z["is_dataset_intercept"]].copy()
    intercepts["dataset"] = intercepts["term"].str.replace("ds_", "", regex=False)
    intercepts = intercepts.set_index("dataset")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})

    # ── panel 1: the 4 shared slopes ──
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
    ax1.set_title("Shared slopes (n=173,072 edges, all 6 datasets pooled)", fontsize=10.5, pad=8)
    ax1.grid(axis="x", color="#e8e7e2", linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax1.spines[s].set_color("#c9c8c2")
    ax1.tick_params(colors="#52514e", labelsize=9)

    # ── panel 2: the 6 per-dataset intercepts (context, not the headline) ──
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

    fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
