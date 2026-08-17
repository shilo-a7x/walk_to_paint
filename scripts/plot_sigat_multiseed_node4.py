"""
Plot: SiGAT 4-term node-entropy regression, mean +- SD across 10 seeds (42-51),
per dataset. Reads outputs/lead4c_sigat_multiseed_node4/aggregated_summary.csv
(built by scripts/lead4c_sigat_multiseed_node4.py) -- pure rendering, no
recomputation, safe to re-run after just tweaking styling.

Usage:
    .venv/bin/python scripts/plot_sigat_multiseed_node4.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV = os.path.join(ROOT, "outputs", "lead4c_sigat_multiseed_node4", "aggregated_summary.csv")
OUT_PNG = os.path.join(ROOT, "outputs", "lead4c_sigat_multiseed_node4", "sigat_multiseed_node4_forest.png")

# Dataset order chosen to visualize the size-driven src_out/tgt_in split found in
# the analysis: the 4 smaller graphs (tgt_in-dominant) grouped together, then the
# 2 largest/densest graphs (src_out-dominant) -- not alphabetical, deliberate.
DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "wiki-elec", "wiki-rfa", "epinions", "slashdot090221"]
GROUP_SPLIT_AFTER = "wiki-rfa"  # horizontal divider drawn after this dataset

# Fixed categorical color per dataset (validated adjacent-pairlist order,
# dataviz skill references/palette.md slots 1-6), assigned in DATASET_ORDER so
# color stays a stable identity cue across all 4 panels.
DATASET_COLOR = {
    "bitcoin-alpha": "#2a78d6",   # slot 1 blue
    "bitcoin-otc":    "#eb6834",  # slot 2 orange
    "wiki-elec":      "#1baf7a",  # slot 3 aqua
    "wiki-rfa":       "#eda100",  # slot 4 yellow
    "epinions":       "#e87ba4",  # slot 5 magenta
    "slashdot090221": "#008300",  # slot 6 green
}
DATASET_LABEL = {
    "bitcoin-alpha": "bitcoin-alpha", "bitcoin-otc": "bitcoin-otc",
    "epinions": "epinions", "slashdot090221": "slashdot090221",
    "wiki-elec": "wiki-elec", "wiki-rfa": "wiki-rfa",
}

TERM_LABEL = {
    "src_out": "src_out\n(how consistently u rates others)",
    "tgt_in": "tgt_in\n(how contested v's reputation is)",
    "src_in": "src_in\n(how consistently u is rated)",
    "tgt_out": "tgt_out\n(how consistently v rates others)",
}
# Top row = the two terms that matter; bottom row = the two that mostly don't.
PANEL_LAYOUT = [["src_out", "tgt_in"], ["src_in", "tgt_out"]]

ROBUST_THRESHOLD = 8  # n_seeds_significant >= this -> filled marker ("robust")


def main():
    agg = pd.read_csv(IN_CSV)
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
                    row["mean_beta"], y, xerr=row["std_beta"],
                    fmt="o", color=color,
                    markerfacecolor=color if robust else "white",
                    markeredgecolor=color, markeredgewidth=1.6,
                    markersize=9, elinewidth=2, capsize=3, capthick=2,
                    zorder=3,
                )

            ax.set_yticks(range(n_ds))
            ax.set_yticklabels([DATASET_LABEL[d] for d in DATASET_ORDER], fontsize=9.5)
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

    fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
