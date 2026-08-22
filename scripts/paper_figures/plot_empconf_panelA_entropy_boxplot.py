"""Plot step for Empirical Confirmation Panel A -- entropy asymmetry boxplot
(H_out vs. H_in per node, six datasets).

Pure rendering: reads aaai2027/figure_data/empconf_panelA_entropy_boxplot.csv
(long format) and empconf_panelA_entropy_ttest.csv (paired-test stats per dataset,
built by extract_empconf_panelA_entropy_boxplot.py -- filename kept for
back-compat, but the caption uses the WILCOXON signed-rank stat, not the
paired t-test; see 2026-07-28 note below). Edit THIS file freely for
color/layout changes -- no recomputation needed.

Wilcoxon, not t-test (switched 2026-07-28): the paired t-test was the initial
choice (per an early, since-clarified request), but the entropy distributions
have a huge zero-inflated point mass (tie_rate 46-79% on 4/6 datasets --
nodes where H_out=H_in=0 exactly), which the t-test doesn't handle
specially and which is exactly the situation the Wilcoxon signed-rank test
(drops exact ties, ranks only the genuine non-zero differences) is designed
for. Wilcoxon is also what's already cited in the paper's own Panel A prose
and in claim1_table.csv -- using the t-test here created a real inconsistency
(text and figure citing two different p-values for the same claim). Both
tests agree on direction/significance for all 6 datasets, so no finding
changes, only which number is reported.

Many nodes are unanimous raters (entropy exactly 0), so on four of the six
datasets the box (25th-75th pct) collapses to a thin line at 0 with a long
whisker/outlier tail above -- that's the real, honestly-plotted distribution
(matches the tie_rate already reported in claim1_table.csv), not a rendering
issue.
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_LONG_CSV = "aaai2027/figure_data/empconf_panelA_entropy_boxplot.csv"
IN_STATS_CSV = "aaai2027/figure_data/empconf_panelA_entropy_ttest.csv"
OUT_PNG = "aaai2027/figures/empconf_panelA_entropy_boxplot.png"

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
DISPLAY_LABEL = {"slashdot090221": "slashdot"}
COLOR_OUT = "#2a78d6"
COLOR_IN = "#eb6834"


def load_long(path):
    data = {ds: {"out": [], "in": []} for ds in DATASET_ORDER}
    with open(path) as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            if ds not in data:
                continue
            data[ds][row["direction"]].append(float(row["entropy"]))
    return data


def load_stats(path):
    stats = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            stats[row["dataset"]] = row
    return stats


def sig_stars(p):
    p = float(p)
    if p < 1e-4:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def main():
    data = load_long(IN_LONG_CSV)
    stat = load_stats(IN_STATS_CSV)

    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    positions_out, positions_in, centers = [], [], []
    width = 0.32
    for i, ds in enumerate(DATASET_ORDER):
        c = i * 1.0
        centers.append(c)
        positions_out.append(c - width / 2 - 0.02)
        positions_in.append(c + width / 2 + 0.02)

    box_kwargs = dict(widths=width, patch_artist=True, showfliers=True,
                       flierprops=dict(marker="o", markersize=2, alpha=0.25, markeredgewidth=0))

    bp_out = ax.boxplot([data[ds]["out"] for ds in DATASET_ORDER], positions=positions_out, **box_kwargs)
    bp_in = ax.boxplot([data[ds]["in"] for ds in DATASET_ORDER], positions=positions_in, **box_kwargs)

    for box in bp_out["boxes"]:
        box.set(facecolor=COLOR_OUT, alpha=0.75, edgecolor="#1c4f8f")
    for box in bp_in["boxes"]:
        box.set(facecolor=COLOR_IN, alpha=0.75, edgecolor="#a8471f")
    for bp, edge in [(bp_out, "#1c4f8f"), (bp_in, "#a8471f")]:
        for part in ["whiskers", "caps", "medians"]:
            for line in bp[part]:
                line.set(color=edge, linewidth=1.1)

    # mean markers (diamonds) -- the prose/caption cite MEAN H_out/H_in, and several
    # datasets have a median (and IQR) collapsed to exactly 0 (high tie rate), so the
    # mean is the only summary visible for those without this marker.
    means_out = [sum(data[ds]["out"]) / len(data[ds]["out"]) for ds in DATASET_ORDER]
    means_in = [sum(data[ds]["in"]) / len(data[ds]["in"]) for ds in DATASET_ORDER]
    ax.scatter(positions_out, means_out, marker="D", s=16, color="#1c4f8f", zorder=5, label=None)
    ax.scatter(positions_in, means_in, marker="D", s=16, color="#a8471f", zorder=5, label=None)

    ax.set_xticks(centers)
    ax.set_xticklabels([DISPLAY_LABEL.get(ds, ds) for ds in DATASET_ORDER], fontsize=8, rotation=12)
    ax.set_ylabel(r"node sign entropy $H$", fontsize=9)
    ax.set_ylim(-0.03, 1.22)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
    ax.grid(True, which="major", axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e")

    for c, ds in zip(centers, DATASET_ORDER):
        s = stat[ds]
        stars = sig_stars(s["wilcoxon_pvalue"])
        direction = "<" if float(s["mean_H_out"]) < float(s["mean_H_in"]) else ">"
        ax.text(c, 1.21, rf"$H_{{\mathrm{{out}}}} {direction} H_{{\mathrm{{in}}}}$" + f"\n{stars}",
                ha="center", va="top", fontsize=6.5, color="#333")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_OUT, alpha=0.75, edgecolor="#1c4f8f", label=r"$\mathrm{H}_{\mathrm{out}}$ (source, rater consistency)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_IN, alpha=0.75, edgecolor="#a8471f", label=r"$\mathrm{H}_{\mathrm{in}}$ (target, reputation contestedness)"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 1.18), ncol=1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"saved {OUT_PNG}")

    print("\nCaption stats (Wilcoxon signed-rank test, H_out vs. H_in, ties dropped):")
    for ds in DATASET_ORDER:
        s = stat[ds]
        print(f"  {ds}: n={s['n_nodes']} W={float(s['wilcoxon_stat']):.1f} p={float(s['wilcoxon_pvalue']):.2e}")


if __name__ == "__main__":
    main()
