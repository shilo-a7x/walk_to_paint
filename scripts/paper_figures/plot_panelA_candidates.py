"""Panel A CANDIDATE visualizations -- alternatives to the boxplot, for
deciding how to show H_out vs H_in per dataset. Same source data as the
real Panel A (`extract_empconf_panelA_entropy_boxplot.py`'s long-format CSV),
no recomputation.

Output: aaai2027/figures/panelA_candidates/*.png

============================================================================
HOW TO TWEAK -- same convention as plot_panelB_candidates.py.
============================================================================
Everything is in CONFIG below. Key knobs:
  - DATASET_ORDER: shared with the real Panel A script -- keep in sync by hand
    if you reorder one, reorder the other (not auto-linked, they're separate files).
  - COLOR_OUT / COLOR_IN: the two series colors, used identically in all 4 candidates.
  - N_BINS: histogram bin count (candidate A).
  - VIOLIN_POINTS: KDE resolution for the violin (candidate B) -- higher = smoother.
  - Nothing past "# Plumbing below" needs touching for a style change.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
IN_LONG_CSV = "aaai2027/figure_data/empconf_panelA_entropy_boxplot.csv"
OUT_DIR = "aaai2027/figures/panelA_candidates"

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
DISPLAY_LABEL = {"slashdot090221": "slashdot"}
COLOR_OUT = "#2a78d6"
COLOR_IN = "#eb6834"
N_BINS = 25
VIOLIN_POINTS = 200

# ============================================================================
# Plumbing below
# ============================================================================


def load_long(path):
    data = {ds: {"out": [], "in": []} for ds in DATASET_ORDER}
    with open(path) as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            if ds not in data:
                continue
            data[ds][row["direction"]].append(float(row["entropy"]))
    for ds in data:
        for d in ("out", "in"):
            data[ds][d] = np.array(data[ds][d])
    return data


def style_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=7)


def label(ds):
    return DISPLAY_LABEL.get(ds, ds)


def candidate_histogram(data):
    """A: small-multiples histogram, log-y (the zero-spike otherwise swamps
    everything else on a linear count axis)."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
    bins = np.linspace(0, 1, N_BINS + 1)
    for ax, ds in zip(axes.flat, DATASET_ORDER):
        ax.hist(data[ds]["out"], bins=bins, color=COLOR_OUT, alpha=0.6, label=r"$H_{out}$")
        ax.hist(data[ds]["in"], bins=bins, color=COLOR_IN, alpha=0.6, label=r"$H_{in}$")
        ax.set_yscale("log")
        ax.set_title(label(ds), fontsize=9)
        style_axes(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes[-1, :]:
        ax.set_xlabel("entropy", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("# nodes (log)", fontsize=8)
    fig.suptitle("Candidate A: per-dataset histograms (log-count y-axis)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "panelA_candidate_histogram.png")


def candidate_violin(data):
    """B: violin plot -- same x-layout as the real boxplot, but shows full
    density shape instead of just quartiles. CAVEAT: matplotlib's violin is
    a KDE, which is NOT bounded at 0 -- it will smear a little below 0 for
    the zero-heavy datasets (bitcoin-alpha/otc especially). That smear is a
    known artifact of the plotting method, not real negative entropy."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    width = 0.32
    positions_out, positions_in, centers = [], [], []
    for i, ds in enumerate(DATASET_ORDER):
        c = i * 1.0
        centers.append(c)
        positions_out.append(c - width / 2 - 0.02)
        positions_in.append(c + width / 2 + 0.02)

    vp_out = ax.violinplot([data[ds]["out"] for ds in DATASET_ORDER], positions=positions_out,
                           widths=width, points=VIOLIN_POINTS, showmeans=True, showextrema=False)
    vp_in = ax.violinplot([data[ds]["in"] for ds in DATASET_ORDER], positions=positions_in,
                          widths=width, points=VIOLIN_POINTS, showmeans=True, showextrema=False)
    for vp, color in [(vp_out, COLOR_OUT), (vp_in, COLOR_IN)]:
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.7)
        vp["cmeans"].set_color("#333")

    ax.axhline(0, color="#c3c2b7", linewidth=0.8, zorder=0)
    ax.set_xticks(centers)
    ax.set_xticklabels([label(ds) for ds in DATASET_ORDER], fontsize=8, rotation=12)
    ax.set_ylabel("node sign entropy $H$", fontsize=9)
    ax.set_ylim(-0.15, 1.05)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_OUT, alpha=0.7, label=r"$H_{out}$"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_IN, alpha=0.7, label=r"$H_{in}$"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Candidate B: violin plot (KDE shape, not bounded at 0 -- see caveat in script)", fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    _save(fig, "panelA_candidate_violin.png")


def candidate_ecdf(data):
    """C: empirical CDF, small multiples -- the flat run at y=P(H=0) at the
    left edge directly shows the zero-mass fraction as a step height, then
    the curve's climb shows the shape of the remaining spread."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
    for ax, ds in zip(axes.flat, DATASET_ORDER):
        for d, color, lbl in [("out", COLOR_OUT, r"$H_{out}$"), ("in", COLOR_IN, r"$H_{in}$")]:
            x = np.sort(data[ds][d])
            y = np.arange(1, len(x) + 1) / len(x)
            ax.step(x, y, where="post", color=color, linewidth=1.4, label=lbl)
        ax.set_title(label(ds), fontsize=9)
        style_axes(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes[-1, :]:
        ax.set_xlabel("entropy", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("cumulative fraction", fontsize=8)
    fig.suptitle("Candidate C: ECDF (step height at x=0 = fraction of unanimous nodes)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "panelA_candidate_ecdf.png")


def candidate_zero_split(data):
    """D: two panels -- (left) bar chart of the zero-mass fraction itself
    (the thing that makes the boxplot look squashed, shown directly as a
    number instead of implicitly); (right) box/violin of ONLY the nonzero
    entropy values, so the real spread is visible without the zero spike
    drowning it out."""
    fig, (ax_bar, ax_box) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})

    width = 0.32
    x = np.arange(len(DATASET_ORDER))
    pct_zero_out = [100 * (data[ds]["out"] == 0).mean() for ds in DATASET_ORDER]
    pct_zero_in = [100 * (data[ds]["in"] == 0).mean() for ds in DATASET_ORDER]
    ax_bar.bar(x - width / 2, pct_zero_out, width=width, color=COLOR_OUT, alpha=0.8, label=r"$H_{out}=0$")
    ax_bar.bar(x + width / 2, pct_zero_in, width=width, color=COLOR_IN, alpha=0.8, label=r"$H_{in}=0$")
    ax_bar.axhline(75, color="#8a8a80", linewidth=0.8, linestyle="--")
    ax_bar.text(len(DATASET_ORDER) - 0.5, 76, "75th-pctile threshold", fontsize=6.5, color="#666", ha="right")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([label(ds) for ds in DATASET_ORDER], fontsize=8, rotation=20)
    ax_bar.set_ylabel("% nodes with H = 0 (unanimous)", fontsize=8)
    ax_bar.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax_bar.set_title("% zero-entropy (unanimous) nodes", fontsize=9)
    style_axes(ax_bar)

    positions_out, positions_in, centers = [], [], []
    for i, ds in enumerate(DATASET_ORDER):
        c = i * 1.0
        centers.append(c)
        positions_out.append(c - width / 2 - 0.02)
        positions_in.append(c + width / 2 + 0.02)
    nz_out = [data[ds]["out"][data[ds]["out"] > 0] for ds in DATASET_ORDER]
    nz_in = [data[ds]["in"][data[ds]["in"] > 0] for ds in DATASET_ORDER]
    bp_out = ax_box.boxplot(nz_out, positions=positions_out, widths=width, patch_artist=True,
                            showfliers=True, flierprops=dict(marker="o", markersize=2, alpha=0.25, markeredgewidth=0))
    bp_in = ax_box.boxplot(nz_in, positions=positions_in, widths=width, patch_artist=True,
                           showfliers=True, flierprops=dict(marker="o", markersize=2, alpha=0.25, markeredgewidth=0))
    for box in bp_out["boxes"]:
        box.set(facecolor=COLOR_OUT, alpha=0.75, edgecolor="#1c4f8f")
    for box in bp_in["boxes"]:
        box.set(facecolor=COLOR_IN, alpha=0.75, edgecolor="#a8471f")
    for bp, edge in [(bp_out, "#1c4f8f"), (bp_in, "#a8471f")]:
        for part in ["whiskers", "caps", "medians"]:
            for line in bp[part]:
                line.set(color=edge, linewidth=1.1)
    ax_box.set_xticks(centers)
    ax_box.set_xticklabels([label(ds) for ds in DATASET_ORDER], fontsize=8, rotation=20)
    ax_box.set_ylabel("entropy (nonzero nodes only)", fontsize=8)
    ax_box.set_title("Spread among the non-unanimous nodes only", fontsize=9)
    style_axes(ax_box)

    fig.suptitle("Candidate D: zero-mass split out, real spread shown separately", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "panelA_candidate_zero_split.png")


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def main():
    data = load_long(IN_LONG_CSV)
    candidate_histogram(data)
    candidate_violin(data)
    candidate_ecdf(data)
    candidate_zero_split(data)


if __name__ == "__main__":
    main()
