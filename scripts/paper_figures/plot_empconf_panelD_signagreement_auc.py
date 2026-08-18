"""Plot step for Empirical Confirmation Panel D -- 4-way sign-agreement AUC.
Pure rendering: reads aaai2027/figure_data/empconf_panelD_signagreement_auc.csv and
..._perdataset.csv (both built by extract_empconf_panelD_signagreement_auc.py). Edit
THIS file freely for color/style changes -- no recomputation needed.

Rebuilt 2026-08-18 as a real multiseed (10-seed) result -- the main panel keeps the
original pooled 4-bar layout (SiGAT only, GINEConv dropped 2026-08-04) but now with error
bars (+-1 SD across the 10 seeds) instead of a single-split point estimate.

Second output, `empconf_panelD_signagreement_auc_perdataset.png`: a per-dataset DOT plot
(not bars) for the same 4 buckets, one colored dot per dataset per bucket, dodged
horizontally so all 6 datasets are visible within each bucket group -- per the
professor's C-ter instruction ("IN D please replace the bars by dots for each dataset -
please keep the colors consistent for datasets all along"), reusing the shared
dataset_style palette so dataset colors match every other figure that uses it. This is a
draft/candidate per the plan -- not yet wired into the main combined figure or decided
whether/how it appears in the paper.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_style import DATASET_ORDER, DATASET_COLORS, DATASET_MARKERS, DATASET_DISPLAY

IN_POOLED_CSV = "aaai2027/figure_data/empconf_panelD_signagreement_auc.csv"
IN_PERDATASET_CSV = "aaai2027/figure_data/empconf_panelD_signagreement_auc_perdataset.csv"
OUT_POOLED_PNG = "aaai2027/figures/empconf_panelD_signagreement_auc.png"
OUT_PERDATASET_PNG = "aaai2027/figures/empconf_panelD_signagreement_auc_perdataset.png"

BUCKET_ORDER = ["in_same", "in_diff", "out_same", "out_diff"]
BUCKET_LABEL = {
    "in_same": "in-same", "in_diff": "in-diff",
    "out_same": "out-same", "out_diff": "out-diff",
}


def fmt_n(n):
    n = float(n)
    if n >= 1_000_000:
        return f"n={n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"n={n/1_000:.0f}k"
    return f"n={n:.0f}"


def plot_pooled():
    rows = list(csv.DictReader(open(IN_POOLED_CSV)))
    data = {r["bucket"]: r for r in rows}

    x = np.arange(len(BUCKET_ORDER))
    width = 0.6
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    YMIN = 0.4
    ax.set_ylim(YMIN, 1.05)
    aucs = [float(data[b]["mean_auc"]) for b in BUCKET_ORDER]
    ses = [float(data[b]["std_auc"]) for b in BUCKET_ORDER]
    ns = [float(data[b]["mean_n"]) for b in BUCKET_ORDER]
    # bars are clipped to start at YMIN (bottom=YMIN) since sklearn AUC is >=0.5 here
    # anyway -- bar() with bottom=YMIN avoids drawing off-axis bar bodies
    bars = ax.bar(x, [a - YMIN for a in aucs], width, bottom=YMIN,
                   yerr=ses, capsize=4, color="#c0392b", label="SiGAT", zorder=3)
    for bar, auc, se, n in zip(bars, aucs, ses, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, auc + se + 0.012, f"{auc:.3f}",
                 ha="center", va="bottom", fontsize=10)
        ax.text(bar.get_x() + bar.get_width() / 2, YMIN + 0.012, fmt_n(n),
                 ha="center", va="bottom", fontsize=8, rotation=90, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_LABEL[b] for b in BUCKET_ORDER], fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel("Test AUC", fontsize=10)
    ax.set_title("SiGAT AUC by sign agreement (mean $\\pm$ 1 SD, 10 splits)", fontsize=11)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_POOLED_PNG), exist_ok=True)
    fig.savefig(OUT_POOLED_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_POOLED_PNG}")


def plot_perdataset():
    rows = list(csv.DictReader(open(IN_PERDATASET_CSV)))
    data = {(r["dataset"], r["bucket"]): r for r in rows}

    x = np.arange(len(BUCKET_ORDER))
    n_ds = len(DATASET_ORDER)
    spread = 0.5
    offsets = np.linspace(-spread / 2, spread / 2, n_ds)

    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    for off, ds in zip(offsets, DATASET_ORDER):
        aucs = [float(data[(ds, b)]["mean_auc"]) for b in BUCKET_ORDER]
        ses = [float(data[(ds, b)]["std_auc"]) for b in BUCKET_ORDER]
        ax.errorbar(x + off, aucs, yerr=ses, fmt=DATASET_MARKERS[ds], color=DATASET_COLORS[ds],
                     markersize=8, capsize=3, linewidth=1.4, linestyle="none",
                     label=DATASET_DISPLAY[ds], zorder=3)

    ax.set_ylim(0.4, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_LABEL[b] for b in BUCKET_ORDER], fontsize=15)
    for xc in (x[:-1] + 0.5):
        ax.axvline(xc, color="#e1e0d9", linewidth=0.8, zorder=1)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("Test AUC", fontsize=15)
    ax.set_title("SiGAT AUC by sign agreement, per dataset (mean $\\pm$ 1 SD, 10 splits)", fontsize=16)
    ax.legend(fontsize=12.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.36), frameon=False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PERDATASET_PNG), exist_ok=True)
    fig.savefig(OUT_PERDATASET_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PERDATASET_PNG}")


def main():
    plot_pooled()
    plot_perdataset()


if __name__ == "__main__":
    main()
