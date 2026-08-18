"""Candidate visualizations for Panel D's per-dataset sign-agreement AUC breakdown --
same "generate several options, let the user pick" pattern as
aaai2027/figures/panelA_candidates/. Pure rendering: reads
aaai2027/figure_data/empconf_panelD_signagreement_auc_perdataset.csv (already computed by
extract_empconf_panelD_signagreement_auc.py -- cheap to rerun, no SiGAT refit needed).
Output: aaai2027/figures/panelD_candidates/panelD_candidate_<name>.png. None of these are
wired into the paper yet -- draft/candidate only, per the user's explicit request to see
options before deciding.

Four candidates:
1. dots       -- one dodged, colored dot per dataset within each bucket group (the
                 professor's literal "replace the bars by dots" ask), +-1 SD whiskers.
2. groupedbar -- the same data as 6 side-by-side thin bars per bucket group instead of
                 dots (a more traditional look, same information).
3. boxstrip   -- per bucket, a boxplot summarizing the spread of the 6 datasets' mean
                 AUCs (quartiles/median across datasets), with each dataset's own point
                 overlaid as a colored dot -- a "box with six middles" reading.
4. heatmap    -- dataset (rows) x bucket (columns) matrix, AUC as both color and
                 annotated text, same visual language as Panel C's entropy heatmap
                 already in the paper.
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

IN_CSV = "aaai2027/figure_data/empconf_panelD_signagreement_auc_perdataset.csv"
OUT_DIR = "aaai2027/figures/panelD_candidates"

BUCKET_ORDER = ["in_same", "in_diff", "out_same", "out_diff"]
BUCKET_LABEL = {"in_same": "in-same", "in_diff": "in-diff", "out_same": "out-same", "out_diff": "out-diff"}


def load():
    rows = list(csv.DictReader(open(IN_CSV)))
    return {(r["dataset"], r["bucket"]): r for r in rows}


def candidate_dots(data):
    x = np.arange(len(BUCKET_ORDER))
    n_ds = len(DATASET_ORDER)
    offsets = np.linspace(-0.25, 0.25, n_ds)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for off, ds in zip(offsets, DATASET_ORDER):
        aucs = [float(data[(ds, b)]["mean_auc"]) for b in BUCKET_ORDER]
        ses = [float(data[(ds, b)]["std_auc"]) for b in BUCKET_ORDER]
        ax.errorbar(x + off, aucs, yerr=ses, fmt=DATASET_MARKERS[ds], color=DATASET_COLORS[ds],
                     markersize=6, capsize=3, linewidth=1.2, linestyle="none",
                     label=DATASET_DISPLAY[ds], zorder=3)
    ax.set_ylim(0.4, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_LABEL[b] for b in BUCKET_ORDER], fontsize=10)
    for xc in (x[:-1] + 0.5):
        ax.axvline(xc, color="#e1e0d9", linewidth=0.8, zorder=1)
    ax.set_ylabel("Test AUC", fontsize=10)
    ax.set_title("1. Dots per dataset", fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    fig.tight_layout()
    return fig


def candidate_groupedbar(data):
    x = np.arange(len(BUCKET_ORDER))
    n_ds = len(DATASET_ORDER)
    width = 0.8 / n_ds
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n_ds)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.set_ylim(0.4, 1.05)
    for off, ds in zip(offsets, DATASET_ORDER):
        aucs = [float(data[(ds, b)]["mean_auc"]) for b in BUCKET_ORDER]
        ses = [float(data[(ds, b)]["std_auc"]) for b in BUCKET_ORDER]
        ax.bar(x + off, [a - 0.4 for a in aucs], width, bottom=0.4, yerr=ses, capsize=2,
               color=DATASET_COLORS[ds], label=DATASET_DISPLAY[ds], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_LABEL[b] for b in BUCKET_ORDER], fontsize=10)
    ax.set_ylabel("Test AUC", fontsize=10)
    ax.set_title("2. Grouped bars per dataset", fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    fig.tight_layout()
    return fig


def candidate_boxstrip(data):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.set_ylim(0.4, 1.05)
    box_data = []
    for b in BUCKET_ORDER:
        box_data.append([float(data[(ds, b)]["mean_auc"]) for ds in DATASET_ORDER])

    bp = ax.boxplot(box_data, positions=np.arange(len(BUCKET_ORDER)), widths=0.5,
                     showfliers=False, patch_artist=True, zorder=2)
    for patch in bp["boxes"]:
        patch.set_facecolor("#eeeeee")
        patch.set_edgecolor("#888888")
    for med in bp["medians"]:
        med.set_color("#444444")

    rng = np.random.default_rng(0)
    for i, b in enumerate(BUCKET_ORDER):
        for ds in DATASET_ORDER:
            jitter = rng.uniform(-0.12, 0.12)
            ax.scatter(i + jitter, float(data[(ds, b)]["mean_auc"]), color=DATASET_COLORS[ds],
                       marker=DATASET_MARKERS[ds], s=40, zorder=3,
                       label=DATASET_DISPLAY[ds] if i == 0 else None)

    ax.set_xticks(np.arange(len(BUCKET_ORDER)))
    ax.set_xticklabels([BUCKET_LABEL[b] for b in BUCKET_ORDER], fontsize=10)
    ax.set_ylabel("Test AUC", fontsize=10)
    ax.set_title("3. Box (across datasets) + dataset dots", fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    fig.tight_layout()
    return fig


def candidate_heatmap(data):
    grid = np.array([[float(data[(ds, b)]["mean_auc"]) for b in BUCKET_ORDER] for ds in DATASET_ORDER])
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(len(DATASET_ORDER)):
        for j in range(len(BUCKET_ORDER)):
            v = grid[i, j]
            color = "black" if 0.62 < v < 0.92 else "white"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10, color=color)
    ax.set_xticks(np.arange(len(BUCKET_ORDER)))
    ax.set_xticklabels([BUCKET_LABEL[b] for b in BUCKET_ORDER], fontsize=10)
    ax.set_yticks(np.arange(len(DATASET_ORDER)))
    ax.set_yticklabels([DATASET_DISPLAY[ds] for ds in DATASET_ORDER], fontsize=10)
    ax.set_title("4. Dataset x bucket heatmap", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Test AUC")
    fig.tight_layout()
    return fig


def main():
    data = load()
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, fn in [("dots", candidate_dots), ("groupedbar", candidate_groupedbar),
                      ("boxstrip", candidate_boxstrip), ("heatmap", candidate_heatmap)]:
        fig = fn(data)
        out_path = os.path.join(OUT_DIR, f"panelD_candidate_{name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
