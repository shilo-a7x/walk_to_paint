"""Diagnostic-only plot of the target_only-rooted Panel B MI-decay run, styled to be a
direct visual match against the production figure (plot_empconf_panelB_mi_decay_linegraph.py)
so the two are easy to compare side by side -- same dataset_style.py colors/markers, same
distance cap (6), same title/label wording, same figure size. The only difference is the
data source and the root_mode note in the corner.

NOT part of the paper's figure pipeline -- writes to outputs/panelB_targetonly_investigation/,
separate from aaai2027/figures/, so this never gets picked up by combine_empconf_panels_abc.py
or cited by mistake. Purpose: check whether root_mode="target_only" (BFS rooted at the target
vertex v alone -- true directed-walk locality, no shortcut through the source vertex u's other
out-edges) changes the shape of the MI-decay curve vs. the production root_mode="both" figure --
see PAPER_CLOSEOUT_LOG.md's "Panel B's post-minimum bump" entry for the open investigation this
is checking, and MAX_LINE_DIST's docstring below for why d=7-8 are excluded here too.

Reads aaai2027/figure_data/empconf_panelB_mi_decay_linegraph_directed_targetonly.csv.
"""
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_style import DATASET_ORDER, DATASET_COLORS, DATASET_MARKERS, DATASET_DISPLAY

IN_CSV = "aaai2027/figure_data/empconf_panelB_mi_decay_linegraph_directed_targetonly.csv"
OUT_PNG = "outputs/panelB_targetonly_investigation/mi_decay_targetonly.png"

COLORS = [DATASET_COLORS[ds] for ds in DATASET_ORDER]
MARKERS = [DATASET_MARKERS[ds] for ds in DATASET_ORDER]
DISPLAY_LABEL = DATASET_DISPLAY

DATASET_RENAME = {"slashdot": "slashdot090221"}

# Same cap as the production script, same reason: beyond distance 6 several datasets show
# an unexplained rebound not mechanistically understood yet (see plot_empconf_panelB_
# mi_decay_linegraph.py's own docstring) -- capping keeps this comparable to that figure
# rather than presenting a different, uncapped view.
MAX_LINE_DIST = 6


def load(path):
    data = {ds: [] for ds in DATASET_ORDER}
    with open(path) as f:
        for row in csv.DictReader(f):
            ds = DATASET_RENAME.get(row["dataset"], row["dataset"])
            if ds not in data:
                continue  # drops synthetic-fog (not one of the 6 paper datasets)
            nmi = row["nmi"]
            if nmi in ("", "nan"):
                continue
            line_dist = int(row["line_dist"])
            if line_dist > MAX_LINE_DIST:
                continue
            data[ds].append((line_dist, float(nmi)))
    for ds in data:
        data[ds].sort()
    return data


def print_decay_ratio_summary(data):
    print(f"=== decay ratio summary (target_only, d=1 / min(d=2..{MAX_LINE_DIST})) ===")
    ratios = {}
    for ds, pts in data.items():
        d = dict(pts)
        if 1 not in d:
            continue
        window = {k: v for k, v in d.items() if 2 <= k <= MAX_LINE_DIST and v > 0}
        if not window:
            continue
        ratios[ds] = d[1] / min(window.values())
    for ds, r in sorted(ratios.items(), key=lambda kv: kv[1]):
        print(f"  {ds:16s} {r:,.0f}x")
    if ratios:
        lo, hi = min(ratios.values()), max(ratios.values())
        print(f"  -> range: {lo:,.0f}x -- {hi:,.0f}x "
              f"(~{math.log10(lo):.1f} to ~{math.log10(hi):.1f} orders of magnitude)")


def main():
    data = load(IN_CSV)
    print_decay_ratio_summary(data)

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    all_x = set()
    for ds, color, marker in zip(DATASET_ORDER, COLORS, MARKERS):
        pts = data[ds]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        all_x.update(xs)
        ax.plot(xs, ys, color=color, marker=marker, markersize=4.5, linewidth=1.6,
                 linestyle="-", zorder=3)

    ax.set_xlabel("Line-graph distance (directed)")
    ax.set_ylabel("Normalized mutual information (NMI)")
    ax.set_title("Sign mutual information vs. line-graph distance", fontsize=11)
    ax.set_xticks(sorted(all_x))
    ax.grid(True, which="major", axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", axis="y", color="#e1e0d9", linewidth=0.4, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")

    dataset_handles = [
        Line2D([0], [0], color=color, marker=marker, markersize=4.5, linewidth=1.6,
               label=DISPLAY_LABEL.get(ds, ds))
        for ds, color, marker in zip(DATASET_ORDER, COLORS, MARKERS)
    ]
    ax.legend(handles=dataset_handles, frameon=False, fontsize=7.5, loc="upper right")
    ax.text(0.02, 0.02, "root_mode=target_only (diagnostic, not a paper figure)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, color="#888", style="italic")
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
