"""Plot step for Empirical Confirmation Panel B (MI decay with distance) --
line-graph / endpoint-based distance version.

Pure rendering: reads aaai2027/figure_data/empconf_panelB_mi_decay_linegraph_directed.csv
(directed, out-neighbors-only BFS -- see extract script's build_graph docstring), built
by extract_empconf_panelB_mi_decay_linegraph.py. Edit THIS file freely for scale/color/
style changes -- no recomputation needed.

**Directed traversal only, capped at line-graph distance 6** (settled 2026-08-23, per
explicit user decision) -- reported as the paper's canonical distance metric throughout,
not one of two options. The undirected variant (still available in
empconf_panelB_mi_decay_linegraph.csv, UNDIRECTED_CSV below, unused by this script) was
dropped from the figure entirely, not just backgrounded -- an earlier version of this
script plotted both (directed solid, undirected dashed) for a "does direction matter"
comparison, but showing an alternative that isn't the paper's own reported metric read as
confusing rather than informative. The distance cap (>6 excluded) avoids presenting an
unexplained rebound in NMI at distance 7-8 (e.g. wiki-rfa directed: ~0 at d=6 -> 0.0027 at
d=7 -> 0.027 at d=8) as a settled part of the decay story -- the mechanism behind that
rebound isn't understood yet.

Plots NMI (not raw MI bits): the 6 datasets have different label imbalance,
so raw MI isn't directly comparable across them (H(Y) differs), while NMI =
MI/H(Y) puts every dataset on the same "fraction of the anchor's own
uncertainty explained by context" scale.
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

UNDIRECTED_CSV = "aaai2027/figure_data/empconf_panelB_mi_decay_linegraph.csv"
DIRECTED_CSV = "aaai2027/figure_data/empconf_panelB_mi_decay_linegraph_directed.csv"
OUT_PNG = "aaai2027/figures/empconf_panelB_mi_decay_linegraph.png"

COLORS = [DATASET_COLORS[ds] for ds in DATASET_ORDER]
MARKERS = [DATASET_MARKERS[ds] for ds in DATASET_ORDER]
DISPLAY_LABEL = DATASET_DISPLAY  # legend label only, internal key unchanged

DATASET_RENAME = {"slashdot": "slashdot090221"}  # DATASET_CONFIGS key vs. paper name


MAX_LINE_DIST = 6  # cap here -- beyond this several datasets show an unexplained
# rebound (e.g. wiki-rfa directed: ~0 at d=6 -> 0.0027 at d=7 -> 0.027 at d=8, a
# ~1000x jump) that isn't mechanistically understood yet; reporting only up to
# d=6 avoids presenting that regime as a settled part of the decay story.


def load(path):
    data = {ds: [] for ds in DATASET_ORDER}
    with open(path) as f:
        for row in csv.DictReader(f):
            ds = DATASET_RENAME.get(row["dataset"], row["dataset"])
            if ds not in data:
                continue  # drops synthetic-fog, not one of the 6 paper datasets
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


def print_decay_ratio_summary(data, label):
    """Prints the exact number backing the paper's prose claim: for each dataset,
    NMI(d=1) / min(NMI(d) for d in 2..MAX_LINE_DIST) -- the collapse from distance 1
    to its lowest point within the reported window. min/max across datasets is the
    range quoted in the text (e.g. "$X\\times$--$Y\\times$")."""
    print(f"=== decay ratio summary ({label}, d=1 / min(d=2..{MAX_LINE_DIST})) ===")
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
    directed = load(DIRECTED_CSV)
    print_decay_ratio_summary(directed, "directed, canonical")

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    all_x = set()
    for ds, color, marker in zip(DATASET_ORDER, COLORS, MARKERS):
        pts = directed[ds]
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
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
