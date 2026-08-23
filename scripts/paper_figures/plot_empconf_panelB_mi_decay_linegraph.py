"""Plot step for Empirical Confirmation Panel B (MI decay with distance) --
line-graph / endpoint-based distance version.

Pure rendering: reads aaai2027/figure_data/empconf_panelB_mi_decay_linegraph_directed_targetonly.csv
(directed BFS, rooted at the anchor edge's own TARGET vertex only -- root_mode=target_only,
true forward-walk locality, no shortcut through the source vertex's other out-edges -- see
extract_empconf_panelB_mi_decay_linegraph.py's build_graph/root_mode docstrings), built by
that same extract script. Edit THIS file freely for scale/color/style changes -- no
recomputation needed.

**root_mode=target_only, directed traversal, capped at distance 6 (settled 2026-08-23,
canon swap from the earlier root_mode=both version, per explicit user decision).** The
root_mode=both variant (BFS rooted at BOTH of the anchor edge's endpoints -- the true,
symmetric line-graph adjacency) is still on disk, unused by this script:
empconf_panelB_mi_decay_linegraph_directed.csv (BOTH_CSV below) and its own rendered PNG,
empconf_panelB_mi_decay_linegraph_ext_d8.png, both kept for provenance/comparison, not
deleted. Swap rationale: target_only matches what a sampled walk actually exposes the model
to (a walk only continues forward past the target vertex, never backtracks through the
source's other edges), which is the more mechanistically relevant notion for this paper's
own local-attention-window argument -- see MECHANISM.md-style investigation notes in
outputs/panelB_targetonly_investigation/. Because target_only is NOT the standard symmetric
line-graph distance (an edge sharing only the source vertex with the anchor is not "distance
1" here), axis/title wording says "directed BFS distance" rather than "line-graph distance"
-- do not call this "line-graph distance" in prose/captions either, see the 2026-08-23 tex
edit for the corrected wording.

The distance cap (>6 excluded) avoids presenting an unexplained rebound in NMI at distance
7-8 (present in both root_mode variants, and in the synthetic-fog null-control graph too --
consistent with the existing estimator-artifact explanation, see PAPER_CLOSEOUT_LOG.md's
"Panel B's post-minimum bump" entry) as a settled part of the decay story -- the mechanism
behind that rebound isn't understood yet.

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

BOTH_CSV = "aaai2027/figure_data/empconf_panelB_mi_decay_linegraph_directed.csv"  # superseded, kept for provenance
TARGETONLY_CSV = "aaai2027/figure_data/empconf_panelB_mi_decay_linegraph_directed_targetonly.csv"  # canonical
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
    data = load(TARGETONLY_CSV)
    print_decay_ratio_summary(data, "target_only, canonical")

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

    ax.set_xlabel("Directed BFS distance (hops)")
    ax.set_ylabel("Normalized mutual information (NMI)")
    ax.set_title("Sign mutual information vs. directed BFS distance", fontsize=11)
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
