"""CANON-SWAP CANDIDATE for Panel B, root_mode=target_only -- staged so adopting this
version is a one-line change (see "TO ADOPT" below), NOT yet wired into the paper
pipeline. Do not point combine_empconf_panels_abc.py at OUT_PNG below without an
explicit decision -- this changes a headline empirical claim (the MI-decay ratio
quoted in the Abstract/Introduction/Section 6.1), see the accompanying prose draft.

Byte-for-byte the same rendering as the real
plot_empconf_panelB_mi_decay_linegraph.py (same dataset_style.py colors/markers,
same distance-6 cap, same 300dpi, same figure size, no diagnostic corner text) --
except for TWO deliberate content changes, both because root_mode=target_only is
NOT the same distance notion as production's root_mode=both:

1. **Data source**: reads the target_only CSV instead of _directed.csv.
2. **Title/axis label**: "line-graph distance" is the standard graph-theory term for
   shortest-path distance in L(G), where two edges are adjacent iff they share ANY
   endpoint -- that's exactly what root_mode=both computes (BFS rooted at BOTH u and
   v). root_mode=target_only roots ONLY at v, so an edge sharing u (but not v) with
   the anchor is not "distance 1" here even though it would be in the true line
   graph -- this is a genuinely different, asymmetric notion, not standard line-graph
   distance. Relabeled "directed BFS distance (hops)" throughout (title, axis) to
   name the actual computation (a directed multi-source BFS rooted at v) without
   overclaiming it's the standard symmetric line-graph metric.

TO ADOPT (do this only after explicit sign-off, and only after editing the tex prose
per the draft in the accompanying message):
  1. Set OUT_PNG below to "aaai2027/figures/empconf_panelB_mi_decay_linegraph.png"
     (overwriting the production file).
  2. Point combine_empconf_panels_abc.py's PANEL_B at the same path (no change needed,
     already points there).
  3. Update aaai2027/WSDM_format_revised.tex's Panel B prose (line ~242) and caption
     (line ~255) per the draft -- new ratio range, new distance-1 definition, new
     terminology ("forward walk-distance" instead of "line-graph distance").
  4. Re-run the brace-balance/citation-key verification pass as usual after any tex edit.
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
OUT_PNG = "outputs/panelB_targetonly_investigation/mi_decay_targetonly_CANON_CANDIDATE.png"

COLORS = [DATASET_COLORS[ds] for ds in DATASET_ORDER]
MARKERS = [DATASET_MARKERS[ds] for ds in DATASET_ORDER]
DISPLAY_LABEL = DATASET_DISPLAY

DATASET_RENAME = {"slashdot": "slashdot090221"}

MAX_LINE_DIST = 6  # same cap and same reason as production -- see that script's docstring


def load(path):
    data = {ds: [] for ds in DATASET_ORDER}
    with open(path) as f:
        for row in csv.DictReader(f):
            ds = DATASET_RENAME.get(row["dataset"], row["dataset"])
            if ds not in data:
                continue
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
