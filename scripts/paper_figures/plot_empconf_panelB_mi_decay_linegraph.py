"""Plot step for Empirical Confirmation Panel B (MI decay with distance) --
line-graph / endpoint-based distance version.

Pure rendering: reads BOTH
aaai2027/figure_data/empconf_panelB_mi_decay_linegraph.csv (undirected, production
default, unchanged) and
aaai2027/figure_data/empconf_panelB_mi_decay_linegraph_directed.csv (directed,
out-neighbors-only BFS -- see extract script's build_graph docstring), both built by
extract_empconf_panelB_mi_decay_linegraph.py. Edit THIS file freely for
scale/color/style changes -- no recomputation needed.

Per the user's call: BOTH directions are shown on ONE plot, one line pair per
dataset -- directed SOLID, undirected DASHED, same color per dataset across both
linestyles, so the "how much does direction matter" comparison is visible directly.

Plots NMI (not raw MI bits): the 6 datasets have different label imbalance,
so raw MI isn't directly comparable across them (H(Y) differs), while NMI =
MI/H(Y) puts every dataset on the same "fraction of the anchor's own
uncertainty explained by context" scale.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

UNDIRECTED_CSV = "aaai2027/figure_data/empconf_panelB_mi_decay_linegraph.csv"
DIRECTED_CSV = "aaai2027/figure_data/empconf_panelB_mi_decay_linegraph_directed.csv"
OUT_PNG = "aaai2027/figures/empconf_panelB_mi_decay_linegraph.png"

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
DISPLAY_LABEL = {
    "bitcoin-alpha": "Bitcoin-alpha", "bitcoin-otc": "Bitcoin-otc", "epinions": "Epinions",
    "slashdot090221": "Slashdot", "wiki-elec": "Wiki-elec", "wiki-rfa": "Wiki-RfA",
}  # legend label only, internal key unchanged

DATASET_RENAME = {"slashdot": "slashdot090221"}  # DATASET_CONFIGS key vs. paper name


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
            data[ds].append((int(row["line_dist"]), float(nmi)))
    for ds in data:
        data[ds].sort()
    return data


def main():
    undirected = load(UNDIRECTED_CSV)
    directed = load(DIRECTED_CSV)

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    all_x = set()
    for ds, color, marker in zip(DATASET_ORDER, COLORS, MARKERS):
        for data, style, alpha in ((directed, "-", 1.0), (undirected, "--", 0.75)):
            pts = data[ds]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            all_x.update(xs)
            ax.plot(xs, ys, color=color, marker=marker, markersize=4.5, linewidth=1.6,
                     linestyle=style, alpha=alpha, zorder=3)

    ax.set_xlabel("edge-to-edge distance (line-graph hops)")
    ax.set_ylabel("normalized mutual information (NMI)")
    ax.set_xticks(sorted(all_x))
    ax.grid(True, which="major", axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", axis="y", color="#e1e0d9", linewidth=0.4, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")

    # two-part legend: dataset (color) and direction (linestyle), kept separate so
    # neither explodes into 12 entries.
    dataset_handles = [
        Line2D([0], [0], color=color, marker=marker, markersize=4.5, linewidth=1.6,
               label=DISPLAY_LABEL.get(ds, ds))
        for ds, color, marker in zip(DATASET_ORDER, COLORS, MARKERS)
    ]
    direction_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.6, label="directed"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.6, alpha=0.75, label="undirected"),
    ]
    leg1 = ax.legend(handles=dataset_handles, frameon=False, fontsize=7.5, loc="upper right")
    ax.add_artist(leg1)
    ax.legend(handles=direction_handles, frameon=False, fontsize=7.5, loc="upper right",
              bbox_to_anchor=(1.0, 0.62))
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
