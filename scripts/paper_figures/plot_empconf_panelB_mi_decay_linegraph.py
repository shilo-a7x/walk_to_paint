"""Plot step for Empirical Confirmation Panel B (MI decay with distance) --
line-graph / endpoint-based distance version.

Pure rendering: reads aaai2027/figure_data/empconf_panelB_mi_decay_linegraph.csv
(built by extract_empconf_panelB_mi_decay_linegraph.py). Edit THIS file freely
for scale/color/style changes -- no recomputation needed.

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

IN_CSV = "aaai2027/figure_data/empconf_panelB_mi_decay_linegraph.csv"
OUT_PNG = "aaai2027/figures/empconf_panelB_mi_decay_linegraph.png"

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
DISPLAY_LABEL = {"slashdot090221": "slashdot"}  # shorter legend label only, internal key unchanged

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
    data = load(IN_CSV)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for ds, color, marker in zip(DATASET_ORDER, COLORS, MARKERS):
        pts = data[ds]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color, marker=marker, markersize=5, linewidth=1.6,
                label=DISPLAY_LABEL.get(ds, ds))

    ax.set_xlabel("edge-to-edge distance (line-graph hops)")
    ax.set_ylabel("normalized mutual information (NMI)")
    ax.set_xticks(sorted({p[0] for pts in data.values() for p in pts}))
    ax.grid(True, which="major", axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", axis="y", color="#e1e0d9", linewidth=0.4, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
