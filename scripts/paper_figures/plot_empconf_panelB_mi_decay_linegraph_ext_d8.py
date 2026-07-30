"""Extended-range (d=8) variant of plot_empconf_panelB_mi_decay_linegraph.py --
separate output, does NOT touch the original plot/PNG since that one is the
citation-anchored figure used in pewter_aaai.tex / the assets checklist.

Pure rendering: reads the same production CSV
(aaai2027/figure_data/empconf_panelB_mi_decay_linegraph.csv), which now has
d=7,8 rows appended (real BFS re-run at --d-max 7, NOT interpolated/
extrapolated) alongside the original d=1-6 rows, left untouched.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_CSV = "aaai2027/figure_data/empconf_panelB_mi_decay_linegraph.csv"
OUT_PNG = "aaai2027/figures/empconf_panelB_mi_decay_linegraph_ext_d8.png"

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
DISPLAY_LABEL = {"slashdot090221": "slashdot"}

DATASET_RENAME = {"slashdot": "slashdot090221"}


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
            data[ds].append((int(row["line_dist"]), float(nmi)))
    for ds in data:
        data[ds].sort()
    return data


def main():
    data = load(IN_CSV)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
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
    ax.text(0.02, 0.02, "extended to d=8 (real BFS re-run, d=1-6 unchanged from the production figure)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, color="#888", style="italic")
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
