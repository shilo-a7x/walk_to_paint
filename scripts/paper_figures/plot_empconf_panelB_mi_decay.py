"""Plot step for Empirical Confirmation Panel B (MI decay with distance).

Pure rendering: reads aaai2027/figure_data/empconf_panelB_mi_decay.csv (built by
extract_empconf_panelB_mi_decay.py) and draws the chart. No data loading beyond
the CSV, no computation -- edit THIS file freely for scale/color/style changes.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

IN_CSV = "aaai2027/figure_data/empconf_panelB_mi_decay.csv"
OUT_PNG = "aaai2027/figures/empconf_panelB_mi_decay.png"
MAX_D = 6  # truncate: report flags d>=7 as sampling noise (too few pairs)

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def load(path):
    data = {ds: [] for ds in DATASET_ORDER}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["mi_bits"] == "nan":
                continue
            d = int(row["d"])
            if d > MAX_D:
                continue
            data[row["dataset"]].append((d, float(row["mi_bits"])))
    for ds in data:
        data[ds].sort()
    return data


def main():
    data = load(IN_CSV)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for ds, color, marker in zip(DATASET_ORDER, COLORS, MARKERS):
        pts = data[ds]
        xs = [p[0] for p in pts]
        ys = [max(p[1], 1e-6) for p in pts]
        ax.plot(xs, ys, color=color, marker=marker, markersize=5, linewidth=1.6, label=ds)

    ax.set_yscale("log")
    ax.set_xlabel("directed BFS distance $d$ between edges")
    ax.set_ylabel("mutual information (bits)")
    ax.set_xticks(range(0, MAX_D + 1))
    ax.grid(True, which="major", axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", axis="y", color="#e1e0d9", linewidth=0.4, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
