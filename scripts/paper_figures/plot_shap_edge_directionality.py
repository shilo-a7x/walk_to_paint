"""Plot step for the SHAP edge directionality figure -- now Panel D of the
Attention Directionality figure (2026-08-11, per the user's call). Pure
rendering: reads aaai2027/figure_data/shap_edge_directionality.csv (built by
extract_shap_edge_directionality.py).

Design (revised 2026-08-11): single compact grouped bar chart, all 6 datasets
in one plot (was a 2x3 grid of line plots -- dropped per the user's call,
"why are there lines anyway", since hop only takes two values and a line
implies a continuous axis). Same forward/backward color code as Panel B
(#2e75b6 / #c0392b); hop 1 is solid, hop 2 is the same color with a hatch, so
all 4 series (fwd/bwd x hop1/hop2) fit in one legend without adding new
colors. Error bars = cluster-robust SE (cluster = target edge_id).
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_CSV = os.path.join(ROOT, "aaai2027", "figure_data", "shap_edge_directionality.csv")
OUT_PNG = os.path.join(ROOT, "aaai2027", "figures", "shap_edge_directionality.png")

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
DISPLAY_LABEL = {
    "bitcoin-alpha": "Bitcoin-alpha", "bitcoin-otc": "Bitcoin-otc", "epinions": "Epinions",
    "slashdot090221": "Slashdot", "wiki-elec": "Wiki-elec", "wiki-rfa": "Wiki-RfA",
}
FWD_COLOR = "#2e75b6"
BWD_COLOR = "#c0392b"
HATCH = "//"


def main():
    rows = list(csv.DictReader(open(IN_CSV)))
    by_ds = {ds: {} for ds in DATASET_ORDER}
    for r in rows:
        by_ds[r["dataset"]][(r["direction"], int(r["hop"]))] = (
            float(r["mean_abs_shap"]), float(r["cluster_se"])
        )

    series = [
        ("fwd", 1, FWD_COLOR, None, "forward, hop 1"),
        ("fwd", 2, FWD_COLOR, HATCH, "forward, hop 2"),
        ("bwd", 1, BWD_COLOR, None, "backward, hop 1"),
        ("bwd", 2, BWD_COLOR, HATCH, "backward, hop 2"),
    ]

    x = np.arange(len(DATASET_ORDER))
    width = 0.19
    offsets = width * np.array([-1.5, -0.5, 0.5, 1.5])

    fig, ax = plt.subplots(figsize=(9.2, 4.3))

    for (direction, hop, color, hatch, label), off in zip(series, offsets):
        y = [by_ds[ds][(direction, hop)][0] for ds in DATASET_ORDER]
        e = [by_ds[ds][(direction, hop)][1] for ds in DATASET_ORDER]
        ax.bar(x + off, y, width, yerr=e, color=color, hatch=hatch, alpha=0.9 if hatch is None else 0.55,
               edgecolor=color, linewidth=0.8, capsize=2, zorder=3,
               error_kw={"linewidth": 0.8, "ecolor": "#333333"})

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABEL.get(d, d) for d in DATASET_ORDER], fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("Mean |SHAP| (probability units)", fontsize=12)
    ax.set_title("Shapley contribution of context edges by hop distance and direction", fontsize=13)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    legend_handles = [
        Patch(facecolor=FWD_COLOR, alpha=0.9, edgecolor=FWD_COLOR, label="Forward, hop 1"),
        Patch(facecolor=FWD_COLOR, alpha=0.55, hatch=HATCH, edgecolor=FWD_COLOR, label="Forward, hop 2"),
        Patch(facecolor=BWD_COLOR, alpha=0.9, edgecolor=BWD_COLOR, label="Backward, hop 1"),
        Patch(facecolor=BWD_COLOR, alpha=0.55, hatch=HATCH, edgecolor=BWD_COLOR, label="Backward, hop 2"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10.5, ncol=2, frameon=True)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
