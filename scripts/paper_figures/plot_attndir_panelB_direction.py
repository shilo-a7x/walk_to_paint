"""Plot step for the Attention Directionality figure, Panel B -- cross-dataset
forward vs. backward attention-mass, LocalAttn4, layer 0. Pure rendering: reads
aaai2027/figure_data/attndir_panelBC_summary.csv (built by
extract_attndir_panelBC_summary.py).

Design (2026-08-04, per the user's call): two grouped positive bars per dataset
(forward, backward) side by side, not a signed forward-backward difference bar --
raw values are more directly readable even though the wiki-elec/wiki-rfa flip is
slightly less visually dramatic this way.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN_CSV = "aaai2027/figure_data/attndir_panelBC_summary.csv"
OUT_PNG = "aaai2027/figures/attndir_panelB_direction.png"

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
DISPLAY_LABEL = {"slashdot090221": "slashdot"}
FWD_COLOR = "#2e75b6"
BWD_COLOR = "#c0392b"


def main():
    rows = {r["dataset"]: r for r in csv.DictReader(open(IN_CSV))}
    layer = rows[DATASET_ORDER[0]]["layer"]

    fwd = [float(rows[d]["forward"]) for d in DATASET_ORDER]
    bwd = [float(rows[d]["backward"]) for d in DATASET_ORDER]

    x = np.arange(len(DATASET_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars_f = ax.bar(x - width / 2, fwd, width, color=FWD_COLOR, label="forward", zorder=3)
    bars_b = ax.bar(x + width / 2, bwd, width, color=BWD_COLOR, label="backward", zorder=3)

    for bar in list(bars_f) + list(bars_b):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_ylim(0, max(fwd + bwd) * 1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABEL.get(d, d) for d in DATASET_ORDER], fontsize=9)
    ax.set_ylabel("attention mass", fontsize=9)
    ax.set_title(f"PEWTER, layer {layer}: forward vs. backward attention mass by dataset\n"
                 "(mean over heads; self-attention excluded by construction)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
