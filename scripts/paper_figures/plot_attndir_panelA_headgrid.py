"""Plot step for the Attention Directionality figure, Panel A -- example per-head
signed attention-mass grid (bitcoin-alpha, layer 0, LocalAttn4). Pure rendering:
reads aaai2027/figure_data/attndir_panelA_headgrid.csv (built by
extract_attndir_panelA_headgrid.py).

2026-08-04 fix vs. the earlier scripts/attention_directionality.py grid plots: those
showed a wider x-range with a dashed red boundary line at +-(window+0.5) -- at a
glance this reads as "the local model has the same reach as full attention but got
truncated," when actually mass is exactly 0 past the window (that's the whole point
of the mask). Per the user's call, this version just sets the x-axis limits to the
window itself -- there is no "beyond the wall" region to show, so nothing is lost by
not drawing it.
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_CSV = "aaai2027/figure_data/attndir_panelA_headgrid.csv"
OUT_PNG = "aaai2027/figures/attndir_panelA_headgrid.png"

NODE_COLOR = "#5b9bd5"
EDGE_COLOR = "#f2a154"


def main():
    rows = list(csv.DictReader(open(IN_CSV)))
    dataset = rows[0]["dataset"]
    layer = int(rows[0]["layer"])
    heads = sorted({int(r["head"]) for r in rows})
    window = max(abs(int(r["d"])) for r in rows)

    by_head = defaultdict(dict)
    for r in rows:
        by_head[int(r["head"])][int(r["d"])] = float(r["mass"])

    nhead = len(heads)
    fig, axes = plt.subplots(1, nhead, figsize=(2.8 * nhead, 3.0), sharey=True, squeeze=False)
    axes = axes[0]

    d_range = list(range(-window, window + 1))
    for h in heads:
        ax = axes[h]
        for d in d_range:
            role_edge = (d % 2 == 0)  # d even (incl. 0) -> same parity as target (odd) -> edge token
            ax.axvspan(d - 0.5, d + 0.5, color=EDGE_COLOR if role_edge else NODE_COLOR,
                       alpha=0.18, lw=0, zorder=0)
        y = [by_head[h][d] for d in d_range]
        ax.plot(d_range, y, color="tab:blue", linewidth=1.6, marker="o", markersize=3, zorder=3)
        ax.fill_between(d_range, y, color="tab:blue", alpha=0.25, zorder=2)
        ax.axvline(0, color="k", linewidth=0.8, alpha=0.6, zorder=2)
        ax.set_xlim(-window - 0.5, window + 0.5)
        ax.set_xticks(d_range)
        ax.set_title(f"head {h}", fontsize=11)
        ax.set_xlabel("d = j − i", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("attention mass", fontsize=10)

    fig.suptitle(f"{dataset}: attention mass per head (orange=edge, blue=vertex offsets)",
                 fontsize=11)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
