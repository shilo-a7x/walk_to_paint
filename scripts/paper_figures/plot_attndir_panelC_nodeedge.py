"""Plot step for the Attention Directionality figure, Panel C -- cross-dataset
node vs. edge attention-mass, LocalAttn4, layer 0. Pure rendering: reads
aaai2027/figure_data/attndir_panelBC_summary.csv (built by
extract_attndir_panelBC_summary.py) for the bar heights and
aaai2027/figure_data/attndir_panelBC_se.csv (built by
attention_directionality_panelB_se.py) for the error bars.

Design (2026-08-04, per the user's call): two grouped positive bars per dataset
(node, edge) side by side, matching Panel B's grouped-bar style, not a signed
node-edge difference bar.

Error bars added 2026-08-18: cluster-robust SE (cluster = target edge_id), 95% CI
(1.96 x SE). Every dataset's node-vs-edge split is significant at this level
(verified separately on the node-minus-edge difference, cluster_robust_se_diff -- see
attention_directionality_panelB_se.py).
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN_CSV = "aaai2027/figure_data/attndir_panelBC_summary.csv"
SE_CSV = "aaai2027/figure_data/attndir_panelBC_se.csv"
OUT_PNG = "aaai2027/figures/attndir_panelC_nodeedge.png"

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
DISPLAY_LABEL = {
    "bitcoin-alpha": "Bitcoin-alpha", "bitcoin-otc": "Bitcoin-otc", "epinions": "Epinions",
    "slashdot090221": "Slashdot", "wiki-elec": "Wiki-elec", "wiki-rfa": "Wiki-RfA",
}
NODE_COLOR = "#5b9bd5"
EDGE_COLOR = "#f2a154"


def main():
    rows = {r["dataset"]: r for r in csv.DictReader(open(IN_CSV))}
    se_rows = {r["dataset"]: r for r in csv.DictReader(open(SE_CSV))}
    layer = rows[DATASET_ORDER[0]]["layer"]

    node = [float(rows[d]["node"]) for d in DATASET_ORDER]
    edge = [float(rows[d]["edge"]) for d in DATASET_ORDER]
    node_err = [1.96 * float(se_rows[d]["node_se"]) for d in DATASET_ORDER]
    edge_err = [1.96 * float(se_rows[d]["edge_se"]) for d in DATASET_ORDER]

    x = np.arange(len(DATASET_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars_n = ax.bar(x - width / 2, node, width, yerr=node_err, capsize=3,
                     color=NODE_COLOR, label="vertex", zorder=3,
                     error_kw={"elinewidth": 1, "zorder": 4})
    bars_e = ax.bar(x + width / 2, edge, width, yerr=edge_err, capsize=3,
                     color=EDGE_COLOR, label="edge", zorder=3,
                     error_kw={"elinewidth": 1, "zorder": 4})

    for bar in list(bars_n) + list(bars_e):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, max(node + edge) * 1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABEL.get(d, d) for d in DATASET_ORDER], fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel("attention mass", fontsize=10)
    ax.set_title("Vertex- vs. edge-token attention mass", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
