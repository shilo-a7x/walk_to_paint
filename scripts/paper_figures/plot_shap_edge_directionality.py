"""Plot step for the SHAP edge directionality figure. Pure rendering: reads
aaai2027/figure_data/shap_edge_directionality.csv (built by
extract_shap_edge_directionality.py).

Design (agreed with the user 2026-08-10): small multiples, one subplot per dataset
(2x3 grid, same layout as Empirical Confirmation Panel C), x-axis = hop distance
(1, 2 -- the only two points the local attention window covers), y-axis =
mean |SHAP| (probability units), two lines per subplot -- forward (solid, same blue
as Attention Directionality Panel B, #2e75b6) and backward (dashed, same red,
#c0392b) -- so this figure reads as a direct companion to Panel B: same
forward/backward color code, but decomposed by distance and measured by actual causal
contribution (Shapley) instead of raw attention weight. Error bars = cluster-robust SE
(cluster = target edge_id).
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_CSV = os.path.join(ROOT, "aaai2027", "figure_data", "shap_edge_directionality.csv")
OUT_PNG = os.path.join(ROOT, "aaai2027", "figures", "shap_edge_directionality.png")

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
DISPLAY_LABEL = {"slashdot090221": "slashdot"}
FWD_COLOR = "#2e75b6"
BWD_COLOR = "#c0392b"


def main():
    rows = list(csv.DictReader(open(IN_CSV)))
    by_ds = {ds: {} for ds in DATASET_ORDER}
    for r in rows:
        by_ds[r["dataset"]][(r["direction"], int(r["hop"]))] = (
            float(r["mean_abs_shap"]), float(r["cluster_se"])
        )

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), sharex=True)
    hops = [1, 2]

    for ax, ds in zip(axes.flat, DATASET_ORDER):
        d = by_ds[ds]
        fwd_y = [d[("fwd", h)][0] for h in hops]
        fwd_e = [d[("fwd", h)][1] for h in hops]
        bwd_y = [d[("bwd", h)][0] for h in hops]
        bwd_e = [d[("bwd", h)][1] for h in hops]

        ax.errorbar(hops, fwd_y, yerr=fwd_e, color=FWD_COLOR, marker="o", linestyle="-",
                    linewidth=1.8, markersize=5, capsize=3, label="forward", zorder=3)
        ax.errorbar(hops, bwd_y, yerr=bwd_e, color=BWD_COLOR, marker="s", linestyle="--",
                    linewidth=1.8, markersize=5, capsize=3, label="backward", zorder=3)

        ax.set_title(DISPLAY_LABEL.get(ds, ds), fontsize=10)
        ax.set_xticks(hops)
        ax.set_xlim(0.7, 2.3)
        ax.set_ylim(0, max(fwd_y + bwd_y) * 1.35)
        ax.grid(axis="y", alpha=0.25, zorder=0)

    for ax in axes[-1, :]:
        ax.set_xlabel("hop distance", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean |SHAP|", fontsize=9)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("PEWTER (local attention): causal contribution of context edges by hop distance and direction",
                 fontsize=10, y=1.07)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
