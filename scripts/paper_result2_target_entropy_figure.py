"""One-off figure builder for the PEWTER paper's Result 2 (entropy sensitivity).

Reuses the already-computed per-edge (src_ent, tgt_ent, y, p) records in
outputs/lead4_entropy_heterogeneity/computed_data.pkl (variant "in_in": tgt_ent
is the target vertex's incoming-sign entropy, matching Assumption 1's node-v
entropy) -- no retraining, no new predictions, just a different, simpler
bucketing (1D on tgt_ent, marginalizing over src_ent) than the existing 2D
(source, target) heatmaps, for a compact paper figure.

Read-only: does not modify outputs/lead4_entropy_heterogeneity/.
"""
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

DATA_PATH = "outputs/lead4_entropy_heterogeneity/computed_data.pkl"
OUT_PATH = "aaai2027/figures/result2_entropy_sensitivity.pdf"
VARIANT = "in_in"
DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
N_BUCKETS = 4
MIN_N = 20

WALK_MODEL = "walk_full"
GNN_MODELS = ["GINEConv", "SiGAT"]


def quantile_edges(values, n_buckets):
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_buckets + 1)))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def bucket_auc(tgt_ent, y, p, edges):
    aucs, ns = [], []
    for i in range(len(edges) - 1):
        mask = (tgt_ent > edges[i]) & (tgt_ent <= edges[i + 1])
        if mask.sum() < MIN_N or len(set(y[mask])) < 2:
            aucs.append(np.nan)
            ns.append(int(mask.sum()))
            continue
        aucs.append(roc_auc_score(y[mask], p[mask]))
        ns.append(int(mask.sum()))
    return np.array(aucs), ns


def main():
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharey=False)
    axes = axes.ravel()

    for ax, ds in zip(axes, DATASETS):
        rec_walk = data[ds][VARIANT][WALK_MODEL]
        # bin edges from the pooled tgt_ent of the walk model (same shared edges across models)
        edges = quantile_edges(rec_walk["tgt_ent"], N_BUCKETS)
        nb = len(edges) - 1

        walk_auc, walk_n = bucket_auc(rec_walk["tgt_ent"], rec_walk["y"], rec_walk["p"], edges)
        ax.plot(range(nb), walk_auc, "o-", color="#1b7837", label="\\method\\ (full attn)", linewidth=2, markersize=5)

        best_gnn_auc = None
        for gnn in GNN_MODELS:
            rec = data[ds][VARIANT].get(gnn)
            if rec is None:
                continue
            gnn_auc, gnn_n = bucket_auc(rec["tgt_ent"], rec["y"], rec["p"], edges)
            ax.plot(range(nb), gnn_auc, "s--", alpha=0.55, linewidth=1.3, markersize=4,
                     label=f"{gnn}")
            best_gnn_auc = gnn_auc if best_gnn_auc is None else np.fmax(best_gnn_auc, gnn_auc)

        ax.set_xticks(range(nb))
        ax.set_xticklabels([f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(nb)],
                            fontsize=7, rotation=20)
        ax.set_title(ds, fontsize=10)
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.25)
        if ax in (axes[0], axes[3]):
            ax.set_ylabel("AUC")
        if ax in axes[3:]:
            ax.set_xlabel("target-vertex entropy (bits)", fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Test AUC by target-vertex sign entropy: \\method\\ vs.\\ GNN baselines", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"saved {OUT_PATH}")

    # print numeric summary for the caption / prose
    print("\ndataset, bucket_range, walk_full_auc, best_gnn_auc, walk_minus_gnn")
    for ds in DATASETS:
        rec_walk = data[ds][VARIANT][WALK_MODEL]
        edges = quantile_edges(rec_walk["tgt_ent"], N_BUCKETS)
        walk_auc, _ = bucket_auc(rec_walk["tgt_ent"], rec_walk["y"], rec_walk["p"], edges)
        best_gnn = None
        for gnn in GNN_MODELS:
            rec = data[ds][VARIANT].get(gnn)
            if rec is None:
                continue
            gnn_auc, _ = bucket_auc(rec["tgt_ent"], rec["y"], rec["p"], edges)
            best_gnn = gnn_auc if best_gnn is None else np.fmax(best_gnn, gnn_auc)
        for i in range(N_BUCKETS):
            print(f"{ds}, {edges[i]:.2f}-{edges[i+1]:.2f}, {walk_auc[i]:.4f}, {best_gnn[i]:.4f}, {walk_auc[i]-best_gnn[i]:+.4f}")


if __name__ == "__main__":
    main()
