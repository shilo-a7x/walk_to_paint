"""
Lead 1 (GNN over-averaging) — Step 2: per-layer sign-decodability probe.

Operationalization note: the plan originally asked for "MI(GNN_embedding@layer
k, sign) vs MI(walk_token@hop k, sign)". A literal high-dim MI estimate on raw
64-dim node embeddings would need a custom kNN/binned estimator not covered by
the project's existing MI utilities (those assume PCA'd low-dim scalar
features). Instead this uses a logistic-regression probe's cross-validated
AUC as the decodability measure for the GNN side -- same question ("how much
sign-relevant info survives in this representation"), standard methodology,
no new estimator to justify.

GNN side: probe AUC on concat([z_u^(k), z_v^(k)]) -> sign, for k = 0 (random
input features, sanity floor), 1, 2 (final layer = same features the trained
discriminator MLP actually uses).

Walk side: doesn't have discrete "layers" of hop-compression the way a GNN
does -- it has Transformer depth (which mixes all positions together, not
hop-by-hop) and an attention *window*. The directly comparable existing
result is full-attention vs LocalAttn4 (restricts to +/-2 hops): walk-model
test AUC barely changes (0.9131 -> 0.9370 on bitcoin-alpha, actually improves
slightly) when context beyond 2 hops is removed, consistent with the
completed MI-collapse finding (signal beyond d=1 is ~zero). Reported here for
side-by-side reference, not recomputed.

Usage:
    python scripts/lead1_layer_probe.py --dataset bitcoin-alpha
"""
import argparse
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def probe_auc(emb, edge_index, y, seed=42):
    u, v = edge_index[0], edge_index[1]
    X = np.concatenate([emb[u], emb[v]], axis=1)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    return scores.mean(), scores.std()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bitcoin-alpha")
    parser.add_argument("--variant", default="seed42",
                         help="subdir under results_our_splits/{dataset}/GINEConv/ holding best_epoch_artifacts.pkl")
    args = parser.parse_args()

    art_path = os.path.join(
        ROOT, "baselines", "GINEConv", "results_our_splits", args.dataset,
        "GINEConv", args.variant, "best_epoch_artifacts.pkl",
    )
    with open(art_path, "rb") as f:
        art = pickle.load(f)

    layer_embeddings = art["layer_embeddings"]
    edge_index = art["edge_index"]
    y = art["y"]

    print(f"\n=== {args.dataset} / GINEConv ({args.variant}) — per-layer probe AUC ===")
    print(f"{len(layer_embeddings)-1}-layer model, {edge_index.shape[1]} test rows\n")
    for k, emb in enumerate(layer_embeddings):
        mean_auc, std_auc = probe_auc(emb, edge_index, y)
        label = "input (random init)" if k == 0 else f"after layer {k}"
        print(f"  k={k:>2} ({label:<22}) probe AUC = {mean_auc:.4f} +/- {std_auc:.4f}")

    print("\n  (final-layer probe AUC vs the model's own trained discriminator AUC "
          "shows how much the nonlinear MLP head recovers beyond a linear probe)")


if __name__ == "__main__":
    main()
