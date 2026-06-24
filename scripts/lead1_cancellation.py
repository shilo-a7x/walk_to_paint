"""
Lead 1 (GNN over-averaging) — Step 3b: cancellation-ratio metric.

For each train-graph node i and each GINEConv layer k, computes the
cancellation ratio of incoming messages:

    ratio_i = || sum_j msg_{j->i} || / sum_j || msg_{j->i} ||

(1 = no cancellation, all incoming messages point the same way; -> 0 = heavy
cancellation, messages destructively interfere in the sum). msg_{j->i} =
ReLU(x_j + lin_k(edge_attr_ji)) is GINEConv's own message function
(torch_geometric.nn.conv.GINEConv.message) — recomputed here directly from
the trained checkpoint's lin_k weights and the actual per-layer input
embeddings (saved as `layer_embeddings` in best_epoch_artifacts.pkl), not a
re-implementation guess.

Correlates ratio_i against node in-degree and against sign-heterogeneity
(fraction of minority-sign in-neighbors).

Usage:
    python scripts/lead1_cancellation.py --dataset bitcoin-alpha --variant seed42
"""
import argparse
import os
import pickle

import numpy as np
import torch
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def per_node_messages(x_in, edge_index, edge_attr, lin_weight, lin_bias):
    """x_in: (N, d) layer-input embeddings. Returns per-node (sum_vec, sum_norms, in_deg)."""
    src, dst = edge_index[0], edge_index[1]
    edge_proj = edge_attr @ lin_weight.T + lin_bias  # (E, d)
    msg = np.maximum(x_in[src] + edge_proj, 0.0)  # ReLU, (E, d)

    n = x_in.shape[0]
    d = x_in.shape[1]
    sum_vec = np.zeros((n, d), dtype=np.float64)
    sum_norms = np.zeros(n, dtype=np.float64)
    in_deg = np.zeros(n, dtype=np.int64)
    norms = np.linalg.norm(msg, axis=1)
    np.add.at(sum_vec, dst, msg)
    np.add.at(sum_norms, dst, norms)
    np.add.at(in_deg, dst, 1)
    return sum_vec, sum_norms, in_deg


def sign_heterogeneity(edge_index, edge_attr, n):
    src, dst = edge_index[0], edge_index[1]
    is_pos = (edge_attr.squeeze(-1) > 0)
    n_pos = np.zeros(n, dtype=np.int64)
    n_neg = np.zeros(n, dtype=np.int64)
    np.add.at(n_pos, dst, is_pos.astype(np.int64))
    np.add.at(n_neg, dst, (~is_pos).astype(np.int64))
    total = n_pos + n_neg
    het = np.zeros(n, dtype=np.float64)
    mask = total > 0
    het[mask] = np.minimum(n_pos[mask], n_neg[mask]) / total[mask]
    return het


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bitcoin-alpha")
    parser.add_argument("--variant", default="seed42")
    parser.add_argument("--min-deg", type=int, default=2,
                         help="only include nodes with >= this many incoming edges (cancellation needs >=2 messages)")
    args = parser.parse_args()

    art_path = os.path.join(
        ROOT, "baselines", "GINEConv", "results_our_splits", args.dataset,
        "GINEConv", args.variant, "best_epoch_artifacts.pkl",
    )
    with open(art_path, "rb") as f:
        art = pickle.load(f)

    splits_path = os.path.join(ROOT, "baselines", "splits", f"{args.dataset}.pt")
    splits = torch.load(splits_path, weights_only=False)
    trn_mask = splits["trn_mask"]
    train_edge_index = splits["edge_index"][:, trn_mask].numpy()
    train_edge_attr = splits["edge_weight"][trn_mask].float().unsqueeze(-1).numpy()

    state_dict = art["state_dict"]
    layer_embeddings = art["layer_embeddings"]
    num_layers = len(layer_embeddings) - 1
    n = layer_embeddings[0].shape[0]

    het = sign_heterogeneity(train_edge_index, train_edge_attr, n)

    print(f"\n=== {args.dataset} / GINEConv ({args.variant}) — cancellation ratio, {num_layers} layers ===")
    for k in range(num_layers):
        lin_w = state_dict[f"convs.{k}.lin.weight"].numpy()
        lin_b = state_dict[f"convs.{k}.lin.bias"].numpy()
        x_in = layer_embeddings[k]

        sum_vec, sum_norms, in_deg = per_node_messages(x_in, train_edge_index, train_edge_attr, lin_w, lin_b)

        valid = in_deg >= args.min_deg
        ratio = np.full(n, np.nan)
        nz = sum_norms > 0
        ratio[nz] = np.linalg.norm(sum_vec[nz], axis=1) / sum_norms[nz]

        mask = valid & nz
        r_deg, p_deg = pearsonr(in_deg[mask], ratio[mask])
        r_het, p_het = pearsonr(het[mask], ratio[mask])

        print(f"  layer {k}: n_nodes={mask.sum()}, mean ratio={ratio[mask].mean():.4f}  "
              f"corr(ratio, in_deg)={r_deg:+.4f} (p={p_deg:.2e})  "
              f"corr(ratio, sign_het)={r_het:+.4f} (p={p_het:.2e})")


if __name__ == "__main__":
    main()
