"""
Lead 1 (GNN over-averaging) — Step 1: degree-stratified AUC gap.

Hypothesis: if sum-aggregation in a GNN loses neighbor evidence (dilution or
cancellation), the walk-model's AUC advantage over a GNN baseline should widen
as target-node degree grows.

Node-ID alignment: the walk-model's canonical edge loader
(scripts/balance_theory_paths.py load_edges_canonical, raw external SNAP ids,
no remap) and baselines/splits/{dataset}.pt (dense ids = sorted-rank of the
same raw ids -- confirmed against old_chats/baselines.json, which shows the
baselines pipeline was built to consume the canonical pipeline's own edge set,
re-indexed densely for the GNN's embedding table) use DIFFERENT integer
node-id spaces but the SAME underlying graph. `rank = sorted(unique raw ids)`
recovers a 100% edge match (99.0% sign-agreement; the 1.0% mismatch matches
the documented reciprocal-edge sign-conflict resolution rate for
bitcoin-alpha in old_chats/baselines.json). This script remaps canonical ids
through `rank` before computing degree, so both models' degree is computed
from the literal same baseline graph (no more independent-quantile workaround).

Usage:
    python scripts/lead1_degree_gap.py --dataset bitcoin-alpha
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.balance_theory_paths import DATASET_CONFIGS, load_edges_canonical, load_pkl

N_BUCKETS = 4


def node_degree_dict(pairs):
    """pairs: iterable of (u, v). Returns {node: total directed-edge touch count}."""
    deg = {}
    for u, v in pairs:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    return deg


def quantile_buckets(scores, n_buckets=N_BUCKETS):
    edges = np.quantile(scores, np.linspace(0, 1, n_buckets + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return np.digitize(scores, edges[1:-1], right=True)


def canonical_rank_map(ds_name):
    """rank[raw_external_id] = dense baseline id (sorted-rank, matches
    baselines/splits/{ds}.pt's node indexing -- see module docstring)."""
    edges = load_edges_canonical(ds_name)
    all_ids = sorted({n for u, v, s in edges for n in (u, v)})
    return {ext_id: i for i, ext_id in enumerate(all_ids)}, edges


def baseline_degree_dict(ds_name):
    splits_path = os.path.join(ROOT, "baselines", "splits", f"{ds_name}.pt")
    splits = torch.load(splits_path, weights_only=False)
    ei = splits["edge_index"]
    return node_degree_dict(zip(ei[0].tolist(), ei[1].tolist()))


def walk_model_side(ds_name, deg, rank):
    cfg = DATASET_CONFIGS[ds_name if ds_name != "slashdot090221" else "slashdot"]
    _, edges = canonical_rank_map(cfg["ds_name"])

    pkl = load_pkl(cfg, split="test")
    edge_ids = pkl["edge_ids"]
    probs = pkl["probabilities"][:, 1]
    targets = pkl["targets"]

    by_edge = {}
    for eid, p, y in zip(edge_ids.tolist(), probs.tolist(), targets.tolist()):
        if eid < 0:
            continue
        by_edge.setdefault(int(eid), {"p": [], "y": y})
        by_edge[int(eid)]["p"].append(p)

    eids, mean_p, y, node_deg = [], [], [], []
    for eid, rec in by_edge.items():
        u, v, _ = edges[eid]
        ru, rv = rank[u], rank[v]
        eids.append(eid)
        mean_p.append(float(np.mean(rec["p"])))
        y.append(rec["y"])
        node_deg.append(max(deg.get(ru, 0), deg.get(rv, 0)))

    return np.array(mean_p), np.array(y), np.array(node_deg)


def gnn_side(ds_name, deg, art_dir="seed42"):
    art_path = os.path.join(
        ROOT, "baselines", "GINEConv", "results_our_splits", ds_name,
        "GINEConv", art_dir, "best_epoch_artifacts.pkl",
    )
    with open(art_path, "rb") as f:
        art = pickle.load(f)

    eu, ev = art["edge_index"][0], art["edge_index"][1]
    node_deg = np.maximum(
        np.array([deg.get(int(u), 0) for u in eu]),
        np.array([deg.get(int(v), 0) for v in ev]),
    )
    return art["pred_p"], art["y"], node_deg


def auc_by_bucket(p, y, node_deg, n_buckets=N_BUCKETS):
    buckets = quantile_buckets(node_deg, n_buckets)
    rows = []
    for b in range(n_buckets):
        mask = buckets == b
        if mask.sum() < 5 or len(set(np.array(y)[mask])) < 2:
            rows.append((b, mask.sum(), node_deg[mask].min(), node_deg[mask].max(), None))
            continue
        auc = roc_auc_score(np.array(y)[mask], np.array(p)[mask])
        rows.append((b, int(mask.sum()), int(node_deg[mask].min()), int(node_deg[mask].max()), auc))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bitcoin-alpha")
    args = parser.parse_args()

    rank, _ = canonical_rank_map(args.dataset)
    deg = baseline_degree_dict(args.dataset)

    walk_p, walk_y, walk_deg = walk_model_side(args.dataset, deg, rank)
    gnn_p, gnn_y, gnn_deg = gnn_side(args.dataset, deg)

    print(f"\n=== {args.dataset} (degree computed on the shared baseline graph, post id-alignment) ===")
    print(f"walk model: {len(walk_p)} unique test edges, overall AUC = {roc_auc_score(walk_y, walk_p):.4f}")
    print(f"GINEConv  : {len(gnn_p)} test rows (bidirectional), overall AUC = {roc_auc_score(gnn_y, gnn_p):.4f}")

    walk_rows = auc_by_bucket(walk_p, walk_y, walk_deg)
    gnn_rows = auc_by_bucket(gnn_p, gnn_y, gnn_deg)

    print(f"\n{'bucket':>6} {'walk_n':>7} {'walk_deg_range':>16} {'walk_auc':>9}   "
          f"{'gnn_n':>6} {'gnn_deg_range':>15} {'gnn_auc':>8}   {'gap':>7}")
    for (wb, wn, wlo, whi, wauc), (gb, gn, glo, ghi, gauc) in zip(walk_rows, gnn_rows):
        gap = f"{(wauc - gauc):+.4f}" if (wauc is not None and gauc is not None) else "n/a"
        wauc_s = f"{wauc:.4f}" if wauc is not None else "n/a"
        gauc_s = f"{gauc:.4f}" if gauc is not None else "n/a"
        print(f"{wb:>6} {wn:>7} {f'[{wlo},{whi}]':>16} {wauc_s:>9}   "
              f"{gn:>6} {f'[{glo},{ghi}]':>15} {gauc_s:>8}   {gap:>7}")


if __name__ == "__main__":
    main()
