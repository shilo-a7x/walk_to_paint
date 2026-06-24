"""
Lead 1 (GNN over-averaging) — Step 4: depth sweep + MAD oversmoothing metric.

MAD (Mean Average Distance, Chen et al. 2020 "Measuring and Relieving the
Over-smoothing Problem for GNNs"): mean pairwise cosine *dissimilarity*
(1 - cosine_similarity) between node embeddings at a given layer. Embedding
collapse (oversmoothing) shows up as MAD -> 0 with depth. Computed on a
random sample of nodes (full O(n^2) is wasteful and unnecessary for a mean
estimate) for tractability on the larger datasets (epinions/slashdot have
80k-130k nodes).

Reads each depth/aggr variant already trained under
results_our_splits/{dataset}/GINEConv/seed42[_aggr{a}_layers{k}]/ and reports
test AUC (from score.csv) alongside final-layer MAD (from
best_epoch_artifacts.pkl's layer_embeddings) side by side.

Usage:
    python scripts/lead1_depth_mad.py --dataset bitcoin-alpha
"""
import argparse
import csv
import os
import pickle
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_N = 2000
SEED = 42


def mad(emb, sample_n=SAMPLE_N, seed=SEED):
    rng = np.random.default_rng(seed)
    n = emb.shape[0]
    idx = rng.choice(n, size=min(sample_n, n), replace=False)
    x = emb[idx]
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    xn = x / norm
    sim = xn @ xn.T
    m = sim.shape[0]
    off_diag = ~np.eye(m, dtype=bool)
    return float((1.0 - sim[off_diag]).mean())


def variant_dirs(ds_name):
    base = os.path.join(ROOT, "baselines", "GINEConv", "results_our_splits", ds_name, "GINEConv")
    out = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            continue
        if name == "seed42":
            out.append((2, "add", name))
            continue
        m = re.match(r"seed42_aggr(\w+)_layers(\d+)", name)
        if m:
            aggr, layers = m.group(1), int(m.group(2))
            out.append((layers, aggr, name))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bitcoin-alpha")
    args = parser.parse_args()

    variants = [(l, a, n) for l, a, n in variant_dirs(args.dataset) if a == "add"]
    variants.sort(key=lambda t: t[0])

    print(f"\n=== {args.dataset} / GINEConv — depth sweep, test AUC + final-layer MAD (sample n={SAMPLE_N}) ===")
    print(f"{'layers':>6} {'tst_auc':>9} {'MAD_final':>10}")
    for layers, aggr, name in variants:
        var_dir = os.path.join(ROOT, "baselines", "GINEConv", "results_our_splits", args.dataset, "GINEConv", name)
        score_path = os.path.join(var_dir, "score.csv")
        art_path = os.path.join(var_dir, "best_epoch_artifacts.pkl")
        if not (os.path.exists(score_path) and os.path.exists(art_path)):
            print(f"{layers:>6}   [missing artifacts in {name}]")
            continue
        with open(score_path) as f:
            row = next(csv.DictReader(f))
        tst_auc = float(row["tst_auc"])
        with open(art_path, "rb") as f:
            art = pickle.load(f)
        final_emb = art["layer_embeddings"][-1]
        m = mad(final_emb)
        print(f"{layers:>6} {tst_auc:>9.4f} {m:>10.4f}")


if __name__ == "__main__":
    main()
