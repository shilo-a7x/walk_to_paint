"""
Balance-theory path analysis at graph depths d=1..5.

For each test edge (u, v) with label Y ∈ {-1, +1}, find ALL directed paths
of exactly length d from u to v (excluding the trivial direct edge u→v and
the trivial 2-cycle via v→u).  Compute the sign product along each path
(balance-theory prediction):  P = ∏ s(w_i → w_{i+1}).

Aggregate per edge: path_balance_score = mean P over all d-paths.

Measure:
  - Coverage:    fraction of test edges with ≥1 path at this depth
  - Pearson r:   correlation(path_balance_score, Y)
  - AUC:         AUC of |path_balance_score| predicting |Y|
  - Sign acc:    fraction where sign(path_balance_score) == sign(Y)
    (only on edges with ≥1 path AND excluding Y-sign ties)

Also run with "de-bidir" mode: remove reciprocal edges (v→u direction for
each (u,v)) from the graph before BFS, to neutralise the bidirectional-edge
confound.

This is model-independent: purely a graph-structural measurement.

Usage
─────
  python scripts/balance_theory_paths.py [--datasets all] [--out outputs/balance_theory]
  python scripts/balance_theory_paths.py --datasets bitcoin-alpha epinions
"""

import os, sys, glob, pickle, argparse, math
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from omegaconf import OmegaConf
from src.utils.config import load_config
from src.data.datasets import get_loader

D_MAX       = 5      # max path length to search
MAX_PATHS   = 200    # max paths to collect per (edge, depth) before stopping
MAX_EDGES   = 10000  # cap on test edges for speed
DFS_BOUND   = 2000   # max DFS nodes expanded per (edge, depth)

# ── Dataset configs ───────────────────────────────────────────────────────────
# ds_name   = canonical name as understood by load_config / get_loader
# exp_dir   = relative path under outputs/transformer_incremental
# best_epoch = epoch index for PKL loading
DATASET_CONFIGS = {
    "bitcoin-alpha": {
        "ds_name":    "bitcoin-alpha",
        "exp_dir":    "outputs/transformer_incremental/bitcoin-alpha_seed42_nw5000000_mw80_bs1024_ep75_20260423-111407",
        "best_epoch": 24,
    },
    "bitcoin-otc": {
        "ds_name":    "bitcoin-otc",
        "exp_dir":    "outputs/transformer_incremental/bitcoin-otc_seed42_nw500000_mw80_bs1024_ep50_20260419-133621",
        "best_epoch": 60,
    },
    "epinions": {
        "ds_name":    "epinions",
        "exp_dir":    "outputs/transformer_incremental/epinions_seed42_nw500000_mw80_bs1024_ep50_20260416-125419",
        "best_epoch": 42,
    },
    "wiki-elec": {
        "ds_name":    "wiki-elec",
        "exp_dir":    "outputs/transformer_incremental/wiki-elec_seed42_nw500000_mw80_bs1024_ep50_20260419-133621",
        "best_epoch": 36,
    },
    "wiki-rfa": {
        "ds_name":    "wiki-rfa",
        "exp_dir":    "outputs/transformer_incremental/wiki-rfa_seed42_nw500000_mw80_bs1024_ep50_20260324-141920",
        "best_epoch": 25,
    },
    "slashdot": {
        "ds_name":    "slashdot090221",
        "exp_dir":    "outputs/transformer_incremental/slashdot090221_seed42_nw5000000_mw80_bs1024_ep50_20260420-134149",
        "best_epoch": 20,
    },
    # Lead 2 Step 4: synthetic "inverted fog" graph (no trained walk-model
    # checkpoint yet -- exp_dir/best_epoch are placeholders, unused by
    # edge_sign_mi_vs_distance_v3.py's MI-only calibration check).
    "synthetic-fog": {
        "ds_name":    "synthetic-fog",
        "exp_dir":    None,
        "best_epoch": None,
    },
}

# ── Canonical edge loader (identical to training pipeline) ────────────────────

_edge_cache: dict = {}

def load_edges_canonical(ds_name: str) -> list:
    """Load edges exactly as done during training: uses load_config + get_loader,
    honouring remove_self_loops and multiedge_handling from the dataset config."""
    if ds_name in _edge_cache:
        return _edge_cache[ds_name]
    orig_dir = os.getcwd()
    os.chdir(ROOT)  # load_config resolves paths relative to CWD
    try:
        cfg = load_config(overrides=[f"dataset.name={ds_name}"])
        edges = get_loader(ds_name)(cfg)
    finally:
        os.chdir(orig_dir)
    _edge_cache[ds_name] = edges
    return edges



# ── PKL loader ────────────────────────────────────────────────────────────────

def load_pkl(cfg, split="test"):
    epoch = cfg["best_epoch"]; ds = cfg["ds_name"]
    exp_dir = os.path.join(ROOT, cfg["exp_dir"])
    pattern = os.path.join(exp_dir, "runs", ds, "E14_HARDNODE_L10",
                           "checkpoints", f"{ds}_predictions",
                           f"epoch_{epoch:03d}", f"{split}_predictions.pkl")
    candidates = glob.glob(pattern)
    if not candidates:
        candidates = glob.glob(os.path.join(exp_dir, "**", f"{split}_predictions.pkl"), recursive=True)
        candidates = [c for c in candidates if f"epoch_{epoch:03d}" in c]
    if not candidates:
        return None
    with open(candidates[0], "rb") as f:
        return pickle.load(f)

# ── Path search: DFS up to depth d from src to tgt ───────────────────────────

def find_paths(src, tgt, adj, depth, exclude_direct=True):
    """Return list of sign-products for paths of exactly `depth` steps from src to tgt.

    - Never uses the edge src→tgt directly (exclude_direct=True)
    - Never revisits a node in the same path
    - Stops early when MAX_PATHS collected or DFS_BOUND nodes expanded
    """
    if depth < 1:
        return []

    products = []
    # Stack items: (current_node, current_sign_product, visited_set)
    stack = [(src, 1, frozenset([src]))]
    expanded = 0

    while stack and len(products) < MAX_PATHS and expanded < DFS_BOUND:
        node, prod, visited = stack.pop()
        expanded += 1
        remaining = depth - (len(visited) - 1)

        for nbr, s in adj.get(node, []):
            if nbr in visited:
                continue
            new_prod = prod * s
            new_remaining = remaining - 1

            # Skip the direct edge src→tgt at depth=1 (trivial)
            if exclude_direct and node == src and nbr == tgt and depth == 1:
                continue

            if new_remaining == 0:
                if nbr == tgt:
                    products.append(new_prod)
                    if len(products) >= MAX_PATHS:
                        break
            else:
                if new_remaining > 0:
                    stack.append((nbr, new_prod, visited | {nbr}))

    return products

# ── De-bidirectionalize ───────────────────────────────────────────────────────

def remove_reciprocal_edges(adj, edges):
    """Build a new adj dict with reversed edges removed.

    For every edge (u, v) in edges, remove v→u from the adjacency list
    if it exists.  This eliminates reciprocal pairs so BFS cannot trivially
    bounce u→v→u.
    """
    edge_set = {(u, v) for u, v, _ in edges}
    new_adj = {}
    for node, nbrs in adj.items():
        filtered = [(nbr, s) for nbr, s in nbrs if (nbr, node) not in edge_set]
        new_adj[node] = filtered
    return new_adj

# ── Metrics ───────────────────────────────────────────────────────────────────

def pearson_r(x, y):
    m = ~np.isnan(x)
    if m.sum() < 10: return float("nan")
    xm, ym = x[m], y[m].astype(float)
    if xm.std() < 1e-12 or ym.std() < 1e-12: return 0.0
    return float(np.corrcoef(xm, ym)[0, 1])

def sign_accuracy(scores, labels):
    """Among edges with a path AND non-zero score: fraction where sign matches."""
    m = ~np.isnan(scores) & (scores != 0)
    if m.sum() < 5: return float("nan")
    return float(np.mean(np.sign(scores[m]) == np.sign(labels[m].astype(float))))

def auc_score(scores, labels):
    m = ~np.isnan(scores)
    if m.sum() < 10 or len(np.unique(labels[m])) < 2: return float("nan")
    try:
        a = roc_auc_score((labels[m] > 0).astype(int), scores[m])
        return max(a, 1 - a)
    except: return float("nan")

# ── Per-dataset analysis ──────────────────────────────────────────────────────

def analyse_dataset(ds_name, cfg, out_dir):
    print(f"\n{'─'*60}\n  {ds_name}\n{'─'*60}")

    # Load edges via canonical training pipeline
    edges = load_edges_canonical(cfg["ds_name"])
    print(f"  Edges: {len(edges)}")

    # Build adjacency
    adj = defaultdict(list)
    for u, v, s in edges:
        adj[u].append((v, s))

    # Load PKL for test edge indices + labels
    pkl = load_pkl(cfg)
    if pkl is None:
        print("  ✗ PKL not found")
        return None

    edge_ids  = pkl["edge_ids"]
    targets   = pkl["targets"]
    edge_label = {}
    for eid, tgt in zip(edge_ids.tolist(), targets.tolist()):
        if eid >= 0:
            edge_label[int(eid)] = int(tgt)

    test_edges_uvy = []
    for eid, lbl in edge_label.items():
        if 0 <= eid < len(edges):
            u, v, _ = edges[eid]
            test_edges_uvy.append((u, v, lbl))

    if len(test_edges_uvy) > MAX_EDGES:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(test_edges_uvy), MAX_EDGES, replace=False)
        test_edges_uvy = [test_edges_uvy[i] for i in idx]

    print(f"  Test edges: {len(test_edges_uvy)}")

    # Bidirectionality stats
    edge_set = {(u, v): s for u, v, s in edges}
    n_bidir = sum(1 for u, v, s in edges if (v, u) in edge_set)
    n_agree = sum(1 for u, v, s in edges if (v, u) in edge_set and edge_set[(v, u)] == s)
    print(f"  Bidirectional pairs: {n_bidir//2} ({100*n_bidir/len(edges):.1f}%), "
          f"agree: {n_agree//2}, disagree: {(n_bidir-n_agree)//2}")

    # Build de-bidirectionalized adj
    adj_debidir = remove_reciprocal_edges(dict(adj), edges)

    results = {}
    for mode, cur_adj in [("full", adj), ("debidir", adj_debidir)]:
        res_mode = {}
        for d in range(1, D_MAX + 1):
            scores = []
            labels = []
            n_with_path = 0
            for u, v, lbl in test_edges_uvy:
                paths = find_paths(u, v, cur_adj, d)
                if paths:
                    n_with_path += 1
                    scores.append(float(np.mean(paths)))
                else:
                    scores.append(float("nan"))
                labels.append(lbl)

            scores_np = np.array(scores)
            labels_np = np.array(labels)
            coverage  = n_with_path / len(test_edges_uvy)

            res_mode[d] = {
                "coverage":  coverage,
                "r":         pearson_r(scores_np, labels_np),
                "sign_acc":  sign_accuracy(scores_np, labels_np),
                "auc":       auc_score(scores_np, labels_np),
                "n_with_path": n_with_path,
            }
            print(f"  [{mode}] d={d}: cov={coverage:.3f}  r={res_mode[d]['r']:+.4f}"
                  f"  sign_acc={res_mode[d]['sign_acc']:.4f}  auc={res_mode[d]['auc']:.4f}")
        results[mode] = res_mode

    # ── Plot ──────────────────────────────────────────────────────────────────
    hops = list(range(1, D_MAX + 1))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ["coverage", "r", "sign_acc", "auc"]
    metric_labels = ["Coverage (fraction of edges\nwith ≥1 path at depth d)",
                     "Pearson r (path balance vs Y)",
                     "Sign accuracy (sign(score)==sign(Y))",
                     "AUC (path balance score vs Y)"]

    for mi, (mname, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[mi // 2, mi % 2]
        for mode, color, ls in [("full", "steelblue", "-"), ("debidir", "darkorange", "--")]:
            vals = [results[mode][d][mname] for d in hops]
            vals_plot = [v if not (isinstance(v, float) and math.isnan(v)) else None for v in vals]
            valid = [(h, v) for h, v in zip(hops, vals_plot) if v is not None]
            if valid:
                hx, vy = zip(*valid)
                ax.plot(hx, vy, marker="o", color=color, linestyle=ls, label=mode)
        ax.axhline(0.5 if mname in ("sign_acc", "auc") else 0,
                   color="gray", linewidth=0.7, linestyle=":")
        ax.set_xlabel("Path depth d")
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel.split("\n")[0])
        ax.legend(fontsize=8)
        ax.set_xticks(hops)

    fig.suptitle(f"{ds_name}: Balance-theory path signal at depth d\n"
                 f"(full graph vs de-bidirectionalized)", fontsize=12)
    fig.tight_layout()
    save_path = os.path.join(out_dir, f"bt_{ds_name}.png")
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
    print(f"  ✓ Saved {os.path.basename(save_path)}")

    return results

# ── Report ────────────────────────────────────────────────────────────────────

def write_report(all_results, out_dir):
    lines = [
        "=" * 72,
        "  BALANCE-THEORY PATH ANALYSIS REPORT",
        "=" * 72,
        "",
        "For each test edge (u,v) with label Y: find all directed paths of",
        "exactly length d from u to v. Compute sign product P = ∏ s(edge_i).",
        "path_balance_score = mean P over all d-paths.",
        "",
        "Key interpretation:",
        "  sign_acc ≈ 0.99 at d=2,3 → balance theory paths strongly predict sign",
        "  sign_acc drops to ≈0.5 at d=2 → no useful signal beyond 1 hop",
        "  coverage at d=3,4 → whether far paths exist at all",
        "",
        "Two modes:",
        "  full     — full graph (includes reciprocal edges)",
        "  debidir  — reciprocal edges removed (v→u stripped for each test edge u→v)",
        "",
    ]
    for ds_name, results in all_results.items():
        lines += [f"{'─'*72}", f"  DATASET: {ds_name}", f"{'─'*72}"]
        for mode in ["full", "debidir"]:
            lines.append(f"  [{mode}]")
            header = f"    {'d':<4}" + "  coverage  Pearson_r  sign_acc  AUC"
            lines.append(header)
            for d in range(1, D_MAX + 1):
                r = results[mode][d]
                lines.append(
                    f"    d={d}   {r['coverage']:.3f}    {r['r']:+.4f}    "
                    f"{r['sign_acc']:.4f}   {r['auc']:.4f}"
                )
            lines.append("")
    path = os.path.join(out_dir, "balance_theory_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n✓ Report: {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--out", default="outputs/balance_theory")
    args = parser.parse_args()

    if args.datasets == ["all"]:
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = args.datasets

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    for ds in datasets:
        if ds not in DATASET_CONFIGS:
            print(f"Unknown dataset: {ds}")
            continue
        result = analyse_dataset(ds, DATASET_CONFIGS[ds], out_dir)
        if result is not None:
            all_results[ds] = result

    if all_results:
        write_report(all_results, out_dir)
    print("\nDone.")

if __name__ == "__main__":
    main()
