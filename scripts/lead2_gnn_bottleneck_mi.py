"""
Lead 2 (GNN bottleneck) -- Step 1: node-level MI(h_v^(1), sign of v's own
out-edges).

h_v^(1) (a GNN's embedding of node v after exactly one message-passing layer)
is built by aggregating over v's own incident edges. The bottleneck
hypothesis (plan-research-leads.md, Lead 2) says: a node q that is 2 hops
from some target node u (because q is a neighbor of some 1-hop relay v of u)
has its sign(v,q) summed/averaged together with v's *other* neighbors before
it can influence any downstream prediction. This script tests the most
direct, literal version of that claim: does h_v^(1) retain recoverable
information about a *specific* one of v's own edges, or does it wash out
under aggregation?

Pairing: for every (v, q) edge in the TRAINING graph (the only edges that
actually fed into h_v^(1) during the forward pass that produced the cached
artifact), one sample (h_v^(1)[v], sign(v,q)). A high-degree v contributes
many samples sharing the same anchor vector.

Ceiling: H(sign) ~= 1 bit (max possible if every neighbor's sign were
retained losslessly) -- NOT outputs/mi_vs_dist's existing numbers, which
measure raw data correlation across BFS distance, a different question (see
plan session notes). mi_pca_bins()'s own "nmi" field is exactly
MI / H(context) i.e. already normalized against this ceiling.
Floor: a permutation-null baseline (shuffle the v->h_v^(1) assignment,
breaking the true correspondence but preserving each node's exact edge
multiplicity) -- should give MI ~= 0.

Usage
-----
  python scripts/lead2_gnn_bottleneck_mi.py --datasets bitcoin-alpha
  python scripts/lead2_gnn_bottleneck_mi.py --datasets all
"""
import os, sys, pickle, argparse, math
from typing import Optional
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.mi_pca_binning_utils import mi_pca_bins

ALL_DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions",
                "wiki-elec", "wiki-rfa", "slashdot090221"]

MODEL_ARTIFACT_PATHS = {
    "GINEConv": "baselines/GINEConv/results_our_splits/{ds}/GINEConv/seed42/best_epoch_artifacts.pkl",
    "CSG":      "baselines/CSG/results_our_splits/{ds}/CSG/seed42/best_epoch_artifacts.pkl",
}


def load_h1(ds_name: str, model_name: str):
    """Returns h_v^(1) as an (num_nodes, dim) float32 ndarray, or None if the
    artifact is missing (caller should train it first -- see plan Step 1)."""
    path = os.path.join(ROOT, MODEL_ARTIFACT_PATHS[model_name].format(ds=ds_name))
    if not os.path.exists(path):
        print(f"  ✗ missing artifact: {path}")
        return None
    with open(path, "rb") as f:
        art = pickle.load(f)
    if model_name == "GINEConv":
        return np.asarray(art["layer_embeddings"][1], dtype=np.float32)
    elif model_name == "CSG":
        return np.asarray(art["layer1_embedding"], dtype=np.float32)
    raise ValueError(model_name)


def load_train_edges(ds_name: str):
    """Returns (src, dst, sign) int64/int64/float32 arrays for TRAINING-split
    edges only -- the only edges that actually fed into h_v^(1) during the
    forward pass that produced the cached artifact."""
    splits_path = os.path.join(ROOT, "baselines", "splits", f"{ds_name}.pt")
    splits = torch.load(splits_path, weights_only=False)
    ei = splits["edge_index"]
    ew = splits["edge_weight"]
    trn_mask = splits["trn_mask"]
    src = ei[0, trn_mask].numpy().astype(np.int64)
    dst = ei[1, trn_mask].numpy().astype(np.int64)
    sign = ew[trn_mask].numpy().astype(np.float32)
    return src, dst, sign


def permutation_null(h1: np.ndarray, src: np.ndarray, dst: np.ndarray, sign: np.ndarray,
                      n_pca: int, n_bins: int, seed: int = 42) -> dict:
    """Floor baseline: permute which node's h_v^(1) is paired with which
    node's out-edges, preserving each node's exact out-degree (edge
    multiplicity) but breaking the true v <-> h_v^(1) correspondence."""
    rng = np.random.default_rng(seed)
    num_nodes = h1.shape[0]
    perm = rng.permutation(num_nodes)
    src_perm = perm[src]
    return mi_pca_bins(h1[src_perm], sign, n_pca=n_pca, n_bins=n_bins)


def analyse(ds_name: str, model_name: str, n_pca: int = 5, n_bins: int = 5) -> Optional[dict]:
    h1 = load_h1(ds_name, model_name)
    if h1 is None:
        return None
    src, dst, sign = load_train_edges(ds_name)

    anchors = h1[src]  # (E_train, dim) -- h_v^(1) for each edge's source node
    result = mi_pca_bins(anchors, sign, n_pca=n_pca, n_bins=n_bins)
    null = permutation_null(h1, src, dst, sign, n_pca=n_pca, n_bins=n_bins)

    h_sign = math.log2(2)  # balanced-binary ceiling reference; report alongside actual H(context)
    return {
        "dataset": ds_name, "model": model_name,
        "n_nodes": h1.shape[0], "n_train_edges": len(src),
        "mi": result["mi"], "nmi": result["nmi"], "n_pairs": result["n_pairs"],
        "best_component": result["best_component"], "n_components": result["n_components"],
        "mi_per_component": result["mi_per_component"],
        "null_mi": null["mi"], "null_nmi": null["nmi"],
        "h_sign_ceiling_bits": h_sign,
    }


def write_report(all_results: list, out_dir: str):
    lines = [
        "=" * 92,
        "  LEAD 2 STEP 1 -- MI(h_v^(1), sign of v's own out-edges)",
        "=" * 92,
        "",
        "Ceiling: H(sign) ~= 1.0 bit (max possible if every neighbor's sign were",
        "retained losslessly by h_v^(1)). 'nmi' = mi_pca_bins' own MI/H(context),",
        "i.e. already normalized against this ceiling -- read nmi close to 1.0 as",
        "near-lossless retention, nmi close to 0 as near-total loss (bottleneck).",
        "'null_mi'/'null_nmi': permutation-null floor (same edge multiplicities,",
        "shuffled v<->h_v^(1) correspondence) -- should be ~0; a real mi well above",
        "this confirms h_v^(1) carries genuine, non-spurious signal.",
        "",
        f"  {'dataset':<16}{'model':<10}{'n_nodes':>9}{'n_train_e':>11}"
        f"{'mi(bits)':>11}{'nmi':>9}{'null_mi':>10}{'null_nmi':>10}{'best_pc':>9}",
    ]
    for r in all_results:
        lines.append(
            f"  {r['dataset']:<16}{r['model']:<10}{r['n_nodes']:>9}{r['n_train_edges']:>11}"
            f"{r['mi']:>11.6f}{r['nmi']:>9.4f}{r['null_mi']:>10.6f}{r['null_nmi']:>10.4f}"
            f"{r['best_component']:>9}"
        )
    lines.append("")
    lines.append("Per-PCA-component MI (bits), one row per dataset x model:")
    for r in all_results:
        comps = ", ".join(f"{m:.6f}" for m in r["mi_per_component"])
        lines.append(f"  {r['dataset']:<16}{r['model']:<10} [{comps}]")

    path = os.path.join(out_dir, "h1_mi_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n✓ Report written to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--models", nargs="+", default=["GINEConv", "CSG"])
    parser.add_argument("--out", default="outputs/lead2_gnn_bottleneck")
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.datasets == ["all"] else args.datasets
    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    for ds in datasets:
        for model_name in args.models:
            print(f"\n── {ds} x {model_name} ──")
            res = analyse(ds, model_name)
            if res is None:
                print(f"  skipped (train artifact first)")
                continue
            print(f"  mi={res['mi']:.6f} bits  nmi={res['nmi']:.4f}  "
                  f"null_mi={res['null_mi']:.6f}  n_pairs={res['n_pairs']:,}")
            all_results.append(res)
            with open(os.path.join(out_dir, f"{ds}_{model_name}_h1_mi.pkl"), "wb") as f:
                pickle.dump(res, f)

    # Merge in any previously-computed results not in this run, so the report
    # stays consolidated across incremental invocations.
    seen = {(r["dataset"], r["model"]) for r in all_results}
    for ds in ALL_DATASETS:
        for model_name in ["GINEConv", "CSG"]:
            if (ds, model_name) in seen:
                continue
            pkl_path = os.path.join(out_dir, f"{ds}_{model_name}_h1_mi.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    all_results.append(pickle.load(f))

    if all_results:
        all_results.sort(key=lambda r: (ALL_DATASETS.index(r["dataset"]), r["model"]))
        write_report(all_results, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
