"""
Node-pair feature autocorrelation by hop distance d (exact, streaming).

Step 2 of the Information/Understanding-Track plan. Pure node-to-node
question, independent of any edge label:

    For a pair of nodes (A, B) where B is at exact BFS hop-distance d
    from A, how much does feature(A) tell you about feature(B)?

This tests whether node-level structural/embedding properties are
"smooth"/correlated across the graph at distance d -- i.e. whether
information propagates between nodes beyond d=1, independent of whether
that shows up as edge-sign MI (which vanishes for d>=2, per
edge_sign_mi_vs_distance_v3.py).

  forward   : out-BFS from anchor A (info flow along edge direction)
  transpose : in-BFS  from anchor A (A^T, info flow against edge direction)

Per-node features (computed once for ALL N nodes):
  out_degree, in_degree, signed_out_ratio, signed_in_ratio,
  clustering_coeff (directed/Fagiolo via nx.clustering), pagerank,
  hits_hub, hits_authority, emb_pca0..emb_pca{N_PCA-1}, emb_l2norm
  (embedding = E14_HARDNODE_L10 checkpoint's input embedding table, looked
  up via the dataset_cache.pt tokenizer's N_<id> tokens, reduced to N_PCA
  components via PCA fit over all graph nodes' embeddings).

For each (feature, d, direction) we report MI(bits) -- exact, via a 5x5
joint histogram over globally percentile-binned feature values -- and
Pearson r -- exact, via streaming sufficient statistics over raw values.
H(feature) (entropy of the global 5-bin marginal) is reported once per
feature as a reference ceiling for MI(d) (data-processing inequality).

Reuses the per-source-node hybrid BFS (CSR-slice / sparse-matvec, with a
generation-counter visited array) from edge_sign_mi_vs_distance_v3.py:
anchors = all nodes with out-degree>0 (forward) / in-degree>0 (transpose);
for each anchor, BFS frontiers at d=1..D_MAX are streamed directly into a
5x5 joint histogram + Pearson sufficient stats per (feature, d, direction)
-- no pair storage, no per-anchor sampling.

Usage
-----
  python scripts/node_mi_structural_embedding.py [--datasets all] [--out outputs/node_mi_structural]
  python scripts/node_mi_structural_embedding.py --datasets bitcoin-alpha --d-max 8
  python scripts/node_mi_structural_embedding.py --datasets epinions --max-anchors 20000
"""

import os, sys, glob, pickle, argparse, math, time
import numpy as np
import networkx as nx
import torch
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.data.datasets import get_loader
from src.model.lit_model import LitEdgeClassifier

D_MAX_DEFAULT      = 8
N_BINS             = 5
N_PCA              = 5
FRONTIER_THRESHOLD = 256

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
}


# ── Canonical edge loader (identical to training pipeline) ────────────────────

_cfg_cache: dict = {}
_edge_cache: dict = {}

def load_dataset_cfg(ds_name: str):
    if ds_name in _cfg_cache:
        return _cfg_cache[ds_name]
    orig_dir = os.getcwd()
    os.chdir(ROOT)
    try:
        cfg = load_config(overrides=[f"dataset.name={ds_name}"])
    finally:
        os.chdir(orig_dir)
    _cfg_cache[ds_name] = cfg
    return cfg


def load_edges_canonical(ds_name: str) -> list:
    if ds_name in _edge_cache:
        return _edge_cache[ds_name]
    cfg = load_dataset_cfg(ds_name)
    orig_dir = os.getcwd()
    os.chdir(ROOT)
    try:
        edges = get_loader(ds_name)(cfg)
    finally:
        os.chdir(orig_dir)
    _edge_cache[ds_name] = edges
    return edges


# ── Embedding loading ─────────────────────────────────────────────────────────

def load_embedding(cfg):
    """Returns (emb_matrix [vocab,dim] float32 ndarray, token2id dict) or (None, None)."""
    dscfg = load_dataset_cfg(cfg["ds_name"])
    cache_path = os.path.join(ROOT, dscfg.dataset.data_dir, "dataset_cache.pt")
    if not os.path.exists(cache_path):
        print(f"  ✗ dataset_cache.pt not found at {cache_path}")
        return None, None

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    token2id = cache["tokenizer"]["token2id"]

    exp_dir = os.path.join(ROOT, cfg["exp_dir"])
    epoch = cfg["best_epoch"]
    pattern = os.path.join(exp_dir, "runs", cfg["ds_name"], "E14_HARDNODE_L10",
                            "checkpoints", f"*-epoch={epoch:02d}-*.ckpt")
    candidates = [c for c in glob.glob(pattern) if "last" not in c]
    if not candidates:
        print(f"  ✗ checkpoint not found: {pattern}")
        return None, None

    model = LitEdgeClassifier.load_from_checkpoint(candidates[0], map_location="cpu")
    emb = model.model.embed.weight.detach().numpy().astype(np.float32)
    return emb, token2id


# ── BFS hop frontiers (hybrid CSR-slice / sparse-matvec, generation counter) ──
# Adapted from edge_sign_mi_vs_distance_v3.py's per-source BFS loop.

def bfs_frontiers(start, indptr, indices, matvec_mat, visited_gen, gen, N, d_max,
                  frontier_threshold=FRONTIER_THRESHOLD):
    """Return dict: d -> np.ndarray of node indices at exact hop-distance d
    from `start` (d=1..d_max, stops early once the frontier is exhausted)."""
    visited_gen[start] = gen
    frontier = np.array([start], np.int32)
    out = {}
    for d in range(1, d_max + 1):
        if len(frontier) == 0:
            break
        if len(frontier) <= frontier_threshold:
            if len(frontier) == 1:
                all_nbrs = indices[indptr[frontier[0]]:indptr[frontier[0] + 1]]
            else:
                parts = [indices[indptr[f]:indptr[f + 1]] for f in frontier]
                all_nbrs = np.concatenate(parts) if len(parts) > 1 else parts[0]
            if len(all_nbrs) == 0:
                break
            mask = visited_gen[all_nbrs] != gen
            new_front = np.unique(all_nbrs[mask])
        else:
            fv = np.zeros(N, np.float32)
            fv[frontier] = 1.0
            candidates = matvec_mat.dot(fv) > 0
            new_front = np.where(candidates & (visited_gen != gen))[0].astype(np.int32)

        if len(new_front) == 0:
            break
        visited_gen[new_front] = gen
        frontier = new_front
        out[d] = frontier
    return out


# ── MI / entropy / Pearson from streaming aggregates ──────────────────────────

def mi_from_joint(J):
    """MI(bits) from an n_bins x n_bins joint count matrix. nan if J is empty."""
    J = J.astype(np.float64)
    n = J.sum()
    if n <= 0:
        return float("nan")
    p = J / n
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p / (px * py)
        terms = np.where(p > 0, p * np.log2(ratio), 0.0)
    return float(terms.sum())


def entropy_from_marginal(counts):
    """H(bits) from a 1D count vector. nan if empty."""
    n = counts.sum()
    if n <= 0:
        return float("nan")
    p = counts / n
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def pearson_from_suffstats(n, sx, sx2, sy, sy2, sxy):
    """Pearson r from streaming sufficient statistics. nan if n too small."""
    if n < 10:
        return float("nan")
    num  = n * sxy - sx * sy
    denx = n * sx2 - sx * sx
    deny = n * sy2 - sy * sy
    if denx <= 1e-12 or deny <= 1e-12:
        return 0.0
    return float(num / math.sqrt(denx * deny))


# ── Per-dataset analysis ──────────────────────────────────────────────────────

STRUCT_FEATURE_NAMES = [
    "out_degree", "in_degree", "signed_out_ratio", "signed_in_ratio",
    "clustering_coeff", "pagerank", "hits_hub", "hits_authority",
]


def analyse_dataset(ds_name: str, cfg: dict, out_dir: str,
                     d_max: int = D_MAX_DEFAULT, max_anchors=None, n_bins: int = N_BINS):
    t_start = time.time()
    print(f"\n{'─'*64}\n  {ds_name}  (D_MAX={d_max})\n{'─'*64}")

    # ── Load edges, build compact-index graph + CSR adjacency ────────────────
    edges = load_edges_canonical(cfg["ds_name"])
    all_nodes = sorted({n for e in edges for n in e[:2]})
    N = len(all_nodes)
    n2i = {n: i for i, n in enumerate(all_nodes)}
    E = len(edges)
    print(f"  Loaded {E:,} edges, {N:,} nodes")

    srcs  = np.empty(E, np.int32)
    dsts  = np.empty(E, np.int32)
    signs = np.empty(E, np.int8)
    for k, (u, v, s) in enumerate(edges):
        srcs[k] = n2i[u]; dsts[k] = n2i[v]; signs[k] = s

    pos_out = np.zeros(N, np.int64); neg_out = np.zeros(N, np.int64)
    pos_in  = np.zeros(N, np.int64); neg_in  = np.zeros(N, np.int64)
    for k in range(E):
        if signs[k] > 0:
            pos_out[srcs[k]] += 1; pos_in[dsts[k]] += 1
        else:
            neg_out[srcs[k]] += 1; neg_in[dsts[k]] += 1

    out_degree_arr = (pos_out + neg_out).astype(np.float64)
    in_degree_arr  = (pos_in + neg_in).astype(np.float64)

    A  = csr_matrix((np.ones(E, bool), (srcs, dsts)), shape=(N, N))
    AT = A.T.tocsr()

    # ── Structural features (whole graph, compact integer ids) ───────────────
    t0 = time.time()
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    G.add_edges_from(zip(srcs.tolist(), dsts.tolist()))

    clustering = nx.clustering(G)
    print(f"  clustering done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    pagerank = nx.pagerank(G)
    print(f"  pagerank done ({time.time()-t0:.1f}s); sum={sum(pagerank.values()):.4f}")

    t0 = time.time()
    try:
        hits_hub, hits_auth = nx.hits(G, max_iter=500)
    except nx.PowerIterationFailedConvergence:
        try:
            hits_hub, hits_auth = nx.hits(G, max_iter=2000, tol=1e-6)
        except nx.PowerIterationFailedConvergence:
            hits_hub  = {i: float("nan") for i in range(N)}
            hits_auth = {i: float("nan") for i in range(N)}
    print(f"  hits done ({time.time()-t0:.1f}s)")

    with np.errstate(invalid="ignore", divide="ignore"):
        signed_out_ratio_arr = np.where(out_degree_arr > 0, pos_out / np.maximum(out_degree_arr, 1), np.nan)
        signed_in_ratio_arr  = np.where(in_degree_arr  > 0, pos_in  / np.maximum(in_degree_arr, 1),  np.nan)

    struct_feats = {
        "out_degree":       out_degree_arr,
        "in_degree":        in_degree_arr,
        "signed_out_ratio": signed_out_ratio_arr,
        "signed_in_ratio":  signed_in_ratio_arr,
        "clustering_coeff": np.array([clustering[i] for i in range(N)], dtype=np.float64),
        "pagerank":         np.array([pagerank[i] for i in range(N)], dtype=np.float64),
        "hits_hub":         np.array([hits_hub[i] for i in range(N)], dtype=np.float64),
        "hits_authority":   np.array([hits_auth[i] for i in range(N)], dtype=np.float64),
    }

    # ── Embedding features (whole graph) ──────────────────────────────────────
    t0 = time.time()
    emb, token2id = load_embedding(cfg)
    embed_feats = {}
    if emb is not None:
        idx_map = np.full(N, -1, dtype=np.int64)
        for i, orig in enumerate(all_nodes):
            tid = token2id.get(f"N_{orig}")
            if tid is not None:
                idx_map[i] = tid
        valid_emb = idx_map >= 0
        n_valid = int(valid_emb.sum())
        n_comp = min(N_PCA, n_valid, emb.shape[1])
        if n_comp >= 1:
            X = np.full((N, emb.shape[1]), np.nan, dtype=np.float32)
            X[valid_emb] = emb[idx_map[valid_emb]]
            l2norm = np.linalg.norm(X, axis=1)

            pca = PCA(n_components=n_comp, random_state=42)
            proj_valid = pca.fit_transform(X[valid_emb])
            proj = np.full((N, n_comp), np.nan, dtype=np.float64)
            proj[valid_emb] = proj_valid

            for k in range(n_comp):
                embed_feats[f"emb_pca{k}"] = proj[:, k]
            embed_feats["emb_l2norm"] = l2norm.astype(np.float64)
            print(f"  embedding loaded ({time.time()-t0:.1f}s); dim={emb.shape[1]}, "
                  f"valid={n_valid}/{N}, PCA explained_var={pca.explained_variance_ratio_[:n_comp].sum():.3f}")
        else:
            print(f"  embedding skipped ({time.time()-t0:.1f}s); too few valid nodes ({n_valid})")
    else:
        print(f"  embedding skipped ({time.time()-t0:.1f}s)")

    all_feats = dict(struct_feats)
    all_feats.update(embed_feats)
    feature_names = list(all_feats.keys())

    # ── Global per-feature percentile binning + entropy reference ────────────
    bin_idx       = {}
    valid_mask    = {}
    n_bins_actual = {}
    entropy       = {}
    for fname, arr in all_feats.items():
        arr = np.asarray(arr, dtype=np.float64)
        vmask = ~np.isnan(arr)
        valid_mask[fname] = vmask
        if not vmask.any():
            bin_idx[fname] = np.full(N, -1, dtype=np.int8)
            n_bins_actual[fname] = 1
            entropy[fname] = float("nan")
            continue
        v = arr[vmask]
        edges_p = np.unique(np.percentile(v, np.linspace(0, 100, n_bins + 1)))
        bidx = np.full(N, -1, dtype=np.int8)
        if len(edges_p) < 2:
            bidx[vmask] = 0
            nb = 1
        else:
            bidx[vmask] = np.clip(np.digitize(v, edges_p[1:-1]), 0, len(edges_p) - 2)
            nb = len(edges_p) - 1
        bin_idx[fname] = bidx
        n_bins_actual[fname] = nb
        counts = np.bincount(bidx[vmask], minlength=nb)
        entropy[fname] = entropy_from_marginal(counts)

    # ── Anchors ────────────────────────────────────────────────────────────────
    # Out-BFS only: in-BFS would be mathematically redundant for this metric
    # (dist_trans(A,B) = dist_fwd(B,A), and MI/Pearson r are symmetric in their
    # two arguments since `feature` is the same function for A and B).
    anchors = np.where(out_degree_arr > 0)[0].astype(np.int32)
    rng = np.random.default_rng(42)
    if max_anchors is not None and len(anchors) > max_anchors:
        anchors = rng.choice(anchors, max_anchors, replace=False)

    # ── Streaming accumulators: J[feature][d], stats[feature][d] ─────────────
    J = {
        fn: {d: np.zeros((n_bins_actual[fn], n_bins_actual[fn]), np.int64)
             for d in range(1, d_max + 1)}
        for fn in feature_names
    }
    # stats[fn][d] = [n, sx, sx2, sy, sy2, sxy]
    stats = {
        fn: {d: [0, 0.0, 0.0, 0.0, 0.0, 0.0] for d in range(1, d_max + 1)}
        for fn in feature_names
    }

    t0 = time.time()
    indptr, indices, matvec_mat = A.indptr, A.indices, AT
    visited_gen = np.zeros(N, np.int32)
    gen = 0
    for ai, anchor in enumerate(anchors):
        if ai % 20_000 == 0 and ai > 0:
            elapsed = time.time() - t0
            eta = elapsed / ai * (len(anchors) - ai)
            print(f"    {ai:,}/{len(anchors):,}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

        gen += 1
        if gen >= 2**30:
            visited_gen[:] = 0
            gen = 1

        frontiers = bfs_frontiers(int(anchor), indptr, indices, matvec_mat,
                                   visited_gen, gen, N, d_max)
        for d, frontier in frontiers.items():
            for fn in feature_names:
                if not valid_mask[fn][anchor]:
                    continue
                fmask = valid_mask[fn][frontier]
                if not fmask.any():
                    continue
                vf = frontier[fmask]
                bA = int(bin_idx[fn][anchor])
                hist = np.bincount(bin_idx[fn][vf], minlength=n_bins_actual[fn])
                J[fn][d][bA, :] += hist

                fA = float(all_feats[fn][anchor])
                s = all_feats[fn][vf]
                m = len(vf)
                ssum = float(s.sum())
                st = stats[fn][d]
                st[0] += m
                st[1] += fA * m
                st[2] += fA * fA * m
                st[3] += ssum
                st[4] += float((s * s).sum())
                st[5] += fA * ssum

    print(f"  BFS+accum done ({time.time()-t0:.1f}s); anchors={len(anchors):,}")

    # ── Final metrics ──────────────────────────────────────────────────────────
    results = {fn: {"H": entropy[fn], "by_d": {}} for fn in feature_names}
    for fn in feature_names:
        for d in range(1, d_max + 1):
            mi = mi_from_joint(J[fn][d])
            n_, sx, sx2, sy, sy2, sxy = stats[fn][d]
            r = pearson_from_suffstats(n_, sx, sx2, sy, sy2, sxy)
            results[fn]["by_d"][d] = {"mi": mi, "r": r, "n": n_}

    # ── Plot: MI vs d, per feature, with H(feature) reference ────────────────
    n_feat = len(feature_names)
    ncols = 4
    nrows = math.ceil(n_feat / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)
    hops = list(range(1, d_max + 1))
    for fi, fname in enumerate(feature_names):
        ax = axes[fi // ncols, fi % ncols]
        vals = [results[fname]["by_d"][d]["mi"] for d in hops]
        valid = [(h, v) for h, v in zip(hops, vals) if not (isinstance(v, float) and math.isnan(v))]
        if valid:
            hx, vy = zip(*valid)
            ax.plot(hx, vy, marker="o", label="MI(d)", color="tab:blue")
        h_val = results[fname]["H"]
        if not math.isnan(h_val):
            ax.axhline(h_val, color="gray", linewidth=0.7, linestyle="--", label="H(feature)")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_title(fname, fontsize=9)
        ax.set_xlabel("hop d")
        ax.set_ylabel("MI (bits)")
        ax.set_xticks(hops)
        if fi == 0:
            ax.legend(fontsize=7)
    for fi in range(n_feat, nrows * ncols):
        axes[fi // ncols, fi % ncols].axis("off")

    fig.suptitle(
        f"{ds_name}: node-pair feature MI(A,B) vs hop distance d "
        f"(anchors={len(anchors):,})",
        fontsize=11)
    fig.tight_layout()
    save_path = os.path.join(out_dir, f"node_mi_{ds_name}.png")
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
    print(f"  ✓ Saved {os.path.basename(save_path)}")
    print(f"  Total time: {time.time()-t_start:.1f}s")

    return {
        "results": results,
        "feature_names": feature_names,
        "n_nodes": N,
        "n_anchors": len(anchors),
        "d_max": d_max,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(all_results, out_dir):
    lines = [
        "=" * 78,
        "  NODE-PAIR FEATURE AUTOCORRELATION BY HOP DISTANCE (exact streaming)",
        "=" * 78,
        "",
        "MI(bits) and Pearson r between feature(A) and feature(B) for node pairs",
        "(A,B) where B is at exact out-BFS hop-distance d from A (A's out-edges).",
        "(in-BFS / transpose is mathematically identical for this metric --",
        " dist_trans(A,B) = dist_fwd(B,A), and MI/Pearson r are symmetric in",
        " their two arguments since `feature` is the same function for A and B.)",
        "",
        "H(feature): entropy (bits, 5-bin global percentile discretisation) over",
        "all N nodes -- a reference ceiling for MI(d) (data-processing inequality).",
        "",
    ]

    for ds_name, res in all_results.items():
        results    = res["results"]
        feat_names = res["feature_names"]
        d_max      = res["d_max"]

        lines += [
            f"{'─'*78}",
            f"  DATASET: {ds_name}   (N={res['n_nodes']:,}, "
            f"anchors={res['n_anchors']:,}, D_MAX={d_max})",
            f"{'─'*78}",
            "",
            "  H(feature) reference (bits):",
        ]
        for fname in feat_names:
            h = results[fname]["H"]
            h_s = f"{h + 0.0:.4f}" if not math.isnan(h) else "nan"
            lines.append(f"    {fname:<20} H = {h_s}")
        lines.append("")

        header = f"  {'feature':<20}" + "".join(f"  d={d}" for d in range(1, d_max + 1))

        lines.append("  MI (bits) by feature x d:")
        lines.append(header)
        for fname in feat_names:
            row = f"  {fname:<20}"
            for d in range(1, d_max + 1):
                mi = results[fname]["by_d"][d]["mi"]
                row += "   nan " if (isinstance(mi, float) and math.isnan(mi)) else f"  {mi:.4f}"
            lines.append(row)
        lines.append("")

        lines.append("  Pearson r by feature x d:")
        lines.append(header)
        for fname in feat_names:
            row = f"  {fname:<20}"
            for d in range(1, d_max + 1):
                r = results[fname]["by_d"][d]["r"]
                row += "   nan " if (isinstance(r, float) and math.isnan(r)) else f"  {r:+.3f}"
            lines.append(row)
        lines.append("")

        lines.append("  n_pairs by d (out_degree, NaN-free reference):")
        n_header = f"  {'':<20}" + "".join(f"  d={d}" for d in range(1, d_max + 1))
        row = f"  {'':<20}"
        for d in range(1, d_max + 1):
            n_ = results["out_degree"]["by_d"][d]["n"]
            row += f"  {n_:>5,}" if n_ < 100_000 else f"  {n_:>5.1e}"
        lines.append(n_header)
        lines.append(row)
        lines.append("")

    path = os.path.join(out_dir, "node_mi_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n✓ Report written to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--out", default="outputs/node_mi_structural")
    parser.add_argument("--d-max", type=int, default=D_MAX_DEFAULT)
    parser.add_argument("--max-anchors", type=int, default=None,
                         help="Cap on anchors per direction (default: exact, all nodes)")
    args = parser.parse_args()

    if args.datasets == ["all"]:
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = args.datasets

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    for ds_name in datasets:
        if ds_name not in DATASET_CONFIGS:
            print(f"Unknown dataset: {ds_name}")
            continue
        result = analyse_dataset(ds_name, DATASET_CONFIGS[ds_name], out_dir,
                                  d_max=args.d_max, max_anchors=args.max_anchors)
        if result is not None:
            all_results[ds_name] = result
            with open(os.path.join(out_dir, f"node_mi_{ds_name}_result.pkl"), "wb") as f:
                pickle.dump(result, f)

    # Merge in any previously-computed results for datasets not in this run,
    # so the report stays consolidated across incremental invocations.
    for ds_name in DATASET_CONFIGS:
        if ds_name in all_results:
            continue
        pkl_path = os.path.join(out_dir, f"node_mi_{ds_name}_result.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                all_results[ds_name] = pickle.load(f)

    if all_results:
        ordered = {k: all_results[k] for k in DATASET_CONFIGS if k in all_results}
        write_report(ordered, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
