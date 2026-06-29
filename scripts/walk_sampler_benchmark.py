"""Phase 2 benchmark (no GPU, no training): compare a coverage sampler against the
uniform baseline on throughput + node/edge coverage + saturation floor.

Edges/splits are taken from data/<ds>/dataset_cache.pt (the exact graph the walk
pipeline uses), so no hydra config is needed.

Usage:
  python scripts/walk_sampler_benchmark.py <ds> --nw 500000 --k 5 --workers 8
"""
import os, sys, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.walk_sampler import sample_random_walks
from src.data.coverage_aware_sampler import k_cover_walks_fast, _build_edge_index

DATA_DIRS = {"bitcoin-alpha": "data/bitcoin-alpha", "bitcoin-otc": "data/bitcoin-otc",
             "epinions": "data/epinions", "wiki-elec": "data/wiki-Elec",
             "wiki-rfa": "data/wiki-RfA", "slashdot090221": "data/slashdot090221"}


def load_graph(ds):
    c = torch.load(os.path.join(DATA_DIRS[ds], "dataset_cache.pt"), weights_only=False)
    splits = {k: set(map(tuple, v)) for k, v in c["splits"].items()}
    edges = []
    seen = set()
    for s in splits.values():
        for e in s:
            e = (int(e[0]), int(e[1]), int(e[2]))
            if e not in seen:
                seen.add(e); edges.append(e)
    return edges, splits


def coverage_stats(walks, edges, splits, k):
    eidx = _build_edge_index(edges)
    m = len(eidx)
    vc = np.zeros(m, dtype=np.int64)
    nodes_seen = set()
    for w in walks:
        if w:
            nodes_seen.add(int(w[0][2:]))
        for i in range(1, len(w), 2):
            try:
                u = int(w[i - 1][2:]); v = int(w[i + 1][2:]); l = int(w[i][2:])
            except (ValueError, IndexError):
                continue
            nodes_seen.add(v)
            e = eidx.get((u, v, l), -1)
            if e >= 0:
                vc[e] += 1
    all_nodes = set()
    for s in splits.values():
        for (a, b, _l) in s:
            all_nodes.add(a); all_nodes.add(b)
    # per split coverage via edge_index membership
    out = {"edge_cov": float((vc > 0).mean()),
           "edge_cov_ge_k": float((vc >= k).mean()),
           "node_cov": len(nodes_seen & all_nodes) / len(all_nodes),
           "min_visit": int(vc.min()), "median_visit": float(np.median(vc)),
           "p10_visit": float(np.percentile(vc, 10)), "mean_visit": float(vc.mean())}
    for name, s in splits.items():
        ids = [eidx[e] for e in s if e in eidx]
        sub = vc[ids] if ids else np.array([0])
        out[f"{name}_cov"] = float((sub > 0).mean())
        out[f"{name}_cov_ge_k"] = float((sub >= k).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("--nw", type=int, default=500000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mw", type=int, default=80)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-uniform", action="store_true")
    a = ap.parse_args()

    edges, splits = load_graph(a.ds)
    print(f"{a.ds}: |E|={len(edges)}  nominal test={len(splits['test'])}  "
          f"budget nw={a.nw} mw={a.mw} k={a.k} workers={a.workers}", flush=True)

    if not a.skip_uniform:
        t = time.time()
        wu = sample_random_walks(edges, num_walks=a.nw, max_walk_length=a.mw,
                                 num_workers=a.workers, seed=a.seed)
        tu = time.time() - t
        su = coverage_stats(wu, edges, splits, a.k)
        print(f"  UNIFORM   gen={tu:6.1f}s  edge_cov={su['edge_cov']:.4f} "
              f"ge{a.k}={su['edge_cov_ge_k']:.4f} node={su['node_cov']:.4f} "
              f"test_cov={su['test_cov']:.4f} min={su['min_visit']} med={su['median_visit']:.0f}",
              flush=True)

    t = time.time()
    wk = k_cover_walks_fast(edges, num_walks=a.nw, max_walk_length=a.mw,
                            num_workers=a.workers, seed=a.seed, k=a.k)
    tk = time.time() - t
    sk = coverage_stats(wk, edges, splits, a.k)
    print(f"  KCOVER k{a.k} gen={tk:6.1f}s  edge_cov={sk['edge_cov']:.4f} "
          f"ge{a.k}={sk['edge_cov_ge_k']:.4f} node={sk['node_cov']:.4f} "
          f"test_cov={sk['test_cov']:.4f} test_ge{a.k}={sk['test_cov_ge_k']:.4f} "
          f"min={sk['min_visit']} med={sk['median_visit']:.0f} p10={sk['p10_visit']:.0f}",
          flush=True)


if __name__ == "__main__":
    main()
