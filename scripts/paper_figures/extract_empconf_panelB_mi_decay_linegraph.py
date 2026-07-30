"""Extract step for Empirical Confirmation Panel B (MI decay with distance) --
LINE-GRAPH / endpoint-based distance, replacing the old directed-source-only BFS.

Distance definition (agreed after discussion, see aaai2027/edge_distance_mutual_information_notes.md):
  For anchor edge (u,v) and context edge (x,y), let shell = the undirected
  shortest-path distance in G from the closer of {x,y} to the closer of {u,v}
  (i.e. min hop distance between any of the anchor's endpoints and any of the
  context edge's endpoints). Line-graph distance = shell + 1 (two edges that
  share a node are at distance 1, never 0 -- matches the standard line-graph
  L(G) construction: edges of G become nodes of L(G), adjacent iff they share
  an endpoint, direction ignored).

Architecture-agnostic by design: does not assume in-edge-only, out-edge-only,
or any specific GNN's aggregation scheme (checked: GINEConv uses real directed
in-edges only; SiGAT uses a mix of undirected/in/out/2-hop-triadic relations --
neither matches a single "hop" definition, so this uses neither, and instead
uses the graph-theoretic standard).

Implementation: per anchor EDGE (not per source node, since the root is now a
PAIR of nodes {u,v}), a multi-source BFS truncated at d_max shells, using a
generation-counter visited array (no O(V) reset between anchors). Every real
edge with at least one endpoint inside the d_max-shell is assigned to its own
minimum shell exactly once (deduplicated via edge id) -- no double-counting
even though both endpoints of a context edge may be reachable.

Real edges only (scripts.balance_theory_paths.load_edges_canonical) -- never
touches baselines/splits*/*.pt, so this is NOT affected by the fabricated-
reverse-edge issue (FABRICATED_REVERSE_EDGES.md).

Parallelism: each anchor's BFS is fully independent (no cross-anchor reuse --
see the discussion this was NOT further optimized to cache per-node BFS
results, since with 128 cores available, multiprocessing over anchor chunks
was the better cost/risk tradeoff). Graph arrays are built ONCE in the main
process and set as module-level globals BEFORE the worker pool is created, so
forked workers (Linux default start method) inherit them via copy-on-write --
no per-worker re-pickling of the graph.
"""
import argparse
import csv
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from scripts.balance_theory_paths import load_edges_canonical, DATASET_CONFIGS

D_MAX_DEFAULT = 4        # max SHELL depth from {u,v}; line-graph distance goes up to D_MAX+1
MIN_PAIRS_FOR_MI = 2000
OUT_CSV_DEFAULT = "aaai2027/figure_data/empconf_panelB_mi_decay_linegraph.csv"
CHUNK_SIZE = 50           # anchors per work item -- small enough for load balancing across hubs

# module-level graph state, set once per dataset by _set_graph() before the pool
# is created; forked workers see it via copy-on-write, no pickling needed.
_N = _E = _edge_x = _edge_y = _edge_s = _adj = _D_MAX = None


def mi_from_cont(c):
    n = int(c.sum())
    if n < MIN_PAIRS_FOR_MI:
        return float("nan"), float("nan"), n
    p = c / n
    py = p.sum(axis=1)
    ps = p.sum(axis=0)
    mi = 0.0
    for i in range(2):
        for j in range(2):
            if p[i, j] > 0 and py[i] > 0 and ps[j] > 0:
                mi += p[i, j] * math.log2(p[i, j] / (py[i] * ps[j]))
    hy = -sum(q * math.log2(q) for q in py if q > 0)
    nmi = mi / hy if hy > 1e-12 else float("nan")
    return float(mi), float(nmi), n


def build_graph(raw_edges):
    nodes = sorted({n for e in raw_edges for n in e[:2]})
    n2i = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    E = len(raw_edges)
    edge_x = np.empty(E, np.int32)
    edge_y = np.empty(E, np.int32)
    edge_s = np.empty(E, np.int8)
    adj = [[] for _ in range(N)]  # adj[w] = list of (neighbor_idx, edge_id), undirected
    for eid, (u, v, s) in enumerate(raw_edges):
        ui, vi = n2i[u], n2i[v]
        edge_x[eid] = ui
        edge_y[eid] = vi
        edge_s[eid] = s
        adj[ui].append((vi, eid))
        adj[vi].append((ui, eid))
    return N, E, edge_x, edge_y, edge_s, adj


def _set_graph(N, E, edge_x, edge_y, edge_s, adj, d_max):
    global _N, _E, _edge_x, _edge_y, _edge_s, _adj, _D_MAX
    _N, _E, _edge_x, _edge_y, _edge_s, _adj, _D_MAX = N, E, edge_x, edge_y, edge_s, adj, d_max


def _process_anchor(eid, shell, gen, cur_gen):
    """Runs one anchor's BFS + accumulation against the module-level graph
    globals. Returns a {line_dist: (yi, s2i)} Counter-ish list of increments."""
    d_max = _D_MAX
    edge_x, edge_y, edge_s, adj = _edge_x, _edge_y, _edge_s, _adj

    u, v, y = int(edge_x[eid]), int(edge_y[eid]), int(edge_s[eid])
    yi = 1 if y > 0 else 0

    roots = (u, v) if u != v else (u,)
    reached_nodes = []
    cur_frontier = []
    for w in roots:
        if gen[w] != cur_gen:
            gen[w] = cur_gen
            shell[w] = 0
            reached_nodes.append(w)
            cur_frontier.append(w)

    for d in range(1, d_max + 1):
        next_frontier = []
        for w in cur_frontier:
            for nb, _eid2 in adj[w]:
                if gen[nb] != cur_gen:
                    gen[nb] = cur_gen
                    shell[nb] = d
                    next_frontier.append(nb)
        if not next_frontier:
            break
        reached_nodes.extend(next_frontier)
        cur_frontier = next_frontier

    candidate_eids = set()
    for w in reached_nodes:
        for _nb, eid2 in adj[w]:
            if eid2 != eid:
                candidate_eids.add(eid2)

    local = {}  # (line_dist, yi, s2i) -> count
    for eid2 in candidate_eids:
        x2, y2, s2 = int(edge_x[eid2]), int(edge_y[eid2]), int(edge_s[eid2])
        sh_x = int(shell[x2]) if gen[x2] == cur_gen else d_max + 1
        sh_y = int(shell[y2]) if gen[y2] == cur_gen else d_max + 1
        d_e = min(sh_x, sh_y)
        if d_e > d_max:
            continue
        line_d = d_e + 1
        s2i = 1 if s2 > 0 else 0
        key = (line_d, yi, s2i)
        local[key] = local.get(key, 0) + 1
    return local


def _worker_chunk(anchor_chunk):
    """Runs a chunk of anchors in this worker process, reusing one shell/gen
    array across the whole chunk (generation-counter trick, no reset needed)."""
    N = _N
    shell = np.full(N, -1, np.int32)
    gen = np.zeros(N, np.int32)
    totals = {}
    for cur_gen, eid in enumerate(anchor_chunk, start=1):
        local = _process_anchor(int(eid), shell, gen, cur_gen)
        for key, cnt in local.items():
            totals[key] = totals.get(key, 0) + cnt
    return totals


def analyse_dataset(ds_name, cfg, d_max, max_anchors, seed=42, workers=None, shuffle_signs=False):
    t0 = time.time()
    raw_edges = load_edges_canonical(cfg["ds_name"])
    N, E, edge_x, edge_y, edge_s, adj = build_graph(raw_edges)
    print(f"\n{'='*64}\n{ds_name}: N={N:,} E={E:,}\n{'='*64}", flush=True)

    rng = np.random.default_rng(seed)
    if shuffle_signs:
        # control test: destroy any real sign correlation while keeping graph
        # structure and the overall sign balance (marginal) exactly fixed --
        # a random PERMUTATION of the sign array, not a resample.
        perm_rng = np.random.default_rng(seed + 1)
        edge_s = edge_s[perm_rng.permutation(E)].copy()
        print("  [SHUFFLE-SIGNS CONTROL: sign labels randomly permuted]", flush=True)
    anchor_ids = np.arange(E)
    if max_anchors is not None and E > max_anchors:
        anchor_ids = rng.choice(E, max_anchors, replace=False)
        print(f"  sampling {len(anchor_ids):,}/{E:,} anchor edges", flush=True)
    else:
        print(f"  using all {E:,} edges as anchors (exact)", flush=True)

    _set_graph(N, E, edge_x, edge_y, edge_s, adj, d_max)

    chunks = [anchor_ids[i:i + CHUNK_SIZE] for i in range(0, len(anchor_ids), CHUNK_SIZE)]
    n_workers = workers or mp.cpu_count()
    print(f"  {len(chunks):,} chunks of <= {CHUNK_SIZE}, {n_workers} workers", flush=True)

    conts = {d: np.zeros((2, 2), np.int64) for d in range(1, d_max + 2)}
    done_chunks = 0
    with mp.get_context("fork").Pool(processes=n_workers) as pool:
        for totals in pool.imap_unordered(_worker_chunk, chunks):
            for (line_d, yi, s2i), cnt in totals.items():
                conts[line_d][yi, s2i] += cnt
            done_chunks += 1
            if done_chunks % max(1, len(chunks) // 20) == 0 or done_chunks == len(chunks):
                elapsed = time.time() - t0
                eta = elapsed / done_chunks * (len(chunks) - done_chunks)
                print(f"    chunk {done_chunks:,}/{len(chunks):,}  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                      flush=True)

    results = {}
    for d in range(1, d_max + 2):
        mi, nmi, n = mi_from_cont(conts[d])
        results[d] = {"mi": mi, "nmi": nmi, "n_pairs": n}
        mi_s = f"{mi:.8f}" if not math.isnan(mi) else "nan"
        print(f"  line_dist={d:<3} n_pairs={n:>15,}  MI={mi_s:>13}  NMI={nmi}", flush=True)

    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["all"])
    ap.add_argument("--d-max", type=int, default=D_MAX_DEFAULT)
    ap.add_argument("--max-anchors", type=int, default=None)
    ap.add_argument("--out", default=OUT_CSV_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--shuffle-signs", action="store_true",
                     help="control test: randomly permute sign labels, keep structure/balance fixed")
    args = ap.parse_args()

    datasets = list(DATASET_CONFIGS.keys()) if args.datasets == ["all"] else args.datasets

    rows = []
    for ds in datasets:
        if ds not in DATASET_CONFIGS:
            print(f"unknown dataset: {ds}")
            continue
        res = analyse_dataset(ds, DATASET_CONFIGS[ds], args.d_max, args.max_anchors, args.seed, args.workers,
                               args.shuffle_signs)
        for d, r in res.items():
            rows.append({
                "dataset": ds, "line_dist": d, "n_pairs": r["n_pairs"],
                "mi_bits": r["mi"], "nmi": r["nmi"],
            })
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "line_dist", "n_pairs", "mi_bits", "nmi"])
            w.writeheader()
            w.writerows(rows)
        print(f"  [checkpoint written to {args.out}]", flush=True)

    print(f"\nDone. wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
