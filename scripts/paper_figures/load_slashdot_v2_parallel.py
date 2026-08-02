"""Parallelized version of load_slashdot_v2.py (itself an instrumented copy
of the user-uploaded `load_slashdot (1).py`, repo root, untouched).

WHY THIS EXISTS: the sequential v2 run (seed=42, 10,000 random anchors,
undirected BFS) measured at ~1.4-1.5s/anchor -> ~4 hour ETA, confirmed live
before being stopped (user's explicit choice: parallelize rather than wait
or shrink the anchor count).

CHANGES vs. load_slashdot_v2.py -- ONLY these two, both non-semantic:
  1. [perf bug fix] Her (and v2's) inner loop does
     `list(g.edges(data=True))[edge_id]` INSIDE the per-anchor for-loop --
     i.e. it rebuilds the full O(E) edge list from scratch on every single
     one of the 10,000 iterations. This is almost certainly the dominant
     cost (E=500,481 edges here => ~5 billion redundant tuple constructions
     across the run), not the undirected-BFS-is-expensive effect seen
     earlier in a different context. Fixed by building the edge list ONCE
     before the loop -- returns the exact same (v1,v2,data) per edge_id,
     so this changes nothing about the result, only removes redundant work.
  2. [parallelization] The per-anchor BFS body (100% identical logic to
     v2/her original -- same undirected graph, same tree-discovery-edge
     recording, same distance bucketing) is farmed out across a fork-based
     multiprocessing Pool, one anchor per unit of work, chunked. Workers
     inherit the graph via copy-on-write fork (no re-pickling, no rewrite
     of the graph into a different data structure) -- same networkx
     g.neighbors() calls, same iteration order, so ties resolve identically
     to what a sequential run would do.

Randomness ordering is preserved to match what a (hypothetical, fast)
sequential run under seed=42 would do: `random.seed(42)` and the anchor
`random.sample(...)` call both happen in the MAIN process before any
forking (workers never call `random`), and the MAX_PAIRS capping step
(the only other place `random` is used) also stays in the main process,
run sequentially AFTER all workers finish and their results are merged --
so it consumes from the exact same, single, seed-42 random stream in the
exact same order as the original script, just with the expensive BFS work
moved off the main process.
"""
import os
import random
import sys
import time
import multiprocessing as mp
from collections import deque

import networkx as nx
import numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT, "aaai2027", "external_review", "slashdot090221.edgelist")
OUT_PNG = os.path.join(ROOT, "slashdot_sign_correlation_v2_seed42_parallel.png")
OUT_CSV = os.path.join(ROOT, "outputs", "panelb_diagnostics", "her_v2_seed42_parallel.csv")

MAX_DIST = 8
MAX_PAIRS = 10_000
N_ANCHORS = 10_000
SEED = 42
CHUNK_SIZE = 20


def load_slashdot(path):
    graph = nx.Graph()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v, label = line.split()
            graph.add_edge(int(u), int(v), sign=int(label))
    return graph


# --------------------------------------------------------------- workers ---
# Set as plain module-level globals (not via Pool initializer/initargs,
# which pickles+pipes its arguments to each worker even under fork). Fork
# inherits process memory via copy-on-write at the moment Pool() is
# constructed, so setting these BEFORE that call gives every worker access
# for free, with no serialization of the graph/edge list at all.
_G = None
_EDGES = None


def _set_globals(g, edges):
    global _G, _EDGES
    _G, _EDGES = g, edges


def _process_anchor(edge_id, max_dist):
    """Identical logic to her sign_distance_correlation's per-anchor body:
    single BFS queue seeded from both endpoints, record only tree/discovery
    edges, bucket by 1-indexed distance."""
    g = _G
    v1, v2, data = _EDGES[edge_id]
    sign = data["sign"]
    local = [[] for _ in range(max_dist)]  # local[d-1] = list of (anchor_sign, context_sign)
    d = {v1: 0, v2: 0} if v1 != v2 else {v1: 0}
    q = deque([v1, v2]) if v1 != v2 else deque([v1])
    while q:
        v = q.popleft()
        if d[v] >= max_dist:
            continue
        for n in g.neighbors(v):
            if n not in d:
                d[n] = d[v] + 1
                q.append(n)
                local[d[n] - 1].append((sign, g.get_edge_data(v, n)["sign"]))
    return local


def _worker_chunk(edge_id_chunk):
    max_dist = MAX_DIST
    out = [[] for _ in range(max_dist)]
    for eid in edge_id_chunk:
        local = _process_anchor(eid, max_dist)
        for i in range(max_dist):
            out[i].extend(local[i])
    return out


def main():
    t0 = time.time()
    g = load_slashdot(DATA_PATH)
    print(f"loaded slashdot090221 (undirected): {g.number_of_nodes():,} nodes, "
          f"{g.number_of_edges():,} edges  ({time.time()-t0:.1f}s)", flush=True)

    random.seed(SEED)
    # [perf bug fix #1] build the edge list ONCE, not once per anchor.
    edges = list(g.edges(data=True))
    edge_numbers = random.sample(range(0, len(edges)), N_ANCHORS)

    chunks = [edge_numbers[i:i + CHUNK_SIZE] for i in range(0, len(edge_numbers), CHUNK_SIZE)]
    n_workers = mp.cpu_count()
    print(f"dispatching {len(edge_numbers):,} anchors in {len(chunks)} chunks "
          f"across {n_workers} workers...", flush=True)

    _set_globals(g, edges)
    coor = [[[], []] for _ in range(MAX_DIST)]
    t1 = time.time()
    done = 0
    with mp.get_context("fork").Pool(processes=n_workers) as pool:
        for chunk_result in pool.imap_unordered(_worker_chunk, chunks):
            for i in range(MAX_DIST):
                for a_sign, c_sign in chunk_result[i]:
                    coor[i][0].append(a_sign)
                    coor[i][1].append(c_sign)
            done += 1
            if done % max(1, len(chunks) // 20) == 0 or done == len(chunks):
                elapsed = time.time() - t1
                eta = elapsed / done * (len(chunks) - done)
                print(f"  chunk {done}/{len(chunks)}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    print(f"BFS phase done: {time.time()-t1:.0f}s", flush=True)
    n_precap = [len(x) for x, y in coor]

    # capping step -- unchanged from her script, sequential, main process,
    # continuing the SAME seed=42 random stream (see module docstring).
    for x, y in coor:
        if len(x) > MAX_PAIRS:
            keep = sorted(random.sample(range(len(x)), MAX_PAIRS))
            kept_x = [x[i] for i in keep]
            kept_y = [y[i] for i in keep]
            x[:] = kept_x
            y[:] = kept_y
    n_postcap = [len(x) for x, y in coor]
    correlations = [numpy.corrcoef(x, y)[0, 1] if len(x) > 1 else float("nan") for x, y in coor]
    distances = list(range(1, MAX_DIST + 1))

    print(f"\n{'d':>2} {'n_pairs(pre-cap)':>18} {'n_pairs(used)':>14} {'phi (=corrcoef)':>16}")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    import csv
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line_dist", "n_pairs_precap", "n_pairs_used", "phi"])
        for d, npre, npost, c in zip(distances, n_precap, n_postcap, correlations):
            print(f"{d:2d} {npre:>18,} {npost:>14,} {c:>16.6f}")
            w.writerow([d, npre, npost, c])
    print(f"\nwrote {OUT_CSV}")

    plt.plot(distances, correlations, marker="o")
    plt.xlabel("BFS distance from edge")
    plt.ylabel("Correlation with edge sign (phi)")
    plt.title("Sign correlation vs. BFS distance (Slashdot) -- her v2, undirected,\n"
              "10k random anchors, seed=42, parallelized")
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    print(f"total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
