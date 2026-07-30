"""Ablation grid isolating exactly which factor(s) explain the gap between
the colleague's load_slashdot.py result ("close to zero") and our own
production Panel B numbers on slashdot090221.

Three binary factors, run as a full 2x2x2 = 8-cell grid:
  - direction:  "directed" (successors only, her bug #1) vs "undirected"
                (both directions, our convention)
  - recording:  "tree" (one edge per BFS-discovered node, her bug #2) vs
                "all" (every edge among all reached nodes, our convention)
  - sampling:   "small_nonrandom" (first 1000 edges in loader order, her
                convention) vs "large_random" (seeded random sample, ours)

Every cell reports BOTH NMI and phi/z/p from the SAME pooled 2x2
contingency table per distance -- there is no separate "run with MI"
config, since both statistics are just two different summaries of the
identical counts (this directly answers "does the statistic choice
matter" for every cell at once, not just one of them).

Cell (directed, tree, small_nonrandom) is her exact method -- reimplemented
here (not reusing her networkx-based script) so all 8 cells go through the
identical counting/statistics code and are only apples-to-apples if that
one cell's numbers are cross-checked against her real script's actual
output (done separately, see chat).

Cell (undirected, all, large_random) IS our existing production method --
reuses the already-published aaai2027/figure_data/empconf_panelB_*.csv
numbers for slashdot rather than recomputing them.

Uses the real production loader (src/data/datasets.py::get_loader), not
the balance_theory_paths wrapper, per standing instruction.
"""
import csv
import math
import multiprocessing as mp
import os
import sys
import time
from collections import deque

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.utils.config import load_config
from src.data.datasets import get_loader

DS_NAME = "slashdot090221"
D_MAX = 8
SEED = 42
SMALL_N = 1000            # her anchor count, non-random (first N in loader order)
LARGE_N_TREE_DIRECTED = 3000    # directed+tree is cheap (confirmed ~110s/1000 anchors)
# undirected+tree is MUCH more expensive per anchor (undirected BFS reaches a
# far bigger frontier -- first attempt at 1000/3000 anchors ran >87min with no
# sign of finishing before being killed; capped much lower here so this
# actually completes, at the cost of a noisier estimate for these 2 cells only)
SMALL_N_UNDIRECTED_TREE = 150
LARGE_N_UNDIRECTED_TREE = 300
LARGE_N_ALL = 20000        # random anchors for the fast (all-edges) path, matches production
MIN_PAIRS = 2000
PROGRESS_EVERY = 25

OUT_CSV = "aaai2027/figure_data/ablation_slashdot_colleague_vs_ours.csv"
PROD_CORR_CSV = "aaai2027/figure_data/empconf_panelB_correlation_check.csv"


def mi_from_cont(c):
    n = int(c.sum())
    if n < MIN_PAIRS:
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


def phi_from_cont(c):
    n = int(c.sum())
    if n < MIN_PAIRS:
        return float("nan")
    n00, n01 = c[0, 0], c[0, 1]
    n10, n11 = c[1, 0], c[1, 1]
    row0, row1 = n00 + n01, n10 + n11
    col0, col1 = n00 + n10, n01 + n11
    denom = math.sqrt(float(row0) * row1 * col0 * col1)
    if denom == 0:
        return float("nan")
    return (n11 * n00 - n10 * n01) / denom


def load_graph():
    cfg = load_config(overrides=[f"dataset.name={DS_NAME}"])
    edges = get_loader(DS_NAME)(cfg)
    nodes = sorted({n for u, v, s in edges for n in (u, v)})
    n2i = {n: i for i, n in enumerate(nodes)}
    N, E = len(nodes), len(edges)
    src = np.array([n2i[u] for u, v, s in edges], dtype=np.int32)
    dst = np.array([n2i[v] for u, v, s in edges], dtype=np.int32)
    sign = np.array([1 if s > 0 else 0 for u, v, s in edges], dtype=np.int8)

    succ = [[] for _ in range(N)]      # out-neighbors only (directed)
    full = [[] for _ in range(N)]      # both directions (undirected)
    for eid in range(E):
        u, v, s = int(src[eid]), int(dst[eid]), int(sign[eid])
        succ[u].append((v, eid))
        full[u].append((v, eid))
        full[v].append((u, eid))
    return N, E, src, dst, sign, succ, full


# ---------------------------------------------------------------------------
# Fast path: "all edges of every reached node" recording -- multiprocessed
# (single-threaded, this config is the ~13-CPU-hour version of what the
# production extract script does in ~365s wall-clock via 128 workers; a
# first sequential attempt at this ran >87min with no end in sight before
# being killed -- this mirrors that script's exact Pool/fork pattern instead
# of re-learning that lesson the slow way twice)
# ---------------------------------------------------------------------------
_N = _E = _src = _dst = _sign = _adj = _D_MAX = None
CHUNK_SIZE = 50


def _set_all_edges_graph(N, E, src, dst, sign, adj, d_max):
    global _N, _E, _src, _dst, _sign, _adj, _D_MAX
    _N, _E, _src, _dst, _sign, _adj, _D_MAX = N, E, src, dst, sign, adj, d_max


def _process_anchor_all(aid, shell, gen, cur_gen):
    d_max = _D_MAX
    src, dst, sign, adj = _src, _dst, _sign, _adj
    u, v, y = int(src[aid]), int(dst[aid]), int(sign[aid])
    roots = (u, v) if u != v else (u,)
    reached = []
    for w in roots:
        if gen[w] != cur_gen:
            gen[w] = cur_gen
            shell[w] = 0
            reached.append(w)
    frontier = list(roots)
    for d in range(1, d_max):
        nxt = []
        for w in frontier:
            for nb, _eid in adj[w]:
                if gen[nb] != cur_gen:
                    gen[nb] = cur_gen
                    shell[nb] = d
                    nxt.append(nb)
        if not nxt:
            break
        reached.extend(nxt)
        frontier = nxt

    cand = set()
    for w in reached:
        for _nb, eid2 in adj[w]:
            if eid2 != aid:
                cand.add(eid2)

    local = {}
    for eid2 in cand:
        x2, y2, s2 = int(src[eid2]), int(dst[eid2]), int(sign[eid2])
        sh_x = int(shell[x2]) if gen[x2] == cur_gen else d_max + 1
        sh_y = int(shell[y2]) if gen[y2] == cur_gen else d_max + 1
        d_e = min(sh_x, sh_y)
        if d_e >= d_max:
            continue
        line_d = d_e + 1
        key = (line_d, y, s2)
        local[key] = local.get(key, 0) + 1
    return local


def _worker_chunk_all(anchor_chunk):
    N = _N
    shell = np.full(N, -1, np.int32)
    gen = np.zeros(N, np.int32)
    totals = {}
    for cur_gen, aid in enumerate(anchor_chunk, start=1):
        local = _process_anchor_all(int(aid), shell, gen, cur_gen)
        for key, cnt in local.items():
            totals[key] = totals.get(key, 0) + cnt
    return totals


def run_all_edges(direction, anchor_ids, N, E, src, dst, sign, succ, full, d_max=D_MAX, workers=None):
    adj = succ if direction == "directed" else full
    conts = {d: np.zeros((2, 2), dtype=np.int64) for d in range(1, d_max + 1)}
    _set_all_edges_graph(N, E, src, dst, sign, adj, d_max)

    chunks = [anchor_ids[i:i + CHUNK_SIZE] for i in range(0, len(anchor_ids), CHUNK_SIZE)]
    n_workers = workers or mp.cpu_count()
    t0 = time.time()
    done = 0
    with mp.get_context("fork").Pool(processes=n_workers) as pool:
        for totals in pool.imap_unordered(_worker_chunk_all, chunks):
            for (line_d, yi, s2i), cnt in totals.items():
                conts[line_d][yi, s2i] += cnt
            done += 1
            if done % max(1, len(chunks) // 10) == 0 or done == len(chunks):
                elapsed = time.time() - t0
                eta = elapsed / done * (len(chunks) - done)
                print(f"    chunk {done}/{len(chunks)} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
    return conts


# ---------------------------------------------------------------------------
# Slow path: "tree discovery edge only" recording (her algorithm's logic)
# ---------------------------------------------------------------------------

def run_tree_edges(direction, anchor_ids, N, E, src, dst, sign, succ, full, d_max=D_MAX):
    adj = succ if direction == "directed" else full
    conts = {d: np.zeros((2, 2), dtype=np.int64) for d in range(1, d_max + 1)}
    t0 = time.time()

    for ai, aid in enumerate(anchor_ids, start=1):
        if ai % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            eta = elapsed / ai * (len(anchor_ids) - ai)
            print(f"    [{ai}/{len(anchor_ids)}] elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
        u, v, y = int(src[aid]), int(dst[aid]), int(sign[aid])
        depth = {u: 0, v: 0}
        q = deque([u, v]) if u != v else deque([u])
        while q:
            w = q.popleft()
            dw = depth[w]
            if dw >= d_max:
                continue
            for nb, eid2 in adj[w]:
                if nb not in depth:
                    depth[nb] = dw + 1
                    q.append(nb)
                    if eid2 != aid:
                        s2 = int(sign[eid2])
                        conts[dw + 1][y, s2] += 1
    return conts


def get_anchor_ids(direction, recording, sampling, E, rng_seed=SEED):
    if sampling == "small_nonrandom":
        n = SMALL_N_UNDIRECTED_TREE if (direction == "undirected" and recording == "tree") else SMALL_N
        return np.arange(min(n, E))
    elif sampling == "large_random_tree":
        n = LARGE_N_UNDIRECTED_TREE if direction == "undirected" else LARGE_N_TREE_DIRECTED
        rng = np.random.default_rng(rng_seed)
        return rng.choice(E, min(n, E), replace=False)
    elif sampling == "large_random_all":
        rng = np.random.default_rng(rng_seed)
        return rng.choice(E, min(LARGE_N_ALL, E), replace=False)
    raise ValueError(sampling)


def report(conts, d_max=D_MAX):
    rows = []
    for d in range(1, d_max + 1):
        c = conts[d]
        mi, nmi, n = mi_from_cont(c)
        phi = phi_from_cont(c)
        rows.append((d, n, mi, nmi, phi))
    return rows


def load_production_slashdot():
    rows = {}
    with open(PROD_CORR_CSV) as f:
        for row in csv.DictReader(f):
            if row["dataset"] == "slashdot":
                d = int(row["line_dist"])
                rows[d] = (int(row["n_pairs"]), float(row["mi_bits"]), float(row["nmi"]),
                           float(row["phi"]))
    return rows


def main():
    t0 = time.time()
    N, E, src, dst, sign, succ, full = load_graph()
    print(f"loaded {DS_NAME}: N={N:,} E={E:,}  ({time.time()-t0:.1f}s)")

    configs = [
        ("directed",   "tree", "small_nonrandom"),
        ("directed",   "tree", "large_random_tree"),
        ("directed",   "all",  "small_nonrandom"),
        ("directed",   "all",  "large_random_all"),
        ("undirected", "tree", "small_nonrandom"),
        ("undirected", "tree", "large_random_tree"),
        ("undirected", "all",  "small_nonrandom"),
        # (undirected, all, large_random_all) == our existing production
        # result -- reused below instead of recomputed.
    ]

    all_rows = []
    for direction, recording, sampling in configs:
        tcfg = time.time()
        print(f"\n>>> starting {direction} / {recording} / {sampling} ...", flush=True)
        anchor_ids = get_anchor_ids(direction, recording, sampling, E)
        if recording == "all":
            conts = run_all_edges(direction, anchor_ids, N, E, src, dst, sign, succ, full)
        else:
            conts = run_tree_edges(direction, anchor_ids, N, E, src, dst, sign, succ, full)
        rows = report(conts)
        elapsed = time.time() - tcfg
        print(f"\n=== {direction} / {recording} / {sampling}  "
              f"(n_anchors={len(anchor_ids):,}, {elapsed:.0f}s) ===", flush=True)
        print(f"{'d':>2} {'n_pairs':>12} {'mi_bits':>12} {'nmi':>12} {'phi':>10}")
        for d, n, mi, nmi, phi in rows:
            mi_s = f"{mi:.8f}" if not math.isnan(mi) else "nan"
            nmi_s = f"{nmi:.6f}" if not math.isnan(nmi) else "nan"
            phi_s = f"{phi:.5f}" if not math.isnan(phi) else "nan"
            print(f"{d:2d} {n:>12,} {mi_s:>12} {nmi_s:>12} {phi_s:>10}")
            all_rows.append({"direction": direction, "recording": recording,
                             "sampling": sampling, "n_anchors": len(anchor_ids),
                             "line_dist": d, "n_pairs": n, "mi_bits": mi, "nmi": nmi, "phi": phi})

    prod = load_production_slashdot()
    print(f"\n=== reference: our existing production result (undirected/all/20000 random) ===")
    print(f"{'d':>2} {'n_pairs':>12} {'mi_bits':>12} {'nmi':>12} {'phi':>10}")
    for d in sorted(prod):
        n, mi, nmi, phi = prod[d]
        print(f"{d:2d} {n:>12,} {mi:.8f} {nmi:.6f} {phi:.5f}")
        all_rows.append({"direction": "undirected", "recording": "all",
                         "sampling": "large_random_all(PRODUCTION)", "n_anchors": 20000,
                         "line_dist": d, "n_pairs": n, "mi_bits": mi, "nmi": nmi, "phi": phi})

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["direction", "recording", "sampling", "n_anchors",
                                          "line_dist", "n_pairs", "mi_bits", "nmi", "phi"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {OUT_CSV}")
    print(f"total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
