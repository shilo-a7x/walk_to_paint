"""
Diagnostic-only raw-pair extractor for the Panel B "bump" investigation
(see PANELB_INVESTIGATION_REPORT.md at repo root for the full writeup).

Collects every (anchor_edge, context_edge) pair discovered by undirected BFS
out to D_MAX on slashdot090221, at reduced scale (3,000 anchors, capped at
4,000 candidate context edges per anchor) so the raw pairs fit in memory and
the sweep completes in under a minute -- this is NOT a byte-for-byte
reproduction of the full-scale production Panel B numbers
(aaai2027/figure_data/empconf_panelB_*.csv, which sweep all ~549K anchors),
it answers the METHODOLOGICAL question (does deduping/proper cluster
bootstrap change the conclusion), using the real production loader
(src/data/datasets.py::get_loader) and the same undirected/all-edges
convention as production throughout.

Output columns (saved as a single int64 array): line_dist, anchor_sign,
context_sign, context_edge_id, pair_key (= min(u,v)*N + max(u,v), a
single-int encoding of the undirected node pair so reciprocal u->v/v->u
edges collapse to the same key).

Run: .venv/bin/python scripts/panelb_diagnostics/extract_panelb_raw_rows.py
Writes: <OUT_DIR>/panelb_raw_rows.npy (~12M rows, ~300MB) -- consumed by
proper_cluster_bootstrap.py in this same directory.
"""
import os
import sys
import time
import multiprocessing as mp

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.data.datasets import get_loader

DS_NAME = "slashdot090221"
D_MAX = 6
SEED = 42
N_ANCHORS = 3000
MAX_CAND_PER_ANCHOR = 4000

OUT_DIR = os.path.join(ROOT, "outputs", "panelb_diagnostics")


def load_graph():
    cfg = load_config(overrides=[f"dataset.name={DS_NAME}"])
    edges = get_loader(DS_NAME)(cfg)
    nodes = sorted({n for u, v, s in edges for n in (u, v)})
    n2i = {n: i for i, n in enumerate(nodes)}
    N, E = len(nodes), len(edges)
    src = np.array([n2i[u] for u, v, s in edges], dtype=np.int32)
    dst = np.array([n2i[v] for u, v, s in edges], dtype=np.int32)
    sign = np.array([1 if s > 0 else 0 for u, v, s in edges], dtype=np.int8)

    full = [[] for _ in range(N)]
    for eid in range(E):
        u, v = int(src[eid]), int(dst[eid])
        full[u].append((v, eid))
        full[v].append((u, eid))
    return N, E, src, dst, sign, full


# --------------------------------------------------------------- workers ---
_N = _E = _src = _dst = _sign = _adj = None


def _init_globals(N, E, src, dst, sign, adj):
    global _N, _E, _src, _dst, _sign, _adj
    _N, _E, _src, _dst, _sign, _adj = N, E, src, dst, sign, adj


def _process_anchor(aid, shell, gen, cur_gen, rng):
    d_max = D_MAX
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

    cand = list(cand)
    if len(cand) > MAX_CAND_PER_ANCHOR:
        idx = rng.choice(len(cand), MAX_CAND_PER_ANCHOR, replace=False)
        cand = [cand[i] for i in idx]

    N = _N
    rows = []  # (line_d, anchor_sign, context_sign, context_edge_id, pair_key)
    for eid2 in cand:
        x2, y2, s2 = int(src[eid2]), int(dst[eid2]), int(sign[eid2])
        sh_x = int(shell[x2]) if gen[x2] == cur_gen else d_max + 1
        sh_y = int(shell[y2]) if gen[y2] == cur_gen else d_max + 1
        d_e = min(sh_x, sh_y)
        if d_e >= d_max:
            continue
        line_d = d_e + 1
        pair_key = min(x2, y2) * N + max(x2, y2)  # single int64-safe encoding
        rows.append((line_d, y, s2, eid2, pair_key))
    return rows


def _worker_chunk(anchor_chunk):
    N = _N
    shell = np.full(N, -1, np.int32)
    gen = np.zeros(N, np.int32)
    out = []
    for cur_gen, aid in enumerate(anchor_chunk, start=1):
        rng = np.random.default_rng(SEED * 1000003 + int(aid))
        out.extend(_process_anchor(int(aid), shell, gen, cur_gen, rng))
    return out


def main():
    t0 = time.time()
    N, E, src, dst, sign, full = load_graph()
    print(f"loaded {DS_NAME}: N={N:,} E={E:,}  ({time.time()-t0:.1f}s)", flush=True)

    rng = np.random.default_rng(SEED)
    anchor_ids = rng.choice(E, min(N_ANCHORS, E), replace=False)

    _init_globals(N, E, src, dst, sign, full)
    chunk_size = 50
    chunks = [anchor_ids[i:i + chunk_size] for i in range(0, len(anchor_ids), chunk_size)]

    all_rows = []
    t1 = time.time()
    with mp.get_context("fork").Pool(processes=mp.cpu_count()) as pool:
        for i, rows in enumerate(pool.imap_unordered(_worker_chunk, chunks), 1):
            all_rows.extend(rows)
            if i % max(1, len(chunks) // 10) == 0:
                print(f"  chunk {i}/{len(chunks)}  elapsed={time.time()-t1:.0f}s  "
                      f"rows_so_far={len(all_rows):,}", flush=True)
    print(f"BFS+gather done: {len(all_rows):,} raw rows, {time.time()-t1:.0f}s", flush=True)

    arr = np.array(all_rows, dtype=np.int64)
    del all_rows
    print(f"array built. total rows={len(arr):,}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "panelb_raw_rows.npy"), arr)
    print(f"saved raw rows to {OUT_DIR}/panelb_raw_rows.npy  total_time={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
