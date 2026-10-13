"""Is Panel B's far-hop NMI bump a pair-reuse artifact?

Panel B counts every (anchor edge, context edge) pair once, so at far hops -- where few distinct
context edges remain -- one context edge is counted once per anchor that reaches it and a handful
of edges can dominate the contingency table (PAPER_CLOSEOUT_LOG.md 2026-07-28: top 10 distinct
edges ~40% of tail pairs on bitcoin-alpha). Per hop, this script reports, on the exact Panel B pair
definition and anchor sample:
  naive_nmi       Panel B's NMI (each pair counted once) -- reproduces the paper CSV
  dedup_nmi       each distinct context edge contributes total weight 1 (its anchors' sign
                  distribution normalised), so heavily reused edges cannot dominate
  n_distinct      distinct context edges at that hop; top10_share = their share of all pairs
  perm_p          cluster-level permutation test: context-edge signs permuted across the distinct
                  context edges at that hop (each edge keeps its anchor counts), 500 draws; p =
                  fraction of permuted naive NMI >= observed. Respects the reuse structure, unlike
                  the naive table's implicit independent-pairs assumption.
  perm_null_mean  mean permuted NMI (the estimator floor under reuse)

Usage:
  .venv/bin/python scripts/panelB_bump_dedup.py --direction undirected --max-anchors 20000
  .venv/bin/python scripts/panelB_bump_dedup.py --direction directed --max-anchors 10000000
"""
import argparse
import csv
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.balance_theory_paths import load_edges_canonical, DATASET_CONFIGS  # noqa: E402
import scripts.topology_sign_mi_vs_distance as base  # noqa: E402  (_G, _neighbors)

CHUNK_SIZE = 50
N_PERM = 500
OUT_DIR = "outputs/topology_mi_vs_distance"


def _worker_chunk(anchors):
    """Per hop and context edge: how many anchors of each sign reach it (counts[d, c, s_anchor])."""
    g = base._G
    d_max, N, E = g["d_max"], g["N"], g["E"]
    ex, ey, s = g["ex"], g["ey"], g["s"]
    counts = np.zeros((d_max + 2, E, 2), np.int32)
    shell = np.zeros(N, np.int32)
    gen = np.zeros(N, np.int32)
    for cur, a in enumerate(anchors, start=1):
        frontier = np.unique(np.array([ex[a], ey[a]], np.int64))
        gen[frontier] = cur
        shell[frontier] = 0
        reached = [frontier]
        for d in range(1, d_max + 1):
            nb, _ = base._neighbors(frontier)
            nb = np.unique(nb)
            nb = nb[gen[nb] != cur]
            if nb.size == 0:
                break
            gen[nb] = cur
            shell[nb] = d
            reached.append(nb)
            frontier = nb
        _, ce = base._neighbors(np.concatenate(reached))
        ce = np.unique(ce)
        ce = ce[ce != a]
        cx, cy = ex[ce], ey[ce]
        de = np.minimum(np.where(gen[cx] == cur, shell[cx], d_max + 1),
                        np.where(gen[cy] == cur, shell[cy], d_max + 1))
        keep = de <= d_max
        np.add.at(counts, (de[keep] + 1, ce[keep], s[a]), 1)
    return counts


def nmi_2x2(c):
    n = c.sum()
    if n <= 0:
        return float("nan")
    p = c / n
    py, ps = p.sum(1), p.sum(0)
    mi = sum(p[i, j] * math.log2(p[i, j] / (py[i] * ps[j])) for i in range(2) for j in range(2) if p[i, j] > 0)
    hy = -sum(q * math.log2(q) for q in py if q > 0)
    return mi / hy if hy > 0 else float("nan")


def table(cnt, sc, w=None):
    """cnt: (n_ctx, 2) anchor-sign counts per context edge; sc: context-edge signs (0/1)."""
    if w is not None:
        cnt = cnt * w[:, None]
    t = np.zeros((2, 2))
    for s_c in (0, 1):
        t[:, s_c] = cnt[sc == s_c].sum(0)
    return t


def analyse(ds, d_max, max_anchors, workers, seed, direction):
    t0 = time.time()
    raw = load_edges_canonical(DATASET_CONFIGS[ds]["ds_name"])
    nodes = sorted({n for e in raw for n in e[:2]})
    n2i = {n: i for i, n in enumerate(nodes)}
    N, E = len(nodes), len(raw)
    ex = np.array([n2i[e[0]] for e in raw], np.int64)
    ey = np.array([n2i[e[1]] for e in raw], np.int64)
    s = np.array([1 if e[2] > 0 else 0 for e in raw], np.int64)
    if direction == "directed":
        src, dst, eids = ex, ey, np.arange(E)
    else:
        src, dst = np.concatenate([ex, ey]), np.concatenate([ey, ex])
        eids = np.concatenate([np.arange(E), np.arange(E)])
    order = np.argsort(src, kind="stable")
    indptr = np.zeros(N + 1, np.int64)
    np.add.at(indptr, src + 1, 1)
    base._G.clear()
    base._G.update(dict(indptr=np.cumsum(indptr), nbr=dst[order], eid=eids[order], ex=ex, ey=ey, s=s,
                        d_max=d_max, N=N, E=E))
    rng = np.random.default_rng(seed)
    anchors = np.arange(E) if E <= max_anchors else rng.choice(E, max_anchors, replace=False)
    size = max(CHUNK_SIZE, len(anchors) // (workers * 4))  # few big chunks: each returns an (hops x E x 2) array
    chunks = [anchors[i:i + size] for i in range(0, len(anchors), size)]
    counts = np.zeros((d_max + 2, E, 2), np.int64)
    with mp.get_context("fork").Pool(workers) as pool:
        for c in pool.imap_unordered(_worker_chunk, chunks):
            counts += c
    print(f"\n{ds} ({direction}): N={N:,} E={E:,} anchors={len(anchors):,}  [{time.time() - t0:.0f}s]", flush=True)

    prng = np.random.default_rng(seed + 7)
    rows = []
    for d in range(1, d_max + 2):
        cnt = counts[d]
        n_c = cnt.sum(1)
        live = n_c > 0
        cnt, sc, n_c = cnt[live], s[live], n_c[live]
        n_pairs = int(n_c.sum())
        if n_pairs < 2000:
            rows.append(dict(dataset=ds, direction=direction, line_dist=d, n_pairs=n_pairs, n_distinct=int(live.sum())))
            continue
        naive = nmi_2x2(table(cnt, sc))
        dedup = nmi_2x2(table(cnt, sc, 1.0 / n_c))
        top10 = float(np.sort(n_c)[-10:].sum() / n_pairs)
        perm = np.array([nmi_2x2(table(cnt, prng.permutation(sc))) for _ in range(N_PERM)])
        p = float((1 + (perm >= naive).sum()) / (1 + N_PERM))
        rows.append(dict(dataset=ds, direction=direction, line_dist=d, n_pairs=n_pairs, n_distinct=int(live.sum()),
                         top10_share=top10, naive_nmi=naive, dedup_nmi=dedup,
                         perm_null_mean=float(perm.mean()), perm_p=p))
        print(f"  hop {d}: pairs={n_pairs:>14,} distinct={int(live.sum()):>8,} top10={top10:.3f}  "
              f"naive NMI={naive:.2e}  dedup NMI={dedup:.2e}  perm null={perm.mean():.2e}  p={p:.3f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["all"])
    ap.add_argument("--direction", choices=["undirected", "directed"], default="undirected")
    ap.add_argument("--d-max", type=int, default=7)
    ap.add_argument("--max-anchors", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    datasets = [d for d in DATASET_CONFIGS if d != "synthetic-fog"] if args.datasets == ["all"] else args.datasets
    out = os.path.join(OUT_DIR, f"panelB_bump_dedup_{args.direction}" + ("_exact" if args.max_anchors >= 10**6 else "") + ".csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for ds in datasets:
        rows += analyse(ds, args.d_max, args.max_anchors, args.workers, args.seed, args.direction)
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
