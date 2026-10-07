"""Panel B with topology distance instead of hop distance, plus the hop x topology consistency check.

Each edge e=(x->y) gets t(e) = log1p([outdeg(x), indeg(x), outdeg(y), indeg(y)])
(src_out, src_in, tgt_out, tgt_in; structure only, signs never used).

Part A -- sign NMI/phi vs topology distance. Random edge pairs, no graph-distance restriction
(20k anchor edges x 2000 random partner edges). Pairs are binned by ||t(a) - t(c)|| into
quantile bins; per bin, NMI/phi between the two edges' REAL signs -- same y as Panel B, x-axis
is topology distance instead of hops.

Part B -- consistency check. Panel B's own pairs (same line-graph hop definition and 20k-anchor
sample as scripts/paper_figures/extract_empconf_panelB_mi_decay_linegraph.py), split by hop AND
by Part A's topology-distance bins. If, inside a fixed topology bin, sign NMI is flat across hops,
the hop curve (bumps included) is a mixture of topology distances.

Both parts also report a null: the same statistic with signs randomly permuted over edges
(structure and sign balance fixed), i.e. the estimator's small-sample floor.

Usage:
  .venv/bin/python scripts/topology_sign_mi_vs_topodist.py --workers 48
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.balance_theory_paths import load_edges_canonical, DATASET_CONFIGS  # noqa: E402
import scripts.topology_sign_mi_vs_distance as base  # noqa: E402  (_G graph state, _neighbors, _nmi_phi_2x2)

N_TBINS = 12
RAND_ANCHORS = 20000
RAND_PARTNERS = 2000
CHUNK_SIZE = 50
OUT_DIR = "outputs/topology_mi_vs_distance"


def _worker_chunk(anchors):
    """Panel B BFS (copied logic of base._worker_chunk) accumulating a hop x topology-bin grid
    of sign pairs, for the real and the permuted (null) signs."""
    g = base._G
    d_max, N = g["d_max"], g["N"]
    ex, ey, s, s0, T, qedges = g["ex"], g["ey"], g["s"], g["s_null"], g["T"], g["qedges"]
    n_d = d_max + 2
    grid = np.zeros((n_d, N_TBINS, 2, 2), np.int64)
    grid0 = np.zeros((n_d, N_TBINS, 2, 2), np.int64)
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
        ce, line_d = ce[keep], de[keep] + 1
        tb = np.searchsorted(qedges, np.linalg.norm(T[ce] - T[a], axis=1), side="right")
        np.add.at(grid, (line_d, tb, s[a], s[ce]), 1)
        np.add.at(grid0, (line_d, tb, s0[a], s0[ce]), 1)
    return grid, grid0


def analyse(ds, d_max, max_anchors, workers, seed, direction="undirected"):
    t0 = time.time()
    raw = load_edges_canonical(DATASET_CONFIGS[ds]["ds_name"])
    nodes = sorted({n for e in raw for n in e[:2]})
    n2i = {n: i for i, n in enumerate(nodes)}
    N, E = len(nodes), len(raw)
    ex = np.array([n2i[e[0]] for e in raw], np.int64)
    ey = np.array([n2i[e[1]] for e in raw], np.int64)
    s = np.array([1 if e[2] > 0 else 0 for e in raw], np.int64)
    s_null = s[np.random.default_rng(seed + 1).permutation(E)].copy()
    outdeg, indeg = np.bincount(ex, minlength=N), np.bincount(ey, minlength=N)
    T = np.log1p(np.stack([outdeg[ex], indeg[ex], outdeg[ey], indeg[ey]], 1).astype(np.float64))

    # ---- Part A: random edge pairs, binned by topology distance
    rng = np.random.default_rng(seed)
    ra = rng.choice(E, min(RAND_ANCHORS, E), replace=False)
    D, sa, sc, sa0, sc0 = [], [], [], [], []
    for i in range(0, len(ra), 1000):
        a = ra[i:i + 1000]
        c = rng.integers(0, E, (len(a), RAND_PARTNERS))
        D.append(np.linalg.norm(T[c] - T[a][:, None, :], axis=2).ravel())
        sa.append(np.repeat(s[a], RAND_PARTNERS))
        sc.append(s[c].ravel())
        sa0.append(np.repeat(s_null[a], RAND_PARTNERS))
        sc0.append(s_null[c].ravel())
    D, sa, sc, sa0, sc0 = map(np.concatenate, (D, sa, sc, sa0, sc0))
    qedges = np.unique(np.quantile(D, np.linspace(0, 1, N_TBINS + 1)[1:-1]))
    tb = np.searchsorted(qedges, D, side="right")
    print(f"\n{ds}: N={N:,} E={E:,} random pairs={len(D):,} topology bins={len(qedges) + 1}", flush=True)
    rows_a = []
    for b in range(len(qedges) + 1):
        m = tb == b
        c = np.zeros((2, 2), np.int64)
        np.add.at(c, (sa[m], sc[m]), 1)
        c0 = np.zeros((2, 2), np.int64)
        np.add.at(c0, (sa0[m], sc0[m]), 1)
        nmi, phi, n = base._nmi_phi_2x2(c)
        nmi0, phi0, _ = base._nmi_phi_2x2(c0)
        rows_a.append(dict(dataset=ds, topo_bin=b, d_lo=float(D[m].min()), d_hi=float(D[m].max()),
                           d_median=float(np.median(D[m])), n_pairs=n, nmi=nmi, phi=phi,
                           null_nmi=nmi0, null_phi=phi0))
        print(f"  A bin {b:>2} dist [{D[m].min():.2f},{D[m].max():.2f}] n={n:>11,} "
              f"NMI={nmi:.2e} phi={phi:+.4f} | null NMI={nmi0:.1e}", flush=True)
    del D, sa, sc, sa0, sc0, tb

    # ---- Part B: Panel B hop pairs, split by topology bin
    if direction == "directed":  # out-neighbours only, as Panel B's --direction directed
        src, dst, eids = ex, ey, np.arange(E)
    else:
        src, dst = np.concatenate([ex, ey]), np.concatenate([ey, ex])
        eids = np.concatenate([np.arange(E), np.arange(E)])
    order = np.argsort(src, kind="stable")
    indptr = np.zeros(N + 1, np.int64)
    np.add.at(indptr, src + 1, 1)
    base._G.clear()
    base._G.update(dict(indptr=np.cumsum(indptr), nbr=dst[order], eid=eids[order], ex=ex, ey=ey, s=s,
                        s_null=s_null, T=T, qedges=qedges, d_max=d_max, N=N))
    anchors = np.arange(E) if E <= max_anchors else np.random.default_rng(seed).choice(E, max_anchors, replace=False)
    chunks = [anchors[i:i + CHUNK_SIZE] for i in range(0, len(anchors), CHUNK_SIZE)]
    n_d = d_max + 2
    grid = np.zeros((n_d, N_TBINS, 2, 2), np.int64)
    grid0 = np.zeros((n_d, N_TBINS, 2, 2), np.int64)
    with mp.get_context("fork").Pool(workers) as pool:
        for g, g0 in pool.imap_unordered(_worker_chunk, chunks):
            grid += g
            grid0 += g0
    rows_b = []
    for d in range(1, d_max + 2):
        for b in range(len(qedges) + 1):
            nmi, phi, n = base._nmi_phi_2x2(grid[d, b])
            nmi0, _, _ = base._nmi_phi_2x2(grid0[d, b])
            rows_b.append(dict(dataset=ds, line_dist=d, topo_bin=b, n_pairs=n, nmi=nmi, phi=phi, null_nmi=nmi0))
        tot, _, n = base._nmi_phi_2x2(grid[d].sum(0))
        print(f"  B hop {d}: n={n:>14,} hop-only NMI={tot:.2e}", flush=True)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)
    return rows_a, rows_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["all"])
    ap.add_argument("--d-max", type=int, default=7)
    ap.add_argument("--max-anchors", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--direction", choices=["undirected", "directed"], default="undirected")
    args = ap.parse_args()
    datasets = [d for d in DATASET_CONFIGS if d != "synthetic-fog"] if args.datasets == ["all"] else args.datasets
    os.makedirs(OUT_DIR, exist_ok=True)
    out_a = os.path.join(OUT_DIR, "sign_nmi_vs_topodist.csv")
    out_b = os.path.join(OUT_DIR, "sign_nmi_hop_x_topodist" + ("_directed" if args.direction == "directed" else "")
                         + ("_exact" if args.max_anchors >= 10**6 else "") + ".csv")
    all_a, all_b = [], []
    for ds in datasets:
        ra, rb = analyse(ds, args.d_max, args.max_anchors, args.workers, args.seed, args.direction)
        all_a += ra
        all_b += rb
        for path, rows in ((out_a, all_a), (out_b, all_b)):
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"\nwrote {out_a}\nwrote {out_b}")


if __name__ == "__main__":
    main()
