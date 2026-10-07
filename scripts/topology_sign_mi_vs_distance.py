"""Is Panel B's edge-sign MI-vs-distance curve (bumps included) explained by topology alone?

Same pair definition as scripts/paper_figures/extract_empconf_panelB_mi_decay_linegraph.py
(undirected line-graph distance, BFS rooted at both anchor endpoints), on a fixed random
sample of anchor edges. Each edge e=(x->y) gets a topology vector
    t(e) = log1p([outdeg(x), indeg(x), outdeg(y), indeg(y)])   (src_out, src_in, tgt_out, tgt_in)
computed from all real edges (structure only, signs never used). Per line distance d, over all
(anchor, context) pairs at that distance:

  real       NMI / phi between the two edges' real signs (re-derives Panel B on this sample)
  surrogate  the same NMI / phi with every sign replaced by a topology-only surrogate sign:
             logistic regression of sign on t(e) (in-sample, descriptive), thresholded so the
             surrogate keeps the real positive rate. If topology explains the sign curve, the
             surrogate curve has the same shape (including the bumps).
  topo_agree NMI / phi between "the two signs agree" and "the two topology vectors are close"
             (||t_a - t_c||): NMI over 10 global-quantile distance bins, phi on a global-median
             split. Thresholds are global (all hops pooled) so hops are comparable.

Pairs are counted once per (anchor, context), like Panel B -- the tail bins inherit Panel B's
known pair-reuse caveat (PAPER_CLOSEOUT_LOG.md 2026-07-28), so read the far bins with care.

Usage:
  .venv/bin/python scripts/topology_sign_mi_vs_distance.py --max-anchors 20000 --workers 48
"""
import argparse
import csv
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.balance_theory_paths import load_edges_canonical, DATASET_CONFIGS  # noqa: E402

N_DBINS = 400          # fine histogram of topology distance, quantile bins derived afterwards
N_QBINS = 10
CHUNK_SIZE = 50
OUT_DIR = "outputs/topology_mi_vs_distance"

# graph state, set once per dataset before the fork pool is created (copy-on-write in workers)
_G = {}


def _neighbors(nodes):
    indptr, nbr, eid = _G["indptr"], _G["nbr"], _G["eid"]
    starts, ends = indptr[nodes], indptr[nodes + 1]
    lengths = ends - starts
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    idx = np.repeat(starts - np.cumsum(lengths) + lengths, lengths) + np.arange(total)
    return nbr[idx], eid[idx]


def _worker_chunk(anchors):
    g = _G
    d_max, N = g["d_max"], g["N"]
    ex, ey, s, sh, T = g["ex"], g["ey"], g["s"], g["sh"], g["T"]
    dbin_w = g["dbin_w"]
    n_d = d_max + 2
    real = np.zeros((n_d, 2, 2), np.int64)
    surr = np.zeros((n_d, 2, 2), np.int64)
    cond = np.zeros((n_d, N_QBINS, N_QBINS, 2, 2), np.int64)
    hist = np.zeros((n_d, 2, N_DBINS), np.int64)
    shell = np.zeros(N, np.int32)
    gen = np.zeros(N, np.int32)
    for cur, a in enumerate(anchors, start=1):
        u, v = ex[a], ey[a]
        frontier = np.unique(np.array([u, v], np.int64))
        gen[frontier] = cur
        shell[frontier] = 0
        reached = [frontier]
        for d in range(1, d_max + 1):
            nb, _ = _neighbors(frontier)
            nb = np.unique(nb)
            nb = nb[gen[nb] != cur]
            if nb.size == 0:
                break
            gen[nb] = cur
            shell[nb] = d
            reached.append(nb)
            frontier = nb
        _, ce = _neighbors(np.concatenate(reached))
        ce = np.unique(ce)
        ce = ce[ce != a]
        cx, cy = ex[ce], ey[ce]
        sx = np.where(gen[cx] == cur, shell[cx], d_max + 1)
        sy = np.where(gen[cy] == cur, shell[cy], d_max + 1)
        de = np.minimum(sx, sy)
        keep = de <= d_max
        ce, line_d = ce[keep], de[keep] + 1
        np.add.at(real, (line_d, s[a], s[ce]), 1)
        np.add.at(surr, (line_d, sh[a], sh[ce]), 1)
        np.add.at(cond, (line_d, g["q"][a], g["q"][ce], s[a], s[ce]), 1)
        dist = np.linalg.norm(T[ce] - T[a], axis=1)
        b = np.minimum((dist / dbin_w).astype(np.int64), N_DBINS - 1)
        np.add.at(hist, (line_d, (s[ce] == s[a]).astype(np.int64), b), 1)
    return real, surr, cond, hist


def _nmi_phi_2x2(c):
    n = c.sum()
    if n < 2000:
        return float("nan"), float("nan"), int(n)
    p = c / n
    py, ps = p.sum(1), p.sum(0)
    mi = sum(p[i, j] * math.log2(p[i, j] / (py[i] * ps[j]))
             for i in range(2) for j in range(2) if p[i, j] > 0)
    hy = -sum(q * math.log2(q) for q in py if q > 0)
    den = math.sqrt(py[0] * py[1] * ps[0] * ps[1])
    phi = (p[1, 1] * p[0, 0] - p[1, 0] * p[0, 1]) / den if den > 0 else float("nan")
    return mi / hy if hy > 0 else float("nan"), phi, int(n)


def _cond_nmi(c):
    """I(s_a; s_c | topology decile of a, of c) / H(s_a): Panel B's real-sign NMI after
    stratifying both edges by their topology-score decile (10x10 strata)."""
    n = c.sum()
    if n < 2000:
        return float("nan")
    cmi = 0.0
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            m = c[i, j]
            k = m.sum()
            if k == 0:
                continue
            p = m / k
            pr, pc = p.sum(1), p.sum(0)
            nz = p > 0
            cmi += (k / n) * float((p[nz] * np.log2(p[nz] / np.outer(pr, pc)[nz])).sum())
    pa = c.sum((0, 1, 3)) / n
    ha = -sum(x * math.log2(x) for x in pa if x > 0)
    return cmi / ha if ha > 0 else float("nan")


def _nmi_rows(c):
    """NMI of the row variable (agree) with the column variable (distance bin), / H(agree)."""
    n = c.sum()
    if n < 2000:
        return float("nan")
    p = c / n
    pr, pc = p.sum(1), p.sum(0)
    nz = p > 0
    mi = float((p[nz] * np.log2(p[nz] / np.outer(pr, pc)[nz])).sum())
    hr = -sum(q * math.log2(q) for q in pr if q > 0)
    return mi / hr if hr > 0 else float("nan")


def analyse(ds, d_max, max_anchors, workers, seed, shuffle=False):
    t0 = time.time()
    raw = load_edges_canonical(DATASET_CONFIGS[ds]["ds_name"])
    nodes = sorted({n for e in raw for n in e[:2]})
    n2i = {n: i for i, n in enumerate(nodes)}
    N, E = len(nodes), len(raw)
    ex = np.array([n2i[e[0]] for e in raw], np.int64)
    ey = np.array([n2i[e[1]] for e in raw], np.int64)
    s = np.array([1 if e[2] > 0 else 0 for e in raw], np.int64)

    # undirected CSR (both directions), same adjacency as Panel B's default
    src = np.concatenate([ex, ey])
    dst = np.concatenate([ey, ex])
    eids = np.concatenate([np.arange(E), np.arange(E)])
    order = np.argsort(src, kind="stable")
    indptr = np.zeros(N + 1, np.int64)
    np.add.at(indptr, src + 1, 1)
    indptr = np.cumsum(indptr)

    outdeg = np.bincount(ex, minlength=N)
    indeg = np.bincount(ey, minlength=N)
    T = np.log1p(np.stack([outdeg[ex], indeg[ex], outdeg[ey], indeg[ey]], 1).astype(np.float64))

    Tz = (T - T.mean(0)) / T.std(0).clip(1e-9)
    lr = LogisticRegression(max_iter=1000).fit(Tz, s)
    score = lr.predict_proba(Tz)[:, 1]
    topo_auc = roc_auc_score(s, score)
    sh = (score >= np.quantile(score, 1.0 - s.mean())).astype(np.int64)
    q = np.minimum((np.argsort(np.argsort(score, kind="stable")) * N_QBINS) // E, N_QBINS - 1)

    if shuffle:
        # null control: permute real signs over edges (structure, sign balance, surrogate and
        # topology strata all fixed) -- measures the estimators' small-sample bias per hop
        s = s[np.random.default_rng(seed + 1).permutation(E)].copy()

    dmax_val = float(np.linalg.norm(T.max(0) - T.min(0)))
    _G.update(dict(indptr=indptr, nbr=dst[order], eid=eids[order], ex=ex, ey=ey, s=s, sh=sh, q=q, T=T,
                   d_max=d_max, N=N, dbin_w=dmax_val / N_DBINS + 1e-12))

    rng = np.random.default_rng(seed)
    anchors = np.arange(E) if E <= max_anchors else rng.choice(E, max_anchors, replace=False)
    chunks = [anchors[i:i + CHUNK_SIZE] for i in range(0, len(anchors), CHUNK_SIZE)]
    print(f"\n{ds}: N={N:,} E={E:,} anchors={len(anchors):,} pos_rate={s.mean():.3f} "
          f"topology-only sign AUC={topo_auc:.4f}", flush=True)

    n_d = d_max + 2
    real = np.zeros((n_d, 2, 2), np.int64)
    surr = np.zeros((n_d, 2, 2), np.int64)
    cond = np.zeros((n_d, N_QBINS, N_QBINS, 2, 2), np.int64)
    hist = np.zeros((n_d, 2, N_DBINS), np.int64)
    with mp.get_context("fork").Pool(workers) as pool:
        for i, (r, su, co, h) in enumerate(pool.imap_unordered(_worker_chunk, chunks), 1):
            real += r
            surr += su
            cond += co
            hist += h
            if i % max(1, len(chunks) // 10) == 0:
                print(f"  {i}/{len(chunks)} chunks, {time.time() - t0:.0f}s", flush=True)

    # global (all-hop) distance thresholds
    pooled = hist[1:].sum((0, 1))
    cdf = np.cumsum(pooled) / pooled.sum()
    median_bin = int(np.searchsorted(cdf, 0.5))
    q_edges = np.unique(np.searchsorted(cdf, np.linspace(0, 1, N_QBINS + 1)[1:-1]))

    rows = []
    for d in range(1, d_max + 2):
        r_nmi, r_phi, n = _nmi_phi_2x2(real[d])
        s_nmi, s_phi, _ = _nmi_phi_2x2(surr[d])
        c_nmi = _cond_nmi(cond[d])
        h = hist[d]
        close = np.stack([h[:, median_bin + 1:].sum(1), h[:, :median_bin + 1].sum(1)], 1)  # rows agree, cols [far, close]
        _, t_phi, _ = _nmi_phi_2x2(close)
        qb = np.stack([seg.sum(1) for seg in np.split(h, q_edges + 1, axis=1)], 1)
        t_nmi = _nmi_rows(qb)
        rows.append(dict(dataset=ds, line_dist=d, n_pairs=n, topo_sign_auc=round(topo_auc, 4),
                         real_nmi=r_nmi, real_phi=r_phi, surrogate_nmi=s_nmi, surrogate_phi=s_phi, real_nmi_given_topology=c_nmi,
                         topo_agree_nmi=t_nmi, topo_agree_phi=t_phi))
        print(f"  d={d}: n={n:>13,}  real NMI={r_nmi:.2e} phi={r_phi:+.4f} | surrogate NMI={s_nmi:.2e} "
              f"| real NMI|topology={c_nmi:.2e} "
              f"phi={s_phi:+.4f} | topo-agree NMI={t_nmi:.2e} phi={t_phi:+.4f}", flush=True)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["all"])
    ap.add_argument("--d-max", type=int, default=7)
    ap.add_argument("--max-anchors", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle-signs", action="store_true")
    args = ap.parse_args()
    datasets = [d for d in DATASET_CONFIGS if d != "synthetic-fog"] if args.datasets == ["all"] else args.datasets
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "topology_mi_vs_distance" + ("_shuffle" if args.shuffle_signs else "") + ".csv")
    rows = []
    for ds in datasets:
        rows += analyse(ds, args.d_max, args.max_anchors, args.workers, args.seed, args.shuffle_signs)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
