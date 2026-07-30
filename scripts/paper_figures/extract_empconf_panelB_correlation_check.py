"""Diagnostic: phi coefficient (== Pearson r == Spearman rho, for two binary
variables -- see note below) between anchor sign and context sign, computed
on the EXACT SAME pipeline as extract_empconf_panelB_mi_decay_linegraph.py
(same datasets, same d_max, same anchor sample/seed, same BFS/shell/counting
logic) -- only the final aggregation statistic changes, from MI/NMI to a
correlation coefficient. Reports both, side by side, from the same underlying
2x2 counts.

Why: the plug-in MI estimator is mathematically constrained to be >= 0, so
sampling noise can only push it UP from the truth, never down -- confirmed in
MI_ESTIMATOR_ELI5.md that a shuffle-signs null control (true MI = 0 by
construction) still shows small POSITIVE NMI that grows as the sample size
collapses at large distance. A correlation coefficient does not have this
one-sided constraint: under the null it is centered at exactly 0, with
symmetric two-sided noise. So this is a clean way to ask: does the
"MI bump" at distance 4-6 look like a real, one-sided, reproducible effect
under a completely different, unbiased-under-the-null statistic, or does it
just look like noise scattered around 0?

Why phi == Pearson r == Spearman rho here: for two binary {0,1} variables,
converting each to ranks (with average-rank ties) is an affine transform of
the raw 0/1 values (all 0s map to one fixed rank, all 1s to another fixed
rank) -- and Pearson correlation is invariant to affine transforms of either
variable. So Spearman-on-binary-data collapses to exactly Pearson r, which
for a 2x2 table is exactly the phi coefficient. No need to materialize raw
per-pair arrays and call scipy -- computed directly and exactly from the same
2x2 count table already being accumulated for MI.

IMPORTANT CAVEAT (does not fix the other issue): switching statistics removes
the MI estimator's specific one-sided bias, but does NOT fix the edge-reuse/
non-independence problem documented in MI_ESTIMATOR_ELI5.md section 7 (the
same handful of real edges getting rediscovered by many different anchors at
large distance) -- that affects the EFFECTIVE sample size / true variance of
ANY statistic computed this way, correlation included. Both the real and
shuffled-null runs here share the identical clustering structure (same
graph, same BFS), so the real-vs-null comparison still isolates "is there a
real signal," but neither one's nominal N should be read as a true
independent-sample count.

Usage: same CLI as the MI script (--datasets, --d-max, --max-anchors,
--shuffle-signs, --out, --seed, --workers).
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
import scripts.paper_figures.extract_empconf_panelB_mi_decay_linegraph as mi_mod
from scripts.paper_figures.extract_empconf_panelB_mi_decay_linegraph import (
    build_graph, _set_graph, _process_anchor, mi_from_cont,
    D_MAX_DEFAULT, MIN_PAIRS_FOR_MI, CHUNK_SIZE,
)

OUT_CSV_DEFAULT = "aaai2027/figure_data/empconf_panelB_correlation_check.csv"


def phi_from_cont(c):
    """c: 2x2 int array, rows=anchor sign (0=neg,1=pos), cols=context sign.
    Returns (phi, se_under_null, z, p_two_sided, n)."""
    n = int(c.sum())
    if n < MIN_PAIRS_FOR_MI:
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    n00, n01 = c[0, 0], c[0, 1]
    n10, n11 = c[1, 0], c[1, 1]
    row0, row1 = n00 + n01, n10 + n11
    col0, col1 = n00 + n10, n01 + n11
    denom = math.sqrt(float(row0) * row1 * col0 * col1)
    if denom == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    phi = (n11 * n00 - n10 * n01) / denom
    # SE of phi under the null (no correlation), standard large-sample approximation: 1/sqrt(n-1)
    se_null = 1.0 / math.sqrt(max(n - 1, 1))
    z = phi / se_null if se_null > 0 else float("nan")
    # two-sided normal p-value via erfc (no scipy dependency needed)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return float(phi), float(se_null), float(z), float(p), n


def analyse_dataset(ds_name, cfg, d_max, max_anchors, seed=42, workers=None, shuffle_signs=False):
    t0 = time.time()
    raw_edges = load_edges_canonical(cfg["ds_name"])
    N, E, edge_x, edge_y, edge_s, adj = build_graph(raw_edges)
    print(f"\n{'='*64}\n{ds_name}: N={N:,} E={E:,}\n{'='*64}", flush=True)

    rng = np.random.default_rng(seed)
    if shuffle_signs:
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
        for totals in pool.imap_unordered(_worker_chunk_local, chunks):
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
        phi, se_null, z, p, _ = phi_from_cont(conts[d])
        results[d] = {"mi": mi, "nmi": nmi, "n_pairs": n, "phi": phi, "se_null": se_null, "z": z, "p": p}
        if isinstance(phi, float) and not math.isnan(phi):
            nmi_s = "nan" if (isinstance(nmi, float) and math.isnan(nmi)) else f"{nmi:.6f}"
            print(f"  line_dist={d:<3} n={n:>15,}  NMI={nmi_s:<10}  phi={phi:+.5f}  z={z:+7.2f}  p={p:.2e}",
                  flush=True)
        else:
            print(f"  line_dist={d:<3} n={n:>15,}  nan", flush=True)

    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    return results


def _worker_chunk_local(anchor_chunk):
    N = mi_mod._N
    shell = np.full(N, -1, np.int32)
    gen = np.zeros(N, np.int32)
    totals = {}
    for cur_gen, eid in enumerate(anchor_chunk, start=1):
        local = _process_anchor(int(eid), shell, gen, cur_gen)
        for key, cnt in local.items():
            totals[key] = totals.get(key, 0) + cnt
    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["all"])
    ap.add_argument("--d-max", type=int, default=D_MAX_DEFAULT)
    ap.add_argument("--max-anchors", type=int, default=None)
    ap.add_argument("--out", default=OUT_CSV_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--shuffle-signs", action="store_true")
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
                "phi": r["phi"], "se_null": r["se_null"], "z": r["z"], "p": r["p"],
            })
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "line_dist", "n_pairs", "mi_bits", "nmi",
                                               "phi", "se_null", "z", "p"])
            w.writeheader()
            w.writerows(rows)
        print(f"  [checkpoint written to {args.out}]", flush=True)

    print(f"\nDone. wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
