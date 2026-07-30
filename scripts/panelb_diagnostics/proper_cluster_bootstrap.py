"""
Corrected cluster-robust bootstrap for the Panel B "bump" question.

Supersedes an earlier, methodologically FLAWED attempt (built and retracted
same session, see PANELB_INVESTIGATION_REPORT.md) that bootstrapped a
DEDUPED sample -- one row per distinct context edge -- to get a "cluster CI".
That's wrong: dropping to a smaller sample size and bootstrapping IT just
reproduces the standard 1/sqrt(n) widening from a smaller n, it doesn't
correct for the actual clustering. It also silently changes the point
estimate by discarding real data, which distorts the sign-imbalance
composition (validated empirically below: dedup shifts % positive by up to
-6.2pp at some distances).

This script separates the two concerns cleanly:
  - The RAW point estimate (all rows, nothing discarded) is kept as the
    number of interest -- repeated encounters of a hub edge are a real
    feature of how often a model would actually encounter that edge, not
    definitionally "bias" to be removed.
  - A PROPER cluster (block) bootstrap gets a valid CI around that SAME raw
    point estimate: resample distinct context edges WITH replacement, but
    each resampled edge contributes ALL of its original rows (not just one)
    -- no information discarded anywhere, only the resampling UNIT changes
    from "row" to "edge".

Self-tests the vectorized "ragged repeat" resampling trick against a
brute-force reference on synthetic data before trusting it on real data.

Requires: outputs/panelb_diagnostics/panelb_raw_rows.npy, produced by
extract_panelb_raw_rows.py in this same directory.
"""
import math
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_NPY = os.path.join(ROOT, "outputs", "panelb_diagnostics", "panelb_raw_rows.npy")
SEED = 42
N_BOOTSTRAP = 500
D_MAX = 6


def phi_from_pairs(a, c):
    n = len(a)
    if n < 10:
        return float("nan"), n
    n11 = int(((a == 1) & (c == 1)).sum())
    n10 = int(((a == 1) & (c == 0)).sum())
    n01 = int(((a == 0) & (c == 1)).sum())
    n00 = int(((a == 0) & (c == 0)).sum())
    row0, row1 = n00 + n01, n10 + n11
    col0, col1 = n00 + n10, n01 + n11
    denom = row0 * row1 * col0 * col1
    if denom == 0:
        return float("nan"), n
    phi = (n11 * n00 - n10 * n01) / math.sqrt(denom)
    return phi, n


def full_cluster_bootstrap(a, c, cluster_ids, n_boot, rng):
    """Resample distinct clusters WITH replacement; each resampled cluster
    contributes ALL of its original rows (not just one) -- no information
    discarded, only the resampling UNIT changes from "row" to "cluster"."""
    order = np.argsort(cluster_ids, kind="stable")
    a_s, c_s = a[order], c[order]
    uniq, start_idx, counts = np.unique(cluster_ids[order], return_index=True, return_counts=True)
    n_clusters = len(uniq)

    phis = np.empty(n_boot)
    ns = np.empty(n_boot, dtype=np.int64)
    for b in range(n_boot):
        chosen = rng.integers(0, n_clusters, n_clusters)
        sizes = counts[chosen]
        total = int(sizes.sum())
        dest_offsets = np.concatenate(([0], np.cumsum(sizes)))[:-1]
        rep_starts = np.repeat(start_idx[chosen], sizes)
        rep_dest_offsets = np.repeat(dest_offsets, sizes)
        local_offset = np.arange(total) - rep_dest_offsets
        src_idx = rep_starts + local_offset
        phi, n = phi_from_pairs(a_s[src_idx], c_s[src_idx])
        phis[b] = phi
        ns[b] = n
    return phis, ns, n_clusters


def _self_test():
    """Verify the vectorized ragged-repeat trick against a brute-force loop
    on a tiny synthetic case before trusting it on real data."""
    a_t = np.array([1, 1, 0, 0, 0, 1, 1, 0])
    c_t = np.array([1, 1, 1, 0, 0, 0, 1, 1])
    cl_t = np.array([0, 0, 1, 1, 1, 2, 3, 3])  # cluster sizes: 2,3,1,2

    order = np.argsort(cl_t, kind="stable")
    a_s, c_s = a_t[order], c_t[order]
    uniq, start_idx, counts = np.unique(cl_t[order], return_index=True, return_counts=True)
    n_clusters = len(uniq)

    rng2 = np.random.default_rng(123)
    chosen = rng2.integers(0, n_clusters, n_clusters)
    sizes = counts[chosen]
    total = int(sizes.sum())
    dest_offsets = np.concatenate(([0], np.cumsum(sizes)))[:-1]
    rep_starts = np.repeat(start_idx[chosen], sizes)
    rep_dest_offsets = np.repeat(dest_offsets, sizes)
    local_offset = np.arange(total) - rep_dest_offsets
    src_idx = rep_starts + local_offset
    vec_a, vec_c = a_s[src_idx], c_s[src_idx]

    ref_a, ref_c = [], []
    for cl in chosen:
        s, e = start_idx[cl], start_idx[cl] + counts[cl]
        ref_a.extend(a_s[s:e].tolist())
        ref_c.extend(c_s[s:e].tolist())
    assert vec_a.tolist() == ref_a, "vectorized cluster bootstrap MISMATCH (a)"
    assert vec_c.tolist() == ref_c, "vectorized cluster bootstrap MISMATCH (c)"
    print("self-test passed: vectorized cluster bootstrap matches brute-force reference exactly")


def main():
    _self_test()

    arr = np.load(IN_NPY)
    line_d, a_sign, c_sign, eid, pair_key = (arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4])
    print(f"loaded {len(arr):,} raw rows from {IN_NPY}")

    print(f"\n{'d':>2} {'n_raw':>10} {'n_distinct':>11} {'pct_discarded_by_dedup':>23} "
          f"{'pct_pos_raw':>12} {'pct_pos_deduped':>16} {'composition_shift':>18}")
    for d in range(1, D_MAX + 1):
        mask = line_d == d
        n_raw = int(mask.sum())
        c_d = c_sign[mask]
        eid_d = eid[mask]
        uniq_eid, first_pos = np.unique(eid_d, return_index=True)
        n_distinct = len(uniq_eid)
        pct_discarded = 100 * (1 - n_distinct / n_raw) if n_raw else float("nan")
        pct_pos_raw = 100 * c_d.mean()
        pct_pos_dedup = 100 * c_d[first_pos].mean()
        shift = pct_pos_dedup - pct_pos_raw
        print(f"{d:2d} {n_raw:>10,} {n_distinct:>11,} {pct_discarded:>22.1f}% "
              f"{pct_pos_raw:>11.2f}% {pct_pos_dedup:>15.2f}% {shift:>+17.2f}pp")

    print(f"\n{'d':>2} {'phi_RAW(all data, nothing discarded)':>38} {'n_raw':>10} {'n_clusters':>11} "
          f"{'proper_cluster_CI_lo':>21} {'proper_cluster_CI_hi':>21} {'excl_0':>7}")
    t0 = time.time()
    for d in range(1, D_MAX + 1):
        mask = np.where(line_d == d)[0]
        a_d, c_d, eid_d = a_sign[mask], c_sign[mask], eid[mask]
        phi_raw, n_raw = phi_from_pairs(a_d, c_d)

        rng = np.random.default_rng(SEED + 7000 + d)
        phis, ns, n_clusters = full_cluster_bootstrap(a_d, c_d, eid_d, N_BOOTSTRAP, rng)
        lo, hi = np.nanpercentile(phis, [2.5, 97.5])
        excl = "YES" if (lo > 0 or hi < 0) else "no"
        print(f"{d:2d} {phi_raw:>38.5f} {n_raw:>10,} {n_clusters:>11,} "
              f"{lo:>21.5f} {hi:>21.5f} {excl:>7}   (mean bootstrap n={ns.mean():,.0f}, "
              f"ran in {time.time()-t0:.1f}s so far)")


if __name__ == "__main__":
    main()
