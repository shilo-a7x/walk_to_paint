"""One-off cheap graph-statistics pass over all 6 canonical datasets --
saved to disk so basic size/degree/connectivity/sign-balance numbers don't
need recomputing every time a new diagnostic (e.g. the Panel B MI/phi "bump"
investigation) needs to reference them.

Uses the same production edge loader as everything else
(scripts.balance_theory_paths.load_edges_canonical / DATASET_CONFIGS) --
real, canonical, training-identical edges, same as extract_empconf_panelB_*.

All statistics here are deliberately CHEAP:
- size/density/degree distribution: O(E) via numpy, exact.
- weak connectivity (components, giant-component fraction): O(V+E) via
  scipy.sparse.csgraph.connected_components, exact.
- sign balance / reciprocity: O(E) via numpy/set membership, exact.
- effective diameter / clustering coefficient: NOT computed exactly (would be
  O(V*E) or worse on epinions' 840K edges) -- estimated from a random sample
  of `SAMPLE_NODES` source nodes (BFS shell sizes for diameter, neighbor-pair
  intersection for local clustering), same sampling philosophy already used
  by extract_empconf_panelB_mi_decay_linegraph.py's anchor subsampling.

Also cross-references the already-computed Panel B "bump" size (NMI rise
from its minimum to the last measured distance, and phi at that same tail
distance -- see empconf_panelB_mi_decay_linegraph.csv /
empconf_panelB_correlation_check.csv) against each cheap graph statistic
via Spearman rank correlation (n=6, so purely suggestive, not a significance
test) -- this is what answers "does any cheap graph property explain why
slashdot's (and wiki-rfa's) bump is the largest?".

Output:
- aaai2027/figure_data/dataset_stats.csv -- one row per dataset, all raw
  numbers (machine-readable, safe to reload for any future analysis).
- aaai2027/DATASET_STATS.md -- the human-readable persistent reference doc
  requested by the user, generated fresh each run (don't hand-edit -- rerun
  this script if a number needs updating).
"""
import csv
import math
import os
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from scripts.balance_theory_paths import load_edges_canonical, DATASET_CONFIGS

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot"]
DISPLAY_NAME = {"slashdot": "slashdot090221"}

SEED = 42
SAMPLE_NODES = 300     # for effective-diameter / clustering-coefficient estimates
BFS_CAP_NODES = 200000  # safety cap so a single BFS never walks more than this many nodes

OUT_CSV = "aaai2027/figure_data/dataset_stats.csv"
OUT_MD = "aaai2027/DATASET_STATS.md"

# ---- Panel B bump numbers, read directly from the existing analysis CSVs so
# this doc can report the bump-vs-stat correlation without re-deriving them.
MI_CSV = "aaai2027/figure_data/empconf_panelB_mi_decay_linegraph.csv"
PHI_CSV = "aaai2027/figure_data/empconf_panelB_correlation_check.csv"


def build_adjacency(edges, N, n2i):
    adj = [[] for _ in range(N)]
    for u, v, s in edges:
        ui, vi = n2i[u], n2i[v]
        adj[ui].append(vi)
        adj[vi].append(ui)
    return [np.array(sorted(set(a)), dtype=np.int64) for a in adj]


def bfs_shells(adj, src, cap=BFS_CAP_NODES):
    N = len(adj)
    visited = np.zeros(N, dtype=bool)
    visited[src] = True
    frontier = [src]
    shell_sizes = []
    n_visited = 1
    d = 0
    while frontier and n_visited < cap:
        nxt = []
        for w in frontier:
            for nb in adj[w]:
                if not visited[nb]:
                    visited[nb] = True
                    nxt.append(nb)
        if not nxt:
            break
        d += 1
        shell_sizes.append(len(nxt))
        n_visited += len(nxt)
        frontier = nxt
    return d, n_visited, shell_sizes


def local_clustering(adj, node):
    nbrs = adj[node]
    k = len(nbrs)
    if k < 2:
        return float("nan")
    nbr_set = set(nbrs.tolist())
    links = 0
    for nb in nbrs:
        for nb2 in adj[nb]:
            if nb2 in nbr_set and nb2 > nb:
                links += 1
    possible = k * (k - 1) / 2
    return links / possible if possible > 0 else float("nan")


def percentile_stats(arr):
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(np.max(arr)),
    }


def gini(arr):
    a = np.sort(np.asarray(arr, dtype=np.float64))
    n = len(a)
    if n == 0 or a.sum() == 0:
        return float("nan")
    cum = np.cumsum(a)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def analyse(ds_key):
    cfg = DATASET_CONFIGS[ds_key]
    edges = load_edges_canonical(cfg["ds_name"])
    E = len(edges)
    nodes = sorted({n for e in edges for n in e[:2]})
    N = len(nodes)
    n2i = {n: i for i, n in enumerate(nodes)}

    u_idx = np.array([n2i[u] for u, v, s in edges], dtype=np.int64)
    v_idx = np.array([n2i[v] for u, v, s in edges], dtype=np.int64)
    signs = np.array([1 if s > 0 else 0 for u, v, s in edges], dtype=np.int8)

    # --- degree (undirected total degree, plus directed in/out) ---
    out_deg = np.bincount(u_idx, minlength=N)
    in_deg = np.bincount(v_idx, minlength=N)
    tot_deg = out_deg + in_deg  # multigraph-safe: counts parallel/self edges as-is

    # --- sign balance ---
    n_pos = int(signs.sum())
    n_neg = E - n_pos

    # --- reciprocity: fraction of directed edges (u,v) whose mirror (v,u) also exists ---
    pair_set = set(zip(u_idx.tolist(), v_idx.tolist()))
    n_recip = sum(1 for u, v in pair_set if (v, u) in pair_set)
    reciprocity = n_recip / len(pair_set) if pair_set else float("nan")

    # --- weak connectivity via scipy sparse ---
    rows = np.concatenate([u_idx, v_idx])
    cols = np.concatenate([v_idx, u_idx])
    data = np.ones(len(rows), dtype=np.int8)
    A = coo_matrix((data, (rows, cols)), shape=(N, N))
    n_components, labels = connected_components(A, directed=False, connection="weak")
    comp_sizes = np.bincount(labels)
    giant_frac_nodes = float(comp_sizes.max() / N)
    giant_id = int(np.argmax(comp_sizes))
    in_giant = labels == giant_id
    giant_edge_mask = in_giant[u_idx] & in_giant[v_idx]
    giant_frac_edges = float(giant_edge_mask.sum() / E)

    # --- density (directed, simple-graph normalization; multi-edges counted as-is) ---
    density = E / (N * (N - 1)) if N > 1 else float("nan")

    # --- sampled effective diameter + clustering coefficient ---
    adj = build_adjacency(edges, N, n2i)
    rng = np.random.default_rng(SEED)
    sample = rng.choice(N, size=min(SAMPLE_NODES, N), replace=False)
    eccentricities, reach_fracs = [], []
    for src in sample:
        d, n_visited, _ = bfs_shells(adj, int(src))
        eccentricities.append(d)
        reach_fracs.append(n_visited / N)
    clustering_sample = rng.choice(N, size=min(SAMPLE_NODES, N), replace=False)
    local_cc = [local_clustering(adj, int(v)) for v in clustering_sample]
    local_cc = [c for c in local_cc if not math.isnan(c)]

    return {
        "dataset": ds_key,
        "N": N, "E": E,
        "density": density,
        "n_pos": n_pos, "n_neg": n_neg, "frac_pos": n_pos / E,
        "reciprocity": reciprocity,
        "n_components": int(n_components),
        "giant_frac_nodes": giant_frac_nodes,
        "giant_frac_edges": giant_frac_edges,
        "out_deg": percentile_stats(out_deg), "in_deg": percentile_stats(in_deg),
        "tot_deg": percentile_stats(tot_deg),
        "degree_gini": gini(tot_deg),
        "frac_degree_1": float(np.mean(tot_deg <= 1)),
        "top1pct_degree_share": float(
            np.sort(tot_deg)[::-1][:max(1, N // 100)].sum() / tot_deg.sum()
        ),
        "sampled_eccentricity_mean": float(np.mean(eccentricities)),
        "sampled_eccentricity_max": int(np.max(eccentricities)),
        "sampled_reach_frac_mean": float(np.mean(reach_fracs)),
        "sampled_clustering_mean": float(np.mean(local_cc)) if local_cc else float("nan"),
    }


def load_bump_metrics():
    """Re-derive the same 'NMI bump' / 'phi at tail' numbers already reported
    to the user, straight from the existing Panel B CSVs (no recomputation of
    the underlying MI/phi values -- just re-reading what's on disk)."""
    from collections import defaultdict
    mi = defaultdict(dict)
    with open(MI_CSV) as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            v = row["nmi"]
            if v not in ("", "nan"):
                mi[ds][int(row["line_dist"])] = float(v)
    phi = defaultdict(dict)
    with open(PHI_CSV) as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            v = row["phi"]
            if v not in ("", "nan"):
                phi[ds][int(row["line_dist"])] = float(v)

    out = {}
    for ds in DATASETS:
        key = DISPLAY_NAME.get(ds, ds)
        m = mi.get(key, mi.get(ds))
        p = phi.get(key, phi.get(ds))
        if not m:
            continue
        dists = sorted(m)
        last = dists[-1]
        min_d = min(dists, key=lambda d: m[d])
        bump = m[last] - m[min_d]
        phi_last = p.get(last, float("nan")) if p else float("nan")
        out[ds] = {"nmi_bump": bump, "phi_last": phi_last, "phi_last_sq": phi_last ** 2
                    if not math.isnan(phi_last) else float("nan")}
    return out


def spearman(xs, ys):
    n = len(xs)
    order_x = sorted(range(n), key=lambda i: xs[i])
    order_y = sorted(range(n), key=lambda i: ys[i])
    rx = [0] * n
    ry = [0] * n
    for pos, i in enumerate(order_x):
        rx[i] = pos
    for pos, i in enumerate(order_y):
        ry[i] = pos
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n ** 2 - 1)) if n > 1 else float("nan")


def main():
    t0 = time.time()
    stats = {}
    for ds in DATASETS:
        print(f"analysing {ds}...", flush=True)
        stats[ds] = analyse(ds)
        print(f"  N={stats[ds]['N']:,} E={stats[ds]['E']:,} done in {time.time()-t0:.0f}s total", flush=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    flat_rows = []
    for ds in DATASETS:
        s = stats[ds]
        flat = {
            "dataset": DISPLAY_NAME.get(ds, ds), "N": s["N"], "E": s["E"], "density": s["density"],
            "frac_pos": s["frac_pos"], "reciprocity": s["reciprocity"],
            "n_components": s["n_components"], "giant_frac_nodes": s["giant_frac_nodes"],
            "giant_frac_edges": s["giant_frac_edges"],
            "out_deg_mean": s["out_deg"]["mean"], "out_deg_p99": s["out_deg"]["p99"], "out_deg_max": s["out_deg"]["max"],
            "in_deg_mean": s["in_deg"]["mean"], "in_deg_p99": s["in_deg"]["p99"], "in_deg_max": s["in_deg"]["max"],
            "tot_deg_mean": s["tot_deg"]["mean"], "tot_deg_median": s["tot_deg"]["median"],
            "tot_deg_std": s["tot_deg"]["std"], "tot_deg_p90": s["tot_deg"]["p90"],
            "tot_deg_p99": s["tot_deg"]["p99"], "tot_deg_max": s["tot_deg"]["max"],
            "degree_gini": s["degree_gini"], "frac_degree_1": s["frac_degree_1"],
            "top1pct_degree_share": s["top1pct_degree_share"],
            "sampled_eccentricity_mean": s["sampled_eccentricity_mean"],
            "sampled_eccentricity_max": s["sampled_eccentricity_max"],
            "sampled_reach_frac_mean": s["sampled_reach_frac_mean"],
            "sampled_clustering_mean": s["sampled_clustering_mean"],
        }
        flat_rows.append(flat)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        w.writeheader()
        w.writerows(flat_rows)
    print(f"wrote {OUT_CSV}")

    bump = load_bump_metrics()
    corr_vars = {
        "N (nodes)": [stats[ds]["N"] for ds in DATASETS],
        "E (edges)": [stats[ds]["E"] for ds in DATASETS],
        "density": [stats[ds]["density"] for ds in DATASETS],
        "mean total degree": [stats[ds]["tot_deg"]["mean"] for ds in DATASETS],
        "degree Gini": [stats[ds]["degree_gini"] for ds in DATASETS],
        "top-1% degree share": [stats[ds]["top1pct_degree_share"] for ds in DATASETS],
        "sampled mean eccentricity": [stats[ds]["sampled_eccentricity_mean"] for ds in DATASETS],
        "sampled max eccentricity": [stats[ds]["sampled_eccentricity_max"] for ds in DATASETS],
        "sampled clustering coeff.": [stats[ds]["sampled_clustering_mean"] for ds in DATASETS],
        "reciprocity": [stats[ds]["reciprocity"] for ds in DATASETS],
        "frac positive signs": [stats[ds]["frac_pos"] for ds in DATASETS],
        "giant component frac (nodes)": [stats[ds]["giant_frac_nodes"] for ds in DATASETS],
    }
    bump_nmi = [bump.get(ds, {}).get("nmi_bump", float("nan")) for ds in DATASETS]
    bump_phi2 = [bump.get(ds, {}).get("phi_last_sq", float("nan")) for ds in DATASETS]

    corr_rows = []
    for name, xs in corr_vars.items():
        rho_nmi = spearman(xs, bump_nmi)
        rho_phi2 = spearman(xs, bump_phi2)
        corr_rows.append((name, rho_nmi, rho_phi2))

    # ---- write the markdown report ----
    lines = []
    lines.append("# Dataset statistics — all 6 canonical datasets")
    lines.append("")
    lines.append(f"Generated by `scripts/paper_figures/compute_dataset_stats.py` "
                 f"(rerun that script if any number needs updating — do not hand-edit this file). "
                 f"Loaded via the production `load_edges_canonical`/`DATASET_CONFIGS` path "
                 f"(same edges used for training and for the Panel B MI/phi analysis).")
    lines.append("")
    lines.append("## Size, density, sign balance, connectivity")
    lines.append("")
    lines.append("| dataset | N (nodes) | E (edges) | density | % positive | reciprocity | "
                 "# weak components | giant comp. (nodes) | giant comp. (edges) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        s = stats[ds]
        name = DISPLAY_NAME.get(ds, ds)
        lines.append(f"| {name} | {s['N']:,} | {s['E']:,} | {s['density']:.2e} | "
                     f"{100*s['frac_pos']:.1f}% | {s['reciprocity']:.3f} | {s['n_components']:,} | "
                     f"{100*s['giant_frac_nodes']:.2f}% | {100*s['giant_frac_edges']:.2f}% |")
    lines.append("")
    lines.append("## Degree distribution (total degree = in + out, undirected count)")
    lines.append("")
    lines.append("| dataset | mean | median | std | p90 | p99 | max | Gini | % degree≤1 | top-1% share |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        s = stats[ds]
        name = DISPLAY_NAME.get(ds, ds)
        td = s["tot_deg"]
        lines.append(f"| {name} | {td['mean']:.2f} | {td['median']:.1f} | {td['std']:.2f} | "
                     f"{td['p90']:.1f} | {td['p99']:.1f} | {td['max']:,} | {s['degree_gini']:.3f} | "
                     f"{100*s['frac_degree_1']:.1f}% | {100*s['top1pct_degree_share']:.1f}% |")
    lines.append("")
    lines.append(f"## Sampled structural stats ({SAMPLE_NODES} random source nodes per dataset, "
                 f"BFS shell expansion capped at {BFS_CAP_NODES:,} visited nodes)")
    lines.append("")
    lines.append("| dataset | mean eccentricity | max eccentricity | mean BFS reach (frac of N) | "
                 "mean local clustering coeff. |")
    lines.append("|---|---|---|---|---|")
    for ds in DATASETS:
        s = stats[ds]
        name = DISPLAY_NAME.get(ds, ds)
        lines.append(f"| {name} | {s['sampled_eccentricity_mean']:.2f} | "
                     f"{s['sampled_eccentricity_max']} | {100*s['sampled_reach_frac_mean']:.1f}% | "
                     f"{s['sampled_clustering_mean']:.4f} |")
    lines.append("")
    lines.append("*Eccentricity here is BFS depth reached from a sampled source node before the "
                 "frontier stops growing (i.e. the local component is exhausted or the cap is hit) "
                 "— a cheap stand-in for graph diameter, not an exact value. 'Mean BFS reach' is "
                 "what fraction of all N nodes that single BFS visited before stopping — directly "
                 "relevant to why Panel B's 'bump' saturates/runs out of pairs at different "
                 "distances per dataset (a small, low-diameter graph exhausts its whole node set "
                 "in a handful of hops; a large sparse one doesn't).*")
    lines.append("")
    lines.append("## Panel B \"bump\" size vs. each cheap graph statistic (Spearman ρ, n=6 datasets)")
    lines.append("")
    lines.append("Bump size = NMI at the last measured distance minus NMI's own minimum "
                 "(the pre-bump dip); phi²(tail) = squared signed correlation coefficient at that "
                 "same last distance (see `empconf_panelB_mi_decay_linegraph.csv` / "
                 "`empconf_panelB_correlation_check.csv`). n=6 is too small for a real significance "
                 "test — read these as suggestive ranking hints, not proof.")
    lines.append("")
    lines.append("| graph statistic | ρ vs. NMI bump | ρ vs. phi²(tail) |")
    lines.append("|---|---|---|")
    for name, rho_nmi, rho_phi2 in corr_rows:
        lines.append(f"| {name} | {rho_nmi:+.2f} | {rho_phi2:+.2f} |")
    lines.append("")

    # pick the strongest candidate(s) to narrate
    best = max(corr_rows, key=lambda r: abs(r[1]) + abs(r[2]))
    lines.append(f"**Strongest candidate by |ρ|: {best[0]}** (ρ={best[1]:+.2f} vs. NMI bump, "
                 f"{best[2]:+.2f} vs. phi²) — see the write-up below for whether this actually "
                 f"explains slashdot's position.")
    lines.append("")
    lines.append("### Raw per-dataset bump numbers used above")
    lines.append("")
    lines.append("| dataset | NMI bump | phi(tail) | phi²(tail) |")
    lines.append("|---|---|---|---|")
    for ds in DATASETS:
        b = bump.get(ds, {})
        name = DISPLAY_NAME.get(ds, ds)
        if b:
            lines.append(f"| {name} | {b['nmi_bump']:.6f} | {b['phi_last']:+.5f} | {b['phi_last_sq']:.6f} |")
    lines.append("")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")
    print(f"\nDone in {time.time()-t0:.0f}s")

    print("\n=== correlation summary (ρ vs NMI bump, ρ vs phi^2 tail) ===")
    for name, rho_nmi, rho_phi2 in sorted(corr_rows, key=lambda r: -(abs(r[1]) + abs(r[2]))):
        print(f"  {name:32} {rho_nmi:+.2f}  {rho_phi2:+.2f}")


if __name__ == "__main__":
    main()
