"""
Edge-sign MI vs directed shortest-path distance.

For random anchor edges e=(u→v, label Y), run a directed BFS from u and at
each distance d collect outgoing edges (a→b, sign S) where dist(u→a) = d.
Compute MI(Y; S) as a function of d.

"Distance" = length of shortest DIRECTED path from u to a.

Answers: do edge signs at directed-hop distance d carry any information about
the target edge sign?  If MI drops to ~0 at d=2, local context is sufficient.
If MI remains elevated at d=4+, long-range context genuinely helps.

Uses full-graph labels (not just test split) to maximise data.

Usage
─────
  python scripts/edge_sign_mi_vs_distance.py [--datasets all] [--out outputs/mi_vs_dist]
  python scripts/edge_sign_mi_vs_distance.py --datasets bitcoin-alpha slashdot
"""
import os, sys, argparse, math
from collections import defaultdict, deque
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.balance_theory_paths import load_edges_canonical, DATASET_CONFIGS

D_MAX               = 8      # max BFS depth
MAX_BFS_NODES       = 50000  # stop BFS after this many nodes (memory guard)
MIN_PAIRS_FOR_MI    = 500    # min total pairs at a depth to report MI
# d=0 and d=1 are computed exactly (all pairs); no sampling needed.
# d>=2: use all edges as anchors but cap pairs per anchor to avoid hub bias.
MAX_PAIRS_PER_ANCHOR_DEEP = 500  # cap per anchor for d>=2 layers


# ── MI for two binary variables ±1 ───────────────────────────────────────────

def mi_bits(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """
    Exact MI(Y;S) in bits for binary Y,S ∈ {-1,+1}.
    Also returns normalised MI = MI / H(Y) (∈ [0,1]).
    Returns (nan, nan) if too few samples.
    """
    n = len(y)
    if n < MIN_PAIRS_FOR_MI:
        return float("nan"), float("nan")

    # 2×2 joint counts
    c = np.zeros((2, 2), dtype=np.int64)
    yi = (y > 0).astype(int)   # +1→1, -1→0
    si = (s > 0).astype(int)
    np.add.at(c, (yi, si), 1)

    p = c / n
    py = p.sum(axis=1)  # marginal Y
    ps = p.sum(axis=0)  # marginal S

    mi = 0.0
    for i in range(2):
        for j in range(2):
            if p[i, j] > 0 and py[i] > 0 and ps[j] > 0:
                mi += p[i, j] * math.log2(p[i, j] / (py[i] * ps[j]))

    hy = -sum(p * math.log2(p) for p in py if p > 0)
    nmi = mi / hy if hy > 1e-12 else float("nan")
    return float(mi), float(nmi)


# ── directed BFS capped at max_nodes ─────────────────────────────────────────

def bfs_layer_map(src: int, adj: dict, max_depth: int, max_nodes: int) -> dict:
    """
    Returns {node: shortest_directed_distance} from src along directed edges.
    Stops when max_depth is reached or max_nodes nodes are visited.
    adj[u] = list of successor node ids (no signs needed here).
    """
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        d = dist[u]
        if d >= max_depth:
            continue
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = d + 1
                q.append(v)
                if len(dist) >= max_nodes:
                    return dist
    return dist


# ── Per-dataset analysis ──────────────────────────────────────────────────────

def analyse_dataset(ds_name: str, cfg: dict, out_dir: str) -> dict:
    print(f"\n{'─'*60}\n  {ds_name}\n{'─'*60}")

    edges = load_edges_canonical(cfg["ds_name"])
    print(f"  Total edges: {len(edges)}")

    # Build adjacency structures
    adj: dict[int, list[int]] = defaultdict(list)          # for BFS (unweighted)
    out_signed: dict[int, list[tuple[int,int]]] = defaultdict(list)  # (v, sign)
    edge_set: set[tuple[int,int]] = set()
    for u, v, s in edges:
        adj[u].append(v)
        out_signed[u].append((v, s))
        edge_set.add((u, v))

    rng = np.random.default_rng(42)

    # ── d=0: EXACT — all (Y, S) pairs from the same source node ──────────────
    # For each node u with out-edges {(v_i, s_i)}, every ordered pair
    # (anchor=(u,v_a,s_a), other=(u,v_b,s_b)) with a≠b contributes (s_a, s_b).
    Y0: list[int] = []
    S0: list[int] = []
    for u, nbrs in out_signed.items():
        if len(nbrs) < 2:
            continue
        ss = [s for _, s in nbrs]
        # Use all ordered pairs — exact enumeration
        # For large hubs (degree > 2000) subsample to keep memory bounded
        if len(ss) > 2000:
            idx = rng.choice(len(ss), 2000, replace=False)
            ss = [ss[i] for i in idx]
        for i, sa in enumerate(ss):
            for j, sb in enumerate(ss):
                if i != j:
                    Y0.append(sa)
                    S0.append(sb)
    print(f"  d=0 exact pairs: {len(Y0):,}")

    # ── d=1: EXACT — all (Y=sign(u→w), S=sign(w→x)) 2-hop chains ────────────
    Y1: list[int] = []
    S1: list[int] = []
    for u, v, y in edges:
        for x, sx in out_signed.get(v, []):
            if x == u:   # skip u→v→u (reciprocal)
                continue
            Y1.append(y)
            S1.append(sx)
    print(f"  d=1 exact pairs: {len(Y1):,}")

    # ── d=2..D_MAX: BFS from ALL anchors, capped per anchor ─────────────────
    Y_by_d: dict[int, list] = {d: [] for d in range(2, D_MAX + 1)}
    S_by_d: dict[int, list] = {d: [] for d in range(2, D_MAX + 1)}

    # For very large graphs subsample anchors to keep runtime manageable
    # (MI estimate converges quickly; 50k anchors is overkill for any dataset)
    MAX_ANCHORS_DEEP = min(len(edges), 50000)
    if len(edges) > MAX_ANCHORS_DEEP:
        anchor_idx = rng.choice(len(edges), MAX_ANCHORS_DEEP, replace=False)
        anchors = [edges[i] for i in anchor_idx]
    else:
        anchors = list(edges)
    print(f"  d>=2 anchors: {len(anchors):,}")

    for anchor_u, anchor_v, y in anchors:
        dist = bfs_layer_map(anchor_u, adj, D_MAX, MAX_BFS_NODES)

        nodes_at: dict[int, list] = defaultdict(list)
        for node, d in dist.items():
            if d >= 2:
                nodes_at[d].append(node)

        for d in range(2, D_MAX + 1):
            layer = nodes_at.get(d, [])
            if not layer:
                continue
            rng.shuffle(layer)
            collected = 0
            for a in layer:
                for b, s in out_signed.get(a, []):
                    if (a == anchor_u and b == anchor_v):
                        continue
                    Y_by_d[d].append(y)
                    S_by_d[d].append(s)
                    collected += 1
                    if collected >= MAX_PAIRS_PER_ANCHOR_DEEP:
                        break
                if collected >= MAX_PAIRS_PER_ANCHOR_DEEP:
                    break

    # ── Compute MI per depth ──────────────────────────────────────────────────
    results = {}
    print(f"\n  {'d':<4} {'n_pairs':>12}  {'MI (bits)':>12}  {'NMI':>8}")
    for d in range(0, D_MAX + 1):
        if d == 0:
            y_arr = np.array(Y0, dtype=np.int8)
            s_arr = np.array(S0, dtype=np.int8)
        elif d == 1:
            y_arr = np.array(Y1, dtype=np.int8)
            s_arr = np.array(S1, dtype=np.int8)
        else:
            y_arr = np.array(Y_by_d[d], dtype=np.int8)
            s_arr = np.array(S_by_d[d], dtype=np.int8)
        mi, nmi = mi_bits(y_arr, s_arr)
        tag = " (exact)" if d <= 1 else ""
        results[d] = {"mi": mi, "nmi": nmi, "n_pairs": len(y_arr)}
        print(f"  d={d}  {len(y_arr):>12,}  {mi:>12.6f}  {nmi:>8.5f}{tag}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    depths     = list(range(0, D_MAX + 1))
    mi_vals    = [results[d]["mi"]  for d in depths]
    nmi_vals   = [results[d]["nmi"] for d in depths]
    n_pairs    = [results[d]["n_pairs"] for d in depths]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, vals, ylabel, title in [
        (axes[0], mi_vals,  "MI (bits)",
         "Mutual Information vs directed distance"),
        (axes[1], nmi_vals, "Normalised MI  (MI / H(Y))",
         "Normalised MI vs directed distance"),
    ]:
        valid = [(d, v) for d, v in zip(depths, vals)
                 if not (isinstance(v, float) and math.isnan(v))]
        if valid:
            dx, vy = zip(*valid)
            ax.plot(dx, vy, marker="o", color="steelblue")
            ax.fill_between(dx, vy, alpha=0.15, color="steelblue")
        ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.set_xlabel("Directed hop distance d  (source → source)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(depths)

    # Secondary axis: n_pairs
    ax2 = axes[0].twinx()
    ax2.bar(depths, n_pairs, alpha=0.12, color="orange", label="n_pairs")
    ax2.set_ylabel("n_pairs (right)", color="orange")

    fig.suptitle(
        f"{ds_name}: MI(sign_target, sign_other) vs directed distance\n"
        f"(d=0,1 exact; d\u22652 all-anchors capped={MAX_PAIRS_PER_ANCHOR_DEEP}/anchor)",
        fontsize=12,
    )
    fig.tight_layout()
    save_path = os.path.join(out_dir, f"mi_vs_dist_{ds_name}.png")
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
    print(f"  ✓ Saved {os.path.basename(save_path)}")
    return results


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(all_results: dict, out_dir: str):
    lines = [
        "=" * 70,
        "  EDGE-SIGN MI vs DIRECTED DISTANCE REPORT",
        "=" * 70,
        "",
        "MI(Y_anchor, S_other) where dist = shortest directed path from",
        "source(anchor_edge) to source(other_edge).",
        "d=0: other outgoing edges from the SAME source node as the anchor.",
        "d=1: edges from direct out-neighbours.",
        "",
        f"Parameters: d=0,1 exact (all pairs); d>=2 up to {MAX_BFS_NODES} BFS nodes,",
        f"            capped at {MAX_PAIRS_PER_ANCHOR_DEEP} pairs/anchor, D_MAX={D_MAX}",
        "",
    ]
    for ds_name, res in all_results.items():
        lines += [f"{'─'*70}", f"  {ds_name}", f"{'─'*70}"]
        lines.append(f"  {'d':<4} {'n_pairs':>10}  {'MI (bits)':>12}  {'NMI':>8}")
        for d in range(0, D_MAX + 1):
            r = res[d]
            lines.append(
                f"  d={d}  {r['n_pairs']:>10}  {r['mi']:>12.6f}  {r['nmi']:>8.5f}"
            )
        lines.append("")

    path = os.path.join(out_dir, "mi_vs_dist_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n✓ Report: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--out", default="outputs/mi_vs_dist")
    args = parser.parse_args()

    datasets = list(DATASET_CONFIGS.keys()) if args.datasets == ["all"] else args.datasets

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    for ds in datasets:
        if ds not in DATASET_CONFIGS:
            print(f"Unknown dataset: {ds}")
            continue
        all_results[ds] = analyse_dataset(ds, DATASET_CONFIGS[ds], out_dir)

    if all_results:
        write_report(all_results, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
