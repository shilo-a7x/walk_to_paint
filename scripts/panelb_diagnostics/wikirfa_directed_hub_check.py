"""Diagnostic-only: hub-concentration + node-role check for wiki-rfa's DIRECTED
Panel B bump at line-graph distance 7-8 (mirrors PANELB_INVESTIGATION_REPORT.md's
Thread A hub-concentration measurement, done there only for slashdot090221
undirected -- this extends the same question to wiki-rfa directed, which was
never checked).

For every anchor edge, directed BFS (successors only) out to d_max=7, but instead
of accumulating a 2x2 contingency table (what the production extractor does),
this records which context edges land at line-graph distance 7 or 8 and how many
times each one is hit across all anchors -- the same "how concentrated is the
pooled sample on a few repeated edges" question Thread A asked for slashdot.

Also reports, for the context edges landing at d=7/8: their own sign mix, and
the in-/out-degree of their source/target nodes -- to check the candidate
mechanism that a small number of very high-in-degree nodes (e.g. admin/RfA
candidates, already documented in CLAUDE.md's Lead4c section as a real
structural feature of the two wiki datasets) dominate the outer shell.

Run: .venv/bin/python scripts/panelb_diagnostics/wikirfa_directed_hub_check.py
"""
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.balance_theory_paths import load_edges_canonical, DATASET_CONFIGS

DS_NAME = "wiki-rfa"
D_MAX = 7  # line-graph distance up to D_MAX+1 = 8
TARGET_LINE_DISTS = {7, 8}


def main():
    t0 = time.time()
    edges = load_edges_canonical(DS_NAME)
    nodes = sorted({n for u, v, s in edges for n in (u, v)})
    n2i = {n: i for i, n in enumerate(nodes)}
    N, E = len(nodes), len(edges)
    edge_x = np.empty(E, np.int32)
    edge_y = np.empty(E, np.int32)
    edge_s = np.empty(E, np.int8)
    adj = [[] for _ in range(N)]        # directed: out-neighbors only
    indeg = np.zeros(N, np.int32)
    outdeg = np.zeros(N, np.int32)
    for eid, (u, v, s) in enumerate(edges):
        ui, vi = n2i[u], n2i[v]
        edge_x[eid], edge_y[eid], edge_s[eid] = ui, vi, s
        adj[ui].append((vi, eid))
        outdeg[ui] += 1
        indeg[vi] += 1
    print(f"N={N:,} E={E:,} loaded in {time.time()-t0:.1f}s")

    shell = np.full(N, -1, np.int32)
    gen = np.zeros(N, np.int32)
    context_hits = Counter()   # eid2 -> count across all anchors, at target dists
    dist_of_hit = {}           # eid2 -> the (min) line_dist it was seen at
    n_target_pairs = 0

    t1 = time.time()
    for cur_gen, eid in enumerate(range(E), start=1):
        u, v, y = int(edge_x[eid]), int(edge_y[eid]), int(edge_s[eid])
        roots = (u, v) if u != v else (u,)
        reached = []
        cur_frontier = []
        for w in roots:
            if gen[w] != cur_gen:
                gen[w] = cur_gen
                shell[w] = 0
                reached.append(w)
                cur_frontier.append(w)
        for d in range(1, D_MAX + 1):
            nxt = []
            for w in cur_frontier:
                for nb, _eid2 in adj[w]:
                    if gen[nb] != cur_gen:
                        gen[nb] = cur_gen
                        shell[nb] = d
                        nxt.append(nb)
            if not nxt:
                break
            reached.extend(nxt)
            cur_frontier = nxt

        candidate_eids = set()
        for w in reached:
            for _nb, eid2 in adj[w]:
                if eid2 != eid:
                    candidate_eids.add(eid2)

        for eid2 in candidate_eids:
            x2, y2 = int(edge_x[eid2]), int(edge_y[eid2])
            sh_x = int(shell[x2]) if gen[x2] == cur_gen else D_MAX + 1
            sh_y = int(shell[y2]) if gen[y2] == cur_gen else D_MAX + 1
            d_e = min(sh_x, sh_y)
            if d_e > D_MAX:
                continue
            line_d = d_e + 1
            if line_d in TARGET_LINE_DISTS:
                context_hits[eid2] += 1
                dist_of_hit.setdefault(eid2, line_d)
                n_target_pairs += 1

        if cur_gen % 20000 == 0:
            print(f"  {cur_gen:,}/{E:,} anchors, {time.time()-t1:.1f}s elapsed, "
                  f"{n_target_pairs:,} target-distance pairs so far")

    print(f"BFS sweep done in {time.time()-t1:.1f}s, total target-distance pairs={n_target_pairs:,}")

    # --- hub concentration ---
    n_distinct = len(context_hits)
    top10 = context_hits.most_common(10)
    top10_share = sum(c for _, c in top10) / n_target_pairs if n_target_pairs else float("nan")
    print(f"\ndistinct context edges hit at d in {{7,8}}: {n_distinct:,}")
    print(f"top-10 edges' share of all target-distance pairs: {top10_share:.1%}")
    print("top-10 edges (eid, count, u, v, sign, indeg[v], outdeg[u]):")
    for eid2, cnt in top10:
        x2, y2, s2 = int(edge_x[eid2]), int(edge_y[eid2]), int(edge_s[eid2])
        print(f"  eid={eid2} count={cnt} u={x2} v={y2} sign={s2} "
              f"indeg(v)={indeg[y2]} outdeg(u)={outdeg[x2]} indeg(u)={indeg[x2]} outdeg(v)={outdeg[y2]}")

    # --- sign mix + degree profile of ALL context edges at d in {7,8} vs. dataset-wide baseline ---
    hit_eids = np.array(list(context_hits.keys()))
    hit_weights = np.array([context_hits[e] for e in hit_eids])
    hit_signs = edge_s[hit_eids].astype(np.float64)
    pos_frac_weighted = float((hit_signs * hit_weights).sum() / hit_weights.sum())
    pos_frac_distinct = float(hit_signs.mean())
    pos_frac_baseline = float((edge_s.astype(np.float64) > 0).mean())
    print(f"\n% positive, target-distance-edges (pair-weighted): {pos_frac_weighted:.1%}")
    print(f"% positive, target-distance-edges (distinct-edge, unweighted): {pos_frac_distinct:.1%}")
    print(f"% positive, dataset-wide baseline (all {E:,} edges): {pos_frac_baseline:.1%}")

    hit_v_indeg = indeg[edge_y[hit_eids]]
    hit_u_outdeg = outdeg[edge_x[hit_eids]]
    print(f"\nmean indeg(v) over distinct target-distance context edges: {hit_v_indeg.mean():.1f} "
          f"(dataset-wide mean indeg: {indeg.mean():.1f}, max indeg: {indeg.max()})")
    print(f"mean outdeg(u) over distinct target-distance context edges: {hit_u_outdeg.mean():.1f} "
          f"(dataset-wide mean outdeg: {outdeg.mean():.1f}, max outdeg: {outdeg.max()})")

    print(f"\ntotal wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
