"""
Rigorous check that her (v2) per-edge BFS distance label is mathematically
identical to production's line-graph distance (min(shell(x), shell(y)) + 1),
now that both use undirected BFS. See PANELB_INVESTIGATION_REPORT.md sec.
3.6 for the accompanying formal proof and full result discussion.

(1) Cross-validates our own shell computation against networkx's ground-truth
    `single_source_shortest_path_length` (same bar the 2026-07-28 investigation
    used for production's shells) -- note the `cutoff=max_dist + 1`: a virtual
    super-node is added connected to both anchor endpoints, so nx's distances
    from the super-node are exactly 1 hop more than the true anchor-relative
    distance; using cutoff=max_dist here (without the +1) undercounts nodes at
    the true boundary distance -- this bit us on a first pass (12 false
    "mismatches", all at exactly ours=max_dist), fixed by the +1.
(2) Checks her per-edge distance label against min(shell)+1 across a large,
    full-coverage sample (500 anchors, EVERY recorded edge checked, not a
    subsample of the check itself) including explicit stress cases: self-loop
    anchors (v1==v2, if any) and nodes with multiple tied potential discovery
    parents (checks the proof's claim that the label doesn't depend on which
    parent BFS happens to pick).

Run: .venv/bin/python scripts/panelb_diagnostics/verify_distance_definitions.py
"""
import os
import random
import multiprocessing as mp
from collections import deque

import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = os.path.join(ROOT, "aaai2027", "external_review", "slashdot090221.edgelist")
MAX_DIST = 8
N_ANCHORS_PART2 = 500

# fork-inherited global, set via _set_global_graph() before Pool() is
# constructed -- see load_slashdot_v2_parallel.py for why this (not
# Pool(initializer=...)) is the right pattern under a fork context.
_G = None


def _set_global_graph(g):
    global _G
    _G = g


def load_graph(path):
    g = nx.Graph()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v, label = line.split()
            g.add_edge(int(u), int(v), sign=int(label))
    return g


def our_shells(g, v1, v2, max_dist=MAX_DIST):
    shell = {v1: 0, v2: 0} if v1 != v2 else {v1: 0}
    frontier = [v1, v2] if v1 != v2 else [v1]
    for dist in range(1, max_dist + 1):
        nxt = []
        for w in frontier:
            for nb in g.neighbors(w):
                if nb not in shell:
                    shell[nb] = dist
                    nxt.append(nb)
        if not nxt:
            break
        frontier = nxt
    return shell


def _check_one_anchor(edge):
    """Worker body: identical logic to the original sequential Part 2 loop
    for a single anchor -- her BFS-based label vs. min(shell)+1, plus the
    tie-parent stress check. Returns (n_checked, n_mismatch, n_tie_edges,
    is_self_loop, [mismatch_log_lines])."""
    g = _G
    v1, v2, data = edge
    self_loop = 1 if v1 == v2 else 0

    her_label = {}
    d = {v1: 0, v2: 0} if v1 != v2 else {v1: 0}
    q = deque([v1, v2]) if v1 != v2 else deque([v1])
    while q:
        v = q.popleft()
        if d[v] >= MAX_DIST:
            continue
        for n in g.neighbors(v):
            if n not in d:
                d[n] = d[v] + 1
                q.append(n)
                her_label[frozenset((v, n))] = d[n]

    shell = our_shells(g, v1, v2)

    total_checked = mismatches = tie_edges_checked = 0
    log_lines = []
    for (a, b), her_d in her_label.items():
        prod_d = min(shell[a], shell[b]) + 1
        total_checked += 1
        b_node = b if shell[b] > shell[a] else a
        parent_shell = shell[b_node] - 1
        n_possible_parents = sum(1 for nb in g.neighbors(b_node)
                                  if nb in shell and shell[nb] == parent_shell)
        if n_possible_parents > 1:
            tie_edges_checked += 1
        if prod_d != her_d:
            mismatches += 1
            log_lines.append(f"  MISMATCH anchor=({v1},{v2}) edge=({a},{b}) "
                              f"her={her_d} prod={prod_d}")

    return total_checked, mismatches, tie_edges_checked, self_loop, log_lines


def main():
    g = load_graph(PATH)
    edges = list(g.edges(data=True))
    n_self_loops = sum(1 for u, v, _ in edges if u == v)
    print(f"graph: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges, "
          f"self-loops={n_self_loops}")

    # --- Part 1: our shell computation vs networkx ground truth ---
    random.seed(1)
    gt_anchors = random.sample(range(len(edges)), 15)
    gt_checked = gt_mismatches = 0
    for eidx in gt_anchors:
        v1, v2, _ = edges[eidx]
        ours = our_shells(g, v1, v2)
        g2 = g.copy()
        g2.add_node("__SUPER__")
        g2.add_edge("__SUPER__", v1)
        if v2 != v1:
            g2.add_edge("__SUPER__", v2)
        nx_dist = nx.single_source_shortest_path_length(g2, "__SUPER__", cutoff=MAX_DIST + 1)
        for node, d in ours.items():
            gt_checked += 1
            nxd = nx_dist.get(node, None)
            if nxd is None or nxd - 1 != d:
                gt_mismatches += 1
                print(f"  GT MISMATCH anchor=({v1},{v2}) node={node} ours={d} "
                      f"nx={nxd - 1 if nxd is not None else None}")
    print(f"\n[Part 1] our shell computation vs networkx ground truth: "
          f"{gt_checked:,} node-shells checked across {len(gt_anchors)} anchors, "
          f"mismatches={gt_mismatches}")

    # --- Part 2: her per-edge distance label vs min(shell)+1, large sample,
    # parallelized across anchors (each anchor's check is fully independent
    # of every other anchor's -- fork-based Pool, globals set before Pool()
    # is constructed so workers inherit the graph via copy-on-write, same
    # pattern as load_slashdot_v2_parallel.py) ---
    random.seed(42)
    big_sample = random.sample(range(len(edges)), N_ANCHORS_PART2)

    _set_global_graph(g)
    with mp.get_context("fork").Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(_check_one_anchor, [edges[i] for i in big_sample])

    total_checked = sum(r[0] for r in results)
    mismatches = sum(r[1] for r in results)
    tie_edges_checked = sum(r[2] for r in results)
    self_loop_anchors = sum(r[3] for r in results)
    for r in results:
        for line in r[4]:
            print(line)

    print(f"\n[Part 2] her label vs min(shell)+1: {total_checked:,} recorded edges "
          f"across {len(big_sample)} anchors ({self_loop_anchors} self-loop anchors)")
    print(f"  mismatches: {mismatches}")
    print(f"  of those, edges whose discovered node had >1 tied possible parent: "
          f"{tie_edges_checked:,} (checks the proof holds regardless of which "
          f"parent BFS happened to pick)")


if __name__ == "__main__":
    main()
