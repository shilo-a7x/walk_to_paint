"""
Demonstrates that the mismatch between our reproduction of the colleague's
`load_slashdot.py` and her own previously-saved plot isn't a bug in the
reproduction -- it's because her `sign_distance_correlation()` calls
`random.sample()` with NO seed set anywhere in the file. Whenever a
distance's pre-cap pair count exceeds her `MAX_PAIRS=10,000`, every run
draws a different random 10k subsample, so the correlation at that distance
differs on every invocation, even on the identical graph with identical code.

See PANELB_INVESTIGATION_REPORT.md at repo root for the full writeup and the
resulting 5-run comparison table.

Strategy: run the BFS/pair-collection step (the expensive part, ~110s for
1,000 anchors) ONCE, then apply the random-subsample-and-correlate step 5
separate times on the exact same underlying pair pool -- isolating how much
variance comes purely from the unseeded random.sample() draw, at zero
re-computation cost.

Run: .venv/bin/python scripts/panelb_diagnostics/demo_slashdot_cap_nondeterminism.py
"""
import os
import random
from collections import deque

import networkx as nx
import numpy
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAX_DIST = 8
MAX_PAIRS = 10_000
PATH = os.path.join(ROOT, "aaai2027", "external_review", "slashdot090221.edgelist")


def load_slashdot(path):
    graph = nx.DiGraph()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v, label = line.split()
            graph.add_edge(int(u), int(v), sign=int(label))
    return graph


def collect_pairs(g, max_dist=MAX_DIST):
    """Her exact BFS/discovery-edge-recording logic (directed, tree-edge
    only), stopped just short of the random-capping step so it can be
    applied repeatedly without re-running the BFS."""
    coor = [[[], []] for _ in range(max_dist)]
    i = 0
    for v1, v2, data in tqdm(g.edges(data=True), total=1000, desc="edges"):
        if i >= 1000:
            break
        i += 1
        sign = data["sign"]
        q = deque([v1, v2])
        d = {v1: 0, v2: 0}
        while q:
            v = q.popleft()
            if d[v] >= max_dist:
                continue
            for n in g.neighbors(v):
                if n not in d:
                    d[n] = d[v] + 1
                    q.append(n)
                    coor[d[n] - 1][0].append(sign)
                    coor[d[n] - 1][1].append(g.get_edge_data(v, n)["sign"])
    return coor


def cap_and_corr(coor, max_pairs=MAX_PAIRS):
    """Exactly her capping+correlation logic, applied fresh (new random draw
    each call since random.sample() is never seeded, matching her script)."""
    out = []
    for x, y in coor:
        if len(x) > max_pairs:
            keep = sorted(random.sample(range(len(x)), max_pairs))
            xs = [x[i] for i in keep]
            ys = [y[i] for i in keep]
        else:
            xs, ys = x, y
        out.append(numpy.corrcoef(xs, ys)[0, 1] if len(xs) > 1 else float("nan"))
    return out


def main():
    print("loading graph + collecting raw (uncapped) pairs (one-time BFS cost)...")
    g = load_slashdot(PATH)
    coor = collect_pairs(g)
    n_precap = [len(x) for x, y in coor]
    print(f"pre-cap pair counts per distance: {n_precap}")

    print("\napplying the SAME capping+correlation step 5 times on the SAME raw pairs")
    print("(only the random.sample() draw changes each time -- no seed in her script):\n")
    print(f"{'run':>4} " + " ".join(f"d={d:<8}" for d in range(1, MAX_DIST + 1)))
    for run in range(5):
        corrs = cap_and_corr(coor)
        print(f"{run:>4} " + " ".join(f"{c:8.4f}" for c in corrs))


if __name__ == "__main__":
    main()
