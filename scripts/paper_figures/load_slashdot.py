"""Load the Slashdot signed edge dataset (see README-slashdot.txt)."""
import random
from collections import deque

import networkx as nx
import numpy
import matplotlib.pyplot as plt
from tqdm import tqdm

DEFAULT_PATH = "slashdot090221.edgelist"
MAX_DIST = 8
MAX_PAIRS = 10_000


def load_slashdot(path=DEFAULT_PATH):
    """Load the Slashdot dataset into a directed, signed graph.

    Each line in the file is "u v label" where label is 1 (positive/trust)
    or 0 (negative/distrust). Returns a nx.DiGraph with a "sign" edge
    attribute holding the original label.
    """
    graph = nx.DiGraph()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v, label = line.split()
            graph.add_edge(int(u), int(v), sign=int(label))
    return graph


def sign_distance_correlation(g, max_dist=MAX_DIST):
    """Correlate each edge's sign with the sign of edges found by BFS.

    For every edge (v1, v2), BFS outward from both endpoints together and,
    for each new edge discovered at hop distance d, pair the anchor edge's
    sign with that edge's sign. Returns a list of length max_dist where
    entry d-1 is the Pearson correlation coefficient at distance d.
    """
    coor = [[[], []] for _ in range(max_dist)]
    edge_bar = tqdm(total=1000, desc="edges", position=0)
    depth_bar = tqdm(total=max_dist, desc="bfs depth", position=1, leave=False)
    i = 0
    for v1, v2, data in g.edges(data=True):
        if (i >= 1000):
            break
        i += 1
        sign = data["sign"]
        q = deque([v1, v2])
        d = {v1: 0, v2: 0}
        depth_bar.reset(total=max_dist)
        depth_shown = 0
        while q:
            v = q.popleft()
            if d[v] >= max_dist:
                continue
            if d[v] != depth_shown:
                depth_bar.update(d[v] - depth_shown)
                depth_shown = d[v]
                depth_bar.set_postfix(visited=len(d), queued=len(q))
            for n in g.neighbors(v):
                if n not in d:
                    d[n] = d[v] + 1
                    q.append(n)
                    coor[d[n] - 1][0].append(sign)
                    coor[d[n] - 1][1].append(g.get_edge_data(v, n)["sign"])
        edge_bar.update(1)
    edge_bar.close()
    depth_bar.close()

    # [diagnostic addition, not a logic change] raw pre-cap pair counts, for
    # reporting alongside the correlation values below.
    n_pairs_precap = [len(x) for x, y in coor]

    for x, y in coor:
        if len(x) > MAX_PAIRS:
            keep = sorted(random.sample(range(len(x)), MAX_PAIRS))
            kept_x = [x[i] for i in keep]
            kept_y = [y[i] for i in keep]
            x[:] = kept_x
            y[:] = kept_y

    n_pairs_postcap = [len(x) for x, y in coor]
    corrs = [numpy.corrcoef(x, y)[0, 1] if len(x) > 1 else float("nan") for x, y in coor]
    return corrs, n_pairs_precap, n_pairs_postcap


if __name__ == "__main__":
    # [diagnostic addition] point at the real exported edgelist (same file
    # handed to the external reviewer) instead of a bare relative filename.
    g = load_slashdot(path="aaai2027/external_review/slashdot090221.edgelist")
    correlations, n_precap, n_postcap = sign_distance_correlation(g)
    distances = range(1, len(correlations) + 1)

    # [diagnostic addition] print raw numbers, not just a plot.
    print("\ndist  n_pairs(pre-cap)  n_pairs(used)  correlation")
    for d, c, npre, npost in zip(distances, correlations, n_precap, n_postcap):
        print(f"{d:4d}  {npre:>15,}  {npost:>13,}  {c: .6f}")

    plt.plot(distances, correlations, marker="o")
    plt.xlabel("BFS distance from edge")
    plt.ylabel("Correlation with edge sign")
    plt.title("Sign correlation vs. BFS distance (Slashdot)")
    plt.savefig("slashdot_sign_correlation.png")
    # [diagnostic addition] plt.show() dropped -- this runs headless here;
    # the saved PNG above is the only output needed.

