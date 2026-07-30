"""Load the Slashdot signed edge dataset (see README-slashdot.txt).

Instrumented copy of the user-uploaded `load_slashdot (1).py` (repo root,
left untouched). Changes from that original, and ONLY these:
  1. [seed] `random.seed(datetime.now().timestamp())` -> `random.seed(42)`,
     per explicit request (pins the run-to-run-unstable MAX_PAIRS cap draw
     to a fixed, reproducible value instead of the current timestamp).
  2. [bugfix] `depth_bar.close()` removed -- `depth_bar` is never
     constructed in this version of the script (only `edge_bar` is), so
     this line raises NameError before any output is produced. Pure
     pre-existing bug, unrelated to the seed change.
  3. [bugfix] `print(str(i+1) + " " + correlations[i])` -> prints both as
     str() -- `correlations[i]` is a float, so `str + float` raises
     TypeError as originally written. Pure pre-existing bug.
  4. [path] DEFAULT_PATH left as-is, but __main__ passes the real repo path
     to the exported edgelist (her bare relative "slashdot090221.edgelist"
     doesn't exist at repo root) -- same fix applied to the first version
     of her script.
  5. [headless] `plt.show()` dropped -- this runs headless here; the saved
     PNG is the only output needed.

No other line, no algorithmic behavior (graph directedness, BFS logic,
anchor sampling, recording convention, or the MAX_PAIRS cap mechanism
itself) was changed.
"""
import random
from collections import deque

import networkx as nx
import numpy
import matplotlib.pyplot as plt
from tqdm import tqdm

random.seed(42)

DEFAULT_PATH = "slashdot090221.edgelist"
MAX_DIST = 8
MAX_PAIRS = 10_000


def load_slashdot(path=DEFAULT_PATH):
    """Load the Slashdot dataset into a directed, signed graph.

    Each line in the file is "u v label" where label is 1 (positive/trust)
    or 0 (negative/distrust). Returns a nx.DiGraph with a "sign" edge
    attribute holding the original label.
    """
    graph = nx.Graph()
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
    edge_bar = tqdm(total=10000, desc="edges", position=0)
    edge_numbers = random.sample(range(0, len(g.edges())), 10000)
    for edge_id in edge_numbers:
        v1, v2, data = list(g.edges(data=True))[edge_id]
        sign = data["sign"]
        q = deque([v1, v2])
        d = {v1: 0, v2: 0}
        depth_shown = 0
        while q:
            v = q.popleft()
            if d[v] >= max_dist:
                continue
            if d[v] != depth_shown:
                depth_shown = d[v]
            for n in g.neighbors(v):
                if n not in d:
                    d[n] = d[v] + 1
                    q.append(n)
                    coor[d[n] - 1][0].append(sign)
                    coor[d[n] - 1][1].append(g.get_edge_data(v, n)["sign"])
        edge_bar.update(1)
    edge_bar.close()
    # [bugfix, see module docstring #2] depth_bar.close() removed -- depth_bar
    # is never constructed in this version.

    for x, y in coor:
        if len(x) > MAX_PAIRS:
            keep = sorted(random.sample(range(len(x)), MAX_PAIRS))
            kept_x = [x[i] for i in keep]
            kept_y = [y[i] for i in keep]
            x[:] = kept_x
            y[:] = kept_y

    return [numpy.corrcoef(x, y)[0, 1] for x, y in coor]


if __name__ == "__main__":
    # [path fix, see module docstring #4] point at the real exported edgelist.
    g = load_slashdot(path="aaai2027/external_review/slashdot090221.edgelist")
    correlations = sign_distance_correlation(g)
    distances = range(1, len(correlations) + 1)

    for i in range(len(correlations)):
        # [bugfix, see module docstring #3] str() around correlations[i].
        print(str(i + 1) + " " + str(correlations[i]))
    plt.plot(distances, correlations, marker="o")
    plt.xlabel("BFS distance from edge")
    plt.ylabel("Correlation with edge sign")
    plt.title("Sign correlation vs. BFS distance (Slashdot)")
    plt.savefig("slashdot_sign_correlation_v2_seed42.png")
    # [headless, see module docstring #5] plt.show() dropped.
