import random
import math
import multiprocessing as mp
from collections import defaultdict
from functools import partial
import numpy as np


def _build_adj(edges):
    """Build adjacency lists and indexable arrays for fast sampling."""
    graph = defaultdict(list)
    for u, v, label in edges:
        graph[u].append((v, label))
    nodes = list(graph.keys())
    # For each node, store separate arrays for neighbors and labels for vectorized choice
    nbrs = {}
    lbls = {}
    for n, items in graph.items():
        if items:
            vs, ls = zip(*items)
            nbrs[n] = np.fromiter(vs, dtype=np.int64)
            lbls[n] = np.fromiter(ls, dtype=np.int64)
        else:
            nbrs[n] = np.array([], dtype=np.int64)
            lbls[n] = np.array([], dtype=np.int64)
    return nodes, nbrs, lbls


def _sample_chunk(
    nodes, nbrs, lbls, num_walks, max_walk_length, start_idx, end_idx, seed
):
    """Sample a chunk of walks [start_idx, end_idx) using NumPy RNG (fast)."""
    rng = np.random.default_rng(seed)
    walks = []
    node_arr = np.fromiter(nodes, dtype=object)
    for _ in range(start_idx, end_idx):
        start = node_arr[rng.integers(0, len(node_arr))]
        walk_nodes = [start]
        walk_tokens = [f"N_{start}"]
        curr = start
        for _ in range(max_walk_length):
            neigh = nbrs.get(curr)
            if neigh is None or len(neigh) == 0:
                break
            idx = rng.integers(0, len(neigh))
            next_node = int(neigh[idx])
            edge_label = int(lbls[curr][idx])
            walk_tokens.append(f"E_{edge_label}")
            walk_tokens.append(f"N_{next_node}")
            walk_nodes.append(next_node)
            curr = next_node
        walks.append(walk_tokens)
    return walks


def sample_random_walks(
    edges, num_walks=100, max_walk_length=16, num_workers=1, seed=None
):
    """
    Fast random-walk sampler with optional multiprocessing.

    Args:
        edges: iterable of (u, v, label)
        num_walks: total number of walks to generate
        max_walk_length: max edges per walk
        num_workers: processes for parallel sampling (>=1)
        seed: base seed for reproducibility (int)
    """
    nodes, nbrs, lbls = _build_adj(edges)
    if not nodes:
        return []

    # Fallback to single process when workers <=1
    num_workers = max(1, int(num_workers))
    base_seed = 42 if seed is None else int(seed)

    if num_workers == 1:
        return _sample_chunk(
            nodes, nbrs, lbls, num_walks, max_walk_length, 0, num_walks, base_seed
        )

    # Split work evenly
    chunk = math.ceil(num_walks / num_workers)
    tasks = []
    start = 0
    for w in range(num_workers):
        end = min(start + chunk, num_walks)
        if start >= end:
            break
        tasks.append((start, end, base_seed + w))
        start = end

    # Each chunk gets its own deterministic seed to avoid overlap
    with mp.Pool(processes=len(tasks)) as pool:
        results = pool.starmap(
            partial(_sample_chunk, nodes, nbrs, lbls, num_walks, max_walk_length),
            tasks,
        )

    # Flatten results
    walks = []
    for chunk_walks in results:
        walks.extend(chunk_walks)
    return walks
