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
    nodes, nbrs, lbls, num_walks, max_walk_length, start_idx, end_idx, base_seed, task_id=0
):
    """Sample a chunk of walks [start_idx, end_idx) using NumPy RNG (fast).
    
    CRITICAL: Each walk uses its absolute index (walk_idx) to seed the RNG.
    This ensures reproducibility regardless of how work is divided across workers.
    
    Args:
        base_seed: Base seed for reproducibility
        task_id: Used to track ordering in multiprocessing
    """
    walks = []
    node_arr = np.fromiter(nodes, dtype=object)
    
    # CRITICAL: Each walk gets deterministic seed based on absolute walk index
    for walk_idx in range(start_idx, end_idx):
        # Seed RNG uniquely for this specific walk (not just per-chunk)
        # This ensures walk[i] is identical regardless of multiprocessing
        rng = np.random.default_rng(base_seed + walk_idx)
        
        # Select starting node deterministically for this walk
        start = node_arr[rng.integers(0, len(node_arr))]
        walk_nodes = [start]
        walk_tokens = [f"N_{start}"]
        curr = start
        
        # Generate walk steps
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
    
    return (task_id, walks)


def sample_random_walks(
    edges, num_walks=100, max_walk_length=16, num_workers=1, seed=None
):
    """
    Fast random-walk sampler with optional multiprocessing.
    
    REPRODUCIBILITY GUARANTEE: Exact same walks in exact same order every run.
    
    How this works:
    1. Each walk[i] uses seed = base_seed + i (deterministic based on walk index)
    2. Starting node for walk[i] is selected using rng seeded with (base_seed + i)
       Example: walk[0] starts at node_arr[rng_0.integers(...)], 
                walk[1] starts at node_arr[rng_1.integers(...)]
    3. All walk steps use this same rng, so walk[i] always identical
    4. Multiprocessing distributes chunks to workers, but each walk[i] maintains its index
    5. Results sorted by walk index (via task_id boundaries) to ensure correct file order
    6. Result: walks[0] through walks[n] always identical and in same order, regardless
       of which worker completes first or how many workers used.

    Args:
        edges: iterable of (u, v, label)
        num_walks: total number of walks to generate
        max_walk_length: max edges per walk
        num_workers: processes for parallel sampling (>=1)
        seed: base seed for reproducibility (int)
        
    Returns:
        List of walks, where walks[i] is always identical and in same order every run.
    """
    nodes, nbrs, lbls = _build_adj(edges)
    if not nodes:
        return []

    # Fallback to single process when workers <=1
    num_workers = max(1, int(num_workers))
    base_seed = 42 if seed is None else int(seed)

    if num_workers == 1:
        _, walks = _sample_chunk(
            nodes, nbrs, lbls, num_walks, max_walk_length, 0, num_walks, base_seed=base_seed, task_id=0
        )
        return walks

    # Split work evenly, track task IDs for deterministic ordering
    chunk = math.ceil(num_walks / num_workers)
    tasks = []
    start = 0
    for w in range(num_workers):
        end = min(start + chunk, num_walks)
        if start >= end:
            break
        # Each task: (start, end, base_seed, task_id)
        # CRITICAL: All workers use same base_seed; each walk adds its index
        tasks.append((start, end, base_seed, w))
        start = end

    # Each worker processes a chunk, but each walk uses its absolute index for seeding
    # Results include task_id to preserve ordering across different systems
    with mp.Pool(processes=len(tasks)) as pool:
        results = pool.starmap(
            partial(_sample_chunk, nodes, nbrs, lbls, num_walks, max_walk_length),
            [(start, end, base_seed, task_id) for start, end, base_seed, task_id in tasks],
        )

    # CRITICAL: Sort results by task_id to ensure deterministic walk order
    # Pool may complete tasks out-of-order, so we must re-sort by task_id.
    # This restores the original walk order: walks[0..chunk-1] from task0, 
    # walks[chunk..2*chunk-1] from task1, etc.
    results.sort(key=lambda x: x[0])  # Sort by task_id
    
    # Flatten results in task order (not completion order)
    walks = []
    for task_id, chunk_walks in results:
        walks.extend(chunk_walks)
    return walks
