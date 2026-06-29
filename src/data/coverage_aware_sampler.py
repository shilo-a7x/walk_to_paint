"""
Coverage-aware walk sampling strategies for signed graphs.

All strategies here are leakage-free: they use only graph topology (edge
existence, node degrees) and training-split edge signs.  Val/test edge *signs*
are NEVER used to guide sampling — they are the classification targets.

Entry point
-----------
    sample_walks(edges, cfg, seed, num_workers, train_set, mask_set)

Dispatches to the chosen strategy via cfg.dataset.walk_strategy.
"""

import heapq
import math
import multiprocessing as mp
import random
from collections import defaultdict
from functools import partial

import numpy as np

from src.data.walk_sampler import sample_random_walks, _build_adj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_edge_index(edges):
    """Map each unique (u, v, label) to a compact 0-based int ID.

    Uses insertion order so IDs are exactly 0..m-1 where m = len(result).
    Duplicate (u,v,label) entries in `edges` (possible with multiedge_handling=keep)
    are deduplicated — first occurrence wins.
    """
    index = {}
    for u, v, label in edges:
        key = (int(u), int(v), int(label))
        if key not in index:
            index[key] = len(index)
    return index


def _walk_to_edge_ids(walk, edge_index):
    """Convert a walk token list to a sorted array of edge indices it traverses."""
    ids = []
    n = len(walk)
    for i, token in enumerate(walk):
        if token.startswith("E_") and i > 0 and i < n - 1:
            try:
                u = int(walk[i - 1][2:])  # N_<id>
                v = int(walk[i + 1][2:])
                label = int(token[2:])
            except (ValueError, IndexError):
                continue
            eid = edge_index.get((u, v, label), -1)
            if eid >= 0:
                ids.append(eid)
    return np.array(ids, dtype=np.int32)


def _sample_one_walk(nbrs, lbls, start, max_walk_length, rng):
    """Walk from `start` using the provided RNG; returns token list."""
    walk_tokens = [f"N_{start}"]
    curr = start
    for _ in range(max_walk_length):
        neigh = nbrs.get(curr)
        if neigh is None or len(neigh) == 0:
            break
        idx = rng.integers(0, len(neigh))
        nxt = int(neigh[idx])
        lbl = int(lbls[curr][idx])
        walk_tokens.append(f"E_{lbl}")
        walk_tokens.append(f"N_{nxt}")
        curr = nxt
    return walk_tokens


def _chunk_tasks(num_items, num_workers):
    """Create deterministic (start, end, task_id) chunks."""
    num_workers = max(1, int(num_workers))
    if num_workers == 1:
        return [(0, num_items, 0)]
    chunk = math.ceil(num_items / num_workers)
    tasks = []
    start = 0
    task_id = 0
    while start < num_items:
        end = min(start + chunk, num_items)
        tasks.append((start, end, task_id))
        start = end
        task_id += 1
    return tasks


def _flatten_chunked(results):
    """Sort by task_id and flatten (task_id, walks) results."""
    results.sort(key=lambda x: x[0])
    out = []
    for _, walks in results:
        out.extend(walks)
    return out


def _sample_from_starts_chunk(
    nbrs, lbls, start_nodes, max_walk_length, base_seed,
    start_idx, end_idx, task_id,
):
    """Worker chunk: sample walks for precomputed start nodes."""
    rng = np.random.default_rng(base_seed + task_id)
    walks = []
    for i in range(start_idx, end_idx):
        start = int(start_nodes[i])
        walks.append(_sample_one_walk(nbrs, lbls, start, max_walk_length, rng))
    return task_id, walks


def _node2vec_chunk(
    nbrs, lbls, nbr_sets, start_nodes, max_walk_length, p, q, base_seed,
    start_idx, end_idx, task_id,
):
    """Worker chunk for node2vec-biased walks."""
    rng = np.random.default_rng(base_seed + task_id)
    walks = []
    empty = set()

    for i in range(start_idx, end_idx):
        start = int(start_nodes[i])
        walk_tokens = [f"N_{start}"]
        curr = start
        prev = None

        for _ in range(max_walk_length):
            neigh = nbrs.get(curr)
            if neigh is None or len(neigh) == 0:
                break

            weights = np.empty(len(neigh), dtype=np.float64)
            prev_nbrs = nbr_sets.get(prev, empty) if prev is not None else empty
            for j, nxt in enumerate(neigh):
                nxt = int(nxt)
                if prev is not None and nxt == prev:
                    weights[j] = 1.0 / p
                elif prev is not None and nxt in prev_nbrs:
                    weights[j] = 1.0
                else:
                    weights[j] = 1.0 / q
            weights /= weights.sum()
            idx = rng.choice(len(neigh), p=weights)
            nxt = int(neigh[idx])
            lbl = int(lbls[curr][idx])
            walk_tokens.append(f"E_{lbl}")
            walk_tokens.append(f"N_{nxt}")
            prev = curr
            curr = nxt

        walks.append(walk_tokens)

    return task_id, walks


def _neg_traversal_chunk(
    nbrs, lbls, rev_nbrs_arr, rev_lbls_arr, neg_arr,
    max_walk_length, base_seed,
    start_idx, end_idx, task_id,
):
    """Worker chunk for training-negative traversal walks."""
    rng = np.random.default_rng(base_seed + task_id)
    walks = []
    n_neg = len(neg_arr)
    prefix_len = max_walk_length // 3

    for _ in range(start_idx, end_idx):
        target_u, target_v, target_lbl = neg_arr[rng.integers(0, n_neg)]

        prefix_tokens = []
        curr = target_u
        for _k in range(prefix_len):
            rn = rev_nbrs_arr.get(curr)
            if rn is None or len(rn) == 0:
                break
            idx = rng.integers(0, len(rn))
            prev = int(rn[idx])
            lbl = int(rev_lbls_arr[curr][idx])
            prefix_tokens = [f"N_{prev}", f"E_{lbl}"] + prefix_tokens
            curr = prev
        prefix_tokens.append(f"N_{target_u}")

        mid_tokens = [f"E_{target_lbl}", f"N_{target_v}"]

        suffix_tokens = []
        curr = target_v
        remaining = max_walk_length - prefix_len - 1
        for _k in range(remaining):
            neigh = nbrs.get(curr)
            if neigh is None or len(neigh) == 0:
                break
            idx = rng.integers(0, len(neigh))
            nxt = int(neigh[idx])
            lbl = int(lbls[curr][idx])
            suffix_tokens.append(f"E_{lbl}")
            suffix_tokens.append(f"N_{nxt}")
            curr = nxt

        walks.append(prefix_tokens + mid_tokens + suffix_tokens)

    return task_id, walks


def _guaranteed_anchor_chunk(
    nbrs, lbls, local_edges, max_walk_length, base_seed, task_id,
):
    """Worker chunk for guaranteed anchor walks."""
    rng_anchor = np.random.default_rng(base_seed + task_id)
    local_walks = []
    for (u, v, label) in local_edges:
        u, v, label = int(u), int(v), int(label)
        walk_tokens = [f"N_{u}", f"E_{label}", f"N_{v}"]
        curr = v
        for _ in range(max_walk_length - 1):
            neigh = nbrs.get(curr)
            if neigh is None or len(neigh) == 0:
                break
            idx = rng_anchor.integers(0, len(neigh))
            nxt = int(neigh[idx])
            lbl = int(lbls[curr][idx])
            walk_tokens.append(f"E_{lbl}")
            walk_tokens.append(f"N_{nxt}")
            curr = nxt
        local_walks.append(walk_tokens)
    return task_id, local_walks


# ---------------------------------------------------------------------------
# Strategy L — Guaranteed target coverage
# ---------------------------------------------------------------------------

def guaranteed_coverage_walks(
    edges, num_walks, max_walk_length, seed,
    num_workers, val_set, test_set,
):
    """Anchor walks to ensure every edge is traversed at least once.

    For every edge (u, v, label) not yet anchored, force a walk through it:
    start at u, take the step to v, then continue with a normal random walk.
    The remaining budget is filled with uniform walks.

    Previously only anchored val/test edges; now targets ALL edges so that
    train and mask edges also receive guaranteed context.  Edges are shuffled
    so that when the budget is smaller than the edge count, no systematic
    subset is always skipped.

    Leakage: none.  Uses edge topology only; the label token is emitted
    identically to how the uniform sampler would emit it.
    """
    nodes, nbrs, lbls = _build_adj(edges)
    if not nodes:
        return []

    base_seed = int(seed)
    # Target all edges; shuffle so budget shortfalls are random not systematic
    rng_shuffle = np.random.default_rng(base_seed + 42)
    target_edges = list(edges)
    rng_shuffle.shuffle(target_edges)

    # Build reverse-lookup: u -> list of (v, label) that are target edges
    target_adj = defaultdict(list)
    for (u, v, label) in target_edges:
        target_adj[int(u)].append((int(v), int(label)))

    walks = []
    walk_idx = 0

    # Phase 1: one guaranteed walk per target edge
    anchor_budget = min(num_walks, len(target_edges))
    anchor_edges = target_edges[:anchor_budget]

    if anchor_budget > 0:
        tasks = _chunk_tasks(anchor_budget, num_workers)
        chunks = [anchor_edges[s:e] for s, e, _ in tasks]
        worker = partial(
            _guaranteed_anchor_chunk,
            nbrs, lbls,
        )
        if len(tasks) == 1:
            _, local = worker(chunks[0], max_walk_length, base_seed + 10_000_000, 0)
            walks.extend(local)
        else:
            with mp.Pool(processes=len(tasks)) as pool:
                results = pool.starmap(
                    worker,
                    [
                        (chunks[i], max_walk_length, base_seed + 10_000_000, tasks[i][2])
                        for i in range(len(tasks))
                    ],
                )
            walks.extend(_flatten_chunked(results))
    walk_idx = len(walks)

    # Phase 2: fill remainder with uniform walks
    remaining = num_walks - walk_idx
    if remaining > 0:
        uniform = sample_random_walks(
            edges, num_walks=remaining, max_walk_length=max_walk_length,
            num_workers=num_workers, seed=base_seed,
        )
        walks.extend(uniform)

    return walks


# ---------------------------------------------------------------------------
# Strategy B_safe — Training-negative emphasis
# ---------------------------------------------------------------------------
#! BUG it has assumtion that neg is 0
def neg_emphasis_walks(
    edges, num_walks, max_walk_length, seed, num_workers,
    train_set, mask_set, neg_fraction,
):
    """Bias walk starts toward nodes incident to TRAINING negative edges.

    Leakage: none.  Only training-split edge signs are used (train_set ∪
    mask_set).  The model already sees these edges as E_0/E_1 tokens.
    Val/test edge signs are never consulted.
    """
    nodes, nbrs, lbls = _build_adj(edges)
    if not nodes:
        return []

    base_seed = int(seed)
    node_arr = np.array(nodes)

    # Nodes incident to training negative edges only
    train_edges = (train_set or set()) | (mask_set or set())
    neg_nodes = list({
        n for (u, v, label) in train_edges if int(label) == 0
        for n in (int(u), int(v))
        if n in nbrs
    })

    if not neg_nodes:
        # Fallback to uniform if no training negatives found
        return sample_random_walks(
            edges, num_walks=num_walks, max_walk_length=max_walk_length,
            num_workers=num_workers, seed=base_seed,
        )

    neg_arr = np.array(neg_nodes)
    n_neg = int(num_walks * neg_fraction)
    n_uniform = num_walks - n_neg

    rng = np.random.default_rng(base_seed)
    starts = np.empty(num_walks, dtype=np.int64)
    if n_neg > 0:
        starts[:n_neg] = neg_arr[rng.integers(0, len(neg_arr), size=n_neg)]
    if n_uniform > 0:
        starts[n_neg:] = node_arr[rng.integers(0, len(node_arr), size=n_uniform)]

    tasks = _chunk_tasks(num_walks, num_workers)
    worker = partial(
        _sample_from_starts_chunk,
        nbrs, lbls, starts, max_walk_length, base_seed + 200_000_000,
    )
    if len(tasks) == 1:
        _, walks = worker(*tasks[0])
        return walks
    with mp.Pool(processes=len(tasks)) as pool:
        results = pool.starmap(worker, tasks)
    return _flatten_chunked(results)


# ---------------------------------------------------------------------------
# Strategy F — Inverse-degree start sampling
# ---------------------------------------------------------------------------

def inv_degree_walks(edges, num_walks, max_walk_length, seed, num_workers):
    """Start probability ∝ 1/degree to counteract the degree² visit bias.

    Leakage: none.  Node degree is pure topology.
    """
    nodes, nbrs, lbls = _build_adj(edges)
    if not nodes:
        return []

    base_seed = int(seed)
    node_arr = np.array(nodes)
    degrees = np.array([len(nbrs.get(n, [])) for n in node_arr], dtype=np.float64)
    degrees = np.where(degrees == 0, 1, degrees)  # avoid /0 for isolated nodes
    weights = 1.0 / degrees
    weights /= weights.sum()

    rng = np.random.default_rng(base_seed)
    start_idx = rng.choice(len(node_arr), size=num_walks, p=weights)
    starts = node_arr[start_idx].astype(np.int64, copy=False)

    tasks = _chunk_tasks(num_walks, num_workers)
    worker = partial(
        _sample_from_starts_chunk,
        nbrs, lbls, starts, max_walk_length, base_seed + 300_000_000,
    )
    if len(tasks) == 1:
        _, walks = worker(*tasks[0])
        return walks
    with mp.Pool(processes=len(tasks)) as pool:
        results = pool.starmap(worker, tasks)
    return _flatten_chunked(results)


# ---------------------------------------------------------------------------
# Strategy G — Node2Vec biased walks
# ---------------------------------------------------------------------------

def node2vec_walks(edges, num_walks, max_walk_length, seed, num_workers, p, q):
    """Node2Vec-style biased transitions using return param p and in-out param q.

    p > 1 discourages returning to the previous node (reduce redundancy).
    q > 1 biases toward DFS-like exploration (spread further from start).

    Leakage: none.  Transition weights are based on node distance in the
    graph topology, never on edge labels.
    """
    nodes, nbrs, lbls = _build_adj(edges)
    if not nodes:
        return []

    base_seed = int(seed)
    node_arr = np.array(nodes)

    # Pre-compute neighbour sets for fast distance lookup
    nbr_sets = {n: set(nbrs[n].tolist()) for n in nodes}

    p, q = float(p), float(q)

    rng = np.random.default_rng(base_seed)
    starts = node_arr[rng.integers(0, len(node_arr), size=num_walks)].astype(np.int64, copy=False)

    tasks = _chunk_tasks(num_walks, num_workers)
    worker = partial(
        _node2vec_chunk,
        nbrs, lbls, nbr_sets, starts, max_walk_length, p, q,
        base_seed + 400_000_000,
    )
    if len(tasks) == 1:
        _, walks = worker(*tasks[0])
        return walks
    with mp.Pool(processes=len(tasks)) as pool:
        results = pool.starmap(worker, tasks)
    return _flatten_chunked(results)


# ---------------------------------------------------------------------------
# Strategy A — Edge-seeded walks (online min-heap)
# ---------------------------------------------------------------------------

def _edge_seeded_chunk(
    nbrs, lbls, edge_list, max_walk_length, base_seed,
    start_idx, end_idx, task_id,
):
    """Worker chunk for edge-seeded walks.

    Each worker maintains its own min-heap and traversal counts, so workers
    are independent and embarrassingly parallel.  The per-worker feedback
    loop still drives coverage within the chunk; cross-chunk coordination is
    lost but inconsequential at high walk counts.
    """
    rng_global = np.random.default_rng(base_seed + task_id)  # reserved for future use
    step_rng = np.random.default_rng(base_seed + task_id + 500_000_000)

    m = len(edge_list)
    counts = np.zeros(m, dtype=np.int64)
    edge_index = {e: i for i, e in enumerate(edge_list)}
    heap_entries = [(0, i) for i in range(m)]
    heapq.heapify(heap_entries)

    walks = []
    for _ in range(start_idx, end_idx):
        # Find least-covered edge via lazy heap
        while True:
            count_snap, eidx = heapq.heappop(heap_entries)
            if count_snap == counts[eidx]:
                break

        u, v, label = edge_list[eidx]
        # Always start at u and force the first step u→v so the selected
        # directed edge is guaranteed to be traversed (starting at v would
        # walk away from u and, in a directed graph, likely never hit u→v).
        walk_tokens = [f"N_{u}", f"E_{label}", f"N_{v}"]
        traversed = [eidx]  # the forced edge is already traversed
        curr = v

        for _ in range(max_walk_length - 1):
            neigh = nbrs.get(curr)
            if neigh is None or len(neigh) == 0:
                break
            idx = step_rng.integers(0, len(neigh))
            nxt = int(neigh[idx])
            lbl = int(lbls[curr][idx])
            walk_tokens.append(f"E_{lbl}")
            walk_tokens.append(f"N_{nxt}")
            eid = edge_index.get((curr, nxt, lbl), -1)
            if eid >= 0:
                traversed.append(eid)
            curr = nxt

        for eid in traversed:
            counts[eid] += 1
            heapq.heappush(heap_entries, (int(counts[eid]), eid))

        walks.append(walk_tokens)

    return task_id, walks


def edge_seeded_walks(edges, num_walks, max_walk_length, seed, num_workers):
    """Seed each walk from the least-traversed edge (adaptive online coverage).

    Maintains a per-worker min-heap of (traversal_count, edge_idx).  Workers
    are independent chunks so this is embarrassingly parallel.  Coverage
    feedback operates within each worker's chunk; cross-worker coordination
    is sacrificed for speed but is negligible at large walk counts.

    Leakage: none.  Traversal counts are topology-derived.  Edge labels are
    read only to build walk tokens (same as uniform sampler).
    """
    nodes, nbrs, lbls = _build_adj(edges)
    if not nodes:
        return []

    base_seed = int(seed)
    edge_list = [(int(u), int(v), int(label)) for u, v, label in edges]

    tasks = _chunk_tasks(num_walks, num_workers)
    worker = partial(
        _edge_seeded_chunk,
        nbrs, lbls, edge_list, max_walk_length, base_seed + 400_000_000,
    )
    if len(tasks) == 1:
        _, walks = worker(*tasks[0])
        return walks
    with mp.Pool(processes=len(tasks)) as pool:
        results = pool.starmap(worker, tasks)
    return _flatten_chunked(results)


# ---------------------------------------------------------------------------
# Strategy H — Neg-traversal walks (training negatives only)
# ---------------------------------------------------------------------------
#! BUG it has assumtion that neg is 0
def neg_traversal_walks(
    edges, num_walks, max_walk_length, seed, num_workers,
    train_set, mask_set,
):
    """Build walks that explicitly traverse a training negative edge.

    For each walk, picks a random TRAINING negative edge (u, v, 0), builds a
    prefix by walking backward from u for k steps, then crosses (u->v), then
    continues from v.  This guarantees the negative edge appears centrally
    with both left and right context.

    Leakage: none.  Only training-split negative edges are used.  Val/test
    edge signs are never consulted.
    """
    nodes, nbrs, lbls = _build_adj(edges)
    # Also build reverse adjacency for backward prefix
    rev_nbrs = defaultdict(list)
    rev_lbls_map = defaultdict(list)
    for u, v, label in edges:
        rev_nbrs[v].append(u)
        rev_lbls_map[v].append(int(label))
    rev_nbrs_arr = {}
    rev_lbls_arr = {}
    for n in rev_nbrs:
        rev_nbrs_arr[n] = np.array(rev_nbrs[n], dtype=np.int64)
        rev_lbls_arr[n] = np.array(rev_lbls_map[n], dtype=np.int64)

    if not nodes:
        return []

    base_seed = int(seed)
    train_edges = (train_set or set()) | (mask_set or set())
    neg_train = [(int(u), int(v), int(l)) for (u, v, l) in train_edges if int(l) == 0]

    if not neg_train:
        return sample_random_walks(
            edges, num_walks=num_walks, max_walk_length=max_walk_length,
            num_workers=num_workers, seed=base_seed,
        )

    neg_arr = neg_train
    n_neg = len(neg_arr)

    tasks = _chunk_tasks(num_walks, num_workers)
    worker = partial(
        _neg_traversal_chunk,
        nbrs, lbls, rev_nbrs_arr, rev_lbls_arr, neg_arr,
        max_walk_length, base_seed + 600_000_000,
    )
    if len(tasks) == 1:
        _, walks = worker(*tasks[0])
        return walks
    with mp.Pool(processes=len(tasks)) as pool:
        results = pool.starmap(worker, tasks)
    return _flatten_chunked(results)


# ---------------------------------------------------------------------------
# Strategy C — Greedy set-cover
# ---------------------------------------------------------------------------

def greedy_set_cover_walks(
    edges, num_walks, max_walk_length, seed, num_workers, multiplier,
):
    """Select num_walks from a larger candidate pool to maximise edge coverage.

    Steps:
    1. Generate num_walks * multiplier candidates with the uniform sampler.
    2. Convert each walk to the set of edge IDs it traverses.
    3. Greedy streaming selection: iterate candidates in random order; add any
       walk that covers at least one new edge; stop when num_walks selected.
       Remaining budget filled with leftover candidates in order.

    Leakage: none.  Selection criterion is edge *presence* (topology), never
    edge *labels*.
    """
    base_seed = int(seed)
    n_candidates = int(num_walks * multiplier)

    edge_index = _build_edge_index(edges)
    m = len(edge_index)

    # Generate candidates in chunks to control memory
    chunk_size = min(100_000, n_candidates)
    candidates = []
    chunk_seed = base_seed + 999_999
    generated = 0
    while generated < n_candidates:
        batch = min(chunk_size, n_candidates - generated)
        chunk = sample_random_walks(
            edges, num_walks=batch, max_walk_length=max_walk_length,
            num_workers=num_workers, seed=chunk_seed + generated,
        )
        for w in chunk:
            eids = _walk_to_edge_ids(w, edge_index)
            candidates.append((w, eids))
        generated += batch

    # Shuffle for unbiased greedy selection
    rng_sel = np.random.default_rng(base_seed + 42)
    order = rng_sel.permutation(len(candidates)).tolist()

    covered = np.zeros(m, dtype=bool)
    selected = []
    leftover = []

    for i in order:
        w, eids = candidates[i]
        if len(selected) >= num_walks:
            break
        if len(eids) == 0 or not covered[eids].all():
            selected.append(w)
            if len(eids) > 0:
                covered[eids] = True
        else:
            leftover.append(w)

    # Fill any remaining budget with leftover candidates
    while len(selected) < num_walks and leftover:
        selected.append(leftover.pop(0))

    # Final top-up with uniform sampling if somehow still short
    if len(selected) < num_walks:
        extra = sample_random_walks(
            edges, num_walks=num_walks - len(selected),
            max_walk_length=max_walk_length,
            num_workers=num_workers, seed=base_seed + 1,
        )
        selected.extend(extra)

    return selected[:num_walks]


# ---------------------------------------------------------------------------
# Strategy J — Coverage-aware restart
# ---------------------------------------------------------------------------

def _cov_restart_chunk(
    nbrs, lbls, edge_list, node_arr, max_walk_length, base_seed,
    start_idx, end_idx, task_id,
):
    """Worker chunk for coverage-restart walks.

    Each worker maintains its own covered-edge state.  Coverage feedback
    operates within the chunk; cross-chunk coordination is dropped for
    parallelism — negligible at high walk counts.
    """
    m = len(edge_list)
    edge_index = {e: i for i, e in enumerate(edge_list)}

    node_edge_map = defaultdict(list)
    for eidx, (u, v, _) in enumerate(edge_list):
        node_edge_map[u].append(eidx)
        node_edge_map[v].append(eidx)

    covered_count = np.zeros(m, dtype=np.int64)
    uncovered_incident = {int(u): len(node_edge_map[u]) for u in node_edge_map}
    nodes_with_uncovered = {u for u, c in uncovered_incident.items() if c > 0}

    rng = np.random.default_rng(base_seed + task_id)
    walks = []
    is_first = True

    for _ in range(start_idx, end_idx):
        if is_first or not nodes_with_uncovered:
            start = int(node_arr[rng.integers(0, len(node_arr))])
            is_first = False
        else:
            candidate_nodes = np.fromiter(nodes_with_uncovered, dtype=np.int64)
            scores = np.array(
                [uncovered_incident[int(n)] for n in candidate_nodes],
                dtype=np.float64,
            )
            probs = scores / scores.sum()
            idx = rng.choice(len(candidate_nodes), p=probs)
            start = int(candidate_nodes[idx])

        walk_tokens = [f"N_{start}"]
        curr = start

        for _ in range(max_walk_length):
            neigh = nbrs.get(curr)
            if neigh is None or len(neigh) == 0:
                break
            idx = rng.integers(0, len(neigh))
            nxt = int(neigh[idx])
            lbl = int(lbls[curr][idx])
            walk_tokens.append(f"E_{lbl}")
            walk_tokens.append(f"N_{nxt}")
            eid = edge_index.get((curr, nxt, lbl), -1)
            if eid >= 0 and covered_count[eid] == 0:
                covered_count[eid] = 1
                u0, v0, _ = edge_list[eid]
                uncovered_incident[u0] = max(0, uncovered_incident[u0] - 1)
                uncovered_incident[v0] = max(0, uncovered_incident[v0] - 1)
                if uncovered_incident[u0] == 0:
                    nodes_with_uncovered.discard(u0)
                if uncovered_incident[v0] == 0:
                    nodes_with_uncovered.discard(v0)
            curr = nxt

        walks.append(walk_tokens)
    return task_id, walks


def cov_restart_walks(edges, num_walks, max_walk_length, seed, num_workers):
    """When a walk dead-ends, restart from the node with most uncovered incident edges.

    Workers are independent chunks each with their own coverage state.
    Coverage feedback operates within each worker; cross-worker coordination
    is dropped for parallelism — negligible at high walk counts.

    Leakage: none.  Restart criterion uses uncovered edge count (topology).
    """
    nodes, nbrs, lbls = _build_adj(edges)
    if not nodes:
        return []

    base_seed = int(seed)
    edge_list = [(int(u), int(v), int(l)) for u, v, l in edges]
    node_arr = np.array(nodes, dtype=np.int64)

    tasks = _chunk_tasks(num_walks, num_workers)
    worker = partial(
        _cov_restart_chunk,
        nbrs, lbls, edge_list, node_arr, max_walk_length,
        base_seed + 800_000_000,
    )
    if len(tasks) == 1:
        _, walks = worker(*tasks[0])
        return walks
    with mp.Pool(processes=len(tasks)) as pool:
        results = pool.starmap(worker, tasks)
    return _flatten_chunked(results)


# ---------------------------------------------------------------------------
# Strategy K_safe — Sign-alternating walks (train edges only)
# ---------------------------------------------------------------------------

def _sign_alt_chunk(
    nbrs, lbls, node_arr, train_edges_set, max_walk_length, base_seed,
    start_idx, end_idx, task_id,
):
    """Worker chunk for sign-alternating walks (stateless per walk)."""
    rng = np.random.default_rng(base_seed + task_id)
    walks = []
    for i in range(start_idx, end_idx):
        # Pick a random start node per walk using a per-walk offset from chunk seed
        start = int(node_arr[rng.integers(0, len(node_arr))])
        walk_tokens = [f"N_{start}"]
        curr = start
        last_sign = None

        for _ in range(max_walk_length):
            neigh = nbrs.get(curr)
            if neigh is None or len(neigh) == 0:
                break

            if last_sign is None:
                idx = rng.integers(0, len(neigh))
            else:
                target_sign = 1 - last_sign
                weights = np.ones(len(neigh), dtype=np.float64)
                for j in range(len(neigh)):
                    nxt_j = int(neigh[j])
                    lbl_j = int(lbls[curr][j])
                    if (curr, nxt_j, lbl_j) in train_edges_set and lbl_j == target_sign:
                        weights[j] = 3.0
                weights /= weights.sum()
                idx = rng.choice(len(neigh), p=weights)

            nxt = int(neigh[idx])
            lbl = int(lbls[curr][idx])
            walk_tokens.append(f"E_{lbl}")
            walk_tokens.append(f"N_{nxt}")
            if (curr, nxt, lbl) in train_edges_set:
                last_sign = lbl
            curr = nxt

        walks.append(walk_tokens)
    return task_id, walks


def sign_alt_walks(
    edges, num_walks, max_walk_length, seed, num_workers,
    train_set, mask_set,
):
    """Prefer crossing an edge of opposite sign to the last traversed edge.

    Only the sign of TRAINING edges is used for transition preference.
    Val/test edge signs are never consulted — those edges are treated as
    neutral at each step.

    Leakage: none.  Training edge signs are visible to the model anyway.
    """
    nodes, nbrs, lbls = _build_adj(edges)
    if not nodes:
        return []

    base_seed = int(seed)
    train_edges_set = (train_set or set()) | (mask_set or set())
    node_arr = np.array(list(nbrs.keys()), dtype=np.int64)

    tasks = _chunk_tasks(num_walks, num_workers)
    worker = partial(
        _sign_alt_chunk,
        nbrs, lbls, node_arr, train_edges_set, max_walk_length,
        base_seed + 700_000_000,
    )
    if len(tasks) == 1:
        _, walks = worker(*tasks[0])
        return walks
    with mp.Pool(processes=len(tasks)) as pool:
        results = pool.starmap(worker, tasks)
    return _flatten_chunked(results)


# ---------------------------------------------------------------------------
# Smart ensemble: Guaranteed + Neg-Emphasis + Edge-Seeded
# ---------------------------------------------------------------------------

def smart_walks(
    edges, num_walks, max_walk_length, seed, num_workers,
    train_set, mask_set, val_set, test_set,
):
    """Combine guaranteed coverage, neg-emphasis, and edge-seeded walks.

    Budget split: 1/3 guaranteed, 1/3 neg-emphasis, 1/3 edge-seeded.
    """
    base_seed = int(seed)
    n_guaranteed = num_walks // 3
    n_neg = num_walks // 3
    n_edge = num_walks - n_guaranteed - n_neg

    walks = []

    if n_guaranteed > 0:
        walks += guaranteed_coverage_walks(
            edges, n_guaranteed, max_walk_length,
            seed=base_seed + 0, num_workers=num_workers,
            val_set=val_set, test_set=test_set,
        )
    if n_neg > 0:
        walks += neg_emphasis_walks(
            edges, n_neg, max_walk_length,
            seed=base_seed + 1_000_000, num_workers=num_workers,
            train_set=train_set, mask_set=mask_set,
            neg_fraction=0.8,
        )
    if n_edge > 0:
        walks += edge_seeded_walks(
            edges, n_edge, max_walk_length,
            seed=base_seed + 2_000_000, num_workers=num_workers,
        )

    return walks[:num_walks]


# ---------------------------------------------------------------------------
# Strategy k_cover — guarantee every edge appears in >= k distinct walks
# ---------------------------------------------------------------------------

def k_cover_walks(
    edges, num_walks, max_walk_length, seed, num_workers, k=1,
):
    """Ensure every edge is traversed by at least k distinct walks.

    Algorithm:
      Iterate over edges in random order (up to k passes).  For each edge
      whose visit count is still < k, emit one anchor walk starting at u,
      forced through the step (u->v, label), then continuing randomly.
      Each anchor walk also increments counts for every other edge it
      traverses, so many edges are satisfied as side-effects without
      needing dedicated anchors.  Once all edges reach k visits (or the
      walk budget is exhausted), the remainder is filled with uniform walks.

    With 500k walks and walk_length=80, each anchor walk covers ~40 edges.
    For Slashdot (549k edges), covering the ~32% zero-visit edges at k=1
    requires ~14k anchor walks (<3% of budget).  k=5 needs ~70k anchors.

    Leakage: none.  Uses edge topology and per-edge visit counts only.
    """
    nodes, nbrs, lbls = _build_adj(edges)
    edge_index = _build_edge_index(edges)
    if not nodes or not edge_index:
        return []

    m = len(edge_index)
    visit_count = np.zeros(m, dtype=np.int32)
    base_seed = int(seed)
    anchor_rng = np.random.default_rng(base_seed + 777_777_777)

    all_edge_keys = list(edge_index.keys())
    # Shuffle once; same order used across passes
    order = np.arange(m, dtype=np.int64)
    anchor_rng.shuffle(order)

    walks = []
    walk_idx = 0

    # Anchor phase: up to k passes over all edges
    for _pass in range(int(k)):
        if walk_idx >= num_walks:
            break
        for pos in order:
            if walk_idx >= num_walks:
                break
            edge_key = all_edge_keys[pos]
            eid = edge_index[edge_key]
            if visit_count[eid] >= k:
                continue  # already satisfied
            u, v, label = edge_key
            # Anchor walk: force first step u -> v
            walk_tokens = [f"N_{u}", f"E_{label}", f"N_{v}"]
            visit_count[eid] += 1
            curr = v
            for _ in range(max_walk_length - 1):
                neigh = nbrs.get(curr)
                if neigh is None or len(neigh) == 0:
                    break
                step_i = anchor_rng.integers(0, len(neigh))
                nxt = int(neigh[step_i])
                lbl = int(lbls[curr][step_i])
                walk_tokens.append(f"E_{lbl}")
                walk_tokens.append(f"N_{nxt}")
                eid2 = edge_index.get((curr, nxt, lbl), -1)
                if eid2 >= 0:
                    visit_count[eid2] += 1
                curr = nxt
            walks.append(walk_tokens)
            walk_idx += 1

    # Fill remaining budget with uniform walks
    remaining = num_walks - walk_idx
    if remaining > 0:
        uniform = sample_random_walks(
            edges, num_walks=remaining, max_walk_length=max_walk_length,
            num_workers=num_workers, seed=base_seed,
        )
        walks.extend(uniform)

    return walks


# ---------------------------------------------------------------------------
# Strategy k_cover (FAST) — parallel, same throughput pattern as uniform/guaranteed
# ---------------------------------------------------------------------------

def _build_adj_with_eids(edges, edge_index):
    """Like _build_adj but also returns, per node, the edge_index id aligned to its
    neighbour array (so a worker can credit visits with O(1) array indexing instead
    of a per-step dict lookup)."""
    nodes, nbrs, lbls = _build_adj(edges)
    eids = {}
    for n in nbrs:
        ns, ls = nbrs[n], lbls[n]
        arr = np.empty(len(ns), dtype=np.int64)
        for i in range(len(ns)):
            arr[i] = edge_index.get((int(n), int(ns[i]), int(ls[i])), -1)
        eids[n] = arr
    return nodes, nbrs, lbls, eids


def _kcover_anchor_chunk(nbrs, lbls, eids, m, anchor_edges, max_walk_length,
                         base_seed, task_id):
    """Worker: for each (u, v, label, eid) anchor, force the step u->v then walk
    randomly; emit raw-id tokens and accumulate a local per-edge visit bincount."""
    rng = np.random.default_rng(base_seed + task_id)
    walks = []
    local_vc = np.zeros(m, dtype=np.int64)
    for (u, v, label, eid) in anchor_edges:
        toks = [f"N_{u}", f"E_{label}", f"N_{v}"]
        if eid >= 0:
            local_vc[eid] += 1
        curr = v
        for _ in range(max_walk_length - 1):
            neigh = nbrs.get(curr)
            if neigh is None or len(neigh) == 0:
                break
            i = int(rng.integers(0, len(neigh)))
            nxt = int(neigh[i])
            lbl = int(lbls[curr][i])
            toks.append(f"E_{lbl}")
            toks.append(f"N_{nxt}")
            e2 = int(eids[curr][i])
            if e2 >= 0:
                local_vc[e2] += 1
            curr = nxt
        walks.append(toks)
    return task_id, walks, local_vc


def k_cover_walks_fast(edges, num_walks, max_walk_length, seed, num_workers, k=1,
                       max_passes=None):
    """Guarantee every edge is traversed >= k times, parallelised.

    Iterative passes: each pass emits one anchor walk per still-under-k edge,
    generated in parallel mp.Pool chunks (same throughput pattern as the uniform
    sampler); visit counts (including every edge an anchor walk passes through as a
    side-effect) are reduced between passes so later passes skip satisfied edges.
    The remaining budget is filled with uniform walks. Node ids stay raw in tokens;
    adjacency is dict-based, so sparse / ~1e9 ids never allocate max_id-sized arrays.
    """
    edge_index = _build_edge_index(edges)
    if not edge_index:
        return []
    nodes, nbrs, lbls, eids = _build_adj_with_eids(edges, edge_index)
    if not nodes:
        return []

    m = len(edge_index)
    if num_walks < m:
        # Anchor-first ordering needs ~|E| walks to even cover every edge once; below
        # that, pass-1 anchors consume the whole budget with no uniform spread and
        # node coverage can drop BELOW plain uniform. Size num_walks >= ~1.5*|E|.
        print(f"WARNING: k_cover num_walks={num_walks} < |E|={m}; coverage/saturation "
              f"will be incomplete and may underperform uniform. Increase num_walks.")
    uniq_edges = list(edge_index.keys())  # index i -> (u,v,label) with eid == i
    visit_count = np.zeros(m, dtype=np.int64)
    base_seed = int(seed)
    order_rng = np.random.default_rng(base_seed + 777_777_777)
    if max_passes is None:
        max_passes = int(k) + 5

    walks = []
    for p in range(max_passes):
        if len(walks) >= num_walks:
            break
        need = np.nonzero(visit_count < k)[0]
        if need.size == 0:
            break
        budget_left = num_walks - len(walks)
        order_rng.shuffle(need)
        if need.size > budget_left:
            need = need[:budget_left]
        anchors = [(uniq_edges[i][0], uniq_edges[i][1], uniq_edges[i][2], int(i))
                   for i in need]

        tasks = _chunk_tasks(len(anchors), num_workers)
        chunks = [anchors[s:e] for s, e, _ in tasks]
        worker = partial(_kcover_anchor_chunk, nbrs, lbls, eids, m)
        pass_seed = base_seed + 10_000_000 * (p + 1)
        if len(tasks) == 1:
            _, w, vc = worker(chunks[0], max_walk_length, pass_seed, 0)
            walks.extend(w)
            visit_count += vc
        else:
            with mp.Pool(processes=len(tasks)) as pool:
                results = pool.starmap(
                    worker,
                    [(chunks[i], max_walk_length, pass_seed, tasks[i][2])
                     for i in range(len(tasks))],
                )
            results.sort(key=lambda r: r[0])
            for _, w, vc in results:
                walks.extend(w)
                visit_count += vc

    remaining = num_walks - len(walks)
    if remaining > 0:
        walks.extend(sample_random_walks(
            edges, num_walks=remaining, max_walk_length=max_walk_length,
            num_workers=num_workers, seed=base_seed,
        ))
    return walks[:num_walks]


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def sample_walks(
    edges, cfg, seed, num_workers=1,
    train_set=None, mask_set=None, val_set=None, test_set=None,
):
    """Dispatch to the configured walk sampling strategy.

    Args:
        edges:       full list of (u, v, label) including val/test
        cfg:         OmegaConf config (uses cfg.dataset.* keys)
        seed:        base integer seed (from reproducibility.seed)
        num_workers: parallel workers for strategies that support it
        train_set:   set of (u,v,label) tuples for training edges
        mask_set:    set of (u,v,label) tuples for mask edges
        val_set:     set of (u,v,label) tuples for val edges (topology only)
        test_set:    set of (u,v,label) tuples for test edges (topology only)

    Returns:
        List of walk token lists.
    """
    strategy = str(getattr(cfg.dataset, "walk_strategy", "uniform"))
    num_walks = int(cfg.dataset.num_walks)
    max_walk_length = int(cfg.dataset.max_walk_length)

    if strategy == "uniform":
        return sample_random_walks(
            edges, num_walks=num_walks, max_walk_length=max_walk_length,
            num_workers=num_workers, seed=seed,
        )

    elif strategy == "guaranteed":
        return guaranteed_coverage_walks(
            edges, num_walks, max_walk_length, seed=seed,
            num_workers=num_workers, val_set=val_set, test_set=test_set,
        )

    elif strategy == "neg_emphasis":
        frac = float(getattr(cfg.dataset, "walk_neg_emphasis_fraction", 0.5))
        return neg_emphasis_walks(
            edges, num_walks, max_walk_length, seed=seed,
            num_workers=num_workers,
            train_set=train_set, mask_set=mask_set, neg_fraction=frac,
        )

    elif strategy == "inv_degree":
        return inv_degree_walks(
            edges, num_walks, max_walk_length, seed=seed, num_workers=num_workers,
        )

    elif strategy == "node2vec":
        p = float(getattr(cfg.dataset, "walk_p", 1.0))
        q = float(getattr(cfg.dataset, "walk_q", 1.0))
        return node2vec_walks(
            edges, num_walks, max_walk_length, seed=seed,
            num_workers=num_workers, p=p, q=q,
        )

    elif strategy == "edge_seeded":
        return edge_seeded_walks(
            edges, num_walks, max_walk_length, seed=seed, num_workers=num_workers,
        )

    elif strategy == "neg_traversal":
        return neg_traversal_walks(
            edges, num_walks, max_walk_length, seed=seed,
            num_workers=num_workers,
            train_set=train_set, mask_set=mask_set,
        )

    elif strategy == "set_cover":
        mult = float(getattr(cfg.dataset, "walk_set_cover_multiplier", 4))
        return greedy_set_cover_walks(
            edges, num_walks, max_walk_length, seed=seed,
            num_workers=num_workers, multiplier=mult,
        )

    elif strategy == "cov_restart":
        return cov_restart_walks(
            edges, num_walks, max_walk_length, seed=seed, num_workers=num_workers,
        )

    elif strategy == "sign_alt":
        return sign_alt_walks(
            edges, num_walks, max_walk_length, seed=seed,
            num_workers=num_workers,
            train_set=train_set, mask_set=mask_set,
        )

    elif strategy == "smart":
        return smart_walks(
            edges, num_walks, max_walk_length, seed=seed,
            num_workers=num_workers,
            train_set=train_set, mask_set=mask_set,
            val_set=val_set, test_set=test_set,
        )

    elif strategy == "k_cover":
        k = int(getattr(cfg.dataset, "walk_k_min", 1))
        return k_cover_walks_fast(
            edges, num_walks, max_walk_length, seed=seed,
            num_workers=num_workers, k=k,
        )

    else:
        raise ValueError(
            f"Unknown walk_strategy={strategy!r}. "
            "Valid: uniform, guaranteed, neg_emphasis, inv_degree, node2vec, "
            "edge_seeded, neg_traversal, set_cover, cov_restart, sign_alt, smart"
        )
