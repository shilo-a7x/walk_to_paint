"""
Walk sampling: builds the random walks that get tokenized into training sequences.

edge_cover_walks guarantees every edge appears in exactly one forced anchor walk,
then fills the remaining walk budget with additional walks that are guaranteed
distinct from every other walk already generated (raises RuntimeError rather than
silently padding with duplicates if the graph can't supply enough).

Entry point: sample_walks(edges, cfg, seed, num_workers, train_set, mask_set)
"""

import hashlib
import math
import multiprocessing as mp
from collections import defaultdict
from functools import partial

import numpy as np


# ---------------------------------------------------------------------------
# Adjacency helpers
# ---------------------------------------------------------------------------

def _build_adj(edges):
    """Build adjacency lists and indexable arrays for fast sampling."""
    graph = defaultdict(list)
    for u, v, label in edges:
        graph[u].append((v, label))
    nodes = list(graph.keys())
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


def _build_rev_adj(edges):
    """Reverse (transpose-graph) adjacency: rev_nbrs[v] = predecessors u of v
    (i.e. edges u->v), rev_lbls[v] = the matching edge labels. Same dict-of-arrays
    shape as _build_adj's (nbrs, lbls), just transposed — dict-based so sparse /
    large node ids never allocate max_id-sized arrays.
    """
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
    return rev_nbrs_arr, rev_lbls_arr


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


# ---------------------------------------------------------------------------
# edge_cover_walks
# ---------------------------------------------------------------------------

def _anchor_chunk(nbrs, lbls, eids, m, anchor_edges, max_walk_length,
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


def _fill_chunk_dedup(
    nbrs, lbls, rev_nbrs, rev_lbls, node_arr, max_walk_length, prefix_cap,
    base_seed, task_id, start_idx, end_idx,
):
    """Worker: generate free-start candidate walks for the fill phase.

    Unlike anchors (forced start, forced first edge), a fill walk's start node is
    free, so every fill candidate draws a random backward prefix from the start
    (0..prefix_cap hops) before continuing forward — a start node with a short or
    deterministic forward path would otherwise produce the same walk every time it's
    picked. Dedup against already-accepted walks happens in the caller
    (_generate_dedup_fill), not here — this worker just proposes candidates and
    returns their hashes alongside.
    """
    rng = np.random.default_rng(base_seed + task_id)
    walks = []
    hashes = []
    for _ in range(start_idx, end_idx):
        start = int(node_arr[rng.integers(0, len(node_arr))])
        prefix_len = int(rng.integers(0, prefix_cap + 1)) if prefix_cap > 0 else 0

        prefix_tokens = []
        curr = start
        for _ in range(prefix_len):
            rn = rev_nbrs.get(curr)
            if rn is None or len(rn) == 0:
                break
            i = int(rng.integers(0, len(rn)))
            prev = int(rn[i])
            lbl = int(rev_lbls[curr][i])
            prefix_tokens = [f"N_{prev}", f"E_{lbl}"] + prefix_tokens
            curr = prev
        actual_prefix_hops = (len(prefix_tokens)) // 2

        toks = prefix_tokens + [f"N_{start}"]
        curr2 = start
        for _ in range(max_walk_length - actual_prefix_hops):
            neigh = nbrs.get(curr2)
            if neigh is None or len(neigh) == 0:
                break
            i = int(rng.integers(0, len(neigh)))
            nxt = int(neigh[i])
            lbl = int(lbls[curr2][i])
            toks.append(f"E_{lbl}")
            toks.append(f"N_{nxt}")
            curr2 = nxt

        h = hashlib.blake2b("|".join(toks).encode(), digest_size=8).digest()
        walks.append(toks)
        hashes.append(h)
    return walks, hashes


def _generate_dedup_fill(
    nbrs, lbls, rev_nbrs, rev_lbls, max_walk_length, num_workers,
    n_needed, seen_hashes, base_seed, max_rounds=10, oversample=1.5,
):
    """Fill `n_needed` walks that are new relative to `seen_hashes` (mutated in
    place — accepted hashes are added as they're accepted).

    Generates candidates in oversampled parallel rounds (more candidates than
    currently needed, since a real fraction will collide with an existing hash),
    keeps only the new ones, and repeats for the shortfall. Oversample escalates
    (doubles) after any round that accepts nothing. Hard guarantee, not
    best-effort: if `max_rounds` is exhausted with a shortfall remaining, the graph
    genuinely does not have `n_needed` additional distinct walks at this
    max_walk_length below the requested budget — raises RuntimeError rather than
    silently padding with duplicates.
    """
    node_arr = np.array(list(nbrs.keys()), dtype=np.int64)
    prefix_cap = max_walk_length // 3
    accepted_walks = []
    n_candidates_generated = 0
    remaining = n_needed
    round_i = 0
    cur_oversample = oversample

    while remaining > 0 and round_i < max_rounds:
        gen_n = max(int(remaining * cur_oversample), remaining)
        tasks = _chunk_tasks(gen_n, num_workers)
        worker = partial(
            _fill_chunk_dedup, nbrs, lbls, rev_nbrs, rev_lbls, node_arr,
            max_walk_length, prefix_cap,
        )
        round_seed = base_seed + 555_555_555 + round_i * 1_000_000
        if len(tasks) == 1:
            w, h = worker(round_seed, 0, tasks[0][0], tasks[0][1])
            batches = [(w, h)]
        else:
            with mp.Pool(processes=len(tasks)) as pool:
                results = pool.starmap(
                    worker,
                    [(round_seed, tasks[i][2], tasks[i][0], tasks[i][1])
                     for i in range(len(tasks))],
                )
            batches = [(w, h) for w, h in results]

        accepted_before_round = len(accepted_walks)
        for w, h in batches:
            n_candidates_generated += len(w)
            for walk, hh in zip(w, h):
                if remaining <= 0:
                    break
                if hh in seen_hashes:
                    continue
                seen_hashes.add(hh)
                accepted_walks.append(walk)
                remaining -= 1

        if len(accepted_walks) == accepted_before_round:
            cur_oversample *= 2.0  # this round found nothing new -- search harder
        round_i += 1

    exhausted = remaining > 0
    if exhausted:
        raise RuntimeError(
            f"_generate_dedup_fill: could not find {n_needed} additional distinct "
            f"walks (found {n_needed - remaining}, short by {remaining}) after "
            f"{round_i} rounds and {n_candidates_generated} candidates generated "
            f"(max_walk_length={max_walk_length}). The graph does not have enough "
            f"genuinely distinct walks left at this budget to honor the "
            f"no-duplicates guarantee -- lower num_walks or raise max_walk_length."
        )

    return accepted_walks, dict(
        n_requested=n_needed, n_accepted=n_needed,
        n_candidates_generated=n_candidates_generated, n_rounds_used=round_i,
        exhausted=False, n_shortfall=0,
    )


def edge_cover_walks(
    edges, num_walks, max_walk_length, seed, num_workers, telemetry_out=None,
):
    """Guarantee every edge appears in exactly one forced anchor walk, then fill
    the remaining budget with genuinely distinct walks (hard-fails via
    `_generate_dedup_fill` if the graph structurally cannot supply enough distinct
    walks at this max_walk_length).

    Two mechanisms:
    1. One forced anchor walk per edge (`[N_u, E_label, N_v]` + random forward
       continuation, via `_anchor_chunk`). Each edge's anchor walk's first 3 tokens
       are that edge's unique `(u, label, v)` key, so no two anchor walks can ever
       collide with each other.
    2. A dedup-fill phase (`_generate_dedup_fill`) for whatever budget remains
       after every edge has its one anchor.
    """
    edge_index = _build_edge_index(edges)
    if not edge_index:
        return []
    nodes, nbrs, lbls, eids = _build_adj_with_eids(edges, edge_index)
    if not nodes:
        return []
    rev_nbrs, rev_lbls = _build_rev_adj(edges)

    m = len(edge_index)
    if num_walks < m:
        print(f"WARNING: edge_cover num_walks={num_walks} < |E|={m}; coverage "
              f"will be incomplete and may underperform uniform. Increase num_walks.")
    uniq_edges = list(edge_index.keys())  # index i -> (u, v, label) with eid == i

    base_seed = int(seed)
    order_rng = np.random.default_rng(base_seed + 777_777_777)
    order = np.arange(m, dtype=np.int64)
    order_rng.shuffle(order)
    n_anchor = min(num_walks, m)
    anchor_edges = [
        (uniq_edges[i][0], uniq_edges[i][1], uniq_edges[i][2], int(i))
        for i in order[:n_anchor]
    ]

    tasks = _chunk_tasks(len(anchor_edges), num_workers)
    chunks = [anchor_edges[s:e] for s, e, _ in tasks]
    worker = partial(_anchor_chunk, nbrs, lbls, eids, m)
    anchor_seed = base_seed + 10_000_000
    if len(tasks) == 1:
        _, walks, _vc = worker(chunks[0], max_walk_length, anchor_seed, 0)
    else:
        with mp.Pool(processes=len(tasks)) as pool:
            results = pool.starmap(
                worker,
                [(chunks[i], max_walk_length, anchor_seed, tasks[i][2])
                 for i in range(len(tasks))],
            )
        results.sort(key=lambda r: r[0])
        walks = []
        for _, w, _vc in results:
            walks.extend(w)

    seen_hashes = {
        hashlib.blake2b("|".join(w).encode(), digest_size=8).digest() for w in walks
    }

    remaining = num_walks - len(walks)
    fill_telemetry = dict(n_requested=0, n_accepted=0, n_candidates_generated=0,
                           n_rounds_used=0, exhausted=False, n_shortfall=0)
    if remaining > 0:
        fill_walks, fill_telemetry = _generate_dedup_fill(
            nbrs, lbls, rev_nbrs, rev_lbls, max_walk_length, num_workers,
            remaining, seen_hashes, base_seed,
        )
        walks.extend(fill_walks)
    walks = walks[:num_walks]

    if telemetry_out is not None:
        telemetry_out["summary"] = dict(m=m, n_anchor_walks=n_anchor)
        telemetry_out["fill"] = fill_telemetry
        corpus_hashes = [
            hashlib.blake2b("|".join(w).encode(), digest_size=8).digest()
            for w in walks
        ]
        n_distinct_total = len(set(corpus_hashes))
        telemetry_out["corpus"] = dict(
            n_walks_total=len(walks),
            n_distinct_total=n_distinct_total,
            dup_rate_total=(1.0 - n_distinct_total / len(walks)) if walks else 0.0,
        )

    return walks


def sample_walks(
    edges, cfg, seed, num_workers=1,
    train_set=None, mask_set=None, val_set=None, test_set=None,
    telemetry_out=None,
):
    """Dispatch to the configured walk sampling strategy.

    Args:
        edges:       full list of (u, v, label) including val/test
        cfg:         OmegaConf config (uses cfg.dataset.* keys)
        seed:        base integer seed (from reproducibility.seed)
        num_workers: parallel workers
        train_set/mask_set/val_set/test_set: unused by edge_cover, kept in the
            signature for call-site compatibility with prepare_data.py.
        telemetry_out: optional dict, filled in place with per-edge duplicate/
                       coverage telemetry.

    Returns:
        List of walk token lists.
    """
    strategy = str(getattr(cfg.dataset, "walk_strategy", "edge_cover"))
    num_walks = int(cfg.dataset.num_walks)
    max_walk_length = int(cfg.dataset.max_walk_length)

    if strategy != "edge_cover":
        raise ValueError(
            f"Unknown walk_strategy={strategy!r}. This release only implements "
            "'edge_cover', the strategy every production config uses."
        )

    return edge_cover_walks(
        edges, num_walks, max_walk_length, seed=seed,
        num_workers=num_workers, telemetry_out=telemetry_out,
    )
