# The `edge_cover` walk sampler — full pipeline, function by function

This traces the production walk sampler (`walk_strategy=edge_cover`, the default in
every `configs/<dataset>.yaml`) from raw edges to the token arrays the model actually
trains on. Written 2026-08-11 in response to a request to verify what the paper's
Methods section claims against what the code actually does — nothing here is
paraphrased from CLAUDE.md or the paper; every step is read directly from
`src/data/prepare_data.py` and `src/data/coverage_aware_sampler.py`.

## 0. Where this sits in the pipeline, and what "edges" means here

Order of calls in `src/data/prepare_data.py` (driven by `run.py`'s data-prep step):

1. `get_edge_list(cfg)` → `get_loader(...)` (`src/data/datasets.py`) — loads and
   postprocesses the raw graph (self-loops removed, multi-edge handling applied per
   dataset). Returns the **full** list of `(u, v, label)` edges.
2. `split_edges(cfg, edges)` — stratified `train_test_split` (sklearn, seeded by
   `reproducibility.seed`) into four **sets** of edge tuples: `train_set`, `mask_set`,
   `val_set`, `test_set` (80/10/10 with train further split 48/32 into train/mask).
   This is a set-membership split only — it does not touch the edges themselves.
3. `get_walks(cfg, edges, train_set, mask_set, val_set, test_set)` → dispatches to
   `sample_walks()` (`src/data/coverage_aware_sampler.py`) → `edge_cover_walks()` when
   `walk_strategy=edge_cover`.

**Critical point: `edges` passed into the sampler is the full list, all four splits
combined, and `edge_cover_walks()` doesn't even accept `train_set`/`mask_set`/
`val_set`/`test_set` as parameters** — it has no notion of which edges are
train/val/test. It samples over pure graph topology and writes each traversed edge's
**true** sign into the walk's token list unconditionally. This is intentional and not
a leakage hole: nothing about *which* edges the walk touches, or what sign token gets
written at sampling time, is later shown to the model as-is. Splitting is enforced
downstream (see §5) at every single batch materialization, not baked into the walk
corpus. The walk sampler's only job is topology-driven traversal; masking is a
completely separate, later concern.

## 1. Anchor phase — guarantee every edge appears at least once

`edge_cover_walks()` (coverage_aware_sampler.py:1594):

1. `_build_edge_index(edges)` — assigns every unique `(u, v, label)` triple a compact
   0-based integer id, first-occurrence-wins for exact duplicates (relevant only for
   `multiedge_handling=keep` datasets — epinions, slashdot090221 — where the same
   triple could in principle repeat; empirically it doesn't, see the Datasets
   paragraph investigation).
2. `_build_adj_with_eids(edges, edge_index)` — forward adjacency (`nbrs[u]` = array of
   `u`'s out-neighbors, `lbls[u]` = matching edge signs, `eids[u]` = matching edge_index
   ids), built once, shared read-only across all workers.
3. Anchor order: `order_rng = np.random.default_rng(base_seed + 777_777_777)` shuffles
   the `m = len(edge_index)` edges once. `n_anchor = min(num_walks, m)` — if the walk
   budget is smaller than the edge count, only that many edges get a direct anchor this
   run (a `WARNING` is printed; production budgets are always well above `|E|`, see
   CLAUDE.md's per-dataset budget table).
4. The first `n_anchor` edges in that shuffled order become `anchor_edges`, chunked
   across `num_workers` via `_chunk_tasks` and run in an `mp.Pool` (or inline if
   `num_workers==1`), each chunk handled by `_kcover_anchor_chunk`
   (coverage_aware_sampler.py:1045), seeded `anchor_seed = base_seed + 10_000_000`,
   further offset by `+ task_id` per chunk inside the worker (so results are
   reproducible regardless of how many workers ran, same pattern as every other
   sampler in this file).

**Per-anchor walk construction** (`_kcover_anchor_chunk`): for edge `(u, v, label)`,
the walk's first three tokens are forced: `[N_u, E_label, N_v]` — this triple *is* the
edge's own unique key, which is what makes anchor-anchor collision impossible by
construction (two different edges can never produce the same forced first-3-tokens,
so no probabilistic argument is needed for that part of the zero-duplication
guarantee). The walk then continues for up to `max_walk_length - 1` more hops as an
ordinary uniform random walk from `v`: at each step, pick a uniformly random
out-neighbor of the current node via the chunk's own `rng`, append `[E_label, N_next]`,
advance. Every edge this continuation happens to cross (not just the anchor edge
itself) increments that edge's `visit_count` in a local per-chunk bincount
(`local_vc`), which is summed back into the walk-level `visit_count` array after each
chunk — this is why far fewer than `|E|` *walks* are needed to reach `|E|`-edge
*coverage*: a single ~80-hop anchor walk typically visits dozens of edges as a
byproduct, so most non-anchor edges get "covered" for free by someone else's anchor
before the fill phase even starts (this byproduct coverage is why the production
budgets in CLAUDE.md's table, e.g. `1×|E|` for epinions, are sufficient at all — pure
anchor-only coverage would need exactly `|E|` walks with zero byproduct help).

## 2. Fill phase — reach the remaining budget with genuinely distinct walks

`remaining = num_walks - len(anchor_walks)`. If `remaining > 0`,
`_generate_dedup_fill()` (coverage_aware_sampler.py:1311) is called with
`seen_hashes` pre-seeded from every anchor walk's own hash (so fill candidates are
checked against the anchors too, not just against each other).

**Per-candidate construction** (`_fill_chunk_dedup`, coverage_aware_sampler.py:1255) —
this is the part that is *not* a plain uniform random walk, and is worth being
precise about:

1. Pick a uniformly random start node (`node_arr = list(nbrs.keys())`, i.e. any node
   that has at least one *outgoing* edge — a pure sink node, in-degree>0 but
   out-degree 0, can never be picked as a start).
2. Draw a random-length **backward prefix**: `prefix_len ~ Uniform(0, prefix_cap)`
   where `prefix_cap = max_walk_length // 3`. Walk backward from the start node via
   the *reverse* adjacency (`rev_nbrs`/`rev_lbls`, built once by `_build_rev_adj`),
   one uniformly random in-edge at a time, prepending `[N_prev, E_label]` tokens as it
   goes. This is what makes the candidate diverge from a plain random walk: a fresh
   backward prefix is drawn **every single time** a start node is sampled, even if
   that same node was already used as a start earlier in this run.
3. From the original start node, continue **forward** as an ordinary uniform random
   walk for `max_walk_length - actual_prefix_hops` more steps (using the *forward*
   adjacency again, same as the anchor phase's continuation).
4. The full token sequence (prefix + start + forward continuation) is hashed
   (`blake2b`, 8-byte digest) and returned alongside the walk.

**Why the backward prefix exists, mechanically**: without it, a node whose forward
path is short and effectively deterministic (e.g. one hop to a degree-1 dead end)
would produce the *exact same walk* every time it's re-sampled as a start — at
production walk-count budgets, a given node gets picked as a fill-phase start many
times, so this was a real, measured source of corpus duplication before the fix (see
`plan-a-fix-for-glimmering-panda.md`, referenced in `coverage_aware_sampler.py`'s
docstrings). The backward prefix diversifies exactly that failure mode, since the
node's set of *incoming* edges (and its own forward continuation appended after a
different backward path) is generally different from its outgoing path alone.

**Acceptance loop** (`_generate_dedup_fill`): candidates are generated in rounds,
`gen_n = max(remaining * oversample, remaining)` per round (`oversample` starts at
`1.5`), seeded `round_seed = base_seed + 555_555_555 + round_i * 1_000_000` (further
`+ task_id` per worker chunk, same pattern as the anchor phase). Each round's
candidates are checked against `seen_hashes` (mutated in place — every accepted
candidate this round is immediately added, so within-round duplicates are also
caught, not just duplicates vs. the anchor phase); genuinely new ones are kept until
`remaining` reaches 0. If a round accepts nothing at all, `oversample` doubles for the
next round (search harder before giving up). **If `max_rounds` (10) is exhausted with
a shortfall still remaining, this raises `RuntimeError`** — there is no silent
fallback to padding with duplicates. This is a real, hard guarantee, not a
best-effort one: production runs have measured 0.000000% duplication at scale (per
CLAUDE.md), and the failure mode if the graph genuinely can't supply enough distinct
walks at the configured `max_walk_length` is a crash with a message telling you to
lower `num_walks`, raise `max_walk_length`, or accept duplicates via a different
strategy — never a silently-corrupted corpus.

## 3. Determinism

Every phase uses a distinct, additively-offset seed derived from the single
`reproducibility.seed` config value (`base_seed`): anchor order shuffle
(`+777_777_777`), anchor walk construction (`+10_000_000`, `+task_id`), fill-phase
rounds (`+555_555_555 + round_i*1_000_000`, `+task_id`). This means the exact same
walk corpus is produced regardless of `num_workers` (chunk boundaries don't change
which seed a given piece of work uses) — the same reproducibility guarantee the
plain uniform sampler (`walk_sampler.py`) documents for itself.

## 4. What happens to the walks after this (why none of the above is a leakage risk)

`get_tokenizer()` builds the vocabulary from the raw walks (still holding every
edge's **true** sign, val/test included). `encode_walks()` then does exactly one
thing per edge-token position: looks up that specific `(u, v, label)` triple in a
`split_lookup` dict (built once from the four split sets) and tags the position with
its `SplitID` (`TRAIN`/`MASK`/`VAL`/`TEST`, or `BAD` for node-token positions) —
**it does not touch the token itself**; the encoded `input_ids` array still holds the
true sign everywhere at this stage. `build_ragged_arrays()` packs `input_ids` +
`split_mask` + `edge_ids` into the flat CSR arrays that get written to
`dataset_cache*.pt` on disk.

**So the on-disk cache contains every edge's true sign, unconditionally, at every
occurrence.** Masking — replacing val/test signs with `<MASK>` and excluding them
from attention — happens only at `StageViewDataset.__getitem__` time
(`src/data/stage_dataset.py`), freshly, on every single batch materialization, train
or eval. This is a separate investigation (masking/leakage, tracked separately) but
it's worth stating here because it's the reason the walk sampler itself can safely
ignore splits entirely: enforcement lives entirely downstream, applied every time
data is actually read, not baked into a preprocessing step that could be
forgotten or bypassed.

## 5. Strategies this doc does NOT cover

`k_cover` / `k_cover_bp` (k≥1-capable generalizations of the same anchor-then-fill
idea, with more machinery for diversifying a *repeated* anchor visit to the same
edge) are implemented in the same file but are not the current default and are not
traced here — `edge_cover` is explicitly `k_cover_bp` with `k` fixed at 1, which is
why its own docstring says most of `k_cover_bp`'s multi-pass/attempt-tracking/
capped-edge machinery is dead code at `k=1` (never executes, since every edge gets
exactly one direct anchor and there is no "second visit" to diversify). See
`coverage_aware_sampler.py`'s own docstrings for those if the sampler choice is ever
revisited.

## Open items for confirmation

- The paper's current draft describes the fill phase as "ordinary random walks
  starting anywhere in the graph" — per §2 above, that's not quite right (backward
  prefix, and "anywhere" really means "any node with out-degree > 0"). Once this doc
  is confirmed, the paper text needs a plain-language (not hash/algorithm-level)
  fix — something like "duplicate walks are found and re-drawn until every one is
  distinct" without naming the specific mechanism, per the earlier discussion about
  not over-specifying implementation detail in the prose.
- The literal `HOW DO YOU COMPARE???` marker in the current Setup paragraph has a
  one-line answer now (content hash of the token sequence) but per the same
  not-too-technical guidance, the paper probably just needs "walks are checked
  against every walk already in the corpus" without naming blake2b specifically.
