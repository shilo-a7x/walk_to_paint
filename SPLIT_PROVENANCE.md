# Split provenance — how the baseline splits were made, why they didn't match the walk model, and the fix

This documents the origin of the walk-vs-GNN split mismatch (reconstructed from git
history + the archived Copilot chat in `old_chats/baselines.json` →
`old_chats/baselines_cleaned.md`) and the corrected, walk-derived generator that
replaces it. Companion: `FABRICATED_REVERSE_EDGES.md` (the symptom report).

## TL;DR

The walk model and the GNN/SGNN baselines were **never** scored on the same held-out
edges. Three independent split procedures existed; the baselines used two of them, the
walk model a third. The fix (`baselines/prepare_splits.py::build_canonical_split`)
re-derives the baseline artifacts directly from the **frozen walk split**
(`data/<ds>/dataset_cache.pt["splits"]`) so test edges are identical, writing to
isolated paths (`baselines/splits_canonical/`) without touching anything old.

## The three split procedures (all seed=42, all "80/10/10")

1. **Walk model** — `src/data/prepare_data.py::split_edges`. Directed edges, **nested
   4-way** stratified split via 3 sequential `sklearn.train_test_split` calls:
   train(0.48) → mask(0.32) → val(0.10) / test(0.10). Persisted to
   `data/<ds>/dataset_cache.pt["splits"]` as `{train, mask, val, test}` sets of raw
   `(u,v,sign)`. This is the **source of truth** (and is never re-run by the fix).
2. **SGA legacy CSVs** — `prepare_splits.py::stratified_split` (now
   `_deprecated_stratified_split`). Directed edges, **3-way** sklearn split
   train(0.8) → val/test. A *different call sequence* than the walk split ⇒ different
   test edges even at the same seed. Consumed by the SGA-GSGNN/SGCN/SiGAT pipeline.
3. **`baselines/splits/<ds>.pt`** — `prepare_splits.py::make_undirected_splits` (now
   `_deprecated_make_undirected_splits`). Canonical (min,max) undirected, conflict→
   negative, **symmetrized to bidirectional** (fabricates a reverse mirror for every
   one-directional edge), then split with **`torch.randperm`** (a different RNG family
   than sklearn) over the canonical pairs. Consumed by GINEConv, CSG, CopulaLSP/SNEA,
   SE-SGformer, bare-SiGAT.

## Why #3 diverged so badly (the originating mistake)

From the chat (interactions 4–7): the team deliberately re-implemented the undirected
split with `torch.randperm` because it **mirrors CopulaLSP's own native
`create_all_masks`**. They reasoned the resulting test set would only differ "slightly"
from the walk model's because reciprocal merges + sign conflicts are "<1% of edges,"
and accepted it ("yeah whatever if it is small anyway"). That reassurance was wrong:

- A `torch.randperm` over a *different-length, re-ordered, undirected* edge array is an
  **independent permutation** of the walk model's sklearn split — not a near-copy.
  Measured overlap was ~10% (i.e. two independent ~10% draws), not ~99%.
- The "<1%" conflict number came right after a **buggy measurement** (the
  `dataset.name` override wasn't loading different files; identical numbers for every
  dataset then a ZeroDivisionError — `baselines_cleaned.md` ~lines 679–733). The
  corrected rate is 0.3–1% of *all* edges, but reciprocal pairs are 20–40% of edges and
  2–12% of those conflict, so conflict→negative flips a real directed label on
  **248–2703 edges per dataset**.

Net symptoms (see `FABRICATED_REVERSE_EDGES.md`): 14–48% of the GNN test set was
fabricated reverse edges, and walk-test vs GNN-test overlapped only ~10%.

## Why `prepare_splits.py` was missing

A filesystem mishap (chat interactions 31–33) wiped several top-level `baselines/`
scripts (`prepare_splits.py`, some `run_*_ordered.sh`, `collect_all_results.py`); the
results subdirs and the `.pt`/CSV data survived, and everything **except**
`prepare_splits.py` was recreated at the time. It was recovered here from
`git show 2f253fa:baselines/prepare_splits.py`.

## The fix — `build_canonical_split`

Reads the frozen walk split and projects it onto the formats the baselines already
consume (so runners need only repointing, no logic change):

- **Directed masks** (`trn/val/tst_mask`, used by every baseline except CopulaLSP) are
  set at the **directed** level: a position is masked only if it is a *real* directed
  edge, in the bin the walk split assigned it. Fabricated reverse positions are left
  unmasked ⇒ **no fabricated edge in any val/test mask**; directed `tst` == walk test
  exactly.
- **`uni_*` masks** (canonical-pair level, used only by CopulaLSP/SNEA's
  `CopulaTrainer`) use a clean/straddle rule: a pair is a clean test/val pair only if
  *all* its real directions are in walk-test / walk-val; anything else (any train
  direction, or a test/val straddle) goes to train, so CopulaLSP never tests on a
  leaked pair. `uni_tst` is therefore the "clean subset" (per-dataset size below).
- **Node ids:** raw loader ids are sparse; baselines need dense `0..N-1` (and
  SE-SGformer's O(N²) matrix *requires* it). Remap via `enumerate(sorted(nodes))` (same
  convention as the legacy code, so `num_nodes`/dense ids match the historical `.pt`),
  and additionally store explicit `raw2dense` / `dense2raw` / `edge_raw_uv` so any
  diagnostic can join walk(raw) ↔ GNN(dense) per-edge on raw `(u,v)`.
- **edge_weight:** real directed sign where the edge is real, else the canonical sign
  (so directed consumers see real labels; CopulaLSP's derived `uni_edge_weight` is
  correct except on conflict pairs, which are excluded from its strict metric).

Build-time assertions (all 6 datasets, all passing): directed test set (mapped to raw)
== walk test set; masks are a clean partition of the real edges; no fabricated edge in
any directed mask; `uni_tst` pairs are all-test (no straddle leak); dense ids
contiguous `0..N-1`; `dense2raw` round-trips; `num_nodes` == unique node count.

## Per-dataset result (canonical split)

| dataset | directed tst (== walk test) | canonical pairs | uni clean-test pairs |
|---|---|---|---|
| bitcoin-alpha | 2,419 | 14,124 | 528 |
| bitcoin-otc | 3,560 | — | 888 |
| epinions | 84,080 | — | 59,533 |
| slashdot090221 | 54,921 | — | 45,724 |
| wiki-elec | 10,370 | — | 9,816 |
| wiki-rfa | 17,722 | — | 16,518 |

## Important downstream nuance — walk coverage

The directed `tst` above is the walk model's **nominal** test set (the frozen split).
Under the old uniform sampler the walk model only *predicted* edges that actually appeared
in a sampled walk, so on larger/sparser graphs its evaluated set was smaller (full coverage
on bitcoin-alpha/otc; e.g. epinions walk_full=73,783 of 84,080).

**RESOLVED (2026-06-29):** the E15 `k_cover` k=5 edge-anchored sampler now drives walk
coverage to ~100% of every dataset's nominal test set, so walk_full == the full directed
`tst` above on all 6 (e.g. epinions walk_full=84,080). The per-edge walk-vs-GNN join (on raw
`(u,v)`) therefore overlaps the full walk-covered set, versus the old ~10% independent
overlap. See `WALK_COVERAGE.md` and CLAUDE.md's SOTA table.

## What is / isn't changed

- **Not changed:** the walk model and its split (frozen, authoritative); the old
  `baselines/splits/*.pt`, SGA CSVs, `results_our_splits/`, and `all_results.csv` (left
  intact; new results go to isolated paths).
- **Changed:** `baselines/prepare_splits.py` (old generators kept, clearly marked
  `_deprecated_*`); new artifacts under `baselines/splits_canonical/`.
