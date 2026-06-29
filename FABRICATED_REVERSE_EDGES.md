# `baselines/splits/<ds>.pt` contains fabricated reverse edges (14-48% of edges)

Discovered while building Lead 4b (`scripts/lead4_twohop_path_consistency.py`):
the walk model and the GNN baselines (GINEConv, SiGAT) reported very different
test-set sizes for the same dataset/split (e.g. bitcoin-alpha: walk n=2419,
GINEConv/SiGAT n=2826). The gap is not a bug in any Lead-4 script -- it's
baked into `baselines/splits/<ds>.pt` itself, and it affects every script
that treats that file's `edge_index` as ground truth for a per-edge or
per-node metric.

## The mechanism

`baselines/splits/<ds>.pt`'s `edge_index`/`edge_weight` is NOT the real
directed graph re-indexed densely. For every real one-directional edge
`u->v` that has **no** real reciprocal `v->u` in the dataset, the split
generator fabricates a `v->u` row and **copies `u->v`'s sign onto it** --
confirmed against `scripts/generate_synthetic_fog_graph.py:184-217`
(`write_splits`), which explicitly documents mirroring "both (u,v) and (v,u),
same sign... matching baselines/splits/*.pt's own convention for the existing
real datasets." Splits are assigned at the canonical (undirected, min/max)
edge level, so a real edge and its fabricated mirror always land in the same
train/val/test split.

Net effect: `baselines/splits/<ds>.pt` is a *symmetrized* graph (every node
pair has edges in both directions, real or not), while the walk model's own
canonical edge loader (`scripts/balance_theory_paths.py` /
`scripts/node_mi_structural_embedding.py`'s `load_edges_canonical`) returns
only the real directed edges, no fabrication.

## Detection

```python
from scripts.node_mi_structural_embedding import DATASET_CONFIGS, load_edges_canonical
from scripts.lead4_entropy_heterogeneity import _ds_key

cfg = DATASET_CONFIGS[_ds_key(ds_name)]
raw_edges = load_edges_canonical(cfg["ds_name"])           # real, raw node ids
nodes = sorted({n for u, v, _ in raw_edges for n in (u, v)})
raw2dense = {n: i for i, n in enumerate(nodes)}             # matches baselines/splits dense ids
real_set = {(raw2dense[u], raw2dense[v]) for u, v, _ in raw_edges}

splits = torch.load(f"baselines/splits/{ds_name}.pt", weights_only=False)
ei = splits["edge_index"]
is_fabricated = [(int(ei[0,i]), int(ei[1,i])) not in real_set for i in range(ei.shape[1])]
```

This is now a reusable helper: `build_real_dense_edge_set(ds_name)` in
`scripts/lead4_entropy_heterogeneity.py`.

## Per-dataset numbers

| dataset | real edges | total dense edges (splits.pt) | fabricated (overall) | fabricated in test split |
|---|---|---|---|---|
| bitcoin-alpha | 24,186 | 28,248 | 4,062 (14.4%) | 389 / 2,826 (13.8%) |
| bitcoin-otc | 35,592 | 42,984 | 7,392 (17.2%) | 735 / 4,300 (17.1%) |
| epinions | 840,799 | 1,422,420 | 581,621 (40.9%) | 58,154 / 142,242 (40.9%) |
| wiki-elec | 103,689 | 201,524 | 97,835 (48.5%) | 9,769 / 20,154 (48.5%) |
| wiki-rfa | 177,211 | 341,514 | 164,303 (48.1%) | 16,434 / 34,154 (48.1%) |
| slashdot090221 | 549,202 | 1,000,962 | 451,760 (45.1%) | 45,089 / 100,098 (45.0%) |

On 4 of 6 datasets, **40-48% of the GNN baselines' nominal test set never
existed in the real data** -- its label is a sign copied from the real edge
in the opposite direction, not an observed rating/vote.

## Why this matters (and when it doesn't)

This contamination is a problem specifically for **edge-level diagnostics
that compare the walk model against a GNN baseline using a metric derived
from "all edges of the graph"** (sign-entropy, 2-hop path consistency,
degree, ...). If that metric is computed from `baselines/splits/<ds>.pt` for
the GNN side and from `load_edges_canonical` for the walk side, the two
models are silently being scored against *different graphs*, and the
GNN-side test set itself contains edges the walk model is never even asked
to predict (because they don't exist).

It is **not** a problem for:
- **SOTA-table aggregate AUC** (CLAUDE.md's reported baseline numbers): those
  come from each baseline's own training/eval pipeline running exactly as
  designed for that architecture (which legitimately treats the graph as
  bidirectional for message passing) -- not a redone comparison against this
  diagnostic's notion of "the real graph." Per-architecture eval is doing
  what it's supposed to do; nothing to fix there.
- Any analysis that explicitly wants "what the GNN actually saw" (e.g.
  reconstructing a GNN's own h^(1) from its real training-time message
  passing) -- see `lead2_gnn_bottleneck_mi.py` below.

## Scripts fixed

- **`scripts/lead4_entropy_heterogeneity.py`** and
  **`scripts/lead4_twohop_path_consistency.py`** are now **canonical-native** and
  no longer touch `baselines/splits/<ds>.pt`'s dense graph at all: they read
  per-edge predictions from `predictions_raw_canonical.pkl` (all models in one
  raw `(u, v)` id space) and build the entropy / 2-hop-path basis from the real
  canonical edge list (`load_edges_canonical`). So the fabricated-reverse-edge
  problem and the `build_real_dense_edge_set` filter are **moot for these two
  scripts** — there is no dense graph and no fabricated row to filter. (The
  earlier `get_gineconv_data`/`get_sigat_data` dense-filter fix has been
  superseded by this rewrite; see CANONICAL_RERUN_FINDINGS.md.)

## Scripts that read `baselines/splits/<ds>.pt`'s `edge_index` and may need review

If you pick up node-entropy work or any other per-edge/per-node diagnostic
that crosses model types, check these:

- **`scripts/lead1_degree_gap.py`** (`baseline_degree_dict`): computes node
  degree from the full (fabricated-inclusive) `edge_index`, and uses that
  *same* (slightly inflated) degree for **both** the walk-model and GNN
  sides of the degree-stratified AUC-gap comparison. Not a cross-model
  ground-truth mismatch like Lead 4's was (both sides share one degree
  definition), but it does mean "degree" is a noisy proxy for true degree on
  the worse-affected datasets. Could be a contributing factor to Lead 1's
  "no consistent pattern" null result on degree-stratified gap -- not
  re-verified here.
- **`scripts/lead2_gnn_bottleneck_mi.py`** (`load_train_edges`): uses the
  full `edge_index` + `trn_mask` and explicitly documents this as "the only
  edges that actually fed into h_v^(1) during the forward pass." This is
  **correct as-is** -- the GNN really did do message passing over the
  fabricated/symmetrized graph, so this script is describing the real
  computation, not re-deriving a "true" graph. Do not apply this fix here.
- **`scripts/lead2_walk_relay_mi.py`** (`build_neighbor_sets`): builds an
  undirected neighbor-set by adding both `s->d` and `d->s` for every row in
  `edge_index`. Checked: this is a no-op with respect to fabrication, because
  a single real directed edge already produces both neighbor entries before
  any fabricated mirror is even considered. **Not affected.**

If you touch any other script that loads `baselines/splits/<ds>.pt` and uses
its `edge_index` as a stand-in for "the real graph" (rather than "what this
specific GNN's forward pass saw"), filter it through
`build_real_dense_edge_set` first.

## SEPARATE ISSUE (RESOLVED 2026-06-29): the walk model's test split and the GNN baselines' test split

> **UPDATE — RESOLVED 2026-06-29.** The two distinct sub-problems below have **both** been
> addressed, so the "no shared ground truth" worry no longer applies to the standing
> diagnostics:
> 1. **Split mismatch (the ~10% overlap)** was a *provenance* bug, not a sampling property:
>    the old `baselines/splits/*.pt` were generated independently of the walk split. Fixed by
>    `baselines/prepare_splits.py::build_canonical_split`, which re-derives every baseline
>    artifact from the frozen walk split into `baselines/splits_canonical/`. After this, the
>    walk and GNN test sets are the **same partition** (`SPLIT_PROVENANCE.md`,
>    `CANONICAL_RERUN_FINDINGS.md`).
> 2. **Walk coverage (walk evaluated only ~85–88% of its own nominal test set)** is fixed by
>    the **E15 `k_cover` k=5** sampler — walk now covers ~100% of every test set, so the
>    per-edge walk∩GINEConv∩SiGAT overlap is the full shared edge set on all 6
>    (`WALK_COVERAGE.md`, `outputs/lead4_entropy_heterogeneity/E15_FULLCOVERAGE_RERUN_NOTE.md`).
>
> Net: Leads 1/4/4b are now computed on identical, full-coverage, same-partition edges for
> all models. The historical description below is retained as provenance of how the problem
> was first found.

The fabrication fix above only addresses *fake* edges polluting the GNN
side's test set. After fixing it, a follow-up check (2026-06-24) found a
second, more serious problem: even restricted to real edges, **the walk
model's test split and the GNN baselines' test split are essentially
independent random samples of the same edge pool, not the same partition.**

```
dataset          total_real   walk_n   gine_n   observed_overlap   expected_if_independent
bitcoin-alpha       24,186     2,419    2,437          238                  244
bitcoin-otc         35,592     3,560    3,565          370                  357
epinions           840,799    73,783   84,088        7,262                7,379
wiki-elec          103,689     8,857   10,385          890                  887
wiki-rfa           177,211    15,280   17,720        1,519                1,528
slashdot090221     549,202    53,962   55,009        5,331                5,405
```

`walk_test ⊆ gnn_test` is **False on every dataset** -- overlap is ~10% of
either set on all 6 datasets, matching almost exactly what you'd expect from
two *independent* random ~10% draws over the same pool (`expected_overlap =
walk_n * gnn_n / total_real`). This is not a rounding/fabrication artifact;
it means CLAUDE.md's "baselines/ uses identical splits (verified) —
comparison is apples-to-apples" does not hold at the edge level: both
pipelines nominally do 80/10/10 seed=42, but something about how each
canonicalizes/orders edges before sampling makes them land on different
edges entirely.

**Why this matters more than the fabrication fix:** any per-edge comparison
that buckets edges by a shared property (sign-entropy, 2-hop consistency,
degree, ...) and compares AUC bucket-by-bucket across the walk model and a
GNN baseline is implicitly assuming the two models were scored on the same
held-out edges. They weren't. Each model's own overall/per-bucket AUC is
real for *its own* test set, but the side-by-side comparison has no shared
ground truth underneath it.

**Not yet fixed -- needs a decision, not just a filter:**
- Intersecting both test sets only keeps ~10% of either side's edges, which
  craters sample size for fine-grained binning (e.g. bitcoin-alpha would have
  ~238 edges total to spread across 6-8 buckets).
- Re-evaluating a GNN baseline on the walk model's exact test split would
  require checking whether those edges leaked into that baseline's own
  training set under its own (different) split -- not safe to assume.
- Likely affects **Lead 1's degree-stratified gap analysis** too (same
  walk-test-vs-GNN-test structure, not re-verified there).

Do not silently "fix" this by intersecting or re-filtering without thinking
through the leakage/power tradeoffs above -- flag and revisit deliberately.
