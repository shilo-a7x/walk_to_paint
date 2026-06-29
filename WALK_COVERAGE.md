# Walk coverage — the walk model does not predict every test edge

> **RESOLVED (2026-06-29).** The edge-anchored `k_cover` sampler (`walk_strategy=k_cover`,
> `walk_k_min=5`) drives node AND edge coverage to ~100% on all 6 datasets while
> matching/beating uniform AUC (and *improving* it on the same edges where coverage was
> low). The E15 SOTA in CLAUDE.md uses it; the coverage caveat below describes the OLD
> uniform sampler and is kept for provenance. Full results:
> `outputs/walk_coverage_analysis/E15_SWEEP_RESULTS.md`; sampler:
> `src/data/coverage_aware_sampler.py::k_cover_walks_fast`; plan:
> `~/.claude/plans/plan-a-fix-for-glimmering-panda.md`. **Caveat: `predictions_raw_canonical.pkl`
> and Leads 1/4/4b still hold the OLD uniform walk predictions until rebuilt (Phase 4).**

## The problem (one line) — OLD uniform sampler

The walk-Transformer can only produce a prediction for a test edge that actually
appears in at least one sampled random walk. On larger/sparser graphs a minority of
held-out edges are never traversed, so the walk model's **evaluated** test set is a
strict subset of its **nominal** test split — while the GNN baselines (which see the
full edge tensor) evaluate every nominal test edge.

## Measured coverage (walk predictions ÷ nominal test split)

| dataset | walk predicts | nominal test | coverage |
|---|---|---|---|
| bitcoin-alpha | 2,419 | 2,419 | 100.0% |
| bitcoin-otc | 3,560 | 3,560 | 100.0% |
| slashdot090221 | 53,962 | 54,921 | 98.3% |
| wiki-rfa | 15,280 | 17,722 | 86.2% |
| epinions | 73,783 | 84,080 | 87.8% |
| wiki-elec | 8,857 | 10,370 | 85.4% |

(`nominal test` = `len(data/<ds>/dataset_cache.pt["splits"]["test"])`; `walk predicts`
= unique test edges with ≥1 prediction in the walk model's `test_predictions.pkl`, e.g.
as surfaced in `outputs/lead4_entropy_heterogeneity/predictions_raw_canonical.pkl`.)

## Why it happens

The pipeline tokenizes each random walk into an alternating node/edge sequence and
supervises masked **edge** tokens. A test edge is scored only if it shows up as an edge
token in some walk. With a finite walk budget (`walk.num_walks`, `max_walk_length`),
edges incident to low-degree / peripheral nodes — more common in the large sparse
graphs — may never be sampled, so they receive no prediction. Dense small graphs
(bitcoin-alpha/otc) reach 100%; the sparser wiki/epinions graphs sit at ~85–88%.

## Why it matters

1. **walk-vs-GNN per-edge comparison.** With the canonical split fix
   (`SPLIT_PROVENANCE.md`), the GNN test set == the walk's *nominal* test set, so a join
   on raw `(u,v)` overlaps ~all walk-covered edges (the table above) instead of the old
   ~10% independent overlap. But the join is still capped at walk coverage — the
   uncovered 12–15% of edges (on wiki/epinions) have no walk prediction to compare.
2. **SOTA-table fairness.** The GNN AUC is over the full nominal test; the walk AUC is
   over its covered subset. These are *almost* the same set but not identical. Current
   convention: report each on what it actually evaluated and footnote coverage. An
   alternative (not yet done) is to restrict the GNN eval to walk-covered edges for a
   strict head-to-head.
3. **Possible mild selection bias.** Uncovered edges are systematically the
   low-degree/peripheral ones. If the walk model (or a GNN) is differentially good on
   peripheral edges, comparing "walk on covered" vs "GNN on all" could shift numbers
   slightly. Unquantified.

## What a future session could check / fix

- **Quantify the bias:** compute GNN AUC on walk-covered vs walk-uncovered edges
  separately (now trivial with the canonical split + stored `dense2raw` — the uncovered
  set = nominal test minus walk-covered). If GNN AUC differs materially between the two,
  the coverage gap is not benign.
- **Raise coverage:** increase `walk.num_walks` / `max_walk_length`, or add
  test-edge-seeded walks (the sampler already supports neg-emphasis / edge-seeded modes;
  see `config.yaml` `walk.*` and `src/data/walk_sampler.py` / `coverage_aware_sampler.py`)
  to guarantee every test edge is traversed ≥1×, then re-measure.
- **Decide the SOTA convention** explicitly (full nominal test + footnote, vs restrict
  GNNs to walk-covered edges) and apply it uniformly in `collect_all_results.py`.

## Scope / status

This is a **property of the walk model**, independent of the split-mismatch bug fixed in
`SPLIT_PROVENANCE.md`. The canonical split does not change coverage; it only makes the
GNN side share the walk's nominal test set so the gap is now cleanly measurable. No code
change has been made for coverage itself — this file is the open record.
