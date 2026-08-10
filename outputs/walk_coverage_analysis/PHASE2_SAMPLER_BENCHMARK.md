# Phase 1+2 — cache isolation + optimized k_cover benchmark (2026-06-28)

## Phase 1 — cache isolation (done)
`src/data/prepare_data.py`: cache filename is now keyed by
`(walk_strategy, num_walks, max_walk_length, seed[, k])` via `_keyed_cache_path()`.
SAVE always targets the keyed path, so the legacy `data/<ds>/dataset_cache.pt` (current
SOTA + all Leads) is **never overwritten**. LOAD reuses the legacy file only for
`uniform` runs whose `num_walks` matches it (verified by walk count), else builds a
keyed cache. Verified: legacy caches byte-unchanged after this session; keyed-path
unit test produces e.g. `dataset_cache__k_cover_k5_nw1500000_mw80_seed42.pt`.

## Phase 2 — optimized parallel k_cover (`k_cover_walks_fast`)
New `k_cover_walks_fast` in `src/data/coverage_aware_sampler.py` (dispatch `k_cover`
now points to it). Parallel anchor passes (mp.Pool, same throughput pattern as the
uniform sampler); per-adjacency `eid` precompute so workers credit visits with O(1)
array indexing (no per-step dict lookup); visit counts reduced between passes so later
passes skip satisfied edges; uniform fill for the remainder. Dict-based adjacency keeps
it sparse-id-safe (wiki-rfa's ~1e9 ids never allocate max_id arrays). Tokens emit raw
ids. Guard added: warns when `num_walks < |E|` (anchor-first regime where it can
underperform uniform).

## Benchmark — recommended per-dataset budgets (k=5)
All achieve **100% node + edge coverage** and **full k=5 saturation floor**
(min visits ≥5, ge5=1.0) unless noted. gen time = walk generation (8 workers).

| dataset | \|E\| | OLD nw | **NEW nw** | gen (s) | edge cov | node cov | %edges ≥5 | min visit |
|---|---|---|---|---|---|---|---|---|
| bitcoin-alpha | 24k | 5,000,000 | **200,000** | 5.4 | 1.000 | 1.000 | 1.000 | 6 |
| bitcoin-otc | 36k | 500,000 | **200,000** | 5.1 | 1.000 | 1.000 | 1.000 | 5 |
| wiki-elec | 104k | 500,000 | **500,000** | 5.6 | 1.000 | 1.000 | 1.000 | 5 |
| wiki-rfa | 177k | 500,000 | **1,000,000** | 15.1 | 1.000 | 1.000 | 1.000 | 5 |
| epinions | 841k | 500,000 | **1,500,000** | 60.7 | 1.000 | 1.000 | 0.914 | 4 |
| slashdot | 549k | 5,000,000 | **3,000,000** | 45.0 | 1.000 | 1.000 | 1.000 | 5 |

Notes:
- **Rule of thumb:** full k=5 needs `nw ≈ 3–8×|E|`. epinions at 1.5M (≈1.8×|E|) hits
  100% coverage and 91% at k≥5 (min 4) — essentially saturated; 2–2.5M would close the
  last ~9% if desired (vs current 500k which only reached 87.8% coverage).
- **k_cover regresses below uniform when `nw < |E|`** (epinions@500k: node cov 0.84 vs
  uniform 0.97 — pass-1 anchors eat the whole budget, no uniform spread). The ≥99.5%
  coverage gate rejects such configs; the new guard warns. Always size `nw ≳ 1.5×|E|`.
- **Throughput** is 1.5–2.5× the uniform sampler at the same budget — acceptable, and
  in absolute terms tiny (≤61s) because the NEW budgets are far smaller than the old
  5M for bitcoin-alpha/slashdot. Net preprocessing time generally DROPS vs current SOTA.
- **Big efficiency win**: bitcoin-alpha 5M→200k (25×), slashdot 5M→3M, while raising
  the under-budgeted 500k datasets (wiki-rfa→1M, epinions→1.5M) to full coverage.

## Validation that the sampler preserves the uniform bulk
On every dataset the median visit count under k_cover ≈ the uniform median at the same
budget (e.g. bitcoin-otc median 340 both) — k_cover only lifts the LOW tail to the k=5
floor; it does not distort the saturation profile the model already benefits from.

## Open for Phase 3
- Confirm epinions: ship 1.5M (91% at k≥5, min 4, full coverage) or bump to ~2M for a
  hard k=5 floor. Recommend **1.5M** (0.F shows AUC benefit plateaus by k≈5; the 9% of
  edges at 4 visits are immaterial) unless a clean floor is wanted.
- Budgets above are the Phase 3 per-dataset `num_walks` (with `walk_strategy=k_cover`,
  `walk_k_min=5`, `max_walk_length=80`).
