# Phase 0(B) — walk duplication measurement, all 6 datasets

Exact-duplicate rate in each production k_cover (k=5) cache — every anchor
walk hashed by its raw token-id sequence. wiki-elec/wiki-rfa were measured
ad-hoc earlier this session (scratchpad `check_walk_duplicates.py`); this run
covers the remaining 4 with `scripts/measure_walk_duplication.py`.

| dataset | cache nw | n_walks | unique | dup rate | max multiplicity |
|---|---|---|---|---|---|
| wiki-elec | dataset_cache__k_cover_k5_nw500000_mw80_seed42.pt | 500,000 | 353,718 | 29.26% | 45 |
| wiki-rfa | dataset_cache__k_cover_k5_nw1000000_mw80_seed42.pt | 1,000,000 | 872,024 | 12.80% | 72 |
| bitcoin-alpha | dataset_cache__k_cover_k5_nw5000000_mw80_seed42.pt | 5,000,000 | 4,704,103 | 5.92% | 1579 |
| bitcoin-otc | dataset_cache__k_cover_k5_nw2000000_mw80_seed42.pt | 2,000,000 | 1,860,724 | 6.96% | 456 |
| epinions | dataset_cache__k_cover_k5_nw3000000_mw80_seed42.pt | 3,000,000 | 2,610,352 | 12.99% | 37 |
| slashdot090221 | dataset_cache__k_cover_k5_nw5000000_mw80_seed42.pt | 5,000,000 | 3,027,219 | 39.46% | 109 |

## Duplicate rate by walk length (tokens; hops = (tokens-1)/2)

| dataset | len=3 | len=5 | len=7 | len=9 | len=11 | len=13 | len=15 |
|---|---|---|---|---|---|---|---|
| wiki-elec | 98.1% | 32.6% | 6.3% | 1.5% | - | - | - |
| wiki-rfa | 96.9% | 36.9% | 8.2% | 1.5% | - | - | - |
| bitcoin-alpha | 100.0% | 91.1% | 51.4% | 19.3% | 7.4% | 3.0% | 1.4% |
| bitcoin-otc | 99.6% | 68.1% | 32.2% | 10.9% | 4.7% | 2.0% | 1.0% |
| epinions | 90.9% | 30.0% | 6.6% | 2.1% | 0.9% | 0.4% | 0.2% |
| slashdot090221 | 99.7% | 59.7% | 22.3% | 8.4% | 3.7% | 2.3% | 1.4% |

## Notes

- Mechanism (confirmed on wiki-elec/wiki-rfa, holds structurally everywhere):
  a length-3 walk `[N_u, E_label, N_v]` occurs when `v` is a dead end (no
  outgoing edges) — the anchor has zero randomness available, so every
  anchor of that edge is byte-identical.
- Not a leakage bug (`stage_dataset.py` masks by edge identity, not
  position/duplication) but is bad training-data hygiene — see
  `~/.claude/plans/plan-a-fix-for-glimmering-panda.md`.
