"""Immediate, cache-free check: run the FULLY-FIXED k_cover_bp (anchor dedup +
fill-phase dedup) directly in memory and report the exact corpus-wide duplicate
rate at production-realistic budget. Doesn't touch any cache files on disk, so
it's safe to run alongside the still-training reference jobs using the old caches.
"""
import sys
import time

sys.path.insert(0, ".")

from scripts.balance_theory_paths import load_edges_canonical
from src.data.coverage_aware_sampler import k_cover_walks_bp

CASES = [
    ("epinions", 1),
    ("epinions", 7),
    ("slashdot090221", 1),
    ("slashdot090221", 7),
]
NUM_WALKS = 2_000_000
MAX_WALK_LENGTH = 80
SEED = 42

for ds_name, k in CASES:
    print(f"\n=== {ds_name} k={k} (nw={NUM_WALKS:,}) ===", flush=True)
    edges = load_edges_canonical(ds_name)
    telemetry = {}
    t0 = time.time()
    walks = k_cover_walks_bp(
        edges, num_walks=NUM_WALKS, max_walk_length=MAX_WALK_LENGTH, seed=SEED,
        num_workers=8, k=k, telemetry_out=telemetry,
    )
    t = time.time() - t0
    corpus = telemetry["corpus"]
    fill = telemetry["fill"]
    anchor_summary = telemetry["summary"]
    print(f"  gen time: {t:.1f}s", flush=True)
    print(f"  CORPUS: n_walks={corpus['n_walks_total']:,} "
          f"n_distinct={corpus['n_distinct_total']:,} "
          f"dup_rate={corpus['dup_rate_total']:.4%}", flush=True)
    print(f"  fill: requested={fill['n_requested']:,} accepted={fill['n_accepted']:,} "
          f"candidates_generated={fill['n_candidates_generated']:,} "
          f"rounds={fill['n_rounds_used']} exhausted={fill['exhausted']} "
          f"shortfall={fill['n_shortfall']:,}", flush=True)
    print(f"  anchor: capped_frac={anchor_summary['capped_frac']:.4%} "
          f"dup_after_retries_frac={anchor_summary['dup_after_retries_frac']:.4%}", flush=True)
    del walks

print("\nDone.")
