"""Phase 0(B) walk-duplication measurement (read-only).

Extends the ad-hoc wiki-elec/wiki-rfa duplicate check (this session's scratchpad
check_walk_duplicates.py) to the 4 remaining production k_cover caches, so all 6
datasets have a duplicate-rate number before the backward-prefix sampler (k_cover_bp,
see ~/.claude/plans/plan-a-fix-for-glimmering-panda.md) is built.

For each dataset's production cache (walk_strategy=k_cover, k=5, num_walks per
E15_final_sota_budgets.csv), hashes every walk's raw token-id sequence
(offsets/flat_input_ids) and reports:
  - overall exact-duplicate rate
  - duplicate rate by walk length bucket (tokens; hops = (tokens-1)/2)
  - max multiplicity of a single repeated sequence

No GPU, no training, writes only outputs/walk_coverage_analysis/PHASE0B_DUP_MEASUREMENT.md.
"""
import os
from collections import Counter

import numpy as np
import torch

# (dataset, cache_path) for the 4 datasets not yet measured this session
# (wiki-elec/wiki-rfa numbers already in hand from the scratchpad checks).
CACHES = [
    ("bitcoin-alpha", "data/bitcoin-alpha/dataset_cache__k_cover_k5_nw5000000_mw80_seed42.pt"),
    ("bitcoin-otc", "data/bitcoin-otc/dataset_cache__k_cover_k5_nw2000000_mw80_seed42.pt"),
    ("epinions", "data/epinions/dataset_cache__k_cover_k5_nw3000000_mw80_seed42.pt"),
    ("slashdot090221", "data/slashdot090221/dataset_cache__k_cover_k5_nw5000000_mw80_seed42.pt"),
]

# Numbers already measured this session (scratchpad check_walk_duplicates.py), kept
# here only so the output doc has all 6 datasets in one table.
PRIOR_RESULTS = {
    "wiki-elec": dict(
        cache="data/wiki-Elec/dataset_cache__k_cover_k5_nw500000_mw80_seed42.pt",
        n_walks=500_000, n_unique=353_718, dup_rate=29.26, max_mult=45,
        by_len={3: 98.1, 5: 32.6, 7: 6.3, 9: 1.5},
    ),
    "wiki-rfa": dict(
        cache="data/wiki-RfA/dataset_cache__k_cover_k5_nw1000000_mw80_seed42.pt",
        n_walks=1_000_000, n_unique=872_024, dup_rate=12.80, max_mult=72,
        by_len={3: 96.9, 5: 36.9, 7: 8.2, 9: 1.5},
    ),
}

OUT_PATH = "outputs/walk_coverage_analysis/PHASE0B_DUP_MEASUREMENT.md"
LEN_BUCKETS = [3, 5, 7, 9, 11, 13, 15]


def measure(cache_path):
    d = torch.load(cache_path, map_location="cpu", weights_only=False)
    enc = d["encoded"]
    offsets = enc["offsets"].numpy()
    flat_ids = enc["flat_input_ids"].numpy()
    n_walks = offsets.shape[0] - 1
    lengths = np.diff(offsets)

    counter = Counter()
    for i in range(n_walks):
        s, e = offsets[i], offsets[i + 1]
        counter[flat_ids[s:e].tobytes()] += 1

    n_unique = len(counter)
    n_dup_walks = n_walks - n_unique
    dup_sizes = [c for c in counter.values() if c > 1]
    max_mult = max(dup_sizes) if dup_sizes else 1

    # Per-walk duplicate membership (walk belongs to a sequence seen >1x)
    is_dup = np.zeros(n_walks, dtype=bool)
    idx = 0
    seen_once = set()
    seen_dup = set()
    for i in range(n_walks):
        s, e = offsets[i], offsets[i + 1]
        key = flat_ids[s:e].tobytes()
        if counter[key] > 1:
            seen_dup.add(key)
    for i in range(n_walks):
        s, e = offsets[i], offsets[i + 1]
        key = flat_ids[s:e].tobytes()
        is_dup[i] = key in seen_dup

    by_len = {}
    for L in LEN_BUCKETS:
        mask = lengths == L
        tot = int(mask.sum())
        if tot == 0:
            continue
        dup = int(is_dup[mask].sum())
        by_len[L] = 100.0 * dup / tot

    return dict(
        cache=cache_path, n_walks=n_walks, n_unique=n_unique,
        dup_rate=100.0 * n_dup_walks / n_walks, max_mult=max_mult, by_len=by_len,
    )


def fmt_by_len(by_len):
    return ", ".join(f"len={L}: {pct:.1f}%" for L, pct in sorted(by_len.items()))


def main():
    results = {}
    for ds, path in CACHES:
        print(f"Measuring {ds} ({path}) ...")
        if not os.path.exists(path):
            print(f"  MISSING: {path}, skipping")
            continue
        results[ds] = measure(path)
        r = results[ds]
        print(f"  n_walks={r['n_walks']:,} unique={r['n_unique']:,} "
              f"dup_rate={r['dup_rate']:.2f}% max_mult={r['max_mult']}")

    lines = [
        "# Phase 0(B) — walk duplication measurement, all 6 datasets",
        "",
        "Exact-duplicate rate in each production k_cover (k=5) cache — every anchor",
        "walk hashed by its raw token-id sequence. wiki-elec/wiki-rfa were measured",
        "ad-hoc earlier this session (scratchpad `check_walk_duplicates.py`); this run",
        "covers the remaining 4 with `scripts/measure_walk_duplication.py`.",
        "",
        "| dataset | cache nw | n_walks | unique | dup rate | max multiplicity |",
        "|---|---|---|---|---|---|",
    ]
    all_ds = list(PRIOR_RESULTS.items()) + [(ds, results[ds]) for ds in results]
    for ds, r in all_ds:
        nw = r.get("n_walks", "?")
        lines.append(
            f"| {ds} | {os.path.basename(r['cache'])} | {nw:,} | "
            f"{r['n_unique']:,} | {r['dup_rate']:.2f}% | {r['max_mult']} |"
        )

    lines += ["", "## Duplicate rate by walk length (tokens; hops = (tokens-1)/2)", "",
              "| dataset | " + " | ".join(f"len={L}" for L in LEN_BUCKETS) + " |",
              "|---|" + "---|" * len(LEN_BUCKETS)]
    for ds, r in all_ds:
        row = [f"{r['by_len'].get(L, float('nan')):.1f}%" if L in r["by_len"] else "-"
               for L in LEN_BUCKETS]
        lines.append(f"| {ds} | " + " | ".join(row) + " |")

    lines += [
        "", "## Notes", "",
        "- Mechanism (confirmed on wiki-elec/wiki-rfa, holds structurally everywhere):",
        "  a length-3 walk `[N_u, E_label, N_v]` occurs when `v` is a dead end (no",
        "  outgoing edges) — the anchor has zero randomness available, so every",
        "  anchor of that edge is byte-identical.",
        "- Not a leakage bug (`stage_dataset.py` masks by edge identity, not",
        "  position/duplication) but is bad training-data hygiene — see",
        "  `~/.claude/plans/plan-a-fix-for-glimmering-panda.md`.",
    ]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
