"""Phase 1.6 sanity checks for k_cover_bp (backward-prefix k_cover sampler).

Synthetic-graph, no-training checks per
~/.claude/plans/plan-a-fix-for-glimmering-panda.md Phase 1.6:
  1. Dead-end edge: attempt #1 matches legacy plain anchor; attempts #2-5 diversify
     (>=3 distinct sequences) via the backward prefix.
  2. Isolated dyad: fully "capped" (zero variance possible in either direction) —
     all attempts identical, never flagged dup_after_retries (expected, not a bug).
  3. Chain-exhaustion: not capped, but only one backward prefix option exists once
     the single predecessor chain is exhausted — attempt #1 and #2 are genuinely new,
     attempts #3-5 must collide and get flagged dup_after_retries=True.
  4. neg_traversal_walks regression: refactoring it to call the new shared
     _build_rev_adj helper must not change its output at all (same seed -> same
     walks), checked against the pre-refactor version read from `git show HEAD`.

Exits nonzero on any failed assertion. No GPU, no training.
"""
import importlib.util
import subprocess
import sys

import numpy as np

sys.path.insert(0, ".")

from src.data.coverage_aware_sampler import (  # noqa: E402
    _build_adj_with_eids,
    _build_edge_index,
    _build_rev_adj,
    _build_rev_adj_with_eids,
    _kcover_anchor_chunk_bp,
    edge_cover_walks,
    k_cover_walks_bp,
)

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def run_attempts(edges, u, v, label, n_attempts, max_walk_length=20,
                  prefix_step=2, max_dedup_retries=3, seed=42):
    """Directly drive _kcover_anchor_chunk_bp for a controlled sequence of attempts
    on one target edge, bypassing the full multi-pass driver's cross-edge side-effect
    dynamics so the test can assert on exact attempt-by-attempt behavior."""
    edge_index = _build_edge_index(edges)
    nodes, nbrs, lbls, eids = _build_adj_with_eids(edges, edge_index)
    rev_nbrs, rev_lbls = _build_rev_adj(edges)
    rev_eids = _build_rev_adj_with_eids(edges, edge_index, rev_nbrs, rev_lbls)
    m = len(edge_index)
    eid = edge_index[(u, v, label)]
    in_deg_u = len(rev_nbrs.get(u, ()))
    out_deg_v = len(nbrs.get(v, ()))
    capped = (in_deg_u == 0) and (out_deg_v == 0)

    seen_hashes = []
    results = []
    for attempt in range(1, n_attempts + 1):
        anchors = [(u, v, label, eid, attempt, capped, list(seen_hashes))]
        _, walks, _local_vc, tel = _kcover_anchor_chunk_bp(
            nbrs, lbls, eids, rev_nbrs, rev_lbls, rev_eids, m,
            anchors, max_walk_length, prefix_step, max_dedup_retries,
            seed, attempt,  # vary the RNG stream per attempt, like separate passes
        )
        eid_out, h, cap_out, dup = tel[0]
        seen_hashes.append(h)
        results.append(dict(walk=walks[0], hash=h, capped=cap_out, dup_after_retries=dup))
    return results, capped


def test_dead_end():
    print("\n--- Test 1: dead-end edge (out_deg(v)=0, in_deg(u)>0) ---")
    # u=1 -> v=0 (v is a dead end); 4 distinct predecessors of u so the backward
    # walk has real branching to diversify into.
    edges = [(1, 0, 1), (2, 1, 0), (3, 1, 1), (4, 1, 0), (5, 1, 1)]
    results, capped = run_attempts(edges, u=1, v=0, label=1, n_attempts=5)

    check("not capped", capped is False)
    check("attempt #1 matches legacy plain anchor",
          results[0]["walk"] == ["N_1", "E_1", "N_0"], detail=str(results[0]["walk"]))
    distinct = len({r["hash"] for r in results})
    check("attempts diversify: >=4 distinct sequences total (spec: >=3 among #2-5)",
          distinct >= 4, detail=f"distinct={distinct}/5")
    for i, r in enumerate(results[1:], start=2):
        print(f"    attempt {i}: walk={r['walk']} dup_after_retries={r['dup_after_retries']}")


def test_isolated_dyad():
    print("\n--- Test 2: isolated dyad (in_deg(u)=0 AND out_deg(v)=0) ---")
    edges = [(1, 0, 1)]
    results, capped = run_attempts(edges, u=1, v=0, label=1, n_attempts=5)

    check("capped == True", capped is True)
    check("all 5 attempts identical (zero variance possible)",
          all(r["walk"] == ["N_1", "E_1", "N_0"] for r in results))
    check("capped edges never flagged dup_after_retries (expected, not a failure)",
          all(r["dup_after_retries"] is False for r in results))


def test_chain_exhaustion():
    print("\n--- Test 3: chain exhaustion (single predecessor, terminates fast, v dead-end) ---")
    # u=1's only predecessor is 2 (edge 2->1); 2 itself has no predecessors, so the
    # backward walk can only ever produce ONE possible 1-step prefix. v=0 is a dead
    # end (out_deg=0), so in_deg(u)=1 != 0 -> not capped, but genuinely only 2
    # distinct walks exist (no-prefix, and the single available prefix).
    edges = [(1, 0, 1), (2, 1, 0)]
    results, capped = run_attempts(edges, u=1, v=0, label=1, n_attempts=5)

    check("not capped (in_deg(u)=1)", capped is False)
    check("attempt #1: no prefix", results[0]["walk"] == ["N_1", "E_1", "N_0"])
    check("attempt #1 not flagged dup", results[0]["dup_after_retries"] is False)
    check("attempt #2: the one available prefix, genuinely new",
          results[1]["walk"] == ["N_2", "E_0", "N_1", "E_1", "N_0"],
          detail=str(results[1]["walk"]))
    check("attempt #2 not flagged dup (first use of the only prefix option)",
          results[1]["dup_after_retries"] is False)
    for i, r in enumerate(results[2:], start=3):
        check(f"attempt #{i}: forced to repeat attempt #2's walk (only option)",
              r["walk"] == results[1]["walk"])
        check(f"attempt #{i}: flagged dup_after_retries=True (honest residual failure)",
              r["dup_after_retries"] is True)


def test_walk_length_never_exceeds_cap():
    print("\n--- Test 3b: prefix+forced-edge+suffix never exceeds max_walk_length hops ---")
    # A chain graph long enough on BOTH sides of the target edge that a real backward
    # prefix AND a long forward suffix can co-occur: 0->1->2->...->9->10 (target edge
    # 10->11) ->12->...->20. Regression test for the bug caught by the actual k=3/5/7
    # Phase 3 training runs: RuntimeError "size of tensor a (163) must match ... (161)"
    # — a fixed-length forward suffix loop that ignored how many hops the backward
    # prefix had already spent.
    chain_before = [(i, i + 1, i % 2) for i in range(10)]       # 0->1->...->9->10
    chain_after = [(10 + i, 11 + i, i % 2) for i in range(10)]  # 10->11->...->19->20
    edges = chain_before + chain_after
    max_walk_length = 20
    results, capped = run_attempts(edges, u=10, v=11, label=0, n_attempts=5,
                                    max_walk_length=max_walk_length)
    max_tokens = 2 * max_walk_length + 1
    lengths = [len(r["walk"]) for r in results]
    check(f"all 5 attempts fit within {max_tokens} tokens (max_walk_length={max_walk_length} hops)",
          all(L <= max_tokens for L in lengths), detail=f"lengths={lengths}")
    check("later attempts (with a real prefix) are actually longer than attempt #1",
          lengths[-1] > lengths[0], detail=f"attempt1={lengths[0]} attempt5={lengths[-1]}")


def test_kcover_bp_smoke():
    print("\n--- Test 4: k_cover_walks_bp end-to-end smoke test (small graph) ---")
    # A slightly bigger synthetic graph combining all 3 topologies above, run through
    # the real multi-pass driver — just confirms it runs, produces the right walk
    # count, hits full coverage, and telemetry is internally consistent.
    edges = (
        [(1, 0, 1), (2, 1, 0), (3, 1, 1), (4, 1, 0), (5, 1, 1)]  # dead-end + branching
        + [(10, 11, 1)]  # isolated dyad
        + [(21, 20, 1), (22, 21, 0)]  # chain exhaustion
    )
    # num_walks=32 is exactly the anchor-only budget needed to bring all 8 edges to
    # k=5 (verified by direct search) -- deliberately NOT larger: this graph is tiny
    # (11 nodes, minimal branching) and its distinct-walk space is exhausted right at
    # the anchor set, so any fill-phase request beyond it now hard-fails per the
    # 2026-07-17 hardening (see test_fill_phase_exhaustion) rather than padding with
    # duplicates -- correct behavior, but this test's purpose is telemetry
    # correctness, not exercising the fill phase, so it stays within the graph's
    # real distinct-walk capacity.
    telemetry = {}
    walks = k_cover_walks_bp(
        edges, num_walks=32, max_walk_length=20, seed=42, num_workers=1,
        k=5, telemetry_out=telemetry,
    )
    check("produced the requested walk count", len(walks) == 32, detail=str(len(walks)))
    m = telemetry["summary"]["m"]
    check("edge_index sees all 8 unique edges", m == 8, detail=f"m={m}")

    per_edge = telemetry["per_edge"]
    check("telemetry keyed by (u,v,label) tuple, not a positional index",
          all(isinstance(k_, tuple) and len(k_) == 3 for k_ in per_edge))
    check("every edge reaches k=5 raw visits",
          all(v["raw_visits"] >= 5 for v in per_edge.values()),
          detail=str({k_: v["raw_visits"] for k_, v in per_edge.items()}))
    check("isolated dyad (10,11,1) is capped", per_edge[(10, 11, 1)]["capped"] is True)
    check("dead-end edge (1,0,1) is not capped", per_edge[(1, 0, 1)]["capped"] is False)
    # (1,0,1) is expected to need only 1 DIRECT attempt here: its 4 predecessor
    # edges' own anchors all forward-continue through node 1's only outgoing edge
    # (1->0), so side-effect credits alone push it well past k=5 (raw_visits >> 5)
    # after pass 1 — it never needs a 2nd direct anchor, which is correct/cheaper
    # behavior, not a diversification failure. The actual multi-attempt
    # diversification mechanism is unit-tested in isolation by test_dead_end()
    # above, where side-effect rescue is deliberately bypassed.
    de = per_edge[(1, 0, 1)]
    check("dead-end edge (1,0,1): side-effect-rescued (raw_visits >> k), "
          "so only needed 1 direct attempt — consistent, not a bug",
          de["raw_visits"] > 5 and de["direct_attempts"] == 1 and de["distinct_hashes"] == 1,
          detail=str(de))


def test_fill_phase_dedup():
    print("\n--- Test 6: fill-phase dedup (the bug found via the real k=1/3/5/7 caches) ---")
    # A moderately-branching random directed graph (~80 nodes, out-degree ~3, so
    # walks up to 8 hops have on the order of 3^8 ~ 6500+ possible distinct paths —
    # ample headroom relative to the 3000-walk budget) PLUS a cluster of dead-end
    # pendants off a shared hub (node 0), the exact shape that broke the old plain-
    # uniform fill: resampling the same low-branching node many times at
    # nw >> |V| produced identical walks every time without a backward prefix.
    rng = np.random.default_rng(0)
    n_nodes = 80
    edges = []
    for u in range(1, n_nodes):
        for _ in range(3):
            v = int(rng.integers(0, n_nodes))
            if v != u:
                edges.append((u, v, int(rng.integers(0, 2))))
    edges = list(set(edges))  # dedup any accidental repeats from the random draw
    # Dead-end pendant cluster off hub node 0 (mirrors the earlier hub-and-spoke case)
    for i in range(1, 15):
        edges.append((0, 1000 + i, i % 2))       # hub -> pendant
        edges.append((1000 + i, 2000 + i, i % 2))  # pendant -> dead end

    telemetry = {}
    walks = k_cover_walks_bp(
        edges, num_walks=3000, max_walk_length=10, seed=42, num_workers=4,
        k=5, telemetry_out=telemetry,
    )
    check("produced the requested walk count", len(walks) == 3000, detail=str(len(walks)))
    corpus = telemetry["corpus"]
    print(f"    corpus: n_walks={corpus['n_walks_total']} "
          f"n_distinct={corpus['n_distinct_total']} dup_rate={corpus['dup_rate_total']:.4%}")
    check("fill-phase dedup drives corpus-wide duplication to near-zero "
          "(graph has ample distinct-walk headroom relative to the budget)",
          corpus["dup_rate_total"] < 0.02, detail=f"dup_rate={corpus['dup_rate_total']:.4%}")
    fill = telemetry["fill"]
    print(f"    fill: requested={fill['n_requested']} accepted={fill['n_accepted']} "
          f"candidates_generated={fill['n_candidates_generated']} "
          f"rounds={fill['n_rounds_used']} exhausted={fill['exhausted']}")
    check("fill phase satisfied its full request without exhausting",
          fill["n_accepted"] == fill["n_requested"] and not fill["exhausted"])


def test_fill_phase_exhaustion():
    print("\n--- Test 7: fill-phase exhaustion HARD-FAILS, never silently duplicates "
          "(2026-07-17 hardening) ---")
    # A TINY graph (3 edges) asked for far more walks than could possibly be
    # distinct at max_walk_length=5 -> fill must raise, not silently pad with
    # duplicates and claim success. This closes the one remaining gap in the
    # "N distinct walks, no matter what" guarantee (the old behavior was a
    # graceful degrade with an honest telemetry flag; per explicit user
    # direction 2026-07-17, "no matter what" now means hard-fail, not degrade).
    edges = [(0, 1, 0), (1, 2, 1), (2, 3, 0)]
    telemetry = {}
    raised = False
    try:
        k_cover_walks_bp(
            edges, num_walks=500, max_walk_length=5, seed=42, num_workers=2,
            k=5, telemetry_out=telemetry,
        )
    except RuntimeError as e:
        raised = True
        msg = str(e)
    check("raises RuntimeError instead of silently padding with duplicates",
          raised, detail=(msg if raised else "no exception raised"))
    if raised:
        check("error message names the shortfall/diagnostics",
              "distinct walks" in msg and "candidates generated" in msg, detail=msg)


def test_edge_cover_anchor_distinctness():
    print("\n--- Test 8: edge_cover anchor phase is provably globally distinct ---")
    # Moderately-branching random graph (same shape as test_fill_phase_dedup) --
    # every edge's anchor walk should be pairwise-distinct by construction alone
    # (each starts with that edge's unique (u, label, v) triple), independent of
    # any RNG luck.
    rng = np.random.default_rng(1)
    n_nodes = 60
    edges = []
    for u in range(1, n_nodes):
        for _ in range(3):
            v = int(rng.integers(0, n_nodes))
            if v != u:
                edges.append((u, v, int(rng.integers(0, 2))))
    edges = list(set(edges))
    m = len(edges)

    telemetry = {}
    walks = edge_cover_walks(
        edges, num_walks=m, max_walk_length=10, seed=7, num_workers=4,
        telemetry_out=telemetry,
    )
    check("produced exactly one walk per edge (num_walks == |E|, no fill needed)",
          len(walks) == m, detail=f"len(walks)={len(walks)} m={m}")
    corpus = telemetry["corpus"]
    check("anchor-only corpus has ZERO duplication (provable, not just measured)",
          corpus["dup_rate_total"] == 0.0, detail=str(corpus))


def test_edge_cover_with_fill():
    print("\n--- Test 9: edge_cover anchors + dedup fill together, budget > |E| ---")
    rng = np.random.default_rng(2)
    n_nodes = 80
    edges = []
    for u in range(1, n_nodes):
        for _ in range(3):
            v = int(rng.integers(0, n_nodes))
            if v != u:
                edges.append((u, v, int(rng.integers(0, 2))))
    edges = list(set(edges))

    telemetry = {}
    walks = edge_cover_walks(
        edges, num_walks=3000, max_walk_length=10, seed=42, num_workers=4,
        telemetry_out=telemetry,
    )
    check("produced the requested walk count", len(walks) == 3000, detail=str(len(walks)))
    corpus = telemetry["corpus"]
    print(f"    corpus: n_walks={corpus['n_walks_total']} "
          f"n_distinct={corpus['n_distinct_total']} dup_rate={corpus['dup_rate_total']:.4%}")
    check("edge_cover (anchor + dedup fill) has zero corpus-wide duplication",
          corpus["dup_rate_total"] == 0.0, detail=f"dup_rate={corpus['dup_rate_total']:.4%}")


def test_neg_traversal_regression():
    print("\n--- Test 5: neg_traversal_walks regression (refactor must not change output) ---")
    old_src = subprocess.run(
        ["git", "show", "HEAD:src/data/coverage_aware_sampler.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    old_path = "/tmp/_old_coverage_aware_sampler_regression_check.py"
    with open(old_path, "w") as f:
        f.write(old_src)
    spec = importlib.util.spec_from_file_location("_old_cas", old_path)
    old_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_mod)

    from src.data import coverage_aware_sampler as new_mod

    edges = [
        (1, 2, 0), (2, 3, 1), (3, 4, 0), (4, 1, 1), (1, 3, 0),
        (5, 1, 1), (2, 5, 0), (3, 5, 1), (5, 2, 0), (4, 5, 1),
    ]
    train_set = {(1, 2, 0), (3, 4, 0), (5, 1, 1)}
    mask_set = {(2, 3, 1), (4, 1, 1)}

    old_walks = old_mod.neg_traversal_walks(
        edges, num_walks=30, max_walk_length=12, seed=7, num_workers=1,
        train_set=train_set, mask_set=mask_set,
    )
    new_walks = new_mod.neg_traversal_walks(
        edges, num_walks=30, max_walk_length=12, seed=7, num_workers=1,
        train_set=train_set, mask_set=mask_set,
    )
    check("neg_traversal_walks output unchanged by the _build_rev_adj refactor",
          old_walks == new_walks,
          detail="" if old_walks == new_walks else "MISMATCH — refactor changed behavior")


if __name__ == "__main__":
    test_dead_end()
    test_isolated_dyad()
    test_chain_exhaustion()
    test_walk_length_never_exceeds_cap()
    test_kcover_bp_smoke()
    test_fill_phase_dedup()
    test_fill_phase_exhaustion()
    test_edge_cover_anchor_distinctness()
    test_edge_cover_with_fill()
    test_neg_traversal_regression()

    print(f"\n{'='*60}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    else:
        print("All sanity checks passed.")
