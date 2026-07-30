"""Phase 2a — k_cover_bp generation-only benchmark at CURRENT production budgets.

For each of the 6 datasets, at the exact num_walks currently shipped in
configs/<dataset>.yaml (full-attention E15 budget), times k_cover_bp generation vs.
the plain uniform sampler at the same budget, and checks the 3 Phase 2a gates from
~/.claude/plans/plan-a-fix-for-glimmering-panda.md:
  1. gen wall-clock within ~1.5-2.5x uniform (the established acceptable envelope)
  2. 100% edge + node coverage, full k=5 floor (no regression vs k_cover_walks_fast)
  3. near-zero duplication: raw-duplicate-% ~= capped-% (i.e. every duplicate has a
     structural reason — dup_after_retries should be ~0%, not just "small")

No training. Writes outputs/walk_coverage_analysis/PHASE2A_BP_BENCHMARK.md
incrementally (one dataset's row appended as soon as it finishes) so a partial run
is still useful if interrupted.
"""
import json
import os
import sys
import time

sys.path.insert(0, ".")

import numpy as np

from scripts.balance_theory_paths import load_edges_canonical
from src.data.coverage_aware_sampler import k_cover_walks_bp
from src.data.walk_sampler import sample_random_walks

# (dataset, num_walks) = current production full-attention budget, from
# configs/<dataset>.yaml / outputs/walk_coverage_analysis/E15_final_sota_budgets.csv
DATASETS = [
    ("bitcoin-alpha", 5_000_000),
    ("bitcoin-otc", 2_000_000),
    ("epinions", 3_000_000),
    ("wiki-elec", 500_000),
    ("wiki-rfa", 1_000_000),
    ("slashdot090221", 5_000_000),
]
MAX_WALK_LENGTH = 80
SEED = 42
K = 5
OUT_MD = "outputs/walk_coverage_analysis/PHASE2A_BP_BENCHMARK.md"
OUT_JSON = "outputs/walk_coverage_analysis/PHASE2A_BP_BENCHMARK.json"


def node_coverage(walks, n_nodes_total):
    covered = set()
    for w in walks:
        for tok in w:
            if tok[0] == "N":
                covered.add(tok)
    return len(covered), n_nodes_total


def bench_one(ds_name, num_walks):
    print(f"\n=== {ds_name} (nw={num_walks:,}) ===", flush=True)
    edges = load_edges_canonical(ds_name)
    all_nodes = set()
    for u, v, _l in edges:
        all_nodes.add(u)
        all_nodes.add(v)
    n_nodes_total = len(all_nodes)
    n_edges_total = len({(u, v, l) for u, v, l in edges})
    print(f"  |E|(unique)={n_edges_total:,} |V|={n_nodes_total:,}", flush=True)

    t0 = time.time()
    uniform_walks = sample_random_walks(
        edges, num_walks=num_walks, max_walk_length=MAX_WALK_LENGTH,
        num_workers=8, seed=SEED,
    )
    t_uniform = time.time() - t0
    print(f"  uniform gen: {t_uniform:.1f}s", flush=True)
    del uniform_walks

    telemetry = {}
    t0 = time.time()
    bp_walks = k_cover_walks_bp(
        edges, num_walks=num_walks, max_walk_length=MAX_WALK_LENGTH, seed=SEED,
        num_workers=8, k=K, telemetry_out=telemetry,
    )
    t_bp = time.time() - t0
    print(f"  k_cover_bp gen: {t_bp:.1f}s (ratio {t_bp / t_uniform:.2f}x uniform)", flush=True)

    n_cov_nodes, _ = node_coverage(bp_walks, n_nodes_total)
    node_cov = n_cov_nodes / n_nodes_total

    per_edge = telemetry["per_edge"]
    m = telemetry["summary"]["m"]
    raw_visits = np.array([v["raw_visits"] for v in per_edge.values()])
    edge_cov = float((raw_visits > 0).mean())
    pct_ge_k = float((raw_visits >= K).mean())
    min_visit = int(raw_visits.min())

    capped_frac = telemetry["summary"]["capped_frac"]
    dup_after_retries_frac = telemetry["summary"]["dup_after_retries_frac"]
    n_capped = telemetry["summary"]["n_capped"]
    n_dup_after_retries = telemetry["summary"]["n_dup_after_retries"]

    result = dict(
        dataset=ds_name, num_walks=num_walks, n_edges=n_edges_total, n_nodes=n_nodes_total,
        t_uniform=t_uniform, t_bp=t_bp, ratio=t_bp / t_uniform,
        edge_coverage=edge_cov, node_coverage=node_cov, pct_ge_k=pct_ge_k, min_visit=min_visit,
        capped_frac=capped_frac, n_capped=n_capped,
        dup_after_retries_frac=dup_after_retries_frac, n_dup_after_retries=n_dup_after_retries,
        m=m,
    )
    print(f"  edge_cov={edge_cov:.4f} node_cov={node_cov:.4f} pct_ge_k={pct_ge_k:.4f} "
          f"min_visit={min_visit} capped={capped_frac:.4%} dup_after_retries={dup_after_retries_frac:.4%}",
          flush=True)
    del bp_walks, telemetry
    return result


def write_md(results):
    lines = [
        "# Phase 2a — k_cover_bp generation-only benchmark at production budgets",
        "",
        "Gates (plan-a-fix-for-glimmering-panda.md Phase 2a): (1) gen time within ~1.5-",
        "2.5x uniform-sampler envelope, (2) 100% edge+node coverage / full k=5 floor,",
        "(3) near-zero duplication (dup_after_retries ~= 0%, raw-dup% ~= capped-%).",
        "",
        "| dataset | nw | \\|E\\| | uniform (s) | k_cover_bp (s) | ratio | edge cov | node cov | %≥k | min visit | capped % | dup_after_retries % |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['dataset']} | {r['num_walks']:,} | {r['n_edges']:,} | "
            f"{r['t_uniform']:.1f} | {r['t_bp']:.1f} | {r['ratio']:.2f}x | "
            f"{r['edge_coverage']:.4f} | {r['node_coverage']:.4f} | {r['pct_ge_k']:.4f} | "
            f"{r['min_visit']} | {r['capped_frac']:.4%} | {r['dup_after_retries_frac']:.4%} |"
        )
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    results = []
    for ds_name, nw in DATASETS:
        r = bench_one(ds_name, nw)
        results.append(r)
        write_md(results)
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_MD} and {OUT_JSON}")


if __name__ == "__main__":
    main()
