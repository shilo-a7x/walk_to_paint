"""Phase 2b — fresh coverage-vs-budget curve for k_cover_bp, per dataset.

Backward-prefix anchors touch more edges per walk as side-effects (prefix edges +
forced edge + suffix edges, vs. forced-edge + suffix only for the old sampler), so
the old k_cover_walks_fast coverage-floor-per-budget table
(outputs/walk_coverage_analysis/PHASE2_SAMPLER_BENCHMARK.md) is not assumed to
transfer. This measures it fresh, generation-only (no training), at a multiplier
grid of num_walks relative to |E| for each dataset — becomes the actual floor
Phase 4's training-budget grid is built from.

Usage: .venv/bin/python scripts/measure_kcover_bp_coverage_curve.py [dataset ...]
  (no args -> all 6 datasets)

Writes outputs/walk_coverage_analysis/PHASE2B_BP_COVERAGE_CURVE.md incrementally.
"""
import json
import os
import sys
import time

sys.path.insert(0, ".")

import numpy as np

from scripts.balance_theory_paths import load_edges_canonical
from src.data.coverage_aware_sampler import k_cover_walks_bp

ALL_DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions",
                "wiki-elec", "wiki-rfa", "slashdot090221"]
MULTIPLIERS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
MAX_WALK_LENGTH = 80
SEED = 42
K = 5
OUT_MD = "outputs/walk_coverage_analysis/PHASE2B_BP_COVERAGE_CURVE.md"
OUT_JSON = "outputs/walk_coverage_analysis/PHASE2B_BP_COVERAGE_CURVE.json"


def node_coverage(walks, n_nodes_total):
    covered = set()
    for w in walks:
        for tok in w:
            if tok[0] == "N":
                covered.add(tok)
    return len(covered) / n_nodes_total


def measure_point(edges, n_edges_total, n_nodes_total, nw):
    telemetry = {}
    t0 = time.time()
    walks = k_cover_walks_bp(
        edges, num_walks=nw, max_walk_length=MAX_WALK_LENGTH, seed=SEED,
        num_workers=8, k=K, telemetry_out=telemetry,
    )
    t = time.time() - t0
    node_cov = node_coverage(walks, n_nodes_total)
    per_edge = telemetry["per_edge"]
    raw_visits = np.array([v["raw_visits"] for v in per_edge.values()])
    edge_cov = float((raw_visits > 0).mean())
    pct_ge_k = float((raw_visits >= K).mean())
    min_visit = int(raw_visits.min())
    del walks
    return dict(
        nw=nw, gen_s=t, edge_coverage=edge_cov, node_coverage=node_cov,
        pct_ge_k=pct_ge_k, min_visit=min_visit,
        capped_frac=telemetry["summary"]["capped_frac"],
        dup_after_retries_frac=telemetry["summary"]["dup_after_retries_frac"],
    )


def run_dataset(ds_name):
    print(f"\n=== {ds_name} ===", flush=True)
    edges = load_edges_canonical(ds_name)
    all_nodes = {u for u, v, _l in edges} | {v for u, v, _l in edges}
    n_nodes_total = len(all_nodes)
    n_edges_total = len({(u, v, l) for u, v, l in edges})
    print(f"  |E|(unique)={n_edges_total:,} |V|={n_nodes_total:,}", flush=True)

    points = []
    for mult in MULTIPLIERS:
        nw = int(round(mult * n_edges_total))
        print(f"  nw={nw:,} ({mult}x|E|) ...", flush=True)
        p = measure_point(edges, n_edges_total, n_nodes_total, nw)
        p["multiplier"] = mult
        points.append(p)
        print(f"    gen={p['gen_s']:.1f}s edge_cov={p['edge_coverage']:.4f} "
              f"node_cov={p['node_coverage']:.4f} pct_ge_k={p['pct_ge_k']:.4f} "
              f"min_visit={p['min_visit']}", flush=True)
        # Early stop: once both coverages hit ~1.0 and the k-floor is fully met,
        # higher multipliers only add cost, not new coverage information — but
        # still record one more point past the floor for context, then stop.
        if p["edge_coverage"] >= 0.999 and p["node_coverage"] >= 0.999 and p["pct_ge_k"] >= 0.999:
            print("    (coverage + k-floor fully met; recording one more point then stopping)",
                  flush=True)
            if mult != MULTIPLIERS[-1]:
                next_mult = MULTIPLIERS[MULTIPLIERS.index(mult) + 1]
                nw2 = int(round(next_mult * n_edges_total))
                p2 = measure_point(edges, n_edges_total, n_nodes_total, nw2)
                p2["multiplier"] = next_mult
                points.append(p2)
                print(f"  nw={nw2:,} ({next_mult}x|E|) gen={p2['gen_s']:.1f}s "
                      f"edge_cov={p2['edge_coverage']:.4f} node_cov={p2['node_coverage']:.4f}",
                      flush=True)
            break
    return dict(dataset=ds_name, n_edges=n_edges_total, n_nodes=n_nodes_total, points=points)


def write_md(all_results):
    lines = [
        "# Phase 2b — fresh k_cover_bp coverage-vs-budget curve, per dataset",
        "",
        "Generation-only (no training). Grid = multiplier x |E|, per dataset.",
        "This is the floor Phase 4's training-budget grid is built from — NOT copied",
        "from the old k_cover_walks_fast benchmark or the old sampler's absolute",
        "walk-count budgets.",
        "",
    ]
    for r in all_results:
        lines.append(f"## {r['dataset']} (|E|={r['n_edges']:,}, |V|={r['n_nodes']:,})")
        lines.append("")
        lines.append("| multiplier | nw | gen (s) | edge cov | node cov | %≥k | min visit | capped % | dup_after_retries % |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for p in r["points"]:
            lines.append(
                f"| {p['multiplier']}x | {p['nw']:,} | {p['gen_s']:.1f} | "
                f"{p['edge_coverage']:.4f} | {p['node_coverage']:.4f} | {p['pct_ge_k']:.4f} | "
                f"{p['min_visit']} | {p['capped_frac']:.4%} | {p['dup_after_retries_frac']:.4%} |"
            )
        lines.append("")
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ALL_DATASETS
    all_results = []
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            all_results = json.load(f)
        done = {r["dataset"] for r in all_results}
        datasets = [d for d in datasets if d not in done]
        print(f"Resuming: already have {sorted(done)}, running {datasets}")

    for ds_name in datasets:
        r = run_dataset(ds_name)
        all_results.append(r)
        write_md(all_results)
        with open(OUT_JSON, "w") as f:
            json.dump(all_results, f, indent=2)
    print(f"\nWrote {OUT_MD} and {OUT_JSON}")


if __name__ == "__main__":
    main()
