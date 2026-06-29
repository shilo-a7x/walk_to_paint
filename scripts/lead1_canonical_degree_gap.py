"""
Lead 1 (GNN over-averaging) RE-RUN on the CANONICAL split.

The original lead1_degree_gap.py bucketed the walk model and GINEConv by
target-node degree and compared per-bucket AUC -- but the two models scored
~independent edge samples (~10% overlap), so a "gap" mixed a real degree effect
with the split mismatch. This driver re-derives the same degree-stratified AUC
gap from `predictions_raw_canonical.pkl`, where walk_full / GINEConv / SiGAT all
score the SAME shared (walk-covered) test edges in one raw `(u, v)` id space, and
degree is computed once from the real canonical edge list. So a residual gap is
now a genuine degree effect, not a split artifact.

    python scripts/lead1_canonical_degree_gap.py
"""
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.node_mi_structural_embedding import DATASET_CONFIGS, load_edges_canonical
from scripts.lead4_entropy_heterogeneity import (
    _ds_key, DATASETS, shared_edge_set, _restrict as restrict,
    CANON_PREDICTIONS_DEFAULT as CANON_PREDICTIONS,
)

N_BUCKETS = 4
GNN_MODELS = ["GINEConv", "SiGAT"]
OUT_DIR = os.path.join(ROOT, "outputs", "lead1_degree_gap_canonical")


def degree_dict(edges):
    deg = {}
    for u, v, _ in edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    return deg


def quantile_buckets(scores, n_buckets=N_BUCKETS):
    edges = np.quantile(scores, np.linspace(0, 1, n_buckets + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return np.digitize(scores, edges[1:-1], right=True)


def auc_by_bucket(p, y, node_deg, buckets, n_buckets=N_BUCKETS):
    rows = []
    y, p, node_deg = np.asarray(y), np.asarray(p), np.asarray(node_deg)
    for b in range(n_buckets):
        mask = buckets == b
        if mask.sum() < 5 or len(set(y[mask])) < 2:
            rows.append((b, int(mask.sum()), None, None, None))
            continue
        rows.append((b, int(mask.sum()), int(node_deg[mask].min()),
                     int(node_deg[mask].max()), roc_auc_score(y[mask], p[mask])))
    return rows


def main():
    with open(CANON_PREDICTIONS, "rb") as f:
        preds = pickle.load(f)
    os.makedirs(OUT_DIR, exist_ok=True)

    report = [
        "# Lead 1 (GNN over-averaging) -- canonical, shared-edge degree gap",
        "",
        "Walk model vs GNN AUC stratified by max-endpoint degree (quantile buckets),",
        "all models on the SAME shared (walk-covered) canonical test edges, degree",
        "from the real canonical edge list. A positive gap = walk beats the GNN in",
        "that degree bucket. The over-averaging hypothesis predicts the gap WIDENS",
        "with degree (more neighbors to dilute/cancel).",
        "",
    ]

    for ds in DATASETS:
        models = preds.get(ds, {})
        keep = shared_edge_set(models)
        if not keep or models.get("walk_full") is None:
            continue
        edges = load_edges_canonical(DATASET_CONFIGS[_ds_key(ds)]["ds_name"])
        deg = degree_dict(edges)

        wk = restrict(models["walk_full"], keep)
        wdeg = np.array([max(deg.get(u, 0), deg.get(v, 0)) for u, v in zip(wk["u"], wk["v"])])
        buckets = quantile_buckets(wdeg)  # shared edges -> identical buckets for all models
        wk_rows = auc_by_bucket(wk["p"], wk["y"], wdeg, buckets)

        report.append(f"### {ds}  (n_shared={len(keep)})")
        report.append("")
        for gnn in GNN_MODELS:
            if models.get(gnn) is None:
                continue
            gn = restrict(models[gnn], keep)
            # align gnn rows to the same edge order as walk so buckets match
            gidx = {(u, v): i for i, (u, v) in enumerate(zip(gn["u"], gn["v"]))}
            order = [gidx[(u, v)] for u, v in zip(wk["u"], wk["v"])]
            gp = np.array(gn["p"])[order]
            gy = np.array(gn["y"])[order]
            gn_rows = auc_by_bucket(gp, gy, wdeg, buckets)

            report.append(f"**walk_full vs {gnn}**")
            report.append("")
            report.append("| bucket | n | deg range | walk AUC | "
                          f"{gnn} AUC | gap (walk-{gnn}) |")
            report.append("|---|---|---|---|---|---|")
            for (b, n, lo, hi, wa), (_, _, _, _, ga) in zip(wk_rows, gn_rows):
                wa_s = f"{wa:.4f}" if wa is not None else "n/a"
                ga_s = f"{ga:.4f}" if ga is not None else "n/a"
                gap = f"{wa - ga:+.4f}" if (wa is not None and ga is not None) else "n/a"
                rng = f"[{lo},{hi}]" if lo is not None else "-"
                report.append(f"| {b} | {n} | {rng} | {wa_s} | {ga_s} | {gap} |")
            report.append("")
            # trend: gap in lowest vs highest degree bucket
            lo_gap = (wk_rows[0][4] - gn_rows[0][4]) if (wk_rows[0][4] and gn_rows[0][4]) else None
            hi_gap = (wk_rows[-1][4] - gn_rows[-1][4]) if (wk_rows[-1][4] and gn_rows[-1][4]) else None
            if lo_gap is not None and hi_gap is not None:
                report.append(f"_low-degree gap {lo_gap:+.4f} -> high-degree gap {hi_gap:+.4f} "
                              f"(widening = {'YES' if hi_gap > lo_gap else 'no'})_")
                report.append("")
            print(f"[{ds}] walk vs {gnn}: low-deg gap "
                  f"{lo_gap if lo_gap is None else round(lo_gap,4)} -> high-deg gap "
                  f"{hi_gap if hi_gap is None else round(hi_gap,4)}")

    path = os.path.join(OUT_DIR, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"\n✓ Report written to {path}")


if __name__ == "__main__":
    main()
