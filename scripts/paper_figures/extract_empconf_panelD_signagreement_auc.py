"""Extract step for Empirical Confirmation Panel D -- 4-way sign-agreement AUC bars.

For target edge e=(u,v) with prediction (y_e, p_e): look at every OTHER real edge
from the full canonical edge list (train+val+test -- ground truth topology, not a
model prediction, so no leakage) incident to v that points INTO v (in-neighbors of
e, relative to e's own direction u->v) or incident to u that points OUT of u
(out-neighbors). Classify each pair (neighbor, e) as same/diff by sign agreement.
Confirmed with the user: one (y_e, p_e) row is contributed per qualifying neighbor
pair -- a target edge can land in multiple of the 4 buckets and be counted multiple
times within one bucket. 4 buckets: in_same, in_diff, out_same, out_diff.

Models: GINEConv, SiGAT (raw -- see CLAUDE.md's Baselines note), reusing the same
shared-edge canonical predictions as Panel C/E
(outputs/lead4_entropy_heterogeneity/predictions_raw_canonical.pkl). AUC computed
per bucket, per model, POOLED across all 6 datasets (matches Panel E's scope).
"""
import os
import pickle
import sys

from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.utils.config import load_config
from src.data.prepare_data import get_edge_list

PRED_PKL = "outputs/lead4_entropy_heterogeneity/predictions_raw_canonical.pkl"
OUT_CSV = "aaai2027/figure_data/empconf_panelD_signagreement_auc.csv"
DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
MODELS = ["GINEConv", "SiGAT"]
BUCKETS = ["in_same", "in_diff", "out_same", "out_diff"]

_cfg_cache = {}


def load_cfg(ds):
    if ds not in _cfg_cache:
        _cfg_cache[ds] = load_config(overrides=[f"dataset.name={ds}"])
    return _cfg_cache[ds]


def build_adjacency(edges):
    """in_edges_of[v] = list of (x, sign) for real edges (x, v).
    out_edges_of[u] = list of (y, sign) for real edges (u, y)."""
    in_edges_of, out_edges_of = {}, {}
    for u, v, s in edges:
        out_edges_of.setdefault(u, []).append((v, s))
        in_edges_of.setdefault(v, []).append((u, s))
    return in_edges_of, out_edges_of


def shared_edge_set(models):
    sets = [set(zip(m["u"], m["v"])) for m in models.values() if m is not None and len(m["u"])]
    if not sets:
        return set()
    shared = sets[0]
    for s in sets[1:]:
        shared &= s
    return shared


def restrict(uvyp, keep):
    u, v, y, p = uvyp["u"], uvyp["v"], uvyp["y"], uvyp["p"]
    idx = [i for i, (a, b) in enumerate(zip(u, v)) if (a, b) in keep]
    return [(u[i], v[i], y[i], p[i]) for i in idx]


def main():
    with open(PRED_PKL, "rb") as f:
        preds = pickle.load(f)

    # bucket[model][bucket_name] = (y_list, p_list)
    pooled = {m: {b: ([], []) for b in BUCKETS} for m in MODELS}
    n_edges_total = {m: 0 for m in MODELS}

    for ds in DATASETS:
        edges = get_edge_list(load_cfg(ds))
        in_edges_of, out_edges_of = build_adjacency(edges)

        models = preds.get(ds, {})
        keep = shared_edge_set({m: models.get(m) for m in MODELS})
        print(f"{ds}: {len(keep)} shared target edges, {len(edges)} total real edges")

        for model in MODELS:
            if models.get(model) is None:
                continue
            rows = restrict(models[model], keep)
            n_edges_total[model] += len(rows)
            for u, v, y_e, p_e in rows:
                target_pos = (y_e == 1)

                for x, s in in_edges_of.get(v, []):
                    if x == u:
                        continue
                    bucket = "in_same" if (s > 0) == target_pos else "in_diff"
                    pooled[model][bucket][0].append(y_e)
                    pooled[model][bucket][1].append(p_e)

                for w, s in out_edges_of.get(u, []):
                    if w == v:
                        continue
                    bucket = "out_same" if (s > 0) == target_pos else "out_diff"
                    pooled[model][bucket][0].append(y_e)
                    pooled[model][bucket][1].append(p_e)

    rows_out = []
    for model in MODELS:
        print(f"\n{model}: {n_edges_total[model]} target-edge instances across 6 datasets")
        for bucket in BUCKETS:
            y_list, p_list = pooled[model][bucket]
            n = len(y_list)
            auc = roc_auc_score(y_list, p_list) if n and len(set(y_list)) > 1 else float("nan")
            print(f"  {bucket:<10} n={n:>10,}  AUC={auc:.4f}" if n else f"  {bucket:<10} n=0")
            rows_out.append({"model": model, "bucket": bucket, "auc": auc, "n": n})

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    import csv
    with open(OUT_CSV, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["model", "bucket", "auc", "n"])
        wtr.writeheader()
        wtr.writerows(rows_out)
    print(f"\nsaved {OUT_CSV} ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
