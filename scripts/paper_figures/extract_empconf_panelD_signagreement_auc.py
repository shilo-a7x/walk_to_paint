"""Extract step for Empirical Confirmation Panel D -- 4-way sign-agreement AUC.

Rebuilt 2026-08-18 as a real multiseed (10-seed) result, per the WSDM closeout plan's
Phase 6 Panel D task. The prior version computed AUC once, pooled across all 6 datasets,
from a single fixed canonical SiGAT reproduction
(outputs/lead4_entropy_heterogeneity/predictions_raw_canonical.pkl). This pulls 10
independently-trained SiGAT predictions per dataset (seeds 42-51), reusing
extract_multiseed_entropy_heatmaps.py's own sigat_raw_seed() -- the same already-verified
infra used for the entropy heatmaps and Table 1's SiGAT row (fits a fresh
LogisticRegression per seed on that seed's own best_epoch_artifacts.pkl +
splits_canonical[_seed<N>], so each seed's predictions come from a real, independently
trained model, not a stale reused one).

For target edge e=(u,v) with prediction (y_e, p_e): look at every OTHER real edge from
the full canonical edge list (train+val+test -- ground truth topology, not a model
prediction, so no leakage; split-independent, computed once per dataset, reused across
all 10 seeds) incident to v that points INTO v, or incident to u that points OUT of u.
Classify each (neighbor, e) pair as same/diff by sign agreement. One (y_e, p_e) row is
contributed per qualifying neighbor pair -- a target edge can land in multiple buckets
and be counted multiple times within one bucket (unchanged from the single-split
version).

Two aggregations from the same per-(dataset, seed, bucket) predictions, written to two
CSVs:
- POOLED (matches the original panel's scope, aaai2027/figure_data/
  empconf_panelD_signagreement_auc.csv): for each seed, pool (y, p) across all 6
  datasets within a bucket, compute one AUC, then mean +- std across the 10 seeds
  (Option 2 convention -- per-split AUC then average, not raw predictions pooled across
  seeds -- matches CLAUDE.md's "Entropy-heatmap multi-split methodology" note).
- PER-DATASET (new, per the professor's C-ter ask -- draft/candidate, placement in the
  paper not yet decided, aaai2027/figure_data/
  empconf_panelD_signagreement_auc_perdataset.csv): AUC per (dataset, bucket, seed),
  mean +- std across the 10 seeds, no cross-dataset pooling.
"""
import csv
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baselines"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from src.utils.config import load_config  # noqa: E402
from src.data.prepare_data import get_edge_list  # noqa: E402
from scripts.paper_figures.extract_multiseed_entropy_heatmaps import sigat_raw_seed, SEEDS  # noqa: E402

OUT_POOLED_CSV = "aaai2027/figure_data/empconf_panelD_signagreement_auc.csv"
OUT_PERDATASET_CSV = "aaai2027/figure_data/empconf_panelD_signagreement_auc_perdataset.csv"
DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
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


def bucket_predictions(rec, in_edges_of, out_edges_of):
    """rec: {'u','v','y','p'} for one (dataset, seed). Returns {bucket: (y_list, p_list)}."""
    out = {b: ([], []) for b in BUCKETS}
    for u, v, y_e, p_e in zip(rec["u"], rec["v"], rec["y"], rec["p"]):
        target_pos = (y_e == 1)
        for x, s in in_edges_of.get(v, []):
            if x == u:
                continue
            b = "in_same" if (s > 0) == target_pos else "in_diff"
            out[b][0].append(y_e)
            out[b][1].append(p_e)
        for w, s in out_edges_of.get(u, []):
            if w == v:
                continue
            b = "out_same" if (s > 0) == target_pos else "out_diff"
            out[b][0].append(y_e)
            out[b][1].append(p_e)
    return out


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(y) and len(set(y)) > 1 else float("nan")


# Cache for sigat_raw_seed()'s output -- the actually expensive step (loading a large
# embedding pickle + fitting a fresh LogisticRegression + predict_proba over the full test
# split, once per (dataset, seed), 60 fits total). SiGAT only ever saves node embeddings
# (best_epoch_artifacts.pkl), never a final per-edge prediction array, so this fit-and-score
# step is unavoidable the FIRST time -- but nothing about it is specific to the
# sign-agreement bucketing done below, so caching the raw (u, v, y, p) output here means any
# future consumer (revisiting this panel with a different bucket/visualization choice, or a
# completely different downstream analysis) can reuse it without refitting. Per the project's
# standing rule (CLAUDE.md): cache the expensive step separately from the cheap one. Delete a
# file here to force a real recompute for that (dataset, seed), e.g. if the underlying SiGAT
# checkpoint changes.
RAW_CACHE_DIR = os.path.join(ROOT, "outputs", "cache", "sigat_raw_predictions")


def sigat_raw_seed_cached(ds, seed):
    cache_path = os.path.join(RAW_CACHE_DIR, f"{ds}__seed{seed}.pkl")
    if os.path.exists(cache_path):
        import pickle
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    rec = sigat_raw_seed(ds, seed)
    if rec is not None and len(rec["u"]):
        os.makedirs(RAW_CACHE_DIR, exist_ok=True)
        import pickle
        with open(cache_path, "wb") as f:
            pickle.dump(rec, f)
    return rec


def main():
    # per_seed_buckets[ds][seed] = {bucket: (y_list, p_list)}
    per_seed_buckets = {ds: {} for ds in DATASETS}

    for ds in DATASETS:
        edges = get_edge_list(load_cfg(ds))
        in_edges_of, out_edges_of = build_adjacency(edges)
        for seed in SEEDS:
            rec = sigat_raw_seed_cached(ds, seed)
            if rec is None or not len(rec["u"]):
                print(f"{ds} seed={seed}: MISSING")
                continue
            per_seed_buckets[ds][seed] = bucket_predictions(rec, in_edges_of, out_edges_of)
        print(f"{ds}: {len(per_seed_buckets[ds])}/{len(SEEDS)} seeds usable")

    # --- per-dataset aggregation ---
    perdataset_rows = []
    for ds in DATASETS:
        for bucket in BUCKETS:
            aucs = []
            for seed, buckets in per_seed_buckets[ds].items():
                y, p = buckets[bucket]
                a = safe_auc(y, p)
                if not np.isnan(a):
                    aucs.append(a)
            if aucs:
                perdataset_rows.append({
                    "dataset": ds, "bucket": bucket,
                    "mean_auc": float(np.mean(aucs)),
                    "std_auc": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                    "n_seeds": len(aucs),
                })
            else:
                perdataset_rows.append({"dataset": ds, "bucket": bucket,
                                         "mean_auc": "", "std_auc": "", "n_seeds": 0})

    # --- pooled aggregation: pool across datasets WITHIN each seed, then mean over seeds ---
    pooled_rows = []
    for bucket in BUCKETS:
        seed_aucs, seed_ns = [], []
        for seed in SEEDS:
            y_all, p_all = [], []
            for ds in DATASETS:
                if seed in per_seed_buckets[ds]:
                    y, p = per_seed_buckets[ds][seed][bucket]
                    y_all.extend(y)
                    p_all.extend(p)
            a = safe_auc(y_all, p_all)
            if not np.isnan(a):
                seed_aucs.append(a)
                seed_ns.append(len(y_all))
        pooled_rows.append({
            "bucket": bucket,
            "mean_auc": float(np.mean(seed_aucs)) if seed_aucs else "",
            "std_auc": float(np.std(seed_aucs, ddof=1)) if len(seed_aucs) > 1 else 0.0,
            "n_seeds": len(seed_aucs),
            "mean_n": round(float(np.mean(seed_ns))) if seed_ns else "",
        })

    os.makedirs("aaai2027/figure_data", exist_ok=True)
    with open(OUT_POOLED_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bucket", "mean_auc", "std_auc", "n_seeds", "mean_n"])
        w.writeheader()
        w.writerows(pooled_rows)
    print(f"\nsaved {OUT_POOLED_CSV}")
    for r in pooled_rows:
        print(f"  {r['bucket']:<10} AUC={r['mean_auc']:.4f} +- {r['std_auc']:.4f}"
              f"  n_seeds={r['n_seeds']}  mean_n={r['mean_n']}")

    with open(OUT_PERDATASET_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "bucket", "mean_auc", "std_auc", "n_seeds"])
        w.writeheader()
        w.writerows(perdataset_rows)
    print(f"saved {OUT_PERDATASET_CSV}")


if __name__ == "__main__":
    main()
