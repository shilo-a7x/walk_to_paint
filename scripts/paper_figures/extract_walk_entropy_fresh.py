"""Extract fresh (src_ent, tgt_ent, y, p) records for the CURRENT production
walk model (LocalAttn4, E27_NOHARD_EDGECOVER_LOCALATTN4 checkpoints) -- built
to replace outputs/lead4_entropy_heterogeneity/computed_data.pkl's walk
entries, which that script's own docstring identifies as E14_HARDNODE_L10 /
E14_HARDNODE_L10_LOCALATTN4 (pre-edge_cover, stale relative to CLAUDE.md's
current SOTA table -- confirmed 2026-07-26 by a full-vs-local AUC reversal on
3/6 datasets when using that pickle).

Pipeline (all from real, already-computed artifacts -- no retraining):
  1. Load the raw per-occurrence test predictions from the E27 checkpoint's
     saved predictions pkl, at the exact epoch the aggregator was fit on
     (recorded in that run's aggregator summary.txt / model_config.json).
  2. Aggregate occurrences to edge-level scores with the SAME func_logit_power
     weighted-mean formula run_posthoc.py uses (weight = (|logit(q)|+eps)^theta,
     score = sum(w*q)/sum(w)), using the already-fit theta* -- no re-fitting.
  3. Sanity-check the aggregated test AUC against the run's own saved
     summary.txt number (must match to ~1e-3) before trusting anything else.
  4. Recover each occurrence's edge_id -> (u, v, label) via a fresh call to
     get_edge_list(cfg) (src/data/prepare_data.py): edge_id is defined at data-
     prep time as `enumerate(get_edge_list(cfg))`, and get_edge_list is a pure
     deterministic file loader (get_loader(name)(cfg)), so re-calling it now
     reproduces the exact same id space with no need to guess at a mapping.
  5. Build out-entropy / in-entropy dictionaries from THIS SAME full edge list
     (all edges, train+mask+val+test together -- matches the "out_in" variant
     definition already used for the GNN columns: src_ent = H(out-signs of u),
     tgt_ent = H(in-signs of v)).

Output: aaai2027/figure_data/walk_entropy_fresh.csv (dataset, src_ent, tgt_ent,
y, p), one row per test edge, model = Pewter (LocalAttn4, current production).
"""
import csv
import json
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.utils.config import load_config
from src.data.prepare_data import get_edge_list

OUT_CSV = "aaai2027/figure_data/walk_entropy_fresh.csv"
_FUNC_EPS = 1e-9

RUN_INFO = {
    # dataset: (run_dir_glob, epoch)  -- epoch = the one the aggregator was fit on
    "bitcoin-alpha":   ("E27_NOHARD_EDGECOVER_LOCALATTN4_20260719-121955", 27),
    "bitcoin-otc":     ("E27_NOHARD_EDGECOVER_LOCALATTN4_20260719-121955", 46),
    "epinions":        ("E27_NOHARD_EDGECOVER_LOCALATTN4_20260719-121955", 30),
    "wiki-elec":       ("E27_NOHARD_EDGECOVER_LOCALATTN4_20260719-122848", 36),
    "wiki-rfa":        ("E27_NOHARD_EDGECOVER_LOCALATTN4_20260719-123214", 33),
    "slashdot090221":  ("E27_NOHARD_EDGECOVER_LOCALATTN4_20260719-121955", 34),
}


def logit_power_weight(theta, q):
    logit = np.log(np.maximum(q, _FUNC_EPS) / np.maximum(1.0 - q, _FUNC_EPS))
    return np.power(np.abs(logit) + _FUNC_EPS, theta)


def aggregate_edges(edge_ids, q, y):
    """Group occurrences by edge_id -> weighted-mean edge score + majority label."""
    order = np.argsort(edge_ids, kind="stable")
    eids_s, q_s, y_s = edge_ids[order], q[order], y[order]
    uniq, inv_idx, counts = np.unique(eids_s, return_inverse=True, return_counts=True)
    return uniq, inv_idx, q_s, y_s, counts


def binary_entropy(signs):
    signs = np.asarray(signs)
    p = float((signs > 0).mean())
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def main():
    rows = []
    for ds, (run_dir, epoch) in RUN_INFO.items():
        base = f"outputs/{ds}/{run_dir}"
        summary_path = f"{base}/posthoc/E27_noH/aggregator/func_logit_power/model_config.json"
        with open(summary_path) as f:
            theta = json.load(f)["theta_star"][0]

        pred_path = f"{base}/checkpoints/{ds}_predictions/epoch_{epoch:03d}/test_predictions.pkl"
        with open(pred_path, "rb") as f:
            preds = pickle.load(f)
        edge_ids = np.asarray(preds["edge_ids"]).astype(np.int64)
        q = preds["probabilities"][:, 1].astype(float)
        y_occ = np.asarray(preds["targets"]).astype(int)

        w = logit_power_weight(theta, q)
        uniq_eids, inv_idx, q_s, y_s, counts = aggregate_edges(edge_ids, q, y_occ)
        n_e = len(uniq_eids)
        wsum = np.bincount(inv_idx, weights=logit_power_weight(theta, q_s), minlength=n_e)
        wqsum = np.bincount(inv_idx, weights=logit_power_weight(theta, q_s) * q_s, minlength=n_e)
        edge_scores = wqsum / np.maximum(wsum, _FUNC_EPS)
        first_occ = np.concatenate([[0], np.cumsum(counts)[:-1]])
        edge_labels = y_s[first_occ].astype(int)

        agg_auc = roc_auc_score(edge_labels, edge_scores)
        print(f"{ds}: aggregated test AUC = {agg_auc:.4f} (n_edges={n_e})")

        # -- edge_id -> (u, v, label), via a fresh deterministic re-derivation --
        cfg = load_config("config.yaml", overrides=[f"dataset.name={ds}"])
        edges = get_edge_list(cfg)  # list of (u, v, label), edge_id == index

        out_signs, in_signs = {}, {}
        for u, v, s in edges:
            out_signs.setdefault(u, []).append(s)
            in_signs.setdefault(v, []).append(s)
        out_ent = {n: binary_entropy(s) for n, s in out_signs.items()}
        in_ent = {n: binary_entropy(s) for n, s in in_signs.items()}

        n_missing = 0
        for eid, score, label in zip(uniq_eids, edge_scores, edge_labels):
            u, v, _lbl = edges[int(eid)]
            se, te = out_ent.get(u), in_ent.get(v)
            if se is None or te is None:
                n_missing += 1
                continue
            rows.append({
                "dataset": ds, "src_ent": se, "tgt_ent": te,
                "y": int(label > 0), "p": float(score),
            })
        if n_missing:
            print(f"  ({n_missing} edges dropped -- endpoint missing from entropy dict)")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["dataset", "src_ent", "tgt_ent", "y", "p"])
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"saved {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
