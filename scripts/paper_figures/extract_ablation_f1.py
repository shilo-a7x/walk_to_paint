"""F1 (and accuracy) for every ablation, computed entirely from already-saved
posthoc artifacts -- zero GPU compute, zero re-fitting.

Each ablation's posthoc run already saved (a) the raw per-walk-occurrence
predictions (test_predictions.pkl: edge_ids, probabilities, targets) and
(b) the fitted func_logit_power weight-power theta* (model_config.json).
Edge-level aggregation is a closed-form weighted mean (run_posthoc.py's
_wfn_edge_scores) -- reusing the already-fit theta*, this is pure numpy
postprocessing on data already on disk, no retraining, no re-optimization.

Covers all four ablations sharing the ABLATION_<TAG>_s<seed> / run_id
"ablation_agg" convention: DIRFLIP, MASKNODE, MASKEDGE, SIGNSCRAMBLE.
Skips any (ablation, dataset, seed) whose artifacts aren't on disk yet
(e.g. SIGNSCRAMBLE, still running) rather than failing the whole script.
"""
import csv
import glob
import json
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
DISPLAY = {"bitcoin-alpha": "Bitcoin-alpha", "bitcoin-otc": "Bitcoin-otc", "epinions": "Epinions",
           "wiki-elec": "Wiki-elec", "wiki-rfa": "Wiki-RfA", "slashdot090221": "Slashdot"}
SEEDS = list(range(42, 52))
ABLATIONS = ["DIRFLIP", "MASKNODE", "MASKEDGE", "SIGNSCRAMBLE"]
RUN_ID = "ablation_agg"
FUNC_EPS = 1e-9

OUT_CSV = "aaai2027/figure_data/ablation_f1.csv"


def latest_exp_dir(dataset, tag, seed):
    matches = sorted(glob.glob(f"outputs/{dataset}/ABLATION_{tag}_s{seed}_*"))
    return matches[-1] if matches else None


def edge_weighted_scores(edge_ids, probs, theta):
    """run_posthoc.py's func_logit_power weighted-mean aggregation,
    reusing an already-fit theta -- exact same formula, no optimization."""
    q = np.clip(probs, FUNC_EPS, 1 - FUNC_EPS)
    w = np.power(np.abs(np.log(q / (1 - q))) + FUNC_EPS, theta)
    w = np.maximum(w, FUNC_EPS)

    order = np.argsort(edge_ids, kind="stable")
    eids_s, w_s, q_s = edge_ids[order], w[order], probs[order]
    uniq, inv, cnts = np.unique(eids_s, return_inverse=True, return_counts=True)
    wsum = np.bincount(inv, weights=w_s, minlength=len(uniq))
    wqsum = np.bincount(inv, weights=w_s * q_s, minlength=len(uniq))
    scores = wqsum / np.maximum(wsum, FUNC_EPS)
    first_occ = np.concatenate([[0], np.cumsum(cnts)[:-1]])
    labels = None  # filled by caller from targets at first_occ
    return scores, first_occ, order, eids_s


def compute_f1_for_run(exp_dir, dataset):
    cfg_path = os.path.join(exp_dir, "posthoc", RUN_ID, "aggregator", "func_logit_power", "model_config.json")
    if not os.path.isfile(cfg_path):
        return None
    theta = json.load(open(cfg_path))["theta_star"][0]

    pred_dirs = sorted(glob.glob(os.path.join(exp_dir, "checkpoints", f"{dataset}_predictions", "epoch_*")))
    if not pred_dirs:
        return None
    test_pkl = os.path.join(pred_dirs[-1], "test_predictions.pkl")
    if not os.path.isfile(test_pkl):
        # fall back to scanning all epoch dirs for one that has it (in case
        # of a stale/partial "last" dir)
        found = None
        for d in reversed(pred_dirs):
            p = os.path.join(d, "test_predictions.pkl")
            if os.path.isfile(p):
                found = p
                break
        if found is None:
            return None
        test_pkl = found

    dd = pickle.load(open(test_pkl, "rb"))
    edge_ids = dd["edge_ids"]
    probs = dd["probabilities"][:, 1].astype(float)
    targets = dd["targets"].astype(int)

    # walk-level F1/accuracy (raw per-occurrence predictions, threshold 0.5)
    walk_pred = (probs >= 0.5).astype(int)
    walk_f1 = f1_score(targets, walk_pred, average='macro')
    walk_acc = accuracy_score(targets, walk_pred)

    # edge-level F1/accuracy (aggregated via the already-fit func_logit_power weights)
    scores, first_occ, order, eids_s = edge_weighted_scores(edge_ids, probs, theta)
    edge_labels = targets[order][first_occ]
    edge_pred = (scores >= 0.5).astype(int)
    edge_f1 = f1_score(edge_labels, edge_pred, average='macro')
    edge_acc = accuracy_score(edge_labels, edge_pred)

    return {
        "walk_f1": walk_f1, "walk_acc": walk_acc,
        "edge_f1": edge_f1, "edge_acc": edge_acc,
        "n_edges": len(edge_labels),
    }


def main():
    rows = []
    print(f"{'ablation':14s} {'dataset':16s} {'n_seeds':>7s} {'walk_f1':>18s} {'edge_f1':>18s}")
    for tag in ABLATIONS:
        for ds in DATASETS:
            per_seed = []
            for seed in SEEDS:
                exp_dir = latest_exp_dir(ds, tag, seed)
                if exp_dir is None:
                    continue
                r = compute_f1_for_run(exp_dir, ds)
                if r is not None:
                    r["seed"] = seed
                    per_seed.append(r)
            if not per_seed:
                print(f"{tag:14s} {DISPLAY[ds]:16s}  NO DATA YET")
                continue
            walk_f1s = np.array([r["walk_f1"] for r in per_seed])
            edge_f1s = np.array([r["edge_f1"] for r in per_seed])
            walk_accs = np.array([r["walk_acc"] for r in per_seed])
            edge_accs = np.array([r["edge_acc"] for r in per_seed])
            print(f"{tag:14s} {DISPLAY[ds]:16s} {len(per_seed):7d} "
                  f"{walk_f1s.mean():8.4f}+-{walk_f1s.std(ddof=1) if len(walk_f1s) > 1 else 0:6.4f} "
                  f"{edge_f1s.mean():8.4f}+-{edge_f1s.std(ddof=1) if len(edge_f1s) > 1 else 0:6.4f}")
            rows.append({
                "ablation": tag, "dataset": ds, "display_name": DISPLAY[ds], "n_seeds": len(per_seed),
                "walk_f1_mean": walk_f1s.mean(), "walk_f1_std": walk_f1s.std(ddof=1) if len(walk_f1s) > 1 else 0.0,
                "walk_acc_mean": walk_accs.mean(),
                "edge_f1_mean": edge_f1s.mean(), "edge_f1_std": edge_f1s.std(ddof=1) if len(edge_f1s) > 1 else 0.0,
                "edge_acc_mean": edge_accs.mean(),
            })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
