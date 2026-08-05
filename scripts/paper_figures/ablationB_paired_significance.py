"""Paired-bootstrap significance check for Ablation B's (paper enumeration;
"Ablation C" in this file's own code/data, pre-dating the A/B relabel)
aggregator comparison.

Why this is needed: the Hanley-McNeil SE bars in the plot are computed
INDEPENDENTLY per aggregator, but all aggregators are evaluated on the exact
same test edges (same underlying q/ds/de/len/rp features, just recombined
differently) -- so the comparisons are paired/correlated, and eyeballing
whether two independent SE bars overlap is not a valid significance test
for "is aggregator A really better than aggregator B on this data." A paired
bootstrap over EDGES (not over independent draws) is the correct tool: each
resample recomputes both aggregators' AUC on the SAME resampled edge set, so
common noise cancels and only the genuine paired difference remains.

Reuses run_posthoc.py's own _func_registry() (the exact weight-function
definitions used to fit theta*) plus each model's already-fit theta* from
the funcsweep_20260727 run, so scores are recomputed identically to the
real sweep, not approximated.
"""
import json
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from run_posthoc import _func_registry

RUN_INFO = {
    "bitcoin-alpha":   ("E27_NOHARD_EDGECOVER_LOCALATTN4_20260719-121955", 27),
    "bitcoin-otc":     ("E27_NOHARD_EDGECOVER_LOCALATTN4_20260719-121955", 46),
    "epinions":        ("E27_NOHARD_EDGECOVER_LOCALATTN4_20260719-121955", 30),
    "wiki-elec":       ("E27_NOHARD_EDGECOVER_LOCALATTN4_20260719-122848", 36),
    "wiki-rfa":        ("E27_NOHARD_EDGECOVER_LOCALATTN4_20260719-123214", 33),
    "slashdot090221":  ("E27_NOHARD_EDGECOVER_LOCALATTN4_20260719-121955", 34),
}
RUN_ID = "funcsweep_20260727"
N_BOOT = 3000
SEED = 42


def load_theta(ds, run_dir, model):
    path = f"outputs/{ds}/{run_dir}/posthoc/{RUN_ID}/aggregator/{model}/model_config.json"
    with open(path) as f:
        return json.load(f)["theta_star"]


def edge_scores(model, theta, edge_ids, q, ds_arr, de_arr, len_arr, rp_arr, y_occ):
    reg = _func_registry()
    w_fn = reg[model][0]
    w = np.maximum(w_fn(theta, q, ds_arr, de_arr, len_arr, rp_arr), 1e-9)
    order = np.argsort(edge_ids, kind="stable")
    eids_s, w_s, q_s, y_s = edge_ids[order], w[order], q[order], y_occ[order]
    uniq, inv_idx, counts = np.unique(eids_s, return_inverse=True, return_counts=True)
    n_e = len(uniq)
    wsum = np.bincount(inv_idx, weights=w_s, minlength=n_e)
    wqsum = np.bincount(inv_idx, weights=w_s * q_s, minlength=n_e)
    scores = wqsum / np.maximum(wsum, 1e-9)
    first_occ = np.concatenate([[0], np.cumsum(counts)[:-1]])
    labels = y_s[first_occ].astype(int)
    return scores, labels


def paired_bootstrap(scores_a, scores_b, labels, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(labels)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = labels[idx]
        if len(np.unique(yb)) < 2:
            diffs[i] = np.nan
            continue
        auc_a = roc_auc_score(yb, scores_a[idx])
        auc_b = roc_auc_score(yb, scores_b[idx])
        diffs[i] = auc_a - auc_b
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_le_0 = float((diffs <= 0).mean())
    return diffs.mean(), lo, hi, p_le_0


def main():
    comparisons = [("func_logit_pos", "func_uniform"),
                   ("func_logit_pos", "func_logit_power"),
                   ("func_logit_len", "func_uniform"),
                   ("func_logit_len", "func_logit_power")]

    for ds, (run_dir, epoch) in RUN_INFO.items():
        base = f"outputs/{ds}/{run_dir}"
        with open(f"{base}/checkpoints/{ds}_predictions/epoch_{epoch:03d}/test_predictions.pkl", "rb") as f:
            preds = pickle.load(f)
        edge_ids = np.asarray(preds["edge_ids"]).astype(np.int64)
        q = preds["probabilities"][:, 1].astype(np.float64)
        y_occ = np.asarray(preds["targets"]).astype(int)
        ds_arr = np.asarray(preds["dist_from_start"], dtype=np.float64)
        len_arr = np.maximum(np.asarray(preds["walk_lengths"], dtype=np.float64), 1.0)
        de_arr = np.maximum(len_arr - ds_arr - 1.0, 0.0)
        rp_arr = ds_arr / len_arr

        cache = {}
        for model_a, model_b in comparisons:
            for m in (model_a, model_b):
                if m not in cache:
                    theta = load_theta(ds, run_dir, m)
                    cache[m] = edge_scores(m, theta, edge_ids, q, ds_arr, de_arr, len_arr, rp_arr, y_occ)
            scores_a, labels = cache[model_a]
            scores_b, _ = cache[model_b]
            mean_diff, lo, hi, p_le_0 = paired_bootstrap(scores_a, scores_b, labels, N_BOOT, SEED)
            sig = "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
            print(f"{ds:16s} {model_a:18s} - {model_b:18s}: "
                  f"mean_diff={mean_diff:+.4f}  95% CI=[{lo:+.4f},{hi:+.4f}]  ({sig})  P(diff<=0)={p_le_0:.3f}")
        print()


if __name__ == "__main__":
    main()
