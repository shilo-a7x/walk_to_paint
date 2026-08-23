"""Diagnostic (not part of the paper pipeline): how much test AUC is explained by a
trivial 2-parameter vertex-reputation model -- out_rate[u] (source's train-pool
out-edge positive rate) + in_rate[v] (target's train-pool in-edge positive rate),
fit with a plain sklearn LogisticRegression -- versus PEWTER's full model.

Written in response to a real concern raised while auditing the mask_edge_tokens
ablation (2026-08-23): edge-sign context turned out to matter less than expected,
which raised the question "why do we beat every GNN/SGNN baseline by 3-5pp then?".
This script is the falsifiable probe used to investigate that, plus a stress test
(shuffle control + paired bootstrap) to rule out data-plumbing artifacts before
trusting the numbers. See TRIVIAL_BASELINE_INVESTIGATION.md for the writeup and
literature context (this 2-feature model is, in substance, the "degree features"
of Leskovec et al. 2010 and the one-shot linear analogue of the Fairness/Goodness
model of Kumar et al. 2016 -- both established baselines on these same datasets).

Usage:
    .venv/bin/python scripts/trivial_baseline_probe.py [--n-boot 2000]

Reads the walk model's own canonical dataset_cache splits (data/<ds>/dataset_cache__*.pt)
and PEWTER's saved test_predictions.pkl per dataset -- read-only, no training, no
checkpoint touched.
"""
import argparse
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines.postprocess_canonical import _edgeid_to_uv

JOBS = [
    ("wiki-elec",
     "data/wiki-Elec/dataset_cache__edge_cover_nw155534_mw80_seed42.pt",
     "outputs/wiki-elec/E32_PY314_LOCALATTN4_20260804-232140/checkpoints/wiki-elec_predictions/epoch_044/test_predictions.pkl"),
    ("bitcoin-alpha",
     "data/bitcoin-alpha/dataset_cache__edge_cover_nw120930_mw80_seed42.pt",
     "outputs/bitcoin-alpha/E32_PY314_LOCALATTN4_20260804-225948/checkpoints/bitcoin-alpha_predictions/epoch_035/test_predictions.pkl"),
    ("bitcoin-otc",
     "data/bitcoin-otc/dataset_cache__edge_cover_nw177960_mw80_seed42.pt",
     "outputs/bitcoin-otc/E32_PY314_LOCALATTN4_20260804-231122/checkpoints/bitcoin-otc_predictions/epoch_035/test_predictions.pkl"),
    ("epinions",
     "data/epinions/dataset_cache__edge_cover_nw840799_mw80_seed42.pt",
     "outputs/epinions/E32_PY314_LOCALATTN4_20260804-225948/checkpoints/epinions_predictions/epoch_030/test_predictions.pkl"),
    ("wiki-rfa",
     "data/wiki-RfA/dataset_cache__edge_cover_nw265817_mw80_seed42.pt",
     "outputs/wiki-rfa/E32_PY314_LOCALATTN4_20260804-232239/checkpoints/wiki-rfa_predictions/epoch_033/test_predictions.pkl"),
    ("slashdot090221",
     "data/slashdot090221/dataset_cache__edge_cover_nw1647606_mw80_seed42.pt",
     "outputs/slashdot090221/E32_PY314_LOCALATTN4_20260804-225948/checkpoints/slashdot090221_predictions/epoch_034/test_predictions.pkl"),
]


def build_rates(train_pool):
    """out_pos/out_tot: source u's train-pool out-edges (# positive, # total).
    in_pos/in_tot: target v's train-pool in-edges (# positive, # total)."""
    out_pos, out_tot, in_pos, in_tot = (
        defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    )
    for u, v, s in train_pool:
        out_tot[u] += 1
        out_pos[u] += 1 if s > 0 else 0
        in_tot[v] += 1
        in_pos[v] += 1 if s > 0 else 0
    global_pos_rate = sum(1 for _, _, s in train_pool if s > 0) / len(train_pool)
    return out_pos, out_tot, in_pos, in_tot, global_pos_rate


def make_features(edges, out_pos, out_tot, in_pos, in_tot, global_pos_rate):
    """X = [out_rate[u], in_rate[v]] per edge; unseen vertex falls back to the
    train-pool global positive rate (cold start, not zero/NaN)."""
    X, Y = [], []
    for u, v, s in edges:
        ro = out_pos[u] / out_tot[u] if out_tot.get(u, 0) > 0 else global_pos_rate
        ri = in_pos[v] / in_tot[v] if in_tot.get(v, 0) > 0 else global_pos_rate
        X.append([ro, ri])
        Y.append(1 if s > 0 else 0)
    return np.array(X), np.array(Y)


def load_full_model_edge_predictions(cache_path, pkl_path):
    """PEWTER's own saved per-walk predictions, averaged to one probability per
    edge (mean over all walk occurrences), keyed by (u, v) to align with the
    trivial baseline's own edges."""
    e2uv = _edgeid_to_uv(cache_path)
    dd = pickle.load(open(pkl_path, "rb"))
    eid = dd["edge_ids"]
    p = dd["probabilities"][:, 1].astype(float)
    y = dd["targets"]
    order = np.argsort(eid)
    eid, p, y = eid[order], p[order], y[order]
    uniq_ids, starts = np.unique(eid, return_index=True)
    ends = np.append(starts[1:], len(eid))
    mean_p = np.array([p[starts[i]:ends[i]].mean() for i in range(len(uniq_ids))])
    labels = y[starts]
    out = {}
    for i, k in enumerate(uniq_ids):
        k = int(k)
        if k in e2uv:
            out[e2uv[k]] = (int(labels[i]), float(mean_p[i]))
    return out


def run(ds, cache_path, pkl_path, n_boot, seed):
    print(f"\n{'=' * 70}\n{ds}\n{'=' * 70}")
    splits = torch.load(cache_path, map_location="cpu", weights_only=False)["splits"]
    train_pool = splits["train"] + splits["mask"]
    val_edges = splits["val"]
    test_edges = splits["test"]

    tp_set, v_set, t_set = set(train_pool), set(val_edges), set(test_edges)
    overlap = len(tp_set & v_set) + len(tp_set & t_set) + len(v_set & t_set)
    print(f"train/val/test triple overlap (should be 0): {overlap}")

    train_pool_vertices = set()
    for u, v, _ in train_pool:
        train_pool_vertices.add(u)
        train_pool_vertices.add(v)
    src_seen = sum(1 for u, _, _ in test_edges if u in train_pool_vertices)
    tgt_seen = sum(1 for _, v, _ in test_edges if v in train_pool_vertices)
    print(f"test edges: {len(test_edges)}  source-seen-in-train-pool: {src_seen}/{len(test_edges)}  "
          f"target-seen-in-train-pool: {tgt_seen}/{len(test_edges)}")

    out_pos, out_tot, in_pos, in_tot, grp = build_rates(train_pool)
    Xval, Yval = make_features(val_edges, out_pos, out_tot, in_pos, in_tot, grp)
    Xtst, Ytst = make_features(test_edges, out_pos, out_tot, in_pos, in_tot, grp)

    clf = LogisticRegression().fit(Xval, Yval)
    Ptst = clf.predict_proba(Xtst)[:, 1]
    auc_trivial = roc_auc_score(Ytst, Ptst)
    print(f"trivial-baseline test AUC (fit on val, scored on test): {auc_trivial:.4f}")
    print(f"  logistic-regression coefficients: out_rate[u]={clf.coef_[0][0]:+.3f}  "
          f"in_rate[v]={clf.coef_[0][1]:+.3f}  intercept={clf.intercept_[0]:+.3f}")

    # shuffle control: permute train-pool signs, keep graph structure fixed --
    # should collapse to ~0.50 if there's no data-plumbing artifact.
    rng = np.random.default_rng(seed)
    shuffled_signs = [s for _, _, s in train_pool]
    rng.shuffle(shuffled_signs)
    shuffled_pool = [(u, v, s) for (u, v, _), s in zip(train_pool, shuffled_signs)]
    out_pos_s, out_tot_s, in_pos_s, in_tot_s, grp_s = build_rates(shuffled_pool)
    Xval_s, Yval_s = make_features(val_edges, out_pos_s, out_tot_s, in_pos_s, in_tot_s, grp_s)
    Xtst_s, Ytst_s = make_features(test_edges, out_pos_s, out_tot_s, in_pos_s, in_tot_s, grp_s)
    clf_s = LogisticRegression().fit(Xval_s, Yval_s)
    auc_shuffled = roc_auc_score(Ytst_s, clf_s.predict_proba(Xtst_s)[:, 1])
    print(f"shuffle-control test AUC (should be ~0.50): {auc_shuffled:.4f}")

    # paired bootstrap vs. the full model, on exactly-shared (u, v) edges only.
    full_uv_p = load_full_model_edge_predictions(cache_path, pkl_path)
    trivial_uv_p = {(u, v): (y_, p_) for (u, v, _), y_, p_ in zip(test_edges, Ytst, Ptst)}
    shared_keys = sorted(set(trivial_uv_p) & set(full_uv_p))
    print(f"shared (u,v) between trivial test set and full-model predictions: "
          f"{len(shared_keys)}/{len(test_edges)}")

    Y_arr = np.array([trivial_uv_p[k][0] for k in shared_keys])
    assert np.array_equal(Y_arr, np.array([full_uv_p[k][0] for k in shared_keys])), \
        "label mismatch on shared edges -- data alignment bug"
    P_triv = np.array([trivial_uv_p[k][1] for k in shared_keys])
    P_full = np.array([full_uv_p[k][1] for k in shared_keys])

    n = len(Y_arr)
    boot_rng = np.random.default_rng(seed + 1)
    boot_triv, boot_full, boot_diff = [], [], []
    for _ in range(n_boot):
        idx = boot_rng.integers(0, n, n)
        yb = Y_arr[idx]
        if len(set(yb)) < 2:
            continue
        boot_triv.append(roc_auc_score(yb, P_triv[idx]))
        boot_full.append(roc_auc_score(yb, P_full[idx]))
        boot_diff.append(boot_full[-1] - boot_triv[-1])
    boot_triv, boot_full, boot_diff = map(np.array, (boot_triv, boot_full, boot_diff))

    lo_t, hi_t = np.percentile(boot_triv, [2.5, 97.5])
    lo_f, hi_f = np.percentile(boot_full, [2.5, 97.5])
    lo_d, hi_d = np.percentile(boot_diff, [2.5, 97.5])
    p_full_le_trivial = float((boot_diff <= 0).mean())

    auc_full_point = roc_auc_score(Y_arr, P_full)
    auc_triv_point = roc_auc_score(Y_arr, P_triv)
    print(f"\nOn the {n} exactly-shared edges (identical test set for both models):")
    print(f"  trivial AUC = {auc_triv_point:.4f}  95% CI [{lo_t:.4f}, {hi_t:.4f}]")
    print(f"  full    AUC = {auc_full_point:.4f}  95% CI [{lo_f:.4f}, {hi_f:.4f}]")
    print(f"  (full - trivial) = {auc_full_point - auc_triv_point:+.4f}  95% CI [{lo_d:+.4f}, {hi_d:+.4f}]")
    print(f"  P(full <= trivial) over {len(boot_diff)} paired bootstrap resamples = {p_full_le_trivial:.3f}")

    return {
        "dataset": ds, "auc_trivial": auc_triv_point, "auc_full": auc_full_point,
        "delta": auc_full_point - auc_triv_point, "delta_ci_lo": lo_d, "delta_ci_hi": hi_d,
        "p_full_le_trivial": p_full_le_trivial, "auc_shuffled": auc_shuffled,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results = [run(ds, cache, pkl, args.n_boot, args.seed) for ds, cache, pkl in JOBS]

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    print(f"{'dataset':16s} {'trivial':>8s} {'full':>8s} {'delta':>8s} {'95% CI':>18s} "
          f"{'P(full<=triv)':>14s} {'shuffle':>8s}")
    for r in results:
        print(f"{r['dataset']:16s} {r['auc_trivial']:8.4f} {r['auc_full']:8.4f} "
              f"{r['delta']:+8.4f} [{r['delta_ci_lo']:+.4f},{r['delta_ci_hi']:+.4f}] "
              f"{r['p_full_le_trivial']:14.3f} {r['auc_shuffled']:8.4f}")


if __name__ == "__main__":
    main()
