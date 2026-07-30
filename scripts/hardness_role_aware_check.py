"""Follow-up check: is role-aware attribution (H_out(u) for source-role errors,
H_in(v) for target-role errors) actually better than the symmetric both-sided map,
or just better-covered? Splits each node's real test error by the ROLE it played
(source vs target) in each edge, and correlates H_out against source-role error,
H_in against target-role error separately. No training -- pure analysis.
"""
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import torch
import numpy as np
from scipy.stats import spearmanr
from scripts.balance_theory_paths import load_edges_canonical
from scripts.hardness_entropy_screen import node_entropy_train_mask, ATTN_VARIANTS, find_run_dir
from scripts.hardness_predictive_validity import DATASETS, load_test_predictions, MIN_OCC


def role_split_error(pred, edges, min_occ=MIN_OCC):
    # Vectorized indexing (~50x faster than a per-occurrence Python list comprehension).
    edges_arr = np.asarray(edges, dtype=np.int64)
    edge_ids = pred["edge_ids"]
    correct = pred["correct"].astype(np.float64)
    n_nodes = max(max(u, v) for u, v, _ in edges) + 1
    src_wrong = np.zeros(n_nodes); src_total = np.zeros(n_nodes)
    tgt_wrong = np.zeros(n_nodes); tgt_total = np.zeros(n_nodes)
    valid = (edge_ids >= 0) & (edge_ids < len(edges))
    eids = edge_ids[valid].astype(np.int64)
    corr = correct[valid]
    us = edges_arr[eids, 0]
    vs = edges_arr[eids, 1]
    np.add.at(src_wrong, us, 1.0 - corr); np.add.at(src_total, us, 1.0)
    np.add.at(tgt_wrong, vs, 1.0 - corr); np.add.at(tgt_total, vs, 1.0)
    src_err = np.full(n_nodes, np.nan); tgt_err = np.full(n_nodes, np.nan)
    src_ok = src_total >= min_occ; tgt_ok = tgt_total >= min_occ
    src_err[src_ok] = src_wrong[src_ok] / src_total[src_ok]
    tgt_err[tgt_ok] = tgt_wrong[tgt_ok] / tgt_total[tgt_ok]
    return src_err, src_ok, tgt_err, tgt_ok


def corr(cand, err, ok):
    nodes = np.nonzero(ok)[0]
    xs, ys = [], []
    for n in nodes:
        if not np.isfinite(cand[n]):
            continue
        xs.append(cand[n]); ys.append(err[n])
    if len(xs) < 10:
        return None, 0
    rho, _ = spearmanr(xs, ys)
    return rho, len(xs)


def main():
    print(f"{'dataset':<16}{'H_out vs src-err':>18}{'H_in vs tgt-err':>18}{'H_in vs src-err':>18}{'H_out vs tgt-err':>18}")
    for ds, (data_dir_name, cache_file) in DATASETS.items():
        edges = load_edges_canonical(ds)
        n_nodes = max(max(u, v) for u, v, _ in edges) + 1
        cache = torch.load(f"data/{data_dir_name}/{cache_file}", map_location="cpu", weights_only=False)
        train_mask_edges = cache["splits"]["train"] + cache["splits"]["mask"]
        h_out, h_in = node_entropy_train_mask(train_mask_edges, n_nodes)

        rows = {"out_src": [], "in_tgt": [], "in_src": [], "out_tgt": []}
        for attn_name, pattern in ATTN_VARIANTS.items():
            run_dir = find_run_dir(ds, pattern)
            if run_dir is None:
                continue
            pred = load_test_predictions(run_dir, ds)
            if pred is None:
                continue
            src_err, src_ok, tgt_err, tgt_ok = role_split_error(pred, edges)
            r1, _ = corr(h_out, src_err, src_ok)   # matched: source-role predicted by out-entropy
            r2, _ = corr(h_in, tgt_err, tgt_ok)     # matched: target-role predicted by in-entropy
            r3, _ = corr(h_in, src_err, src_ok)     # mismatched (cross-check)
            r4, _ = corr(h_out, tgt_err, tgt_ok)    # mismatched (cross-check)
            if r1 is not None: rows["out_src"].append(r1)
            if r2 is not None: rows["in_tgt"].append(r2)
            if r3 is not None: rows["in_src"].append(r3)
            if r4 is not None: rows["out_tgt"].append(r4)

        def m(k):
            return np.mean(rows[k]) if rows[k] else float("nan")
        print(f"{ds:<16}{m('out_src'):>18.4f}{m('in_tgt'):>18.4f}{m('in_src'):>18.4f}{m('out_tgt'):>18.4f}")


if __name__ == "__main__":
    main()
