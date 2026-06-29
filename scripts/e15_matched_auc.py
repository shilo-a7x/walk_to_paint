"""Evaluate an E15 k_cover run: new walk-model test AUC on the FULL (now ~100%)
test set and on the SOTA-covered subset (matched edges), vs SOTA walk + GNN refs.

new model: mean-prob per-edge from its test_predictions.pkl (decoded to raw uv via
its keyed cache). SOTA walk + GNN: from predictions_raw_canonical.pkl (SOTA-covered
edges) and gineconv_raw/sigat_raw (full test). mean-prob ~ func_logit_power within
~0.001 (validated in Phase 0.F), so cross-aggregator bias is negligible.

Usage: python scripts/e15_matched_auc.py <ds> <new_test_predictions.pkl> <new_cache.pt>
"""
import os, sys, pickle
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from walk_coverage_bias import gineconv_raw, sigat_raw  # inlined dense->raw GNN preds
PKL = os.path.join(ROOT, "outputs/lead4_entropy_heterogeneity/predictions_raw_canonical.pkl")


def edgeid_to_uv(cache_path):
    c = torch.load(cache_path, weights_only=False)
    enc = c["encoded"]; iid = enc["flat_input_ids"].numpy()
    sm = enc["flat_split_mask"].numpy(); eid = enc["flat_edge_ids"].numpy()
    id2t = c["tokenizer"]["id2token"]; mx = max(int(i) for i in id2t)
    id2node = np.full(mx + 1, -1, np.int64)
    for t, tok in id2t.items():
        if tok.startswith("N_"):
            id2node[int(t)] = int(tok[2:])
    em = sm != -1; idx = np.nonzero(em)[0]; ev = eid[idx]
    uniq, first = np.unique(ev, return_index=True); fp = idx[first]
    u = id2node[iid[fp - 1]]; v = id2node[iid[fp + 1]]
    return {int(e): (int(u[i]), int(v[i])) for i, e in enumerate(uniq)}


def new_peredge(pred_pkl, cache):
    d = pickle.load(open(pred_pkl, "rb"))
    eid = d["edge_ids"]; p = d["probabilities"][:, 1].astype(float); y = d["targets"]
    o = np.argsort(eid); e = eid[o]; p = p[o]; y = y[o]
    u, st = np.unique(e, return_index=True); en = np.append(st[1:], len(e))
    mp = np.array([p[st[i]:en[i]].mean() for i in range(len(u))]); yy = y[st]
    e2uv = edgeid_to_uv(cache)
    return {e2uv[int(u[i])]: (mp[i], int(yy[i])) for i in range(len(u)) if int(u[i]) in e2uv}


def auc_on(m, keys):
    yy = [m[k][1] for k in keys]; pp = [m[k][0] for k in keys]
    return roc_auc_score(yy, pp) if len(set(yy)) > 1 else float("nan")


def gnn_auc(rec, sel=None):
    uv = list(zip(rec["u"], rec["v"])); y = np.array(rec["y"]); p = np.array(rec["p"])
    if sel is None:
        idx = np.arange(len(uv))
    else:
        pos = {k: i for i, k in enumerate(uv)}; idx = [pos[k] for k in sel if k in pos]
    return roc_auc_score(y[idx], p[idx]) if len(set(y[idx])) > 1 else float("nan")


def main():
    ds, pred_pkl, cache = sys.argv[1], sys.argv[2], sys.argv[3]
    canon = pickle.load(open(PKL, "rb"))
    wf = canon[ds]["walk_full"]
    sota = {(u, v): (p, y) for u, v, p, y in zip(wf["u"], wf["v"], wf["p"], wf["y"])}
    new = new_peredge(pred_pkl, cache)
    matched = [k for k in new if k in sota]
    gi, si = gineconv_raw(ds), sigat_raw(ds)
    print(f"\n=== {ds} ===")
    print(f"new edges (full)={len(new)}  SOTA-covered={len(sota)}  matched={len(matched)}")
    print(f"  walk NEW   full(100%%)         AUC = {auc_on(new, list(new)):.4f}")
    print(f"  walk NEW   matched(SOTA edges) AUC = {auc_on(new, matched):.4f}")
    print(f"  walk SOTA  matched(SOTA edges) AUC = {auc_on(sota, matched):.4f}")
    if gi:
        print(f"  GINEConv   full(100%%)={gnn_auc(gi):.4f}   matched={gnn_auc(gi, matched):.4f}")
    if si:
        print(f"  SiGAT      full(100%%)={gnn_auc(si):.4f}   matched={gnn_auc(si, matched):.4f}")


if __name__ == "__main__":
    main()
