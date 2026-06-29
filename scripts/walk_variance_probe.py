"""Phase 0.F (read-only): does walk variance / saturation contribute to AUC?

For each test edge the SOTA run stored one prediction per walk-occurrence
(test_predictions.pkl: edge_ids, probabilities[:,1], targets). We ask: if we
aggregate only k occurrences per edge (mean of probabilities) instead of all of
them, how does test AUC change with k? If AUC(k=1) << AUC(all) the ensemble /
saturation genuinely helps -> chasing multi-visit saturation is worthwhile. If
AUC(1) ~ AUC(all) a single walk per edge suffices -> coverage>=1 is enough.

Aggregator here is a plain mean-of-prob proxy for func_logit_power (we care about
the TREND in k, not the absolute SOTA number).
"""
import sys, pickle
import numpy as np
from sklearn.metrics import roc_auc_score

KS = [1, 2, 3, 5, 8, 16, 32, 10 ** 9]  # 1e9 == "all occurrences"


def probe(path, seed=42):
    d = pickle.load(open(path, "rb"))
    eid = d["edge_ids"]
    prob = d["probabilities"][:, 1].astype(np.float64)
    targ = d["targets"].astype(np.int64)
    rng = np.random.default_rng(seed)

    # group occurrence indices by edge id
    order = np.argsort(eid, kind="stable")
    eid_s, prob_s, targ_s = eid[order], prob[order], targ[order]
    uniq, starts = np.unique(eid_s, return_index=True)
    ends = np.append(starts[1:], len(eid_s))
    n_occ = ends - starts
    y = targ_s[starts]  # target is constant per edge

    res = {"path": path, "n_edges": int(uniq.size),
           "occ_mean": float(n_occ.mean()), "occ_median": float(np.median(n_occ)),
           "auc_by_k": {}}
    for k in KS:
        agg = np.empty(uniq.size)
        for i in range(uniq.size):
            s, e = starts[i], ends[i]
            m = e - s
            if k >= m:
                agg[i] = prob_s[s:e].mean()
            else:
                sel = rng.choice(m, size=k, replace=False)
                agg[i] = prob_s[s:e][sel].mean()
        a = roc_auc_score(y, agg) if len(np.unique(y)) > 1 else float("nan")
        res["auc_by_k"][("all" if k > 10 ** 8 else k)] = float(a)
    return res


PATHS = {
 "bitcoin-alpha": "outputs/transformer_incremental/bitcoin-alpha_seed42_nw5000000_mw80_bs1024_ep75_20260423-111407/runs/bitcoin-alpha/E14_HARDNODE_L10/checkpoints/bitcoin-alpha_predictions/epoch_024/test_predictions.pkl",
 "epinions": "outputs/transformer_incremental/epinions_seed42_nw500000_mw80_bs1024_ep50_20260416-125419/runs/epinions/E14_HARDNODE_L10/checkpoints/epinions_predictions/epoch_042/test_predictions.pkl",
 "wiki-elec": "outputs/transformer_incremental/wiki-elec_seed42_nw500000_mw80_bs1024_ep50_20260419-133621/runs/wiki-elec/E14_HARDNODE_L10/checkpoints/wiki-elec_predictions/epoch_036/test_predictions.pkl",
 "slashdot090221": "outputs/transformer_incremental/slashdot090221_seed42_nw5000000_mw80_bs1024_ep50_20260420-134149/runs/slashdot090221/E14_HARDNODE_L10/checkpoints/slashdot090221_predictions/epoch_020/test_predictions.pkl",
}

if __name__ == "__main__":
    only = sys.argv[1:] or list(PATHS)
    hdr = ["dataset", "n_edges", "occ_med"] + [str(k if k < 10**8 else "all") for k in KS]
    print(("{:14s} {:>8s} {:>7s}" + " {:>7s}" * len(KS)).format(*hdr))
    for ds in only:
        r = probe(PATHS[ds])
        row = [ds, r["n_edges"], f"{r['occ_median']:.0f}"] + \
              [f"{r['auc_by_k'][(k if k<10**8 else 'all')]:.4f}" for k in KS]
        print(("{:14s} {:>8d} {:>7s}" + " {:>7s}" * len(KS)).format(*row))
