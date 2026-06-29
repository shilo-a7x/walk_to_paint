"""Phase 0.E (read-only): is the walk-coverage gap benign?

For each dataset, compute GNN (GINEConv, SiGAT) test AUC on the walk-COVERED
subset vs the walk-UNCOVERED subset of the full nominal test set. Reuses the
canonical dense->raw machinery in baselines/postprocess_canonical.py so GNN
full-test preds land in the same raw (u,v) id space as the walk-covered set
(taken from predictions_raw_canonical.pkl walk_full).
"""
import os, pickle
import numpy as np
import torch
from sklearn import linear_model
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = os.path.join(ROOT, "baselines", "splits_canonical")
RESCAN = "results_our_splits_canonical"
DATASETS = ["bitcoin-alpha", "bitcoin-otc", "wiki-elec", "wiki-rfa", "slashdot090221", "epinions"]
PKL = os.path.join(ROOT, "outputs/lead4_entropy_heterogeneity/predictions_raw_canonical.pkl")


# --- inlined from baselines/postprocess_canonical.py (import chain is stale) ---
def _dense2raw(ds):
    sd = torch.load(os.path.join(CANON, f"{ds}.pt"), weights_only=False)
    return sd["dense2raw"], sd


def gineconv_raw(ds):
    p = os.path.join(ROOT, "baselines", "GINEConv", RESCAN, ds, "GINEConv", "seed42",
                     "best_epoch_artifacts.pkl")
    if not os.path.exists(p):
        return None
    a = pickle.load(open(p, "rb"))
    d2r, _ = _dense2raw(ds)
    eu, ev = a["edge_index"][0], a["edge_index"][1]
    u = [int(d2r[int(x)]) for x in eu]
    v = [int(d2r[int(x)]) for x in ev]
    return {"u": u, "v": v, "y": a["y"].astype(int).tolist(), "p": a["pred_p"].tolist()}


def sigat_raw(ds):
    p = os.path.join(ROOT, "baselines", "SGA", RESCAN, ds, "SiGAT", "seed42",
                     "best_epoch_artifacts.pkl")
    if not os.path.exists(p):
        return None
    z = pickle.load(open(p, "rb"))["final_embedding"]
    z = z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z)
    d2r, sd = _dense2raw(ds)
    ei, ew = sd["edge_index"], sd["edge_weight"]
    trn, tst = sd["trn_mask"], sd["tst_mask"]

    def rows(mask):
        idx = ei[:, mask]; w = ew[mask]
        return idx[0].numpy(), idx[1].numpy(), (w.numpy() > 0).astype(int)
    tu, tv, ty = rows(trn)
    su, sv, sy = rows(tst)
    clf = linear_model.LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(np.concatenate([z[tu], z[tv]], axis=1), ty)
    pp = clf.predict_proba(np.concatenate([z[su], z[sv]], axis=1))[:, 1]
    u = [int(d2r[int(x)]) for x in su]
    v = [int(d2r[int(x)]) for x in sv]
    return {"u": u, "v": v, "y": sy.tolist(), "p": pp.tolist()}


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def main():
    canon = pickle.load(open(PKL, "rb"))
    print(f"{'dataset':14s} {'model':9s} {'cov_n':>7s} {'cov_auc':>8s} {'unc_n':>7s} "
          f"{'unc_auc':>8s} {'d_auc':>7s} {'cov_negR':>8s} {'unc_negR':>8s}")
    rows = []
    for ds in DATASETS:
        covered_uv = set(zip(canon[ds]["walk_full"]["u"], canon[ds]["walk_full"]["v"]))
        for name, fn in (("GINEConv", gineconv_raw), ("SiGAT", sigat_raw)):
            r = fn(ds)
            if r is None:
                continue
            u, v, y, p = r["u"], r["v"], np.asarray(r["y"]), np.asarray(r["p"])
            covmask = np.array([(uu, vv) in covered_uv for uu, vv in zip(u, v)])
            yc, pc = y[covmask], p[covmask]
            yu, pu = y[~covmask], p[~covmask]
            # y in {0,1}? GNN y is {0,1} (pos=1). neg rate = mean(y==0)
            cov_negR = float((yc == 0).mean()) if yc.size else float("nan")
            unc_negR = float((yu == 0).mean()) if yu.size else float("nan")
            a_c, a_u = auc(yc, pc), auc(yu, pu)
            print(f"{ds:14s} {name:9s} {yc.size:7d} {a_c:8.4f} {yu.size:7d} {a_u:8.4f} "
                  f"{a_c-a_u:7.4f} {cov_negR:8.3f} {unc_negR:8.3f}")
            rows.append(dict(dataset=ds, model=name, cov_n=int(yc.size), cov_auc=a_c,
                             unc_n=int(yu.size), unc_auc=a_u, cov_negR=cov_negR, unc_negR=unc_negR))
    import json
    json.dump(rows, open(os.path.join(ROOT, "outputs/walk_coverage_analysis/bias_phase0E.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
