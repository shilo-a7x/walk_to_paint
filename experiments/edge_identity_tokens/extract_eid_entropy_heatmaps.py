"""EID (H3) port of scripts/paper_figures/extract_multiseed_entropy_heatmaps.py: the entropy-binned
AUC heatmap (src_out entropy x tgt_in entropy, 4x4 fixed bins, per-split-then-average over seeds
42-51), for H3, production Pewter (local attention) and SiGAT side by side, plus H3-SiGAT and
H3-production deltas per cell.

Reuses the production module unchanged (binning, entropy join, SiGAT refit, record cache); only the
walk-model prediction source differs: H3's test_predictions.pkl from the thesis campaign
(EIDREG_H3_<DS>_s<seed>). EID caches are built from production's per-seed dataset cache, so edge ids
map to (u, v) through the same dataset_cache__edge_cover_nw<N>_mw80_seed<seed>.pt.

Output: experiments/edge_identity_tokens/thesis_figure_data/eid_entropy_heatmaps.csv
Usage:  .venv/bin/python experiments/edge_identity_tokens/extract_eid_entropy_heatmaps.py
"""
import csv
import glob
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper_figures"))
import extract_multiseed_entropy_heatmaps as base  # noqa: E402

OUT_CSV = os.path.join(ROOT, "experiments", "edge_identity_tokens", "thesis_figure_data", "eid_entropy_heatmaps.csv")


def h3_raw_seed(ds, seed):
    tag = f"EIDREG_H3_{ds.upper().replace('-', '')}_s{seed}_"
    dirs = sorted(glob.glob(os.path.join(ROOT, "outputs", ds, f"{tag}*")))
    preds = sorted(glob.glob(os.path.join(dirs[-1], "checkpoints", "*_predictions", "epoch_*",
                                          "test_predictions.pkl"))) if dirs else []
    if not preds:
        return None
    cache = os.path.join(ROOT, base._DATA_DIR[ds], f"dataset_cache__edge_cover_nw{base.NUM_WALKS[ds]}_mw80_seed{seed}.pt")
    d = pickle.load(open(preds[-1], "rb"))
    eid = d["edge_ids"]
    p = d["probabilities"][:, 1].astype(float)
    y = d["targets"]
    o = np.argsort(eid)
    e, p, y = eid[o], p[o], y[o]
    uq, st = np.unique(e, return_index=True)
    en = np.append(st[1:], len(e))
    mp = np.array([p[st[i]:en[i]].mean() for i in range(len(uq))])  # mean-prob, same as production port
    e2uv = base._edgeid_to_uv(cache)
    out = {"u": [], "v": [], "y": [], "p": []}
    for i, k in enumerate(uq):
        if int(k) in e2uv:
            u, v = e2uv[int(k)]
            out["u"].append(u); out["v"].append(v); out["y"].append(int(y[st[i]])); out["p"].append(float(mp[i]))
    return out


def main():
    rows = []
    for ds in base.DATASETS:
        print(f"\n=== {ds} ===", flush=True)
        sign_dicts = base.build_sign_dicts(base.load_edges_canonical(base.DATASET_CONFIGS[base._ds_key(ds)]["ds_name"]))
        recs = {
            "h3": base.per_seed_records(ds, sign_dicts, h3_raw_seed, "EID_H3"),
            "production": base.per_seed_records(ds, sign_dicts, lambda d, s: base.walk_raw_seed(d, s, "local"),
                                                "PEWTER(local)"),
            "sigat": base.per_seed_records(ds, sign_dicts, base.sigat_raw_seed, "SiGAT"),
        }
        grids = {}
        for k, r in recs.items():
            stack, edges, mean_n = base.grids_from_records(r, base.MIN_CELL_N_DELTA)
            if stack is not None:
                grids[k] = (np.nanmean(stack, 0), np.nanstd(stack, 0, ddof=1), np.isfinite(stack).sum(0), mean_n, edges)
        if "h3" not in grids:
            continue
        edges = grids["h3"][4]
        for i in range(base.N_BINS):
            for j in range(base.N_BINS):
                row = {"dataset": ds, "src_bin_lo": edges[i], "src_bin_hi": edges[i + 1],
                       "tgt_bin_lo": edges[j], "tgt_bin_hi": edges[j + 1]}
                for k, (m, s, n, mn, _) in grids.items():
                    row.update({f"{k}_mean_auc": m[i, j], f"{k}_std_auc": s[i, j], f"{k}_n_splits": int(n[i, j]),
                                f"{k}_mean_cell_n": round(float(mn[i, j])) if np.isfinite(mn[i, j]) else ""})
                for a, b in (("h3", "sigat"), ("h3", "production"), ("production", "sigat")):
                    if a in grids and b in grids:
                        row[f"delta_{a}_minus_{b}"] = grids[a][0][i, j] - grids[b][0][i, j]
                rows.append(row)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
