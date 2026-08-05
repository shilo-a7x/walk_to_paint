"""Extract fresh (src_ent, tgt_ent, y, p) records for BOTH current production
walk-model variants -- full attention (E31_PY314_MIGRATION) and LocalAttn4
(E32_PY314_LOCALATTN4), post-migration -- for Result 2's dedicated 4-row
heatmap (Pewter full, Pewter local, SiGAT, GINEConv). Sibling of
extract_walk_entropy_fresh.py (which only does LocalAttn4, for Panel C's
now-reverted walk column); this one covers both variants since Result 2 shows
all 4 rows even though 2 of them (the GNNs) duplicate Panel C.

Same pipeline as extract_walk_entropy_fresh.py (see that file for the full
method writeup): aggregate raw per-occurrence test predictions with the
already-fit func_logit_power weights, sanity-check the aggregated AUC against
each run's own saved summary.txt, then recover each occurrence's (u, v) via a
fresh deterministic get_edge_list(cfg) call and compute out/in-entropy from
that same full edge list.

**Updated 2026-08-04** (post-migration rebuild): both RUN_INFO dicts repointed
from the pre-migration E25/E26 (full) and E27 (LocalAttn4) checkpoints to the
post-migration E31_PY314_MIGRATION (full -- adopted after it turned out to be
mislabeled, see CLAUDE.md "Current SOTA") and E32_PY314_LOCALATTN4 (LocalAttn4)
checkpoints. Both used a uniform posthoc run-id (E31_posthoc/E32_posthoc) this
time, not the old per-dataset x5/floor/x3/p1_5x scheme. slashdot090221's
LocalAttn4 entry is temporarily omitted below -- its E32 training is still in
progress; add it back once its posthoc completes.
"""
import csv
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.utils.config import load_config
from src.data.prepare_data import get_edge_list

OUT_CSV = "aaai2027/figure_data/result2_walk_entropy_fresh.csv"
_FUNC_EPS = 1e-9

# (run_dir, epoch, posthoc_run_id) per dataset, per variant
FULL_RUN_INFO = {
    "bitcoin-alpha":  ("E31_PY314_MIGRATION_20260802-130454", 35, "E31_posthoc"),
    "bitcoin-otc":    ("E31_PY314_MIGRATION_20260802-135051", 46, "E31_posthoc"),
    "epinions":       ("E31_PY314_MIGRATION_20260802-130454", 41, "E31_posthoc"),
    "wiki-elec":      ("E31_PY314_MIGRATION_20260802-135051", 37, "E31_posthoc"),
    "wiki-rfa":       ("E31_PY314_MIGRATION_20260802-135051", 33, "E31_posthoc"),
    "slashdot090221": ("E31_PY314_MIGRATION_20260802-130454", 51, "E31_posthoc"),
}
LOCAL_RUN_INFO = {
    "bitcoin-alpha":   ("E32_PY314_LOCALATTN4_20260804-225948", 35, "E32_posthoc"),
    "bitcoin-otc":     ("E32_PY314_LOCALATTN4_20260804-231122", 35, "E32_posthoc"),
    "epinions":        ("E32_PY314_LOCALATTN4_20260804-225948", 30, "E32_posthoc"),
    "wiki-elec":       ("E32_PY314_LOCALATTN4_20260804-232140", 44, "E32_posthoc"),
    "wiki-rfa":        ("E32_PY314_LOCALATTN4_20260804-232239", 33, "E32_posthoc"),
    "slashdot090221":  ("E32_PY314_LOCALATTN4_20260804-225948", 34, "E32_posthoc"),
}
VARIANTS = [("Pewter (full attn)", FULL_RUN_INFO), ("Pewter (LocalAttn4)", LOCAL_RUN_INFO)]


def logit_power_weight(theta, q):
    logit = np.log(np.maximum(q, _FUNC_EPS) / np.maximum(1.0 - q, _FUNC_EPS))
    return np.power(np.abs(logit) + _FUNC_EPS, theta)


def binary_entropy(signs):
    signs = np.asarray(signs)
    p = float((signs > 0).mean())
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


_ENTROPY_CACHE = {}


def get_entropy_dicts(ds):
    if ds not in _ENTROPY_CACHE:
        cfg = load_config("config.yaml", overrides=[f"dataset.name={ds}"])
        edges = get_edge_list(cfg)
        out_signs, in_signs = {}, {}
        for u, v, s in edges:
            out_signs.setdefault(u, []).append(s)
            in_signs.setdefault(v, []).append(s)
        out_ent = {n: binary_entropy(s) for n, s in out_signs.items()}
        in_ent = {n: binary_entropy(s) for n, s in in_signs.items()}
        _ENTROPY_CACHE[ds] = (edges, out_ent, in_ent)
    return _ENTROPY_CACHE[ds]


def main():
    rows = []
    for model_name, run_info in VARIANTS:
        for ds, (run_dir, epoch, run_id) in run_info.items():
            base = f"outputs/{ds}/{run_dir}"
            with open(f"{base}/posthoc/{run_id}/aggregator/func_logit_power/model_config.json") as f:
                theta = json.load(f)["theta_star"][0]
            with open(f"{base}/checkpoints/{ds}_predictions/epoch_{epoch:03d}/test_predictions.pkl", "rb") as f:
                import pickle
                preds = pickle.load(f)

            edge_ids = np.asarray(preds["edge_ids"]).astype(np.int64)
            q = preds["probabilities"][:, 1].astype(float)
            y_occ = np.asarray(preds["targets"]).astype(int)

            order = np.argsort(edge_ids, kind="stable")
            eids_s, q_s, y_s = edge_ids[order], q[order], y_occ[order]
            uniq, inv_idx, counts = np.unique(eids_s, return_inverse=True, return_counts=True)
            n_e = len(uniq)
            w_s = logit_power_weight(theta, q_s)
            wsum = np.bincount(inv_idx, weights=w_s, minlength=n_e)
            wqsum = np.bincount(inv_idx, weights=w_s * q_s, minlength=n_e)
            edge_scores = wqsum / np.maximum(wsum, _FUNC_EPS)
            first_occ = np.concatenate([[0], np.cumsum(counts)[:-1]])
            edge_labels = y_s[first_occ].astype(int)

            agg_auc = roc_auc_score(edge_labels, edge_scores)
            print(f"{model_name} / {ds}: aggregated test AUC = {agg_auc:.4f} (n_edges={n_e})")

            edges, out_ent, in_ent = get_entropy_dicts(ds)
            n_missing = 0
            for eid, score, label in zip(uniq, edge_scores, edge_labels):
                u, v, _lbl = edges[int(eid)]
                se, te = out_ent.get(u), in_ent.get(v)
                if se is None or te is None:
                    n_missing += 1
                    continue
                rows.append({
                    "dataset": ds, "model": model_name, "src_ent": se, "tgt_ent": te,
                    "y": int(label > 0), "p": float(score),
                })
            if n_missing:
                print(f"  ({n_missing} edges dropped -- endpoint missing from entropy dict)")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["dataset", "model", "src_ent", "tgt_ent", "y", "p"])
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"saved {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
