"""Q5 pre-filter #1 (predictive validity): correlate each candidate hardness
map/metric against the REAL per-node error rate of an already-trained
checkpoint on held-out TEST edges. No retraining needed. Answers: which
candidate (if any) actually tracks genuine downstream difficulty?

Candidates screened:
  - Learned miner maps (accuracy/margin/brier/loss-based hardness, and D/R
    training-regime variants E17-E21 — see VARIANT_TAGS / SCREEN_VARIANTS).
  - Zero-training structural priors (degree, signed-ratio extremity) as a
    sanity baseline: if these correlate comparably to the learned miner, that
    calls into question whether the miner is adding anything.
"""
import sys, os, pickle, glob
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import torch
import numpy as np
from scipy.stats import spearmanr
from scripts.balance_theory_paths import load_edges_canonical

DATASETS = {
    "bitcoin-alpha": ("bitcoin-alpha", "dataset_cache__k_cover_k5_nw5000000_mw80_seed42.pt"),
    "bitcoin-otc": ("bitcoin-otc", "dataset_cache__k_cover_k5_nw1000000_mw80_seed42.pt"),
    "epinions": ("epinions", "dataset_cache__k_cover_k5_nw2000000_mw80_seed42.pt"),
    "wiki-elec": ("wiki-Elec", "dataset_cache__k_cover_k5_nw500000_mw80_seed42.pt"),
    "wiki-rfa": ("wiki-RfA", "dataset_cache__k_cover_k5_nw1000000_mw80_seed42.pt"),
    "slashdot090221": ("slashdot090221", "dataset_cache__k_cover_k5_nw3000000_mw80_seed42.pt"),
}

# Learned-miner training-regime variants (Q3: D/R axis), single hardness tensor each.
REGIME_VARIANT_TAGS = {
    "E17_static": "E17_HARDNODE_KCOVER_REMINE",
    "E18_dynpool": "E18_HARDNODE_DYNPOOL",
    "E19_static_R": "E19_HARDNODE_STATIC_R",
    "E20_dynpool_R": "E20_HARDNODE_DYNPOOL_R",
}

# Hardness-definition variants (Q5), all mined under the static (E17) recipe from
# ONE eval pass — dict with keys accuracy/margin/brier/loss (see --save-variants).
METRIC_VARIANTS_TAG = "E21_HARDNODE_METRIC_SCREEN"

MIN_OCC = 5  # min edge occurrences at a node for a stable error-rate estimate


def find_ckpt_run_dir(ds):
    cands = sorted(glob.glob(f"outputs/{ds}/E17_HARDNODE_KCOVER_REMINE_LOCALATTN4_*"))
    return cands[0] if cands else None


def load_test_predictions(run_dir, ds_key):
    pattern = os.path.join(run_dir, "checkpoints", f"{ds_key}_predictions", "epoch_*", "test_predictions.pkl")
    cands = sorted(glob.glob(pattern))
    if not cands:
        return None
    with open(cands[-1], "rb") as f:
        return pickle.load(f)


def structural_candidates(train_mask_edges, n_nodes_guess):
    """Zero-training candidate hardness signals from graph structure alone.
    hardness_degree = 1/(1+outdeg) (low-degree nodes = less evidence = harder)
    hardness_ratio_extremity = 1 - |2*pos_frac - 1| (balanced sign mix = harder)

    IMPORTANT: must be computed from TRAIN+MASK edges only (same info the miner/
    main model ever see) — using the full edge set (incl. VAL/TEST) would let a
    test edge's own sign leak into its own node's "structural hardness" feature,
    inflating the correlation especially for low-out-degree nodes.
    """
    out_pos = np.zeros(n_nodes_guess, dtype=np.float64)
    out_total = np.zeros(n_nodes_guess, dtype=np.float64)
    for u, v, label in train_mask_edges:
        out_total[u] += 1.0
        if label == 1:
            out_pos[u] += 1.0
    outdeg = out_total.copy()
    ratio = np.full(n_nodes_guess, np.nan)
    has_out = out_total > 0
    ratio[has_out] = out_pos[has_out] / out_total[has_out]
    hardness_degree = 1.0 / (1.0 + outdeg)
    hardness_ratio = np.full(n_nodes_guess, np.nan)
    hardness_ratio[has_out] = 1.0 - np.abs(2.0 * ratio[has_out] - 1.0)
    return {"struct_inv_outdeg": hardness_degree, "struct_ratio_extremity": hardness_ratio}


def main():
    results = {}
    for ds, (data_dir_name, cache_file) in DATASETS.items():
        run_dir = find_ckpt_run_dir(ds)
        if run_dir is None:
            print(f"{ds}: NO E17 LocalAttn4 run dir found, skip")
            continue
        pred = load_test_predictions(run_dir, ds)
        if pred is None:
            print(f"{ds}: no test_predictions.pkl found, skip")
            continue

        edges = load_edges_canonical(ds)  # [(u, v, label), ...] ordered, index = edge_id
        # Vectorized indexing (~50x faster than a per-occurrence Python list
        # comprehension over millions of test-prediction rows -- measured on epinions).
        edges_arr = np.asarray(edges, dtype=np.int64)
        edge_ids = pred["edge_ids"]
        correct = pred["correct"].astype(np.float64)

        n_nodes_guess = max(max(u, v) for u, v, _ in edges) + 1
        wrong_sum = np.zeros(n_nodes_guess, dtype=np.float64)
        total_cnt = np.zeros(n_nodes_guess, dtype=np.float64)

        valid = (edge_ids >= 0) & (edge_ids < len(edges))
        eids = edge_ids[valid].astype(np.int64)
        corr = correct[valid]
        us = edges_arr[eids, 0]
        vs = edges_arr[eids, 1]
        np.add.at(wrong_sum, us, 1.0 - corr)
        np.add.at(total_cnt, us, 1.0)
        np.add.at(wrong_sum, vs, 1.0 - corr)
        np.add.at(total_cnt, vs, 1.0)

        node_error = np.full(n_nodes_guess, np.nan)
        mask_stable = total_cnt >= MIN_OCC
        node_error[mask_stable] = wrong_sum[mask_stable] / total_cnt[mask_stable]
        raw_nodes = np.nonzero(mask_stable)[0]

        cache_path = f"data/{data_dir_name}/{cache_file}"
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        token2id = cache["tokenizer"]["token2id"]

        def correlate(hmap_by_raw_node):
            errs, hs = [], []
            for n in raw_nodes:
                h = hmap_by_raw_node.get(n) if isinstance(hmap_by_raw_node, dict) else None
                if h is None or (isinstance(h, float) and np.isnan(h)):
                    continue
                errs.append(node_error[n])
                hs.append(h)
            if len(errs) < 10:
                return None
            rho, p = spearmanr(errs, hs)
            return (rho, p, len(errs))

        def correlate_tokenized(hmap):
            errs, hs = [], []
            for n in raw_nodes:
                tok = token2id.get(f"N_{n}")
                if tok is None or tok >= len(hmap):
                    continue
                errs.append(node_error[n])
                hs.append(hmap[tok])
            if len(errs) < 10:
                return None
            rho, p = spearmanr(errs, hs)
            return (rho, p, len(errs))

        row = {"n_stable_nodes": int(mask_stable.sum())}

        # 1) Q3 training-regime variants (single accuracy-hardness tensor each)
        for vname, tag in REGIME_VARIANT_TAGS.items():
            hpath = f"outputs/{ds}/{tag}/hardness_map.pt"
            if not os.path.exists(hpath):
                row[vname] = None
                continue
            hd = torch.load(hpath, weights_only=False)
            hmap = (hd["hardness"] if isinstance(hd, dict) and "hardness" in hd else hd).numpy()
            row[vname] = correlate_tokenized(hmap)

        # 2) Q5 hardness-definition variants (accuracy/margin/brier/loss, static recipe)
        vpath = f"outputs/{ds}/{METRIC_VARIANTS_TAG}/hardness_map.pt.variants.pt"
        if os.path.exists(vpath):
            variants = torch.load(vpath, weights_only=False)
            for mname in ["accuracy", "margin", "brier", "loss"]:
                hmap = variants[mname].numpy()
                row[f"metric_{mname}"] = correlate_tokenized(hmap)
        else:
            for mname in ["accuracy", "margin", "brier", "loss"]:
                row[f"metric_{mname}"] = None

        # 3) Zero-training structural baselines (TRAIN+MASK only, see docstring)
        train_mask_edges = cache["splits"]["train"] + cache["splits"]["mask"]
        struct = structural_candidates(train_mask_edges, n_nodes_guess)
        for sname, arr in struct.items():
            d = {n: arr[n] for n in raw_nodes}
            row[sname] = correlate(d)

        results[ds] = row

    all_cols = (
        list(REGIME_VARIANT_TAGS.keys())
        + [f"metric_{m}" for m in ["accuracy", "margin", "brier", "loss"]]
        + ["struct_inv_outdeg", "struct_ratio_extremity"]
    )
    print(f"\n{'dataset':<16}{'n_stable':>10}" + "".join(f"{c:>22}" for c in all_cols))
    for ds, row in results.items():
        cells = []
        for c in all_cols:
            r = row.get(c)
            cells.append(f"{r[0]:.4f}" if r else "NA")
        print(f"{ds:<16}{row['n_stable_nodes']:>10}" + "".join(f"{c:>22}" for c in cells))

    return results, all_cols


if __name__ == "__main__":
    main()
