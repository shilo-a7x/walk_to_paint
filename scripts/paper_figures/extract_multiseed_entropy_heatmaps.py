"""Extract step for two figures agreed with the user (2026-08-10):

1. SiGAT entropy-vs-AUC heatmap, mean+-std over 10 splits (Option 2: each split's
   cell AUC computed independently on its own ~10% test slice, then averaged --
   NOT pooled test predictions; picked explicitly by the user as "much simpler and
   reliable" over pooling, see CLAUDE.md "Entropy-heatmap multi-split methodology").
   Same fixed 4x4 entropy bins as the existing single-split
   extract_empconf_panelC_gnn_entropy_heatmap.py.
2. PEWTER-vs-SiGAT delta heatmap: same bins, same per-split-then-average method,
   delta(i,j) = mean_pewter_auc(i,j) - mean_sigat_auc(i,j).

Both models' per-seed per-edge predictions are pulled fresh here (not from the old
single-split outputs/lead4_entropy_heterogeneity/computed_data.pkl):
- SiGAT: baselines/SGA/results_our_splits_canonical/<ds>/SiGAT/seed<N>/
  best_epoch_artifacts.pkl (fit a fresh LogisticRegression on that seed's own train
  split, exactly mirroring baselines/postprocess_canonical.py::sigat_raw, just
  parameterized over seed).
- PEWTER (local attention): outputs/<ds>/MULTISEED_s<N>_local_*/checkpoints/
  <ds>_predictions/epoch_*/test_predictions.pkl (scripts/run_multiseed_pewter.py's
  output), mean-probability aggregation over each edge's walk occurrences (same
  approximation baselines/postprocess_canonical.py::walk_raw uses -- within ~0.003 of
  func_logit_power per that script's own note).

Entropy values (src_ent, tgt_ent) are computed ONCE per dataset from the fixed real
dense edge set (split-independent, confirmed in CLAUDE.md) via
lead4_entropy_heterogeneity.py's own build_sign_dicts/collect_model_records, reused
unchanged across all 10 seeds -- only the per-seed (y, p) test predictions differ.
"""
import csv
import glob
import os
import pickle
import sys

import numpy as np
from sklearn import linear_model
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baselines"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from baselines.postprocess_canonical import _DATA_DIR, _edgeid_to_uv  # noqa: E402
from scripts.attention_directionality import LOCAL_RUN_INFO  # noqa: E402
from scripts.lead4_entropy_heterogeneity import build_sign_dicts, collect_model_records  # noqa: E402
from scripts.node_mi_structural_embedding import DATASET_CONFIGS, load_edges_canonical  # noqa: E402

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
SEEDS = [42] + list(range(43, 52))
VARIANT = "out_in"  # same atomic-entropy axes as empconf_panelC (src_out, tgt_in)
N_BINS = 4
MIN_CELL_N_SIGAT_HEATMAP = 30  # standalone SiGAT-alone heatmap -- unchanged, not in the paper
MIN_CELL_N_DELTA = 50  # PEWTER-vs-SiGAT delta heatmap -- bumped 2026-08-11 per the user's call

# Cache of the expensive per-seed entropy-joined (u,v,y,p,src_ent,tgt_ent) records --
# added 2026-08-11 per the user's standing rule ("every plot we do we need to save the
# data so if we just want to change colors or threshold... we dont need to do the whole
# thing again"). This is the part that's actually slow: SiGAT refits a fresh
# LogisticRegression per seed (60 fits total) and PEWTER reloads/re-aggregates large
# prediction pickles per seed -- neither depends on N_BINS/min_cell_n, so bin-edge or
# threshold changes (like the MIN_CELL_N_DELTA bump above) never need to redo this part
# again once cached. Delete a cache file to force a real recompute for that
# dataset/model (e.g. after a checkpoint changes) -- same "safe to delete/regenerate"
# convention as the figure_data/*.csv outputs.
CACHE_DIR = os.path.join(ROOT, "outputs", "cache", "multiseed_entropy_records")

# Production num_walks budget per dataset (CLAUDE.md "Walk sampler" table) -- needed
# to build each seed's keyed cache filename.
NUM_WALKS = {
    "bitcoin-alpha": 120930, "bitcoin-otc": 177960, "epinions": 840799,
    "slashdot090221": 1647606, "wiki-elec": 155534, "wiki-rfa": 265817,
}

OUT_SIGAT_CSV = os.path.join(ROOT, "aaai2027", "figure_data", "empconf_panelC_sigat_10split.csv")
OUT_DELTA_CSV = os.path.join(ROOT, "aaai2027", "figure_data", "pewter_sigat_delta_heatmap.csv")


def _ds_key(ds_name):
    return "slashdot" if ds_name == "slashdot090221" else ds_name


def sigat_raw_seed(ds, seed):
    """Mirrors baselines/postprocess_canonical.py::sigat_raw, parameterized over seed."""
    art_path = os.path.join(ROOT, "baselines", "SGA", "results_our_splits_canonical",
                             ds, "SiGAT", f"seed{seed}", "best_epoch_artifacts.pkl")
    if not os.path.exists(art_path):
        return None
    z = pickle.load(open(art_path, "rb"))["final_embedding"]
    z = z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z)

    if seed == 42:
        split_path = os.path.join(ROOT, "baselines", "splits_canonical", f"{ds}.pt")
    else:
        split_path = os.path.join(ROOT, "baselines", f"splits_canonical_seed{seed}", f"{ds}.pt")
    if not os.path.exists(split_path):
        return None
    sd = __import__("torch").load(split_path, weights_only=False)
    d2r = sd["dense2raw"]
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


def walk_raw_seed(ds, seed, variant="local"):
    """Mirrors baselines/postprocess_canonical.py::walk_raw, parameterized over seed.
    seed=42 predates the MULTISEED_* naming (it's the original, already-adopted
    E32_PY314_LOCALATTN4 checkpoint, "reused/backfilled" per run_multiseed_pewter.py's
    own docstring) -- pull it from LOCAL_RUN_INFO's pinned (run_dir, epoch) instead of
    globbing for a MULTISEED_s42_local_* dir that doesn't exist."""
    if seed == 42:
        if variant != "local" or ds not in LOCAL_RUN_INFO:
            return None
        run_dir_name, epoch = LOCAL_RUN_INFO[ds]
        pred_pkl = os.path.join(ROOT, "outputs", ds, run_dir_name, "checkpoints",
                                 f"{ds}_predictions", f"epoch_{epoch:03d}", "test_predictions.pkl")
        if not os.path.exists(pred_pkl):
            return None
    else:
        run_glob = os.path.join(ROOT, "outputs", ds, f"MULTISEED_s{seed}_{variant}_*")
        run_dirs = sorted(glob.glob(run_glob))
        if not run_dirs:
            return None
        pred_pkl = None
        for run_dir in reversed(run_dirs):
            g = glob.glob(os.path.join(run_dir, "checkpoints", "*_predictions",
                                        "epoch_*", "test_predictions.pkl"))
            if g:
                pred_pkl = sorted(g)[-1]
                break
        if pred_pkl is None:
            return None

    nw = NUM_WALKS[ds]
    cache = os.path.join(ROOT, _DATA_DIR[ds], f"dataset_cache__edge_cover_nw{nw}_mw80_seed{seed}.pt")
    if not os.path.exists(cache):
        return None

    d = pickle.load(open(pred_pkl, "rb"))
    eid = d["edge_ids"]; p = d["probabilities"][:, 1].astype(float); y = d["targets"]
    o = np.argsort(eid); e = eid[o]; p = p[o]; y = y[o]
    uq, st = np.unique(e, return_index=True); en = np.append(st[1:], len(e))
    mp = np.array([p[st[i]:en[i]].mean() for i in range(len(uq))]); yy = y[st]
    e2uv = _edgeid_to_uv(cache)
    U, V, Y, P = [], [], [], []
    for i in range(len(uq)):
        k = int(uq[i])
        if k in e2uv:
            uu, vv = e2uv[k]
            U.append(uu); V.append(vv); Y.append(int(yy[i])); P.append(float(mp[i]))
    return {"u": U, "v": V, "y": Y, "p": P}


def binned_auc_grid(src_ent, tgt_ent, y, p, n_bins, min_cell_n):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    src_bin = np.clip(np.digitize(src_ent, edges[1:-1]), 0, n_bins - 1)
    tgt_bin = np.clip(np.digitize(tgt_ent, edges[1:-1]), 0, n_bins - 1)
    y_bin = (np.asarray(y) > 0).astype(int)

    auc_grid = np.full((n_bins, n_bins), np.nan)
    n_grid = np.zeros((n_bins, n_bins), dtype=int)
    for i in range(n_bins):
        for j in range(n_bins):
            mask = (src_bin == i) & (tgt_bin == j)
            n = int(mask.sum())
            n_grid[i, j] = n
            if n < min_cell_n or len(set(y_bin[mask])) < 2:
                continue
            auc_grid[i, j] = roc_auc_score(y_bin[mask], p[mask])
    return auc_grid, n_grid, edges


def per_seed_records(ds, sign_dicts, model_loader, model_name):
    """Load + entropy-join each seed's predictions ONCE (the expensive part --
    pickle/artifact loading, LogisticRegression refit for SiGAT). Returns the raw
    per-seed entropy-joined records so they can be binned at more than one
    min_cell_n threshold without redoing this work. Cached to disk (CACHE_DIR) so a
    second run (e.g. to change min_cell_n or bin edges) skips this entirely."""
    cache_path = os.path.join(CACHE_DIR, f"{ds}__{model_name}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            records = pickle.load(f)
        print(f"    {model_name}: loaded {len(records)} cached seed records from {cache_path}")
        return records

    records = []
    for seed in SEEDS:
        rec = model_loader(ds, seed)
        if rec is None or not len(rec["u"]):
            print(f"    {model_name} seed={seed}: MISSING")
            continue
        test_uvyp = list(zip(rec["u"], rec["v"], rec["y"], rec["p"]))
        entry = collect_model_records(test_uvyp, sign_dicts, VARIANT, min_total=1)
        if entry is None:
            print(f"    {model_name} seed={seed}: too few usable edges after entropy join")
            continue
        records.append(entry)
        print(f"    {model_name} seed={seed}: n={len(entry['y'])}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(records, f)
    return records


def grids_from_records(records, min_cell_n):
    """Bin already-loaded per-seed records at a given min_cell_n threshold (cheap --
    no re-loading/re-fitting, just re-thresholding the same predictions)."""
    if not records:
        return None, None, None
    grids, n_grids, edges = [], [], None
    for entry in records:
        auc_grid, n_grid, edges = binned_auc_grid(
            entry["src_ent"], entry["tgt_ent"], entry["y"], entry["p"], N_BINS, min_cell_n)
        grids.append(auc_grid)
        n_grids.append(n_grid)
    # mean cell sample size over valid (n>=min_cell_n) splits only, to match what
    # actually went into the mean AUC -- splits below min_cell_n contribute NaN to
    # auc_grid and shouldn't inflate the reported "n" for that cell.
    auc_stack = np.stack(grids, axis=0)
    n_stack = np.stack(n_grids, axis=0).astype(float)
    n_stack[~np.isfinite(auc_stack)] = np.nan
    mean_n = np.nanmean(n_stack, axis=0)
    return auc_stack, edges, mean_n


def main():
    sigat_rows, delta_rows = [], []
    for ds in DATASETS:
        print(f"\n{'=' * 78}\n{ds}\n{'=' * 78}")
        edges_list = load_edges_canonical(DATASET_CONFIGS[_ds_key(ds)]["ds_name"])
        sign_dicts = build_sign_dicts(edges_list)

        sigat_records = per_seed_records(ds, sign_dicts, sigat_raw_seed, "SiGAT")
        walk_records = per_seed_records(
            ds, sign_dicts, lambda d, s: walk_raw_seed(d, s, "local"), "PEWTER(local)")

        # Standalone SiGAT-alone heatmap: unchanged threshold (MIN_CELL_N_SIGAT_HEATMAP).
        sigat_stack, bin_edges, sigat_mean_n = grids_from_records(sigat_records, MIN_CELL_N_SIGAT_HEATMAP)
        if sigat_stack is None:
            print(f"  ✗ no usable SiGAT splits for {ds}, skipping")
            continue
        sigat_mean = np.nanmean(sigat_stack, axis=0)
        sigat_std = np.nanstd(sigat_stack, axis=0, ddof=1)
        sigat_n = np.sum(np.isfinite(sigat_stack), axis=0)

        # Delta plot: higher threshold (MIN_CELL_N_DELTA), per the user's call --
        # only the delta plot's reliability bar was raised, not the SiGAT-alone one.
        sigat_stack_d, _, sigat_mean_n_d = grids_from_records(sigat_records, MIN_CELL_N_DELTA)
        walk_stack, _, walk_mean_n = grids_from_records(walk_records, MIN_CELL_N_DELTA)
        if sigat_stack_d is not None:
            sigat_mean_d = np.nanmean(sigat_stack_d, axis=0)
            sigat_n_d = np.sum(np.isfinite(sigat_stack_d), axis=0)

        for i in range(N_BINS):
            for j in range(N_BINS):
                sigat_rows.append({
                    "dataset": ds,
                    "src_bin_lo": bin_edges[i], "src_bin_hi": bin_edges[i + 1],
                    "tgt_bin_lo": bin_edges[j], "tgt_bin_hi": bin_edges[j + 1],
                    "mean_auc": sigat_mean[i, j], "std_auc": sigat_std[i, j],
                    "n_splits_valid": int(sigat_n[i, j]),
                    "mean_cell_n": (round(float(sigat_mean_n[i, j]))
                                    if np.isfinite(sigat_mean_n[i, j]) else ""),
                })

        if walk_stack is not None and sigat_stack_d is not None:
            walk_mean = np.nanmean(walk_stack, axis=0)
            walk_n = np.sum(np.isfinite(walk_stack), axis=0)
            for i in range(N_BINS):
                for j in range(N_BINS):
                    both_valid = np.isfinite(walk_mean[i, j]) and np.isfinite(sigat_mean_d[i, j])
                    delta_rows.append({
                        "dataset": ds,
                        "src_bin_lo": bin_edges[i], "src_bin_hi": bin_edges[i + 1],
                        "tgt_bin_lo": bin_edges[j], "tgt_bin_hi": bin_edges[j + 1],
                        "pewter_mean_auc": walk_mean[i, j], "sigat_mean_auc": sigat_mean_d[i, j],
                        "delta": (walk_mean[i, j] - sigat_mean_d[i, j]) if both_valid else np.nan,
                        "n_splits_valid_pewter": int(walk_n[i, j]),
                        "n_splits_valid_sigat": int(sigat_n_d[i, j]),
                        "mean_cell_n_pewter": (round(float(walk_mean_n[i, j]))
                                                if np.isfinite(walk_mean_n[i, j]) else ""),
                        "mean_cell_n_sigat": (round(float(sigat_mean_n_d[i, j]))
                                               if np.isfinite(sigat_mean_n_d[i, j]) else ""),
                    })
        else:
            print(f"  ✗ no usable PEWTER(local)/SiGAT(min_n={MIN_CELL_N_DELTA}) splits for {ds} "
                  f"-- delta heatmap skipped for this dataset")

    os.makedirs(os.path.dirname(OUT_SIGAT_CSV), exist_ok=True)
    with open(OUT_SIGAT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "src_bin_lo", "src_bin_hi", "tgt_bin_lo",
                                           "tgt_bin_hi", "mean_auc", "std_auc", "n_splits_valid",
                                           "mean_cell_n"])
        w.writeheader(); w.writerows(sigat_rows)
    print(f"\nwrote {OUT_SIGAT_CSV} ({len(sigat_rows)} rows)")

    if delta_rows:
        with open(OUT_DELTA_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "src_bin_lo", "src_bin_hi", "tgt_bin_lo",
                                               "tgt_bin_hi", "pewter_mean_auc", "sigat_mean_auc",
                                               "delta", "n_splits_valid_pewter", "n_splits_valid_sigat",
                                               "mean_cell_n_pewter", "mean_cell_n_sigat"])
            w.writeheader(); w.writerows(delta_rows)
        print(f"wrote {OUT_DELTA_CSV} ({len(delta_rows)} rows)")


if __name__ == "__main__":
    main()
