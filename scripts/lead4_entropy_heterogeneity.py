"""
Lead 4 -- node sign-entropy heterogeneity vs. AUC.

Hypothesis: all models (walk transformer, GINEConv, SiGAT) do well when both
endpoints of a test edge sit in a locally homogeneous (low sign-entropy)
neighborhood, but GNNs degrade more than the walk model as endpoint sign
heterogeneity (entropy) rises -- i.e. the walk model's advantage is
concentrated in the high-entropy region.

For each test edge (u, v): bin source-node sign-entropy x target-node
sign-entropy into a 2D grid and compute AUC within each cell, for 4 models:
  - walk_full        (E14_HARDNODE_L10, full attention)
  - walk_localattn4  (E14_HARDNODE_L10_LOCALATTN4, +-2 hop banded attention)
  - GINEConv         (baselines/GINEConv, dense re-indexed ids)
  - SiGAT            (baselines/SGA,      dense re-indexed ids)

Entropy is computed from ALL edges of the dataset (train+val+test), not just
train. This is purely a diagnostic/evaluation grouping of *already-trained*
models' predictions -- it does not feed anything back into training, so
there is no train/test leakage concern in the sense that matters (no
information flows into a model's weights or selection). Using all edges
gives a less noisy, more representative estimate of each node's "true"
sign-heterogeneity than train-only counts would, especially for low-degree
nodes. (Contrast with e.g. lead3_attention_ambiguity.py's ambiguity score,
which IS read by the model indirectly via training dynamics and therefore
must stay train-only to avoid leaking the val/test answer into a "model-free"
diagnostic that's interpreted as if it were train-time-only information.)

No retraining: walk-model predictions come from existing
*_predictions/epoch_*/test_predictions.pkl artifacts (edge_ids decoded back
to (u, v, sign) via the canonical edge list, same trick as
lead1_degree_gap.py / balance_theory_paths.py). A single edge is visited by
MANY walks, so test_predictions.pkl has several per-walk probabilities per
edge_id -- these must be combined with the SAME edge-level aggregator used
for the project's reported SOTA numbers (per CLAUDE.md: func_logit_power,
weighted mean of per-walk q by |logit(q)|^theta), not a naive unweighted
mean. theta* is NOT refit here -- it's read straight from the already-run
run_posthoc.py aggregator artifact's model_config.json (theta_star), found
under <exp_dir>/runs/<ds>/E14_HARDNODE_L10/posthoc/*epoch=<N>-*enc_fixed*/
aggregator/func_logit_power/ for the full-attention model, and
outputs/<ds>/E14_HARDNODE_L10_LOCALATTN4_*/posthoc/localattn4_posthoc/
aggregator/func_logit_power/ for LocalAttn4 -- see find_func_logit_power_theta().
GINEConv predictions come
straight from its best_epoch_artifacts.pkl (pred_p/y/edge_index already on
the test set). SiGAT has no saved predictions, only final_embedding -- its
test-set predictions are reconstructed by refitting the same sklearn
LogisticRegression on concatenated embeddings that baselines/SGA/
run_with_our_splits.py's own evaluate() used (deterministic, lbfgs solver,
no retraining of SiGAT itself; this classifier fit still uses train/test
masks as originally designed -- that split is about predicting unseen edges,
unrelated to the entropy-grouping question above).

Entropy variants (4): for each test edge (u, v), source/target entropy can
each be computed from OUT-edges, IN-edges, or IN+OUT (combined) signs of that
node, over ALL edges of the dataset:
  out_out      : H(out-signs of u),       H(out-signs of v)
  in_in        : H(in-signs of u),        H(in-signs of v)
  out_in       : H(out-signs of u),       H(in-signs of v)   [most "causal":
                 u's own outgoing voting pattern vs. v's incoming reputation]
  inout_inout  : H(all incident signs of u), H(all incident signs of v)

Binary Shannon entropy in bits: H(p) = -p*log2(p) - (1-p)*log2(1-p), p =
fraction of positive-sign edges in the relevant set. H=0 -> perfectly
homogeneous (all same sign), H=1 -> maximally heterogeneous (50/50 split).
Nodes with zero edges in the relevant direction are dropped (no entropy
signal available).

Compute/plot are separable so you can iterate on plot styling (colormap,
binning resolution, vmin/vmax, ...) without re-loading model artifacts /
refitting SiGAT's classifier each time:

    # compute raw per-edge (entropy, y, p) records once, save to disk
    python scripts/lead4_entropy_heterogeneity.py --mode compute --datasets all

    # replot from saved data, as many times as you like, with new styling
    python scripts/lead4_entropy_heterogeneity.py --mode plot --cmap viridis \
        --n-buckets 5 --min-cell-n 15

    # default: do both in one go (same behavior as before this split existed)
    python scripts/lead4_entropy_heterogeneity.py --mode all --datasets all

`--mode compute` (and `all`) also saves the full per-edge raw predictions for
every (dataset, model) -- independent of entropy variant -- to
`predictions_raw.pkl`: {ds: {model: {"u","v","y","p"}}}, node ids in that
model's own id space (raw tokenizer ids for walk models, dense
baselines/splits ids for GINEConv/SiGAT). Kept around for any future
investigation beyond this specific entropy question, so model artifacts
don't need reloading / SiGAT's classifier doesn't need refitting again.
"""
import argparse
import glob
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import linear_model
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.node_mi_structural_embedding import (
    DATASET_CONFIGS, load_edges_canonical,
)

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
MODELS = ["walk_full", "walk_localattn4", "GINEConv", "SiGAT"]
VARIANTS = {
    "out_out":     ("out", "out"),
    "in_in":       ("in", "in"),
    "out_in":      ("out", "in"),
    "inout_inout": ("inout", "inout"),
}
N_BUCKETS_DEFAULT = 4
MIN_CELL_N_DEFAULT = 20
MIN_TOTAL_DEFAULT = 40  # drop a (dataset, variant, model) entirely below this many usable edges
OUT_DIR_DEFAULT = os.path.join(ROOT, "outputs", "lead4_entropy_heterogeneity")
DATA_FILENAME = "computed_data.pkl"
PREDICTIONS_FILENAME = "predictions_raw.pkl"


def _ds_key(ds_name):
    """balance_theory_paths/node_mi_structural_embedding.py key slashdot090221
    by 'slashdot', not its own ds_name -- mirrors lead1_degree_gap.py."""
    return "slashdot" if ds_name == "slashdot090221" else ds_name


# ── Entropy machinery ──────────────────────────────────────────────────────────

def binary_entropy(signs):
    signs = np.asarray(signs)
    p = float((signs > 0).mean())
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def build_sign_dicts(edge_triples):
    """edge_triples: iterable of (u, v, sign). Returns (out_signs, in_signs,
    inout_signs), each {node: list of +-1 signs}."""
    out_signs, in_signs, inout_signs = {}, {}, {}
    for u, v, s in edge_triples:
        out_signs.setdefault(u, []).append(s)
        inout_signs.setdefault(u, []).append(s)
        in_signs.setdefault(v, []).append(s)
        inout_signs.setdefault(v, []).append(s)
    return out_signs, in_signs, inout_signs


def entropy_lookup(sign_dicts):
    """sign_dicts: (out_signs, in_signs, inout_signs) -> dict direction ->
    {node: entropy}."""
    out_signs, in_signs, inout_signs = sign_dicts
    return {
        "out": {n: binary_entropy(s) for n, s in out_signs.items()},
        "in": {n: binary_entropy(s) for n, s in in_signs.items()},
        "inout": {n: binary_entropy(s) for n, s in inout_signs.items()},
    }


# ── Per-model data loaders: each returns (test_uvyp, sign_dicts) ──────────────
# test_uvyp: list of (u, v, y, p) -- y in {0,1} (1 = positive sign), p =
# predicted P(sign=+1). sign_dicts: ALL-edges (out_signs, in_signs,
# inout_signs) in THIS model's own node-id space.

def load_walk_pkl(ds_name, variant):
    cfg = DATASET_CONFIGS[_ds_key(ds_name)]
    if variant == "full":
        epoch = cfg["best_epoch"]
        exp_dir = os.path.join(ROOT, cfg["exp_dir"])
        pattern = os.path.join(exp_dir, "runs", cfg["ds_name"], "E14_HARDNODE_L10",
                                "checkpoints", f"{cfg['ds_name']}_predictions",
                                f"epoch_{epoch:03d}", "test_predictions.pkl")
    else:
        run_dirs = sorted(glob.glob(os.path.join(
            ROOT, "outputs", ds_name, "E14_HARDNODE_L10_LOCALATTN4_*")))
        if not run_dirs:
            return None
        pattern = os.path.join(run_dirs[-1], "checkpoints", f"{ds_name}_predictions",
                                "epoch_*", "test_predictions.pkl")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    with open(candidates[0], "rb") as f:
        return pickle.load(f)


def find_func_logit_power_theta(ds_name, variant):
    """theta* for the func_logit_power edge aggregator (|logit(q)|^theta
    weighted mean of per-walk q), read from the already-run run_posthoc.py
    artifact -- never refit here. Per user direction: use the '_enc_fixed'
    posthoc run for full attention (the one matching CLAUDE.md's reported
    SOTA AUCs)."""
    cfg = DATASET_CONFIGS[_ds_key(ds_name)]
    if variant == "full":
        epoch = cfg["best_epoch"]
        exp_dir = os.path.join(ROOT, cfg["exp_dir"])
        pattern = os.path.join(exp_dir, "runs", cfg["ds_name"], "E14_HARDNODE_L10", "posthoc",
                                f"*epoch={epoch}-*enc_fixed*", "aggregator",
                                "func_logit_power", "model_config.json")
    else:
        run_dirs = sorted(glob.glob(os.path.join(
            ROOT, "outputs", ds_name, "E14_HARDNODE_L10_LOCALATTN4_*")))
        if not run_dirs:
            return None
        pattern = os.path.join(run_dirs[-1], "posthoc", "localattn4_posthoc", "aggregator",
                                "func_logit_power", "model_config.json")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    with open(candidates[0]) as f:
        cfg_json = json.load(f)
    return float(cfg_json["theta_star"][0])


_FUNC_EPS = 1e-9


def aggregate_func_logit_power(edge_ids, probs, targets, theta):
    """Edge-level score = weighted mean of per-walk q, weight = |logit(q)|^theta
    -- the exact formula run_posthoc.py's func_logit_power aggregator uses
    (run_posthoc.py:_wfn_edge_scores / _FUNC_EPS). theta is read from a saved
    artifact, not refit. Returns (edge_ids, scores, labels), one row per
    distinct edge_id, sorted ascending by edge_id."""
    eps = _FUNC_EPS
    probs = np.clip(np.asarray(probs, dtype=np.float64), eps, 1 - eps)
    logit = np.log(probs / (1 - probs))
    w = np.power(np.abs(logit) + eps, theta)

    edge_ids = np.asarray(edge_ids)
    targets = np.asarray(targets)
    valid = edge_ids >= 0
    edge_ids, w, probs, targets = edge_ids[valid], w[valid], probs[valid], targets[valid]

    order = np.argsort(edge_ids, kind="stable")
    eids_s, w_s, p_s, y_s = edge_ids[order], w[order], probs[order], targets[order]
    uniq_eids, inv_idx, counts = np.unique(eids_s, return_inverse=True, return_counts=True)
    wsum = np.bincount(inv_idx, weights=w_s, minlength=len(uniq_eids))
    wpsum = np.bincount(inv_idx, weights=w_s * p_s, minlength=len(uniq_eids))
    scores = wpsum / np.maximum(wsum, eps)
    first_occ = np.concatenate([[0], np.cumsum(counts)[:-1]])
    labels = y_s[first_occ].astype(int)
    return uniq_eids, scores, labels


def get_walk_model_data(ds_name, variant):
    """variant: 'full' or 'localattn4'. Entropy sign_dicts come from the full
    canonical edge list (load_edges_canonical already returns ALL edges --
    train+val+test -- in raw tokenizer id space, no dataset_cache split
    bookkeeping needed). Per-edge predicted probability is the func_logit_power
    aggregator's weighted mean across that edge's walk occurrences (theta read
    from the existing run_posthoc.py artifact, see find_func_logit_power_theta),
    matching how the project's reported SOTA AUCs are computed -- NOT a naive
    unweighted mean over occurrences."""
    cfg = DATASET_CONFIGS[_ds_key(ds_name)]
    edges = load_edges_canonical(cfg["ds_name"])
    sign_dicts = build_sign_dicts(edges)

    pkl = load_walk_pkl(ds_name, variant)
    if pkl is None:
        return None, None
    theta = find_func_logit_power_theta(ds_name, variant)
    if theta is None:
        raise RuntimeError(
            f"No func_logit_power aggregator artifact found for {ds_name}/{variant} -- "
            "run run_posthoc.py with --artifacts aggregator --agg-models func_logit_power first."
        )

    edge_ids = pkl["edge_ids"]
    probs = pkl["probabilities"][:, 1]
    targets = pkl["targets"]

    agg_eids, agg_scores, agg_labels = aggregate_func_logit_power(edge_ids, probs, targets, theta)

    test_uvyp = []
    for eid, score, y in zip(agg_eids.tolist(), agg_scores.tolist(), agg_labels.tolist()):
        if eid >= len(edges):
            continue
        u, v, _ = edges[eid]
        test_uvyp.append((u, v, y, score))

    return test_uvyp, sign_dicts


def get_gineconv_data(ds_name):
    art_path = os.path.join(ROOT, "baselines", "GINEConv", "results_our_splits",
                             ds_name, "GINEConv", "seed42", "best_epoch_artifacts.pkl")
    if not os.path.exists(art_path):
        return None, None
    with open(art_path, "rb") as f:
        art = pickle.load(f)

    splits_path = os.path.join(ROOT, "baselines", "splits", f"{ds_name}.pt")
    splits = torch.load(splits_path, weights_only=False)
    ei, ew = splits["edge_index"], splits["edge_weight"]
    all_triples = list(zip(ei[0].tolist(), ei[1].tolist(), ew.int().tolist()))
    sign_dicts = build_sign_dicts(all_triples)

    eu, ev = art["edge_index"][0].tolist(), art["edge_index"][1].tolist()
    test_uvyp = list(zip(eu, ev, art["y"].tolist(), art["pred_p"].tolist()))
    return test_uvyp, sign_dicts


def get_sigat_data(ds_name):
    art_path = os.path.join(ROOT, "baselines", "SGA", "results_our_splits",
                             ds_name, "SiGAT", "seed42", "best_epoch_artifacts.pkl")
    if not os.path.exists(art_path):
        return None, None
    with open(art_path, "rb") as f:
        art = pickle.load(f)
    z = art["final_embedding"]

    splits_path = os.path.join(ROOT, "baselines", "splits", f"{ds_name}.pt")
    splits = torch.load(splits_path, weights_only=False)
    ei, ew = splits["edge_index"], splits["edge_weight"]
    trn, tst = splits["trn_mask"], splits["tst_mask"]

    all_triples = list(zip(ei[0].tolist(), ei[1].tolist(), ew.int().tolist()))
    sign_dicts = build_sign_dicts(all_triples)

    def mask_to_np(mask):
        idx = ei[:, mask]
        w = ew[mask].int()
        return np.stack([idx[0].numpy(), idx[1].numpy(), w.numpy()], axis=1)

    # train/test masks here are only about fitting/evaluating the logistic
    # regression read-out (predicting unseen edges) -- unrelated to the
    # entropy grouping above, which intentionally uses ALL edges.
    train_np, test_np = mask_to_np(trn), mask_to_np(tst)
    train_X, train_y = train_np[:, :2], (train_np[:, 2] > 0).astype(int)
    test_X, test_y = test_np[:, :2], (test_np[:, 2] > 0).astype(int)
    train_feats = np.concatenate([z[train_X[:, 0]], z[train_X[:, 1]]], axis=1)
    test_feats = np.concatenate([z[test_X[:, 0]], z[test_X[:, 1]]], axis=1)

    clf = linear_model.LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(train_feats, train_y)
    pred_p = clf.predict_proba(test_feats)[:, 1]

    test_uvyp = list(zip(test_X[:, 0].tolist(), test_X[:, 1].tolist(),
                          test_y.tolist(), pred_p.tolist()))
    return test_uvyp, sign_dicts


LOADERS = {
    "walk_full": lambda ds: get_walk_model_data(ds, "full"),
    "walk_localattn4": lambda ds: get_walk_model_data(ds, "localattn4"),
    "GINEConv": get_gineconv_data,
    "SiGAT": get_sigat_data,
}


# ── Compute stage: raw per-edge (u, v, y, p) records, no binning ─────────────
# LOADERS[model](ds_name) is the expensive part (loads/aggregates walk-model
# predictions, or refits SiGAT's classifier) -- called exactly ONCE per
# (dataset, model), cached, then reused for every entropy variant instead of
# redundantly re-running it once per variant.

def collect_model_records(test_uvyp, sign_dicts, variant, min_total):
    ent = entropy_lookup(sign_dicts)
    src_dir, tgt_dir = VARIANTS[variant]
    src_ent_d, tgt_ent_d = ent[src_dir], ent[tgt_dir]

    src_ent, tgt_ent, y, p = [], [], [], []
    for u, v, yi, pi in test_uvyp:
        se, te = src_ent_d.get(u), tgt_ent_d.get(v)
        if se is None or te is None:
            continue
        src_ent.append(se); tgt_ent.append(te); y.append(yi); p.append(pi)
    if len(y) < min_total:
        return None
    return {
        "src_ent": np.array(src_ent), "tgt_ent": np.array(tgt_ent),
        "y": np.array(y), "p": np.array(p),
    }


def compute_all(datasets, variants, min_total):
    """Returns (entropy_data, predictions):
      - entropy_data[ds][variant][model] = record dict or None (for binning/plotting)
      - predictions[ds][model] = {"u","v","y","p"} or None -- raw per-edge
        predictions, variant-independent, kept for any downstream investigation
        beyond this entropy question (node ids in that model's own id space --
        raw tokenizer ids for walk models, dense baselines/splits ids for
        GINEConv/SiGAT, see module docstring)."""
    entropy_data, predictions = {}, {}
    for ds_name in datasets:
        print(f"\n{'=' * 72}\n{ds_name}\n{'=' * 72}")
        entropy_data[ds_name] = {}
        predictions[ds_name] = {}

        # load/aggregate each model's predictions exactly once per dataset,
        # reused below for every entropy variant
        loaded = {}
        for model in MODELS:
            test_uvyp, sign_dicts = LOADERS[model](ds_name)
            loaded[model] = (test_uvyp, sign_dicts)
            if test_uvyp is None:
                predictions[ds_name][model] = None
                print(f"  {model}: no data")
                continue
            u, v, y, p = zip(*test_uvyp)
            predictions[ds_name][model] = {
                "u": np.array(u), "v": np.array(v),
                "y": np.array(y), "p": np.array(p),
            }
            overall_auc = roc_auc_score(predictions[ds_name][model]["y"],
                                         predictions[ds_name][model]["p"])
            print(f"  {model}: n={len(test_uvyp)}, overall_auc={overall_auc:.4f}")

        for variant in variants:
            print(f"  variant={variant}")
            entropy_data[ds_name][variant] = {}
            for model in MODELS:
                test_uvyp, sign_dicts = loaded[model]
                if test_uvyp is None:
                    entropy_data[ds_name][variant][model] = None
                    continue
                rec = collect_model_records(test_uvyp, sign_dicts, variant, min_total)
                if rec is None:
                    print(f"    {model}: insufficient samples after entropy lookup")
                entropy_data[ds_name][variant][model] = rec

    return entropy_data, predictions


def save_data(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"\n✓ Computed data saved to {path}")


def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Plot stage: binning + per-cell AUC + figures + report ────────────────────

def quantile_bin_edges(values, n_buckets):
    """Per-dataset quantile edges -- equal SAMPLE COUNT per bin, but bin
    *boundaries* differ from dataset to dataset, which makes side-by-side
    heatmaps visually inconsistent (e.g. bin 3 of bitcoin-alpha covers a
    different entropy range than bin 3 of epinions). Use binning='fixed' for
    cross-dataset comparability."""
    values = np.asarray(values, dtype=np.float64)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_buckets + 1)))
    if len(edges) < 2:
        return np.array([values.min() - 1e-9, values.max() + 1e-9])
    edges = edges.copy()
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def fixed_bin_edges(n_buckets):
    """Global edges over the full possible entropy range [0, 1] bit, shared by
    every dataset/variant/model -- makes heatmaps directly comparable across
    datasets (e.g. for the combined all-datasets plot) at the cost of
    possibly-uneven sample counts per cell."""
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def bin_edges(values, n_buckets, binning):
    if binning == "fixed":
        return fixed_bin_edges(n_buckets)
    return quantile_bin_edges(values, n_buckets)


def digitize(values, edges):
    return np.clip(np.digitize(values, edges[1:-1], right=True), 0, len(edges) - 2)


def auc_grid(src_ent, tgt_ent, y, p, n_buckets, min_n, binning="quantile"):
    src_edges = bin_edges(src_ent, n_buckets, binning)
    tgt_edges = bin_edges(tgt_ent, n_buckets, binning)
    src_bins = digitize(src_ent, src_edges)
    tgt_bins = digitize(tgt_ent, tgt_edges)
    n_src, n_tgt = len(src_edges) - 1, len(tgt_edges) - 1

    auc = np.full((n_tgt, n_src), np.nan)
    counts = np.zeros((n_tgt, n_src), dtype=int)
    y, p = np.asarray(y), np.asarray(p)
    for i in range(n_src):
        for j in range(n_tgt):
            mask = (src_bins == i) & (tgt_bins == j)
            counts[j, i] = int(mask.sum())
            if counts[j, i] >= min_n and len(np.unique(y[mask])) == 2:
                auc[j, i] = roc_auc_score(y[mask], p[mask])
    return auc, counts, src_edges, tgt_edges


def _bad_cmap(cmap_name):
    """NaN cells (no AUC) default to fully transparent in imshow, which reads
    as a misleading blank-white gap. Force them to a visible light gray so
    'no signal here' is distinguishable from 'plot background'."""
    cm = matplotlib.colormaps[cmap_name].copy()
    cm.set_bad(color="#dcdcdc")
    return cm


def _annotate_cell(ax, i, j, auc_val, count, min_n):
    """Three distinct cases, each labeled differently so a blank/gray cell's
    meaning is never ambiguous:
      - count == 0           : truly empty cell (no test edges landed here) -- no text
      - 0 < count < min_n     : under the min-sample threshold -- 'n=.. (<min_n)'
      - count >= min_n, NaN   : enough samples but only one class present
                                (e.g. a near-zero-entropy bucket where every
                                edge has the same sign) -- AUC is undefined,
                                not a bug -- 'n=.. (1 class)'
      - otherwise             : normal 'auc\n(n=..)'
    """
    if count == 0:
        return
    if not np.isnan(auc_val):
        ax.text(i, j, f"{auc_val:.2f}\n(n={count})", ha="center", va="center", fontsize=7)
    elif count < min_n:
        ax.text(i, j, f"n={count}\n(<{min_n})", ha="center", va="center",
                fontsize=6, color="dimgray")
    else:
        ax.text(i, j, f"n={count}\n(1 class)", ha="center", va="center",
                fontsize=6, color="dimgray")


def _draw_heatmap(ax, auc, counts, src_edges, tgt_edges, cmap, vmin, vmax, min_n, title):
    masked = np.ma.masked_invalid(auc)
    im = ax.imshow(masked, origin="lower", cmap=_bad_cmap(cmap), vmin=vmin, vmax=vmax,
                    aspect="auto")
    for j in range(auc.shape[0]):
        for i in range(auc.shape[1]):
            _annotate_cell(ax, i, j, auc[j, i], counts[j, i], min_n)
    ax.set_xticks(range(auc.shape[1]))
    ax.set_xticklabels([f"{src_edges[i]:.2f}-{src_edges[i+1]:.2f}" for i in range(auc.shape[1])],
                        fontsize=6, rotation=40, ha="right")
    ax.set_yticks(range(auc.shape[0]))
    ax.set_yticklabels([f"{tgt_edges[j]:.2f}-{tgt_edges[j+1]:.2f}" for j in range(auc.shape[0])],
                        fontsize=6)
    if title:
        ax.set_title(title, fontsize=9)
    return im


def plot_variant(ds_name, variant, grids, out_dir, cmap, vmin, vmax, dpi, min_n, suffix=""):
    """grids: {model: (auc_grid, counts, src_edges, tgt_edges) or None}."""
    models_present = [m for m in MODELS if grids.get(m) is not None]
    if not models_present:
        return
    n = len(models_present)
    # dedicated colorbar column (width_ratios) instead of stealing space from
    # the rightmost subplot via ax= -- avoids the colorbar overlapping plot 4.
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * n + 1, 4.5),
                              gridspec_kw={"width_ratios": [1] * n + [0.06]})
    im = None
    for ax, model in zip(axes[:n], models_present):
        auc, counts, src_edges, tgt_edges = grids[model]
        im = _draw_heatmap(ax, auc, counts, src_edges, tgt_edges, cmap, vmin, vmax, min_n, model)
        ax.set_xlabel("source-node entropy (bits)")
        ax.set_ylabel("target-node entropy (bits)")
    fig.colorbar(im, cax=axes[n], label="AUC")
    fig.suptitle(f"{ds_name}: AUC vs. (source, target) sign-entropy -- variant={variant}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(out_dir, f"{ds_name}_{variant}{suffix}.png")
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_variant_combined(variant, ds_grids, out_dir, cmap, vmin, vmax, dpi, min_n, suffix=""):
    """One figure per variant, all datasets x all models, using the SAME fixed
    bin edges throughout (caller must pass grids computed with binning='fixed')
    so rows/cols are genuinely comparable across datasets.
    ds_grids: {dataset: {model: (auc, counts, src_edges, tgt_edges) or None}}."""
    datasets_present = [d for d in ds_grids if any(v is not None for v in ds_grids[d].values())]
    if not datasets_present:
        return
    n_rows, n_cols = len(datasets_present), len(MODELS)
    fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(4.2 * n_cols + 1, 3.6 * n_rows),
                              gridspec_kw={"width_ratios": [1] * n_cols + [0.05]},
                              squeeze=False)
    im = None
    for r, ds_name in enumerate(datasets_present):
        for c, model in enumerate(MODELS):
            ax = axes[r][c]
            grid = ds_grids[ds_name].get(model)
            if grid is None:
                ax.axis("off")
                continue
            auc, counts, src_edges, tgt_edges = grid
            title = model if r == 0 else None
            im = _draw_heatmap(ax, auc, counts, src_edges, tgt_edges, cmap, vmin, vmax, min_n, title)
            if c == 0:
                ax.set_ylabel(f"{ds_name}\ntarget entropy", fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("source entropy (bits)", fontsize=8)
        for c in range(n_cols):
            if ds_grids[ds_name].get(MODELS[c]) is None:
                continue
        axes[r][n_cols].axis("off")
    if im is not None:
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cax, label="AUC")
    fig.suptitle(f"All datasets: AUC vs. (source, target) sign-entropy -- variant={variant}")
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    save_path = os.path.join(out_dir, f"ALL_DATASETS_{variant}{suffix}.png")
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  saved {save_path}")


def low_high_summary_md(ds_name, variant, grids):
    """Markdown table: lowest-joint-entropy cell (both axes lowest bucket) vs.
    highest-joint cell (both axes highest bucket), per model."""
    lines = [
        f"**variant = `{variant}`**",
        "",
        "| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |",
        "|---|---|---|---|",
    ]
    for model in MODELS:
        if grids.get(model) is None:
            lines.append(f"| {model} | no data | no data | n/a |")
            continue
        auc, counts, _, _ = grids[model]
        lo, hi = auc[0, 0], auc[-1, -1]
        lo_s = f"{lo:.4f}" if not np.isnan(lo) else "n/a"
        hi_s = f"{hi:.4f}" if not np.isnan(hi) else "n/a"
        drop = (lo - hi) if (not np.isnan(lo) and not np.isnan(hi)) else float("nan")
        drop_s = f"{drop:+.4f}" if not np.isnan(drop) else "n/a"
        lines.append(f"| {model} | {lo_s} (n={counts[0,0]}) | {hi_s} (n={counts[-1,-1]}) | {drop_s} |")
    lines.append("")
    return lines


def grid_to_text(ds_name, variant, n_buckets, binning, model, auc, counts, src_edges, tgt_edges, min_n):
    """Plain-text dump of one (dataset, variant, bucket-count, model) heatmap --
    same data as the PNG, for people who'd rather grep/diff numbers than look
    at pictures. One row per cell, tab-separated."""
    lines = [
        f"=== {ds_name} | variant={variant} | binning={binning} | n_buckets={n_buckets} | model={model} ===",
        "src_entropy_range\ttgt_entropy_range\tauc\tn\tnote",
    ]
    for j in range(auc.shape[0]):
        for i in range(auc.shape[1]):
            src_r = f"{src_edges[i]:.4f}-{src_edges[i+1]:.4f}"
            tgt_r = f"{tgt_edges[j]:.4f}-{tgt_edges[j+1]:.4f}"
            c = int(counts[j, i])
            a = auc[j, i]
            if c == 0:
                a_s, note = "", "empty"
            elif np.isnan(a):
                a_s, note = "", ("below_min_n" if c < min_n else "single_class")
            else:
                a_s, note = f"{a:.4f}", ""
            lines.append(f"{src_r}\t{tgt_r}\t{a_s}\t{c}\t{note}")
    lines.append("")
    return lines


def plot_all(data, variants, out_dir, n_buckets_list, min_cell_n, cmap, vmin, vmax, dpi,
             binning, combined):
    text_lines = [
        "LEAD 4: NODE SIGN-ENTROPY HETEROGENEITY vs. AUC -- raw per-cell data",
        "Same numbers as the PNGs / report.md tables, in plain text. See report.md",
        "for methodology (entropy definition, binning, aggregator, etc).",
        "",
    ]
    report_lines = [
        "# Lead 4: Node Sign-Entropy Heterogeneity vs. AUC",
        "",
        "## Methodology",
        "",
        "For each test edge `(u, v)`, bin source-node sign-entropy x target-node",
        "sign-entropy into a 2D quantile grid and compute AUC within each cell, for",
        "4 models: `walk_full` (E14_HARDNODE_L10, full attention), `walk_localattn4`",
        "(E14_HARDNODE_L10_LOCALATTN4, +-2 hop banded attention), `GINEConv`, `SiGAT`",
        "(both dense-id baselines).",
        "",
        "**Entropy** is binary Shannon entropy in bits over a node's sign labels,",
        "`H(p) = -p*log2(p) - (1-p)*log2(1-p)`, where `p` = fraction of positive-sign",
        "edges in the relevant direction. `H=0` -> homogeneous (all one sign),",
        "`H=1` -> maximally heterogeneous (50/50 split). Computed over **all edges**",
        "of the dataset (train+val+test) -- this is a diagnostic grouping of",
        "already-trained models' predictions, not a training-time feature, so there",
        "is no leakage concern; using all edges instead of train-only gives a less",
        "noisy estimate of each node's true heterogeneity, especially at low degree.",
        "Nodes with zero edges in the relevant direction are dropped.",
        "",
        "**Entropy variants** -- source/target entropy can each be computed from",
        "out-edges, in-edges, or in+out (combined) signs of that node:",
        "",
        "| variant | source | target |",
        "|---|---|---|",
        "| `out_out` | H(out-signs of u) | H(out-signs of v) |",
        "| `in_in` | H(in-signs of u) | H(in-signs of v) |",
        "| `out_in` | H(out-signs of u) | H(in-signs of v) |",
        "| `inout_inout` | H(all incident signs of u) | H(all incident signs of v) |",
        "",
        "**No retraining**: walk-model predictions come from existing",
        "`*_predictions/epoch_*/test_predictions.pkl` artifacts (`edge_ids` decoded",
        "back to `(u, v, sign)` via the canonical edge list). Each edge is visited by",
        "many walks, so per-walk probabilities are combined into one edge-level score",
        "via the SAME aggregator used for the project's reported SOTA numbers --",
        "`func_logit_power`: weighted mean of per-walk `q` with weight",
        "`|logit(q)|^theta`. `theta*` is read directly from the existing",
        "`run_posthoc.py` aggregator artifact (`model_config.json`), never refit here:",
        "the `*_posthoc_enc_fixed` run for full attention, `localattn4_posthoc` for",
        "LocalAttn4. GINEConv predictions",
        "come straight from its `best_epoch_artifacts.pkl`. SiGAT has no saved",
        "predictions, only `final_embedding` -- its test-set predictions are",
        "reconstructed by refitting the same `sklearn.linear_model.LogisticRegression`",
        "on concatenated embeddings that `baselines/SGA/run_with_our_splits.py`'s own",
        "`evaluate()` used (deterministic, no retraining of SiGAT itself; this",
        "classifier fit still uses the train/test split masks -- unrelated to the",
        "all-edges entropy computation above).",
        "",
        f"**Binning**: `binning={binning}` grid(s) per (dataset, variant), swept over",
        f"bucket counts {n_buckets_list} (filenames carry a `_{{binning}}_b{{n}}` suffix);",
        f"a cell needs >= {min_cell_n} samples and both classes present to get an AUC.",
        "`quantile` binning gives equal sample counts per bin but DIFFERENT entropy",
        "ranges per dataset (bin edges aren't comparable across datasets/plots).",
        "`fixed` binning uses the same global `[0, 1]`-bit edges for every dataset, so",
        "heatmaps -- including the combined all-datasets plot below -- are visually",
        "comparable at the cost of uneven sample counts per cell.",
        "",
        "Gray cells with `n=.. (<min_n)` are below the minimum-sample threshold; gray",
        "cells with `n=.. (1 class)` have enough samples but only one true sign",
        "present (common in near-zero-entropy buckets, where by construction almost",
        "all edges share one sign) so AUC is undefined -- not a bug. Fully blank cells",
        "have zero samples.",
        "",
        "**Raw text data**: every heatmap cell (entropy ranges, AUC, n, note) is also",
        "dumped as plain tab-separated text in `raw_data.txt` next to this report --",
        "same numbers as the PNGs/tables, for grepping/diffing instead of eyeballing",
        "pictures.",
        "",
        "**Reproducing / extending this analysis**: raw per-edge records",
        "`(src_ent, tgt_ent, y, p)` for every (dataset, variant, model) are cached in",
        f"`{DATA_FILENAME}` next to this report. To regenerate model predictions and",
        "entropy from scratch: `--mode compute`. To re-bin / restyle plots (colormap,",
        "binning mode, bucket-count sweep, cell threshold, vmin/vmax) without",
        "recomputation: `--mode plot`. The same `collect_model_records` /",
        "`compute_all` / `save_data` pattern can be reused for other per-edge metrics",
        "in the same spirit (e.g. degree instead of entropy) by swapping out",
        "`entropy_lookup` for a different per-node lookup.",
        "",
        "## Results",
        "",
    ]

    combined_out = os.path.join(out_dir, "combined")
    if combined:
        os.makedirs(combined_out, exist_ok=True)

    for ds_name, ds_data in data.items():
        ds_out = os.path.join(out_dir, ds_name)
        os.makedirs(ds_out, exist_ok=True)
        print(f"\n{'=' * 72}\n{ds_name}\n{'=' * 72}")
        report_lines.append(f"### {ds_name}")
        report_lines.append("")

        for variant in variants:
            if variant not in ds_data:
                continue
            print(f"  variant={variant}")
            summary_grids = None  # use the first (smallest) bucket count for the summary table
            for n_buckets in n_buckets_list:
                grids = {}
                for model in MODELS:
                    rec = ds_data[variant].get(model)
                    if rec is None:
                        grids[model] = None
                        continue
                    grids[model] = auc_grid(rec["src_ent"], rec["tgt_ent"], rec["y"], rec["p"],
                                             n_buckets, min_cell_n, binning)
                suffix = f"_{binning}_b{n_buckets}"
                plot_variant(ds_name, variant, grids, ds_out, cmap, vmin, vmax, dpi, min_cell_n, suffix)
                report_lines.append(f"![{ds_name} {variant} b{n_buckets}]"
                                     f"({ds_name}/{ds_name}_{variant}{suffix}.png)")
                report_lines.append("")
                for model in MODELS:
                    if grids.get(model) is None:
                        continue
                    auc, counts, src_edges, tgt_edges = grids[model]
                    text_lines.extend(grid_to_text(ds_name, variant, n_buckets, binning, model,
                                                    auc, counts, src_edges, tgt_edges, min_cell_n))
                if summary_grids is None:
                    summary_grids = grids
            report_lines.extend(low_high_summary_md(ds_name, variant, summary_grids))

    if combined:
        print(f"\n{'=' * 72}\ncombined (all datasets)\n{'=' * 72}")
        report_lines.append("### Combined (all datasets)")
        report_lines.append("")
        for variant in variants:
            for n_buckets in n_buckets_list:
                ds_grids = {}
                for ds_name, ds_data in data.items():
                    if variant not in ds_data:
                        continue
                    grids = {}
                    for model in MODELS:
                        rec = ds_data[variant].get(model)
                        if rec is None:
                            grids[model] = None
                            continue
                        # combined plot always uses 'fixed' binning regardless of the
                        # per-dataset --binning choice -- quantile edges differ per
                        # dataset and would make rows visually incomparable.
                        grids[model] = auc_grid(rec["src_ent"], rec["tgt_ent"], rec["y"], rec["p"],
                                                 n_buckets, min_cell_n, "fixed")
                    ds_grids[ds_name] = grids
                suffix = f"_fixed_b{n_buckets}"
                plot_variant_combined(variant, ds_grids, combined_out, cmap, vmin, vmax, dpi,
                                       min_cell_n, suffix)
                report_lines.append(f"![ALL_DATASETS {variant} b{n_buckets}]"
                                     f"(combined/ALL_DATASETS_{variant}{suffix}.png)")
                report_lines.append("")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n✓ Report written to {report_path}")

    text_path = os.path.join(out_dir, "raw_data.txt")
    with open(text_path, "w") as f:
        f.write("\n".join(text_lines))
    print(f"✓ Raw text data written to {text_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compute", "plot", "all"], default="all")
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    parser.add_argument("--out", default=OUT_DIR_DEFAULT)
    parser.add_argument("--data-path", default=None,
                         help=f"path to computed-data pickle (default: <out>/{DATA_FILENAME})")
    parser.add_argument("--predictions-path", default=None,
                         help=f"path to raw-predictions pickle (default: <out>/{PREDICTIONS_FILENAME})")
    parser.add_argument("--min-total", type=int, default=MIN_TOTAL_DEFAULT,
                         help="drop a (dataset, variant, model) entirely below this many usable edges (compute stage)")
    parser.add_argument("--binning", choices=["quantile", "fixed"], default="fixed",
                         help="'fixed' (default) uses global [0,1]-bit edges, comparable across "
                              "datasets; 'quantile' uses per-dataset equal-sample-count edges")
    parser.add_argument("--n-buckets", type=int, nargs="+", default=[8],
                         help="one or more bucket counts to sweep, each producing its own plot set")
    parser.add_argument("--min-cell-n", type=int, default=MIN_CELL_N_DEFAULT)
    parser.add_argument("--cmap", default="RdYlGn")
    parser.add_argument("--vmin", type=float, default=0.5)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--no-combined", action="store_true",
                         help="skip the all-datasets combined plot per variant")
    args = parser.parse_args()

    datasets = DATASETS if args.datasets == ["all"] else args.datasets
    data_path = args.data_path or os.path.join(args.out, DATA_FILENAME)
    predictions_path = args.predictions_path or os.path.join(args.out, PREDICTIONS_FILENAME)

    if args.mode in ("compute", "all"):
        data, predictions = compute_all(datasets, args.variants, args.min_total)
        save_data(data, data_path)
        save_data(predictions, predictions_path)
        print(f"✓ Raw per-edge predictions (all models, all datasets) saved to {predictions_path}")
    if args.mode == "plot":
        data = load_data(data_path)
        if args.datasets != ["all"]:
            data = {k: v for k, v in data.items() if k in datasets}

    if args.mode in ("plot", "all"):
        plot_all(data, args.variants, args.out, args.n_buckets, args.min_cell_n,
                  args.cmap, args.vmin, args.vmax, args.dpi, args.binning, not args.no_combined)


if __name__ == "__main__":
    main()
