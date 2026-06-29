"""
Lead 4 -- node sign-entropy heterogeneity vs. AUC (canonical, shared-edge).

Hypothesis (original): all models (walk transformer, GINEConv, SiGAT) do well
when both endpoints of a test edge sit in a locally homogeneous (low
sign-entropy) neighborhood, but GNNs degrade *more* than the walk model as
endpoint sign heterogeneity (entropy) rises -- i.e. the walk model's advantage
is concentrated in the high-entropy region. On the corrected, same-edge split
this turns out NOT to hold: every model degrades by similar amounts (see
report.md / CANONICAL_RERUN_FINDINGS.md). The diagnostic itself remains the
right tool, so this is its standing, canonical implementation.

Same-edge ground truth. Every model's per-edge predictions are read from
`predictions_raw_canonical.pkl` (built by baselines/postprocess_canonical.py
after the canonical-split baseline reruns), where ALL models live in the SAME
raw `(u, v)` id space. We restrict every model to the SHARED edge set -- the
intersection of (u, v) across models, which equals the walk-covered test edges
(walk_full is a strict subset of GINEConv/SiGAT's test set). So every cell of
every heatmap compares the four models on the *identical* edges with the
*identical* ground truth, and the per-cell sample count `n` is the same for all
four models. No model reloading, no SiGAT refit, no walk re-aggregation here --
those happened upstream; this script only groups already-computed predictions.

For each shared test edge (u, v): bin source-node sign-entropy x target-node
sign-entropy into a 2D grid and compute AUC within each cell, for 4 models:
  - walk_full        (E14_HARDNODE_L10, full attention)
  - walk_localattn4  (E14_HARDNODE_L10_LOCALATTN4, +-2 hop banded attention)
  - GINEConv         (baselines/GINEConv)
  - SiGAT            (baselines/SGA)

Entropy is computed from ALL edges of the dataset (train+val+test) via the
canonical edge list (`load_edges_canonical`, raw ids -- the same id space the
predictions live in, so there is no dense/raw remap and no fabricated-reverse-
edge contamination to filter out; that was an artifact of the old
`baselines/splits/*.pt` dense graph and is gone on the canonical split). This is
purely a diagnostic grouping of already-trained models' predictions, so there is
no train/test leakage concern; using all edges gives a less noisy estimate of
each node's true sign-heterogeneity than train-only counts.

Entropy variants (4): for each test edge (u, v), source/target entropy can each
be computed from OUT-edges, IN-edges, or IN+OUT (combined) signs of that node:
  out_out      : H(out-signs of u),         H(out-signs of v)
  in_in        : H(in-signs of u),          H(in-signs of v)
  out_in       : H(out-signs of u),         H(in-signs of v)
  inout_inout  : H(all incident signs of u),H(all incident signs of v)

Binary Shannon entropy in bits: H(p) = -p*log2(p) - (1-p)*log2(1-p), p =
fraction of positive-sign edges in the relevant set. H=0 -> homogeneous (all
same sign), H=1 -> maximally heterogeneous (50/50). Nodes with zero edges in the
relevant direction are dropped (no signal available).

Compute/plot are separable so plot styling (colormap, binning, bucket sweep,
vmin/vmax) can be iterated without recomputing entropy / re-binning:

    # restrict to shared edges, compute per-edge (entropy, y, p), save to disk
    python scripts/lead4_entropy_heterogeneity.py --mode compute --datasets all

    # replot from saved data, as many times as you like, with new styling
    python scripts/lead4_entropy_heterogeneity.py --mode plot --cmap viridis \
        --n-buckets 2 3 4 --min-cell-n 15

    # default: do both in one go
    python scripts/lead4_entropy_heterogeneity.py --mode all --datasets all
"""
import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.node_mi_structural_embedding import (
    DATASET_CONFIGS, load_edges_canonical,
)

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
MODELS = ["walk_full", "walk_localattn4", "GINEConv", "SiGAT"]
WALK_MODELS = ("walk_full", "walk_localattn4")  # the rest are GNNs (walk-vs-GNN contrast)
VARIANTS = {
    "out_out":     ("out", "out"),
    "in_in":       ("in", "in"),
    "out_in":      ("out", "in"),
    "inout_inout": ("inout", "inout"),
}
N_BUCKETS_DEFAULT = [2, 4, 8, 16, 32]  # powers of 2
MIN_CELL_N_DEFAULT = 20
MIN_TOTAL_DEFAULT = 40  # drop a (dataset, variant, model) entirely below this many usable edges
OUT_DIR_DEFAULT = os.path.join(ROOT, "outputs", "lead4_entropy_heterogeneity")
CANON_PREDICTIONS_DEFAULT = os.path.join(OUT_DIR_DEFAULT, "predictions_raw_canonical.pkl")
DATA_FILENAME = "computed_data.pkl"


def _ds_key(ds_name):
    """node_mi_structural_embedding.py keys slashdot090221 by 'slashdot'."""
    return "slashdot" if ds_name == "slashdot090221" else ds_name


def _fmt_signed(x):
    """Signed float, with -0.0000 normalized to +0.0000 (no negative zero)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if x == 0:
        x = 0.0
    return f"{x:+.4f}"


def _fmt_edge(x, nd=2):
    """Bin-edge label, with the -1e-9 padding on the lowest edge (and any other
    rounds-to-zero value) shown as 0.00 instead of -0.00 (no negative zero on
    the axes / in raw_data.txt)."""
    v = round(float(x), nd)
    if v == 0:
        v = 0.0
    return f"{v:.{nd}f}"


# ── Shared-edge restriction ──────────────────────────────────────────────────
# The canonical predictions pkl carries each model on its OWN full test set
# (walk covers a subset on sparse graphs, GNNs cover all of it). Intersecting on
# (u, v) gives the honest same-edges comparison set = the walk-covered edges.

def shared_edge_set(models):
    sets = [set(zip(list(m["u"]), list(m["v"]))) for m in models.values()
            if m is not None and len(m["u"])]
    if not sets:
        return set()
    shared = sets[0]
    for s in sets[1:]:
        shared &= s
    return shared


def _restrict(uvyp, keep):
    u, v, y, p = uvyp["u"], uvyp["v"], uvyp["y"], uvyp["p"]
    idx = [i for i, (a, b) in enumerate(zip(list(u), list(v))) if (a, b) in keep]
    return {
        "u": [int(u[i]) for i in idx], "v": [int(v[i]) for i in idx],
        "y": [int(y[i]) for i in idx], "p": [float(p[i]) for i in idx],
    }


def load_shared_predictions(path, datasets=None):
    """Read the canonical per-edge predictions and restrict every model to the
    SHARED edge set per dataset. Returns {ds: {model: {u,v,y,p} | None}} with
    every present model on identical (u, v). Also prints the shared-edge overall
    AUC table (the headline same-edges comparison). Shared by Lead 4 and 4b so
    both bucket exactly the same edges."""
    with open(path, "rb") as f:
        preds = pickle.load(f)
    datasets = datasets or [d for d in DATASETS if d in preds]

    out = {}
    print(f"\n{'=' * 78}\nSHARED-EDGE (walk-covered) overall AUCs -- canonical split\n{'=' * 78}")
    print(f"{'dataset':<16}{'n_shared':>10}  " + "  ".join(f"{m:>16}" for m in MODELS))
    for ds in datasets:
        models = preds.get(ds, {})
        keep = shared_edge_set(models)
        out[ds] = {}
        cells = []
        for m in MODELS:
            if models.get(m) is None:
                out[ds][m] = None
                cells.append(float("nan"))
                continue
            r = _restrict(models[m], keep)
            out[ds][m] = r
            auc = (roc_auc_score(r["y"], r["p"])
                   if len(set(r["y"])) > 1 and len(r["y"]) else float("nan"))
            cells.append(auc)
        print(f"{ds:<16}{len(keep):>10}  " + "  ".join(f"{c:>16.4f}" for c in cells))
    return out


# ── Entropy machinery ────────────────────────────────────────────────────────

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
    out_signs, in_signs, inout_signs = sign_dicts
    return {
        "out": {n: binary_entropy(s) for n, s in out_signs.items()},
        "in": {n: binary_entropy(s) for n, s in in_signs.items()},
        "inout": {n: binary_entropy(s) for n, s in inout_signs.items()},
    }


# ── Compute stage: raw per-edge (src_ent, tgt_ent, y, p) records ──────────────

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


def compute_all(predictions, datasets, variants, min_total):
    """predictions: shared-edge {ds: {model: {u,v,y,p}}}. Builds ONE sign-entropy
    basis per dataset from the real canonical edge list (raw ids), shared by all
    four models. Returns entropy_data[ds][variant][model] = record | None."""
    entropy_data = {}
    for ds_name in datasets:
        print(f"\n{'=' * 72}\n{ds_name}\n{'=' * 72}")
        edges = load_edges_canonical(DATASET_CONFIGS[_ds_key(ds_name)]["ds_name"])
        sign_dicts = build_sign_dicts(edges)
        entropy_data[ds_name] = {v: {} for v in variants}

        for model in MODELS:
            r = predictions.get(ds_name, {}).get(model)
            if r is None or not len(r["u"]):
                for variant in variants:
                    entropy_data[ds_name][variant][model] = None
                print(f"  {model}: no data")
                continue
            auc = (roc_auc_score(r["y"], r["p"]) if len(set(r["y"])) > 1 else float("nan"))
            print(f"  {model}: n={len(r['u'])}, shared_overall_auc={auc:.4f}")
            test_uvyp = list(zip(r["u"], r["v"], r["y"], r["p"]))
            for variant in variants:
                entropy_data[ds_name][variant][model] = collect_model_records(
                    test_uvyp, sign_dicts, variant, min_total)
    return entropy_data


def save_data(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"\n✓ Computed data saved to {path}")


def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Plot stage: binning + per-cell AUC + heatmaps + report ────────────────────

def quantile_bin_edges(values, n_buckets):
    values = np.asarray(values, dtype=np.float64)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_buckets + 1)))
    if len(edges) < 2:
        return np.array([values.min() - 1e-9, values.max() + 1e-9])
    edges = edges.copy()
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def fixed_bin_edges(n_buckets):
    """Global edges over the full [0, 1]-bit entropy range, shared by every
    dataset -- makes heatmaps directly comparable across datasets."""
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


def auc_grid(src_ent, tgt_ent, y, p, n_buckets, min_n, binning="fixed"):
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
    cm = matplotlib.colormaps[cmap_name].copy()
    cm.set_bad(color="#dcdcdc")
    return cm


def _annotate_cell(ax, i, j, auc_val, count, min_n):
    if count == 0:
        return
    if not np.isnan(auc_val):
        ax.text(i, j, f"{auc_val:.2f}\n(n={count})", ha="center", va="center", fontsize=7)
    elif count < min_n:
        ax.text(i, j, f"n={count}\n(<{min_n})", ha="center", va="center", fontsize=6, color="dimgray")
    else:
        ax.text(i, j, f"n={count}\n(1 class)", ha="center", va="center", fontsize=6, color="dimgray")


def _draw_heatmap(ax, auc, counts, src_edges, tgt_edges, cmap, vmin, vmax, min_n, title):
    masked = np.ma.masked_invalid(auc)
    im = ax.imshow(masked, origin="lower", cmap=_bad_cmap(cmap), vmin=vmin, vmax=vmax, aspect="auto")
    for j in range(auc.shape[0]):
        for i in range(auc.shape[1]):
            _annotate_cell(ax, i, j, auc[j, i], counts[j, i], min_n)
    ax.set_xticks(range(auc.shape[1]))
    ax.set_xticklabels([f"{_fmt_edge(src_edges[i])}-{_fmt_edge(src_edges[i+1])}" for i in range(auc.shape[1])],
                        fontsize=6, rotation=40, ha="right")
    ax.set_yticks(range(auc.shape[0]))
    ax.set_yticklabels([f"{_fmt_edge(tgt_edges[j])}-{_fmt_edge(tgt_edges[j+1])}" for j in range(auc.shape[0])],
                        fontsize=6)
    if title:
        ax.set_title(title, fontsize=9)
    return im


_GRAY_LEGEND = [Patch(facecolor="#dcdcdc", edgecolor="gray",
                      label="no AUC (empty / <min_n / single-class cell)")]


def plot_variant(ds_name, variant, grids, out_dir, cmap, vmin, vmax, dpi, min_n, suffix=""):
    models_present = [m for m in MODELS if grids.get(m) is not None]
    if not models_present:
        return
    n = len(models_present)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * n + 1, 4.7),
                              gridspec_kw={"width_ratios": [1] * n + [0.06]})
    im = None
    for ax, model in zip(axes[:n], models_present):
        auc, counts, src_edges, tgt_edges = grids[model]
        tag = "[walk]" if model in WALK_MODELS else "[GNN]"
        im = _draw_heatmap(ax, auc, counts, src_edges, tgt_edges, cmap, vmin, vmax, min_n,
                            f"{model} {tag}")
        ax.set_xlabel("source-node entropy (bits)")
        ax.set_ylabel("target-node entropy (bits)")
    fig.colorbar(im, cax=axes[n], label="AUC")
    fig.legend(handles=_GRAY_LEGEND, loc="lower center", fontsize=7, ncol=1, frameon=False)
    fig.suptitle(f"{ds_name}: AUC vs. (source, target) sign-entropy -- variant={variant} "
                 f"(same shared edges for all models; n identical across cells)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(out_dir, f"{ds_name}_{variant}{suffix}.png")
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_variant_combined(variant, ds_grids, out_dir, cmap, vmin, vmax, dpi, min_n, suffix=""):
    datasets_present = [d for d in ds_grids if any(v is not None for v in ds_grids[d].values())]
    if not datasets_present:
        return
    n_rows, n_cols = len(datasets_present), len(MODELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols + 1.2, 3.6 * n_rows),
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
            tag = "[walk]" if model in WALK_MODELS else "[GNN]"
            title = f"{model} {tag}" if r == 0 else None
            im = _draw_heatmap(ax, auc, counts, src_edges, tgt_edges, cmap, vmin, vmax, min_n, title)
            if c == 0:
                ax.set_ylabel(f"{ds_name}\ntarget entropy", fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("source entropy (bits)", fontsize=8)
    if im is not None:
        cax = fig.add_axes([0.93, 0.15, 0.013, 0.7])
        fig.colorbar(im, cax=cax, label="AUC")
    fig.legend(handles=_GRAY_LEGEND, loc="lower center", fontsize=8, ncol=1, frameon=False)
    fig.suptitle(f"All datasets: AUC vs. (source, target) sign-entropy -- variant={variant}  "
                 f"(columns = models, walk vs GNN; same shared edges)", y=0.995)
    fig.tight_layout(rect=[0, 0.02, 0.91, 0.97])
    save_path = os.path.join(out_dir, f"ALL_DATASETS_{variant}{suffix}.png")
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  saved {save_path}")


def low_high_summary_md(ds_name, variant, grids):
    """Markdown: lowest-joint-entropy cell vs. highest-joint cell AUC, per model."""
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
        lines.append(f"| {model} | {lo_s} (n={counts[0,0]}) | {hi_s} (n={counts[-1,-1]}) | {_fmt_signed(drop)} |")
    lines.append("")
    return lines


def grid_to_text(ds_name, variant, n_buckets, binning, model, auc, counts, src_edges, tgt_edges, min_n):
    lines = [
        f"=== {ds_name} | variant={variant} | binning={binning} | n_buckets={n_buckets} | model={model} ===",
        "src_entropy_range\ttgt_entropy_range\tauc\tn\tnote",
    ]
    for j in range(auc.shape[0]):
        for i in range(auc.shape[1]):
            src_r = f"{_fmt_edge(src_edges[i], 4)}-{_fmt_edge(src_edges[i+1], 4)}"
            tgt_r = f"{_fmt_edge(tgt_edges[j], 4)}-{_fmt_edge(tgt_edges[j+1], 4)}"
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


REPORT_HEADER = [
    "# Lead 4: Node Sign-Entropy Heterogeneity vs. AUC (canonical, shared edges)",
    "",
    "## Methodology",
    "",
    "For each shared test edge `(u, v)`, bin source-node sign-entropy x",
    "target-node sign-entropy into a 2D grid and compute AUC within each cell,",
    "for 4 models: `walk_full` (E14_HARDNODE_L10, full attention),",
    "`walk_localattn4` (E14_HARDNODE_L10_LOCALATTN4, +-2 hop banded attention),",
    "`GINEConv`, `SiGAT`.",
    "",
    "**Same-edge ground truth.** Per-edge predictions are read from",
    "`predictions_raw_canonical.pkl` (built by `baselines/postprocess_canonical.py`",
    "after the canonical-split baseline reruns); every model lives in the same raw",
    "`(u, v)` id space. We restrict all models to the **shared edge set** (the",
    "intersection of `(u, v)`, which equals the walk-covered test edges -- walk is",
    "a strict subset of the GNN test sets). So every cell compares the four models",
    "on the *identical* edges with the *identical* ground truth, and the per-cell",
    "sample count `n` is the **same for all four models**. (This is the fix for the",
    "old version, which bucketed each model over its own ~independent test sample;",
    "see CANONICAL_RERUN_FINDINGS.md.)",
    "",
    "**Entropy** is binary Shannon entropy in bits over a node's sign labels,",
    "`H(p) = -p*log2(p) - (1-p)*log2(1-p)`, `p` = fraction of positive-sign edges in",
    "the relevant direction. `H=0` -> homogeneous, `H=1` -> maximally heterogeneous.",
    "Computed over **all edges** of the dataset (train+val+test) from the canonical",
    "edge list -- a diagnostic grouping of already-trained models' predictions, not a",
    "training-time feature, so no leakage. Nodes with zero edges in the relevant",
    "direction are dropped.",
    "",
    "**Entropy variants** -- source/target entropy from out-edges, in-edges, or",
    "in+out (combined) signs of that node:",
    "",
    "| variant | source | target |",
    "|---|---|---|",
    "| `out_out` | H(out-signs of u) | H(out-signs of v) |",
    "| `in_in` | H(in-signs of u) | H(in-signs of v) |",
    "| `out_in` | H(out-signs of u) | H(in-signs of v) |",
    "| `inout_inout` | H(all incident signs of u) | H(all incident signs of v) |",
    "",
    "**No retraining / recomputation here**: predictions are read straight from the",
    "canonical pkl (walk-model func_logit_power aggregation, GINEConv saved",
    "predictions, and the SiGAT logistic read-out all happened upstream). This",
    "script only intersects edges, computes entropy, bins, and plots.",
    "",
    "Gray cells (`n=.. (<min_n)` / `n=.. (1 class)` / empty) have no defined AUC.",
    "Raw per-cell numbers are also dumped to `raw_data.txt`. Per-edge records",
    f"`(src_ent, tgt_ent, y, p)` are cached in `{DATA_FILENAME}` for re-binning via",
    "`--mode plot`.",
    "",
    "## Results",
    "",
]


def plot_all(data, variants, out_dir, n_buckets_list, min_cell_n, cmap, vmin, vmax, dpi,
             binning, combined):
    text_lines = [
        "LEAD 4: NODE SIGN-ENTROPY HETEROGENEITY vs. AUC -- raw per-cell data (canonical, shared edges)",
        "Same numbers as the PNGs / report.md tables, in plain text.",
        "",
    ]
    report_lines = list(REPORT_HEADER)

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
            summary_grids = None
            for n_buckets in n_buckets_list:
                grids = {}
                for model in MODELS:
                    rec = ds_data[variant].get(model)
                    grids[model] = None if rec is None else auc_grid(
                        rec["src_ent"], rec["tgt_ent"], rec["y"], rec["p"], n_buckets, min_cell_n, binning)
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
                        grids[model] = None if rec is None else auc_grid(
                            rec["src_ent"], rec["tgt_ent"], rec["y"], rec["p"], n_buckets, min_cell_n, "fixed")
                    ds_grids[ds_name] = grids
                suffix = f"_fixed_b{n_buckets}"
                plot_variant_combined(variant, ds_grids, combined_out, cmap, vmin, vmax, dpi,
                                       min_cell_n, suffix)
                report_lines.append(f"![ALL_DATASETS {variant} b{n_buckets}]"
                                     f"(combined/ALL_DATASETS_{variant}{suffix}.png)")
                report_lines.append("")

    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n✓ Report written to {os.path.join(out_dir, 'report.md')}")
    with open(os.path.join(out_dir, "raw_data.txt"), "w") as f:
        f.write("\n".join(text_lines))
    print(f"✓ Raw text data written to {os.path.join(out_dir, 'raw_data.txt')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compute", "plot", "all"], default="all")
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    parser.add_argument("--out", default=OUT_DIR_DEFAULT)
    parser.add_argument("--predictions", default=CANON_PREDICTIONS_DEFAULT,
                         help="canonical per-edge predictions pkl (from baselines/postprocess_canonical.py)")
    parser.add_argument("--data-path", default=None,
                         help=f"computed-data pickle (default: <out>/{DATA_FILENAME})")
    parser.add_argument("--min-total", type=int, default=MIN_TOTAL_DEFAULT)
    parser.add_argument("--binning", choices=["quantile", "fixed"], default="fixed")
    parser.add_argument("--n-buckets", type=int, nargs="+", default=N_BUCKETS_DEFAULT,
                         help="bucket counts to sweep (start from 2)")
    parser.add_argument("--min-cell-n", type=int, default=MIN_CELL_N_DEFAULT)
    parser.add_argument("--cmap", default="RdYlGn")
    parser.add_argument("--vmin", type=float, default=0.5)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--no-combined", action="store_true")
    args = parser.parse_args()

    datasets = DATASETS if args.datasets == ["all"] else args.datasets
    data_path = args.data_path or os.path.join(args.out, DATA_FILENAME)

    if args.mode in ("compute", "all"):
        predictions = load_shared_predictions(args.predictions, datasets)
        data = compute_all(predictions, datasets, args.variants, args.min_total)
        save_data(data, data_path)
    if args.mode == "plot":
        data = load_data(data_path)
        if args.datasets != ["all"]:
            data = {k: v for k, v in data.items() if k in datasets}

    if args.mode in ("plot", "all"):
        plot_all(data, args.variants, args.out, args.n_buckets, args.min_cell_n,
                  args.cmap, args.vmin, args.vmax, args.dpi, args.binning, not args.no_combined)


if __name__ == "__main__":
    main()
