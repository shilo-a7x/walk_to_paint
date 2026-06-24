"""
Lead 4b -- 2-hop sign-path-consistency heterogeneity vs. AUC.

Same spirit as lead4_entropy_heterogeneity.py (binned AUC vs. a heterogeneity
score, with compute/plot stages separated so plot styling can be iterated on
without recomputation) but with a DIFFERENT heterogeneity definition, and
ONE value per EDGE rather than a (source, target) pair feeding a 2D grid.

For a test edge (u, v): the edge's heterogeneity score is computed entirely
from the TARGET node v's own 2-hop out-paths. Every (v -> m) out-edge
followed by every (m -> k) out-edge of m forms a 2-hop path v->m->k. The
path is consistent if both edges share the same sign (+/+ or -/-),
inconsistent otherwise (+/- or -/+). Worked example: v has out-edges (v,x)
negative and (v,y) positive; x has 3 out-edges (2 positive, 1 negative); y
has 2 out-edges (2 positive). v has 3+2=5 two-hop paths total: via x,
(neg,pos) x2 inconsistent + (neg,neg) x1 consistent; via y, (pos,pos) x2
consistent. So 3/5 of v's two-hop paths are consistent -> p=3/5.
p = consistent-fraction, turned into the SAME binary Shannon entropy used by
lead4_entropy_heterogeneity.py: H(p) = -p*log2(p) - (1-p)*log2(1-p).
H=0 -> a path's second-hop sign is fully predictable from the first hop
(locally "balanced" 2-hop neighborhood around v), H=1 -> maximally
unpredictable (50/50 consistent/inconsistent). Edges whose target v has zero
2-hop out-paths (no out-edges, or none of v's out-neighbors have any
out-edges themselves) are dropped -- no signal available, same convention as
the original script.

Computed over ALL edges of the dataset (train+val+test), for the same
no-leakage reasoning as lead4_entropy_heterogeneity.py: this is a diagnostic
grouping of already-trained models' predictions, not a training-time
feature.

One dimension, not two: unlike the original entropy's out/in/inout x out/in
variant menu and its (source, target) 2D grid, this measure is intrinsically
edge-level (anchored at the target) -- so the AUC-vs-heterogeneity plot here
is a 1D binned bar chart (one bar group per entropy bucket, one bar per
model), not a heatmap.

Reuses already-computed per-edge predictions: NO model re-loading, NO
walk-occurrence aggregation, NO SiGAT classifier refit. Reads
predictions_raw.pkl produced by `lead4_entropy_heterogeneity.py --mode
compute` (or `all`) -- outputs/lead4_entropy_heterogeneity/predictions_raw.pkl
by default -- which already holds, per (dataset, model), per-edge (u, v, y,
p) in that model's own node-id space (raw tokenizer ids for
walk_full/walk_localattn4, dense baselines/splits ids for GINEConv/SiGAT).
Only the all-edges adjacency needed for the new heterogeneity score is built
from scratch here, once per (dataset, id-space) and shared across the 2
models that use each space.

    # compute per-edge (ent, y, p) once, save to disk
    python scripts/lead4_twohop_path_consistency.py --mode compute --datasets all

    # replot from saved data, as many times as you like, with new styling
    python scripts/lead4_twohop_path_consistency.py --mode plot --n-buckets 5 --min-cell-n 15

    # default: do both in one go
    python scripts/lead4_twohop_path_consistency.py --mode all --datasets all
"""
import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.node_mi_structural_embedding import DATASET_CONFIGS, load_edges_canonical
from scripts.lead4_entropy_heterogeneity import (
    DATASETS, MODELS, _ds_key, bin_edges, digitize,
    OUT_DIR_DEFAULT as LEAD4_OUT_DIR_DEFAULT,
    PREDICTIONS_FILENAME as LEAD4_PREDICTIONS_FILENAME,
)

OUT_DIR_DEFAULT = os.path.join(ROOT, "outputs", "lead4_twohop_path_consistency")
DATA_FILENAME = "computed_data.pkl"
MIN_TOTAL_DEFAULT = 40
MIN_CELL_N_DEFAULT = 20


# ── 2-hop path-consistency entropy (per node, looked up by target only) ──────

def build_out_adj(edge_triples):
    """edge_triples: iterable of (u, v, sign) -> {u: [(v, sign), ...]}."""
    out_adj = {}
    for u, v, s in edge_triples:
        out_adj.setdefault(u, []).append((v, s))
    return out_adj


def binary_entropy_from_p(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def twohop_consistency_entropy(out_adj):
    """{node: entropy} for every node with >=1 two-hop out-path (n->m->k),
    per the module docstring's worked example. Nodes with zero two-hop paths
    are dropped entirely (no key) -- no signal available. Used here ONLY as
    the target-node lookup for an edge's heterogeneity score."""
    entropy = {}
    for n, edges1 in out_adj.items():
        total, consistent = 0, 0
        for m, s1 in edges1:
            for _, s2 in out_adj.get(m, []):
                total += 1
                if s1 == s2:
                    consistent += 1
        if total == 0:
            continue
        entropy[n] = binary_entropy_from_p(consistent / total)
    return entropy


def get_dataset_adjacencies(ds_name):
    """Returns (raw_adj, dense_adj): all-edges out-adjacency in the two
    node-id spaces used across the 4 models -- raw tokenizer ids
    (walk_full/walk_localattn4, from the canonical edge list) and dense
    baselines/splits ids (GINEConv/SiGAT, from baselines/splits/<ds>.pt) --
    built once per dataset, reused for both models sharing each space."""
    cfg = DATASET_CONFIGS[_ds_key(ds_name)]
    raw_edges = load_edges_canonical(cfg["ds_name"])
    raw_adj = build_out_adj(raw_edges)

    splits_path = os.path.join(ROOT, "baselines", "splits", f"{ds_name}.pt")
    splits = torch.load(splits_path, weights_only=False)
    ei, ew = splits["edge_index"], splits["edge_weight"]
    dense_triples = list(zip(ei[0].tolist(), ei[1].tolist(), ew.int().tolist()))
    dense_adj = build_out_adj(dense_triples)

    return raw_adj, dense_adj


ID_SPACE = {
    "walk_full": "raw", "walk_localattn4": "raw",
    "GINEConv": "dense", "SiGAT": "dense",
}


# ── Compute stage: raw per-edge (ent, y, p) records ───────────────────────────
# ent is a property of the EDGE, taken entirely from its target node v's
# 2-hop path-consistency entropy -- u plays no role in this measure.

def collect_model_records(uvyp, ent_lookup, min_total):
    v, y, p = uvyp["v"], uvyp["y"], uvyp["p"]
    ent, yy, pp = [], [], []
    for vi, yi, pi in zip(v, y, p):
        e = ent_lookup.get(int(vi))
        if e is None:
            continue
        ent.append(e); yy.append(yi); pp.append(pi)
    if len(yy) < min_total:
        return None
    return {"ent": np.array(ent), "y": np.array(yy), "p": np.array(pp)}


def compute_all(datasets, predictions_path, min_total):
    with open(predictions_path, "rb") as f:
        predictions = pickle.load(f)

    data = {}
    for ds_name in datasets:
        print(f"\n{'=' * 72}\n{ds_name}\n{'=' * 72}")
        if ds_name not in predictions:
            print(f"  no cached predictions for {ds_name} in {predictions_path}, skipping")
            data[ds_name] = {}
            continue

        raw_adj, dense_adj = get_dataset_adjacencies(ds_name)
        ent_by_space = {
            "raw": twohop_consistency_entropy(raw_adj),
            "dense": twohop_consistency_entropy(dense_adj),
        }

        data[ds_name] = {}
        for model in MODELS:
            uvyp = predictions[ds_name].get(model)
            if uvyp is None:
                data[ds_name][model] = None
                print(f"  {model}: no data")
                continue
            rec = collect_model_records(uvyp, ent_by_space[ID_SPACE[model]], min_total)
            if rec is None:
                print(f"  {model}: insufficient samples after entropy lookup")
            else:
                auc = roc_auc_score(rec["y"], rec["p"])
                print(f"  {model}: n={len(rec['y'])}, overall_auc={auc:.4f}")
            data[ds_name][model] = rec

    return data


def save_data(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"\n✓ Computed data saved to {path}")


def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Plot stage: 1D binning + per-bucket AUC + bar charts + report ────────────
# bin_edges/digitize (quantile vs. fixed edges) are imported unchanged from
# lead4_entropy_heterogeneity.py; everything below is 1D since this
# heterogeneity score is a single number per edge, not a (source, target) pair.

def auc_by_bin(ent, y, p, n_buckets, min_n, binning):
    edges = bin_edges(ent, n_buckets, binning)
    bins = digitize(ent, edges)
    n_bins = len(edges) - 1

    auc = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    y, p = np.asarray(y), np.asarray(p)
    for i in range(n_bins):
        mask = bins == i
        counts[i] = int(mask.sum())
        if counts[i] >= min_n and len(np.unique(y[mask])) == 2:
            auc[i] = roc_auc_score(y[mask], p[mask])
    return auc, counts, edges


MODEL_COLOR = {m: c for m, c in zip(MODELS, plt.cm.tab10(np.linspace(0, 1, len(MODELS))))}


def _annotate_bar(ax, x, auc_val, count, min_n, top):
    """Mirrors lead4_entropy_heterogeneity.py's _annotate_cell three cases,
    adapted to a bar (label placed above the bar / baseline)."""
    if count == 0:
        return
    if not np.isnan(auc_val):
        ax.text(x, auc_val + 0.015, f"{auc_val:.2f}\n(n={count})", ha="center", va="bottom", fontsize=6)
    elif count < min_n:
        ax.text(x, 0.02, f"n={count}\n(<{min_n})", ha="center", va="bottom", fontsize=6, color="dimgray")
    else:
        ax.text(x, 0.02, f"n={count}\n(1 class)", ha="center", va="bottom", fontsize=6, color="dimgray")


def _draw_bars(ax, results, min_n, title=None, show_xlabels=True):
    """results: {model: (auc, counts, edges) or None}. Returns the shared bin
    edges (taken from whichever model has data) for the caller's x-tick labels."""
    models_present = [m for m in MODELS if results.get(m) is not None]
    if not models_present:
        ax.axis("off")
        return None
    edges_ref = results[models_present[0]][2]
    n_bins = len(edges_ref) - 1
    width = 0.8 / len(models_present)
    for mi, model in enumerate(models_present):
        auc, counts, _ = results[model]
        x = np.arange(n_bins) + (mi - (len(models_present) - 1) / 2) * width
        heights = np.nan_to_num(auc, nan=0.0)
        ax.bar(x, heights, width=width, color=MODEL_COLOR[model], label=model)
        for xi, a, c in zip(x, auc, counts):
            _annotate_bar(ax, xi, a, c, min_n, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks(range(n_bins))
    if show_xlabels:
        ax.set_xticklabels([f"{edges_ref[i]:.2f}-{edges_ref[i+1]:.2f}" for i in range(n_bins)],
                            fontsize=7, rotation=30, ha="right")
    else:
        ax.set_xticklabels([])
    if title:
        ax.set_title(title, fontsize=9)
    return edges_ref


def plot_models(ds_name, results, out_dir, dpi, min_n, suffix=""):
    """results: {model: (auc, counts, edges) or None}."""
    if not any(v is not None for v in results.values()):
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _draw_bars(ax, results, min_n, title=f"{ds_name}: AUC vs. target-node 2-hop path-consistency entropy")
    ax.set_xlabel("target-node 2-hop path-consistency entropy (bits)")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save_path = os.path.join(out_dir, f"{ds_name}{suffix}.png")
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_combined(ds_results, out_dir, dpi, min_n, suffix=""):
    """ds_results: {dataset: {model: (auc, counts, edges) or None}}. Caller
    must pass results computed with binning='fixed' so bucket edges (and
    therefore x-axis ranges) are comparable across the dataset subplots."""
    datasets_present = [d for d in ds_results if any(v is not None for v in ds_results[d].values())]
    if not datasets_present:
        return
    n_cols = 2
    n_rows = (len(datasets_present) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 3.6 * n_rows), squeeze=False)
    for idx, ds_name in enumerate(datasets_present):
        r, c = divmod(idx, n_cols)
        _draw_bars(axes[r][c], ds_results[ds_name], min_n, title=ds_name)
    for idx in range(len(datasets_present), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")
    handles, labels = next(
        ax.get_legend_handles_labels() for row in axes for ax in row if ax.has_data()
    )
    fig.legend(handles, labels, loc="upper center", ncol=len(MODELS), fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("All datasets: AUC vs. target-node 2-hop path-consistency entropy", y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_path = os.path.join(out_dir, f"ALL_DATASETS{suffix}.png")
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  saved {save_path}")


def low_high_summary_md(results):
    """Markdown table: lowest-entropy bucket vs. highest-entropy bucket AUC,
    per model."""
    lines = [
        "| model | low-entropy AUC (n) | high-entropy AUC (n) | drop (low - high) |",
        "|---|---|---|---|",
    ]
    for model in MODELS:
        if results.get(model) is None:
            lines.append(f"| {model} | no data | no data | n/a |")
            continue
        auc, counts, _ = results[model]
        lo, hi = auc[0], auc[-1]
        lo_s = f"{lo:.4f}" if not np.isnan(lo) else "n/a"
        hi_s = f"{hi:.4f}" if not np.isnan(hi) else "n/a"
        drop = (lo - hi) if (not np.isnan(lo) and not np.isnan(hi)) else float("nan")
        drop_s = f"{drop:+.4f}" if not np.isnan(drop) else "n/a"
        lines.append(f"| {model} | {lo_s} (n={counts[0]}) | {hi_s} (n={counts[-1]}) | {drop_s} |")
    lines.append("")
    return lines


def bins_to_text(ds_name, n_buckets, binning, model, auc, counts, edges, min_n):
    """Plain-text dump of one (dataset, bucket-count, model) bar chart -- same
    data as the PNG, for grepping/diffing instead of eyeballing pictures."""
    lines = [
        f"=== {ds_name} | binning={binning} | n_buckets={n_buckets} | model={model} ===",
        "entropy_range\tauc\tn\tnote",
    ]
    for i in range(len(auc)):
        r = f"{edges[i]:.4f}-{edges[i+1]:.4f}"
        c = int(counts[i])
        a = auc[i]
        if c == 0:
            a_s, note = "", "empty"
        elif np.isnan(a):
            a_s, note = "", ("below_min_n" if c < min_n else "single_class")
        else:
            a_s, note = f"{a:.4f}", ""
        lines.append(f"{r}\t{a_s}\t{c}\t{note}")
    lines.append("")
    return lines


def plot_all(data, out_dir, n_buckets_list, min_cell_n, dpi, binning, combined):
    text_lines = [
        "LEAD 4b: 2-HOP SIGN-PATH-CONSISTENCY ENTROPY (PER EDGE, TARGET-NODE-ONLY) vs. AUC",
        "Same numbers as the PNGs / report.md tables, in plain text. See report.md",
        "for methodology.",
        "",
    ]
    report_lines = [
        "# Lead 4b: 2-Hop Sign-Path-Consistency Entropy vs. AUC",
        "",
        "## Methodology",
        "",
        "Part of Lead 4 (see lead4_entropy_heterogeneity.py / report.md for the",
        "original sign-entropy version, which used a 2D source x target grid). This",
        "variant gives EACH TEST EDGE a single heterogeneity score, taken entirely",
        "from its TARGET node `v`'s own 2-hop out-paths -- `u` plays no role.",
        "",
        "**2-hop path-consistency entropy**: for the target node `v` of edge",
        "`(u, v)`, every `(v->m)` out-edge followed by every `(m->k)` out-edge of",
        "`m` forms a 2-hop path. A path is *consistent* if both edges share the",
        "same sign (`+/+` or `-/-`), *inconsistent* otherwise (`+/-` or `-/+`).",
        "`p` = fraction of `v`'s 2-hop paths that are consistent, turned into the",
        "same binary Shannon entropy as the original Lead 4:",
        "`H(p) = -p*log2(p) - (1-p)*log2(1-p)`. `H=0` -> a path's second-hop sign",
        "is fully predictable from the first hop (locally \"balanced\" 2-hop",
        "neighborhood around `v`), `H=1` -> maximally unpredictable (50/50",
        "consistent/inconsistent). Worked example: `v` has out-edges `(v,x)`",
        "negative and `(v,y)` positive; `x` has 3 out-edges (2 positive, 1",
        "negative); `y` has 2 out-edges (2 positive) -- 5 two-hop paths total, 3",
        "consistent (1 via `x`, 2 via `y`), so `p=3/5`. Edges whose target has zero",
        "2-hop out-paths are dropped (no signal). Computed over **all edges** of",
        "the dataset (train+val+test) -- a diagnostic grouping of already-trained",
        "models' predictions, not a training-time feature, so there is no leakage",
        "concern.",
        "",
        "Because the score is intrinsically edge-level (anchored at the target,",
        "not a source/target pair), the AUC-vs-heterogeneity plot here is a 1D",
        "binned bar chart (one bucket group per entropy range, one bar per model)",
        "instead of the original script's 2D heatmap.",
        "",
        "**No retraining, no recomputation of model predictions**: this script reads",
        f"`{LEAD4_PREDICTIONS_FILENAME}` (default location:",
        f"`{os.path.join(LEAD4_OUT_DIR_DEFAULT, LEAD4_PREDICTIONS_FILENAME)}`),",
        "produced by `lead4_entropy_heterogeneity.py --mode compute`, which already",
        "has per-edge `(u, v, y, p)` for all 4 models in their own node-id space",
        "(raw tokenizer ids for the walk models, dense `baselines/splits` ids for",
        "GINEConv/SiGAT). Only the all-edges adjacency needed for the new",
        "heterogeneity score is built here, once per (dataset, id-space).",
        "",
        f"**Binning**: `binning={binning}`, swept over bucket counts",
        f"{n_buckets_list} (filenames carry a `_{{binning}}_b{{n}}` suffix); a",
        f"bucket needs >= {min_cell_n} samples and both classes present to get an",
        "AUC. `fixed` binning uses the same global `[0, 1]`-bit edges for every",
        "dataset, so bar charts -- including the combined all-datasets plot -- are",
        "directly comparable at the cost of uneven sample counts per bucket;",
        "`quantile` binning gives equal sample counts per bucket but different",
        "entropy ranges per dataset.",
        "",
        "Bars annotated `n=.. (<min_n)` are below the minimum-sample threshold;",
        "`n=.. (1 class)` means enough samples but only one true sign present, so",
        "AUC is undefined -- not a bug. Missing bars/labels mean zero samples in",
        "that bucket.",
        "",
        "**Raw text data**: every bucket's numbers are also dumped as plain",
        "tab-separated text in `raw_data.txt` next to this report.",
        "",
        "**Reproducing / extending this analysis**: raw per-edge records",
        "`(ent, y, p)` for every (dataset, model) are cached in",
        f"`{DATA_FILENAME}` next to this report. `--mode compute` (re-reads",
        f"`{LEAD4_PREDICTIONS_FILENAME}`, recomputes adjacency/entropy/binning);",
        "`--mode plot` re-bins/restyles without recomputation.",
        "",
        "## Results",
        "",
    ]

    combined_out = os.path.join(out_dir, "combined")
    if combined:
        os.makedirs(combined_out, exist_ok=True)

    for ds_name, ds_data in data.items():
        if not ds_data:
            continue
        ds_out = os.path.join(out_dir, ds_name)
        os.makedirs(ds_out, exist_ok=True)
        print(f"\n{'=' * 72}\n{ds_name}\n{'=' * 72}")
        report_lines.append(f"### {ds_name}")
        report_lines.append("")

        summary_results = None  # use the first (smallest) bucket count for the summary table
        for n_buckets in n_buckets_list:
            results = {}
            for model in MODELS:
                rec = ds_data.get(model)
                if rec is None:
                    results[model] = None
                    continue
                results[model] = auc_by_bin(rec["ent"], rec["y"], rec["p"], n_buckets, min_cell_n, binning)
            suffix = f"_{binning}_b{n_buckets}"
            plot_models(ds_name, results, ds_out, dpi, min_cell_n, suffix)
            report_lines.append(f"![{ds_name} b{n_buckets}]({ds_name}/{ds_name}{suffix}.png)")
            report_lines.append("")
            for model in MODELS:
                if results.get(model) is None:
                    continue
                auc, counts, edges = results[model]
                text_lines.extend(bins_to_text(ds_name, n_buckets, binning, model,
                                                auc, counts, edges, min_cell_n))
            if summary_results is None:
                summary_results = results
        if summary_results is not None:
            report_lines.extend(low_high_summary_md(summary_results))

    if combined:
        print(f"\n{'=' * 72}\ncombined (all datasets)\n{'=' * 72}")
        report_lines.append("### Combined (all datasets)")
        report_lines.append("")
        for n_buckets in n_buckets_list:
            ds_results = {}
            for ds_name, ds_data in data.items():
                if not ds_data:
                    continue
                results = {}
                for model in MODELS:
                    rec = ds_data.get(model)
                    if rec is None:
                        results[model] = None
                        continue
                    # combined plot always uses 'fixed' binning regardless of the
                    # per-dataset --binning choice -- quantile edges differ per
                    # dataset and would make subplots visually incomparable.
                    results[model] = auc_by_bin(rec["ent"], rec["y"], rec["p"], n_buckets, min_cell_n, "fixed")
                ds_results[ds_name] = results
            suffix = f"_fixed_b{n_buckets}"
            plot_combined(ds_results, combined_out, dpi, min_cell_n, suffix)
            report_lines.append(f"![ALL_DATASETS b{n_buckets}](combined/ALL_DATASETS{suffix}.png)")
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
    parser.add_argument("--out", default=OUT_DIR_DEFAULT)
    parser.add_argument("--data-path", default=None,
                         help=f"path to computed-data pickle (default: <out>/{DATA_FILENAME})")
    parser.add_argument("--predictions-path", default=None,
                         help="path to lead4_entropy_heterogeneity.py's cached per-edge "
                              f"predictions (default: {os.path.join(LEAD4_OUT_DIR_DEFAULT, LEAD4_PREDICTIONS_FILENAME)})")
    parser.add_argument("--min-total", type=int, default=MIN_TOTAL_DEFAULT,
                         help="drop a (dataset, model) entirely below this many usable edges (compute stage)")
    parser.add_argument("--binning", choices=["quantile", "fixed"], default="fixed",
                         help="'fixed' (default) uses global [0,1]-bit edges, comparable across "
                              "datasets; 'quantile' uses per-dataset equal-sample-count edges")
    parser.add_argument("--n-buckets", type=int, nargs="+", default=[8],
                         help="one or more bucket counts to sweep, each producing its own plot set")
    parser.add_argument("--min-cell-n", type=int, default=MIN_CELL_N_DEFAULT)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--no-combined", action="store_true",
                         help="skip the all-datasets combined plot")
    args = parser.parse_args()

    datasets = DATASETS if args.datasets == ["all"] else args.datasets
    data_path = args.data_path or os.path.join(args.out, DATA_FILENAME)
    predictions_path = args.predictions_path or os.path.join(
        LEAD4_OUT_DIR_DEFAULT, LEAD4_PREDICTIONS_FILENAME)

    if args.mode in ("compute", "all"):
        if not os.path.exists(predictions_path):
            raise FileNotFoundError(
                f"{predictions_path} not found -- run "
                "`python scripts/lead4_entropy_heterogeneity.py --mode compute` first; "
                "this script reuses its cached per-edge predictions instead of "
                "reloading models / refitting SiGAT itself."
            )
        data = compute_all(datasets, predictions_path, args.min_total)
        save_data(data, data_path)
    if args.mode == "plot":
        data = load_data(data_path)
        if args.datasets != ["all"]:
            data = {k: v for k, v in data.items() if k in datasets}

    if args.mode in ("plot", "all"):
        plot_all(data, args.out, args.n_buckets, args.min_cell_n,
                  args.dpi, args.binning, not args.no_combined)


if __name__ == "__main__":
    main()
