"""
Lead 4b -- 2-hop sign-path-consistency heterogeneity vs. AUC (canonical, shared-edge).

Same spirit as lead4_entropy_heterogeneity.py (binned AUC vs. a heterogeneity
score, compute/plot stages separated so plot styling can be iterated without
recomputation) but with a DIFFERENT heterogeneity definition and ONE value per
EDGE (anchored at the target node v) instead of a (source, target) pair feeding a
2D grid -- so the plot is a 1D binned bar chart (one bucket group per entropy
range, one bar per model) rather than a heatmap.

Same-edge ground truth, same machinery as Lead 4: per-edge predictions are read
from `predictions_raw_canonical.pkl` and restricted to the SHARED edge set via
`lead4_entropy_heterogeneity.load_shared_predictions`, so every model is bucketed
over the *identical* edges. A direct consequence: for a given bucket the sample
count `n` is the SAME for all models (same edges, same target-node entropy
lookup), so `n` is shown ONCE under each x-axis tick instead of per bar.

**2-hop path-consistency entropy**, for edge `(u, v)`:
  - `out`   : forward 2-hop paths from the TARGET `v -> m -> k` (using only
              out-edges, twice). Consistent if `sign(v->m) == sign(m->k)`.
  - `in`    : backward 2-hop paths into the SOURCE `s -> t -> u` (using only
              in-edges, twice -- i.e. who points at `u`, and who points at
              that). Consistent if `sign(s->t) == sign(t->u)`.
  - `inout` : pool the `out` counts (from `v`) and the `in` counts (into `u`)
              for this specific edge into ONE (total, consistent) tally before
              computing entropy -- not a separate undirected traversal.
A path is *consistent* if both its edges share the same sign (`+/+` or `-/-`).
`p` = consistent fraction, turned into binary Shannon entropy
`H(p) = -p*log2(p) - (1-p)*log2(1-p)`: `H=0` -> a path's second-hop sign is
fully predictable from the first (locally "balanced"), `H=1` -> maximally
unpredictable. `out`/`in` are per-node lookups (`v` / `u` respectively);
`inout` is necessarily per-EDGE since it mixes counts from both endpoints.
Edges with zero applicable 2-hop paths (for that variant) are dropped.
Each variant gets its own bar-chart set + combined plot.

Computed over ALL edges of the dataset (train+val+test) from the canonical edge
list (raw ids -- same id space the predictions live in), a diagnostic grouping of
already-trained models' predictions, no leakage. No model reloading / no SiGAT
refit: those happened upstream; this only builds adjacency, bins, and plots.

    # compute per-edge (ent, y, p) for every variant, save to disk
    python scripts/lead4_twohop_path_consistency.py --mode compute --datasets all

    # replot from saved data with new styling / bucket sweep
    python scripts/lead4_twohop_path_consistency.py --mode plot --n-buckets 2 3 4

    # default: both
    python scripts/lead4_twohop_path_consistency.py --mode all --datasets all
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.node_mi_structural_embedding import DATASET_CONFIGS, load_edges_canonical
from scripts import lead4_entropy_heterogeneity as L4
from scripts.lead4_entropy_heterogeneity import (
    DATASETS, MODELS, WALK_MODELS, _ds_key, _fmt_signed, _fmt_edge,
    load_shared_predictions, save_data, load_data,
    CANON_PREDICTIONS_DEFAULT, N_BUCKETS_DEFAULT,
)

VARIANTS = ["out", "in", "inout"]
OUT_DIR_DEFAULT = os.path.join(ROOT, "outputs", "lead4_twohop_path_consistency")
DATA_FILENAME = "computed_data.pkl"
MIN_TOTAL_DEFAULT = 40
MIN_CELL_N_DEFAULT = 20

# walk = cool (blues), GNN = warm (oranges) -- the walk-vs-GNN contrast we care about
MODEL_COLOR = {
    "walk_full":       "#08519c",  # dark blue
    "walk_localattn4": "#6baed6",  # light blue
    "GINEConv":        "#a63603",  # dark orange
    "SiGAT":           "#fd8d3c",  # light orange
}


def _model_label(model):
    return f"{model} ({'walk' if model in WALK_MODELS else 'GNN'})"


# ── 2-hop path-consistency counts ────────────────────────────────────────────

def build_adj_out(edge_triples):
    """{u: [(v, s)]} -- out-edges. Used to walk forward from a node."""
    adj = {}
    for u, v, s in edge_triples:
        adj.setdefault(u, []).append((v, s))
    return adj


def build_adj_in(edge_triples):
    """{v: [(u, s)]} -- in-edges (predecessors). Used to walk backward into a
    node, i.e. n's in-adjacency entry for `n` lists `(predecessor, sign)`."""
    adj = {}
    for u, v, s in edge_triples:
        adj.setdefault(v, []).append((u, s))
    return adj


def binary_entropy_from_p(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def twohop_counts(adj):
    """{node: (total, consistent)} for every node with >=1 two-hop path
    n -> m -> k traversed via `adj` (works identically whether `adj` is
    out-adjacency, giving forward paths, or in-adjacency, giving backward
    paths -- consistency only compares the two hop signs). Nodes with zero
    two-hop paths are omitted."""
    counts = {}
    for n, edges1 in adj.items():
        total, consistent = 0, 0
        for m, s1 in edges1:
            for _, s2 in adj.get(m, []):
                total += 1
                if s1 == s2:
                    consistent += 1
        if total > 0:
            counts[n] = (total, consistent)
    return counts


def _entropy_from_count(c):
    if c is None or c[0] == 0:
        return None
    total, consistent = c
    return binary_entropy_from_p(consistent / total)


def _entropy_pooled(c1, c2):
    """Pool two (total, consistent) tallies into one entropy -- combine the
    counts BEFORE computing entropy, not an average of two entropies."""
    total = (c1[0] if c1 else 0) + (c2[0] if c2 else 0)
    if total == 0:
        return None
    consistent = (c1[1] if c1 else 0) + (c2[1] if c2 else 0)
    return binary_entropy_from_p(consistent / total)


def make_entropy_fns(out_counts, in_counts):
    """Per-edge (u, v) -> entropy, for each variant.
      out   : v's forward 2-hop consistency (v -> m -> k), anchored at TARGET.
      in    : u's backward 2-hop consistency (s -> t -> u), anchored at SOURCE.
      inout : pool v's out-counts and u's in-counts into ONE tally for this
              edge -- necessarily per-edge (mixes both endpoints' counts),
              unlike out/in which are really per-node lookups.
    """
    return {
        "out": lambda u, v: _entropy_from_count(out_counts.get(v)),
        "in": lambda u, v: _entropy_from_count(in_counts.get(u)),
        "inout": lambda u, v: _entropy_pooled(out_counts.get(v), in_counts.get(u)),
    }


# ── Compute stage ────────────────────────────────────────────────────────────

def collect_model_records(uvyp, entropy_fn, min_total):
    u, v, y, p = uvyp["u"], uvyp["v"], uvyp["y"], uvyp["p"]
    ent, yy, pp = [], [], []
    for ui, vi, yi, pi in zip(u, v, y, p):
        e = entropy_fn(int(ui), int(vi))
        if e is None:
            continue
        ent.append(e); yy.append(yi); pp.append(pi)
    if len(yy) < min_total:
        return None
    return {"ent": np.array(ent), "y": np.array(yy), "p": np.array(pp)}


def compute_all(predictions, datasets, variants, min_total):
    """predictions: shared-edge {ds: {model: {u,v,y,p}}}. One pair of 2-hop
    count tables (out-anchored, in-anchored) per dataset from the real
    canonical edge list, shared by all models. Returns
    data[ds][variant][model] = record | None."""
    data = {}
    for ds_name in datasets:
        print(f"\n{'=' * 72}\n{ds_name}\n{'=' * 72}")
        edges = load_edges_canonical(DATASET_CONFIGS[_ds_key(ds_name)]["ds_name"])
        out_counts = twohop_counts(build_adj_out(edges))
        in_counts = twohop_counts(build_adj_in(edges))
        entropy_fns = make_entropy_fns(out_counts, in_counts)
        data[ds_name] = {var: {} for var in variants}
        for var in variants:
            fn = entropy_fns[var]
            for model in MODELS:
                r = predictions.get(ds_name, {}).get(model)
                if r is None or not len(r["u"]):
                    data[ds_name][var][model] = None
                    continue
                rec = collect_model_records(r, fn, min_total)
                data[ds_name][var][model] = rec
                if rec is not None and var == variants[0] and model == MODELS[0]:
                    auc = roc_auc_score(rec["y"], rec["p"]) if len(set(rec["y"])) > 1 else float("nan")
                    print(f"  [{var}] {model}: n={len(rec['y'])}, auc={auc:.4f}")
    return data


# ── Plot stage: 1D binning + per-bucket AUC + bar charts + report ─────────────

def auc_by_bin(ent, y, p, n_buckets, min_n, binning):
    edges = L4.bin_edges(ent, n_buckets, binning)
    bins = L4.digitize(ent, edges)
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


def _ylo(results, models_present):
    """y-axis bottom: 0.5, dropped only as far as needed to show any sub-0.5
    bucket (so 'start from 0.5' holds in the common all-good case but real
    below-chance buckets aren't silently clipped)."""
    finite = [a for m in models_present for a in results[m][0] if not np.isnan(a)]
    return min(0.5, min(finite) - 0.02) if finite else 0.5


def _draw_bars(ax, results, min_n, title=None, show_xlabels=True):
    """results: {model: (auc, counts, edges) | None}. Bars grow from a 0.5
    baseline (height = AUC - 0.5). n is identical across models (shared edges),
    so it is shown ONCE under each x tick. Returns shared bin edges."""
    models_present = [m for m in MODELS if results.get(m) is not None]
    if not models_present:
        ax.axis("off")
        return None
    edges_ref = results[models_present[0]][2]
    counts_ref = results[models_present[0]][1]  # same for every model
    n_bins = len(edges_ref) - 1
    width = 0.8 / len(models_present)
    for mi, model in enumerate(models_present):
        auc, _, _ = results[model]
        x = np.arange(n_bins) + (mi - (len(models_present) - 1) / 2) * width
        heights = np.where(np.isnan(auc), 0.0, auc - 0.5)
        ax.bar(x, heights, width=width, bottom=0.5, color=MODEL_COLOR[model],
               label=_model_label(model))
        for xi, a in zip(x, auc):
            if not np.isnan(a):
                ax.text(xi, max(a, 0.5) + 0.004, f"{a:.2f}", rotation=90,
                        ha="center", va="bottom", fontsize=5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(_ylo(results, models_present), 1.04)
    ax.set_xticks(range(n_bins))
    if show_xlabels:
        ax.set_xticklabels([f"{_fmt_edge(edges_ref[i])}-{_fmt_edge(edges_ref[i+1])}\n[n={counts_ref[i]}]"
                            for i in range(n_bins)], fontsize=6)
    else:
        ax.set_xticklabels([])
    if title:
        ax.set_title(title, fontsize=9)
    return edges_ref


def plot_models(ds_name, variant, results, out_dir, dpi, min_n, suffix=""):
    if not any(v is not None for v in results.values()):
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    _draw_bars(ax, results, min_n,
               title=f"{ds_name}: AUC vs. target-node 2-hop path-consistency entropy ({variant})")
    ax.set_xlabel("target-node 2-hop path-consistency entropy (bits)  [n = shared bucket size]")
    ax.set_ylabel("AUC (bars from 0.5 baseline)")
    ax.legend(fontsize=8, loc="lower left", title="walk = blue · GNN = orange", title_fontsize=8)
    fig.tight_layout()
    save_path = os.path.join(out_dir, f"{ds_name}_{variant}{suffix}.png")
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_combined(variant, ds_results, out_dir, dpi, min_n, suffix=""):
    datasets_present = [d for d in ds_results if any(v is not None for v in ds_results[d].values())]
    if not datasets_present:
        return
    n_cols = 2
    n_rows = (len(datasets_present) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3.8 * n_rows), squeeze=False)
    for idx, ds_name in enumerate(datasets_present):
        r, c = divmod(idx, n_cols)
        _draw_bars(axes[r][c], ds_results[ds_name], min_n, title=ds_name)
    for idx in range(len(datasets_present), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")
    handles, labels = next(
        ax.get_legend_handles_labels() for row in axes for ax in row if ax.has_data())
    fig.suptitle(f"All datasets: AUC vs. target-node 2-hop path-consistency entropy "
                 f"({variant} paths)  ·  n shown under each tick is shared across models", y=0.995)
    fig.legend(handles, labels, loc="upper center", ncol=len(MODELS), fontsize=9,
               title="walk = blue · GNN = orange", title_fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_path = os.path.join(out_dir, f"ALL_DATASETS_{variant}{suffix}.png")
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  saved {save_path}")


def low_high_summary_md(variant, results):
    lines = [
        f"**variant = `{variant}` paths**",
        "",
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
        lines.append(f"| {model} | {lo_s} (n={counts[0]}) | {hi_s} (n={counts[-1]}) | {_fmt_signed(drop)} |")
    lines.append("")
    return lines


def bins_to_text(ds_name, variant, n_buckets, binning, model, auc, counts, edges, min_n):
    lines = [
        f"=== {ds_name} | variant={variant} | binning={binning} | n_buckets={n_buckets} | model={model} ===",
        "entropy_range\tauc\tn\tnote",
    ]
    for i in range(len(auc)):
        r = f"{_fmt_edge(edges[i], 4)}-{_fmt_edge(edges[i+1], 4)}"
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


REPORT_HEADER = [
    "# Lead 4b: 2-Hop Sign-Path-Consistency Entropy vs. AUC (canonical, shared edges)",
    "",
    "## Methodology",
    "",
    "Part of Lead 4 (see `lead4_entropy_heterogeneity.py` / its report for the",
    "node sign-entropy version). This variant gives EACH shared test edge a",
    "heterogeneity score from 2-hop sign-path consistency, so the AUC-vs-",
    "heterogeneity plot is a 1D binned bar chart (not a 2D heatmap).",
    "",
    "**Same-edge ground truth.** Predictions are read from",
    "`predictions_raw_canonical.pkl` and restricted to the **shared edge set**",
    "(intersection of `(u, v)` across models = walk-covered edges) via",
    "`lead4_entropy_heterogeneity.load_shared_predictions`. So all models are",
    "bucketed over the *identical* edges -- and the per-bucket sample count `n` is",
    "the **same for every model**, shown once under each x-axis tick.",
    "",
    "**2-hop path-consistency entropy.** A path is *consistent* if both its edge",
    "signs match. `p` = consistent fraction, turned into binary Shannon entropy",
    "`H(p) = -p*log2(p) - (1-p)*log2(1-p)`. `H=0` -> second-hop sign fully",
    "predictable from the first (locally balanced), `H=1` -> maximally",
    "unpredictable. Computed over **all edges** (train+val+test) from the",
    "canonical edge list -- a diagnostic grouping, no leakage.",
    "",
    "**Path-direction variants**, for edge `(u, v)`:",
    "",
    "| variant | traversal | consistency | anchored at |",
    "|---|---|---|---|",
    "| `out` | forward, OUT-edges both hops (`v->m->k`) | `sign(v->m) == sign(m->k)` | target `v` |",
    "| `in` | backward, IN-edges both hops (`s->t->u`) | `sign(s->t) == sign(t->u)` | source `u` |",
    "| `inout` | pool `out`'s tally (from `v`) and `in`'s tally (into `u`) into ONE count for this edge, then take entropy | both | edge `(u,v)` |",
    "",
    "`out`/`in` are really per-node lookups (by `v` / `u` respectively); `inout`",
    "is necessarily per-edge since it mixes both endpoints' counts before taking",
    "entropy (not an average of two entropies). Edges with zero applicable 2-hop",
    "paths for a variant are dropped.",
    "",
    "Bars grow from a **0.5 baseline** (height = AUC - 0.5); the y-axis starts at",
    "0.5 unless a bucket dips below chance. Walk models are blue, GNNs orange.",
    "`n=.. (<min_n)` / `n=.. (1 class)` buckets have no defined AUC (no bar). Raw",
    f"numbers are dumped to `raw_data.txt`; per-edge `(ent, y, p)` cached in",
    f"`{DATA_FILENAME}` for re-binning via `--mode plot`.",
    "",
    "## Results",
    "",
]


def plot_all(data, out_dir, variants, n_buckets_list, min_cell_n, dpi, binning, combined):
    text_lines = [
        "LEAD 4b: 2-HOP SIGN-PATH-CONSISTENCY ENTROPY vs. AUC -- raw data (canonical, shared edges)",
        "Same numbers as the PNGs / report.md tables, in plain text.",
        "",
    ]
    report_lines = list(REPORT_HEADER)

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

        for variant in variants:
            if variant not in ds_data:
                continue
            report_lines.append(f"**{variant} paths**")
            report_lines.append("")
            summary_results = None
            for n_buckets in n_buckets_list:
                results = {}
                for model in MODELS:
                    rec = ds_data[variant].get(model)
                    results[model] = None if rec is None else auc_by_bin(
                        rec["ent"], rec["y"], rec["p"], n_buckets, min_cell_n, binning)
                suffix = f"_{binning}_b{n_buckets}"
                plot_models(ds_name, variant, results, ds_out, dpi, min_cell_n, suffix)
                report_lines.append(f"![{ds_name} {variant} b{n_buckets}]"
                                     f"({ds_name}/{ds_name}_{variant}{suffix}.png)")
                report_lines.append("")
                for model in MODELS:
                    if results.get(model) is None:
                        continue
                    auc, counts, edges = results[model]
                    text_lines.extend(bins_to_text(ds_name, variant, n_buckets, binning, model,
                                                    auc, counts, edges, min_cell_n))
                if summary_results is None:
                    summary_results = results
            if summary_results is not None:
                report_lines.extend(low_high_summary_md(variant, summary_results))

    if combined:
        print(f"\n{'=' * 72}\ncombined (all datasets)\n{'=' * 72}")
        report_lines.append("### Combined (all datasets)")
        report_lines.append("")
        for variant in variants:
            for n_buckets in n_buckets_list:
                ds_results = {}
                for ds_name, ds_data in data.items():
                    if not ds_data or variant not in ds_data:
                        continue
                    results = {}
                    for model in MODELS:
                        rec = ds_data[variant].get(model)
                        # combined always 'fixed' binning so x ranges are comparable
                        results[model] = None if rec is None else auc_by_bin(
                            rec["ent"], rec["y"], rec["p"], n_buckets, min_cell_n, "fixed")
                    ds_results[ds_name] = results
                suffix = f"_fixed_b{n_buckets}"
                plot_combined(variant, ds_results, combined_out, dpi, min_cell_n, suffix)
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
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
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
        plot_all(data, args.out, args.variants, args.n_buckets, args.min_cell_n,
                  args.dpi, args.binning, not args.no_combined)


if __name__ == "__main__":
    main()
