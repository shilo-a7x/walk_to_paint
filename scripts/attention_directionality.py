"""Attention directionality analysis (Result 3 / checklist #22 candidate design).

Question: for a masked target edge, does attention lean on what came BEFORE it in the
walk (source-side, "backward") or what comes AFTER it (target-side, "forward"), and does
that mass sit on node tokens or edge tokens? This is a claim about the attention
mechanism's own geometry -- a separate concept from the H_out>H_in entropy-asymmetry
result (Lead 4c), which is about which architecture is hurt by which entropy term.

Sibling of scripts/attention_analysis.py (Step 3 of the original local-attention
investigation), which only measured unsigned |i-j| distance to decide the LocalAttn4
window size. This script keeps the sign of (j-i) and adds a node/edge role split. See
ATTENTION_MATH.md for the full underlying math and a side-by-side explanation of what
each script measures.

Uses the CURRENT production checkpoints (E25/E26 full attention, E27 LocalAttn4,
edge_cover sampler) -- NOT the stale E14-era checkpoints scripts/attention_analysis.py
points at via node_mi_structural_embedding.py's DATASET_CONFIGS. Run-dir/epoch pins below
are copied from scripts/paper_figures/extract_result2_walk_entropy_fresh.py, already
verified fresh there (each reproduces CLAUDE.md's exact test AUC).

Inference only -- no retraining, no checkpoint modification.

Usage
-----
  python scripts/attention_directionality.py [--datasets all] [--variants full,local]
  python scripts/attention_directionality.py --datasets bitcoin-alpha --variants full,local --max-samples 2000
"""

import os, sys, glob, pickle, argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

torch.backends.mha.set_fastpath_enabled(False)

from src.model.lit_model import LitEdgeClassifier
from src.model.model import LocalAttentionEncoderLayer
from src.data.stage_dataset import StageViewDataset, ragged_collate_fn
from src.data.prepare_data import _keyed_cache_path
from src.data.dataset_cache import load_dataset_cache

MAX_SAMPLES_DEFAULT = 20000
PLOT_RADIUS_DEFAULT = 24
NODE_COLOR = "#5b9bd5"
EDGE_COLOR = "#f2a154"

# (run_dir, epoch) per dataset, per variant -- copied from
# scripts/paper_figures/extract_result2_walk_entropy_fresh.py (FULL_RUN_INFO /
# LOCAL_RUN_INFO), dropping the posthoc run_id (not needed here, no aggregation step).
FULL_RUN_INFO = {
    "bitcoin-alpha":   ("E25_BUDGET_alpha_x5_20260717-162755", 35),
    "bitcoin-otc":     ("E25_BUDGET_otc_x5_20260717-163145", 28),
    "epinions":        ("E25_BUDGET_epinions_floor_20260717-163552", 42),
    "wiki-elec":       ("E26_WIKI_elec_p1_5x_20260719-113223", 36),
    "wiki-rfa":        ("E26_WIKI_rfa_p1_5x_20260719-113223", 24),
    "slashdot090221":  ("E25_BUDGET_slashdot_x3_20260717-190227", 39),
}
LOCAL_RUN_INFO = {
    # Updated 2026-08-04 (post-migration rebuild) -- E27 (pre-migration) replaced with
    # E32_PY314_LOCALATTN4 (post-migration, correctly-configured LocalAttn4 retrain; E31
    # turned out to be full attention, see CLAUDE.md "Current SOTA"). Epochs match the ones
    # each dataset's run_posthoc.py actually selected (checkpoints/<ds>_predictions/epoch_*),
    # not just the highest val_auc_epoch filename, for consistency with Result 1/2's numbers.
    # slashdot090221 omitted -- its E32 training is still in progress; add back once done.
    "bitcoin-alpha":   ("E32_PY314_LOCALATTN4_20260804-225948", 35),
    "bitcoin-otc":      ("E32_PY314_LOCALATTN4_20260804-231122", 35),
    "epinions":        ("E32_PY314_LOCALATTN4_20260804-225948", 30),
    "wiki-elec":       ("E32_PY314_LOCALATTN4_20260804-232140", 44),
    "wiki-rfa":        ("E32_PY314_LOCALATTN4_20260804-232239", 33),
    "slashdot090221":  ("E32_PY314_LOCALATTN4_20260804-225948", 34),
}
RUN_INFO_BY_VARIANT = {"full": FULL_RUN_INFO, "local": LOCAL_RUN_INFO}
ALL_DATASETS = list(FULL_RUN_INFO.keys())


# ── Attention-recording encoder layers ──────────────────────────────────────────

class AttentionRecordingEncoderLayer(nn.TransformerEncoderLayer):
    """Full-attention (E25/E26) checkpoints use stock nn.TransformerEncoderLayer.
    Same trick as scripts/attention_analysis.py: force need_weights=True,
    average_attn_weights=False so the [B,nhead,S,S] softmax survives, unaveraged."""

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        x, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        self.last_attn_weights = attn_weights.detach()
        return self.dropout1(x)


class LocalAttentionRecordingEncoderLayer(LocalAttentionEncoderLayer):
    """LocalAttn4 (E27) checkpoints use src/model/model.py's LocalAttentionEncoderLayer,
    which calls F.scaled_dot_product_attention directly -- that fused kernel never
    returns attention weights at all. This subclass replicates its exact math
    (QK^T/sqrt(d_h) + additive window/padding mask, then softmax) by hand instead, so
    the weights can be captured. Only _sa_block is overridden; forward() (the
    eager-path / NaN-guard override) is inherited unchanged from LocalAttentionEncoderLayer."""

    def _sa_block(self, x, attn_mask, fully_masked_rows, is_causal=False):
        mha = self.self_attn
        bsz, seq_len, embed_dim = x.shape
        nhead = mha.num_heads
        head_dim = embed_dim // nhead

        qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, nhead, head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, nhead, head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, nhead, head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            scores = scores + attn_mask  # attn_mask already canonicalized to additive float (-inf/0) by forward()
        attn_weights = torch.softmax(scores, dim=-1)
        self.last_attn_weights = attn_weights.detach()

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, embed_dim)

        if fully_masked_rows is not None:
            rows = fully_masked_rows.squeeze(1).unsqueeze(-1)
            attn_out = attn_out.masked_fill(rows, 0.0)

        out = mha.out_proj(attn_out)
        return self.dropout1(out)


# ── Model / data loading ───────────────────────────────────────────────────────

def load_model_and_dataset(ds_name, variant, stage="test"):
    run_info = RUN_INFO_BY_VARIANT[variant]
    if ds_name not in run_info:
        print(f"  ✗ no {variant} run pinned for {ds_name}")
        return None
    run_dir, epoch = run_info[ds_name]

    ckpt_dir = os.path.join(ROOT, "outputs", ds_name, run_dir, "checkpoints")
    pattern = os.path.join(ckpt_dir, f"*-epoch={epoch:02d}-*.ckpt")
    candidates = [c for c in glob.glob(pattern) if "last" not in c]
    if not candidates:
        print(f"  ✗ checkpoint not found: {pattern}")
        return None
    ckpt_path = sorted(candidates)[0]

    lit_model = LitEdgeClassifier.load_from_checkpoint(ckpt_path, map_location="cpu")
    model = lit_model.model
    cfg = lit_model.cfg  # OmegaConf, self-describing (see CLAUDE.md posthoc fix)

    is_local = getattr(model, "local_attention_window", None) is not None
    if is_local != (variant == "local"):
        print(f"  ✗ variant mismatch: requested '{variant}' but checkpoint's "
              f"local_attention_window={model.local_attention_window!r}")
        return None

    rec_cls = LocalAttentionRecordingEncoderLayer if is_local else AttentionRecordingEncoderLayer
    for layer in model.transformer.layers:
        layer.__class__ = rec_cls

    cache_path = os.path.join(ROOT, _keyed_cache_path(cfg))
    if not os.path.exists(cache_path):
        legacy = os.path.join(ROOT, str(cfg.dataset.data_dir), "dataset_cache.pt")
        if os.path.exists(legacy):
            cache_path = legacy
        else:
            print(f"  ✗ dataset cache not found: {cache_path}")
            return None

    cache_data = load_dataset_cache(cache_path, use_mmap=False)
    tokenizer = cache_data["tokenizer"]
    pad_id = int(tokenizer["PAD_ID"])
    ignore_index = int(tokenizer["UNK_LABEL_ID"])

    ds = StageViewDataset(cache_data, stage=stage)
    max_len = int(ds.lengths.max().item())
    collate = ragged_collate_fn(pad_id, ignore_index)

    return {
        "model": model,
        "dataset": ds,
        "collate": collate,
        "ignore_index": ignore_index,
        "max_len": max_len,
        "nlayers": len(model.transformer.layers),
        "nhead": model.transformer.layers[0].self_attn.num_heads,
        "window": getattr(model, "local_attention_window", None),
        "ckpt_path": ckpt_path,
        "cache_path": cache_path,
    }


# ── Core analysis ───────────────────────────────────────────────────────────────

def analyse_dataset(ds_name, variant, out_dir, stage="test", max_samples=MAX_SAMPLES_DEFAULT,
                     batch_size=64, device="cpu", plot_radius=PLOT_RADIUS_DEFAULT):
    print(f"\n{'=' * 80}\nDATASET: {ds_name}  [{variant}]\n{'=' * 80}")
    t0 = time.time()

    bundle = load_model_and_dataset(ds_name, variant, stage=stage)
    if bundle is None:
        return None

    model = bundle["model"].to(device).eval()
    ds = bundle["dataset"]
    ignore_index = bundle["ignore_index"]
    nlayers, nhead = bundle["nlayers"], bundle["nhead"]
    max_dist = bundle["max_len"] - 1
    n_bins = 2 * max_dist + 1  # signed bins, index = d + max_dist, d in [-max_dist, max_dist]
    window = bundle["window"]

    n = len(ds)
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_samples, replace=False).tolist()
        ds_run = torch.utils.data.Subset(ds, idx)
    else:
        ds_run = ds

    loader = torch.utils.data.DataLoader(
        ds_run, batch_size=batch_size, shuffle=False, collate_fn=bundle["collate"]
    )

    print(f"  N={n:,} {stage} walks  (using {len(ds_run):,}), "
          f"nlayers={nlayers}, nhead={nhead}, max_len={bundle['max_len']}, "
          f"window={window}, ckpt={os.path.basename(bundle['ckpt_path'])}")

    hist = np.zeros((nlayers, nhead, n_bins), dtype=np.float64)
    # mass[:, :, 0..4] = [forward_node, forward_edge, backward_node, backward_edge, self]
    mass = np.zeros((nlayers, nhead, 5), dtype=np.float64)
    n_targets = 0

    with torch.no_grad():
        for batch in loader:
            input_ids, labels, attention_mask, metadata = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            target_mask = labels != ignore_index
            rows, cols = target_mask.nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue

            _ = model(input_ids, attention_mask=attention_mask)

            rows_d = rows.to(device)
            cols_d = cols.to(device)
            S = input_ids.shape[1]
            am = attention_mask.float()  # [B, S]
            j_idx = torch.arange(S, device=device)
            is_edge_pos = (j_idx % 2 == 1)  # [S] bool -- odd position = edge token (Step 0, ATTENTION_MATH.md)

            dist = (j_idx.unsqueeze(0) - cols_d.unsqueeze(1))  # [M, S] signed, d = j - i
            dist_flat = dist.reshape(-1)

            edge_j = is_edge_pos.unsqueeze(0).expand_as(dist)  # [M, S]
            node_j = ~edge_j
            fwd = dist > 0
            bwd = dist < 0
            selfd = dist == 0
            fwd_node_mask = (fwd & node_j).float()
            fwd_edge_mask = (fwd & edge_j).float()
            bwd_node_mask = (bwd & node_j).float()
            bwd_edge_mask = (bwd & edge_j).float()
            self_mask = selfd.float()

            for l, layer in enumerate(model.transformer.layers):
                attn = layer.last_attn_weights  # [B, nhead, S, S]
                sel = attn[rows_d, :, cols_d, :]  # [M, nhead, S]
                valid = sel * am[rows_d].unsqueeze(1)  # [M, nhead, S]

                for h in range(nhead):
                    hist[l, h] += np.bincount(
                        (dist_flat + max_dist).cpu().numpy(),
                        weights=valid[:, h, :].reshape(-1).cpu().numpy(),
                        minlength=n_bins,
                    )

                mass[l, :, 0] += (valid * fwd_node_mask.unsqueeze(1)).sum(dim=2).sum(dim=0).cpu().numpy()
                mass[l, :, 1] += (valid * fwd_edge_mask.unsqueeze(1)).sum(dim=2).sum(dim=0).cpu().numpy()
                mass[l, :, 2] += (valid * bwd_node_mask.unsqueeze(1)).sum(dim=2).sum(dim=0).cpu().numpy()
                mass[l, :, 3] += (valid * bwd_edge_mask.unsqueeze(1)).sum(dim=2).sum(dim=0).cpu().numpy()
                mass[l, :, 4] += (valid * self_mask.unsqueeze(1)).sum(dim=2).sum(dim=0).cpu().numpy()

            n_targets += rows.numel()

    pmf = hist / max(n_targets, 1)      # [nlayers, nhead, n_bins]
    mass_frac = mass / max(n_targets, 1)  # [nlayers, nhead, 5]

    forward_total = mass_frac[:, :, 0] + mass_frac[:, :, 1]
    backward_total = mass_frac[:, :, 2] + mass_frac[:, :, 3]
    node_total = mass_frac[:, :, 0] + mass_frac[:, :, 2]
    edge_total = mass_frac[:, :, 1] + mass_frac[:, :, 3]
    self_total = mass_frac[:, :, 4]

    print(f"  n_targets={n_targets:,}  elapsed={time.time() - t0:.1f}s")
    print(f"  mean forward mass={forward_total.mean():.4f}  backward mass={backward_total.mean():.4f}  "
          f"self mass={self_total.mean():.4f}")
    print(f"  mean node mass={node_total.mean():.4f}  edge mass={edge_total.mean():.4f}")

    result = {
        "ds_name": ds_name, "variant": variant, "n_targets": n_targets,
        "nlayers": nlayers, "nhead": nhead, "max_dist": max_dist, "window": window,
        "pmf": pmf, "mass_frac": mass_frac,
        "forward_total": forward_total, "backward_total": backward_total,
        "node_total": node_total, "edge_total": edge_total, "self_total": self_total,
    }
    plot_variant(result, out_dir, plot_radius=plot_radius)
    return result


# ── Plotting ────────────────────────────────────────────────────────────────────

def plot_variant(res, out_dir, plot_radius=PLOT_RADIUS_DEFAULT, layers=None, suffix=""):
    """One subplot PER (layer, head) -- a full nlayers x nhead grid, not overlaid lines.
    The LocalAttn4 boundary is drawn at +-(window+0.5), i.e. the true wall between the
    last allowed bin and the first disallowed one, not on top of the last data point --
    overlaying it exactly on the last nonzero bin (the old behavior) made the descending
    line segment down to the next (exactly-zero) bin look like it was crossing the
    boundary, even though the underlying data is exactly zero beyond the window
    (verified numerically, see chat).

    layers: optional list of layer indices to restrict the grid to (e.g. [0] for
    layer-0-only) -- purely a plotting-time filter over the already-computed pmf array
    in `res`, does not touch or require recomputation. suffix is appended to the output
    filename so a filtered plot never overwrites the full all-layers one."""
    ds_name, variant = res["ds_name"], res["variant"]
    nlayers, nhead = res["nlayers"], res["nhead"]
    max_dist, window, pmf = res["max_dist"], res["window"], res["pmf"]
    r = min(plot_radius, max_dist)
    ds_range = list(range(-r, r + 1))
    layers_to_plot = list(layers) if layers is not None else list(range(nlayers))
    n_rows = len(layers_to_plot)

    fig_width = max(3.4 * nhead, 8.5)  # floor so the (multi-line) suptitle always fits
    fig, axes = plt.subplots(n_rows, nhead, figsize=(fig_width, 2.6 * n_rows),
                              squeeze=False, sharex=True)
    for row_idx, l in enumerate(layers_to_plot):
        for h in range(nhead):
            ax = axes[row_idx, h]
            for d in ds_range:
                role_edge = (d % 2 == 0)  # d even (incl. 0) -> same parity as target (odd) -> edge token
                ax.axvspan(d - 0.5, d + 0.5, color=EDGE_COLOR if role_edge else NODE_COLOR,
                           alpha=0.15, lw=0, zorder=0)
            y = [pmf[l, h, d + max_dist] for d in ds_range]
            ax.plot(ds_range, y, color="tab:blue", linewidth=1.4, zorder=3)
            ax.fill_between(ds_range, y, color="tab:blue", alpha=0.25, zorder=2)
            ax.axvline(0, color="k", linewidth=0.8, alpha=0.6, zorder=2)
            if window is not None:
                wall = window + 0.5
                ax.axvline(wall, color="crimson", linestyle="--", linewidth=1.1, alpha=0.85, zorder=2)
                ax.axvline(-wall, color="crimson", linestyle="--", linewidth=1.1, alpha=0.85, zorder=2)
            if row_idx == 0:
                ax.set_title(f"head {h}", fontsize=10)
            if h == 0:
                ax.set_ylabel(f"layer {l}", fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel("d = j − i", fontsize=8)
            ax.tick_params(labelsize=7)
    layer_note = f"layers {layers_to_plot}" if layers is not None else "all layers"
    title_lines = [
        f"{ds_name} [{variant}]: signed attention mass per (layer, head) -- {layer_note}  "
        f"(n_targets={res['n_targets']:,})",
        "orange bg = edge-token offsets, blue bg = node-token offsets",
    ]
    if window is not None:
        title_lines.append(f"dashed red = ±{window} LocalAttn4 window wall (mass is exactly 0 past it)")
    fig.suptitle("\n".join(title_lines), fontsize=10)
    fig.tight_layout()
    out_png = os.path.join(out_dir, f"attention_directionality_{ds_name}_{variant}{suffix}.png")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {os.path.basename(out_png)}")


def plot_summary(ds_name, results_by_variant, out_dir, layers=None, suffix=""):
    """layers: optional list of layer indices to restrict the mean to (e.g. [0] for
    layer-0-only); None means mean over all layers & heads, as before."""
    variants = [v for v in ("full", "local") if v in results_by_variant]
    if not variants:
        return
    cats = ["forward\n(node+edge)", "backward\n(node+edge)", "node\n(fwd+bwd)", "edge\n(fwd+bwd)", "self"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    width = 0.35
    x = np.arange(len(cats))
    for i, variant in enumerate(variants):
        res = results_by_variant[variant]
        sl = layers if layers is not None else slice(None)
        vals = [
            res["forward_total"][sl].mean(), res["backward_total"][sl].mean(),
            res["node_total"][sl].mean(), res["edge_total"][sl].mean(), res["self_total"][sl].mean(),
        ]
        ax.bar(x + (i - (len(variants) - 1) / 2) * width, vals, width, label=variant)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8)
    layer_note = f"layers {list(layers)}" if layers is not None else "all layers"
    ax.set_ylabel(f"avg attention mass (mean over {layer_note} & heads)")
    ax.set_title(f"{ds_name}: forward/backward and node/edge mass summary ({layer_note})")
    ax.legend()
    fig.tight_layout()
    out_png = os.path.join(out_dir, f"attention_directionality_{ds_name}_summary{suffix}.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  ✓ Saved {os.path.basename(out_png)}")


# ── Report ──────────────────────────────────────────────────────────────────────

_CAT_NAMES = ["fwd_node", "fwd_edge", "bwd_node", "bwd_edge", "self"]


def _dominant(mf):
    """Which of the 5 raw mass categories this (layer,head) concentrates on, and how much."""
    k = int(np.argmax(mf))
    return _CAT_NAMES[k], float(mf[k])


def write_report(all_results, out_dir):
    """Writes both a plain-text table (attention_directionality_report.txt) and a
    markdown report (attention_directionality_report.md) with the same per-(layer,head)
    numbers, a derived "dominant category" column for at-a-glance scanning, and a
    cross-dataset summary table up top."""
    txt_lines = []
    txt_lines.append("ATTENTION DIRECTIONALITY ANALYSIS (checklist #22 candidate design)")
    txt_lines.append("=" * 80)
    txt_lines.append("")
    txt_lines.append("Signed offset d = j - i for masked target position i (always an edge")
    txt_lines.append("token -- odd position). d>0 = forward/target-side (toward v's future in")
    txt_lines.append("the walk), d<0 = backward/source-side (toward u's history). Role by")
    txt_lines.append("parity: even nonzero d -> edge token, odd d -> node token, d=0 -> self.")
    txt_lines.append("All mass fractions are averaged over sampled masked targets in the test split.")
    txt_lines.append("'dominant' = the single largest of the 5 raw categories for that (layer,head)")
    txt_lines.append("-- the spike location you'd see in the plot.")
    txt_lines.append("")

    md_lines = []
    md_lines.append("# Attention directionality analysis (checklist #22 candidate design)")
    md_lines.append("")
    md_lines.append("Signed offset `d = j - i` for masked target position `i` (always an edge "
                     "token -- odd position). `d>0` = forward/target-side (toward `v`'s future "
                     "in the walk), `d<0` = backward/source-side (toward `u`'s history). Role by "
                     "parity: even nonzero `d` -> edge token, odd `d` -> node token, `d=0` -> self. "
                     "All mass fractions are averaged over every masked target in the test split "
                     "(no subsampling unless `--max-samples` was set). `dominant` = the single "
                     "largest of the 5 raw categories for that (layer,head) -- the spike location "
                     "you'd see in the plot.")
    md_lines.append("")
    md_lines.append("**Multi-target-per-walk note:** a single walk commonly contains more than one "
                     "test-split edge (measured directly on bitcoin-alpha: 78% of test walks have "
                     ">1 target, mean 4.8, max 58) -- every StageViewDataset walk masks ALL of its "
                     "test-split edges simultaneously, and this script's `rows,cols = "
                     "target_mask.nonzero()` already scores each one as an independent query row "
                     "with its own signed-offset frame, so multi-target walks are handled correctly "
                     "and are not an edge case, they're the majority case. The one caveat: this script "
                     "cannot currently tell whether mass landing on a nearby edge-token position is "
                     "landing on a REAL labeled edge or on ANOTHER masked-out target's `[MASK]` "
                     "placeholder -- both look identical here (same odd position, same "
                     "forward/backward role), only the token identity differs. "
                     "`scripts/measure_local_context_availability.py` (CLAUDE.md, the E30 "
                     "short-walk-truncation investigation) measures exactly this distinction for a "
                     "related question; extending that same real-vs-masked split to this script "
                     "would be a natural follow-up if it matters for the paper claim, not done here.")
    md_lines.append("")

    # Cross-dataset summary table
    md_lines.append("## Cross-dataset summary (mean over layers & heads)")
    md_lines.append("")
    md_lines.append("| dataset | variant | forward | backward | self | node | edge |")
    md_lines.append("|---|---|---|---|---|---|---|")
    for (ds_name, variant), res in all_results.items():
        md_lines.append(
            f"| {ds_name} | {variant} | {res['forward_total'].mean():.4f} | "
            f"{res['backward_total'].mean():.4f} | {res['self_total'].mean():.4f} | "
            f"{res['node_total'].mean():.4f} | {res['edge_total'].mean():.4f} |"
        )
    md_lines.append("")

    for (ds_name, variant), res in all_results.items():
        txt_lines.append("-" * 80)
        txt_lines.append(f"DATASET: {ds_name}  [{variant}]   (n_targets={res['n_targets']:,}, "
                          f"nlayers={res['nlayers']}, nhead={res['nhead']}, "
                          f"max_dist={res['max_dist']}, window={res['window']})")
        txt_lines.append("-" * 80)
        txt_lines.append(f"  {'layer':<6}{'head':<6}{'fwd_node':>10}{'fwd_edge':>10}"
                          f"{'bwd_node':>10}{'bwd_edge':>10}{'self':>8}{'eff_signed_d':>14}"
                          f"{'dominant':>12}")

        md_lines.append(f"## {ds_name} [{variant}]")
        md_lines.append("")
        md_lines.append(f"`n_targets={res['n_targets']:,}`, `nlayers={res['nlayers']}`, "
                         f"`nhead={res['nhead']}`, `max_dist={res['max_dist']}`, `window={res['window']}`")
        md_lines.append("")
        md_lines.append("| layer | head | fwd_node | fwd_edge | bwd_node | bwd_edge | self | eff_signed_d | dominant |")
        md_lines.append("|---|---|---|---|---|---|---|---|---|")

        pmf, max_dist = res["pmf"], res["max_dist"]
        d_vals = np.arange(-max_dist, max_dist + 1)
        for l in range(res["nlayers"]):
            for h in range(res["nhead"]):
                mf = res["mass_frac"][l, h]
                eff_signed = float((pmf[l, h] * d_vals).sum())
                dom_name, dom_val = _dominant(mf)
                txt_lines.append(
                    f"  {l:<6}{h:<6}{mf[0]:>10.4f}{mf[1]:>10.4f}{mf[2]:>10.4f}{mf[3]:>10.4f}"
                    f"{mf[4]:>8.4f}{eff_signed:>14.3f}{dom_name + f' ({dom_val:.2f})':>12}"
                )
                md_lines.append(
                    f"| {l} | {h} | {mf[0]:.4f} | {mf[1]:.4f} | {mf[2]:.4f} | {mf[3]:.4f} | "
                    f"{mf[4]:.4f} | {eff_signed:+.3f} | **{dom_name}** ({dom_val:.2f}) |"
                )
        txt_lines.append("")
        txt_lines.append(f"  overall forward mass:  {res['forward_total'].mean():.4f}")
        txt_lines.append(f"  overall backward mass: {res['backward_total'].mean():.4f}")
        txt_lines.append(f"  overall node mass:     {res['node_total'].mean():.4f}")
        txt_lines.append(f"  overall edge mass:     {res['edge_total'].mean():.4f}")
        txt_lines.append(f"  overall self mass:     {res['self_total'].mean():.4f}")
        txt_lines.append("")

        md_lines.append("")
        md_lines.append(
            f"Overall: forward **{res['forward_total'].mean():.4f}**, "
            f"backward **{res['backward_total'].mean():.4f}**, "
            f"node **{res['node_total'].mean():.4f}**, "
            f"edge **{res['edge_total'].mean():.4f}**, "
            f"self **{res['self_total'].mean():.4f}**"
        )
        md_lines.append("")

    report_path = os.path.join(out_dir, "attention_directionality_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(txt_lines))
    print(f"\n✓ Report written to {report_path}")

    md_path = os.path.join(out_dir, "attention_directionality_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"✓ Report written to {md_path}")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--variants", default="full,local")
    parser.add_argument("--out", default="outputs/attention_directionality")
    parser.add_argument("--stage", default="test", choices=["val", "test"])
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_DEFAULT,
                        help="cap on sampled test walks; <=0 means unlimited (use every test walk)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--plot-radius", type=int, default=PLOT_RADIUS_DEFAULT)
    parser.add_argument("--replot-only", action="store_true",
                        help="skip inference; reload cached *_result.pkl files and just "
                             "regenerate plots/report (e.g. after a plotting-code change)")
    parser.add_argument("--layers", default=None,
                        help="comma-separated layer indices to restrict PLOTS to (e.g. '0' for "
                             "layer-0-only). Purely a plotting-time filter over the cached pmf/mass "
                             "arrays -- never touches the *_result.pkl files, which always keep every "
                             "layer. Output filenames get a _layerN suffix so filtered plots never "
                             "overwrite the all-layers ones. The .txt/.md report is unaffected (always "
                             "full, all layers).")
    args = parser.parse_args()
    layers_filter = None
    layer_suffix = ""
    if args.layers is not None:
        layers_filter = [int(x) for x in args.layers.split(",") if x.strip() != ""]
        layer_suffix = "_layer" + "-".join(str(x) for x in layers_filter)

    datasets = ALL_DATASETS if args.datasets == ["all"] else args.datasets
    max_samples = None if args.max_samples <= 0 else args.max_samples
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in RUN_INFO_BY_VARIANT:
            parser.error(f"unknown variant '{v}', must be 'full' and/or 'local'")

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    if device != "cpu" and device.isdigit():
        device = f"cuda:{device}"

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    per_dataset = {}
    for ds_name in datasets:
        if ds_name not in ALL_DATASETS:
            print(f"Unknown dataset: {ds_name}")
            continue
        per_dataset[ds_name] = {}
        for variant in variants:
            if args.replot_only:
                pkl_path = os.path.join(out_dir, f"attention_directionality_{ds_name}_{variant}_result.pkl")
                if not os.path.exists(pkl_path):
                    print(f"  ✗ --replot-only but no cached result at {pkl_path}, skipping")
                    continue
                with open(pkl_path, "rb") as f:
                    result = pickle.load(f)
                plot_variant(result, out_dir, plot_radius=args.plot_radius,
                             layers=layers_filter, suffix=layer_suffix)
            else:
                result = analyse_dataset(ds_name, variant, out_dir, stage=args.stage,
                                          max_samples=max_samples, batch_size=args.batch_size,
                                          device=device, plot_radius=args.plot_radius)
                if result is not None:
                    with open(os.path.join(out_dir, f"attention_directionality_{ds_name}_{variant}_result.pkl"), "wb") as f:
                        pickle.dump(result, f)
                    if layers_filter is not None:
                        # analyse_dataset already plotted the all-layers version; also emit the
                        # layer-filtered one from the same in-memory result, no recompute needed.
                        plot_variant(result, out_dir, plot_radius=args.plot_radius,
                                     layers=layers_filter, suffix=layer_suffix)
            if result is not None:
                all_results[(ds_name, variant)] = result
                per_dataset[ds_name][variant] = result
        if len(per_dataset[ds_name]) > 1:
            plot_summary(ds_name, per_dataset[ds_name], out_dir)
            if layers_filter is not None:
                plot_summary(ds_name, per_dataset[ds_name], out_dir,
                             layers=layers_filter, suffix=layer_suffix)

    # Merge in any previously-computed results for combos not in this run.
    for ds_name in ALL_DATASETS:
        for variant in RUN_INFO_BY_VARIANT:
            if (ds_name, variant) in all_results:
                continue
            pkl_path = os.path.join(out_dir, f"attention_directionality_{ds_name}_{variant}_result.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    all_results[(ds_name, variant)] = pickle.load(f)

    if all_results:
        ordered = {(d, v): all_results[(d, v)] for d in ALL_DATASETS for v in ("full", "local")
                   if (d, v) in all_results}
        write_report(ordered, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
