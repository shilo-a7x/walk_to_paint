"""
Attention effective-distance analysis (Step 3 of the Information/Understanding
Track plan). Inference only -- no retraining.

Question: does any attention head, in any layer, actually attend beyond a
2-hop window around the masked target position? This directly informs
Step 4's local-attention-window choice.

Token layout (alternating node/edge): pos 0=N_u0, 1=E_s1, 2=N_u1, 3=E_s2, ...
One graph-hop = +2 token positions, so 2 graph-hops = |i-j| <= 4 token
positions around a masked position i.

For each masked target position i (labels != ignore_index) and each
(layer, head), we record:
  - effective_distance = sum_j attn[i,j] * |i-j|  (over valid, non-padded j)
  - the full attention-mass distribution over |i-j|
averaged over all sampled masked positions in the test split.

Usage
-----
  python scripts/attention_analysis.py [--datasets all] [--out outputs/attention_analysis]
  python scripts/attention_analysis.py --datasets bitcoin-alpha --max-samples 2000
"""

import os, sys, glob, pickle, argparse, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The TransformerEncoderLayer fast path (torch._transformer_encoder_layer_fwd)
# bypasses _sa_block entirely in eval mode, so attention weights never get
# recorded unless it's disabled.
torch.backends.mha.set_fastpath_enabled(False)

from src.model.lit_model import LitEdgeClassifier
from src.data.stage_dataset import StageViewDataset, ragged_collate_fn
from node_mi_structural_embedding import DATASET_CONFIGS, load_dataset_cfg  # noqa: E402

MAX_SAMPLES_DEFAULT = 20000
HOP_TOKEN_WIDTH = 2  # 1 graph-hop = 2 token positions
ONE_HOP_RADIUS = 2   # |i-j| <= 2  <=>  within 1 graph-hop
TWO_HOP_RADIUS = 4   # |i-j| <= 4  <=>  within 2 graph-hops


# ── Attention-recording encoder layer ──────────────────────────────────────────

class AttentionRecordingEncoderLayer(nn.TransformerEncoderLayer):
    """Same as nn.TransformerEncoderLayer, but stashes the per-head attention
    weights from the self-attention block on self.last_attn_weights, shape
    [batch, nhead, seq, seq]."""

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


# ── Model / data loading ───────────────────────────────────────────────────────

def load_model_and_dataset(ds_name, cfg, stage="test"):
    dscfg = load_dataset_cfg(cfg["ds_name"])
    cache_path = os.path.join(ROOT, dscfg.dataset.data_dir, "dataset_cache.pt")
    if not os.path.exists(cache_path):
        print(f"  ✗ dataset_cache.pt not found at {cache_path}")
        return None

    cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
    tokenizer = cache_data["tokenizer"]
    pad_id = int(tokenizer["PAD_ID"])
    ignore_index = int(tokenizer["UNK_LABEL_ID"])

    exp_dir = os.path.join(ROOT, cfg["exp_dir"])
    epoch = cfg["best_epoch"]
    pattern = os.path.join(exp_dir, "runs", cfg["ds_name"], "E14_HARDNODE_L10",
                            "checkpoints", f"*-epoch={epoch:02d}-*.ckpt")
    candidates = [c for c in glob.glob(pattern) if "last" not in c]
    if not candidates:
        print(f"  ✗ checkpoint not found: {pattern}")
        return None

    lit_model = LitEdgeClassifier.load_from_checkpoint(candidates[0], map_location="cpu")
    model = lit_model.model
    for layer in model.transformer.layers:
        layer.__class__ = AttentionRecordingEncoderLayer

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
    }


# ── Core analysis ───────────────────────────────────────────────────────────────

def analyse_dataset(ds_name, cfg, out_dir, stage="test", max_samples=MAX_SAMPLES_DEFAULT,
                     batch_size=64, device="cpu", return_per_example=False, node_id_of=None):
    """return_per_example: if True, also returns result["per_example"], a list of
    per-masked-target dicts (eff_dist, frac_beyond_1hop, both averaged over layers/heads
    -- already computed internally below, just not retained by default -- plus u/v node
    ids decoded from the flanking node tokens if node_id_of is given). Used by Lead 3
    Step 3 (scripts/lead3_attention_ambiguity.py) to bin per-example attention behavior
    by a per-edge ambiguity score; dataset-level aggregates (returned either way) are
    unaffected by this flag."""
    print(f"\n{'=' * 80}\nDATASET: {ds_name}\n{'=' * 80}")
    t0 = time.time()

    bundle = load_model_and_dataset(ds_name, cfg, stage=stage)
    if bundle is None:
        return None

    model = bundle["model"].to(device).eval()
    ds = bundle["dataset"]
    ignore_index = bundle["ignore_index"]
    nlayers, nhead = bundle["nlayers"], bundle["nhead"]
    max_dist = bundle["max_len"] - 1

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
          f"nlayers={nlayers}, nhead={nhead}, max_len={bundle['max_len']}")

    sum_eff = np.zeros((nlayers, nhead), dtype=np.float64)
    hist = np.zeros((nlayers, nhead, max_dist + 1), dtype=np.float64)
    n_targets = 0
    per_example = [] if return_per_example else None

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
            dist = (j_idx.unsqueeze(0) - cols_d.unsqueeze(1)).abs()  # [M, S]
            dist_flat = dist.reshape(-1)

            if return_per_example:
                M = rows.numel()
                eff_sum_per_example = torch.zeros(M, device=device)
                frac_beyond_1hop_sum_per_example = torch.zeros(M, device=device)
                beyond_1hop_mask = (dist > ONE_HOP_RADIUS).float()  # [M, S]

            for l, layer in enumerate(model.transformer.layers):
                attn = layer.last_attn_weights  # [B, nhead, S, S]
                sel = attn[rows_d, :, cols_d, :]  # [M, nhead, S]
                valid = sel * am[rows_d].unsqueeze(1)  # [M, nhead, S]
                eff = (valid * dist.unsqueeze(1).float()).sum(dim=2)  # [M, nhead]
                sum_eff[l] += eff.sum(dim=0).cpu().numpy()
                for h in range(nhead):
                    hist[l, h] += np.bincount(
                        dist_flat.cpu().numpy(),
                        weights=valid[:, h, :].reshape(-1).cpu().numpy(),
                        minlength=max_dist + 1,
                    )
                if return_per_example:
                    eff_sum_per_example += eff.mean(dim=1)  # avg over heads, this layer
                    frac_beyond = (valid * beyond_1hop_mask.unsqueeze(1)).sum(dim=2)  # [M, nhead]
                    frac_beyond_1hop_sum_per_example += frac_beyond.mean(dim=1)

            if return_per_example:
                eff_mean = (eff_sum_per_example / nlayers).cpu().numpy()
                frac_beyond_1hop_mean = (frac_beyond_1hop_sum_per_example / nlayers).cpu().numpy()
                ids_np = input_ids.cpu().numpy()
                am_np = attention_mask.cpu().numpy()
                rows_np, cols_np = rows.numpy(), cols.numpy()
                for m in range(rows.numel()):
                    row, i = int(rows_np[m]), int(cols_np[m])
                    rec = {"eff_dist": float(eff_mean[m]),
                           "frac_beyond_1hop": float(frac_beyond_1hop_mean[m]),
                           "u": None, "v": None}
                    if node_id_of is not None:
                        u_pos, v_pos = i - 1, i + 1
                        if 0 <= u_pos < S and 0 <= v_pos < S \
                                and am_np[row, u_pos] and am_np[row, v_pos]:
                            utok, vtok = int(ids_np[row, u_pos]), int(ids_np[row, v_pos])
                            rec["u"] = node_id_of.get(utok)
                            rec["v"] = node_id_of.get(vtok)
                    per_example.append(rec)

            n_targets += rows.numel()

    eff_dist = sum_eff / max(n_targets, 1)  # [nlayers, nhead]
    pmf = hist / max(n_targets, 1)          # [nlayers, nhead, max_dist+1]
    frac_within_2hop = pmf[:, :, :TWO_HOP_RADIUS + 1].sum(axis=-1)
    frac_beyond_2hop = 1.0 - frac_within_2hop

    print(f"  n_targets={n_targets:,}  elapsed={time.time() - t0:.1f}s")
    print(f"  mean effective_distance (over all layers/heads) = {eff_dist.mean():.3f}")
    print(f"  mean attention mass beyond 2 hops (|i-j|>{TWO_HOP_RADIUS}) = "
          f"{frac_beyond_2hop.mean():.4f}")

    # ── Plot: per-layer attention-mass-vs-distance, one line per head ──────────
    plot_dist = min(max_dist, 32)
    fig, axes = plt.subplots(1, nlayers, figsize=(4.5 * nlayers, 4), squeeze=False)
    for l in range(nlayers):
        ax = axes[0, l]
        for h in range(nhead):
            ax.plot(range(plot_dist + 1), pmf[l, h, :plot_dist + 1], label=f"head {h}")
        ax.axvline(TWO_HOP_RADIUS, color="k", linestyle="--", alpha=0.6, label="2-hop boundary")
        ax.set_title(f"layer {l}")
        ax.set_xlabel("|i - j| (token positions)")
        ax.set_ylabel("avg attention mass")
        if l == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"{ds_name}: attention mass vs. distance (n_targets={n_targets:,})")
    fig.tight_layout()
    out_png = os.path.join(out_dir, f"effective_distance_{ds_name}.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  ✓ Saved {os.path.basename(out_png)}")

    result = {
        "n_targets": n_targets,
        "nlayers": nlayers,
        "nhead": nhead,
        "max_dist": max_dist,
        "eff_dist": eff_dist,
        "frac_within_2hop": frac_within_2hop,
        "frac_beyond_2hop": frac_beyond_2hop,
        "pmf": pmf,
    }
    if return_per_example:
        result["per_example"] = per_example
    return result


# ── Report ──────────────────────────────────────────────────────────────────────

def write_report(all_results, out_dir):
    lines = []
    lines.append("ATTENTION EFFECTIVE-DISTANCE ANALYSIS (Step 3)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("For each masked target position i, effective_distance =")
    lines.append("sum_j attn[i,j] * |i-j|, averaged over all sampled masked")
    lines.append("positions in the test split (over valid, non-padded j).")
    lines.append("")
    lines.append("Token layout alternates node/edge: 1 graph-hop = 2 token")
    lines.append(f"positions, so |i-j| <= {TWO_HOP_RADIUS} <=> within 2 graph-hops.")
    lines.append("frac_beyond_2hop = fraction of attention mass with |i-j| > "
                  f"{TWO_HOP_RADIUS}.")
    lines.append("")

    for ds_name, res in all_results.items():
        lines.append("-" * 80)
        lines.append(f"DATASET: {ds_name}   (n_targets={res['n_targets']:,}, "
                      f"nlayers={res['nlayers']}, nhead={res['nhead']}, "
                      f"max_dist={res['max_dist']})")
        lines.append("-" * 80)
        lines.append(f"  {'layer':<6}{'head':<6}{'eff_dist':>10}{'frac_within_2hop':>18}"
                      f"{'frac_beyond_2hop':>18}")
        for l in range(res["nlayers"]):
            for h in range(res["nhead"]):
                lines.append(
                    f"  {l:<6}{h:<6}{res['eff_dist'][l, h]:>10.3f}"
                    f"{res['frac_within_2hop'][l, h]:>18.4f}"
                    f"{res['frac_beyond_2hop'][l, h]:>18.4f}"
                )
        lines.append("")
        lines.append(f"  overall mean effective_distance: {res['eff_dist'].mean():.3f}")
        lines.append(f"  overall mean frac_beyond_2hop:   {res['frac_beyond_2hop'].mean():.4f}")
        lines.append(f"  max frac_beyond_2hop (any head): {res['frac_beyond_2hop'].max():.4f}")
        lines.append("")

    report_path = os.path.join(out_dir, "attention_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n✓ Report written to {report_path}")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--out", default="outputs/attention_analysis")
    parser.add_argument("--stage", default="test", choices=["val", "test"])
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.datasets == ["all"]:
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = args.datasets

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    if device != "cpu" and device.isdigit():
        device = f"cuda:{device}"

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    for ds_name in datasets:
        if ds_name not in DATASET_CONFIGS:
            print(f"Unknown dataset: {ds_name}")
            continue
        result = analyse_dataset(ds_name, DATASET_CONFIGS[ds_name], out_dir,
                                  stage=args.stage, max_samples=args.max_samples,
                                  batch_size=args.batch_size, device=device)
        if result is not None:
            all_results[ds_name] = result
            with open(os.path.join(out_dir, f"attention_{ds_name}_result.pkl"), "wb") as f:
                pickle.dump(result, f)

    # Merge in any previously-computed results for datasets not in this run.
    for ds_name in DATASET_CONFIGS:
        if ds_name in all_results:
            continue
        pkl_path = os.path.join(out_dir, f"attention_{ds_name}_result.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                all_results[ds_name] = pickle.load(f)

    if all_results:
        ordered = {k: all_results[k] for k in DATASET_CONFIGS if k in all_results}
        write_report(ordered, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
