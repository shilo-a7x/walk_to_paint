"""Diagnostic (read-only): does a masked target edge's attention allocate more/less
mass to node-token vs edge-token context positions on epinions vs wiki-elec/wiki-rfa?

Context: EID's winning architecture on epinions (its only dataset with a real win over
production) uses node_embed_dim=0 (no dedicated node-embedding channel), vs wiki-elec/
wiki-rfa's node_embed_dim=64 and much larger edge_embed_rank -- a plausible
capacity-mismatch/overfitting story. This script asks, on REAL trained checkpoints and
REAL val/test data (not just the hyperparameter table): when the model is predicting a
masked target edge's sign, how much self-attention mass does that target position put
on node-token positions in its local context vs. edge-token positions (including its
own position, which always carries visible identity in EID, just never a visible sign)?

Mechanism/positional note (see MECHANISM.md sec 4): the identity channel and the sign
channel are NOT separate sequence positions -- both are summed (or concatenated) into
the SAME key vector at an edge position (`_content_embed` + `sign_embedding`, see
eid_src/model/model.py::EdgeIdentityTransformerModel.forward). So attention weights can
only be broken down by TOKEN POSITION (node position vs edge position), never further
into "attention to the identity channel" vs "attention to the sign channel" within a
single edge position -- that distinction lives inside the key vector's subspace, not in
which position gets attended to. This script is explicit about that limitation rather
than fabricating a finer split.

Extraction method: EID's local-attention layer (LocalAttentionEncoderLayer, subclassed
unmodified from production's src/model/model.py) already bypasses nn.MultiheadAttention
and manually computes q/k/v via F.linear + a manual reshape before calling
F.scaled_dot_product_attention (see that file's docstring -- done for mask-broadcast
efficiency, not for weight access, but it means real per-head attention weights are one
straightforward substitution away: replace the SDPA call with the mathematically
identical manual softmax(q @ k^T / sqrt(d) + mask) @ v, and stash the softmax output).
Monkey-patches `LocalAttentionEncoderLayer._sa_block` for the duration of this script's
forward passes only; restores the original method before exiting. No production or EID
source file is modified, no config/checkpoint file is touched, nothing under outputs/ is
written.

Usage:
  .venv/bin/python experiments/edge_identity_tokens/inspect_attention_node_vs_edge.py \
      --device 0 --n-instances 400
"""
import argparse
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from experiments.edge_identity_tokens.eid_src.model.lit_model import EIDLitEdgeClassifier
from experiments.edge_identity_tokens.eid_src.data.prepare_eid_data import prepare_eid_data
from experiments.edge_identity_tokens.run_eid import EID_CACHE_PATH, ensure_eid_cache
from src.utils.config import get_seed
from src.model.model import LocalAttentionEncoderLayer

# Real, on-disk, checkpointed noablation runs -- picked for having a saved .ckpt with a
# real val_auc-selected checkpoint (not just last.ckpt), same seed where available.
CHECKPOINTS = {
    "epinions": "outputs/epinions/EID_MULTISEED_s43_noablation_20260916-025121/checkpoints/"
                 "epinions-EID_MULTISEED_s43_noablation-epoch=30-val_auc_epoch=0.9561.ckpt",
    "wiki-elec": "outputs/wiki-elec/EID_MULTISEED_s43_noablation_20260916-014814/checkpoints/"
                 "wiki-elec-EID_MULTISEED_s43_noablation-epoch=24-val_auc_epoch=0.8640.ckpt",
    "wiki-rfa": "outputs/wiki-rfa/EID_MULTISEED_s42_noablation_20260916-011059/checkpoints/"
                "wiki-rfa-EID_MULTISEED_s42_noablation-epoch=19-val_auc_epoch=0.8791.ckpt",
}

_CAPTURED = {}


def _sa_block_capture(self, x, attn_mask, fully_masked_rows, is_causal=False):
    """Drop-in replacement for LocalAttentionEncoderLayer._sa_block that computes the
    identical operation (manual q/k/v -> scaled dot-product -> softmax -> weighted sum
    of v) but via an explicit softmax instead of F.scaled_dot_product_attention, so the
    per-head attention weight matrix can be captured. attn_mask here is already the
    canonicalized additive float mask (0 / -inf) built in TransformerModel.forward."""
    mha = self.self_attn
    bsz, seq_len, embed_dim = x.shape
    nhead = mha.num_heads
    head_dim = embed_dim // nhead

    qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(bsz, seq_len, nhead, head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, nhead, head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, nhead, head_dim).transpose(1, 2)

    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        scores = scores + attn_mask  # additive mask, 0 or -inf, broadcasts (1,1,L,L)/(B,1,L,L)
    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)  # fully-masked rows: softmax(-inf,...) = nan
    attn_out = torch.matmul(weights, v)
    attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, embed_dim)

    if fully_masked_rows is not None:
        rows = fully_masked_rows.squeeze(1).unsqueeze(-1)
        attn_out = attn_out.masked_fill(rows, 0.0)

    out = mha.out_proj(attn_out)
    _CAPTURED.setdefault("layers", []).append(weights.detach().to("cpu"))
    return self.dropout1(out)


def load_eid_model_and_test_loader(ckpt_path, device):
    model = EIDLitEdgeClassifier.load_from_checkpoint(ckpt_path, map_location="cpu")
    cfg = model.cfg
    eid_cache_path = EID_CACHE_PATH.format(
        dataset=cfg.dataset.name, num_walks=int(cfg.dataset.num_walks), seed=get_seed(cfg)
    )
    ensure_eid_cache(cfg, eid_cache_path)
    data_module = prepare_eid_data(cfg, eid_cache_path)
    model = model.to(device)
    model.eval()
    return model, data_module["test"], cfg


def collect_instances(model, test_loader, device, n_target, seed=0):
    """Run real batches through the (patched) model, collect per-target-instance
    attention-mass-by-column-role for as many masked target positions as n_target,
    using a fixed torch seed so batch order (shuffle=False for test loaders normally,
    but be defensive) is reproducible across datasets."""
    torch.manual_seed(seed)
    ignore_idx = model.cfg.model.ignore_index
    per_layer_masses = None  # filled on first batch once nlayers is known
    n_collected = 0
    n_self_only_targets = 0

    with torch.no_grad():
        for batch in test_loader:
            if n_collected >= n_target:
                break
            input_ids, labels, attention_mask, metadata = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            sign_ids = metadata["sign_ids"].to(device)
            positions = metadata["positions"].to(device)

            target_mask = labels != ignore_idx
            if not target_mask.any():
                continue

            node_mask = (positions >= 0) & (positions % 2 == 0)
            edge_mask = (positions >= 0) & (positions % 2 == 1)
            valid_mask = attention_mask.bool()

            _CAPTURED.clear()
            _ = model.model(input_ids, sign_ids, attention_mask=attention_mask, node_mask=node_mask)
            layer_weights = _CAPTURED["layers"]  # list[nlayers] of (B, H, L, L)
            nlayers = len(layer_weights)
            if per_layer_masses is None:
                per_layer_masses = [defaultdict(list) for _ in range(nlayers)]

            b_idx, t_pos = target_mask.nonzero(as_tuple=True)
            for b, tpos in zip(b_idx.tolist(), t_pos.tolist()):
                if n_collected >= n_target:
                    break
                # layer_weights are stashed on CPU (see _sa_block_capture) to save GPU
                # memory across many batches, so the column-role masks must be on CPU too.
                col_valid = valid_mask[b].cpu()  # (L,)
                col_node = node_mask[b].cpu() & col_valid
                col_edge = edge_mask[b].cpu() & col_valid
                col_self = torch.zeros_like(col_valid)
                col_self[tpos] = True
                col_edge_other = col_edge & (~col_self)

                any_row_had_weight = False
                for li in range(nlayers):
                    w = layer_weights[li][b].mean(dim=0)[tpos]  # head-averaged, (L,)
                    total = float(w.sum().item())
                    if total <= 1e-8:
                        continue
                    any_row_had_weight = True
                    mass_node = float(w[col_node].sum().item())
                    mass_edge_other = float(w[col_edge_other].sum().item())
                    mass_self = float(w[col_self].sum().item())
                    mass_invalid = max(0.0, total - mass_node - mass_edge_other - mass_self)
                    per_layer_masses[li]["node"].append(mass_node / total)
                    per_layer_masses[li]["edge_other"].append(mass_edge_other / total)
                    per_layer_masses[li]["self"].append(mass_self / total)
                    per_layer_masses[li]["invalid_residual"].append(mass_invalid / total)
                if any_row_had_weight:
                    n_collected += 1
                    if col_edge_other.sum().item() == 0:
                        n_self_only_targets += 1

    return per_layer_masses, n_collected, n_self_only_targets


def summarize(per_layer_masses):
    nlayers = len(per_layer_masses)
    per_layer_summary = []
    for li in range(nlayers):
        d = per_layer_masses[li]
        per_layer_summary.append({
            "node_mean": float(np.mean(d["node"])),
            "edge_other_mean": float(np.mean(d["edge_other"])),
            "self_mean": float(np.mean(d["self"])),
            "n": len(d["node"]),
        })
    # overall: pool across layers (each instance contributes nlayers rows)
    all_node = np.concatenate([np.array(per_layer_masses[li]["node"]) for li in range(nlayers)])
    all_edge_other = np.concatenate([np.array(per_layer_masses[li]["edge_other"]) for li in range(nlayers)])
    all_self = np.concatenate([np.array(per_layer_masses[li]["self"]) for li in range(nlayers)])
    overall = {
        "node_mean": float(all_node.mean()), "node_std": float(all_node.std()),
        "edge_other_mean": float(all_edge_other.mean()), "edge_other_std": float(all_edge_other.std()),
        "self_mean": float(all_self.mean()), "self_std": float(all_self.std()),
    }
    last_layer = per_layer_summary[-1]
    return per_layer_summary, overall, last_layer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--n-instances", type=int, default=400)
    p.add_argument("--datasets", type=str, default="epinions,wiki-elec,wiki-rfa")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.device))
        device = torch.device("cuda:0")

    orig_sa_block = LocalAttentionEncoderLayer._sa_block
    LocalAttentionEncoderLayer._sa_block = _sa_block_capture

    results = {}
    try:
        for ds in args.datasets.split(","):
            ckpt = CHECKPOINTS[ds]
            print(f"\n=== {ds} ===")
            print(f"checkpoint: {ckpt}")
            model, test_loader, cfg = load_eid_model_and_test_loader(ckpt, device)
            print(
                f"edge_embed_rank={getattr(cfg.model, 'edge_embed_rank', None)} "
                f"node_embed_dim={getattr(cfg.model, 'node_embed_dim', None)} "
                f"edge_sign_combine={getattr(cfg.model, 'edge_sign_combine', None)} "
                f"local_attention_window={getattr(cfg.model, 'local_attention_window', None)} "
                f"eid_reveal_holdout_identity={getattr(cfg.model, 'eid_reveal_holdout_identity', None)}"
            )
            per_layer_masses, n_collected, n_self_only = collect_instances(
                model, test_loader, device, args.n_instances, seed=42
            )
            per_layer_summary, overall, last_layer = summarize(per_layer_masses)
            results[ds] = {
                "n_instances": n_collected,
                "n_self_only_targets": n_self_only,
                "per_layer": per_layer_summary,
                "overall": overall,
                "last_layer": last_layer,
            }
            print(f"n_instances collected: {n_collected} (self-only-context: {n_self_only})")
            print("per-layer (node / other-edge / self):")
            for li, s in enumerate(per_layer_summary):
                print(f"  layer {li}: node={s['node_mean']:.4f} edge_other={s['edge_other_mean']:.4f} "
                      f"self={s['self_mean']:.4f}")
            print(f"overall pooled: node={overall['node_mean']:.4f}+-{overall['node_std']:.4f} "
                  f"edge_other={overall['edge_other_mean']:.4f}+-{overall['edge_other_std']:.4f} "
                  f"self={overall['self_mean']:.4f}+-{overall['self_std']:.4f}")
            print(f"last layer only: node={last_layer['node_mean']:.4f} "
                  f"edge_other={last_layer['edge_other_mean']:.4f} self={last_layer['self_mean']:.4f}")

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    finally:
        LocalAttentionEncoderLayer._sa_block = orig_sa_block

    print("\n\n=== SUMMARY TABLE (overall, pooled across all layers) ===")
    print(f"{'dataset':<12} {'n':>5} {'node':>8} {'edge_other':>11} {'self':>8}")
    for ds, r in results.items():
        o = r["overall"]
        print(f"{ds:<12} {r['n_instances']:>5} {o['node_mean']:>8.4f} {o['edge_other_mean']:>11.4f} "
              f"{o['self_mean']:>8.4f}")

    print("\n=== SUMMARY TABLE (last layer only) ===")
    print(f"{'dataset':<12} {'node':>8} {'edge_other':>11} {'self':>8}")
    for ds, r in results.items():
        l = r["last_layer"]
        print(f"{ds:<12} {l['node_mean']:>8.4f} {l['edge_other_mean']:>11.4f} {l['self_mean']:>8.4f}")


if __name__ == "__main__":
    main()
