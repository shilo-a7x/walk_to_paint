"""
Permanent regression test + benchmark for LocalAttentionEncoderLayer (src/model/model.py).

Three independent checks, see MASKING.md for the full design rationale:

1. Synthetic correctness — a small hand-built batch with known padding, known
   disallowed positions, and a known window size. Verifies the actual attention
   weight matrix (computed independently, not via SDPA) is nonzero exactly where
   (in-window AND not-padding AND not-disallowed), for every real query position,
   and that the real layer's output matches that independent computation exactly
   for every real (non-padding, non-disallowed) position.
2. No-NaN on real data, real code path — no forward hooks, model.eval(), plain
   `model(input_ids, attention_mask=...)` calls, exactly how real training/eval
   invokes the model. Confirms 0% NaN on the actual production call pattern.
3. Overhead benchmark vs. full attention — wall-clock + peak memory,
   forward+backward, at production shape (embedding_dim=32, nhead=8, nlayers=5,
   seq_len=161, batch=1024, matching config.yaml / plan-performance.md). Compares
   full attention, the prior per-layer-merge design, and the current merge-once
   design side by side.

Run: .venv/bin/python scripts/test_local_attention_masking.py
"""
import sys, time, warnings
sys.path.insert(0, "/home/eng/shilo_avital/yolo_lab/walk_to_paint")
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.model import LocalAttentionEncoderLayer

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Synthetic correctness check
# ══════════════════════════════════════════════════════════════════════════════

def test_synthetic_correctness():
    print("=" * 70)
    print("1. SYNTHETIC CORRECTNESS CHECK")
    print("=" * 70)

    d_model, nhead, window = 8, 2, 2
    bsz, seq_len = 2, 12
    head_dim = d_model // nhead

    layer = LocalAttentionEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=16, dropout=0.0, batch_first=True
    ).to(DEVICE).eval()

    # Batch item 0: fully real (no padding), but positions 5,6 are "disallowed"
    #   (simulating hidden-split edges — real tokens, but attention_mask=0).
    # Batch item 1: real length 8, padded for the last 4 positions.
    attention_mask = torch.ones(bsz, seq_len, dtype=torch.long)
    attention_mask[0, 5] = 0
    attention_mask[0, 6] = 0
    attention_mask[1, 8:] = 0
    attention_mask = attention_mask.to(DEVICE)

    key_padding_mask = ~attention_mask.bool()  # (B, L) True = ignore
    pos = torch.arange(seq_len, device=DEVICE)
    window_bool = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs() > window  # (L, L)
    merged_bool = window_bool.view(1, 1, seq_len, seq_len) | key_padding_mask.view(bsz, 1, 1, seq_len)
    # (B, 1, L, L) bool, True = blocked — exactly what TransformerModel.forward builds.

    x = torch.randn(bsz, seq_len, d_model, device=DEVICE)

    # --- Independent reference: naive masked softmax attention, no SDPA ---
    mha = layer.self_attn
    qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(bsz, seq_len, nhead, head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, nhead, head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, nhead, head_dim).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)          # (B, H, L, L)
    scores = scores.masked_fill(merged_bool, float("-inf"))
    ref_weights = torch.softmax(scores, dim=-1)                     # NaN at fully-masked rows

    # Rows that are fully masked (padding/disallowed positions whose entire window
    # is also padding/disallowed) — the only place NaN can appear.
    fully_masked = merged_bool.all(dim=-1)                          # (B, 1, L)
    real_rows = ~(key_padding_mask.unsqueeze(1))                    # (B,1,L) — real query positions

    # (a) Every REAL query position must have well-defined (non-NaN) reference weights.
    real_rows_flat = real_rows.expand(-1, nhead, -1)
    assert not ref_weights[real_rows_flat].isnan().any(), "real position produced NaN in reference softmax!"
    print("  [OK] every real (non-padding/non-disallowed) query position has finite reference weights")

    # (b) Reference weights are exactly zero at every blocked (i,j) pair, for real rows.
    blocked_at_real_rows = merged_bool.expand(-1, nhead, -1, -1) & real_rows_flat.unsqueeze(-1)
    assert (ref_weights[blocked_at_real_rows] == 0.0).all(), "blocked position got nonzero weight!"
    print("  [OK] blocked positions (out-of-window OR padding OR disallowed) get exactly zero weight")

    # (c) Reference weights are (generically) nonzero at every allowed (i,j) pair, for real rows.
    allowed_at_real_rows = (~merged_bool).expand(-1, nhead, -1, -1) & real_rows_flat.unsqueeze(-1)
    assert (ref_weights[allowed_at_real_rows] > 0.0).all(), "allowed position got exactly zero weight!"
    print("  [OK] allowed positions (in-window AND not-padding AND not-disallowed) get nonzero weight")

    # (d) Confirm NaN really does occur in the naive reference at fully-masked rows —
    #     documents the mechanism this design guards against.
    if fully_masked.any():
        fm = fully_masked.expand(-1, nhead, -1)
        assert ref_weights[fm].isnan().all(), "expected NaN at fully-masked rows in naive reference"
        print(f"  [OK] naive reference does produce NaN at the {int(fully_masked.sum())} fully-masked row(s) — as expected")
    else:
        print("  [--] no fully-masked rows in this synthetic batch (nothing to confirm here)")

    # Zero-fill the fully-masked rows in the reference too, matching what the real
    # layer's masked_fill does, so the comparison below isolates "does the attention
    # sub-computation match" rather than tripping over the documented NaN at rows
    # nothing downstream reads anyway.
    ref_weights_expanded = ref_weights
    sa_raw = (ref_weights_expanded @ v).transpose(1, 2).reshape(bsz, seq_len, d_model)
    fm_embed = fully_masked.squeeze(1).unsqueeze(-1).expand(-1, -1, d_model)
    sa_raw = sa_raw.masked_fill(fm_embed, 0.0)
    sa_out = mha.out_proj(sa_raw)          # dropout1 is identity (dropout=0.0, eval mode)

    # Replicate the rest of the (norm_first=False, the default) layer computation —
    # the attention sub-block above feeds into the same residual/LayerNorm/FFN
    # pipeline nn.TransformerEncoderLayer.forward uses.
    assert not layer.norm_first
    x1 = layer.norm1(x + sa_out)
    ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x1))))
    ref_out = layer.norm2(x1 + ff_out)

    # --- Real layer output ---
    with torch.no_grad():
        real_out = layer(x, src_mask=merged_bool)

    assert not real_out.isnan().any(), "real layer produced NaN!"
    print("  [OK] real layer output has zero NaN")

    real_rows_embed = real_rows.squeeze(1).unsqueeze(-1).expand(-1, -1, d_model)  # (B, L, D)
    diff = (real_out - ref_out)[real_rows_embed].abs().max().item()
    assert diff < 1e-4, f"real layer output diverges from reference at real positions: max diff {diff}"
    print(f"  [OK] real layer output matches independent reference at every real position (max diff {diff:.2e})")

    print("SYNTHETIC CORRECTNESS: PASS\n")


# ══════════════════════════════════════════════════════════════════════════════
# 1b. eval() vs train()-mode equivalence (masking code path, dropout excluded)
# ══════════════════════════════════════════════════════════════════════════════

def test_eval_train_equivalence():
    print("=" * 70)
    print("1b. EVAL vs TRAIN-MODE EQUIVALENCE (isolates masking from dropout)")
    print("=" * 70)

    d_model, nhead, window = 16, 4, 2
    bsz, seq_len = 3, 14
    torch.manual_seed(42)
    layer = LocalAttentionEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=32, dropout=0.3, batch_first=True
    ).to(DEVICE)

    attention_mask = torch.ones(bsz, seq_len, dtype=torch.long)
    attention_mask[1, 10:] = 0
    attention_mask[2, 4] = 0
    attention_mask = attention_mask.to(DEVICE)
    key_padding_mask = ~attention_mask.bool()
    pos = torch.arange(seq_len, device=DEVICE)
    window_bool = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs() > window
    merged = window_bool.view(1, 1, seq_len, seq_len) | key_padding_mask.view(bsz, 1, 1, seq_len)
    x = torch.randn(bsz, seq_len, d_model, device=DEVICE)

    layer.eval()
    with torch.no_grad():
        out_eval = layer(x, src_mask=merged)

    # Force self.training=True but zero every dropout probability first — isolates
    # whether self.training itself (not dropout noise) changes anything else.
    layer.train()
    orig = (layer.self_attn.dropout, layer.dropout.p, layer.dropout1.p, layer.dropout2.p)
    layer.self_attn.dropout = 0.0
    layer.dropout.p = layer.dropout1.p = layer.dropout2.p = 0.0
    with torch.no_grad():
        out_train_nodrop = layer(x, src_mask=merged)
    layer.self_attn.dropout, layer.dropout.p, layer.dropout1.p, layer.dropout2.p = orig

    diff = (out_eval - out_train_nodrop).abs().max().item()
    assert diff < 1e-6, f"eval vs train (dropout excluded) diverge: {diff}"
    print(f"  [OK] eval() output bit-matches train()-with-dropout-forced-to-zero (max diff {diff:.2e})")
    print("  -> 'self.training' appears exactly once in LocalAttentionEncoderLayer (the dropout_p")
    print("     ternary, grep-verified) -- forward() has no other branch conditioned on train/eval,")
    print("     so there is no second, divergent code path to silently differ between them.")
    print("EVAL/TRAIN EQUIVALENCE: PASS\n")


# ══════════════════════════════════════════════════════════════════════════════
# 1c. Shape/config sweep — different batch sizes, head counts, dims, windows
# ══════════════════════════════════════════════════════════════════════════════

def test_shape_sweep():
    print("=" * 70)
    print("1c. SHAPE/CONFIG SWEEP (batch size, seq_len, nhead, embedding_dim, window)")
    print("=" * 70)

    configs = [
        dict(d_model=8, nhead=1, seq_len=10, bsz=1, window=2),    # degenerate: nhead=1, bsz=1
        dict(d_model=9, nhead=3, seq_len=13, bsz=5, window=1),    # odd nhead
        dict(d_model=32, nhead=8, seq_len=161, bsz=4, window=4),  # production shape, small batch
        dict(d_model=16, nhead=4, seq_len=7, bsz=2, window=0),    # window=0 (self-attend only)
    ]
    for cfg in configs:
        d_model, nhead, seq_len, bsz, window = (
            cfg["d_model"], cfg["nhead"], cfg["seq_len"], cfg["bsz"], cfg["window"]
        )
        head_dim = d_model // nhead
        layer = LocalAttentionEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model, dropout=0.0, batch_first=True
        ).to(DEVICE).eval()

        attention_mask = torch.ones(bsz, seq_len, dtype=torch.long)
        if bsz > 1:
            attention_mask[0, seq_len // 2:] = 0   # padding on item 0
        if bsz > 2:
            attention_mask[1, 1] = 0               # a lone disallowed position on item 1
        attention_mask = attention_mask.to(DEVICE)
        key_padding_mask = ~attention_mask.bool()
        pos = torch.arange(seq_len, device=DEVICE)
        window_bool = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs() > window
        merged = window_bool.view(1, 1, seq_len, seq_len)
        if key_padding_mask.any():
            merged = merged | key_padding_mask.view(bsz, 1, 1, seq_len)

        x = torch.randn(bsz, seq_len, d_model, device=DEVICE)
        with torch.no_grad():
            out = layer(x, src_mask=merged)
        assert not out.isnan().any(), f"NaN with config {cfg}"
        assert out.shape == (bsz, seq_len, d_model), f"shape mismatch with config {cfg}"
        print(f"  [OK] d_model={d_model} nhead={nhead} (head_dim={head_dim}) seq_len={seq_len} "
              f"bsz={bsz} window={window}: no NaN, correct output shape")
    print("SHAPE/CONFIG SWEEP: PASS\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. No-NaN on real data, real production code path
# ══════════════════════════════════════════════════════════════════════════════

def test_no_nan_production_path():
    print("=" * 70)
    print("2. NO-NAN ON REAL DATA (production code path — no hooks)")
    print("=" * 70)

    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score
    from src.model.lit_model import LitEdgeClassifier
    from src.data.stage_dataset import StageViewDataset, ragged_collate_fn

    cache_path = "data/epinions/dataset_cache__k_cover_k5_nw2000000_mw80_seed42.pt"
    ckpt_path = (
        "outputs/epinions/E16_NOHARD_KCOVER_K5_NW2000000_local_20260707-012846"
        "/checkpoints/epinions-E16_NOHARD_KCOVER_K5_NW2000000_local-epoch=29-val_auc_epoch=0.9553.ckpt"
    )
    import os
    if not (os.path.exists(cache_path) and os.path.exists(ckpt_path)):
        print(f"  [SKIP] cache or checkpoint not found ({cache_path} / {ckpt_path})\n")
        return

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    tokenizer = cache["tokenizer"]
    test_dataset = StageViewDataset(cache, stage="test", dynamic_train_masking=False)
    collate = ragged_collate_fn(int(tokenizer["PAD_ID"]), int(tokenizer["UNK_LABEL_ID"]))
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, collate_fn=collate, num_workers=4)

    lit = LitEdgeClassifier.load_from_checkpoint(ckpt_path, map_location="cpu")
    model = lit.model.to(DEVICE).eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, labels, attn_mask, _meta = batch
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            logits = model(input_ids, attention_mask=attn_mask)   # plain call, no hooks
            tgt = labels != -1
            all_logits.append(logits[tgt].float().cpu())
            all_labels.append(labels[tgt].cpu())

    logits = torch.cat(all_logits)
    labels_all = torch.cat(all_labels)
    n_nan = logits.isnan().any(dim=1).sum().item()
    n_total = len(labels_all)
    print(f"  Supervised positions: {n_total:,}")
    print(f"  NaN positions       : {n_nan:,} ({100*n_nan/n_total:.4f}%)")
    assert n_nan == 0, f"found {n_nan} NaN positions in production code path!"
    auc = roc_auc_score(labels_all.numpy(), torch.softmax(logits, 1)[:, 1].numpy())
    print(f"  AUC                 : {auc:.6f}")
    print("NO-NAN PRODUCTION PATH: PASS\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Overhead benchmark vs. full attention
# ══════════════════════════════════════════════════════════════════════════════

class _PerLayerMergeLocalLayer(nn.TransformerEncoderLayer):
    """Reproduces the PRIOR (2026-07-17, first-pass) fix: forward() forced eager,
    but _sa_block still re-merges window+kp on every layer call. Kept here only
    for benchmark comparison — not used anywhere else."""

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask, mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask), other_name="src_mask", target_type=src.dtype,
        )
        src_mask = F._canonical_mask(
            mask=src_mask, mask_name="src_mask", other_type=None, other_name="",
            target_type=src.dtype, check_other=False,
        )
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        mha = self.self_attn
        bsz, seq_len, embed_dim = x.shape
        nhead = mha.num_heads
        head_dim = embed_dim // nhead
        qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, nhead, head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, nhead, head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, nhead, head_dim).transpose(1, 2)
        mask = attn_mask.view(1, 1, seq_len, seq_len) if attn_mask is not None else None
        if key_padding_mask is not None and key_padding_mask.any():
            kp = key_padding_mask.view(bsz, 1, 1, seq_len)
            mask = kp if mask is None else mask + kp
        dropout_p = mha.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, embed_dim)
        out = mha.out_proj(attn_out)
        return self.dropout1(out)


def _build_encoder(layer_cls, d_model, nhead, dim_feedforward, dropout, nlayers):
    layer = layer_cls(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                       dropout=dropout, batch_first=True)
    return nn.TransformerEncoder(layer, num_layers=nlayers).to(DEVICE)


def _benchmark(encoder, x, make_masks, label, n_iters=20, warmup=5):
    """make_masks() is called fresh on every iteration (inside the timed region) —
    for the merge-once design this must include the actual window|padding OR, so the
    benchmark reflects real per-batch cost, not an amortized-away one-time build."""
    encoder.train()  # matches training-time code path (fast path always disabled here anyway)
    opt = torch.optim.SGD(encoder.parameters(), lr=1e-3)

    def step():
        opt.zero_grad()
        attn_mask, src_key_padding_mask = make_masks()
        out = encoder(x, mask=attn_mask, src_key_padding_mask=src_key_padding_mask)
        loss = out.sum()
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step()
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize(DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        step()
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize(DEVICE)
    t1 = time.perf_counter()
    ms_per_iter = (t1 - t0) / n_iters * 1000
    peak_mb = torch.cuda.max_memory_allocated(DEVICE) / 1e6 if DEVICE.startswith("cuda") else float("nan")
    print(f"  {label:32s}: {ms_per_iter:7.2f} ms/iter, peak {peak_mb:8.1f} MB")
    return ms_per_iter, peak_mb


def benchmark_overhead():
    print("=" * 70)
    print("3. OVERHEAD BENCHMARK vs. full attention (production shape)")
    print("=" * 70)
    if not DEVICE.startswith("cuda"):
        print("  [SKIP] no CUDA device available\n")
        return

    d_model, nhead, dim_feedforward, dropout, nlayers = 32, 8, 16, 0.5, 5
    bsz, seq_len, window = 1024, 161, 4

    x = torch.randn(bsz, seq_len, d_model, device=DEVICE)
    # Realistic padding: ~1.2% average per CLAUDE.md/plan-performance.md — simulate
    # variable walk lengths within the batch.
    lengths = torch.randint(low=int(seq_len * 0.85), high=seq_len + 1, size=(bsz,))
    attention_mask = torch.zeros(bsz, seq_len, dtype=torch.long)
    for i, l in enumerate(lengths):
        attention_mask[i, :l] = 1
    attention_mask = attention_mask.to(DEVICE)
    key_padding_mask = ~attention_mask.bool()

    pos = torch.arange(seq_len, device=DEVICE)
    window_bool_2d = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs() > window
    window_bool_4d = window_bool_2d.view(1, 1, seq_len, seq_len)

    print(f"  shape: d_model={d_model}, nhead={nhead}, nlayers={nlayers}, "
          f"seq_len={seq_len}, batch={bsz}, avg padding ~{(1-attention_mask.float().mean()).item()*100:.1f}%\n")

    # Full attention: stock layer, mask=None (only key_padding_mask), can use encoder fast paths.
    full_enc = _build_encoder(nn.TransformerEncoderLayer, d_model, nhead, dim_feedforward, dropout, nlayers)
    _benchmark(full_enc, x, lambda: (None, key_padding_mask), "Full attention")

    # Prior per-layer-merge design (kept only for this comparison): raw masks passed
    # in unmerged every iteration, exactly as before — the OR/add happens inside
    # _sa_block, once per layer (5x), which is what made it the slower design.
    perlayer_enc = _build_encoder(_PerLayerMergeLocalLayer, d_model, nhead, dim_feedforward, dropout, nlayers)
    _benchmark(perlayer_enc, x, lambda: (window_bool_4d, key_padding_mask),
               "Local (prior: per-layer merge)")

    # Current merge-once design: build the actual window|padding OR fresh every
    # iteration (matching TransformerModel.forward building it fresh every batch) —
    # NOT precomputed outside the loop, so this measures the real per-forward-pass cost.
    from src.model.model import LocalAttentionEncoderLayer as CurrentLocalLayer
    current_enc = _build_encoder(CurrentLocalLayer, d_model, nhead, dim_feedforward, dropout, nlayers)

    def make_merged_masks():
        merged = window_bool_4d | key_padding_mask.view(bsz, 1, 1, seq_len)
        return merged, None

    _benchmark(current_enc, x, make_merged_masks, "Local (current: merge-once)")

    print("\n  (historical reference, plan-performance.md: original unfixed local attention "
        "measured ~37% slower / ~4x memory vs. full attention)")
    print("OVERHEAD BENCHMARK: done\n")


if __name__ == "__main__":
    test_synthetic_correctness()
    test_eval_train_equivalence()
    test_shape_sweep()
    test_no_nan_production_path()
    benchmark_overhead()
    print("=" * 70)
    print("ALL CHECKS COMPLETE")
    print("=" * 70)
