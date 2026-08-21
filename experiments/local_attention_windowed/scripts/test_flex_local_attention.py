"""
Prototype correctness check + benchmark for FlexAttentionLocalLayer
(experiments/local_attention_windowed/localattn/flex_local_attention.py) against the
current production dense-masked LocalAttentionEncoderLayer
(experiments/local_attention_windowed/localattn/model.py -- a copy of the real
src/model/model.py, per this experiment's isolation requirement).

This is a standalone prototype -- it does not touch or import anything from the real
repo's src/. Everything needed is copied into experiments/local_attention_windowed/.
The package is named `localattn` (not `src`) specifically to avoid colliding with the
real repo's own `src` package on sys.path when run from the repo root.

Run: .venv/bin/python experiments/local_attention_windowed/scripts/test_flex_local_attention.py
(run from the repo root, so the sys.path insert below finds this experiment's own package)
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

from localattn.model import LocalAttentionEncoderLayer
from localattn.flex_local_attention import FlexAttentionLocalLayer, build_flex_local_mask

DEVICE = os.environ.get("EXP_DEVICE") or ("cuda:3" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def _copy_weights(dense_layer, flex_layer):
    flex_layer.load_state_dict(dense_layer.state_dict())


# ══════════════════════════════════════════════════════════════════════════════
# 1. Synthetic correctness: FlexAttentionLocalLayer vs. LocalAttentionEncoderLayer
# ══════════════════════════════════════════════════════════════════════════════

def test_correctness_vs_dense():
    print("=" * 70)
    print("1. FLEX-ATTENTION LOCAL vs. DENSE-MASKED LOCAL -- CORRECTNESS")
    print("=" * 70)
    if not DEVICE.startswith("cuda"):
        print("  [SKIP] flex_attention requires CUDA in this torch build\n")
        return

    d_model, nhead, window = 32, 8, 4
    bsz, seq_len = 6, 47  # deliberately not a multiple of any block size
    block_size = 16

    dense = LocalAttentionEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=0.0, batch_first=True
    ).to(DEVICE).eval()
    flex = FlexAttentionLocalLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=0.0, batch_first=True
    ).to(DEVICE).eval()
    _copy_weights(dense, flex)

    # Same padding/disallowed pattern shape as test_local_attention_masking.py's
    # synthetic check: some fully-real rows, some with mid-sequence disallowed
    # positions, some with trailing padding, plus one row engineered to be fully
    # masked (tests the fully_masked_rows zero-fill path in both layers).
    attention_mask = torch.ones(bsz, seq_len, dtype=torch.long)
    attention_mask[0, 20] = 0
    attention_mask[0, 21] = 0
    attention_mask[1, 30:] = 0
    attention_mask[2, 5:9] = 0          # a wider disallowed block
    attention_mask[3, max(0, 10 - window):10 + window + 1] = 0  # engineer a fully-masked row at q=10
    attention_mask = attention_mask.to(DEVICE)

    x = torch.randn(bsz, seq_len, d_model, device=DEVICE)

    # --- dense reference path (same construction test_local_attention_masking.py uses) ---
    key_padding_mask = ~attention_mask.bool()
    pos = torch.arange(seq_len, device=DEVICE)
    window_bool = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs() > window
    merged_bool = window_bool.view(1, 1, seq_len, seq_len) | key_padding_mask.view(bsz, 1, 1, seq_len)
    with torch.no_grad():
        dense_out = dense(x, src_mask=merged_bool)

    # --- flex-attention path ---
    block_mask, fully_masked_rows = build_flex_local_mask(
        attention_mask, window, seq_len, block_size, DEVICE
    )
    with torch.no_grad():
        flex_out = flex(x, block_mask=block_mask, fully_masked_rows=fully_masked_rows)

    real_rows = attention_mask.bool()  # (B, L) -- positions whose output must match
    real_rows_embed = real_rows.unsqueeze(-1).expand(-1, -1, d_model)

    assert not flex_out.isnan().any(), "flex layer produced NaN!"
    print("  [OK] flex layer output has zero NaN")

    diff = (dense_out - flex_out)[real_rows_embed].abs().max().item()
    print(f"  max diff at real positions: {diff:.2e}")
    assert diff < 1e-3, f"flex layer diverges from dense reference at real positions: max diff {diff}"
    print("  [OK] flex layer output matches dense-masked reference at every real position")

    # Positions the dense design would call "fully masked" (query with no real,
    # in-window key) should be zero in both -- confirms the max-pool-based
    # fully_masked_rows diagnostic agrees with the dense design's isneginf-based one.
    dense_fully_masked = merged_bool.all(dim=-1).squeeze(1)  # (B, L)
    assert torch.equal(dense_fully_masked, fully_masked_rows), (
        "fully_masked_rows diagnostic disagrees between dense and flex implementations"
    )
    print("  [OK] fully_masked_rows diagnostic agrees exactly between both implementations")
    print("CORRECTNESS vs. DENSE: PASS\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Overhead benchmark: full attention vs. dense-masked local vs. flex-attention local
# ══════════════════════════════════════════════════════════════════════════════

def _build_dense_encoder(layer_cls, d_model, nhead, dim_feedforward, dropout, nlayers):
    layer = layer_cls(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                       dropout=dropout, batch_first=True)
    return nn.TransformerEncoder(layer, num_layers=nlayers).to(DEVICE)


def _benchmark_dense(encoder, x, make_masks, label, n_iters=20, warmup=5):
    encoder.train()
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
    torch.cuda.synchronize(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        step()
    torch.cuda.synchronize(DEVICE)
    t1 = time.perf_counter()
    ms_per_iter = (t1 - t0) / n_iters * 1000
    peak_mb = torch.cuda.max_memory_allocated(DEVICE) / 1e6
    print(f"  {label:32s}: {ms_per_iter:7.2f} ms/iter, peak {peak_mb:8.1f} MB")
    return ms_per_iter, peak_mb


def _benchmark_flex(layers, x, attention_mask, window, block_size, label, n_iters=20, warmup=5):
    params = [p for layer in layers for p in layer.parameters()]
    opt = torch.optim.SGD(params, lr=1e-3)
    for layer in layers:
        layer.train()
    bsz, seq_len, _ = x.shape

    def step():
        opt.zero_grad()
        # Rebuilt fresh every iteration -- matches the dense benchmark's "make_masks()
        # inside the timed region" convention (real per-forward-pass cost, not amortized).
        block_mask, fully_masked_rows = build_flex_local_mask(
            attention_mask, window, seq_len, block_size, DEVICE
        )
        h = x
        for layer in layers:
            h = layer(h, block_mask=block_mask, fully_masked_rows=fully_masked_rows)
        loss = h.sum()
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        step()
    torch.cuda.synchronize(DEVICE)
    t1 = time.perf_counter()
    ms_per_iter = (t1 - t0) / n_iters * 1000
    peak_mb = torch.cuda.max_memory_allocated(DEVICE) / 1e6
    print(f"  {label:32s}: {ms_per_iter:7.2f} ms/iter, peak {peak_mb:8.1f} MB")
    return ms_per_iter, peak_mb


def benchmark_overhead():
    print("=" * 70)
    print("2. OVERHEAD BENCHMARK: full vs. dense-masked local vs. flex-attention local")
    print("=" * 70)
    if not DEVICE.startswith("cuda"):
        print("  [SKIP] no CUDA device available\n")
        return

    d_model, nhead, dim_feedforward, dropout, nlayers = 32, 8, 16, 0.5, 5
    bsz, seq_len, window = 1024, 161, 4

    x = torch.randn(bsz, seq_len, d_model, device=DEVICE)
    lengths = torch.randint(low=int(seq_len * 0.85), high=seq_len + 1, size=(bsz,))
    attention_mask = torch.zeros(bsz, seq_len, dtype=torch.long)
    for i, l in enumerate(lengths):
        attention_mask[i, :l] = 1
    attention_mask = attention_mask.to(DEVICE)
    key_padding_mask = ~attention_mask.bool()

    pos = torch.arange(seq_len, device=DEVICE)
    window_bool_4d = ((pos.unsqueeze(0) - pos.unsqueeze(1)).abs() > window).view(1, 1, seq_len, seq_len)

    print(f"  shape: d_model={d_model}, nhead={nhead}, nlayers={nlayers}, "
          f"seq_len={seq_len}, batch={bsz}, window={window}, "
          f"avg padding ~{(1-attention_mask.float().mean()).item()*100:.1f}%\n")

    full_enc = _build_dense_encoder(nn.TransformerEncoderLayer, d_model, nhead, dim_feedforward, dropout, nlayers)
    _benchmark_dense(full_enc, x, lambda: (None, key_padding_mask), "Full attention")

    dense_local_enc = _build_dense_encoder(LocalAttentionEncoderLayer, d_model, nhead, dim_feedforward, dropout, nlayers)

    def make_merged_masks():
        merged = window_bool_4d | key_padding_mask.view(bsz, 1, 1, seq_len)
        return merged, None

    _benchmark_dense(dense_local_enc, x, make_merged_masks, "Local (current: dense-masked)")

    for block_size in (16, 32, 64):
        flex_layers = nn.ModuleList([
            FlexAttentionLocalLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                     dropout=dropout, batch_first=True).to(DEVICE)
            for _ in range(nlayers)
        ])
        _benchmark_flex(flex_layers, x, attention_mask, window, block_size,
                         f"Local (flex-attention, block={block_size})")

    print("OVERHEAD BENCHMARK: done\n")


if __name__ == "__main__":
    test_correctness_vs_dense()
    benchmark_overhead()
    print("=" * 70)
    print("ALL CHECKS COMPLETE")
    print("=" * 70)
