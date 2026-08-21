"""
Prototype correctness check + benchmark for BandedLocalAttentionLayer
(experiments/local_attention_windowed/localattn/banded_local_attention.py) against the
current production dense-masked LocalAttentionEncoderLayer
(experiments/local_attention_windowed/localattn/model.py -- a copy of the real
src/model/model.py, per this experiment's isolation requirement).

Standalone prototype -- does not import anything from the real repo's src/.

Run: .venv/bin/python experiments/local_attention_windowed/scripts/test_banded_local_attention.py
(run from the repo root; set EXP_DEVICE=cuda:N to pick a specific GPU)
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

from localattn.model import LocalAttentionEncoderLayer
from localattn.banded_local_attention import BandedLocalAttentionLayer, build_banded_local_mask

DEVICE = os.environ.get("EXP_DEVICE") or ("cuda:3" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def _copy_weights(dense_layer, banded_layer):
    banded_layer.load_state_dict(dense_layer.state_dict())


# ══════════════════════════════════════════════════════════════════════════════
# 1. Synthetic correctness: BandedLocalAttentionLayer vs. LocalAttentionEncoderLayer
# ══════════════════════════════════════════════════════════════════════════════

def test_correctness_vs_dense():
    print("=" * 70)
    print("1. BANDED (UNFOLD) LOCAL vs. DENSE-MASKED LOCAL -- CORRECTNESS")
    print("=" * 70)

    d_model, nhead, window = 32, 8, 4
    bsz, seq_len = 6, 47

    dense = LocalAttentionEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=0.0, batch_first=True
    ).to(DEVICE).eval()
    banded = BandedLocalAttentionLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=0.0, batch_first=True, window=window
    ).to(DEVICE).eval()
    _copy_weights(dense, banded)

    attention_mask = torch.ones(bsz, seq_len, dtype=torch.long)
    attention_mask[0, 20] = 0
    attention_mask[0, 21] = 0
    attention_mask[1, 30:] = 0
    attention_mask[2, 5:9] = 0
    attention_mask[3, max(0, 10 - window):10 + window + 1] = 0  # engineered fully-masked row at q=10
    attention_mask[4, 0] = 0    # boundary case: padding at the very first position
    attention_mask[5, -1] = 0   # boundary case: padding at the very last position
    attention_mask = attention_mask.to(DEVICE)

    x = torch.randn(bsz, seq_len, d_model, device=DEVICE)

    key_padding_mask = ~attention_mask.bool()
    pos = torch.arange(seq_len, device=DEVICE)
    window_bool = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs() > window
    merged_bool = window_bool.view(1, 1, seq_len, seq_len) | key_padding_mask.view(bsz, 1, 1, seq_len)
    with torch.no_grad():
        dense_out = dense(x, src_mask=merged_bool)

    allow_windows, fully_masked_rows = build_banded_local_mask(attention_mask, window, seq_len, DEVICE)
    with torch.no_grad():
        banded_out = banded(x, allow_windows=allow_windows, fully_masked_rows=fully_masked_rows)

    real_rows = attention_mask.bool()
    real_rows_embed = real_rows.unsqueeze(-1).expand(-1, -1, d_model)

    assert not banded_out.isnan().any(), "banded layer produced NaN!"
    print("  [OK] banded layer output has zero NaN")

    diff = (dense_out - banded_out)[real_rows_embed].abs().max().item()
    print(f"  max diff at real positions: {diff:.2e}")
    assert diff < 1e-4, f"banded layer diverges from dense reference at real positions: max diff {diff}"
    print("  [OK] banded layer output matches dense-masked reference at every real position")

    dense_fully_masked = merged_bool.all(dim=-1).squeeze(1)
    assert torch.equal(dense_fully_masked, fully_masked_rows), (
        "fully_masked_rows diagnostic disagrees between dense and banded implementations"
    )
    print("  [OK] fully_masked_rows diagnostic agrees exactly between both implementations")
    print("CORRECTNESS vs. DENSE: PASS\n")


def test_shape_sweep():
    print("=" * 70)
    print("1b. SHAPE/CONFIG SWEEP")
    print("=" * 70)
    configs = [
        dict(d_model=8, nhead=1, seq_len=10, bsz=1, window=2),
        dict(d_model=9, nhead=3, seq_len=13, bsz=5, window=1),
        dict(d_model=32, nhead=8, seq_len=161, bsz=4, window=4),
        dict(d_model=16, nhead=4, seq_len=7, bsz=2, window=0),
    ]
    for cfg in configs:
        d_model, nhead, seq_len, bsz, window = (
            cfg["d_model"], cfg["nhead"], cfg["seq_len"], cfg["bsz"], cfg["window"]
        )
        dense = LocalAttentionEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model, dropout=0.0, batch_first=True
        ).to(DEVICE).eval()
        banded = BandedLocalAttentionLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model, dropout=0.0, batch_first=True, window=window
        ).to(DEVICE).eval()
        _copy_weights(dense, banded)

        attention_mask = torch.ones(bsz, seq_len, dtype=torch.long)
        if bsz > 1:
            attention_mask[0, seq_len // 2:] = 0
        if bsz > 2:
            attention_mask[1, 1] = 0
        attention_mask = attention_mask.to(DEVICE)
        key_padding_mask = ~attention_mask.bool()
        pos = torch.arange(seq_len, device=DEVICE)
        window_bool = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs() > window
        merged = window_bool.view(1, 1, seq_len, seq_len)
        if key_padding_mask.any():
            merged = merged | key_padding_mask.view(bsz, 1, 1, seq_len)

        x = torch.randn(bsz, seq_len, d_model, device=DEVICE)
        with torch.no_grad():
            dense_out = dense(x, src_mask=merged)
        allow_windows, fully_masked_rows = build_banded_local_mask(attention_mask, window, seq_len, DEVICE)
        with torch.no_grad():
            banded_out = banded(x, allow_windows=allow_windows, fully_masked_rows=fully_masked_rows)

        assert not banded_out.isnan().any(), f"NaN with config {cfg}"
        assert banded_out.shape == (bsz, seq_len, d_model), f"shape mismatch with config {cfg}"
        real_rows_embed = attention_mask.bool().unsqueeze(-1).expand(-1, -1, d_model)
        diff = (dense_out - banded_out)[real_rows_embed].abs().max().item()
        assert diff < 1e-4, f"config {cfg} diverges: max diff {diff}"
        print(f"  [OK] d_model={d_model} nhead={nhead} seq_len={seq_len} bsz={bsz} window={window}: "
              f"no NaN, correct shape, max diff {diff:.2e}")
    print("SHAPE/CONFIG SWEEP: PASS\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Overhead benchmark: full attention vs. dense-masked local vs. banded local
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


def _benchmark_banded(layers, x, attention_mask, window, label, n_iters=20, warmup=5):
    params = [p for layer in layers for p in layer.parameters()]
    opt = torch.optim.SGD(params, lr=1e-3)
    for layer in layers:
        layer.train()
    bsz, seq_len, _ = x.shape

    def step():
        opt.zero_grad()
        allow_windows, fully_masked_rows = build_banded_local_mask(attention_mask, window, seq_len, DEVICE)
        h = x
        for layer in layers:
            h = layer(h, allow_windows=allow_windows, fully_masked_rows=fully_masked_rows)
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
    print("2. OVERHEAD BENCHMARK: full vs. dense-masked local vs. banded (unfold) local")
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

    banded_layers = nn.ModuleList([
        BandedLocalAttentionLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                   dropout=dropout, batch_first=True, window=window).to(DEVICE)
        for _ in range(nlayers)
    ])
    _benchmark_banded(banded_layers, x, attention_mask, window, "Local (banded/unfold)")

    print("OVERHEAD BENCHMARK: done\n")


if __name__ == "__main__":
    test_correctness_vs_dense()
    test_shape_sweep()
    benchmark_overhead()
    print("=" * 70)
    print("ALL CHECKS COMPLETE")
    print("=" * 70)
