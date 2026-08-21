import torch
import torch.nn as nn
import torch.nn.functional as F


def build_banded_local_mask(padding_mask, window, seq_len, device):
    """Builds the per-query-window validity mask for banded local attention, once per
    forward pass (mirrors MASKING.md's merge-once design principle).

    padding_mask: (B, L) bool or {0,1}, True/1 = real (attendable-as-key) token, same
        convention as TransformerModel.forward's `attention_mask`.

    Returns:
      allow_windows: (B, L, 2*window+1) bool -- allow_windows[b, i, w] is True iff
        offset w (0..2*window, centered on `window`) points at a real, in-bounds,
        attendable key for query i. Positions past the sequence boundary (from the
        implicit zero-pad) are always False here, same as an out-of-window position
        would be in the dense design.
      fully_masked_rows: (B, L) bool -- True where a query has zero attendable keys
        in its window (only place softmax(-inf,...,-inf)=NaN can occur).
    """
    allow_kv = padding_mask.bool() if padding_mask.dtype != torch.bool else padding_mask
    bsz = allow_kv.shape[0]
    # Pad with False (invalid) on both ends -- out-of-bounds window offsets must never
    # be treated as valid, unlike zero-padding a value tensor where 0 could coincide
    # with a real value.
    padded = F.pad(allow_kv, (window, window), value=False)  # (B, L + 2*window)
    allow_windows = padded.unfold(1, 2 * window + 1, 1)  # (B, L, 2*window+1), a view
    fully_masked_rows = ~allow_windows.any(dim=-1)  # (B, L)
    return allow_windows, fully_masked_rows


class BandedLocalAttentionLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer whose self-attention gathers only the +-window band of
    keys/values per query (via unfold) and computes scores/softmax/output on that
    (B, H, L, 2*window+1) band directly -- never materializes an (L, L) score or mask
    tensor, unlike LocalAttentionEncoderLayer's dense masked SDPA. Genuinely O(L*w)
    rather than O(L^2) in both the score computation and the mask/validity bookkeeping.

    Unlike FlexAttentionLocalLayer (see flex_local_attention.py, rejected after
    benchmarking -- 4.5-5x slower / ~3.5x more memory than the dense design at this
    project's real L=161/window=4 shape, kernel-launch and BlockMask-construction
    overhead dominates at this tiny scale), this implementation supports attention
    dropout directly (via F.dropout on the explicit weight tensor), matching the
    dense design's `mha.dropout` semantics exactly rather than dropping it.
    """

    def __init__(self, *args, window, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = window

    def forward(self, src, allow_windows=None, fully_masked_rows=None, is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), allow_windows, fully_masked_rows, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, allow_windows, fully_masked_rows, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, allow_windows, fully_masked_rows, is_causal=False):
        mha = self.self_attn
        bsz, seq_len, embed_dim = x.shape
        nhead = mha.num_heads
        head_dim = embed_dim // nhead
        window = self.window

        qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, nhead, head_dim).transpose(1, 2)  # (B, H, L, Dh)
        k = k.view(bsz, seq_len, nhead, head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, nhead, head_dim).transpose(1, 2)

        k_pad = F.pad(k, (0, 0, window, window))  # (B, H, L+2w, Dh), zero-padded
        v_pad = F.pad(v, (0, 0, window, window))
        # unfold the sequence dim: (B, H, L, Dh, 2w+1) -- a strided view, no copy yet.
        k_windows = k_pad.unfold(2, 2 * window + 1, 1)
        v_windows = v_pad.unfold(2, 2 * window + 1, 1)

        scores = torch.einsum("bhid,bhidw->bhiw", q, k_windows) / (head_dim ** 0.5)
        if allow_windows is not None:
            scores = scores.masked_fill(~allow_windows.unsqueeze(1), float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # (B, H, L, 2w+1); NaN at fully-masked rows
        dropout_p = mha.dropout if self.training else 0.0
        if dropout_p > 0.0:
            weights = F.dropout(weights, p=dropout_p)

        attn_out = torch.einsum("bhiw,bhidw->bhid", weights, v_windows)  # (B, H, L, Dh)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, embed_dim)

        if fully_masked_rows is not None:
            # Same mitigation as LocalAttentionEncoderLayer._sa_block: overwrite (not
            # arithmetically combine) the NaN before any later layer can read it via
            # 0.0 * NaN = NaN.
            rows = fully_masked_rows.unsqueeze(-1)  # (B, L, 1)
            attn_out = attn_out.masked_fill(rows, 0.0)

        out = mha.out_proj(attn_out)
        return self.dropout1(out)
