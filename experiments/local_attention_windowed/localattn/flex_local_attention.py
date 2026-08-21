import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


def build_flex_local_mask(padding_mask, window, seq_len, block_size, device):
    """Builds the BlockMask + fully-masked-row diagnostic for banded local attention,
    once per forward pass (mirrors MASKING.md's merge-once design principle for the
    dense-masked implementation).

    padding_mask: (B, L) bool or {0,1}, True/1 = real (attendable-as-key) token,
        False/0 = padding or disallowed-split position. Same convention as
        TransformerModel.forward's `attention_mask` in the current dense-masked model.
    Returns (block_mask, fully_masked_rows) where fully_masked_rows is (B, L) bool,
    True at query positions whose entire window contains no attendable key -- the only
    place softmax(-inf,...,-inf)=NaN can occur, same root cause as the dense design
    (see MASKING.md's "NaN bug" section).
    """
    allow_kv = padding_mask.bool() if padding_mask.dtype != torch.bool else padding_mask
    bsz = allow_kv.shape[0]

    def mask_mod(b, h, q_idx, kv_idx):
        windowed = (q_idx - kv_idx).abs() <= window
        return windowed & allow_kv[b, kv_idx]

    block_mask = create_block_mask(
        mask_mod, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len,
        device=device, BLOCK_SIZE=block_size,
    )

    # fully_masked_rows[b, q] = True iff q has zero attendable keys in its window.
    # Computed as a 1D sliding-window OR (max-pool) over allow_kv -- O(L), not O(L^2) --
    # so this diagnostic doesn't reintroduce the quadratic cost the layer is meant to avoid.
    kernel = 2 * window + 1
    pooled = F.max_pool1d(
        allow_kv.float().unsqueeze(1), kernel_size=kernel, stride=1, padding=window
    ).squeeze(1)
    fully_masked_rows = pooled == 0.0  # (B, L) bool
    return block_mask, fully_masked_rows


class FlexAttentionLocalLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer whose self-attention uses torch.nn.attention.flex_attention
    with a banded (sliding-window) BlockMask, instead of LocalAttentionEncoderLayer's dense
    L x L masked SDPA. Genuinely sub-quadratic: flex_attention's block-sparse kernel skips
    fully-out-of-window blocks rather than computing and masking them.

    Unlike LocalAttentionEncoderLayer, this layer does NOT take a pre-merged (B,1,L,L)
    src_mask -- building one would defeat the point (that materialization IS the O(L^2)
    cost this design avoids). Instead it takes a small (B, L) padding_mask and a
    precomputed BlockMask (built once per forward pass by the caller, shared across
    layers -- same merge-once principle as the dense design, see MASKING.md).

    Known gap vs. the dense implementation: flex_attention's signature has no dropout_p
    parameter, so attention dropout (mha.dropout) is not applied here -- only FFN/residual
    dropout (dropout1/dropout2, inherited from nn.TransformerEncoderLayer) is. Numerical
    equivalence checks below therefore use dropout=0.0, matching the existing
    test_local_attention_masking.py precedent for its synthetic correctness check.
    """

    def forward(self, src, block_mask=None, fully_masked_rows=None, is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), block_mask, fully_masked_rows, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, block_mask, fully_masked_rows, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, block_mask, fully_masked_rows, is_causal=False):
        mha = self.self_attn
        bsz, seq_len, embed_dim = x.shape
        nhead = mha.num_heads
        head_dim = embed_dim // nhead

        qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, nhead, head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, nhead, head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, nhead, head_dim).transpose(1, 2)

        attn_out = flex_attention(q, k, v, block_mask=block_mask)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, embed_dim)

        if fully_masked_rows is not None:
            # Same mitigation as LocalAttentionEncoderLayer._sa_block: overwrite (not
            # arithmetically combine) the output at fully-masked rows before any later
            # layer can read a NaN value at that position via 0.0 * NaN = NaN.
            rows = fully_masked_rows.unsqueeze(-1)  # (B, L, 1) broadcasts against (B, L, E)
            attn_out = attn_out.masked_fill(rows, 0.0)

        out = mha.out_proj(attn_out)
        return self.dropout1(out)
