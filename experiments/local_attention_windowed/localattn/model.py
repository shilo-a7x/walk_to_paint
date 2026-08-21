import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class LocalAttentionEncoderLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer whose self-attention bypasses nn.MultiheadAttention and
    calls scaled_dot_product_attention directly with a pre-merged mask.

    Two PyTorch quirks drive this design (full story: MASKING.md):

    - nn.MultiheadAttention (F.multi_head_attention_forward) requires any per-batch-item
      mask to be pre-expanded to exactly (bsz*nhead, L, L) — no broadcast across heads.
      Our window+padding mask varies per batch item on essentially every batch, so the
      stock path would materialise 8x (nhead) more memory than necessary.
      F.scaled_dot_product_attention broadcasts a (B,1,L,L) mask across heads for free —
      that's the whole reason this subclass exists.
    - nn.TransformerEncoderLayer.forward has a fused-kernel fast path that activates in
      eval mode whenever no forward hooks are attached (i.e. every real val/test loop) and
      skips _sa_block entirely, regardless of whether it's been overridden. That fused
      kernel produces NaN on padding/disallowed positions whose entire local window is
      also padding/disallowed (softmax(-inf,...,-inf)), which then propagates to real
      positions in later layers via 0.0 * NaN = NaN. forward() is overridden below to
      always take the eager path so _sa_block (not the fused kernel) runs unconditionally.

    The window+padding mask itself is built once per forward pass in
    TransformerModel.forward (not here) and passed in as a single already-merged
    src_mask, so no per-layer re-merge cost.
    """

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        # Only padding/disallowed-split positions can ever be fully masked (real
        # positions always keep their own diagonal — window includes distance 0, and a
        # real position's key is never excluded) — so this identifies exactly the rows
        # that would otherwise produce softmax(-inf,...,-inf) = NaN.
        fully_masked_rows = torch.isneginf(src_mask).all(dim=-1) if src_mask is not None else None

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, fully_masked_rows, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, fully_masked_rows, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

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

        dropout_p = mha.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, embed_dim)

        if fully_masked_rows is not None:
            # softmax(-inf,...,-inf) really does produce NaN here — this doesn't prevent
            # that, it overwrites it (masked_fill = direct assignment, not an arithmetic
            # op that would inherit the NaN) before any later layer can read this position
            # as a key/value and multiply it into a real position's output via
            # 0.0 * NaN = NaN. Safe unconditionally: these rows are exactly the
            # padding/disallowed-split positions, which are never a supervised/label
            # position and are never attended to as a key by any real position in any
            # layer (the same mask excludes them as keys everywhere) — nothing downstream
            # depends on their actual value.
            # fully_masked_rows is (1,1,L) when the mask is batch-independent (no
            # padding this batch) or (B,1,L) when merged with per-item padding —
            # squeeze+unsqueeze broadcasts correctly against (bsz, L, E) either way.
            rows = fully_masked_rows.squeeze(1).unsqueeze(-1)
            attn_out = attn_out.masked_fill(rows, 0.0)

        out = mha.out_proj(attn_out)
        return self.dropout1(out)


def get_sinusoidal_encoding(length, dim):
    pe = torch.zeros(length, dim)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TransformerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.node_context_mode = str(getattr(cfg.model, "node_context_mode", "none"))
        self.node_mask_prob = float(getattr(cfg.model, "node_mask_prob", 0.0))
        self.node_noise_sigma = float(getattr(cfg.model, "node_noise_sigma", 0.0))
        local_attention_window = getattr(cfg.model, "local_attention_window", None)
        self.local_attention_window = (
            int(local_attention_window) if local_attention_window is not None else None
        )
        pad_id = cfg.model.pad_id
        self.embed = nn.Embedding(
            cfg.model.vocab_size, cfg.model.embedding_dim, padding_idx=pad_id
        )
        max_length = 2 * cfg.dataset.max_walk_length + 1
        self.register_buffer(
            "pos_encoder",
            get_sinusoidal_encoding(max_length, cfg.model.embedding_dim),
        )
        encoder_layer_cls = (
            LocalAttentionEncoderLayer
            if self.local_attention_window is not None
            else nn.TransformerEncoderLayer
        )
        encoder_layer = encoder_layer_cls(
            d_model=cfg.model.embedding_dim,
            nhead=cfg.model.nhead,
            dim_feedforward=cfg.model.hidden_dim,
            dropout=cfg.model.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.model.nlayers
        )
        self.out = nn.Linear(cfg.model.embedding_dim, cfg.model.num_classes)

    def forward(self, input_ids, attention_mask=None, node_mask=None):
        x = self.embed(input_ids)

        if self.training and node_mask is not None:
            if self.node_context_mode == "mask_unscaled" and self.node_mask_prob > 0.0:
                keep = torch.rand_like(node_mask, dtype=torch.float) >= self.node_mask_prob
                node_keep = (~node_mask) | keep
                x = x * node_keep.unsqueeze(-1).to(x.dtype)
            elif self.node_context_mode == "noise" and self.node_noise_sigma > 0.0:
                noise = torch.randn_like(x) * self.node_noise_sigma
                x = x + noise * node_mask.unsqueeze(-1).to(x.dtype)

        x = x + self.pos_encoder[: input_ids.size(1)]
        if attention_mask is not None:
            # Convert mask to shape [batch_size, seq_len] with bool type
            # True = to be ignored, False = to attend
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        if self.local_attention_window is not None:
            seq_len = input_ids.size(1)
            pos = torch.arange(seq_len, device=input_ids.device)
            dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
            attn_mask = dist > self.local_attention_window  # (L, L) bool
            attn_mask = attn_mask.view(1, 1, seq_len, seq_len)
            if key_padding_mask is not None and key_padding_mask.any():
                # Fold padding/disallowed-split positions into the same mask, once,
                # instead of passing them separately for every layer to re-merge.
                attn_mask = attn_mask | key_padding_mask.view(-1, 1, 1, seq_len)
            src_key_padding_mask = None  # already folded into attn_mask above
        else:
            attn_mask = None
            src_key_padding_mask = key_padding_mask

        x = self.transformer(x, mask=attn_mask, src_key_padding_mask=src_key_padding_mask)
        return self.out(x)
