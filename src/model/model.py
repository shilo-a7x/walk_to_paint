import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class LocalAttentionEncoderLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer whose self-attention bypasses nn.MultiheadAttention's
    mask handling and calls scaled_dot_product_attention directly with a mask
    broadcastable over heads, shape (bsz, 1, L, S).

    Why: nn.MultiheadAttention combines an explicit attn_mask with
    src_key_padding_mask by materializing a (bsz * nhead, L, S) tensor (see
    F.multi_head_attention_forward), which the SDPA backward pass then has to
    retain per layer. At this model's shape (embedding_dim=32, nhead=8 -> head_dim=4,
    seq_len up to 161, batch=1024) that turns a banded local-attention mask plus the
    routine padding mask (always present once batching variable-length walks) into a
    ~4x memory / ~35-40% wall-clock regression vs. full attention, measured on an L40S
    (see plan-performance.md Issue 1). Keeping the mask at (bsz, 1, L, S) lets SDPA's
    own kernel broadcast across heads instead of materializing per-head copies, which
    recovers attention-call cost to within noise of full attention's masked case.
    Only used for the local_attention_window path; full attention keeps the stock
    nn.TransformerEncoderLayer untouched.
    """

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

        mask = None
        if attn_mask is not None:
            mask = attn_mask.view(1, 1, seq_len, seq_len)
        if key_padding_mask is not None:
            kp = key_padding_mask.view(bsz, 1, 1, seq_len)
            mask = kp if mask is None else mask + kp

        dropout_p = mha.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, embed_dim)
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
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        if self.local_attention_window is not None:
            seq_len = input_ids.size(1)
            pos = torch.arange(seq_len, device=input_ids.device)
            dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
            attn_mask = dist > self.local_attention_window
        else:
            attn_mask = None

        x = self.transformer(x, mask=attn_mask, src_key_padding_mask=src_key_padding_mask)
        return self.out(x)
