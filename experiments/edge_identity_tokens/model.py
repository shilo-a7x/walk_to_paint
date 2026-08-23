"""TransformerModel variant for the edge-identity-token experiment.

Minimal copy of src/model/model.py::TransformerModel's core forward path (embed +
positional encoding + transformer + local-attention masking + classification head),
with one addition: a sign_embedding(3, d) added to the token embedding at every
position, exactly like positional encoding is added -- class 0/1 = negative/positive
sign, class 2 = "not applicable" (node position) or "hidden" (masked/target edge
position, see dataset.py). Skips the production model's other ablation-specific
branches (zero_node_tokens, node replacement noise, etc.) -- not relevant to this
pilot and would only add noise to interpreting the result.
"""
import math

import torch
import torch.nn as nn

from src.model.model import LocalAttentionEncoderLayer, get_sinusoidal_encoding


class EdgeIdentityTransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, hidden_dim, dropout, nlayers,
                 num_classes, max_walk_length, pad_id, local_attention_window=None):
        super().__init__()
        self.local_attention_window = local_attention_window
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.sign_embedding = nn.Embedding(3, embedding_dim)  # 0=neg, 1=pos, 2=n/a or hidden

        max_length = 2 * max_walk_length + 1
        self.register_buffer("pos_encoder", get_sinusoidal_encoding(max_length, embedding_dim))

        encoder_layer_cls = (
            LocalAttentionEncoderLayer if local_attention_window is not None else nn.TransformerEncoderLayer
        )
        encoder_layer = encoder_layer_cls(
            d_model=embedding_dim, nhead=nhead, dim_feedforward=hidden_dim,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.out = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids, sign_ids, attention_mask=None):
        x = self.embed(input_ids) + self.sign_embedding(sign_ids)
        x = x + self.pos_encoder[: input_ids.size(1)]

        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        if self.local_attention_window is not None:
            seq_len = input_ids.size(1)
            pos = torch.arange(seq_len, device=input_ids.device)
            dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
            attn_mask = dist > self.local_attention_window
            attn_mask = attn_mask.view(1, 1, seq_len, seq_len)
            if key_padding_mask is not None and key_padding_mask.any():
                attn_mask = attn_mask | key_padding_mask.view(-1, 1, 1, seq_len)
            src_key_padding_mask = None
        else:
            attn_mask = None
            src_key_padding_mask = key_padding_mask

        x = self.transformer(x, mask=attn_mask, src_key_padding_mask=src_key_padding_mask)
        return self.out(x)
