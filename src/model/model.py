import torch.nn as nn
import torch
import math


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
        pad_id = cfg.model.pad_id
        self.embed = nn.Embedding(
            cfg.model.vocab_size, cfg.model.embedding_dim, padding_idx=pad_id
        )
        max_length = 2 * cfg.dataset.max_walk_length + 1
        self.register_buffer(
            "pos_encoder",
            get_sinusoidal_encoding(max_length, cfg.model.embedding_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
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

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return self.out(x)
