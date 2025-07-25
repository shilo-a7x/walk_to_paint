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
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.model.embedding_dim)
        self.register_buffer(
            "pos_encoder",
            get_sinusoidal_encoding(
                cfg.dataset.max_walk_length, cfg.model.embedding_dim
            ),
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
        self.out = nn.Linear(cfg.model.embedding_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids) + self.pos_encoder[: input_ids.size(1)]
        x = self.transformer(x)
        return self.out(x)
