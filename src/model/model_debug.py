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


class TransformerModelDebug(nn.Module):
    """Debug version of TransformerModel with detailed logging and dimension checks."""

    def __init__(self, cfg, debug=False):
        super().__init__()
        self.debug = debug

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

        # Store config for debugging
        self.cfg = cfg

    def _debug_print(self, msg):
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(f"[DEBUG] {msg}")

    def _check_tensor_stats(self, tensor, name):
        """Check and print tensor statistics if debug mode is enabled."""
        if self.debug:
            self._debug_print(
                f"{name}: shape={tensor.shape}, "
                f"range=[{tensor.min():.4f}, {tensor.max():.4f}], "
                f"mean={tensor.mean():.4f}, std={tensor.std():.4f}"
            )

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        # Debug input
        if self.debug:
            self._debug_print(f"Input - batch_size: {batch_size}, seq_len: {seq_len}")
            self._debug_print(
                f"Input IDs range: [{input_ids.min()}, {input_ids.max()}]"
            )
            if attention_mask is not None:
                self._debug_print(
                    f"Attention mask sum: {attention_mask.sum()} / {attention_mask.numel()}"
                )

        # Embedding layer
        x = self.embed(input_ids)
        self._check_tensor_stats(x, "Embeddings")

        # Add positional encoding
        pos_enc = self.pos_encoder[:seq_len]
        self._check_tensor_stats(pos_enc, "Positional encoding")

        x = x + pos_enc
        self._check_tensor_stats(x, "After adding pos encoding")

        # Prepare attention mask
        if attention_mask is not None:
            # Convert mask to shape [batch_size, seq_len] with bool type
            # True = to be ignored, False = to attend
            src_key_padding_mask = ~attention_mask.bool()
            if self.debug:
                self._debug_print(f"Padding mask shape: {src_key_padding_mask.shape}")
                self._debug_print(
                    f"Ignored positions: {src_key_padding_mask.sum()} / {src_key_padding_mask.numel()}"
                )
        else:
            src_key_padding_mask = None

        # Transformer layers
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        self._check_tensor_stats(x, "Transformer output")

        # Output layer
        logits = self.out(x)
        self._check_tensor_stats(logits, "Final logits")

        # Additional checks
        if self.debug:
            # Check if logits are reasonable
            probs = torch.softmax(logits, dim=-1)
            self._debug_print(
                f"Probability range: [{probs.min():.4f}, {probs.max():.4f}]"
            )
            self._debug_print(f"Sum of probs (first position): {probs[0, 0].sum():.4f}")

            # Check predictions
            preds = logits.argmax(dim=-1)
            unique_preds = torch.unique(preds)
            self._debug_print(f"Unique predictions: {unique_preds.tolist()}")

        return logits


class TransformerModel(nn.Module):
    """Original TransformerModel - kept unchanged for compatibility."""

    def __init__(self, cfg):
        super().__init__()
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

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids) + self.pos_encoder[: input_ids.size(1)]
        if attention_mask is not None:
            # Convert mask to shape [batch_size, seq_len] with bool type
            # True = to be ignored, False = to attend
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return self.out(x)
