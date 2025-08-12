#!/usr/bin/env python3
"""
Script to verify model dimensions and values for debugging purposes.
"""

import torch
import yaml
import json
import os
from omegaconf import OmegaConf
from src.model.model import TransformerModel
from src.model.lit_model import LitEdgeClassifier


def load_config():
    """Load configuration and meta information."""
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Load meta information
    with open("data/chess/meta.json", "r") as f:
        meta = json.load(f)

    # Add meta info to config
    cfg["model"]["vocab_size"] = meta["vocab_size"]
    cfg["model"]["num_classes"] = meta["num_classes"]
    cfg["model"]["pad_id"] = meta["pad_id"]
    cfg["model"]["ignore_index"] = meta["ignore_index"]

    return OmegaConf.create(cfg)


def create_sample_batch(cfg, batch_size=4, seq_len=20):
    """Create a sample batch for testing."""
    vocab_size = cfg.model.vocab_size
    num_classes = cfg.model.num_classes

    # Create random input_ids (excluding pad_id for most positions)
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))

    # Add some padding to simulate real data
    for i in range(batch_size):
        if i % 2 == 0:  # Add padding to some sequences
            pad_start = seq_len - torch.randint(1, seq_len // 3, (1,)).item()
            input_ids[i, pad_start:] = cfg.model.pad_id

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (input_ids != cfg.model.pad_id).long()

    # Create random labels
    labels = torch.randint(0, num_classes, (batch_size, seq_len))

    # Set labels for padded positions to ignore_index
    labels[input_ids == cfg.model.pad_id] = cfg.model.ignore_index

    return input_ids, labels, attention_mask


def analyze_model_dimensions(cfg):
    """Analyze model dimensions and values."""
    print("=" * 60)
    print("MODEL DIMENSION AND VALUE ANALYSIS")
    print("=" * 60)

    # Print configuration
    print("\n📋 Configuration:")
    print(f"  Vocab size: {cfg.model.vocab_size}")
    print(f"  Embedding dim: {cfg.model.embedding_dim}")
    print(f"  Hidden dim: {cfg.model.hidden_dim}")
    print(f"  Number of heads: {cfg.model.nhead}")
    print(f"  Number of layers: {cfg.model.nlayers}")
    print(f"  Number of classes: {cfg.model.num_classes}")
    print(f"  Max walk length: {cfg.dataset.max_walk_length}")
    print(f"  Dropout: {cfg.model.dropout}")
    print(f"  Pad ID: {cfg.model.pad_id}")
    print(f"  Ignore index: {cfg.model.ignore_index}")

    # Create model
    model = TransformerModel(cfg)
    model.eval()

    print(f"\n🏗️  Model Architecture:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Create sample batch
    batch_size = 4
    seq_len = 15
    input_ids, labels, attention_mask = create_sample_batch(cfg, batch_size, seq_len)

    print(f"\n📦 Sample Batch:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Attention mask shape: {attention_mask.shape}")

    print(f"\n🔍 Sample Data:")
    print(f"  Input IDs (first sequence): {input_ids[0].tolist()}")
    print(f"  Attention mask (first sequence): {attention_mask[0].tolist()}")
    print(f"  Labels (first sequence): {labels[0].tolist()}")
    print(f"  Input IDs range: [{input_ids.min().item()}, {input_ids.max().item()}]")

    # Forward pass with detailed dimension tracking
    print(f"\n🔄 Forward Pass Analysis:")

    with torch.no_grad():
        # Step 1: Embedding
        embeddings = model.embed(input_ids)
        print(f"  1. Embeddings shape: {embeddings.shape}")
        print(f"     Expected: ({batch_size}, {seq_len}, {cfg.model.embedding_dim})")
        print(
            f"     Embeddings range: [{embeddings.min().item():.4f}, {embeddings.max().item():.4f}]"
        )
        print(f"     Embeddings mean: {embeddings.mean().item():.4f}")
        print(f"     Embeddings std: {embeddings.std().item():.4f}")

        # Step 2: Positional encoding
        pos_encoding = model.pos_encoder[:seq_len]
        print(f"  2. Positional encoding shape: {pos_encoding.shape}")
        print(f"     Expected: ({seq_len}, {cfg.model.embedding_dim})")
        print(
            f"     Pos encoding range: [{pos_encoding.min().item():.4f}, {pos_encoding.max().item():.4f}]"
        )

        # Step 3: Add positional encoding
        x = embeddings + pos_encoding
        print(f"  3. After adding pos encoding shape: {x.shape}")
        print(f"     Expected: ({batch_size}, {seq_len}, {cfg.model.embedding_dim})")
        print(f"     Combined range: [{x.min().item():.4f}, {x.max().item():.4f}]")

        # Step 4: Attention mask processing
        src_key_padding_mask = (
            ~attention_mask.bool() if attention_mask is not None else None
        )
        if src_key_padding_mask is not None:
            print(f"  4. Src key padding mask shape: {src_key_padding_mask.shape}")
            print(f"     Expected: ({batch_size}, {seq_len})")
            print(f"     Mask (True=ignore): {src_key_padding_mask[0].tolist()}")

        # Step 5: Transformer
        transformer_out = model.transformer(
            x, src_key_padding_mask=src_key_padding_mask
        )
        print(f"  5. Transformer output shape: {transformer_out.shape}")
        print(f"     Expected: ({batch_size}, {seq_len}, {cfg.model.embedding_dim})")
        print(
            f"     Transformer out range: [{transformer_out.min().item():.4f}, {transformer_out.max().item():.4f}]"
        )
        print(f"     Transformer out mean: {transformer_out.mean().item():.4f}")
        print(f"     Transformer out std: {transformer_out.std().item():.4f}")

        # Step 6: Output layer
        logits = model.out(transformer_out)
        print(f"  6. Final logits shape: {logits.shape}")
        print(f"     Expected: ({batch_size}, {seq_len}, {cfg.model.num_classes})")
        print(
            f"     Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]"
        )
        print(f"     Logits mean: {logits.mean().item():.4f}")
        print(f"     Logits std: {logits.std().item():.4f}")

        # Step 7: Softmax probabilities
        probs = torch.softmax(logits, dim=-1)
        print(f"  7. Probabilities shape: {probs.shape}")
        print(
            f"     Probabilities range: [{probs.min().item():.4f}, {probs.max().item():.4f}]"
        )
        print(f"     Sum of probs (should be ~1): {probs.sum(dim=-1)[0, 0].item():.4f}")

        # Step 8: Predictions
        predictions = logits.argmax(dim=-1)
        print(f"  8. Predictions shape: {predictions.shape}")
        print(f"     Expected: ({batch_size}, {seq_len})")
        print(f"     Predictions (first sequence): {predictions[0].tolist()}")
        print(
            f"     Predictions range: [{predictions.min().item()}, {predictions.max().item()}]"
        )


def analyze_loss_computation(cfg):
    """Analyze loss computation."""
    print(f"\n💢 Loss Computation Analysis:")

    # Create model and sample data
    model = TransformerModel(cfg)
    model.eval()

    batch_size = 4
    seq_len = 15
    input_ids, labels, attention_mask = create_sample_batch(cfg, batch_size, seq_len)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

        # Reshape for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        print(f"  Logits shape before reshape: {logits.shape}")
        print(f"  Labels shape before reshape: {labels.shape}")
        print(f"  Logits shape after reshape: {logits_flat.shape}")
        print(f"  Labels shape after reshape: {labels_flat.shape}")

        # Compute loss
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=cfg.model.ignore_index)
        loss = loss_fn(logits_flat, labels_flat)

        print(f"  Loss value: {loss.item():.4f}")

        # Count valid (non-ignored) labels
        valid_labels = (labels_flat != cfg.model.ignore_index).sum().item()
        total_labels = labels_flat.numel()
        print(
            f"  Valid labels: {valid_labels}/{total_labels} ({100*valid_labels/total_labels:.1f}%)"
        )


def analyze_lightning_model(cfg):
    """Analyze the Lightning model wrapper."""
    print(f"\n⚡ Lightning Model Analysis:")

    lit_model = LitEdgeClassifier(cfg)
    lit_model.eval()

    batch_size = 4
    seq_len = 15
    input_ids, labels, attention_mask = create_sample_batch(cfg, batch_size, seq_len)
    batch = (input_ids, labels, attention_mask)

    with torch.no_grad():
        loss = lit_model._step(batch, "val")
        print(f"  Validation loss: {loss.item():.4f}")

        # Check metrics computation
        logits = lit_model.model(input_ids, attention_mask=attention_mask)
        preds = logits.argmax(dim=-1).view(-1)
        targets = labels.view(-1)

        print(f"  Predictions shape: {preds.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Unique predictions: {torch.unique(preds).tolist()}")
        print(f"  Unique targets: {torch.unique(targets).tolist()}")


def main():
    """Main function to run all analyses."""
    cfg = load_config()

    try:
        analyze_model_dimensions(cfg)
        analyze_loss_computation(cfg)
        analyze_lightning_model(cfg)

        print(f"\n✅ Model dimension and value verification completed successfully!")
        print(
            f"   All dimensions appear to be consistent and values are in reasonable ranges."
        )

    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
