#!/usr/bin/env python3
"""
Training Performance Analysis
=============================
Profiles training time complexity for the Transformer model.
Analyzes how training time scales with:
- batch_size
- walk_length (sequence length)
- embedding_dim
- hidden_dim
- nhead
- nlayers
- num_samples (dataset size)

Usage:
    # Using dataset-specific config (recommended)
    python analyze_training_performance.py dataset.name=epinions
    
    # Or with explicit config file
    python analyze_training_performance.py --config config.yaml dataset.name=epinions
    
    # With additional overrides
    python analyze_training_performance.py dataset.name=wiki-rfa training.batch_size=256
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from src.model.model import TransformerModel
from src.utils.config import load_config


def create_synthetic_batch(batch_size, seq_len, vocab_size, num_classes):
    """Create synthetic batch for profiling"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    labels = torch.randint(0, num_classes, (batch_size, seq_len))
    return input_ids, attention_mask, labels


def profile_forward_pass(model, batch, device, num_warmup=5, num_runs=20):
    """Profile forward pass time"""
    input_ids, attention_mask, _ = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids, attention_mask)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Actual timing
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_ids, attention_mask)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    
    return np.mean(times), np.std(times)


def profile_training_step(model, batch, device, num_warmup=5, num_runs=20):
    """Profile full training step (forward + backward + optimizer)"""
    input_ids, attention_mask, labels = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Warmup
    for _ in range(num_warmup):
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        loss.backward()
        optimizer.step()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Actual timing
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        loss.backward()
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times), np.std(times)


def theoretical_complexity(seq_len, embed_dim, hidden_dim, nhead, nlayers, batch_size):
    """
    Calculate theoretical FLOPs for Transformer
    
    Per layer:
    - Multi-head attention: O(seq_len^2 * embed_dim)
    - Feedforward: O(seq_len * embed_dim * hidden_dim)
    
    Returns FLOPs (approximate)
    """
    # Attention FLOPs per layer
    # Q, K, V projections: 3 * seq_len * embed_dim^2
    # Attention scores: seq_len^2 * embed_dim
    # Attention output: seq_len^2 * embed_dim
    # Output projection: seq_len * embed_dim^2
    attention_flops = (
        3 * seq_len * batch_size * embed_dim * embed_dim  # QKV projections
        + batch_size * nhead * seq_len * seq_len * (embed_dim // nhead)  # Attention
        + seq_len * batch_size * embed_dim * embed_dim  # Output projection
    )
    
    # Feedforward FLOPs per layer
    # Two linear layers: seq_len * (embed_dim * hidden_dim + hidden_dim * embed_dim)
    ff_flops = seq_len * batch_size * 2 * embed_dim * hidden_dim
    
    # Total FLOPs
    total_flops = nlayers * (attention_flops + ff_flops)
    
    return {
        'attention_flops': attention_flops * nlayers,
        'ff_flops': ff_flops * nlayers,
        'total_flops': total_flops,
        'flops_per_sample': total_flops / batch_size
    }


def analyze_batch_size_scaling(cfg, device):
    """Analyze how training time scales with batch size"""
    print("\n" + "="*80)
    print("BATCH SIZE SCALING ANALYSIS")
    print("="*80)
    
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    seq_len = 2 * cfg.dataset.max_walk_length + 1
    
    results = []
    
    for bs in batch_sizes:
        # Create model
        model = TransformerModel(cfg).to(device)
        batch = create_synthetic_batch(
            bs, seq_len, cfg.model.vocab_size, cfg.model.num_classes
        )
        
        # Profile
        fwd_time, fwd_std = profile_forward_pass(model, batch, device)
        train_time, train_std = profile_training_step(model, batch, device)
        
        # Calculate per-sample time
        per_sample_fwd = fwd_time / bs * 1000  # ms
        per_sample_train = train_time / bs * 1000  # ms
        
        # Theoretical complexity
        theory = theoretical_complexity(
            seq_len, cfg.model.embedding_dim, cfg.model.hidden_dim,
            cfg.model.nhead, cfg.model.nlayers, bs
        )
        
        results.append({
            'batch_size': bs,
            'forward_time_ms': fwd_time * 1000,
            'forward_std_ms': fwd_std * 1000,
            'training_time_ms': train_time * 1000,
            'training_std_ms': train_std * 1000,
            'per_sample_fwd_ms': per_sample_fwd,
            'per_sample_train_ms': per_sample_train,
            'total_flops': theory['total_flops'],
            'flops_per_sample': theory['flops_per_sample']
        })
        
        print(f"\nBatch Size: {bs}")
        print(f"  Forward:  {fwd_time*1000:7.2f} ± {fwd_std*1000:5.2f} ms")
        print(f"  Training: {train_time*1000:7.2f} ± {train_std*1000:5.2f} ms")
        print(f"  Per-sample (forward):  {per_sample_fwd:6.3f} ms")
        print(f"  Per-sample (training): {per_sample_train:6.3f} ms")
        print(f"  Total FLOPs: {theory['total_flops']/1e9:.2f} GFLOPs")
        
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return pd.DataFrame(results)


def analyze_sequence_length_scaling(cfg, device):
    """Analyze how training time scales with sequence length (walk_length)"""
    print("\n" + "="*80)
    print("SEQUENCE LENGTH SCALING ANALYSIS")
    print("="*80)
    
    walk_lengths = [10, 20, 40, 60, 80, 100]
    batch_size = cfg.training.batch_size
    
    results = []
    
    for wl in walk_lengths:
        seq_len = 2 * wl + 1
        
        # Create temporary config with modified walk length
        temp_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
        temp_cfg.dataset.max_walk_length = wl
        
        # Create model
        model = TransformerModel(temp_cfg).to(device)
        batch = create_synthetic_batch(
            batch_size, seq_len, cfg.model.vocab_size, cfg.model.num_classes
        )
        
        # Profile
        fwd_time, fwd_std = profile_forward_pass(model, batch, device)
        train_time, train_std = profile_training_step(model, batch, device)
        
        # Theoretical complexity
        theory = theoretical_complexity(
            seq_len, cfg.model.embedding_dim, cfg.model.hidden_dim,
            cfg.model.nhead, cfg.model.nlayers, batch_size
        )
        
        results.append({
            'walk_length': wl,
            'seq_len': seq_len,
            'forward_time_ms': fwd_time * 1000,
            'forward_std_ms': fwd_std * 1000,
            'training_time_ms': train_time * 1000,
            'training_std_ms': train_std * 1000,
            'total_flops': theory['total_flops'],
            'attention_flops': theory['attention_flops'],
            'ff_flops': theory['ff_flops']
        })
        
        print(f"\nWalk Length: {wl} (seq_len={seq_len})")
        print(f"  Forward:  {fwd_time*1000:7.2f} ± {fwd_std*1000:5.2f} ms")
        print(f"  Training: {train_time*1000:7.2f} ± {train_std*1000:5.2f} ms")
        print(f"  Attention FLOPs: {theory['attention_flops']/1e9:.2f} GFLOPs")
        print(f"  FF FLOPs: {theory['ff_flops']/1e9:.2f} GFLOPs")
        
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return pd.DataFrame(results)


def analyze_model_dimensions(cfg, device):
    """Analyze how training time scales with model dimensions"""
    print("\n" + "="*80)
    print("MODEL DIMENSION SCALING ANALYSIS")
    print("="*80)
    
    embedding_dims = [16, 32, 64, 128]
    batch_size = cfg.training.batch_size
    seq_len = 2 * cfg.dataset.max_walk_length + 1
    
    results = []
    
    for emb_dim in embedding_dims:
        # Create temporary config
        temp_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
        temp_cfg.model.embedding_dim = emb_dim
        temp_cfg.model.hidden_dim = emb_dim  # Keep same ratio
        # Adjust nhead to be valid divisor
        temp_cfg.model.nhead = min(cfg.model.nhead, emb_dim // 4) or 1
        
        # Create model
        model = TransformerModel(temp_cfg).to(device)
        batch = create_synthetic_batch(
            batch_size, seq_len, cfg.model.vocab_size, cfg.model.num_classes
        )
        
        # Profile
        fwd_time, fwd_std = profile_forward_pass(model, batch, device)
        train_time, train_std = profile_training_step(model, batch, device)
        
        # Model size
        num_params = sum(p.numel() for p in model.parameters())
        
        # Theoretical complexity
        theory = theoretical_complexity(
            seq_len, emb_dim, emb_dim, temp_cfg.model.nhead, 
            cfg.model.nlayers, batch_size
        )
        
        results.append({
            'embedding_dim': emb_dim,
            'hidden_dim': emb_dim,
            'num_params': num_params,
            'nhead': temp_cfg.model.nhead,
            'forward_time_ms': fwd_time * 1000,
            'training_time_ms': train_time * 1000,
            'total_flops': theory['total_flops']
        })
        
        print(f"\nEmbedding Dim: {emb_dim}, Hidden Dim: {emb_dim}")
        print(f"  Parameters: {num_params:,}")
        print(f"  Forward:  {fwd_time*1000:7.2f} ms")
        print(f"  Training: {train_time*1000:7.2f} ms")
        print(f"  Total FLOPs: {theory['total_flops']/1e9:.2f} GFLOPs")
        
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return pd.DataFrame(results)


def analyze_transformer_layers(cfg, device):
    """Analyze how training time scales with number of transformer layers"""
    print("\n" + "="*80)
    print("TRANSFORMER LAYERS SCALING ANALYSIS")
    print("="*80)
    
    layer_counts = [1, 2, 3, 4, 5, 6]
    batch_size = cfg.training.batch_size
    seq_len = 2 * cfg.dataset.max_walk_length + 1
    
    results = []
    
    for nlayers in layer_counts:
        # Create temporary config
        temp_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
        temp_cfg.model.nlayers = nlayers
        
        # Create model
        model = TransformerModel(temp_cfg).to(device)
        batch = create_synthetic_batch(
            batch_size, seq_len, cfg.model.vocab_size, cfg.model.num_classes
        )
        
        # Profile
        fwd_time, fwd_std = profile_forward_pass(model, batch, device)
        train_time, train_std = profile_training_step(model, batch, device)
        
        # Model size
        num_params = sum(p.numel() for p in model.parameters())
        
        # Theoretical complexity
        theory = theoretical_complexity(
            seq_len, cfg.model.embedding_dim, cfg.model.hidden_dim,
            cfg.model.nhead, nlayers, batch_size
        )
        
        results.append({
            'nlayers': nlayers,
            'num_params': num_params,
            'forward_time_ms': fwd_time * 1000,
            'training_time_ms': train_time * 1000,
            'total_flops': theory['total_flops']
        })
        
        print(f"\nNumber of Layers: {nlayers}")
        print(f"  Parameters: {num_params:,}")
        print(f"  Forward:  {fwd_time*1000:7.2f} ms")
        print(f"  Training: {train_time*1000:7.2f} ms")
        print(f"  Total FLOPs: {theory['total_flops']/1e9:.2f} GFLOPs")
        
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return pd.DataFrame(results)


def estimate_epoch_time(cfg, num_samples, time_per_batch_ms):
    """Estimate full epoch training time"""
    num_batches = np.ceil(num_samples / cfg.training.batch_size)
    epoch_time_s = num_batches * time_per_batch_ms / 1000
    return epoch_time_s, num_batches


def generate_visualizations(results_dict, output_dir):
    """Generate comprehensive visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Batch size scaling
    if 'batch_size' in results_dict:
        df = results_dict['batch_size']
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Total time
        axes[0].plot(df['batch_size'], df['training_time_ms'], 'o-', label='Training')
        axes[0].plot(df['batch_size'], df['forward_time_ms'], 's--', label='Forward')
        axes[0].set_xlabel('Batch Size')
        axes[0].set_ylabel('Time (ms)')
        axes[0].set_title('Training Time vs Batch Size')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log', base=2)
        
        # Per-sample time
        axes[1].plot(df['batch_size'], df['per_sample_train_ms'], 'o-')
        axes[1].set_xlabel('Batch Size')
        axes[1].set_ylabel('Per-Sample Time (ms)')
        axes[1].set_title('Per-Sample Training Time vs Batch Size')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'batch_size_scaling.png', dpi=150)
        print(f"\nSaved: {output_dir / 'batch_size_scaling.png'}")
        plt.close()
    
    # 2. Sequence length scaling
    if 'sequence_length' in results_dict:
        df = results_dict['sequence_length']
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Time vs seq_len
        axes[0].plot(df['seq_len'], df['training_time_ms'], 'o-', label='Training')
        axes[0].plot(df['seq_len'], df['forward_time_ms'], 's--', label='Forward')
        axes[0].set_xlabel('Sequence Length')
        axes[0].set_ylabel('Time (ms)')
        axes[0].set_title('Training Time vs Sequence Length')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # FLOPs breakdown
        axes[1].plot(df['seq_len'], df['attention_flops']/1e9, 'o-', label='Attention')
        axes[1].plot(df['seq_len'], df['ff_flops']/1e9, 's--', label='Feedforward')
        axes[1].set_xlabel('Sequence Length')
        axes[1].set_ylabel('GFLOPs')
        axes[1].set_title('FLOPs vs Sequence Length')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sequence_length_scaling.png', dpi=150)
        print(f"Saved: {output_dir / 'sequence_length_scaling.png'}")
        plt.close()
    
    # 3. Model dimensions
    if 'dimensions' in results_dict:
        df = results_dict['dimensions']
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Time vs embedding dim
        axes[0].plot(df['embedding_dim'], df['training_time_ms'], 'o-', label='Training')
        axes[0].plot(df['embedding_dim'], df['forward_time_ms'], 's--', label='Forward')
        axes[0].set_xlabel('Embedding Dimension')
        axes[0].set_ylabel('Time (ms)')
        axes[0].set_title('Training Time vs Embedding Dimension')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Params vs time
        axes[1].scatter(df['num_params'], df['training_time_ms'])
        axes[1].set_xlabel('Number of Parameters')
        axes[1].set_ylabel('Training Time (ms)')
        axes[1].set_title('Training Time vs Model Size')
        axes[1].grid(True, alpha=0.3)
        axes[1].ticklabel_format(style='plain', axis='x')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dimension_scaling.png', dpi=150)
        print(f"Saved: {output_dir / 'dimension_scaling.png'}")
        plt.close()
    
    # 4. Transformer layers
    if 'layers' in results_dict:
        df = results_dict['layers']
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        ax.plot(df['nlayers'], df['training_time_ms'], 'o-', label='Training')
        ax.plot(df['nlayers'], df['forward_time_ms'], 's--', label='Forward')
        ax.set_xlabel('Number of Transformer Layers')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Training Time vs Number of Layers')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'layers_scaling.png', dpi=150)
        print(f"Saved: {output_dir / 'layers_scaling.png'}")
        plt.close()


def print_summary_recommendations(cfg, results_dict):
    """Print summary and recommendations for speeding up training"""
    print("\n" + "="*80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*80)
    
    print("\n📊 CURRENT CONFIGURATION:")
    print(f"  Dataset: {cfg.dataset.name}")
    print(f"  Walk Length: {cfg.dataset.max_walk_length}")
    print(f"  Num Walks: {cfg.dataset.num_walks:,}")
    print(f"  Batch Size: {cfg.training.batch_size}")
    print(f"  Embedding Dim: {cfg.model.embedding_dim}")
    print(f"  Hidden Dim: {cfg.model.hidden_dim}")
    print(f"  Num Heads: {cfg.model.nhead}")
    print(f"  Num Layers: {cfg.model.nlayers}")
    print(f"  Epochs: {cfg.training.epochs}")
    
    # Get baseline timing
    if 'batch_size' in results_dict:
        df = results_dict['batch_size']
        baseline_row = df[df['batch_size'] == cfg.training.batch_size]
        if not baseline_row.empty:
            baseline_time = baseline_row.iloc[0]['training_time_ms']
            epoch_time, num_batches = estimate_epoch_time(
                cfg, cfg.dataset.num_walks, baseline_time
            )
            total_time = epoch_time * cfg.training.epochs
            
            print(f"\n⏱️  ESTIMATED TRAINING TIME:")
            print(f"  Time per batch: {baseline_time:.2f} ms")
            print(f"  Batches per epoch: {int(num_batches):,}")
            print(f"  Time per epoch: {epoch_time/60:.2f} minutes")
            print(f"  Total training time: {total_time/3600:.2f} hours")
    
    print("\n🚀 RECOMMENDATIONS TO SPEED UP TRAINING:")
    
    print("\n1. BATCH SIZE:")
    print("   - Larger batch sizes reduce per-sample overhead")
    print("   - Current:", cfg.training.batch_size)
    if 'batch_size' in results_dict:
        df = results_dict['batch_size']
        best_bs = df.loc[df['per_sample_train_ms'].idxmin(), 'batch_size']
        speedup = df[df['batch_size'] == cfg.training.batch_size].iloc[0]['per_sample_train_ms'] / \
                  df[df['batch_size'] == best_bs].iloc[0]['per_sample_train_ms']
        print(f"   - Optimal: {int(best_bs)} (speedup: {speedup:.2f}x per sample)")
        print(f"   - ACTION: Increase to {int(best_bs)} if GPU memory allows")
    
    print("\n2. SEQUENCE LENGTH (WALK LENGTH):")
    print("   - Attention is O(L²), most expensive operation")
    print("   - Current walk_length:", cfg.dataset.max_walk_length)
    print("   - Current seq_len:", 2 * cfg.dataset.max_walk_length + 1)
    if 'sequence_length' in results_dict:
        df = results_dict['sequence_length']
        current_wl = cfg.dataset.max_walk_length
        current_time = df[df['walk_length'] == current_wl].iloc[0]['training_time_ms']
        reduced_wl = 40
        if reduced_wl in df['walk_length'].values:
            reduced_time = df[df['walk_length'] == reduced_wl].iloc[0]['training_time_ms']
            speedup = current_time / reduced_time
            print(f"   - Reducing to {reduced_wl} would give {speedup:.2f}x speedup")
            print(f"   - ACTION: Try walk_length=40 or 60 to balance quality and speed")
    
    print("\n3. MODEL DIMENSIONS:")
    print("   - Smaller dimensions = faster training")
    print("   - Current embedding_dim:", cfg.model.embedding_dim)
    print("   - Current hidden_dim:", cfg.model.hidden_dim)
    if 'dimensions' in results_dict:
        df = results_dict['dimensions']
        current_dim = cfg.model.embedding_dim
        if current_dim in df['embedding_dim'].values:
            current_time = df[df['embedding_dim'] == current_dim].iloc[0]['training_time_ms']
            smaller_dim = 32
            if smaller_dim in df['embedding_dim'].values:
                smaller_time = df[df['embedding_dim'] == smaller_dim].iloc[0]['training_time_ms']
                speedup = current_time / smaller_time
                print(f"   - Reducing to {smaller_dim} would give {speedup:.2f}x speedup")
                print(f"   - ACTION: Try embedding_dim={smaller_dim}, hidden_dim={smaller_dim}")
    
    print("\n4. NUMBER OF LAYERS:")
    print("   - Time scales linearly with layers")
    print("   - Current nlayers:", cfg.model.nlayers)
    if 'layers' in results_dict:
        df = results_dict['layers']
        fewer_layers = max(2, cfg.model.nlayers - 1)
        if fewer_layers in df['nlayers'].values:
            current_time = df[df['nlayers'] == cfg.model.nlayers].iloc[0]['training_time_ms']
            fewer_time = df[df['nlayers'] == fewer_layers].iloc[0]['training_time_ms']
            speedup = current_time / fewer_time
            print(f"   - Reducing to {fewer_layers} would give {speedup:.2f}x speedup")
            print(f"   - ACTION: Try nlayers={fewer_layers}")
    
    print("\n5. DATASET SIZE:")
    print("   - Consider using fewer walks during hyperparameter search")
    print(f"   - Current num_walks: {cfg.dataset.num_walks:,}")
    print(f"   - ACTION: Use 1M-2M walks for Optuna trials, full dataset for final training")
    
    print("\n6. OTHER OPTIMIZATIONS:")
    print("   - Use mixed precision training (torch.cuda.amp)")
    print("   - Enable torch.compile() (PyTorch 2.0+)")
    print("   - Use gradient accumulation for larger effective batch sizes")
    print("   - Consider Flash Attention for very long sequences")
    
    print("\n💡 QUICK WIN COMBINATIONS:")
    print("   Option A (Fast experimentation):")
    print("      walk_length=40, embedding_dim=32, nlayers=2, batch_size=1024")
    print("   Option B (Balanced):")
    print("      walk_length=60, embedding_dim=64, nlayers=3, batch_size=512")
    print("   Option C (Current - slower but potentially better quality):")
    print(f"      walk_length={cfg.dataset.max_walk_length}, "
          f"embedding_dim={cfg.model.embedding_dim}, "
          f"nlayers={cfg.model.nlayers}, "
          f"batch_size={cfg.training.batch_size}")


def main():
    parser = argparse.ArgumentParser(description='Analyze training performance')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to base config file')
    parser.add_argument('--output-dir', type=str, default='outputs/performance_analysis',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('overrides', nargs=argparse.REMAINDER,
                        help='Override config values (e.g., dataset.name=epinions)')
    args = parser.parse_args()
    
    # Load config with dataset-specific overrides (same as run.py)
    cfg = load_config(args.config, overrides=args.overrides)
    device = torch.device(args.device)
    
    print("="*80)
    print("TRAINING PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Dataset: {cfg.dataset.name}")
    
    # Run analyses
    results = {}
    
    print("\n🔬 Running profiling experiments...")
    print("(This may take several minutes)")
    
    results['batch_size'] = analyze_batch_size_scaling(cfg, device)
    results['sequence_length'] = analyze_sequence_length_scaling(cfg, device)
    results['dimensions'] = analyze_model_dimensions(cfg, device)
    results['layers'] = analyze_transformer_layers(cfg, device)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, df in results.items():
        csv_path = output_dir / f'{name}_scaling.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    generate_visualizations(results, output_dir)
    
    # Print recommendations
    print_summary_recommendations(cfg, results)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
