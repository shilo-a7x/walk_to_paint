"""
Profile transformer model forward pass performance with COMPLETE SYNTHETIC DATA.
No dataset dependencies - pure model profiling with random inputs.
"""

import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from omegaconf import OmegaConf
from src.model.model import TransformerModel
import argparse


class PureTransformerProfiler:
    def __init__(self, device="cuda", vocab_size=5000, num_classes=2):
        """Initialize profiler with minimal config needed for the model."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        print(f"✓ Initialized profiler on {self.device}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Num classes: {self.num_classes}")

    def _create_minimal_config(
        self, embedding_dim, hidden_dim, nhead, nlayers, dropout
    ):
        """Create minimal config needed for TransformerModel."""
        cfg = OmegaConf.create(
            {
                "model": {
                    "vocab_size": self.vocab_size,
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "nhead": nhead,
                    "nlayers": nlayers,
                    "dropout": dropout,
                    "num_classes": self.num_classes,
                    "pad_id": 0,
                    "ignore_index": -100,
                },
                "dataset": {
                    "max_walk_length": 200,  # Needed for positional encoding buffer
                },
            }
        )
        return cfg

    def _create_model(self, embedding_dim, hidden_dim, nhead, nlayers, dropout):
        """Create a model with specified architecture."""
        cfg = self._create_minimal_config(
            embedding_dim, hidden_dim, nhead, nlayers, dropout
        )
        model = TransformerModel(cfg).to(self.device)
        model.eval()
        return model

    def _warmup(self, model, num_warmup=10):
        """Warm up GPU/CPU with dummy forward passes."""
        dummy_input = torch.randint(0, self.vocab_size, (8, 20)).to(self.device)
        dummy_mask = torch.ones(8, 20, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input, dummy_mask)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _generate_synthetic_batch(self, batch_size, seq_length):
        """Generate completely synthetic batch with random inputs."""
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_length)).to(
            self.device
        )
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bool).to(
            self.device
        )
        return input_ids, attention_mask

    def profile_batch_size(self, batch_size, seq_length, model, num_runs=50):
        """
        Profile a specific batch size and sequence length.

        Args:
            batch_size: Number of samples in batch
            seq_length: Sequence length
            model: The model to profile
            num_runs: Number of forward passes to average over

        Returns:
            dict with timing statistics
        """
        times = []

        with torch.no_grad():
            for _ in range(num_runs):
                input_ids, attention_mask = self._generate_synthetic_batch(
                    batch_size, seq_length
                )

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = model(input_ids, attention_mask)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "median_ms": np.median(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "total_tokens": batch_size * seq_length,
        }

    def profile_model_configs(
        self,
        embedding_dims=[16, 32, 64, 128],
        hidden_dims=[16, 32, 64, 128],
        nheads=[2, 4, 8],
        nlayers_list=[2, 3, 4, 5],
        dropout_vals=[0.0, 0.1, 0.5],
        batch_sizes=[16, 32, 64, 128, 256],
        seq_lengths=[32, 64, 128],
        num_runs=50,
    ):
        """
        Profile across different model configurations, batch sizes, and sequence lengths.
        """
        results = []

        base_emb = 32
        base_hidden = 32
        base_nhead = 8
        base_nlayers = 3
        base_dropout = 0.1
        base_seq = 64

        print(
            f"\nBase config: emb={base_emb}, hidden={base_hidden}, nhead={base_nhead}, "
            f"nlayers={base_nlayers}, dropout={base_dropout}, seq={base_seq}"
        )
        print("=" * 80)

        # Test 1: Vary embedding dimension
        print("\n1. Testing Embedding Dimensions...")
        for emb_dim in embedding_dims:
            if emb_dim % base_nhead != 0:
                continue
            model = self._create_model(
                emb_dim, base_hidden, base_nhead, base_nlayers, base_dropout
            )
            self._warmup(model)

            for bs in batch_sizes:
                print(
                    f"  emb_dim={emb_dim}, batch_size={bs}, seq_len={base_seq}...",
                    end=" ",
                    flush=True,
                )
                stats = self.profile_batch_size(bs, base_seq, model, num_runs)
                results.append(
                    {
                        "config_type": "embedding_dim",
                        "embedding_dim": emb_dim,
                        "hidden_dim": base_hidden,
                        "nhead": base_nhead,
                        "nlayers": base_nlayers,
                        "dropout": base_dropout,
                        "batch_size": bs,
                        "seq_length": base_seq,
                        **stats,
                    }
                )
                print(f"✓ {stats['mean_ms']:.2f}ms (±{stats['std_ms']:.2f})")

            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Test 2: Vary hidden dimension
        print("\n2. Testing Hidden Dimensions...")
        for hidden_dim in hidden_dims:
            model = self._create_model(
                base_emb, hidden_dim, base_nhead, base_nlayers, base_dropout
            )
            self._warmup(model)

            for bs in batch_sizes:
                print(
                    f"  hidden_dim={hidden_dim}, batch_size={bs}, seq_len={base_seq}...",
                    end=" ",
                    flush=True,
                )
                stats = self.profile_batch_size(bs, base_seq, model, num_runs)
                results.append(
                    {
                        "config_type": "hidden_dim",
                        "embedding_dim": base_emb,
                        "hidden_dim": hidden_dim,
                        "nhead": base_nhead,
                        "nlayers": base_nlayers,
                        "dropout": base_dropout,
                        "batch_size": bs,
                        "seq_length": base_seq,
                        **stats,
                    }
                )
                print(f"✓ {stats['mean_ms']:.2f}ms (±{stats['std_ms']:.2f})")

            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Test 3: Vary number of heads
        print("\n3. Testing Number of Attention Heads...")
        for nhead in nheads:
            if base_emb % nhead != 0:
                print(f"  Skipping nhead={nhead} (emb_dim={base_emb} not divisible)")
                continue

            model = self._create_model(
                base_emb, base_hidden, nhead, base_nlayers, base_dropout
            )
            self._warmup(model)

            for bs in batch_sizes:
                print(
                    f"  nhead={nhead}, batch_size={bs}, seq_len={base_seq}...",
                    end=" ",
                    flush=True,
                )
                stats = self.profile_batch_size(bs, base_seq, model, num_runs)
                results.append(
                    {
                        "config_type": "nhead",
                        "embedding_dim": base_emb,
                        "hidden_dim": base_hidden,
                        "nhead": nhead,
                        "nlayers": base_nlayers,
                        "dropout": base_dropout,
                        "batch_size": bs,
                        "seq_length": base_seq,
                        **stats,
                    }
                )
                print(f"✓ {stats['mean_ms']:.2f}ms (±{stats['std_ms']:.2f})")

            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Test 4: Vary number of layers
        print("\n4. Testing Number of Layers...")
        for nlayers in nlayers_list:
            model = self._create_model(
                base_emb, base_hidden, base_nhead, nlayers, base_dropout
            )
            self._warmup(model)

            for bs in batch_sizes:
                print(
                    f"  nlayers={nlayers}, batch_size={bs}, seq_len={base_seq}...",
                    end=" ",
                    flush=True,
                )
                stats = self.profile_batch_size(bs, base_seq, model, num_runs)
                results.append(
                    {
                        "config_type": "nlayers",
                        "embedding_dim": base_emb,
                        "hidden_dim": base_hidden,
                        "nhead": base_nhead,
                        "nlayers": nlayers,
                        "dropout": base_dropout,
                        "batch_size": bs,
                        "seq_length": base_seq,
                        **stats,
                    }
                )
                print(f"✓ {stats['mean_ms']:.2f}ms (±{stats['std_ms']:.2f})")

            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Test 5: Vary sequence length
        print("\n5. Testing Sequence Lengths...")
        model = self._create_model(
            base_emb, base_hidden, base_nhead, base_nlayers, base_dropout
        )
        self._warmup(model)

        for seq_len in seq_lengths:
            for bs in batch_sizes:
                print(f"  seq_len={seq_len}, batch_size={bs}...", end=" ", flush=True)
                stats = self.profile_batch_size(bs, seq_len, model, num_runs)
                results.append(
                    {
                        "config_type": "seq_length",
                        "embedding_dim": base_emb,
                        "hidden_dim": base_hidden,
                        "nhead": base_nhead,
                        "nlayers": base_nlayers,
                        "dropout": base_dropout,
                        "batch_size": bs,
                        "seq_length": seq_len,
                        **stats,
                    }
                )
                print(f"✓ {stats['mean_ms']:.2f}ms (±{stats['std_ms']:.2f})")

        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Test 6: Vary dropout
        print("\n6. Testing Dropout Values...")
        for dropout in dropout_vals:
            model = self._create_model(
                base_emb, base_hidden, base_nhead, base_nlayers, dropout
            )
            self._warmup(model)

            for bs in batch_sizes:
                print(
                    f"  dropout={dropout}, batch_size={bs}, seq_len={base_seq}...",
                    end=" ",
                    flush=True,
                )
                stats = self.profile_batch_size(bs, base_seq, model, num_runs)
                results.append(
                    {
                        "config_type": "dropout",
                        "embedding_dim": base_emb,
                        "hidden_dim": base_hidden,
                        "nhead": base_nhead,
                        "nlayers": base_nlayers,
                        "dropout": dropout,
                        "batch_size": bs,
                        "seq_length": base_seq,
                        **stats,
                    }
                )
                print(f"✓ {stats['mean_ms']:.2f}ms (±{stats['std_ms']:.2f})")

            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        df = pd.DataFrame(results)

        # Add derived metrics
        df["throughput_tokens_per_sec"] = (df["total_tokens"] / df["mean_ms"]) * 1000
        df["time_per_sample_ms"] = df["mean_ms"] / df["batch_size"]
        df["time_per_token_ms"] = df["mean_ms"] / df["total_tokens"]

        return df

    def save_results(self, df, output_dir="outputs/profiling"):
        """Save profiling results to CSV and generate plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save CSV
        csv_path = output_path / "profiling_results_synthetic.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved results to {csv_path}")

        # Generate plots
        self._plot_by_config_type(df, output_path)
        self._plot_batch_size_comparison(df, output_path)
        self._plot_time_per_sample(df, output_path)
        self._generate_summary(df, output_path)

    def _plot_by_config_type(self, df, output_path):
        """Create separate plots for each config type."""
        config_types = df["config_type"].unique()

        for config_type in config_types:
            subset = df[df["config_type"] == config_type]

            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # Get the varying parameter
            param_col = config_type
            unique_vals = sorted(subset[param_col].unique())

            # Plot 1: Absolute time vs batch size
            ax = axes[0]
            for val in unique_vals:
                data = subset[subset[param_col] == val].sort_values("batch_size")
                ax.plot(
                    data["batch_size"],
                    data["mean_ms"],
                    marker="o",
                    label=f"{param_col}={val}",
                )

            ax.set_xlabel("Batch Size", fontsize=12)
            ax.set_ylabel("Forward Pass Time (ms)", fontsize=12)
            ax.set_title(
                f"Impact of {param_col} on Performance", fontsize=14, fontweight="bold"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log", base=2)

            # Plot 2: Time per sample
            ax = axes[1]
            for val in unique_vals:
                data = subset[subset[param_col] == val].sort_values("batch_size")
                ax.plot(
                    data["batch_size"],
                    data["time_per_sample_ms"],
                    marker="o",
                    label=f"{param_col}={val}",
                )

            ax.set_xlabel("Batch Size", fontsize=12)
            ax.set_ylabel("Time per Sample (ms)", fontsize=12)
            ax.set_title(
                f"Per-Sample Time: {param_col}", fontsize=14, fontweight="bold"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log", base=2)

            plt.tight_layout()
            plt.savefig(output_path / f"profile_{config_type}.png", dpi=300)
            plt.close()
            print(f"✓ Saved plot: profile_{config_type}.png")

    def _plot_batch_size_comparison(self, df, output_path):
        """Plot batch size scaling for each config type."""
        fig, ax = plt.subplots(figsize=(12, 7))

        config_types = df["config_type"].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(config_types)))

        for config_type, color in zip(config_types, colors):
            subset = df[df["config_type"] == config_type]
            batch_means = subset.groupby("batch_size")["mean_ms"].mean()
            ax.plot(
                batch_means.index,
                batch_means.values,
                marker="o",
                label=f"{config_type}",
                color=color,
                linewidth=2,
            )

        ax.set_xlabel("Batch Size", fontsize=12)
        ax.set_ylabel("Mean Forward Pass Time (ms)", fontsize=12)
        ax.set_title(
            "Batch Size Scaling Across Config Types", fontsize=14, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)

        plt.tight_layout()
        plt.savefig(output_path / "batch_size_comparison.png", dpi=300)
        plt.close()
        print(f"✓ Saved plot: batch_size_comparison.png")

    def _plot_time_per_sample(self, df, output_path):
        """Plot time per sample showing batching efficiency."""
        fig, ax = plt.subplots(figsize=(12, 7))

        config_types = df["config_type"].unique()

        for config_type in config_types:
            subset = df[df["config_type"] == config_type]
            batch_means = subset.groupby("batch_size")["time_per_sample_ms"].mean()
            ax.plot(
                batch_means.index,
                batch_means.values,
                marker="o",
                label=f"{config_type}",
                linewidth=2,
            )

        ax.set_xlabel("Batch Size", fontsize=12)
        ax.set_ylabel("Time per Sample (ms)", fontsize=12)
        ax.set_title(
            "Batching Efficiency: Time per Sample", fontsize=14, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")

        plt.tight_layout()
        plt.savefig(output_path / "time_per_sample.png", dpi=300)
        plt.close()
        print(f"✓ Saved plot: time_per_sample.png")

    def _generate_summary(self, df, output_path):
        """Generate text summary of profiling results."""
        summary_path = output_path / "profiling_summary_synthetic.txt"

        with open(summary_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("TRANSFORMER MODEL PROFILING SUMMARY - PURE SYNTHETIC DATA\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Device: {self.device}\n")
            f.write(f"Vocab Size: {self.vocab_size}\n")
            f.write(f"Num Classes: {self.num_classes}\n\n")

            f.write("-" * 80 + "\n")
            f.write("PERFORMANCE SUMMARY BY CONFIG TYPE\n")
            f.write("-" * 80 + "\n\n")

            for config_type in df["config_type"].unique():
                subset = df[df["config_type"] == config_type]
                f.write(f"\n{config_type.upper()}:\n")
                f.write("-" * 40 + "\n")

                # Find best and worst for this config type
                best_idx = subset["mean_ms"].idxmin()
                worst_idx = subset["mean_ms"].idxmax()

                best = subset.loc[best_idx]
                worst = subset.loc[worst_idx]

                f.write(
                    f"  Fastest: {config_type}={best[config_type]}, BS={best['batch_size']}, "
                    f"Time={best['mean_ms']:.2f}ms\n"
                )
                f.write(
                    f"  Slowest: {config_type}={worst[config_type]}, BS={worst['batch_size']}, "
                    f"Time={worst['mean_ms']:.2f}ms\n"
                )

                # Best throughput
                best_throughput_idx = subset["throughput_tokens_per_sec"].idxmax()
                best_throughput = subset.loc[best_throughput_idx]
                f.write(
                    f"  Best Throughput: {config_type}={best_throughput[config_type]}, "
                    f"BS={best_throughput['batch_size']}, "
                    f"{best_throughput['throughput_tokens_per_sec']:.0f} tokens/sec\n"
                )

            f.write("\n" + "=" * 80 + "\n")
            f.write("OVERALL RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")

            # Overall best throughput
            best_overall = df.loc[df["throughput_tokens_per_sec"].idxmax()]
            f.write(f"Best Overall Throughput:\n")
            f.write(f"  Config Type: {best_overall['config_type']}\n")
            f.write(f"  Embedding dim: {best_overall['embedding_dim']}\n")
            f.write(f"  Hidden dim: {best_overall['hidden_dim']}\n")
            f.write(f"  Num heads: {best_overall['nhead']}\n")
            f.write(f"  Num layers: {best_overall['nlayers']}\n")
            f.write(f"  Batch size: {best_overall['batch_size']}\n")
            f.write(f"  Sequence length: {best_overall['seq_length']}\n")
            f.write(
                f"  Throughput: {best_overall['throughput_tokens_per_sec']:.0f} tokens/sec\n"
            )
            f.write(f"  Time per sample: {best_overall['time_per_sample_ms']:.3f}ms\n")
            f.write(f"  Time per token: {best_overall['time_per_token_ms']:.6f}ms\n\n")

        print(f"✓ Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile transformer model - pure synthetic data"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run profiling on",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=5000,
        help="Vocabulary size for synthetic data",
    )
    parser.add_argument(
        "--num-classes", type=int, default=2, help="Number of output classes"
    )
    parser.add_argument(
        "--num-runs", type=int, default=50, help="Number of runs per configuration"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/profiling",
        help="Output directory for results",
    )
    parser.add_argument(
        "--embedding-dims",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128],
        help="Embedding dimensions to test",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128],
        help="Hidden dimensions to test",
    )
    parser.add_argument(
        "--nheads",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Number of attention heads to test",
    )
    parser.add_argument(
        "--nlayers",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
        help="Number of transformer layers to test",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="Sequence lengths to test",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("TRANSFORMER PROFILING - PURE SYNTHETIC DATA")
    print("=" * 80)

    # Initialize profiler
    profiler = PureTransformerProfiler(
        device=args.device,
        vocab_size=args.vocab_size,
        num_classes=args.num_classes,
    )

    # Run profiling
    df = profiler.profile_model_configs(
        embedding_dims=args.embedding_dims,
        hidden_dims=args.hidden_dims,
        nheads=args.nheads,
        nlayers_list=args.nlayers,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        num_runs=args.num_runs,
    )

    # Save results
    profiler.save_results(df, args.output_dir)

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - profiling_results_synthetic.csv")
    print(f"  - profiling_summary_synthetic.txt")
    print(f"  - profile_*.png (one per config type)")


if __name__ == "__main__":
    main()
