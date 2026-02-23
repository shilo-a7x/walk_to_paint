"""
Profile transformer model forward pass performance with SYNTHETIC DATA.
Uses realistic walk length distributions matching the epinions dataset.
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
from src.utils.config import load_config, get_seed
from src.data.dataset_cache import load_dataset_cache, cache_exists
import argparse
import random
from pytorch_lightning import seed_everything


class SyntheticDataTransformerProfiler:
    def __init__(self, config_path="config.yaml", overrides=None, device="cuda"):
        """Initialize profiler with config and synthesize realistic walk lengths."""
        # Load config with overrides (like run.py does)
        self.cfg = load_config(config_path, overrides=overrides)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Set seed for reproducibility
        seed = get_seed(self.cfg)
        seed_everything(seed, workers=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Analyze walk length distribution from cache
        self._analyze_walk_lengths_from_cache()

        print(f"\n✓ Initialized profiler on {self.device}")
        print(f"  Dataset: {self.cfg.dataset.name}")

    def _analyze_walk_lengths_from_cache(self):
        """Load walk_lengths tensor from cache to understand distribution."""
        dataset_name = self.cfg.dataset.name
        data_dir = self.cfg.dataset.data_dir
        cache_path = Path(data_dir) / "dataset_cache.pt"

        if not cache_exists(str(cache_path)):
            print(f"⚠️  Cache not found at {cache_path}")
            print("Using default walk length distribution from config")
            self.walk_length_stats = {
                "min": 5,
                "max": self.cfg.dataset.max_walk_length,
                "mean": self.cfg.dataset.max_walk_length * 0.7,
                "median": self.cfg.dataset.max_walk_length * 0.7,
                "std": self.cfg.dataset.max_walk_length * 0.2,
            }
            self._set_default_tokenizer_fields()
            return

        print(
            f"Loading walk lengths and tokenizer state from cache (this may take a moment)..."
        )
        cache_data = load_dataset_cache(str(cache_path))
        walk_lengths = cache_data.get("encoded", {}).get("walk_lengths", None)

        # Load tokenizer metadata to set pad_id, vocab_size, etc.
        self._load_tokenizer_fields(cache_data)

        if walk_lengths is None:
            print("⚠️  No walk_lengths in cache, using config defaults")
            self.walk_length_stats = {
                "min": 5,
                "max": self.cfg.dataset.max_walk_length,
                "mean": self.cfg.dataset.max_walk_length * 0.7,
                "median": self.cfg.dataset.max_walk_length * 0.7,
                "std": self.cfg.dataset.max_walk_length * 0.2,
            }
            return

        # Analyze actual walk lengths
        actual_lengths = (
            walk_lengths[walk_lengths > 0].float().numpy()
        )  # Exclude padding

        self.walk_length_stats = {
            "min": int(actual_lengths.min()),
            "max": int(actual_lengths.max()),
            "mean": float(actual_lengths.mean()),
            "median": float(np.median(actual_lengths)),
            "std": float(actual_lengths.std()),
        }

        print(
            f"Walk length stats: min={self.walk_length_stats['min']}, "
            f"max={self.walk_length_stats['max']}, "
            f"mean={self.walk_length_stats['mean']:.1f}, "
            f"median={self.walk_length_stats['median']:.1f}"
        )

        # Cleanup
        del cache_data
        torch.cuda.empty_cache() if self.device.type == "cuda" else None

    def _load_tokenizer_fields(self, cache_data):
        """Load tokenizer metadata from cache and set it in config."""
        tokenizer_state = cache_data.get("tokenizer", {})
        metadata = cache_data.get("metadata", {})
        self.cfg.model.pad_id = tokenizer_state.get("PAD_ID", 0)
        self.cfg.model.vocab_size = tokenizer_state.get("vocab_size", 5000)
        self.cfg.model.num_classes = metadata.get("num_classes", 2)

    def _set_default_tokenizer_fields(self):
        """Set default tokenizer fields when cache is not available."""
        self.cfg.model.pad_id = 0
        self.cfg.model.vocab_size = 5000
        self.cfg.model.num_classes = 2

    def _create_model(self, embedding_dim, hidden_dim, nhead, nlayers, dropout):
        """Create a model with specified architecture."""
        cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        cfg.model.embedding_dim = embedding_dim
        cfg.model.hidden_dim = hidden_dim
        cfg.model.nhead = nhead
        cfg.model.nlayers = nlayers
        cfg.model.dropout = dropout

        model = TransformerModel(cfg).to(self.device)
        model.eval()
        return model, cfg

    def _warmup(self, model, num_warmup=10):
        """Warm up GPU/CPU with dummy forward passes."""
        dummy_input = torch.randint(0, self.cfg.model.vocab_size, (8, 20)).to(
            self.device
        )
        dummy_mask = torch.ones(8, 20, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input, dummy_mask)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _generate_batch_with_varying_lengths(
        self, batch_size, min_length=None, max_length=None
    ):
        """Generate a batch with varying realistic walk lengths."""
        if min_length is None:
            min_length = int(self.walk_length_stats["min"])
        if max_length is None:
            max_length = int(self.walk_length_stats["max"])

        # Sample walk lengths from a distribution similar to the data
        mean = self.walk_length_stats["mean"]
        std = self.walk_length_stats["std"]

        lengths = np.random.normal(mean, std, batch_size)
        lengths = np.clip(lengths, min_length, max_length).astype(int)

        max_seq_len = lengths.max()

        # Create padded batch
        input_ids = torch.randint(
            0, self.cfg.model.vocab_size, (batch_size, max_seq_len)
        )
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

        for i, length in enumerate(lengths):
            attention_mask[i, :length] = True

        return input_ids.to(self.device), attention_mask.to(self.device), lengths

    def profile_batch_size(self, batch_size, model, num_runs=50):
        """
        Profile a specific batch size using synthetic data with realistic walk lengths.

        Args:
            batch_size: Number of samples in batch
            model: The model to profile
            num_runs: Number of forward passes to average over

        Returns:
            dict with timing statistics
        """
        times = []
        batch_walk_lengths = []

        with torch.no_grad():
            for _ in range(num_runs):
                input_ids, attention_mask, lengths = (
                    self._generate_batch_with_varying_lengths(batch_size)
                )
                batch_walk_lengths.append(lengths)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = model(input_ids, attention_mask)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

        # Flatten walk lengths
        all_lengths = np.concatenate(batch_walk_lengths)

        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "median_ms": np.median(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "mean_walk_length": np.mean(all_lengths),
            "median_walk_length": np.median(all_lengths),
            "min_walk_length": np.min(all_lengths),
            "max_walk_length": np.max(all_lengths),
        }

    def profile_model_configs(
        self,
        embedding_dims=[16, 32, 64, 128],
        hidden_dims=[16, 32, 64, 128],
        nheads=[2, 4, 8],
        nlayers_list=[2, 3, 4, 5],
        dropout_vals=[0.0, 0.1, 0.5],
        batch_sizes=[16, 32, 64, 128, 256],
        num_runs=50,
    ):
        """
        Profile across different model configurations and batch sizes.

        Tests one parameter at a time while keeping others at base config values.
        """
        results = []

        base_emb = self.cfg.model.embedding_dim
        base_hidden = self.cfg.model.hidden_dim
        base_nhead = self.cfg.model.nhead
        base_nlayers = self.cfg.model.nlayers
        base_dropout = self.cfg.model.dropout
        base_batch = self.cfg.training.batch_size

        print(
            f"\nBase config: emb={base_emb}, hidden={base_hidden}, nhead={base_nhead}, "
            f"nlayers={base_nlayers}, dropout={base_dropout}, batch={base_batch}"
        )
        print("=" * 80)

        # Test 1: Vary embedding dimension
        print("\n1. Testing Embedding Dimensions...")
        for emb_dim in embedding_dims:
            # Hidden dim must be divisible by nhead
            if emb_dim % base_nhead != 0:
                continue
            model, cfg = self._create_model(
                emb_dim, base_hidden, base_nhead, base_nlayers, base_dropout
            )
            self._warmup(model)

            for bs in batch_sizes:
                print(f"  emb_dim={emb_dim}, batch_size={bs}...", end=" ", flush=True)
                stats = self.profile_batch_size(bs, model, num_runs)
                results.append(
                    {
                        "config_type": "embedding_dim",
                        "embedding_dim": emb_dim,
                        "hidden_dim": base_hidden,
                        "nhead": base_nhead,
                        "nlayers": base_nlayers,
                        "dropout": base_dropout,
                        "batch_size": bs,
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
            model, cfg = self._create_model(
                base_emb, hidden_dim, base_nhead, base_nlayers, base_dropout
            )
            self._warmup(model)

            for bs in batch_sizes:
                print(
                    f"  hidden_dim={hidden_dim}, batch_size={bs}...",
                    end=" ",
                    flush=True,
                )
                stats = self.profile_batch_size(bs, model, num_runs)
                results.append(
                    {
                        "config_type": "hidden_dim",
                        "embedding_dim": base_emb,
                        "hidden_dim": hidden_dim,
                        "nhead": base_nhead,
                        "nlayers": base_nlayers,
                        "dropout": base_dropout,
                        "batch_size": bs,
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
            # Embedding dim must be divisible by nhead
            if base_emb % nhead != 0:
                print(f"  Skipping nhead={nhead} (emb_dim={base_emb} not divisible)")
                continue

            model, cfg = self._create_model(
                base_emb, base_hidden, nhead, base_nlayers, base_dropout
            )
            self._warmup(model)

            for bs in batch_sizes:
                print(f"  nhead={nhead}, batch_size={bs}...", end=" ", flush=True)
                stats = self.profile_batch_size(bs, model, num_runs)
                results.append(
                    {
                        "config_type": "nhead",
                        "embedding_dim": base_emb,
                        "hidden_dim": base_hidden,
                        "nhead": nhead,
                        "nlayers": base_nlayers,
                        "dropout": base_dropout,
                        "batch_size": bs,
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
            model, cfg = self._create_model(
                base_emb, base_hidden, base_nhead, nlayers, base_dropout
            )
            self._warmup(model)

            for bs in batch_sizes:
                print(f"  nlayers={nlayers}, batch_size={bs}...", end=" ", flush=True)
                stats = self.profile_batch_size(bs, model, num_runs)
                results.append(
                    {
                        "config_type": "nlayers",
                        "embedding_dim": base_emb,
                        "hidden_dim": base_hidden,
                        "nhead": base_nhead,
                        "nlayers": nlayers,
                        "dropout": base_dropout,
                        "batch_size": bs,
                        **stats,
                    }
                )
                print(f"✓ {stats['mean_ms']:.2f}ms (±{stats['std_ms']:.2f})")

            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Test 5: Vary dropout (less important for inference timing, but included)
        print("\n5. Testing Dropout Values...")
        for dropout in dropout_vals:
            model, cfg = self._create_model(
                base_emb, base_hidden, base_nhead, base_nlayers, dropout
            )
            self._warmup(model)

            for bs in batch_sizes:
                print(f"  dropout={dropout}, batch_size={bs}...", end=" ", flush=True)
                stats = self.profile_batch_size(bs, model, num_runs)
                results.append(
                    {
                        "config_type": "dropout",
                        "embedding_dim": base_emb,
                        "hidden_dim": base_hidden,
                        "nhead": base_nhead,
                        "nlayers": base_nlayers,
                        "dropout": dropout,
                        "batch_size": bs,
                        **stats,
                    }
                )
                print(f"✓ {stats['mean_ms']:.2f}ms (±{stats['std_ms']:.2f})")

            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        df = pd.DataFrame(results)

        # Add derived metrics
        df["throughput_samples_per_sec"] = (df["batch_size"] / df["mean_ms"]) * 1000
        df["time_per_sample_ms"] = df["mean_ms"] / df["batch_size"]

        return df

    def save_results(self, df, output_dir="outputs/profiling"):
        """Save profiling results to CSV and generate plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save CSV
        csv_path = output_path / "profiling_results.csv"
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
            # Get base config (use first row as representative)
            base_row = subset.iloc[0]

            # Group by batch size and compute mean
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
        summary_path = output_path / "profiling_summary.txt"

        with open(summary_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("TRANSFORMER MODEL PROFILING SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Dataset: {self.cfg.dataset.name}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Data Type: Synthetic with realistic walk lengths from cache\n\n")

            f.write("Walk Length Distribution:\n")
            for key, val in self.walk_length_stats.items():
                f.write(f"  {key}: {val}\n")
            f.write("\n")

            f.write("Base Model Configuration:\n")
            f.write(f"  - Embedding Dim: {self.cfg.model.embedding_dim}\n")
            f.write(f"  - Hidden Dim: {self.cfg.model.hidden_dim}\n")
            f.write(f"  - Num Layers: {self.cfg.model.nlayers}\n")
            f.write(f"  - Num Heads: {self.cfg.model.nhead}\n")
            f.write(f"  - Dropout: {self.cfg.model.dropout}\n")
            f.write(f"  - Vocab Size: {self.cfg.model.vocab_size}\n\n")

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
                best_throughput_idx = subset["throughput_samples_per_sec"].idxmax()
                best_throughput = subset.loc[best_throughput_idx]
                f.write(
                    f"  Best Throughput: {config_type}={best_throughput[config_type]}, "
                    f"BS={best_throughput['batch_size']}, "
                    f"{best_throughput['throughput_samples_per_sec']:.0f} samples/sec\n"
                )

            f.write("\n" + "=" * 80 + "\n")
            f.write("OVERALL RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")

            # Overall best throughput
            best_overall = df.loc[df["throughput_samples_per_sec"].idxmax()]
            f.write(f"Best Overall Throughput:\n")
            f.write(f"  Config Type: {best_overall['config_type']}\n")
            f.write(f"  Embedding dim: {best_overall['embedding_dim']}\n")
            f.write(f"  Hidden dim: {best_overall['hidden_dim']}\n")
            f.write(f"  Num heads: {best_overall['nhead']}\n")
            f.write(f"  Num layers: {best_overall['nlayers']}\n")
            f.write(f"  Batch size: {best_overall['batch_size']}\n")
            f.write(
                f"  Throughput: {best_overall['throughput_samples_per_sec']:.0f} samples/sec\n"
            )
            f.write(
                f"  Time per sample: {best_overall['time_per_sample_ms']:.3f}ms\n\n"
            )

        print(f"✓ Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile transformer model with synthetic data"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to base config file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run profiling on",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=50,
        help="Number of batches to profile per configuration",
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
        default=[16, 32, 64],
        help="Embedding dimensions to test",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[16, 32, 64],
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
        "overrides",
        nargs=argparse.REMAINDER,
        help="Config overrides (e.g., dataset.name=epinions)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("TRANSFORMER PROFILING WITH SYNTHETIC DATA")
    print("(using realistic walk lengths from dataset cache)")
    print("=" * 80)

    # Initialize profiler
    profiler = SyntheticDataTransformerProfiler(
        args.config,
        overrides=args.overrides,
        device=args.device,
    )

    # Run profiling
    df = profiler.profile_model_configs(
        embedding_dims=args.embedding_dims,
        hidden_dims=args.hidden_dims,
        nheads=args.nheads,
        nlayers_list=args.nlayers,
        batch_sizes=args.batch_sizes,
        num_runs=args.num_runs,
    )

    # Save results
    profiler.save_results(df, args.output_dir)

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - profiling_results.csv")
    print(f"  - profiling_summary.txt")
    print(f"  - profile_*.png (one per config type)")
    print(f"  - batch_size_comparison.png")
    print(f"  - time_per_sample.png")


if __name__ == "__main__":
    main()
