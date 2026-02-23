#!/usr/bin/env python3
"""
Benchmark script to measure tokenizer performance improvements.
"""

import time
import sys
from src.utils.config import load_config
from src.data.prepare_data import (
    get_edge_list,
    split_edges,
    get_walks,
    get_tokenizer,
    encode_walks,
)


def benchmark_dataset(dataset_name):
    """Benchmark tokenizer performance on a dataset"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {dataset_name}")
    print(f"{'='*60}")

    cfg = load_config("config.yaml", overrides=[f"dataset.name={dataset_name}"])

    # Disable caching to force fresh computation
    cfg.preprocess.use_cache = False
    cfg.preprocess.save = False

    # Load data
    t0 = time.time()
    edges = get_edge_list(cfg)
    t_edges = time.time() - t0
    print(f"✓ Loaded edges: {len(edges)} edges in {t_edges:.2f}s")

    # Split edges
    t0 = time.time()
    train_set, mask_set, val_set, test_set = split_edges(cfg, edges)
    t_split = time.time() - t0
    print(f"✓ Split edges in {t_split:.2f}s")

    # Sample walks
    t0 = time.time()
    walks = get_walks(cfg, edges)
    t_walks = time.time() - t0
    print(f"✓ Sampled {len(walks)} walks in {t_walks:.2f}s")

    # Build tokenizer (the critical optimization)
    t0 = time.time()
    tokenizer = get_tokenizer(cfg, walks, edges)
    t_tokenizer = time.time() - t0
    print(f"✓ Built tokenizer (vocab={tokenizer.vocab_size}) in {t_tokenizer:.2f}s")

    # Encode walks (the other critical optimization)
    t0 = time.time()
    input_lists, split_lists = encode_walks(
        walks, tokenizer, train_set, mask_set, val_set, test_set
    )
    t_encode = time.time() - t0
    print(f"✓ Encoded walks in {t_encode:.2f}s")

    # Summary
    total = t_edges + t_split + t_walks + t_tokenizer + t_encode
    print(f"\nTiming Breakdown:")
    print(f"  Edge loading:    {t_edges:6.2f}s ({100*t_edges/total:5.1f}%)")
    print(f"  Edge splitting:  {t_split:6.2f}s ({100*t_split/total:5.1f}%)")
    print(f"  Walk sampling:   {t_walks:6.2f}s ({100*t_walks/total:5.1f}%)")
    print(f"  Tokenizer build: {t_tokenizer:6.2f}s ({100*t_tokenizer/total:5.1f}%) ⚡")
    print(f"  Walk encoding:   {t_encode:6.2f}s ({100*t_encode/total:5.1f}%) ⚡")
    print(f"  {'─'*50}")
    print(f"  Total:           {total:6.2f}s")

    # Performance metrics
    walks_per_sec = len(walks) / t_encode
    tokens_per_walk = sum(len(w) for w in walks) / len(walks)
    tokens_per_sec = walks_per_sec * tokens_per_walk

    print(f"\nPerformance Metrics:")
    print(f"  Walks/sec:       {walks_per_sec:,.0f}")
    print(f"  Tokens/walk:     {tokens_per_walk:.1f}")
    print(f"  Tokens/sec:      {tokens_per_sec:,.0f}")

    return {
        "dataset": dataset_name,
        "edges": len(edges),
        "walks": len(walks),
        "vocab": tokenizer.vocab_size,
        "t_tokenizer": t_tokenizer,
        "t_encode": t_encode,
        "total": total,
        "walks_per_sec": walks_per_sec,
        "tokens_per_sec": tokens_per_sec,
    }


def main():
    datasets = ["bitcoin-alpha-binary"]

    if len(sys.argv) > 1:
        datasets = sys.argv[1:]

    print("🚀 TOKENIZER OPTIMIZATION BENCHMARK")
    print("=" * 60)

    results = []
    for dataset in datasets:
        try:
            result = benchmark_dataset(dataset)
            results.append(result)
        except Exception as e:
            print(f"❌ Failed to benchmark {dataset}: {e}")
            import traceback

            traceback.print_exc()

    # Summary table
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Dataset':<25} {'Walks':>10} {'Encode(s)':>10} {'Walks/s':>12}")
        print(f"{'-'*60}")
        for r in results:
            print(
                f"{r['dataset']:<25} {r['walks']:>10} {r['t_encode']:>10.2f} {r['walks_per_sec']:>12,.0f}"
            )


if __name__ == "__main__":
    main()
