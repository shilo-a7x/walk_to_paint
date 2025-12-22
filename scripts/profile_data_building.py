#!/usr/bin/env python3
"""
Analyze data building speed and recommend optimizations.
Profile different num_walks and max_walk_length combinations.
"""
import time
import json
from pathlib import Path
from omegaconf import OmegaConf
import torch
from src.utils.config import load_config
from src.data.datasets import get_loader
from src.data.walk_sampler import sample_random_walks


def profile_data_creation(dataset_name, num_walks_list, max_walk_length_list, output_file=None):
    """
    Profile data creation for different walk parameters.
    
    Args:
        dataset_name: "wiki-rfa", "epinions", or "slashdot090221"
        num_walks_list: list of num_walks to test, e.g., [100000, 500000, 1000000]
        max_walk_length_list: list of max_walk_length to test, e.g., [50, 80, 100]
        output_file: save results to JSON
    """
    cfg = load_config(config_path="config.yaml")
    cfg.dataset.name = dataset_name
    
    # Load edge list once (shared across all runs)
    print(f"Loading {dataset_name} edge list...")
    t0 = time.time()
    edges = get_loader(dataset_name)(cfg)
    load_time = time.time() - t0
    print(f"  Loaded {len(edges)} edges in {load_time:.2f}s\n")
    
    results = []
    num_params = len(num_walks_list) * len(max_walk_length_list)
    current = 0
    
    for num_walks in num_walks_list:
        for max_walk_length in max_walk_length_list:
            current += 1
            print(f"[{current}/{num_params}] Profiling num_walks={num_walks:,}, max_walk_length={max_walk_length}...")
            
            cfg.dataset.num_walks = num_walks
            cfg.dataset.max_walk_length = max_walk_length
            
            # Profile walk sampling
            t0 = time.time()
            walks = sample_random_walks(cfg, edges)
            walk_time = time.time() - t0
            
            result = {
                "num_walks": num_walks,
                "max_walk_length": max_walk_length,
                "walk_sampling_time": walk_time,
                "total_edges": len(edges),
                "walks_per_second": num_walks / walk_time,
                "time_per_1m_walks": (walk_time / num_walks) * 1_000_000,
            }
            
            print(f"  ✅ {walk_time:.2f}s ({result['walks_per_second']:.0f} walks/sec)")
            results.append(result)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"PROFILING SUMMARY ({dataset_name})")
    print(f"{'='*80}")
    print(f"{'num_walks':>15} {'max_len':>10} {'time(s)':>10} {'walks/sec':>12}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['num_walks']:>15,} {r['max_walk_length']:>10} {r['walk_sampling_time']:>10.2f} {r['walks_per_second']:>12.0f}")
    
    # Find best tradeoffs
    print(f"\n{'='*80}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*80}")
    
    # Sort by time
    sorted_by_time = sorted(results, key=lambda x: x['walk_sampling_time'])
    fastest = sorted_by_time[0]
    print(f"\n⚡ Fastest: {fastest['num_walks']:,} walks, length={fastest['max_walk_length']}")
    print(f"   Time: {fastest['walk_sampling_time']:.2f}s")
    
    # Find sweet spots (< 120s but reasonable num_walks)
    under_2min = [r for r in results if r['walk_sampling_time'] < 120]
    if under_2min:
        print(f"\n🎯 Under 2 minutes ({len(under_2min)} options):")
        for r in sorted(under_2min, key=lambda x: -x['num_walks'])[:3]:
            print(f"   {r['num_walks']:,} walks, length={r['max_walk_length']}: {r['walk_sampling_time']:.2f}s")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")
    
    return results


def estimate_preprocessing_time(dataset_name, num_walks, max_walk_length, profiling_data=None):
    """
    Estimate total preprocessing time based on profiling data.
    """
    # Load or run profiling
    if profiling_data is None:
        print(f"No profiling data provided, running quick profile...")
        profiling_data = profile_data_creation(dataset_name, [num_walks], [max_walk_length])
    
    if not profiling_data:
        return None
    
    # Find closest match in profiling data
    matching = [p for p in profiling_data 
                if p['num_walks'] == num_walks and p['max_walk_length'] == max_walk_length]
    
    if not matching:
        print(f"⚠️  No exact match in profiling data, using closest match...")
        matching = profiling_data
    
    walk_time = matching[0]['walk_sampling_time']
    
    # Estimate other stages (from previous runs, these are much faster)
    # Adjust based on dataset
    if dataset_name == "wiki-rfa":
        other_time = 50  # get_edge_list + split + tokenizer + encode + pad
    elif dataset_name == "epinions":
        other_time = 30
    elif dataset_name == "slashdot090221":
        other_time = 40
    else:
        other_time = 40  # default
    
    total_time = walk_time + other_time
    
    print(f"\n⏱️  Time Estimate for {dataset_name}:")
    print(f"   num_walks: {num_walks:,}")
    print(f"   max_walk_length: {max_walk_length}")
    print(f"   Walk sampling: {walk_time:.1f}s")
    print(f"   Other stages: ~{other_time}s")
    print(f"   TOTAL: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    return total_time


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python profile_data_building.py <dataset_name> [--quick] [--full]")
        print("\nExamples:")
        print("  python profile_data_building.py wiki-rfa --quick")
        print("  python profile_data_building.py epinions --full")
        sys.exit(1)
    
    dataset = sys.argv[1]
    
    # Quick profile (fewer combinations)
    if "--quick" in sys.argv:
        num_walks = [100000, 500000, 1000000]
        max_lengths = [50, 80]
    # Full profile (more combinations)
    elif "--full" in sys.argv:
        num_walks = [100000, 250000, 500000, 750000, 1000000]
        max_lengths = [40, 60, 80, 100]
    # Default: quick
    else:
        num_walks = [100000, 500000, 1000000]
        max_lengths = [50, 80]
    
    output_file = f"profiling_{dataset}.json"
    
    try:
        results = profile_data_creation(dataset, num_walks, max_lengths, output_file)
        
        # Estimate time for best trial from optuna
        print(f"\n{'='*80}")
        print(f"COMPARISON: Optuna Best vs. Quick Options")
        print(f"{'='*80}")
        
        optuna_best = {
            "wiki-rfa": (434857, 82),
            "epinions": (3181, 24),  # Would need to check actual
            "slashdot090221": (1995392, 89),  # Would need to check actual
        }
        
        if dataset in optuna_best:
            walks, length = optuna_best[dataset]
            estimate_preprocessing_time(dataset, walks, length, results)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
