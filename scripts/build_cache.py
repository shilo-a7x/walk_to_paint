"""Build (and save) the keyed dataset cache for a given walk config, then exit.
CPU-only; no training. Used by the sweep orchestrator so full+local attention cells
for the same (dataset, num_walks) share one cache built exactly once.

Usage: python scripts/build_cache.py dataset.name=<ds> dataset.walk_strategy=k_cover \
         dataset.walk_k_min=5 dataset.num_walks=<nw>
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.data.prepare_data import prepare_data

if __name__ == "__main__":
    cfg = load_config("config.yaml", overrides=sys.argv[1:])
    cfg.training.use_cuda = False  # no GPU for cache build
    prepare_data(cfg)  # builds + saves the keyed cache if missing
    print("CACHE_BUILD_DONE")
