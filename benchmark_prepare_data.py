"""
Benchmark prepare_data.py: build-from-scratch (no save) vs build+save vs load-from-cache.
Uses the toy dataset so it runs quickly.
"""

import os
import time
import shutil
import tempfile

from omegaconf import OmegaConf

from src.data.prepare_data import (
    get_edge_list,
    split_edges,
    get_walks,
    get_tokenizer,
    encode_walks,
    build_ragged_arrays,
    build_runtime_cache_data,
    compute_class_weights_from_base,
)
from src.data.dataset_cache import save_dataset_cache, load_dataset_cache, cache_exists
from src.data.stage_dataset import create_stage_dataloaders
from src.utils.config import get_seed, load_config

import torch
from src.data.prepare_data import SplitID


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_cfg(data_dir: str, use_cache: bool, save: bool):
    cfg = load_config("config_toy.yaml")
    cfg.dataset.data_dir = data_dir
    cfg.preprocess.use_cache = use_cache
    cfg.preprocess.save = save
    cfg.training.num_workers = 0       # no subprocess overhead in benchmarks
    cfg.training.persistent_workers = False
    cfg.training.pin_memory = False
    cfg.training.prefetch_factor = None
    return cfg


def _build_from_scratch(cfg):
    """Run the full build pipeline, return (cache_data, tokenizer, timings)."""
    timings = {}

    t = time.perf_counter(); edges = get_edge_list(cfg);              timings["get_edge_list"] = time.perf_counter() - t
    t = time.perf_counter(); train_s, mask_s, val_s, test_s = split_edges(cfg, edges);  timings["split_edges"] = time.perf_counter() - t
    t = time.perf_counter(); walks = get_walks(cfg, edges);            timings["get_walks"] = time.perf_counter() - t
    t = time.perf_counter(); tokenizer = get_tokenizer(cfg, walks, edges); timings["get_tokenizer"] = time.perf_counter() - t

    cfg.model.vocab_size   = tokenizer.vocab_size
    cfg.model.num_classes  = tokenizer.num_edge_tokens
    cfg.model.pad_id       = tokenizer.PAD_ID
    cfg.model.ignore_index = tokenizer.UNK_LABEL_ID
    cfg.model.unk_id       = tokenizer.UNK_ID
    cfg.model.mask_id      = tokenizer.MASK_ID

    t = time.perf_counter()
    input_lists, split_lists, eid_list = encode_walks(
        walks, tokenizer, edges, train_s, mask_s, val_s, test_s
    )
    timings["encode_walks"] = time.perf_counter() - t

    t = time.perf_counter()
    ragged = build_ragged_arrays(input_lists, split_lists, eid_list, tokenizer)
    del input_lists, split_lists, eid_list
    timings["build_ragged_arrays"] = time.perf_counter() - t

    class_weights = compute_class_weights_from_base(
        ragged["flat_input_ids"], ragged["flat_split_mask"],
        tokenizer, cfg.model.num_classes, cfg.model.ignore_index,
    )
    cfg.model.class_weights = class_weights

    splits_dict = {"train": train_s, "mask": mask_s, "val": val_s, "test": test_s}
    metadata = {
        "vocab_size": cfg.model.vocab_size, "num_classes": cfg.model.num_classes,
        "pad_id": cfg.model.pad_id, "ignore_index": cfg.model.ignore_index,
        "class_weights": class_weights, "dataset_name": cfg.dataset.name,
        "seed": get_seed(cfg),
    }
    cache_data = build_runtime_cache_data(
        tokenizer,
        ragged["offsets"], ragged["flat_input_ids"],
        ragged["flat_split_mask"], ragged["flat_edge_ids"],
        splits_dict, metadata,
    )
    return cache_data, tokenizer, timings


def _time_save(cache_data, tokenizer, cache_path):
    t = time.perf_counter()
    save_dataset_cache(
        cache_path,
        tokenizer,
        cache_data["encoded"]["offsets"],
        cache_data["encoded"]["flat_input_ids"],
        cache_data["encoded"]["flat_split_mask"],
        cache_data["encoded"]["flat_edge_ids"],
        cache_data["splits"],
        cache_data["metadata"],
    )
    return time.perf_counter() - t


def _time_load(cache_path):
    t = time.perf_counter()
    data = load_dataset_cache(cache_path)
    return time.perf_counter() - t, data



# ── main ──────────────────────────────────────────────────────────────────────

def main():
    toy_data_dir = "data/toy"
    real_cache   = os.path.join(toy_data_dir, "dataset_cache.pt")

    print("=" * 60)
    print("BENCHMARK: prepare_data paths (toy dataset)")
    print("=" * 60)

    # ── 1. Build from scratch (no save) ──────────────────────────────────────
    print("\n[1] BUILD FROM SCRATCH (no save)")
    cfg1 = _make_cfg(toy_data_dir, use_cache=False, save=False)
    t_total = time.perf_counter()
    cache_data, _tok1, timings1 = _build_from_scratch(cfg1)
    total1 = time.perf_counter() - t_total

    print(f"  {'step':<35} {'time (s)':>10}")
    print(f"  {'-'*35} {'-'*10}")
    for k, v in timings1.items():
        print(f"  {k:<35} {v:>10.3f}")
    print(f"  {'TOTAL':<35} {total1:>10.3f}")

    # ── 2. Build from scratch + save ─────────────────────────────────────────
    print("\n[2] BUILD FROM SCRATCH + SAVE TO DISK")
    cfg2 = _make_cfg(toy_data_dir, use_cache=False, save=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_cache = os.path.join(tmp, "dataset_cache.pt")
        # copy the toy edge file so loader can find it
        shutil.copy(os.path.join(toy_data_dir, "out.toy"), tmp)
        cfg2.dataset.data_dir = tmp

        t_total = time.perf_counter()
        cache_data2, tokenizer2, timings2 = _build_from_scratch(cfg2)
        t_save = _time_save(cache_data2, tokenizer2, tmp_cache)
        
        total2 = time.perf_counter() - t_total

        print(f"  {'step':<35} {'time (s)':>10}")
        print(f"  {'-'*35} {'-'*10}")
        for k, v in timings2.items():
            print(f"  {k:<35} {v:>10.3f}")
        print(f"  {'save_dataset_cache':<35} {t_save:>10.3f}")
        print(f"  {'TOTAL':<35} {total2:>10.3f}")

        # ── 3. Load from cache ────────────────────────────────────────────────
        print("\n[3] LOAD FROM CACHE")
        cfg3 = _make_cfg(tmp, use_cache=True, save=False)

        t_total = time.perf_counter()
        t_load, _ = _time_load(tmp_cache)
        total3 = time.perf_counter() - t_total

        print(f"  {'step':<35} {'time (s)':>10}")
        print(f"  {'-'*35} {'-'*10}")
        print(f"  {'load_dataset_cache':<35} {t_load:>10.3f}")
        print(f"  {'TOTAL':<35} {total3:>10.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    rows = [
        ("Build (no save)",    total1),
        ("Build + save",       total2),
        ("Load from cache",    total3),
    ]
    baseline = total1
    for name, t in rows:
        pct = 100.0 * t / baseline
        print(f"  {name:<25} {t:>8.3f}s   ({pct:6.1f}% of build time)")

    fastest = min(rows, key=lambda x: x[1])
    print(f"\n  Fastest path: {fastest[0]}  ({fastest[1]:.3f}s)")


if __name__ == "__main__":
    main()
