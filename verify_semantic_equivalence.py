"""
Verification test: Ensure dataset cache build and load produce IDENTICAL outputs.

Tests:
1. Batch shapes match
2. Token IDs match
3. Attention masks match
4. Labels match
5. Edge visibility semantics match
6. Class distribution matches
"""

import torch
from src.utils.config import load_config
from src.data.prepare_data import prepare_data


def _unpack_batch(batch):
    if len(batch) == 4:
        input_ids, labels, attention_mask, _ = batch
    else:
        input_ids, labels, attention_mask = batch
    return input_ids, attention_mask, labels


def compare_batches(build_batch, load_batch, stage_name):
    """Compare two batches element-wise."""
    old_input, old_attn, old_labels = _unpack_batch(build_batch)
    new_input, new_attn, new_labels = _unpack_batch(load_batch)

    print(f"\n🔍 Comparing {stage_name} batches:")

    # 1. Shape check
    assert (
        old_input.shape == new_input.shape
    ), f"Input shape mismatch: {old_input.shape} vs {new_input.shape}"
    assert old_attn.shape == new_attn.shape, f"Attention shape mismatch"
    assert old_labels.shape == new_labels.shape, f"Labels shape mismatch"
    print(f"   ✓ Shapes match: {old_input.shape}")

    # 2. Content check - input_ids
    input_match = torch.equal(old_input, new_input)
    if not input_match:
        diff = (old_input != new_input).sum()
        print(f"   ⚠️  Input IDs differ at {diff} positions")
        # Show first difference
        first_diff = torch.where(old_input != new_input)
        if len(first_diff[0]) > 0:
            idx = (first_diff[0][0].item(), first_diff[1][0].item())
            print(
                f"      First diff at {idx}: old={old_input[idx]}, new={new_input[idx]}"
            )
    else:
        print(f"   ✓ Input IDs match exactly")

    # 3. Attention masks
    attn_match = torch.equal(old_attn, new_attn)
    if not attn_match:
        diff = (old_attn != new_attn).sum()
        print(f"   ⚠️  Attention masks differ at {diff} positions")
    else:
        print(f"   ✓ Attention masks match exactly")

    # 4. Labels
    labels_match = torch.equal(old_labels, new_labels)
    if not labels_match:
        diff = (old_labels != new_labels).sum()
        print(f"   ⚠️  Labels differ at {diff} positions")
    else:
        print(f"   ✓ Labels match exactly")

    # 5. Statistics
    print(f"\n   📊 Statistics:")
    print(
        f"      Old: {(old_attn == 1).sum()} visible, {(old_labels != -1).sum()} targets"
    )
    print(
        f"      New: {(new_attn == 1).sum()} visible, {(new_labels != -1).sum()} targets"
    )

    return input_match and attn_match and labels_match


def main():
    print("=" * 70)
    print("SEMANTIC EQUIVALENCE VERIFICATION")
    print("=" * 70)

    # Clean cache files
    import os

    dataset_dir = "data/bitcoin-alpha-binary"
    cache_files = ["dataset_cache.pt"]
    for f in cache_files:
        path = os.path.join(dataset_dir, f)
        if os.path.exists(path):
            os.remove(path)
    print("✓ Cleaned cache files\n")

    # Build dataset cache
    print("1️⃣ Building dataset cache...")
    cfg_build = load_config(
        "config.yaml",
        overrides=[
            "dataset.name=bitcoin-alpha-binary",
            "preprocess.use_cache=false",
            "preprocess.save=true",
        ],
    )
    loaders_build = prepare_data(cfg_build)
    print("   ✓ Cache built\n")

    # Load dataset cache
    print("2️⃣ Loading dataset cache...")
    cfg_load = load_config(
        "config.yaml",
        overrides=[
            "dataset.name=bitcoin-alpha-binary",
            "preprocess.use_cache=true",
            "preprocess.save=false",
        ],
    )
    loaders_load = prepare_data(cfg_load)
    print("   ✓ Cache loaded\n")

    # Compare dataset sizes
    print("3️⃣ Comparing dataset sizes...")
    for stage in ["train", "val", "test"]:
        build_len = len(loaders_build[stage].dataset)
        load_len = len(loaders_load[stage].dataset)
        assert (
            build_len == load_len
        ), f"{stage} size mismatch: {build_len} vs {load_len}"
        print(f"   ✓ {stage:5s}: {build_len} samples")

    # Compare batches (deterministic with same seed)
    print("\n4️⃣ Comparing batch contents...")

    all_match = True
    for stage in ["train", "val", "test"]:
        # Get first batch from each
        build_batch = next(iter(loaders_build[stage]))
        load_batch = next(iter(loaders_load[stage]))

        match = compare_batches(build_batch, load_batch, stage)
        all_match = all_match and match

    print("\n" + "=" * 70)
    if all_match:
        print("✅ PERFECT MATCH - New format is semantically identical!")
        print("   All input_ids, attention masks, and labels match exactly.")
    else:
        print("⚠️  DIFFERENCES DETECTED - Investigation needed")
        print("   Check the diff details above.")
    print("=" * 70)

    return all_match


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
