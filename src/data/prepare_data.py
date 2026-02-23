import os
import random
import time
import torch
import numpy as np
from enum import IntEnum
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from src.data.datasets import get_loader
from src.data.tokenizer import Tokenizer
from src.data.walk_sampler import sample_random_walks
from src.data.dataset_cache import (
    save_dataset_cache,
    load_dataset_cache,
    cache_exists,
)
from src.data.stage_dataset import create_stage_dataloaders
from src.data.walk_dataset import WalkDataset
from src.utils.config import get_seed


class SplitID(IntEnum):
    TRAIN = 0  # context in train
    MASK = 1  # train targets (32%)
    VAL = 2  # val targets (10%)
    TEST = 3  # test targets (10%)
    BAD = -1  # non-edge positions (nodes)


def get_edge_list(cfg):
    print(f"Loading {cfg.dataset.name} dataset...")
    data = get_loader(cfg.dataset.name)(cfg)  # list of (u, v, label)
    # add vee mark emoji
    print(f"Success! ✅")
    return data


def split_edges(cfg, edges):
    print(f"Splitting edges for {cfg.dataset.name} dataset...")
    # Get seed for reproducibility
    seed = get_seed(cfg)

    # Validate ratios sum to 1.0
    train_ratio = float(cfg.dataset.train_ratio)
    mask_ratio = float(cfg.dataset.mask_ratio)
    val_ratio = float(cfg.dataset.val_ratio)
    test_ratio = float(cfg.dataset.test_ratio)
    total = train_ratio + mask_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Train, mask, val, and test ratios must sum to 1. Got {total:.6f}."
        )

    print(
        f"Ratios: train={train_ratio:.3f}, mask={mask_ratio:.3f}, val={val_ratio:.3f}, test={test_ratio:.3f}"
    )

    # Extract labels for stratification (3rd element of each edge tuple)
    edges_array = np.array(edges)
    labels = np.array([e[2] for e in edges])

    # HIERARCHICAL STRATIFIED SPLITTING: Three levels to maintain class balance
    # Each split preserves the original class distribution

    print(f"Performing stratified edge split with seed={seed}...")

    # Original class balance for validation
    original_pos_count = np.sum(labels == 1)
    original_pos_pct = 100.0 * original_pos_count / len(labels)
    print(
        f"Original class balance: {original_pos_count}/{len(labels)} positive ({original_pos_pct:.2f}%)"
    )

    # Step 1: Split TRAIN from remaining (train_ratio vs (1 - train_ratio))
    train_edges, remaining_edges, _, remaining_labels = train_test_split(
        edges_array,
        labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Step 2: Split MASK from temp
    # Recalculate mask ratio relative to remaining edges
    mask_ratio_of_remaining = mask_ratio / (1.0 - train_ratio)
    mask_edges, temp_edges, _, temp_labels = train_test_split(
        remaining_edges,
        remaining_labels,
        train_size=mask_ratio_of_remaining,
        stratify=remaining_labels,
        random_state=seed,
    )

    # Step 3: Split VAL from TEST
    # Recalculate test ratio relative to remaining edges
    test_ratio_of_temp = test_ratio / (val_ratio + test_ratio)
    val_edges, test_edges, _, _ = train_test_split(
        temp_edges,
        temp_labels,
        test_size=test_ratio_of_temp,
        stratify=temp_labels,
        random_state=seed,
    )

    # Convert numpy arrays back to list of tuples
    split = {
        "train": [tuple(e) for e in train_edges],
        "mask": [tuple(e) for e in mask_edges],
        "val": [tuple(e) for e in val_edges],
        "test": [tuple(e) for e in test_edges],
    }

    # Validate split sizes
    n_total = len(edges)
    n_train = len(split["train"])
    n_mask = len(split["mask"])
    n_val = len(split["val"])
    n_test = len(split["test"])

    actual_train_ratio = n_train / n_total
    actual_mask_ratio = n_mask / n_total
    actual_val_ratio = n_val / n_total
    actual_test_ratio = n_test / n_total

    print(f"Split sizes: train={n_train}, mask={n_mask}, val={n_val}, test={n_test}")
    print(
        f"Actual ratios: train={actual_train_ratio:.4f}, mask={actual_mask_ratio:.4f}, "
        f"val={actual_val_ratio:.4f}, test={actual_test_ratio:.4f}"
    )

    # Verify class balance in each split
    for split_name in ["train", "mask", "val", "test"]:
        split_edges_list = split[split_name]
        split_labels = [e[2] for e in split_edges_list]
        split_pos_count = sum(1 for l in split_labels if l == 1)
        split_pos_pct = 100.0 * split_pos_count / len(split_labels)
        diff_pct = abs(split_pos_pct - original_pos_pct)
        status = "✓" if diff_pct <= 2.0 else "✗"
        print(
            f"  {split_name:6s}: {split_pos_count:7d}/{len(split_labels):7d} positive ({split_pos_pct:6.2f}%) "
            f"diff={diff_pct:.2f}% {status}"
        )

    print(f"Stratified splitting complete! ✅")

    # Lookup sets for fast membership
    train_set = {tuple(t) for t in split["train"]}
    mask_set = {tuple(t) for t in split["mask"]}
    val_set = {tuple(t) for t in split["val"]}
    test_set = {tuple(t) for t in split["test"]}
    print(f"Success! ✅")
    return train_set, mask_set, val_set, test_set


def get_walks(cfg, edges):
    print(f"Sampling random walks from {cfg.dataset.name} dataset...")

    # Get worker count - support both old and new config keys
    walk_workers = int(
        getattr(
            cfg.preprocess,
            "num_workers",
            getattr(cfg.preprocess, "walk_num_workers", 1),
        )
    )

    # Use canonical seed from reproducibility config (imported at top)
    walk_seed = get_seed(cfg)

    walks = sample_random_walks(
        edges,
        num_walks=int(cfg.dataset.num_walks),
        max_walk_length=cfg.dataset.max_walk_length,
        num_workers=walk_workers,
        seed=walk_seed,
    )
    print(f"Success! ✅")
    return walks


def get_tokenizer(cfg, walks, edges):
    print(f"Building tokenizer for {cfg.dataset.name} dataset...")
    tokenizer = Tokenizer()
    tokenizer.fit(walks, edges=edges)
    print(f"Success! ✅")
    return tokenizer


def encode_walks(
    walks,
    tokenizer: Tokenizer,
    edges,
    train_set,
    mask_set,
    val_set,
    test_set,
):
    """Highly optimized walk encoding with vectorized operations and minimal overhead"""

    # Pre-build split lookup once (keep as dict for O(1) lookup)
    split_lookup = {}
    split_lookup.update({t: SplitID.TEST for t in test_set})
    split_lookup.update({t: SplitID.VAL for t in val_set})
    split_lookup.update({t: SplitID.MASK for t in mask_set})
    split_lookup.update({t: SplitID.TRAIN for t in train_set})
    split_lookup_get = split_lookup.get  # Cache method

    # Cache all tokenizer lookups
    is_edge = tokenizer.is_edge
    parse_node = tokenizer.parse_node
    parse_edge_label = tokenizer.parse_edge_label
    unk_id = tokenizer.UNK_ID
    token2id = tokenizer.token2id
    token2id_get = token2id.get  # Cache dict.get method

    # Edge lookup for metadata (edge_id per (u, v, label))
    edge_to_id = {
        (int(u), int(v), int(label)): idx for idx, (u, v, label) in enumerate(edges)
    }

    # Pre-allocate result lists
    input_ids = []
    edge_split_masks = []
    edge_ids_list = []
    walk_ids_list = []
    positions_list = []
    walk_lengths_list = []

    # Process walks with minimal function calls
    BAD = SplitID.BAD
    for walk_idx, walk in enumerate(walks):
        walk_len = len(walk)

        # Pre-allocate arrays for this walk
        x = [0] * walk_len
        split_mask = [BAD] * walk_len
        edge_ids = [-1] * walk_len
        positions = list(range(walk_len))
        walk_lengths = [walk_len] * walk_len
        walk_ids = [walk_idx] * walk_len

        # Vectorize the main loop
        for i in range(walk_len):
            token = walk[i]
            # Single dictionary lookup per token
            x[i] = token2id_get(token, unk_id)

            # Only check edges (most tokens are nodes, so this branch is rare)
            if is_edge(token):
                u = parse_node(walk[i - 1]) if i > 0 else None
                v = parse_node(walk[i + 1]) if i < walk_len - 1 else None
                label = parse_edge_label(token)
                split_mask[i] = split_lookup_get((u, v, label), BAD)
                if u is not None and v is not None and label is not None:
                    edge_ids[i] = edge_to_id.get((int(u), int(v), int(label)), -1)

        # Convert to tensors once per walk
        input_ids.append(torch.tensor(x, dtype=torch.long))
        edge_split_masks.append(torch.tensor(split_mask, dtype=torch.long))
        edge_ids_list.append(torch.tensor(edge_ids, dtype=torch.long))
        walk_ids_list.append(torch.tensor(walk_ids, dtype=torch.long))
        positions_list.append(torch.tensor(positions, dtype=torch.long))
        walk_lengths_list.append(torch.tensor(walk_lengths, dtype=torch.long))

    return (
        input_ids,
        edge_split_masks,
        edge_ids_list,
        walk_ids_list,
        positions_list,
        walk_lengths_list,
    )


def _stage_views_from_base(
    input_ids: torch.Tensor,
    edge_split_mask: torch.Tensor,
    tokenizer: Tokenizer,
    edge_ids: torch.Tensor,
    walk_ids: torch.Tensor,
    positions: torch.Tensor,
    walk_lengths: torch.Tensor,
):
    """
    Build per-stage (input_ids, labels, attention_mask) with those rules:
      Train: allowed = TRAIN; target = MASK
      Val:   allowed = TRAIN|MASK; target = VAL
      Test:  allowed = TRAIN|MASK|VAL; target = TEST
    """
    ignore_index = tokenizer.UNK_LABEL_ID
    pad_id = int(tokenizer.PAD_ID)
    mask_id = int(tokenizer.MASK_ID)

    # base attention (nodes + all edges visible initially)
    base_attn = (input_ids != pad_id).long()
    is_edge_pos = edge_split_mask != SplitID.BAD
    # map token_id -> class_id (for labels)
    id2class = torch.full((tokenizer.vocab_size,), ignore_index, dtype=torch.long)
    for class_id, edge_tok in tokenizer.id2edge_label.items():
        tok_id = tokenizer.token2id.get(edge_tok, None)
        if tok_id is not None:
            id2class[tok_id] = int(class_id)

    def build_for_stage(allowed_splits, target_split: int):
        # attention: start from base and zero-out disallowed edges
        attn = base_attn.clone()
        allowed_edges = torch.zeros_like(edge_split_mask, dtype=torch.bool)
        for s in allowed_splits:
            allowed_edges |= edge_split_mask == s
        disallowed_edges = is_edge_pos & (~allowed_edges)
        attn[disallowed_edges] = 0

        # input ids: mask only target edges; mask disallowed edges too
        x = input_ids.clone()
        target_edges = edge_split_mask == target_split
        # labels from original token ids (before overwrite)
        labels = torch.full_like(input_ids, ignore_index)
        if target_edges.any():
            labels[target_edges] = id2class[x[target_edges]]
        # replace target edges with MASK token id
        x[target_edges] = mask_id
        # mask disallowed edges but don't give them labels (keep ignore_index)
        x[disallowed_edges] = mask_id

        return x, labels, attn

    # Stage definitions - include target splits in allowed splits for attention
    train_allowed = [SplitID.TRAIN, SplitID.MASK]  # Include MASK (target) in allowed
    val_allowed = [
        SplitID.TRAIN,
        SplitID.MASK,
        SplitID.VAL,
    ]  # Include VAL (target) in allowed
    test_allowed = [
        SplitID.TRAIN,
        SplitID.MASK,
        SplitID.VAL,
        SplitID.TEST,
    ]  # Include TEST (target) in allowed

    train_x, train_y, train_attn = build_for_stage(train_allowed, SplitID.MASK)
    val_x, val_y, val_attn = build_for_stage(val_allowed, SplitID.VAL)
    test_x, test_y, test_attn = build_for_stage(test_allowed, SplitID.TEST)

    train_meta = {
        "edge_ids": edge_ids,
        "walk_ids": walk_ids,
        "positions": positions,
        "walk_lengths": walk_lengths,
    }
    val_meta = {
        "edge_ids": edge_ids,
        "walk_ids": walk_ids,
        "positions": positions,
        "walk_lengths": walk_lengths,
    }
    test_meta = {
        "edge_ids": edge_ids,
        "walk_ids": walk_ids,
        "positions": positions,
        "walk_lengths": walk_lengths,
    }

    return (
        (train_x, train_y, train_attn, train_meta),
        (val_x, val_y, val_attn, val_meta),
        (test_x, test_y, test_attn, test_meta),
    )


def pad_and_build_stage_tensors(
    cfg,
    input_ids_list,
    edge_split_masks_list,
    edge_ids_list,
    walk_ids_list,
    positions_list,
    walk_lengths_list,
    tokenizer,
):
    print(f"Padding and building stage tensors for {cfg.dataset.name} dataset...")
    pad_id = int(tokenizer.PAD_ID)
    # pad base
    input_ids = pad_sequence(
        input_ids_list, batch_first=True, padding_value=pad_id
    ).long()
    edge_split_mask = pad_sequence(
        edge_split_masks_list, batch_first=True, padding_value=SplitID.BAD
    ).long()
    edge_ids = pad_sequence(edge_ids_list, batch_first=True, padding_value=-1).long()
    walk_ids = pad_sequence(walk_ids_list, batch_first=True, padding_value=-1).long()
    positions = pad_sequence(positions_list, batch_first=True, padding_value=-1).long()
    walk_lengths = pad_sequence(
        walk_lengths_list, batch_first=True, padding_value=-1
    ).long()

    # derive stage-specific views
    train_pack, val_pack, test_pack = _stage_views_from_base(
        input_ids,
        edge_split_mask,
        tokenizer,
        edge_ids,
        walk_ids,
        positions,
        walk_lengths,
    )
    print(f"Success! ✅")
    return (train_pack, val_pack, test_pack)


def make_dataloaders(cfg, train_pack, val_pack, test_pack):
    print(f"Creating DataLoaders for {cfg.dataset.name} dataset...")
    batch_size = int(cfg.training.batch_size)

    # Unpack all packs (always 4-tuple with metadata)
    train_x, train_y, train_attn, train_meta = train_pack
    val_x, val_y, val_attn, val_meta = val_pack
    test_x, test_y, test_attn, test_meta = test_pack

    train_ds = WalkDataset(train_x, train_y, train_attn, train_meta)
    val_ds = WalkDataset(val_x, val_y, val_attn, val_meta)
    test_ds = WalkDataset(test_x, test_y, test_attn, test_meta)

    # configurable worker options (set in config under training)
    try:
        num_workers = int(getattr(cfg.training, "num_workers", 4))
    except Exception:
        num_workers = 4
    try:
        pin_memory = bool(getattr(cfg.training, "pin_memory", True))
    except Exception:
        pin_memory = True
    try:
        persistent = bool(getattr(cfg.training, "persistent_workers", True))
    except Exception:
        persistent = True
    try:
        prefetch = int(getattr(cfg.training, "prefetch_factor", 2))
    except Exception:
        prefetch = 2

    # seeded generator for reproducible shuffling
    # get_seed is imported at top of file
    base_seed = get_seed(cfg)
    g = torch.Generator()
    g.manual_seed(base_seed)

    def worker_init_fn(worker_id):
        # seed python, numpy and torch in each worker deterministically
        if base_seed is None:
            return
        seed = base_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent and num_workers > 0),
        prefetch_factor=prefetch,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent and num_workers > 0),
        prefetch_factor=prefetch,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=g,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent and num_workers > 0),
        prefetch_factor=prefetch,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=g,
    )

    print(f"DataLoader config:")
    print(f"  num_workers: {num_workers}")
    print(f"  pin_memory: {pin_memory}")
    print(f"  persistent_workers: {persistent and num_workers > 0}")
    print(f"  prefetch_factor: {prefetch}")
    print(f"Success! ✅")
    print(f"DataLoader Config Summary:")
    print(
        f"  Train:  num_workers={num_workers}, pin_memory={pin_memory}, persistent={persistent}, prefetch={prefetch}"
    )
    print(
        f"  Val:    num_workers={num_workers}, pin_memory={pin_memory}, persistent={persistent}, prefetch={prefetch}"
    )
    print(
        f"  Test:   num_workers={num_workers}, pin_memory={pin_memory}, persistent={persistent}, prefetch={prefetch}"
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}


def compute_class_weights_from_train(train_pack, ignore_index, num_classes):
    """Compute class weights from train split only (inverse frequency).

    Args:
        train_pack: Tuple of (input_ids, labels, attention_mask, metadata) from train split
        ignore_index: Label value to ignore in computation
        num_classes: Number of classes

    Returns:
        List of weights (one per class), normalized
    """
    # Unpack 4-tuple (always with metadata)
    _, labels, _, _ = train_pack

    # Flatten and filter out ignore_index
    all_labels = labels.view(-1)
    valid_labels = all_labels[all_labels != ignore_index]

    if len(valid_labels) == 0:
        print("⚠️  Warning: No valid labels in train split, using uniform weights")
        return [1.0] * num_classes

    # Count samples per class
    class_counts = []
    for i in range(num_classes):
        count = (valid_labels == i).sum().item()
        class_counts.append(count)

    # Inverse frequency formula
    total = len(valid_labels)
    weights = []
    for count in class_counts:
        if count > 0:
            weight = total / (num_classes * count)
        else:
            weight = 1.0
        weights.append(weight)

    # Normalize to sum to num_classes
    weight_sum = sum(weights)
    weights = [w / weight_sum * num_classes for w in weights]

    return weights


def prepare_data(cfg):
    # Load from dataset cache (only supported format)
    dataset_cache_path = os.path.join(cfg.dataset.data_dir, "dataset_cache.pt")

    if cfg.preprocess.use_cache and cache_exists(dataset_cache_path):
        print(f"Loading dataset cache from {dataset_cache_path}...")
        cache_data = load_dataset_cache(dataset_cache_path)

        # Update config with metadata
        cfg.model.vocab_size = cache_data["metadata"]["vocab_size"]
        cfg.model.num_classes = cache_data["metadata"]["num_classes"]
        cfg.model.pad_id = cache_data["metadata"]["pad_id"]
        cfg.model.ignore_index = cache_data["metadata"]["ignore_index"]

        # Compute class weights if not in cache
        if "class_weights" in cache_data["metadata"]:
            cfg.model.class_weights = cache_data["metadata"]["class_weights"]

        # Get file size for reporting
        size_mb = os.path.getsize(dataset_cache_path) / (1024 * 1024)
        print(f"Success! ✅ (loaded {size_mb:.1f} MB)")

        # Create dataloaders from dataset cache
        batch_size = int(cfg.training.batch_size)
        num_workers = int(getattr(cfg.training, "num_workers", 4))
        return create_stage_dataloaders(cache_data, batch_size, num_workers)
    # Profile data creation steps to help diagnose slow preprocessing
    timings = {}

    t0 = time.time()
    edges = get_edge_list(cfg)
    timings["get_edge_list"] = time.time() - t0

    t0 = time.time()
    train_set, mask_set, val_set, test_set = split_edges(cfg, edges)
    timings["split_edges"] = time.time() - t0

    t0 = time.time()
    walks = get_walks(cfg, edges)
    timings["get_walks"] = time.time() - t0

    t0 = time.time()
    tokenizer = get_tokenizer(cfg, walks, edges)
    timings["get_tokenizer"] = time.time() - t0

    cfg.model.vocab_size = tokenizer.vocab_size
    cfg.model.num_classes = tokenizer.num_edge_tokens
    cfg.model.pad_id = tokenizer.PAD_ID
    cfg.model.ignore_index = tokenizer.UNK_LABEL_ID

    t0 = time.time()
    (
        input_lists,
        split_lists,
        edge_ids_list,
        walk_ids_list,
        positions_list,
        walk_lengths_list,
    ) = encode_walks(walks, tokenizer, edges, train_set, mask_set, val_set, test_set)
    timings["encode_walks"] = time.time() - t0

    t0 = time.time()
    train_pack, val_pack, test_pack = pad_and_build_stage_tensors(
        cfg,
        input_lists,
        split_lists,
        edge_ids_list,
        walk_ids_list,
        positions_list,
        walk_lengths_list,
        tokenizer,
    )
    timings["pad_and_build_stage_tensors"] = time.time() - t0

    # Compute class weights from train split only (mandatory for fair loss)
    class_weights = compute_class_weights_from_train(
        train_pack, cfg.model.ignore_index, cfg.model.num_classes
    )
    cfg.model.class_weights = class_weights
    print(f"✓ Class weights computed from train split: {class_weights}")

    # Save dataset cache (always save in new approach)
    if cfg.preprocess.save:
        print(f"Saving dataset cache to {dataset_cache_path}...")

        # Get base tensors from padding step
        pad_id = int(tokenizer.PAD_ID)
        input_ids = pad_sequence(
            input_lists, batch_first=True, padding_value=pad_id
        ).long()
        edge_split_mask = pad_sequence(
            split_lists, batch_first=True, padding_value=SplitID.BAD
        ).long()
        attention_base = (input_ids != pad_id).long()

        edge_ids = pad_sequence(
            edge_ids_list, batch_first=True, padding_value=-1
        ).long()
        walk_ids = pad_sequence(
            walk_ids_list, batch_first=True, padding_value=-1
        ).long()
        positions = pad_sequence(
            positions_list, batch_first=True, padding_value=-1
        ).long()
        walk_lengths = pad_sequence(
            walk_lengths_list, batch_first=True, padding_value=-1
        ).long()

        # Prepare splits dict
        splits_dict = {
            "train": train_set,
            "mask": mask_set,
            "val": val_set,
            "test": test_set,
        }

        # Prepare metadata
        metadata = {
            "vocab_size": cfg.model.vocab_size,
            "num_classes": cfg.model.num_classes,
            "pad_id": cfg.model.pad_id,
            "ignore_index": cfg.model.ignore_index,
            "class_weights": class_weights,
            "dataset_name": cfg.dataset.name,
            "seed": get_seed(cfg),
        }

        # Save dataset cache
        size_mb = save_dataset_cache(
            dataset_cache_path,
            walks,
            tokenizer,
            input_ids,
            edge_split_mask,
            attention_base,
            splits_dict,
            metadata,
            edge_ids=edge_ids,
            walk_ids=walk_ids,
            positions=positions,
            walk_lengths=walk_lengths,
        )
        print(f"✓ Dataset cache saved ({size_mb:.1f} MB)")

    # Print a short profile summary for debugging
    try:
        print("Data creation profiling (seconds):")
        for k, v in timings.items():
            print(f"  {k}: {v:.2f}s")
    except Exception:
        pass

    return make_dataloaders(cfg, train_pack, val_pack, test_pack)
