import os
import time
import torch
import numpy as np
from enum import IntEnum
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor

from src.data.datasets import get_loader
from src.data.tokenizer import Tokenizer
from src.data.walk_sampler import sample_random_walks
from src.data.dataset_cache import (
    save_dataset_cache,
    load_dataset_cache,
    cache_exists,
)
from src.data.stage_dataset import create_stage_dataloaders
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


def get_walks(cfg, edges, train_set=None, mask_set=None, val_set=None, test_set=None):
    print(f"Sampling random walks from {cfg.dataset.name} dataset...")

    walk_workers = int(getattr(cfg.preprocess, "num_workers", 1))

    # Use canonical seed from reproducibility config (imported at top)
    walk_seed = get_seed(cfg)

    strategy = str(getattr(cfg.dataset, "walk_strategy", "uniform"))

    if strategy == "uniform":
        walks = sample_random_walks(
            edges,
            num_walks=int(cfg.dataset.num_walks),
            max_walk_length=cfg.dataset.max_walk_length,
            num_workers=walk_workers,
            seed=walk_seed,
        )
    else:
        from src.data.coverage_aware_sampler import sample_walks as _sample_coverage_walks
        walks = _sample_coverage_walks(
            edges=edges,
            cfg=cfg,
            seed=walk_seed,
            num_workers=walk_workers,
            train_set=train_set,
            mask_set=mask_set,
            val_set=val_set,
            test_set=test_set,
        )

    print(f"Success! ✅")
    return walks


def get_tokenizer(cfg, walks, edges):
    print(f"Building tokenizer for {cfg.dataset.name} dataset...")
    tokenizer = Tokenizer()
    tokenizer.fit(walks, edges=edges)
    print(f"Success! ✅")
    return tokenizer


# ---------------------------------------------------------------------------
# Module-level globals used by multiprocess worker processes.
# Set once via _worker_init(); never accessed in the main process.
# ---------------------------------------------------------------------------
_w_token2id = None
_w_edge_tokens = None
_w_node_id_lookup = None
_w_edge_label_lookup = None
_w_unk_id = None
_w_split_lookup = None
_w_edge_to_id = None
_w_bad_val = None


def _worker_init(
    token2id,
    edge_tokens,
    node_id_lookup,
    edge_label_lookup,
    unk_id,
    split_lookup,
    edge_to_id,
    bad_val,
):
    """ProcessPoolExecutor initializer: sets read-only globals once per worker."""
    global _w_token2id, _w_edge_tokens, _w_node_id_lookup, _w_edge_label_lookup
    global _w_unk_id, _w_split_lookup, _w_edge_to_id, _w_bad_val
    _w_token2id = token2id
    _w_edge_tokens = edge_tokens
    _w_node_id_lookup = node_id_lookup
    _w_edge_label_lookup = edge_label_lookup
    _w_unk_id = unk_id
    _w_split_lookup = split_lookup
    _w_edge_to_id = edge_to_id
    _w_bad_val = bad_val


def _encode_chunk(walks_chunk):
    """Encode a slice of walks using module-level worker globals.

    Called both from worker processes (via ProcessPoolExecutor) and directly
    from the main process when num_workers <= 1.
    """
    token2id_get = _w_token2id.get
    edge_tokens_set = _w_edge_tokens
    node_id_lookup = _w_node_id_lookup
    edge_label_lookup = _w_edge_label_lookup
    unk_id = _w_unk_id
    split_lookup_get = _w_split_lookup.get
    edge_to_id = _w_edge_to_id
    BAD = _w_bad_val

    input_ids = []
    edge_split_masks = []
    edge_ids_list = []

    for walk in walks_chunk:
        walk_len = len(walk)
        x = [0] * walk_len
        split_mask = [BAD] * walk_len
        edge_ids = [-1] * walk_len

        for i in range(walk_len):
            token = walk[i]
            x[i] = token2id_get(token, unk_id)
            if token in edge_tokens_set:
                u = node_id_lookup.get(walk[i - 1]) if i > 0 else None
                v = node_id_lookup.get(walk[i + 1]) if i < walk_len - 1 else None
                label = edge_label_lookup.get(token)
                split_mask[i] = split_lookup_get((u, v, label), BAD)
                if u is not None and v is not None and label is not None:
                    edge_ids[i] = edge_to_id.get((u, v, label), -1)

        input_ids.append(np.array(x, dtype=np.int64))
        edge_split_masks.append(np.array(split_mask, dtype=np.int64))
        edge_ids_list.append(np.array(edge_ids, dtype=np.int64))

    return input_ids, edge_split_masks, edge_ids_list


def encode_walks(
    walks,
    tokenizer: Tokenizer,
    edges,
    train_set,
    mask_set,
    val_set,
    test_set,
    num_workers: int = 1,
):
    """Walk encoding returning numpy int64 arrays.

    Returns 3 lists (input_ids, edge_split_masks, edge_ids) as np.int64 arrays per walk.
    positions / walk_ids / walk_lengths are reconstructable from the walk lengths after
    CSR array construction and are therefore NOT computed here.

    When num_workers > 1, walks are split into equal chunks and encoded in parallel
    using ProcessPoolExecutor.  Shared read-only state (tokenizer dicts, split lookup,
    edge→id map) is passed once via the pool initializer to avoid per-task pickling.
    """
    # Pre-build split lookup (O(1) per edge, built once here in the main process)
    split_lookup = {}
    split_lookup.update({t: SplitID.TEST for t in test_set})
    split_lookup.update({t: SplitID.VAL for t in val_set})
    split_lookup.update({t: SplitID.MASK for t in mask_set})
    split_lookup.update({t: SplitID.TRAIN for t in train_set})

    edge_to_id = {
        (int(u), int(v), int(label)): idx for idx, (u, v, label) in enumerate(edges)
    }

    init_args = (
        tokenizer.token2id,
        tokenizer._edge_tokens,
        tokenizer._token_to_node_id,
        tokenizer._token_to_edge_label,
        tokenizer.UNK_ID,
        split_lookup,
        edge_to_id,
        int(SplitID.BAD),
    )

    if num_workers <= 1:
        # Single-process path: zero fork/pickle overhead
        _worker_init(*init_args)
        return _encode_chunk(walks)

    # Multi-process path: split walks into chunks, one future per worker
    n = len(walks)
    chunk_size = max(1, (n + num_workers - 1) // num_workers)
    chunks = [walks[i : i + chunk_size] for i in range(0, n, chunk_size)]

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=init_args,
    ) as executor:
        chunk_results = list(executor.map(_encode_chunk, chunks))

    # Merge results in original walk order
    input_ids = []
    edge_split_masks = []
    edge_ids_list = []
    for ids, splits, eids in chunk_results:
        input_ids.extend(ids)
        edge_split_masks.extend(splits)
        edge_ids_list.extend(eids)

    return input_ids, edge_split_masks, edge_ids_list


def build_ragged_arrays(input_ids_list, split_masks_list, edge_ids_list, tokenizer):
    """Build CSR ragged arrays from per-walk numpy arrays.  No global padding.

    Dtype safety guards:
      offsets:        int32 if N+1 <= 2^31-1 AND T <= 2^31-1, else int64
      flat_input_ids: int16 if vocab_size <= 32767, int32 if <= 2^31-1, else int64
      flat_split_mask: always int8 (values are only -1..3)
      flat_edge_ids:  int32 if max_edge_id <= 2^31-1, else int64
    """
    N = len(input_ids_list)
    lengths = np.array([len(a) for a in input_ids_list], dtype=np.int64)
    T = int(lengths.sum())

    # offsets dtype: protect against huge T or N
    if (N + 1) > np.iinfo(np.int32).max or T > np.iinfo(np.int32).max:
        offsets_dtype = np.int64
    else:
        offsets_dtype = np.int32
    offsets_np = np.zeros(N + 1, dtype=offsets_dtype)
    np.cumsum(lengths, out=offsets_np[1:])

    # input_ids dtype
    vocab_size = tokenizer.vocab_size
    if vocab_size <= np.iinfo(np.int16).max:
        ids_dtype = np.int16
    elif vocab_size <= np.iinfo(np.int32).max:
        ids_dtype = np.int32
    else:
        ids_dtype = np.int64

    # edge_ids dtype
    max_edge_id = max(
        (int(a.max()) for a in edge_ids_list if len(a) > 0 and int(a.max()) >= 0),
        default=0,
    )
    edge_ids_dtype = np.int32 if max_edge_id <= np.iinfo(np.int32).max else np.int64

    flat_input_ids_np = np.empty(T, dtype=ids_dtype)
    flat_split_mask_np = np.empty(T, dtype=np.int8)
    flat_edge_ids_np = np.full(T, -1, dtype=edge_ids_dtype)

    for i in range(N):
        s = int(offsets_np[i])
        e = int(offsets_np[i + 1])
        flat_input_ids_np[s:e] = input_ids_list[i]
        flat_split_mask_np[s:e] = split_masks_list[i]
        flat_edge_ids_np[s:e] = edge_ids_list[i]

    print(
        f"  Ragged arrays: N={N}, T={T}, ids_dtype={ids_dtype.__name__}, offsets_dtype={offsets_dtype.__name__}"
    )
    return {
        "offsets": torch.from_numpy(offsets_np),
        "flat_input_ids": torch.from_numpy(flat_input_ids_np),
        "flat_split_mask": torch.from_numpy(flat_split_mask_np),
        "flat_edge_ids": torch.from_numpy(flat_edge_ids_np),
    }


def build_runtime_cache_data(
    tokenizer,
    offsets,
    flat_input_ids,
    flat_split_mask,
    flat_edge_ids,
    splits_dict,
    metadata,
):
    tokenizer_state = {
        "token2id": tokenizer.token2id,
        "id2token": tokenizer.id2token,
        "edge_label2id": tokenizer.edge_label2id,
        "id2edge_label": tokenizer.id2edge_label,
        "PAD_ID": tokenizer.PAD_ID,
        "MASK_ID": tokenizer.MASK_ID,
        "UNK_ID": tokenizer.UNK_ID,
        "UNK_LABEL_ID": tokenizer.UNK_LABEL_ID,
        "vocab_size": tokenizer.vocab_size,
        "num_edge_tokens": tokenizer.num_edge_tokens,
    }

    return {
        "version": "2.0",
        "tokenizer": tokenizer_state,
        "encoded": {
            "offsets": offsets,
            "flat_input_ids": flat_input_ids,
            "flat_split_mask": flat_split_mask,
            "flat_edge_ids": flat_edge_ids,
        },
        "splits": splits_dict,
        "metadata": metadata,
    }


def compute_class_weights_from_base(
    input_ids: torch.Tensor,
    edge_split_mask: torch.Tensor,
    tokenizer: "Tokenizer",
    num_classes: int,
    ignore_index: int,
) -> list:
    """Compute class weights from train-target (MASK) edges in base tensors.

    Avoids materializing any stage views — reads labels directly from the
    MASK positions in edge_split_mask.
    """
    # Build token_id -> class_id mapping
    id2class = torch.full((tokenizer.vocab_size,), ignore_index, dtype=torch.long)
    for class_id, edge_tok in tokenizer.id2edge_label.items():
        tok_id = tokenizer.token2id.get(edge_tok, None)
        if tok_id is not None:
            id2class[tok_id] = int(class_id)

    # Extract class labels at train-target (MASK) positions
    target_positions = edge_split_mask == SplitID.MASK
    if not target_positions.any():
        print("\u26a0\ufe0f  Warning: No MASK positions found, using uniform weights")
        return [1.0] * num_classes

    raw_labels = id2class[input_ids[target_positions].long()]
    valid_labels = raw_labels[raw_labels != ignore_index]

    if len(valid_labels) == 0:
        print(
            "\u26a0\ufe0f  Warning: No valid labels in train split, using uniform weights"
        )
        return [1.0] * num_classes

    # Inverse frequency formula
    class_counts = [(valid_labels == i).sum().item() for i in range(num_classes)]
    total = len(valid_labels)
    weights = [
        total / (num_classes * count) if count > 0 else 1.0 for count in class_counts
    ]
    weight_sum = sum(weights)
    weights = [w / weight_sum * num_classes for w in weights]
    return weights


def _dataloader_kwargs(cfg) -> dict:
    """Extract DataLoader construction kwargs from cfg. Single source of truth for both paths."""
    return dict(
        batch_size=int(cfg.training.batch_size),
        num_workers=int(getattr(cfg.training, "num_workers", 4)),
        pin_memory=bool(getattr(cfg.training, "pin_memory", True)),
        persistent_workers=bool(getattr(cfg.training, "persistent_workers", True)),
        prefetch_factor=int(getattr(cfg.training, "prefetch_factor", 2)),
        use_bucket_batching=bool(getattr(cfg.training, "bucket_batching", True)),
        bucket_width=int(getattr(cfg.training, "bucket_width", 16)),
        seed=int(get_seed(cfg)),
    )


def _keyed_cache_path(cfg):
    """Cache filename keyed by (walk_strategy, num_walks, max_walk_length, seed[, k]).

    Isolation guarantee: every distinct sampling config writes its OWN cache file,
    so a new sampler can never silently load or overwrite the legacy
    `dataset_cache.pt` that the current SOTA runs and all research Leads depend on.
    The legacy file is read-only here (reused for uniform runs whose budget matches;
    see prepare_data) and is never a SAVE target.
    """
    strategy = str(getattr(cfg.dataset, "walk_strategy", "uniform"))
    nw = int(cfg.dataset.num_walks)
    mw = int(cfg.dataset.max_walk_length)
    seed = int(get_seed(cfg))
    parts = [strategy, f"nw{nw}", f"mw{mw}", f"seed{seed}"]
    if "k_cover" in strategy:
        parts.insert(1, f"k{int(getattr(cfg.dataset, 'walk_k_min', 1))}")
    return os.path.join(cfg.dataset.data_dir, "dataset_cache__" + "_".join(parts) + ".pt")


def _cache_num_walks(cache_path, use_mmap=False):
    """Cheaply read the walk count (== len(offsets)-1) from a cache file."""
    cd = load_dataset_cache(cache_path, use_mmap=use_mmap)
    return int(cd["encoded"]["offsets"].shape[0] - 1)


def prepare_data(cfg):
    strategy = str(getattr(cfg.dataset, "walk_strategy", "uniform"))
    keyed_path = _keyed_cache_path(cfg)
    legacy_path = os.path.join(cfg.dataset.data_dir, "dataset_cache.pt")
    # SAVE always targets the keyed path (legacy dataset_cache.pt is never overwritten).
    dataset_cache_path = keyed_path

    # Warn when caller asked for cache but it doesn't exist yet — avoids silent rebuild.
    if cfg.preprocess.use_cache and not cache_exists(keyed_path) and not cache_exists(legacy_path):
        print(
            f"⚠️  use_cache=True but no cache found at {dataset_cache_path} "
            "\u2014 building from scratch."
        )

    use_mmap = bool(getattr(cfg.preprocess, "use_mmap", False))
    load_path = None
    if cfg.preprocess.use_cache:
        if cache_exists(keyed_path):
            load_path = keyed_path
        elif strategy == "uniform" and cache_exists(legacy_path):
            # Back-compat: reuse the legacy uniform cache only if its budget matches,
            # so existing SOTA caches are not needlessly rebuilt.
            legacy_nw = _cache_num_walks(legacy_path, use_mmap=use_mmap)
            if legacy_nw == int(cfg.dataset.num_walks):
                load_path = legacy_path
                print(f"Reusing legacy cache {legacy_path} (uniform, nw={legacy_nw} matches).")
            else:
                print(f"WARNING: legacy cache nw={legacy_nw} != config nw="
                      f"{int(cfg.dataset.num_walks)} -- building keyed cache instead.")

    if load_path is not None:
        print(f"Loading dataset cache from {load_path} (mmap={use_mmap})...")
        cache_data = load_dataset_cache(load_path, use_mmap=use_mmap)

        # Update config with metadata
        cfg.model.vocab_size = cache_data["metadata"]["vocab_size"]
        cfg.model.num_classes = cache_data["metadata"]["num_classes"]
        cfg.model.pad_id = cache_data["metadata"]["pad_id"]
        cfg.model.ignore_index = cache_data["metadata"]["ignore_index"]
        cfg.model.unk_id = int(cache_data["tokenizer"]["UNK_ID"])
        cfg.model.mask_id = int(cache_data["tokenizer"]["MASK_ID"])

        if "class_weights" in cache_data["metadata"]:
            cfg.model.class_weights = cache_data["metadata"]["class_weights"]
        else:
            # Old cache built before class-weight support: warn loudly so the user
            # knows the model will fall back to uniform weights inside LitEdgeClassifier.
            print(
                f"⚠️  WARNING: cache at {dataset_cache_path} has no class_weights in metadata.\n"
                "   The model will use uniform class weights, which may skew results\n"
                "   on imbalanced datasets. Delete the cache and re-run to rebuild it."
            )

        size_mb = os.path.getsize(dataset_cache_path) / (1024 * 1024)
        print(f"Success! ✅ (loaded {size_mb:.1f} MB)")

        return create_stage_dataloaders(
            cache_data,
            dynamic_train_masking=bool(
                getattr(cfg.model, "dynamic_train_masking", False)
            ),
            **_dataloader_kwargs(cfg),
        )
    # Profile data creation steps to help diagnose slow preprocessing
    timings = {}

    t0 = time.time()
    edges = get_edge_list(cfg)
    timings["get_edge_list"] = time.time() - t0

    t0 = time.time()
    train_set, mask_set, val_set, test_set = split_edges(cfg, edges)
    timings["split_edges"] = time.time() - t0

    t0 = time.time()
    walks = get_walks(
        cfg, edges,
        train_set=train_set, mask_set=mask_set,
        val_set=val_set, test_set=test_set,
    )
    timings["get_walks"] = time.time() - t0

    t0 = time.time()
    tokenizer = get_tokenizer(cfg, walks, edges)
    timings["get_tokenizer"] = time.time() - t0

    cfg.model.vocab_size = tokenizer.vocab_size
    cfg.model.num_classes = tokenizer.num_edge_tokens
    cfg.model.pad_id = tokenizer.PAD_ID
    cfg.model.ignore_index = tokenizer.UNK_LABEL_ID
    cfg.model.unk_id = tokenizer.UNK_ID
    cfg.model.mask_id = tokenizer.MASK_ID

    t0 = time.time()
    (
        input_lists,
        split_lists,
        edge_ids_list,
    ) = encode_walks(
        walks,
        tokenizer,
        edges,
        train_set,
        mask_set,
        val_set,
        test_set,
        num_workers=int(cfg.preprocess.num_workers),
    )
    timings["encode_walks"] = time.time() - t0

    t0 = time.time()
    ragged = build_ragged_arrays(input_lists, split_lists, edge_ids_list, tokenizer)
    del input_lists, split_lists, edge_ids_list  # free memory
    timings["build_ragged_arrays"] = time.time() - t0

    # Compute class weights directly from flat arrays.
    class_weights = compute_class_weights_from_base(
        ragged["flat_input_ids"],
        ragged["flat_split_mask"],
        tokenizer,
        cfg.model.num_classes,
        cfg.model.ignore_index,
    )
    cfg.model.class_weights = class_weights
    print(f"✓ Class weights computed from train split: {class_weights}")

    splits_dict = {
        "train": train_set,
        "mask": mask_set,
        "val": val_set,
        "test": test_set,
    }
    metadata = {
        "vocab_size": cfg.model.vocab_size,
        "num_classes": cfg.model.num_classes,
        "pad_id": cfg.model.pad_id,
        "ignore_index": cfg.model.ignore_index,
        "class_weights": class_weights,
        "dataset_name": cfg.dataset.name,
        "seed": get_seed(cfg),
    }
    cache_data = build_runtime_cache_data(
        tokenizer,
        ragged["offsets"],
        ragged["flat_input_ids"],
        ragged["flat_split_mask"],
        ragged["flat_edge_ids"],
        splits_dict,
        metadata,
    )

    if cfg.preprocess.save:
        print(f"Saving dataset cache to {dataset_cache_path}...")
        size_mb = save_dataset_cache(
            dataset_cache_path,
            tokenizer,
            ragged["offsets"],
            ragged["flat_input_ids"],
            ragged["flat_split_mask"],
            ragged["flat_edge_ids"],
            splits_dict,
            metadata,
        )
        print(f"✓ Dataset cache saved ({size_mb:.1f} MB)")

    try:
        print("Data creation profiling (seconds):")
        for k, v in timings.items():
            print(f"  {k}: {v:.2f}s")
    except Exception:
        pass

    return create_stage_dataloaders(
        cache_data,
        dynamic_train_masking=bool(getattr(cfg.model, "dynamic_train_masking", False)),
        **_dataloader_kwargs(cfg),
    )
