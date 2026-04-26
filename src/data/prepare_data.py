import os
import time
import torch
import numpy as np
from enum import IntEnum
from torch.nn.utils.rnn import pad_sequence
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


def get_walks(cfg, edges):
    print(f"Sampling random walks from {cfg.dataset.name} dataset...")

    walk_workers = int(getattr(cfg.preprocess, "num_workers", 1))

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


# ---------------------------------------------------------------------------
# Module-level globals used by multiprocess worker processes.
# Set once via _worker_init(); never accessed in the main process.
# ---------------------------------------------------------------------------
_w_token2id        = None
_w_edge_tokens     = None
_w_node_id_lookup  = None
_w_edge_label_lookup = None
_w_unk_id          = None
_w_split_lookup    = None
_w_edge_to_id      = None
_w_bad_val         = None


def _worker_init(token2id, edge_tokens, node_id_lookup, edge_label_lookup,
                 unk_id, split_lookup, edge_to_id, bad_val):
    """ProcessPoolExecutor initializer: sets read-only globals once per worker."""
    global _w_token2id, _w_edge_tokens, _w_node_id_lookup, _w_edge_label_lookup
    global _w_unk_id, _w_split_lookup, _w_edge_to_id, _w_bad_val
    _w_token2id          = token2id
    _w_edge_tokens       = edge_tokens
    _w_node_id_lookup    = node_id_lookup
    _w_edge_label_lookup = edge_label_lookup
    _w_unk_id            = unk_id
    _w_split_lookup      = split_lookup
    _w_edge_to_id        = edge_to_id
    _w_bad_val           = bad_val


def _encode_chunk(walks_chunk):
    """Encode a slice of walks using module-level worker globals.

    Called both from worker processes (via ProcessPoolExecutor) and directly
    from the main process when num_workers <= 1.
    """
    token2id_get      = _w_token2id.get
    edge_tokens_set   = _w_edge_tokens
    node_id_lookup    = _w_node_id_lookup
    edge_label_lookup = _w_edge_label_lookup
    unk_id            = _w_unk_id
    split_lookup_get  = _w_split_lookup.get
    edge_to_id        = _w_edge_to_id
    BAD               = _w_bad_val

    input_ids      = []
    edge_split_masks = []
    edge_ids_list  = []

    for walk in walks_chunk:
        walk_len = len(walk)
        x          = [0]   * walk_len
        split_mask = [BAD] * walk_len
        edge_ids   = [-1]  * walk_len

        for i in range(walk_len):
            token = walk[i]
            x[i] = token2id_get(token, unk_id)
            if token in edge_tokens_set:
                u     = node_id_lookup.get(walk[i - 1]) if i > 0 else None
                v     = node_id_lookup.get(walk[i + 1]) if i < walk_len - 1 else None
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
    positions / walk_ids / walk_lengths are trivially reconstructable after padding and
    are therefore NOT computed here — see pad_and_build_stage_tensors.

    When num_workers > 1, walks are split into equal chunks and encoded in parallel
    using ProcessPoolExecutor.  Shared read-only state (tokenizer dicts, split lookup,
    edge→id map) is passed once via the pool initializer to avoid per-task pickling.
    """
    # Pre-build split lookup (O(1) per edge, built once here in the main process)
    split_lookup = {}
    split_lookup.update({t: SplitID.TEST  for t in test_set})
    split_lookup.update({t: SplitID.VAL   for t in val_set})
    split_lookup.update({t: SplitID.MASK  for t in mask_set})
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
    input_ids      = []
    edge_split_masks = []
    edge_ids_list  = []
    for ids, splits, eids in chunk_results:
        input_ids.extend(ids)
        edge_split_masks.extend(splits)
        edge_ids_list.extend(eids)

    return input_ids, edge_split_masks, edge_ids_list


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
    tokenizer,
):
    """Pad variable-length numpy arrays and reconstruct all base tensors.

    Opt 2a: positions / walk_ids / walk_lengths are reconstructed vectorially from
            the padding mask — no per-walk allocation or pad_sequence call needed.
    Opt 2b: inputs are numpy int64 arrays; numpy fill + torch.from_numpy is ~10x
            faster than pad_sequence on lists of torch tensors.
    """
    print(f"Padding and building stage tensors for {cfg.dataset.name} dataset...")
    pad_id = int(tokenizer.PAD_ID)
    N      = len(input_ids_list)
    max_len = max(len(a) for a in input_ids_list)

    def _np_pad(arrays: list, pad_val: int) -> torch.Tensor:
        """Fill a pre-allocated numpy array and return as a contiguous LongTensor."""
        out = np.full((N, max_len), pad_val, dtype=np.int64)
        for i, a in enumerate(arrays):
            out[i, :len(a)] = a
        return torch.from_numpy(out)

    input_ids       = _np_pad(input_ids_list,        pad_id)
    edge_split_mask = _np_pad(edge_split_masks_list, int(SplitID.BAD))
    edge_ids        = _np_pad(edge_ids_list,         -1)
    attention_base  = (input_ids != pad_id).long()

    # Reconstruct trivially-computable tensors from the padding mask.
    # real_mask[w, i] is True iff position i in walk w is a real (non-padded) token.
    real_mask   = attention_base.bool()                                          # [N, max_len]

    positions   = torch.arange(max_len, dtype=torch.long)                       # [max_len]
    positions   = positions.unsqueeze(0).expand(N, -1).clone()                  # [N, max_len]
    positions[~real_mask] = -1

    walk_lengths = real_mask.long().sum(1, keepdim=True).expand(N, max_len).clone()  # [N, max_len]
    walk_lengths[~real_mask] = -1

    walk_ids    = torch.arange(N, dtype=torch.long)                             # [N]
    walk_ids    = walk_ids.unsqueeze(1).expand(N, max_len).clone()              # [N, max_len]
    walk_ids[~real_mask] = -1

    print(f"Success! ✅")
    return {
        "input_ids":       input_ids,
        "edge_split_mask": edge_split_mask,
        "attention_base":  attention_base,
        "edge_ids":        edge_ids,
        "walk_ids":        walk_ids,
        "positions":       positions,
        "walk_lengths":    walk_lengths,
    }


def build_runtime_cache_data(
    tokenizer,
    input_ids,
    edge_split_mask,
    attention_base,
    splits_dict,
    metadata,
    edge_ids,
    walk_ids,
    positions,
    walk_lengths,
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
        "version": "1.2",
        "tokenizer": tokenizer_state,
        "encoded": {
            "input_ids": input_ids,
            "edge_split_mask": edge_split_mask,
            "attention_base": attention_base,
            "edge_ids": edge_ids,
            "walk_ids": walk_ids,
            "positions": positions,
            "walk_lengths": walk_lengths,
        },
        "splits": splits_dict,
        "metadata": metadata,
    }


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


def compute_class_weights_from_base(
    input_ids: torch.Tensor,
    edge_split_mask: torch.Tensor,
    tokenizer: "Tokenizer",
    num_classes: int,
    ignore_index: int,
) -> list:
    """Compute class weights from train-target (MASK) edges in base tensors.

    Avoids materializing any stage views — reads labels directly from the
    MASK positions in edge_split_mask.  Drop-in replacement for the old
    compute_class_weights_from_train(train_pack, ...) flow.
    """
    # Build token_id -> class_id mapping (same logic as _stage_views_from_base)
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

    raw_labels = id2class[input_ids[target_positions]]
    valid_labels = raw_labels[raw_labels != ignore_index]

    if len(valid_labels) == 0:
        print("\u26a0\ufe0f  Warning: No valid labels in train split, using uniform weights")
        return [1.0] * num_classes

    # Inverse frequency (identical formula to compute_class_weights_from_train)
    class_counts = [(valid_labels == i).sum().item() for i in range(num_classes)]
    total = len(valid_labels)
    weights = [
        total / (num_classes * count) if count > 0 else 1.0
        for count in class_counts
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
    )


def prepare_data(cfg):
    dataset_cache_path = os.path.join(cfg.dataset.data_dir, "dataset_cache.pt")

    # Warn when caller asked for cache but it doesn't exist yet — avoids silent rebuild.
    if cfg.preprocess.use_cache and not cache_exists(dataset_cache_path):
        print(
            f"⚠️  use_cache=True but no cache found at {dataset_cache_path} "
            "\u2014 building from scratch."
        )

    if cfg.preprocess.use_cache and cache_exists(dataset_cache_path):
        use_mmap = bool(getattr(cfg.preprocess, "use_mmap", False))
        print(f"Loading dataset cache from {dataset_cache_path} (mmap={use_mmap})...")
        cache_data = load_dataset_cache(dataset_cache_path, use_mmap=use_mmap)

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
            dynamic_train_masking=bool(getattr(cfg.model, "dynamic_train_masking", False)),
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
    walks = get_walks(cfg, edges)
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
        walks, tokenizer, edges, train_set, mask_set, val_set, test_set,
        num_workers=int(cfg.preprocess.num_workers),
    )
    timings["encode_walks"] = time.time() - t0

    # Free the walks list — it is no longer needed and holds ~1-2 GB at real scale.
    del walks

    t0 = time.time()
    base_tensors = pad_and_build_stage_tensors(
        cfg,
        input_lists,
        split_lists,
        edge_ids_list,
        tokenizer,
    )
    timings["pad_and_build_stage_tensors"] = time.time() - t0

    input_ids       = base_tensors["input_ids"]
    edge_split_mask = base_tensors["edge_split_mask"]
    attention_base  = base_tensors["attention_base"]
    edge_ids        = base_tensors["edge_ids"]
    walk_ids        = base_tensors["walk_ids"]
    positions       = base_tensors["positions"]
    walk_lengths    = base_tensors["walk_lengths"]

    # Compute class weights directly from base tensors — no stage view materialization.
    class_weights = compute_class_weights_from_base(
        input_ids, edge_split_mask, tokenizer, cfg.model.num_classes, cfg.model.ignore_index
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
        input_ids,
        edge_split_mask,
        attention_base,
        splits_dict,
        metadata,
        edge_ids,
        walk_ids,
        positions,
        walk_lengths,
    )

    # Save dataset cache if requested.
    if cfg.preprocess.save:
        print(f"Saving dataset cache to {dataset_cache_path}...")

        size_mb = save_dataset_cache(
            dataset_cache_path,
            tokenizer,
            input_ids,
            edge_split_mask,
            attention_base,
            splits_dict,
            metadata,
            edge_ids=edge_ids,
        )
        print(f"✓ Dataset cache saved ({size_mb:.1f} MB)")

    # Print a short profile summary for debugging
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
