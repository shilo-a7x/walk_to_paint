import os
import json
import random
import torch
from enum import IntEnum
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.data.datasets import get_loader
from src.data.tokenizer import Tokenizer
from src.data.walk_sampler import sample_random_walks


class SplitID(IntEnum):
    TRAIN = 0  # context in train
    MASK = 1  # train targets (10%)
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
    split_path = os.path.join(cfg.dataset.data_dir, cfg.dataset.edge_split_file)
    if cfg.preprocess.use_cache and os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
    else:
        edges_copy = list(edges)
        random.shuffle(edges_copy)
        n_total = len(edges_copy)
        n_train = int(cfg.dataset.train_ratio * n_total)
        n_mask = int(cfg.dataset.mask_ratio * n_total)
        n_val = int(cfg.dataset.val_ratio * n_total)

        n_test = n_total - n_train - n_mask - n_val
        if n_test < 0:
            raise ValueError("Ratios sum to > 1. Adjust train/mask/val ratios.")

        split = {
            "train": edges_copy[:n_train],
            "mask": edges_copy[n_train : n_train + n_mask],
            "val": edges_copy[n_train + n_mask : n_train + n_mask + n_val],
            "test": edges_copy[n_train + n_mask + n_val :],
        }
        if cfg.preprocess.save:
            with open(split_path, "w") as f:
                json.dump(split, f)

    # lookup sets for fast membership
    train_set = {tuple(t) for t in split["train"]}
    mask_set = {tuple(t) for t in split["mask"]}
    val_set = {tuple(t) for t in split["val"]}
    test_set = {tuple(t) for t in split["test"]}
    print(f"Success! ✅")
    return train_set, mask_set, val_set, test_set


def get_walks(cfg, edges):
    print(f"Sampling random walks from {cfg.dataset.name} dataset...")
    walks_path = os.path.join(cfg.dataset.data_dir, cfg.dataset.walks_file)
    if cfg.preprocess.use_cache and os.path.exists(walks_path):
        with open(walks_path) as f:
            walks = json.load(f)
            print(f"Success! ✅")
            return walks
    walks = sample_random_walks(
        edges,
        num_walks=int(cfg.dataset.num_walks),
        max_walk_length=cfg.dataset.max_walk_length,
    )
    if cfg.preprocess.save:
        with open(walks_path, "w") as f:
            json.dump(walks, f)
    print(f"Success! ✅")
    return walks


def get_tokenizer(cfg, walks, edges):
    print(f"Building tokenizer for {cfg.dataset.name} dataset...")
    tokenizer_path = os.path.join(cfg.dataset.data_dir, cfg.dataset.tokenizer_file)
    if cfg.preprocess.use_cache and os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.load(tokenizer_path)
        print(f"Success! ✅")
        return tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit(walks, edges=edges)
    if cfg.preprocess.save:
        tokenizer.save(tokenizer_path)
    print(f"Success! ✅")
    return tokenizer


def encode_walks(
    cfg, walks, tokenizer: Tokenizer, train_set, mask_set, val_set, test_set
):
    input_ids, edge_split_masks = [], []

    split_lookup = {}
    split_lookup.update({t: SplitID.TEST for t in test_set})
    split_lookup.update({t: SplitID.VAL for t in val_set})
    split_lookup.update({t: SplitID.MASK for t in mask_set})
    split_lookup.update({t: SplitID.TRAIN for t in train_set})
    for walk in walks:
        x, split_mask = [], []
        for i, token in enumerate(walk):
            if tokenizer.is_edge(token):
                u = tokenizer.parse_node(walk[i - 1]) if i > 0 else None
                v = tokenizer.parse_node(walk[i + 1]) if i < len(walk) - 1 else None
                label = tokenizer.parse_edge_label(token)
                t = (u, v, label)
                split = split_lookup.get(t, SplitID.BAD)
                x.append(tokenizer.encode(token)[0])
                split_mask.append(split)
            else:
                x.append(tokenizer.encode(token)[0])
                split_mask.append(SplitID.BAD)
        input_ids.append(torch.tensor(x, dtype=torch.long))
        edge_split_masks.append(torch.tensor(split_mask, dtype=torch.long))

    return input_ids, edge_split_masks


def _stage_views_from_base(
    cfg, input_ids: torch.Tensor, edge_split_mask: torch.Tensor, tokenizer: Tokenizer
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

        # input ids: mask only target edges; pad disallowed edges
        x = input_ids.clone()
        target_mask = edge_split_mask == target_split
        # labels from original token ids (before overwrite)
        labels = torch.full_like(input_ids, ignore_index)
        if target_mask.any():
            labels[target_mask] = id2class[x[target_mask]]
        # replace target edges with MASK token id
        x[target_mask] = mask_id
        # hide disallowed edges completely
        x[disallowed_edges] = pad_id

        return x, labels, attn

    # Stage definitions
    train_allowed = [SplitID.TRAIN]
    val_allowed = [SplitID.TRAIN, SplitID.MASK]
    test_allowed = [SplitID.TRAIN, SplitID.MASK, SplitID.VAL]

    train_x, train_y, train_attn = build_for_stage(train_allowed, SplitID.MASK)
    val_x, val_y, val_attn = build_for_stage(val_allowed, SplitID.VAL)
    test_x, test_y, test_attn = build_for_stage(test_allowed, SplitID.TEST)

    return (
        (train_x, train_y, train_attn),
        (val_x, val_y, val_attn),
        (test_x, test_y, test_attn),
    )


def pad_and_build_stage_tensors(cfg, input_ids_list, edge_split_masks_list, tokenizer):
    print(f"Padding and building stage tensors for {cfg.dataset.name} dataset...")
    enc_path = os.path.join(cfg.dataset.data_dir, cfg.dataset.encoded_file)
    meta_path = os.path.join(cfg.dataset.data_dir, cfg.dataset.meta_file)
    if cfg.preprocess.use_cache and os.path.exists(enc_path):
        loaded = torch.load(enc_path)
        print(f"Success! ✅")
        return loaded
    pad_id = int(tokenizer.PAD_ID)
    # pad base
    input_ids = pad_sequence(
        input_ids_list, batch_first=True, padding_value=pad_id
    ).long()
    edge_split_mask = pad_sequence(
        edge_split_masks_list, batch_first=True, padding_value=SplitID.BAD
    ).long()
    # derive stage-specific views
    train_pack, val_pack, test_pack = _stage_views_from_base(
        cfg, input_ids, edge_split_mask, tokenizer
    )
    if cfg.preprocess.save:
        torch.save((train_pack, val_pack, test_pack), enc_path)
        meta = {
            "vocab_size": tokenizer.vocab_size,
            "num_classes": tokenizer.num_edge_tokens,
            "pad_id": pad_id,
            "ignore_index": tokenizer.UNK_LABEL_ID,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)
    print(f"Success! ✅")
    return (train_pack, val_pack, test_pack)


def make_dataloaders(cfg, train_pack, val_pack, test_pack):
    print(f"Creating DataLoaders for {cfg.dataset.name} dataset...")
    batch_size = int(cfg.training.batch_size)

    train_ds = TensorDataset(*train_pack)
    val_ds = TensorDataset(*val_pack)
    test_ds = TensorDataset(*test_pack)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Success! ✅")
    return {"train": train_loader, "val": val_loader, "test": test_loader}


def prepare_data(cfg):
    enc_path = os.path.join(cfg.dataset.data_dir, cfg.dataset.encoded_file)
    meta_path = os.path.join(cfg.dataset.data_dir, cfg.dataset.meta_file)
    if (
        cfg.preprocess.use_cache
        and os.path.exists(enc_path)
        and os.path.exists(meta_path)
    ):
        print(f"Loading preprocessed data from {enc_path}...")
        loaded = torch.load(enc_path)
        train_pack, val_pack, test_pack = loaded
        with open(meta_path, "r") as f:
            meta = json.load(f)
        cfg.model.vocab_size = meta["vocab_size"]
        cfg.model.num_classes = meta["num_classes"]
        cfg.model.pad_id = meta["pad_id"]
        cfg.model.ignore_index = meta["ignore_index"]
        print(f"Success! ✅")
        return make_dataloaders(cfg, train_pack, val_pack, test_pack)
    edges = get_edge_list(cfg)
    train_set, mask_set, val_set, test_set = split_edges(cfg, edges)
    walks = get_walks(cfg, edges)
    tokenizer = get_tokenizer(cfg, walks, edges)
    cfg.model.vocab_size = tokenizer.vocab_size
    cfg.model.num_classes = tokenizer.num_edge_tokens
    cfg.model.pad_id = tokenizer.PAD_ID
    cfg.model.ignore_index = tokenizer.UNK_LABEL_ID
    input_lists, split_lists = encode_walks(
        cfg, walks, tokenizer, train_set, mask_set, val_set, test_set
    )
    train_pack, val_pack, test_pack = pad_and_build_stage_tensors(
        cfg, input_lists, split_lists, tokenizer
    )
    return make_dataloaders(cfg, train_pack, val_pack, test_pack)
