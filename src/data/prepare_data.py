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
    TRAIN = 0
    VAL = 1
    TEST = 2
    BAD = -1


def get_edge_list(cfg):
    print(f"Loading {cfg.dataset.name} dataset...")
    return get_loader(cfg.dataset.name)(cfg)  # list of (u, v, label)


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
        n_val = int(cfg.dataset.val_ratio * n_total)

        split = {
            "train": edges_copy[:n_train],
            "val": edges_copy[n_train : n_train + n_val],
            "test": edges_copy[n_train + n_val :],
        }
        if cfg.preprocess.save:
            with open(split_path, "w") as f:
                json.dump(split, f)

    # lookup sets for fast membership
    train_set = {tuple(t) for t in split["train"]}
    val_set = {tuple(t) for t in split["val"]}
    test_set = {tuple(t) for t in split["test"]}
    return train_set, val_set, test_set


def get_walks(cfg, edges):
    print(f"Sampling random walks from {cfg.dataset.name} dataset...")
    walks_path = os.path.join(cfg.dataset.data_dir, cfg.dataset.walks_file)
    if cfg.preprocess.use_cache and os.path.exists(walks_path):
        with open(walks_path) as f:
            return json.load(f)
    walks = sample_random_walks(
        edges,
        num_walks=int(cfg.dataset.num_walks),
        max_walk_length=cfg.dataset.max_walk_length,
    )
    if cfg.preprocess.save:
        with open(walks_path, "w") as f:
            json.dump(walks, f)
    return walks


def get_tokenizer(cfg, walks, edges):
    print(f"Building tokenizer for {cfg.dataset.name} dataset...")
    tokenizer_path = os.path.join(cfg.dataset.data_dir, cfg.dataset.tokenizer_file)
    if cfg.preprocess.use_cache and os.path.exists(tokenizer_path):
        return Tokenizer.load(tokenizer_path)
    tokenizer = Tokenizer()
    tokenizer.fit(walks, edges=edges)
    if cfg.preprocess.save:
        tokenizer.save(tokenizer_path)
    return tokenizer


def encode_walks(cfg, walks, tokenizer: Tokenizer, train_set, val_set, test_set):
    print(f"Encoding walks for {cfg.dataset.name} dataset...")
    input_ids, labels, edge_split_masks = [], [], []
    ignore_index = cfg.dataset.ignore_index
    p_mask = cfg.dataset.mask_prob
    split_lookup = {}
    split_lookup.update({t: SplitID.TEST for t in test_set})
    split_lookup.update({t: SplitID.VAL for t in val_set})
    split_lookup.update({t: SplitID.TRAIN for t in train_set})
    for walk in walks:
        x, y, split_mask = [], [], []
        for i, token in enumerate(walk):
            if tokenizer.is_edge(token):
                u = tokenizer.parse_node(walk[i - 1]) if i > 0 else None
                v = tokenizer.parse_node(walk[i + 1]) if i < len(walk) - 1 else None
                label = tokenizer.parse_edge_label(token)
                t = (u, v, label)
                split = split_lookup.get(t, SplitID.BAD)
                should_mask = random.random() < p_mask
                x.append(
                    tokenizer.MASK_ID if should_mask else tokenizer.encode(token)[0]
                )
                y.append(
                    tokenizer.encode_edge_label(token) if should_mask else ignore_index
                )
                split_mask.append(split)
            else:
                x.append(tokenizer.encode(token)[0])
                y.append(ignore_index)
                split_mask.append(SplitID.BAD)
        input_ids.append(torch.tensor(x, dtype=torch.long))
        labels.append(torch.tensor(y, dtype=torch.long))
        edge_split_masks.append(torch.tensor(split_mask, dtype=torch.long))
    return input_ids, labels, edge_split_masks


def pad_and_split_labels(cfg, input_ids, labels, edge_split_masks, tokenizer):
    enc_path = os.path.join(cfg.dataset.data_dir, cfg.dataset.encoded_file)
    meta_path = os.path.join(cfg.dataset.data_dir, cfg.dataset.meta_file)
    if cfg.preprocess.use_cache and os.path.exists(enc_path):
        return torch.load(enc_path)
    ignore_index = cfg.dataset.ignore_index
    pad_id = tokenizer.PAD_ID
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=ignore_index)
    edge_split_mask = pad_sequence(
        edge_split_masks, batch_first=True, padding_value=SplitID.BAD
    ).long()
    train_labels = labels.clone()
    val_labels = labels.clone()
    test_labels = labels.clone()
    train_labels[edge_split_mask != SplitID.TRAIN] = ignore_index
    val_labels[edge_split_mask != SplitID.VAL] = ignore_index
    test_labels[edge_split_mask != SplitID.TEST] = ignore_index
    attention_mask = (input_ids != pad_id).long()
    if cfg.preprocess.save:
        torch.save(
            (input_ids, train_labels, val_labels, test_labels, attention_mask), enc_path
        )
        meta = {
            "vocab_size": tokenizer.vocab_size,
            "num_classes": tokenizer.num_edge_tokens,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)
    return input_ids, train_labels, val_labels, test_labels, attention_mask


def make_dataloaders(
    cfg, input_ids, train_labels, val_labels, test_labels, attention_mask
):
    train_ds = TensorDataset(input_ids, train_labels, attention_mask)
    val_ds = TensorDataset(input_ids, val_labels, attention_mask)
    test_ds = TensorDataset(input_ids, test_labels, attention_mask)
    batch_size = cfg.training.batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def prepare_data(cfg):
    enc_path = os.path.join(cfg.dataset.data_dir, cfg.dataset.encoded_file)
    meta_path = os.path.join(cfg.dataset.data_dir, cfg.dataset.meta_file)
    if (
        cfg.preprocess.use_cache
        and os.path.exists(enc_path)
        and os.path.exists(meta_path)
    ):
        input_ids, train_labels, val_labels, test_labels, attention_mask = torch.load(
            enc_path
        )
        with open(meta_path, "r") as f:
            meta = json.load(f)
        cfg.model.vocab_size = meta["vocab_size"]
        cfg.model.num_classes = meta["num_classes"]
        return make_dataloaders(
            cfg,
            input_ids,
            train_labels,
            val_labels,
            test_labels,
            attention_mask,
        )
    edges = get_edge_list(cfg)
    train_set, val_set, test_set = split_edges(cfg, edges)
    walks = get_walks(cfg, edges)
    tokenizer = get_tokenizer(cfg, walks, edges)
    cfg.model.vocab_size = tokenizer.vocab_size
    cfg.model.num_classes = tokenizer.num_edge_tokens
    cfg.dataset.num_edge_labels = tokenizer.num_edge_tokens
    encoded, labels, splits = encode_walks(
        cfg, walks, tokenizer, train_set, val_set, test_set
    )
    (
        input_ids,
        train_labels,
        val_labels,
        test_labels,
        attention_mask,
    ) = pad_and_split_labels(cfg, encoded, labels, splits, tokenizer)
    dataloaders = make_dataloaders(
        cfg,
        input_ids,
        train_labels,
        val_labels,
        test_labels,
        attention_mask,
    )
    return dataloaders
