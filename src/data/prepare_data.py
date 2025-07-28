import os
import json
import random
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence

from src.data.datasets import get_loader
from src.data.tokenizer import Tokenizer
from src.data.walk_sampler import sample_random_walks


def get_edge_list(cfg):
    return get_loader(cfg.dataset.name)(cfg)


def get_walks(cfg, edges):
    if cfg.preprocess.use_cache and os.path.exists(cfg.dataset.walks_file):
        with open(cfg.dataset.walks_file) as f:
            return json.load(f)
    walks = sample_random_walks(
        edges,
        num_walks=cfg.dataset.num_walks,
        max_walk_length=cfg.dataset.max_walk_length,
    )
    if cfg.preprocess.save:
        path = os.path.join(cfg.dataset.data_dir, cfg.dataset.walks_file)
        with open(path, "w") as f:
            json.dump(walks, f)
    return walks


def get_tokenizer(cfg, walks, edges):
    if cfg.preprocess.use_cache and os.path.exists(cfg.dataset.tokenizer_file):
        return Tokenizer.load(cfg.dataset.tokenizer_file)
    tokenizer = Tokenizer()
    tokenizer.fit(walks, edges=edges)
    if cfg.preprocess.save:
        path = os.path.join(cfg.dataset.data_dir, cfg.dataset.tokenizer_file)
        tokenizer.save(path)
    return tokenizer


def encode_walks(cfg, walks, tokenizer):
    input_ids, labels = [], []
    for walk in walks:
        x, y = [], []
        for token in tokenizer.encode(walk):
            if tokenizer.is_edge(token) and random.random() < cfg.dataset.mask_prob:
                x.append(tokenizer.MASK_ID)
                y.append(tokenizer.encode_edge_label(token))
            else:
                x.append(token)
                y.append(cfg.dataset.ignore_index)
        input_ids.append(torch.tensor(x, dtype=torch.long))
        labels.append(torch.tensor(y, dtype=torch.long))
    return input_ids, labels


def pad_and_save_encoded(cfg, input_ids, labels, tokenizer):
    if cfg.preprocess.use_cache and os.path.exists(cfg.dataset.encoded_file):
        return torch.load(cfg.dataset.encoded_file)
    pad_id = tokenizer.PAD_ID
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    labels = pad_sequence(
        labels, batch_first=True, padding_value=cfg.dataset.ignore_index
    )
    attention_mask = (input_ids != pad_id).long()
    path = os.path.join(cfg.dataset.data_dir, cfg.dataset.encoded_file)
    if cfg.preprocess.save:
        torch.save((input_ids, labels, attention_mask), path)
    return input_ids, labels, attention_mask


def get_splits(cfg, dataset_size):
    if cfg.preprocess.use_cache and os.path.exists(cfg.dataset.splits_file):
        with open(cfg.dataset.splits_file) as f:
            return json.load(f)

    indices = list(range(dataset_size))
    random.shuffle(indices)
    n_train = int(cfg.dataset.train_ratio * dataset_size)
    n_val = int(cfg.dataset.val_ratio * dataset_size)

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

    if cfg.preprocess.save:
        path = os.path.join(cfg.dataset.data_dir, cfg.dataset.splits_file)
        with open(path, "w") as f:
            json.dump(splits, f)
    return splits


def make_dataloaders(cfg, input_ids, labels, attention_mask, splits):
    dataset = TensorDataset(input_ids, labels, attention_mask)

    def make_loader(indices, shuffle):
        subset = Subset(dataset, indices)
        return DataLoader(
            subset,
            batch_size=cfg.training.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    return {
        "train": make_loader(splits["train"], shuffle=True),
        "val": make_loader(splits["val"], shuffle=False),
        "test": make_loader(splits["test"], shuffle=False),
    }


def prepare_data(cfg):
    edges = get_edge_list(cfg)
    walks = get_walks(cfg, edges)
    tokenizer = get_tokenizer(cfg, walks, edges)
    cfg.model.vocab_size = tokenizer.vocab_size
    cfg.model.num_classes = tokenizer.num_edge_tokens
    cfg.dataset.num_edge_labels = tokenizer.num_edge_tokens
    encoded, labels = encode_walks(cfg, walks, tokenizer)
    input_ids, labels, attention_mask = pad_and_save_encoded(
        cfg, encoded, labels, tokenizer
    )
    splits = get_splits(cfg, len(input_ids))
    dataloaders = make_dataloaders(cfg, input_ids, labels, attention_mask, splits)
    return dataloaders
