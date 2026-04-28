"""
Stage-view dataset: ragged (CSR) v2.0 + padded v1.x backward compat.

Ragged mode: walks stored as flat arrays + offsets. __getitem__ slices by walk.
             Collate pads to batch-max length (not global max).
             BucketBatchSampler groups walks by length for fewer wasted padding tokens.

Padded mode: existing v1.x behavior unchanged.
"""

import math
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from enum import IntEnum


class SplitID(IntEnum):
    TRAIN = 0
    MASK = 1
    VAL = 2
    TEST = 3
    BAD = -1


class StageViewDataset(Dataset):
    """Stage-specific view over a cached dataset. Supports ragged (v2.0) and padded (v1.x)."""

    def __init__(self, cache_data: dict, stage: str = "train", dynamic_train_masking: bool = False):
        enc = cache_data["encoded"]
        self._ragged = "offsets" in enc

        tokenizer = cache_data["tokenizer"]
        self.mask_id = tokenizer["MASK_ID"]
        self.ignore_index = tokenizer["UNK_LABEL_ID"]
        self.dynamic_train_masking = bool(dynamic_train_masking)
        self.stage = stage

        self.id2class = torch.full(
            (tokenizer["vocab_size"],), self.ignore_index, dtype=torch.long
        )
        for class_id, edge_tok in tokenizer["id2edge_label"].items():
            tok_id = tokenizer["token2id"].get(edge_tok)
            if tok_id is not None:
                self.id2class[tok_id] = int(class_id)

        if stage == "train":
            self.allowed_splits = [SplitID.TRAIN, SplitID.MASK]
            self.target_split = SplitID.MASK
        elif stage == "val":
            self.allowed_splits = [SplitID.TRAIN, SplitID.MASK, SplitID.VAL]
            self.target_split = SplitID.VAL
        elif stage == "test":
            self.allowed_splits = [SplitID.TRAIN, SplitID.MASK, SplitID.VAL, SplitID.TEST]
            self.target_split = SplitID.TEST
        else:
            raise ValueError("Stage must be 'train', 'val', or 'test'")

        if self._ragged:
            self.offsets = enc["offsets"]                      # int32 or int64 [N+1]
            self.flat_input_ids = enc["flat_input_ids"]        # int16/int32/int64 [T]
            self.flat_split_mask = enc["flat_split_mask"]      # int8 [T]
            self.flat_edge_ids = enc["flat_edge_ids"]          # int32/int64 [T]
            self._N = len(self.offsets) - 1
            # lengths[i] = real token count for walk i — used by BucketBatchSampler
            self.lengths = (self.offsets[1:] - self.offsets[:-1]).to(torch.long)
        else:
            # Padded mode (v1.x) — original code unchanged
            self.input_ids = enc["input_ids"]
            self.edge_split_mask = enc["edge_split_mask"]
            self.attention_base = enc["attention_base"]
            self.edge_ids = enc.get("edge_ids")

            N, seq_len = self.attention_base.shape
            if enc.get("walk_ids") is not None:
                self.walk_ids = enc["walk_ids"]
            else:
                self.walk_ids = torch.arange(N, dtype=torch.long).unsqueeze(1).expand(N, seq_len)
            if enc.get("positions") is not None:
                self.positions = enc["positions"]
            else:
                self.positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(N, seq_len)
            if enc.get("walk_lengths") is not None:
                self.walk_lengths = enc["walk_lengths"]
            else:
                lengths = self.attention_base.sum(dim=1, dtype=torch.long)
                self.walk_lengths = lengths.unsqueeze(1).expand(N, seq_len)

    def __len__(self):
        if self._ragged:
            return self._N
        return len(self.input_ids)

    # ------------------------------------------------------------------ #
    # Compatibility properties for code that accesses padded-mode attrs   #
    # (e.g. LitEdgeClassifier._build_dynamic_train_pool).                 #
    # In ragged mode, expose the flat (1-D) arrays under the old names.   #
    # ------------------------------------------------------------------ #
    @property
    def input_ids(self):
        if self._ragged:
            return self.flat_input_ids
        return self.__dict__["input_ids"]

    @input_ids.setter
    def input_ids(self, value):
        self.__dict__["input_ids"] = value

    @property
    def edge_split_mask(self):
        if self._ragged:
            return self.flat_split_mask
        return self.__dict__["edge_split_mask"]

    @edge_split_mask.setter
    def edge_split_mask(self, value):
        self.__dict__["edge_split_mask"] = value

    @property
    def edge_ids(self):
        if self._ragged:
            return self.flat_edge_ids
        return self.__dict__.get("edge_ids")

    @edge_ids.setter
    def edge_ids(self, value):
        self.__dict__["edge_ids"] = value

    def __getitem__(self, idx):
        if self._ragged:
            return self._getitem_ragged(idx)
        return self._getitem_padded(idx)

    def _getitem_ragged(self, idx):
        s = int(self.offsets[idx])
        e = int(self.offsets[idx + 1])
        L = e - s

        # Cast to long: dtype change always creates a new tensor (no aliasing into flat storage)
        input_ids = self.flat_input_ids[s:e].to(torch.long)
        split_mask = self.flat_split_mask[s:e].to(torch.long)
        attention_mask = torch.ones(L, dtype=torch.long)

        is_edge = split_mask != int(SplitID.BAD)
        target_edges = split_mask == int(self.target_split)

        allowed_edges = torch.zeros(L, dtype=torch.bool)
        for split_id in self.allowed_splits:
            allowed_edges |= (split_mask == int(split_id))
        disallowed_edges = is_edge & (~allowed_edges)

        labels = torch.full((L,), self.ignore_index, dtype=torch.long)
        if target_edges.any() and not (self.stage == "train" and self.dynamic_train_masking):
            labels[target_edges] = self.id2class[input_ids[target_edges]]

        if not (self.stage == "train" and self.dynamic_train_masking):
            input_ids[target_edges] = self.mask_id
        input_ids[disallowed_edges] = self.mask_id
        attention_mask[disallowed_edges] = 0

        edge_ids = self.flat_edge_ids[s:e].to(torch.long)
        edge_classes = self.id2class[self.flat_input_ids[s:e].to(torch.long)]

        metadata = {
            "edge_ids": edge_ids,
            "walk_ids": torch.tensor(idx, dtype=torch.long),
            "positions": torch.arange(L, dtype=torch.long),
            "walk_lengths": torch.tensor(L, dtype=torch.long),
            "edge_split_mask": split_mask,
            "edge_classes": edge_classes,
        }
        return input_ids, labels, attention_mask, metadata

    def _getitem_padded(self, idx):
        input_ids = self.input_ids[idx].clone()
        split_mask = self.edge_split_mask[idx]
        attention_mask = self.attention_base[idx].clone()

        is_edge = split_mask != SplitID.BAD
        target_edges = split_mask == self.target_split

        allowed_edges = torch.zeros_like(split_mask, dtype=torch.bool)
        for split_id in self.allowed_splits:
            allowed_edges |= split_mask == split_id
        disallowed_edges = is_edge & (~allowed_edges)

        labels = torch.full_like(input_ids, self.ignore_index)
        if target_edges.any() and not (self.stage == "train" and self.dynamic_train_masking):
            labels[target_edges] = self.id2class[input_ids[target_edges]]

        if not (self.stage == "train" and self.dynamic_train_masking):
            input_ids[target_edges] = self.mask_id
        input_ids[disallowed_edges] = self.mask_id
        attention_mask[disallowed_edges] = 0

        if (
            self.edge_ids is not None
            and self.walk_ids is not None
            and self.positions is not None
            and self.walk_lengths is not None
        ):
            metadata = {
                "edge_ids": self.edge_ids[idx],
                "walk_ids": self.walk_ids[idx],
                "positions": self.positions[idx],
                "walk_lengths": self.walk_lengths[idx],
                "edge_split_mask": split_mask,
                "edge_classes": self.id2class[self.input_ids[idx]],
            }
            return input_ids, labels, attention_mask, metadata

        return input_ids, labels, attention_mask


def ragged_collate_fn(pad_id: int, ignore_index: int):
    """Returns a collate function that pads variable-length ragged samples to batch max length."""
    def collate(batch):
        inputs, labels, attns, metas = zip(*batch)

        input_ids = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
        labels_t = pad_sequence(labels, batch_first=True, padding_value=ignore_index)
        attention_mask = pad_sequence(attns, batch_first=True, padding_value=0)

        edge_ids = pad_sequence([m["edge_ids"] for m in metas], batch_first=True, padding_value=-1)
        # walk_ids and walk_lengths are scalars per item — stack into [B]
        walk_ids = torch.stack([m["walk_ids"] for m in metas])
        positions = pad_sequence([m["positions"] for m in metas], batch_first=True, padding_value=-1)
        walk_lengths = torch.stack([m["walk_lengths"] for m in metas])
        edge_split_mask = pad_sequence(
            [m["edge_split_mask"] for m in metas], batch_first=True, padding_value=int(SplitID.BAD)
        )
        edge_classes = pad_sequence(
            [m["edge_classes"] for m in metas], batch_first=True, padding_value=ignore_index
        )
        metadata = {
            "edge_ids": edge_ids,
            "walk_ids": walk_ids,
            "positions": positions,
            "walk_lengths": walk_lengths,
            "edge_split_mask": edge_split_mask,
            "edge_classes": edge_classes,
        }
        return input_ids, labels_t, attention_mask, metadata

    return collate


class BucketBatchSampler(Sampler):
    """Groups walks by length bucket, batches within bucket, shuffles batch order each epoch.

    Epoch shuffling uses (seed + call_count) so each pass through the DataLoader
    uses a different permutation without needing external set_epoch() calls.
    """

    def __init__(self, lengths, batch_size: int, bucket_width: int = 16, shuffle: bool = True, seed: int = 0):
        self.batch_size = batch_size
        self.bucket_width = bucket_width
        self.shuffle = shuffle
        self.seed = seed
        self._iter_count = 0

        # Build buckets: bucket_id (length // bucket_width) -> list of walk indices
        self._buckets: dict = {}
        for idx, length in enumerate(lengths):
            bid = int(length) // bucket_width
            if bid not in self._buckets:
                self._buckets[bid] = []
            self._buckets[bid].append(idx)

        self._num_batches = sum(
            math.ceil(len(idxs) / batch_size) for idxs in self._buckets.values()
        )

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._iter_count)
        self._iter_count += 1

        all_batches = []
        for bucket_idxs in self._buckets.values():
            idxs = list(bucket_idxs)
            if self.shuffle:
                rng.shuffle(idxs)
            for start in range(0, len(idxs), self.batch_size):
                all_batches.append(idxs[start : start + self.batch_size])

        if self.shuffle:
            rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch


def create_stage_dataloaders(
    cache_data: dict,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = None,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    dynamic_train_masking: bool = False,
    use_bucket_batching: bool = True,
    bucket_width: int = 16,
    seed: int = 0,
):
    ragged = "offsets" in cache_data.get("encoded", {})

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if pin_memory:
        try:
            key = "flat_input_ids" if ragged else "input_ids"
            t = cache_data["encoded"][key]
            if t.untyped_storage().filename() is not None:
                pin_memory = False
        except (AttributeError, TypeError, KeyError):
            pass

    dataloader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(persistent_workers)
        dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_dataset = StageViewDataset(cache_data, stage="train", dynamic_train_masking=dynamic_train_masking)
    val_dataset   = StageViewDataset(cache_data, stage="val",   dynamic_train_masking=False)
    test_dataset  = StageViewDataset(cache_data, stage="test",  dynamic_train_masking=False)

    if ragged:
        tokenizer = cache_data["tokenizer"]
        collate = ragged_collate_fn(int(tokenizer["PAD_ID"]), int(tokenizer["UNK_LABEL_ID"]))

        if use_bucket_batching:
            train_sampler = BucketBatchSampler(
                lengths=train_dataset.lengths.tolist(),
                batch_size=batch_size,
                bucket_width=bucket_width,
                shuffle=True,
                seed=seed,
            )
            # batch_sampler is mutually exclusive with batch_size/shuffle/sampler
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                collate_fn=collate,
                **dataloader_kwargs,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate,
                **dataloader_kwargs,
            )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate, **dataloader_kwargs,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate, **dataloader_kwargs,
        )
    else:
        # Padded mode — full backward compat, no collate_fn needed
        padded_kwargs = {"batch_size": batch_size, **dataloader_kwargs}
        train_loader = DataLoader(train_dataset, shuffle=True,  **padded_kwargs)
        val_loader   = DataLoader(val_dataset,   shuffle=False, **padded_kwargs)
        test_loader  = DataLoader(test_dataset,  shuffle=False, **padded_kwargs)

    return {"train": train_loader, "val": val_loader, "test": test_loader}
