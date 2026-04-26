"""
Stage-view dataset that builds train/val/test views on-the-fly.

Stores base tensors once and applies stage-specific masking in __getitem__().
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from enum import IntEnum


class SplitID(IntEnum):
    TRAIN = 0
    MASK = 1
    VAL = 2
    TEST = 3
    BAD = -1


class StageViewDataset(Dataset):
    """Create stage-specific views on-the-fly for a cached dataset."""

    def __init__(self, cache_data: dict, stage: str = "train", dynamic_train_masking: bool = False):
        self.input_ids = cache_data["encoded"]["input_ids"]
        self.edge_split_mask = cache_data["encoded"]["edge_split_mask"]
        self.attention_base = cache_data["encoded"]["attention_base"]

        self.edge_ids = cache_data["encoded"].get("edge_ids")

        # walk_ids / positions / walk_lengths are reconstructable from attention_base
        # and are no longer stored in the cache file.  Prefer stored values (runtime
        # build path); fall back to reconstruction when loading from a v1.2+ cache.
        N, seq_len = self.attention_base.shape
        enc = cache_data["encoded"]

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
            lengths = self.attention_base.sum(dim=1, dtype=torch.long)  # [N]
            self.walk_lengths = lengths.unsqueeze(1).expand(N, seq_len)

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
            self.allowed_splits = [
                SplitID.TRAIN,
                SplitID.MASK,
                SplitID.VAL,
                SplitID.TEST,
            ]
            self.target_split = SplitID.TEST
        else:
            raise ValueError("Stage must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
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

        # Match old format order: (input_ids, labels, attention_mask)
        return input_ids, labels, attention_mask


def create_stage_dataloaders(
    cache_data: dict,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = None,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    dynamic_train_masking: bool = False,
):

    if pin_memory is None:
        pin_memory = True if torch.cuda.is_available() else False

    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(persistent_workers)
        dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_dataset = StageViewDataset(
        cache_data,
        stage="train",
        dynamic_train_masking=dynamic_train_masking,
    )
    val_dataset = StageViewDataset(cache_data, stage="val", dynamic_train_masking=False)
    test_dataset = StageViewDataset(cache_data, stage="test", dynamic_train_masking=False)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **dataloader_kwargs,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
