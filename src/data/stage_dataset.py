"""
Stage-view dataset that builds train/val/test views on-the-fly.

Stores base tensors once and applies stage-specific masking in __getitem__().
"""

import torch
from torch.utils.data import Dataset
from enum import IntEnum


class SplitID(IntEnum):
    TRAIN = 0
    MASK = 1
    VAL = 2
    TEST = 3
    BAD = -1


class StageViewDataset(Dataset):
    """Create stage-specific views on-the-fly for a cached dataset."""

    def __init__(self, cache_data: dict, stage: str = "train"):
        self.input_ids = cache_data["encoded"]["input_ids"]
        self.edge_split_mask = cache_data["encoded"]["edge_split_mask"]
        self.attention_base = cache_data["encoded"]["attention_base"]

        self.edge_ids = cache_data["encoded"].get("edge_ids")
        self.walk_ids = cache_data["encoded"].get("walk_ids")
        self.positions = cache_data["encoded"].get("positions")
        self.walk_lengths = cache_data["encoded"].get("walk_lengths")

        tokenizer = cache_data["tokenizer"]
        self.mask_id = tokenizer["MASK_ID"]
        self.ignore_index = tokenizer["UNK_LABEL_ID"]

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
        if target_edges.any():
            labels[target_edges] = self.id2class[input_ids[target_edges]]

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
            }
            return input_ids, labels, attention_mask, metadata

        # Match old format order: (input_ids, labels, attention_mask)
        return input_ids, labels, attention_mask


def create_stage_dataloaders(cache_data: dict, batch_size: int, num_workers: int = 0):
    from torch.utils.data import DataLoader

    train_dataset = StageViewDataset(cache_data, stage="train")
    val_dataset = StageViewDataset(cache_data, stage="val")
    test_dataset = StageViewDataset(cache_data, stage="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
