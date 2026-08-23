"""Edge-identity-token variant of src/data/stage_dataset.py.

This is a full copy of the production file (SplitID, _RaggedCollate,
BucketBatchSampler, create_stage_dataloaders are byte-identical in spirit --
copied here rather than imported so this experiment can be modified without any
risk of touching production code, per this project's `experiments/` isolation
convention, e.g. `experiments/local_attention_windowed/`). The ONLY class with a
real behavioral change is `StageViewDataset` -> `EdgeIdentityStageViewDataset`,
and only two things differ inside it (see inline comments at each diff):

  1. `id2class` is built from the EID cache's per-edge `edge_sign_lookup`
     (build_cache.py) instead of the production tokenizer's 2-token
     `id2edge_label` map -- production's id2class has exactly one valid entry per
     sign class (2 total); this dataset's id2class has one valid entry per real
     edge (tens of thousands), because every edge now has its own unique token id.
  2. `_getitem_ragged` hides a target/disallowed edge's SIGN only
     (`sign_ids[...] = sign_na_id`), never its IDENTITY (`input_ids` is left
     untouched at target positions) -- production hides both simultaneously by
     construction, since the token id *is* the sign there. See
     experiments/edge_identity_tokens/MECHANISM.md section 3 for the full
     rationale and a worked example.

Everything else (split logic, disallowed-edge exclusion from attention, bucket
batching, collate padding) is unchanged from production.
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


SIGN_NA = 2  # "not applicable" (vertex position) or "hidden" (masked/held-out edge)


class EdgeIdentityStageViewDataset(Dataset):
    """Stage-specific view over an edge-identity-token cache (see build_cache.py)."""

    def __init__(
        self,
        cache_data: dict,
        stage: str = "train",
        dynamic_train_masking: bool = False,
        walk_flip: torch.Tensor = None,
    ):
        if dynamic_train_masking:
            # Not supported yet in this experiment -- production's dynamic-resplit
            # hides a target by overwriting input_ids with <MASK> (see
            # LitEdgeClassifier._build_dynamic_targets_for_batch), which would hide
            # IDENTITY too, defeating this experiment's whole point. A sign-only-hide
            # variant of dynamic resplit is a real possible extension, just not
            # implemented here -- see MECHANISM.md.
            raise NotImplementedError(
                "dynamic_train_masking is not supported by EdgeIdentityStageViewDataset "
                "yet -- keep model.dynamic_train_masking=false in the EID run config."
            )

        enc = cache_data["encoded"]
        tokenizer = cache_data["tokenizer"]
        self.mask_id = tokenizer["MASK_ID"]
        self.ignore_index = tokenizer["UNK_LABEL_ID"]
        self.dynamic_train_masking = False
        self.walk_flip = walk_flip
        self.stage = stage
        self.sign_na_id = SIGN_NA

        # --- DIFF 1: id2class built from the per-edge sign lookup, not the 2-token
        # production map. Covers the FULL new vocab (old_vocab_size + num_edges):
        # -1 (ignore) everywhere except at a real edge-identity token id, where it
        # holds that edge's fixed true sign class (0/1). This lets every other piece
        # of production machinery that reads `id2class[input_ids]` to recover the
        # true class per position (dynamic resplit's pool builder, sign-scramble
        # ablation, if ever enabled here) work completely unmodified against this
        # dataset -- see MECHANISM.md and eid_src/model/lit_model.py.
        old_vocab_size = int(tokenizer["old_vocab_size"])
        num_edges = int(tokenizer["num_edges"])
        vocab_size = int(tokenizer["vocab_size"])
        edge_sign_lookup = tokenizer["edge_sign_lookup"]  # (num_edges,) int8, 0/1

        self.id2class = torch.full((vocab_size,), self.ignore_index, dtype=torch.long)
        self.id2class[old_vocab_size : old_vocab_size + num_edges] = edge_sign_lookup.long()

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

        self.offsets = enc["offsets"]
        self.flat_input_ids = enc["flat_input_ids"]
        self.flat_split_mask = enc["flat_split_mask"]
        self.flat_edge_ids = enc["flat_edge_ids"]
        self._N = len(self.offsets) - 1
        self.lengths = (self.offsets[1:] - self.offsets[:-1]).to(torch.long)
        for arr in [self.offsets, self.flat_input_ids, self.flat_split_mask, self.flat_edge_ids]:
            arr.share_memory_()

    def __len__(self):
        return self._N

    @property
    def input_ids(self):
        return self.flat_input_ids

    @property
    def edge_split_mask(self):
        return self.flat_split_mask

    @property
    def edge_ids(self):
        return self.flat_edge_ids

    def __getitem__(self, idx):
        return self._getitem_ragged(idx)

    def _getitem_ragged(self, idx):
        s = int(self.offsets[idx])
        e = int(self.offsets[idx + 1])
        L = e - s

        input_ids = self.flat_input_ids[s:e].to(torch.long)
        split_mask = self.flat_split_mask[s:e].to(torch.long)
        edge_ids = self.flat_edge_ids[s:e].to(torch.long)

        if self.walk_flip is not None and bool(self.walk_flip[idx]):
            input_ids = input_ids.flip(0)
            split_mask = split_mask.flip(0)
            edge_ids = edge_ids.flip(0)

        attention_mask = torch.ones(L, dtype=torch.long)

        is_edge = split_mask != int(SplitID.BAD)
        target_edges = split_mask == int(self.target_split)

        allowed_edges = torch.zeros(L, dtype=torch.bool)
        for split_id in self.allowed_splits:
            allowed_edges |= split_mask == int(split_id)
        disallowed_edges = is_edge & (~allowed_edges)

        # True sign class per position, from the raw (possibly reversed) ids --
        # identical in spirit to production's `edge_classes = self.id2class[input_ids]`,
        # just resolved against the per-edge id2class built in __init__.
        edge_classes = self.id2class[input_ids]

        labels = torch.full((L,), self.ignore_index, dtype=torch.long)
        labels[target_edges] = edge_classes[target_edges]

        # sign_ids: what the model is actually allowed to see about each position's
        # sign. Starts as a copy of the true class, then gets hidden (SIGN_NA) at
        # target and disallowed positions -- mirrors `labels` construction above,
        # but drives the model's INPUT rather than the loss target.
        sign_ids = edge_classes.clone()
        sign_ids[target_edges] = self.sign_na_id
        sign_ids[disallowed_edges] = self.sign_na_id
        sign_ids[~is_edge] = self.sign_na_id  # vertex positions: always "n/a"

        # --- DIFF 2: identity (input_ids) is left untouched at target_edges -- this
        # is the one behavioral difference from production's _getitem_ragged, which
        # does `input_ids[target_edges] = self.mask_id` here. Disallowed edges still
        # get their identity hidden too, same as production (harmless either way,
        # since they're excluded from attention below regardless -- kept purely for
        # convention-consistency with "hidden edges render as <MASK>").
        input_ids[disallowed_edges] = self.mask_id
        attention_mask[disallowed_edges] = 0

        metadata = {
            "edge_ids": edge_ids,
            "walk_ids": torch.tensor(idx, dtype=torch.long),
            "positions": torch.arange(L, dtype=torch.long),
            "walk_lengths": torch.tensor(L, dtype=torch.long),
            "edge_split_mask": split_mask,
            "edge_classes": edge_classes,
            "sign_ids": sign_ids,
        }
        return input_ids, labels, attention_mask, metadata


class _EIDRaggedCollate:
    """Same as production's _RaggedCollate, plus sign_ids padding."""

    def __init__(self, pad_id: int, ignore_index: int, sign_na_id: int = SIGN_NA):
        self.pad_id = pad_id
        self.ignore_index = ignore_index
        self.sign_na_id = sign_na_id

    def __call__(self, batch):
        inputs, labels, attns, metas = zip(*batch)

        input_ids = pad_sequence(inputs, batch_first=True, padding_value=self.pad_id)
        labels_t = pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)
        attention_mask = pad_sequence(attns, batch_first=True, padding_value=0)

        edge_ids = pad_sequence(
            [m["edge_ids"] for m in metas], batch_first=True, padding_value=-1
        )
        walk_ids = torch.stack([m["walk_ids"] for m in metas])
        positions = pad_sequence(
            [m["positions"] for m in metas], batch_first=True, padding_value=-1
        )
        walk_lengths = torch.stack([m["walk_lengths"] for m in metas])
        edge_split_mask = pad_sequence(
            [m["edge_split_mask"] for m in metas],
            batch_first=True,
            padding_value=int(SplitID.BAD),
        )
        edge_classes = pad_sequence(
            [m["edge_classes"] for m in metas],
            batch_first=True,
            padding_value=self.ignore_index,
        )
        sign_ids = pad_sequence(
            [m["sign_ids"] for m in metas],
            batch_first=True,
            padding_value=self.sign_na_id,
        )
        metadata = {
            "edge_ids": edge_ids,
            "walk_ids": walk_ids,
            "positions": positions,
            "walk_lengths": walk_lengths,
            "edge_split_mask": edge_split_mask,
            "edge_classes": edge_classes,
            "sign_ids": sign_ids,
        }
        return input_ids, labels_t, attention_mask, metadata


def eid_ragged_collate_fn(pad_id: int, ignore_index: int):
    return _EIDRaggedCollate(pad_id, ignore_index)


class BucketBatchSampler(Sampler):
    """Identical to production's BucketBatchSampler (src/data/stage_dataset.py)."""

    def __init__(
        self,
        lengths,
        batch_size: int,
        bucket_width: int = 16,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.batch_size = batch_size
        self.bucket_width = bucket_width
        self.shuffle = shuffle
        self.seed = seed
        self._iter_count = 0

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


def create_eid_stage_dataloaders(
    cache_data: dict,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = None,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    dynamic_train_masking: bool = False,
    randomize_walk_direction: bool = False,
    use_bucket_batching: bool = True,
    bucket_width: int = 16,
    seed: int = 0,
):
    """Same shape/behavior as production's create_stage_dataloaders, using
    EdgeIdentityStageViewDataset + the sign_ids-aware collate function."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if pin_memory:
        try:
            t = cache_data["encoded"]["flat_input_ids"]
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

    walk_flip = None
    if randomize_walk_direction:
        n_walks = len(cache_data["encoded"]["offsets"]) - 1
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + 8_675_309)
        walk_flip = torch.rand(n_walks, generator=gen) < 0.5

    train_dataset = EdgeIdentityStageViewDataset(
        cache_data, stage="train",
        dynamic_train_masking=dynamic_train_masking,
        walk_flip=walk_flip,
    )
    val_dataset = EdgeIdentityStageViewDataset(
        cache_data, stage="val", dynamic_train_masking=False, walk_flip=walk_flip,
    )
    test_dataset = EdgeIdentityStageViewDataset(
        cache_data, stage="test", dynamic_train_masking=False, walk_flip=walk_flip,
    )

    tokenizer = cache_data["tokenizer"]
    collate = eid_ragged_collate_fn(
        int(tokenizer["PAD_ID"]), int(tokenizer["UNK_LABEL_ID"])
    )

    if use_bucket_batching:
        train_sampler = BucketBatchSampler(
            lengths=train_dataset.lengths.tolist(),
            batch_size=batch_size, bucket_width=bucket_width, shuffle=True, seed=seed,
        )
        train_loader = DataLoader(
            train_dataset, batch_sampler=train_sampler, collate_fn=collate, **dataloader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, **dataloader_kwargs,
        )

    if use_bucket_batching:
        val_sampler = BucketBatchSampler(
            lengths=val_dataset.lengths.tolist(),
            batch_size=batch_size, bucket_width=bucket_width, shuffle=False, seed=seed,
        )
        val_loader = DataLoader(
            val_dataset, batch_sampler=val_sampler, collate_fn=collate, **dataloader_kwargs,
        )
        test_sampler = BucketBatchSampler(
            lengths=test_dataset.lengths.tolist(),
            batch_size=batch_size, bucket_width=bucket_width, shuffle=False, seed=seed,
        )
        test_loader = DataLoader(
            test_dataset, batch_sampler=test_sampler, collate_fn=collate, **dataloader_kwargs,
        )
    else:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, **dataloader_kwargs,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, **dataloader_kwargs,
        )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
