"""StageViewDataset variant for the edge-identity-token experiment.

Differs from the production src/data/stage_dataset.py::StageViewDataset in exactly one
respect: at a masked/hidden edge position (this stage's prediction target, or a
split-excluded held-out edge), the edge's IDENTITY token is left visible -- only its
SIGN is hidden (via a parallel sign_ids array, forced to class 2 = "unknown/hidden" at
those positions). Production behavior replaces the whole token with <MASK>, conflating
"which edge" with "what sign"; this experiment's whole point is separating them.

Single-split pilot only (no dynamic_train_masking here, to keep the new sign_ids
masking logic simple and easy to verify correct on the first pass -- see
experiments/edge_identity_tokens/README.md for the scope note). Reuses the exact same
split/attention-exclusion logic as production otherwise (SplitID, target masking,
disallowed-edge exclusion from attention) -- only the two lines that decide what
replaces a hidden position's *identity* token, plus the new sign_ids array, differ.
"""
import torch
from torch.utils.data import Dataset

from src.data.stage_dataset import SplitID


class EdgeIdentityStageViewDataset(Dataset):
    def __init__(self, cache_data: dict, stage: str = "train"):
        enc = cache_data["encoded"]
        tokenizer = cache_data["tokenizer"]
        self.mask_id = tokenizer["MASK_ID"]
        self.ignore_index = tokenizer["UNK_LABEL_ID"]
        self.stage = stage
        self.sign_na_id = 2  # "not applicable" / "hidden" class for sign_embedding

        # id2class is unused here (sign is read directly from the precomputed
        # flat_sign_ids array built by build_cache.py, not re-derived from token id --
        # the whole point of this scheme is that identity no longer implies sign).

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

        self.offsets = enc["offsets"]
        self.flat_input_ids = enc["flat_input_ids"]
        self.flat_split_mask = enc["flat_split_mask"]
        self.flat_edge_ids = enc["flat_edge_ids"]
        self.flat_sign_ids = enc["flat_sign_ids"]
        self._N = len(self.offsets) - 1
        self.lengths = (self.offsets[1:] - self.offsets[:-1]).to(torch.long)
        for arr in [self.offsets, self.flat_input_ids, self.flat_split_mask,
                    self.flat_edge_ids, self.flat_sign_ids]:
            arr.share_memory_()

    def __len__(self):
        return self._N

    @property
    def edge_ids(self):
        return self.flat_edge_ids

    def __getitem__(self, idx):
        s = int(self.offsets[idx])
        e = int(self.offsets[idx + 1])
        L = e - s

        input_ids = self.flat_input_ids[s:e].to(torch.long)
        split_mask = self.flat_split_mask[s:e].to(torch.long)
        edge_ids = self.flat_edge_ids[s:e].to(torch.long)
        sign_ids = self.flat_sign_ids[s:e].to(torch.long)

        attention_mask = torch.ones(L, dtype=torch.long)

        is_edge = split_mask != int(SplitID.BAD)
        target_edges = split_mask == int(self.target_split)

        allowed_edges = torch.zeros(L, dtype=torch.bool)
        for split_id in self.allowed_splits:
            allowed_edges |= split_mask == int(split_id)
        disallowed_edges = is_edge & (~allowed_edges)

        labels = torch.full((L,), self.ignore_index, dtype=torch.long)
        labels[target_edges] = sign_ids[target_edges]

        # THE difference from production: input_ids (edge identity) is NOT overwritten
        # at target_edges -- only sign_ids is hidden there. disallowed_edges are still
        # excluded from attention entirely (attention_mask=0), so what their identity
        # token shows is moot; masked here anyway for consistency with production.
        sign_ids[target_edges] = self.sign_na_id
        sign_ids[disallowed_edges] = self.sign_na_id
        input_ids[disallowed_edges] = self.mask_id
        attention_mask[disallowed_edges] = 0

        metadata = {
            "edge_ids": edge_ids,
            "walk_ids": torch.tensor(idx, dtype=torch.long),
            "positions": torch.arange(L, dtype=torch.long),
            "sign_ids": sign_ids,
        }
        return input_ids, labels, attention_mask, metadata


class _EIDCollate:
    def __init__(self, pad_id: int, ignore_index: int, sign_na_id: int = 2):
        self.pad_id = pad_id
        self.ignore_index = ignore_index
        self.sign_na_id = sign_na_id

    def __call__(self, batch):
        from torch.nn.utils.rnn import pad_sequence
        inputs, labels, attns, metas = zip(*batch)
        input_ids = pad_sequence(inputs, batch_first=True, padding_value=self.pad_id)
        labels_t = pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)
        attention_mask = pad_sequence(attns, batch_first=True, padding_value=0)
        sign_ids = pad_sequence(
            [m["sign_ids"] for m in metas], batch_first=True, padding_value=self.sign_na_id
        )
        edge_ids = pad_sequence(
            [m["edge_ids"] for m in metas], batch_first=True, padding_value=-1
        )
        walk_ids = torch.stack([m["walk_ids"] for m in metas])
        positions = pad_sequence(
            [m["positions"] for m in metas], batch_first=True, padding_value=-1
        )
        metadata = {"edge_ids": edge_ids, "walk_ids": walk_ids, "positions": positions,
                     "sign_ids": sign_ids}
        return input_ids, labels_t, attention_mask, metadata


def eid_collate_fn(pad_id: int, ignore_index: int):
    return _EIDCollate(pad_id, ignore_index)
