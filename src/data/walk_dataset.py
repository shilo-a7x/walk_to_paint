import torch
from torch.utils.data import Dataset


class WalkDataset(Dataset):
    """Dataset that returns walk metadata alongside model inputs.

    Returns: (input_ids, labels, attention_mask, metadata)
    where metadata is a dict with:
      - edge_ids
      - walk_ids
      - positions
      - walk_lengths
    """

    def __init__(self, input_ids, labels, attention_mask, metadata):
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask
        self.metadata = metadata

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        meta = {k: v[idx] for k, v in self.metadata.items()}
        return (
            self.input_ids[idx],
            self.labels[idx],
            self.attention_mask[idx],
            meta,
        )
