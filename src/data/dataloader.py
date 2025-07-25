from torch.utils.data import DataLoader, Dataset, random_split
import torch

class WalkDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.inputs[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def build_data_module(inputs, labels, cfg):
    dataset = WalkDataset(inputs, labels)
    N = len(dataset)
    train_len = int(N * cfg.dataset.train_ratio)
    val_len = int(N * cfg.dataset.val_ratio)
    test_len = N - train_len - val_len
    train, val, test = random_split(dataset, [train_len, val_len, test_len])

    return {
        "train": DataLoader(train, batch_size=cfg.training.batch_size, shuffle=True),
        "val": DataLoader(val, batch_size=cfg.training.batch_size),
        "test": DataLoader(test, batch_size=cfg.training.batch_size),
    }