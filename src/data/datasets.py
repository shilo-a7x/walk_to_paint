import os


def load_chess(cfg):
    """Load the Chess dataset from an edge list file."""
    path = os.path.join(cfg.dataset.data_dir, cfg.dataset.edge_list_file)
    edges = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("%"):
                continue  # Skip comments
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # Invalid line
            u, v, label = int(parts[0]), int(parts[1]), int(parts[2])
            edges.append((u, v, label))
    return edges


def load_bitcoin(cfg):
    """Load the Bitcoin dataset from a csv file."""
    path = os.path.join(cfg.dataset.data_dir, cfg.dataset.edge_list_file)
    edges = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue  # Invalid line
            u, v, label = int(parts[0]), int(parts[1]), int(parts[2])
            edges.append((u, v, label))
    return edges


def load_toy(cfg):
    """Load the toy dataset from an edge list file."""
    path = os.path.join(cfg.dataset.data_dir, cfg.dataset.edge_list_file)
    edges = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("%"):
                continue  # Skip comments
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # Invalid line
            u, v, label = int(parts[0]), int(parts[1]), int(parts[2])
            edges.append((u, v, label))
    return edges


# 🔁 Registry of dataset loaders
DATASET_LOADERS = {
    "chess": load_chess,
    "bitcoin-alpha": load_bitcoin,
    "bitcoin-alpha-binary": load_bitcoin,
    "bitcoin-otc": load_bitcoin,
    "bitcoin-otc-binary": load_bitcoin,
    "toy": load_toy,
}


def get_loader(name):
    """Return the loader function for a given dataset name."""
    if name not in DATASET_LOADERS:
        raise ValueError(f"Dataset '{name}' is not supported.")
    return DATASET_LOADERS[name]
