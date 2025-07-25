import os


def load_chess(cfg):
    path = os.path.join(cfg.dataset.data_dir, cfg.dataset.name, "raw.txt")
    edges = []
    with open(path) as f:
        for line in f:
            u, v, label, *_ = line.strip().split()
            edges.append((int(u), int(v), int(label)))
    return edges


def load_bitcoin(cfg):
    # TODO: custom format loader
    raise NotImplementedError("Bitcoin loader not implemented.")


DATASET_LOADERS = {
    "chess": load_chess,
    "bitcoin": load_bitcoin,
}


def get_loader(name):
    return DATASET_LOADERS[name]
