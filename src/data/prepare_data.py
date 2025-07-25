import os
import random
from src.data.tokenizer import Tokenizer
from src.data.load_graph import load_edges
from src.data.walk_sampler import sample_walks
from src.data.dataloader import build_data_module


def prepare_all(cfg):
    dataset_dir = os.path.join(cfg.dataset.data_dir, cfg.dataset.name)
    edge_path = os.path.join(dataset_dir, "raw.txt")
    edges = load_edges(edge_path)

    walks = sample_walks(
        edges, num_walks=cfg.dataset.num_walks, walk_length=cfg.dataset.max_walk_length
    )
    tokenizer = Tokenizer()
    tokenizer.fit(walks)
    encoded = [tokenizer.encode(w) for w in walks]

    # Mask edges (even indices are nodes, odd are edge labels)
    inputs, labels = [], []
    for seq in encoded:
        inp, lbl = [], []
        for i, tok in enumerate(seq):
            if i % 2 == 1 and random.random() < cfg.dataset.mask_prob:
                inp.append(
                    tokenizer.token2id["[MASK]"]
                    if "[MASK]" in tokenizer.token2id
                    else 0
                )
                lbl.append(tok)
            else:
                inp.append(tok)
                lbl.append(cfg.dataset.ignore_index)
        inputs.append(inp)
        labels.append(lbl)

    return build_data_module(inputs, labels, cfg), tokenizer
