from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.data.dataset_cache import load_dataset_cache, save_dataset_cache
from src.data.prepare_data import build_runtime_cache_data
from src.data.stage_dataset import create_stage_dataloaders
from src.data.tokenizer import Tokenizer
from src.model.lit_model import LitEdgeClassifier, SPLIT_MASK, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL


def _build_cfg():
    return OmegaConf.create(
        {
            "reproducibility": {"seed": 42},
            "dataset": {
                "name": "toy-dynamic",
                "data_dir": "data/toy-dynamic",
                "max_walk_length": 2,
                "num_walks": 3,
                "train_ratio": 0.48,
                "mask_ratio": 0.32,
                "val_ratio": 0.10,
                "test_ratio": 0.10,
            },
            "preprocess": {"use_cache": False, "save": False},
            "training": {
                "epochs": 4,
                "batch_size": 2,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "use_cuda": False,
            },
            "model": {
                "embedding_dim": 8,
                "hidden_dim": 16,
                "nhead": 2,
                "nlayers": 1,
                "dropout": 0.1,
                "dynamic_train_masking": True,
                "dynamic_train_mask_seed_offset": 0,
                "node_context_mode": "none",
            },
        }
    )


def _build_synthetic_cache_data():
    tok = Tokenizer()
    walks = [
        ["N_0", "E_0", "N_1", "E_1", "N_2"],
        ["N_2", "E_1", "N_3", "E_0", "N_0"],
        ["N_1", "E_1", "N_3", "E_0", "N_2"],
    ]
    tok.fit(walks)

    input_ids = torch.tensor(
        [tok.encode(walk) for walk in walks],
        dtype=torch.long,
    )
    edge_split_mask = torch.tensor(
        [
            [-1, SPLIT_TRAIN, -1, SPLIT_MASK, -1],
            [-1, SPLIT_TRAIN, -1, SPLIT_MASK, -1],
            [-1, SPLIT_VAL, -1, SPLIT_TEST, -1],
        ],
        dtype=torch.long,
    )
    attention_base = torch.ones_like(input_ids)
    edge_ids = torch.tensor(
        [
            [-1, 0, -1, 1, -1],
            [-1, 2, -1, 3, -1],
            [-1, 4, -1, 5, -1],
        ],
        dtype=torch.long,
    )
    walk_ids = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
        ],
        dtype=torch.long,
    )
    positions = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    walk_lengths = torch.tensor(
        [
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
        ],
        dtype=torch.long,
    )
    splits_dict = {
        "train": {(0, 1, 0), (2, 3, 1)},
        "mask": {(1, 2, 1), (3, 0, 0)},
        "val": {(1, 3, 1)},
        "test": {(3, 2, 0)},
    }
    metadata = {
        "vocab_size": tok.vocab_size,
        "num_classes": tok.num_edge_tokens,
        "pad_id": tok.PAD_ID,
        "ignore_index": tok.UNK_LABEL_ID,
        "class_weights": [1.0, 1.0],
        "dataset_name": "toy-dynamic",
        "seed": 42,
    }
    cache_data = build_runtime_cache_data(
        tok,
        input_ids,
        edge_split_mask,
        attention_base,
        splits_dict,
        metadata,
        edge_ids,
        walk_ids,
        positions,
        walk_lengths,
    )
    return cache_data, tok


def _assert_items_equal(a, b):
    a_input, a_labels, a_attn, a_meta = a
    b_input, b_labels, b_attn, b_meta = b
    assert torch.equal(a_input, b_input)
    assert torch.equal(a_labels, b_labels)
    assert torch.equal(a_attn, b_attn)
    assert set(a_meta) == set(b_meta)
    for key in a_meta:
        assert torch.equal(a_meta[key], b_meta[key])


def test_runtime_and_loaded_cache_stage_views_are_identical(tmp_path: Path):
    cache_data, tok = _build_synthetic_cache_data()
    cache_path = tmp_path / "dataset_cache.pt"

    save_dataset_cache(
        str(cache_path),
        tok,
        cache_data["encoded"]["input_ids"],
        cache_data["encoded"]["edge_split_mask"],
        cache_data["encoded"]["attention_base"],
        cache_data["splits"],
        cache_data["metadata"],
        edge_ids=cache_data["encoded"]["edge_ids"],
    )

    loaded_cache = load_dataset_cache(str(cache_path))
    runtime_loaders = create_stage_dataloaders(cache_data, batch_size=2, num_workers=0)
    loaded_loaders = create_stage_dataloaders(loaded_cache, batch_size=2, num_workers=0)

    for split in ("train", "val", "test"):
        runtime_ds = runtime_loaders[split].dataset
        loaded_ds = loaded_loaders[split].dataset
        assert len(runtime_ds) == len(loaded_ds)
        for idx in range(len(runtime_ds)):
            _assert_items_equal(runtime_ds[idx], loaded_ds[idx])


def test_dynamic_train_targets_resample_without_val_test_leakage():
    cfg = _build_cfg()
    cache_data, _ = _build_synthetic_cache_data()

    cfg.model.vocab_size = cache_data["metadata"]["vocab_size"]
    cfg.model.num_classes = cache_data["metadata"]["num_classes"]
    cfg.model.pad_id = cache_data["metadata"]["pad_id"]
    cfg.model.ignore_index = cache_data["metadata"]["ignore_index"]
    cfg.model.mask_id = cache_data["tokenizer"]["MASK_ID"]
    cfg.model.unk_id = cache_data["tokenizer"]["UNK_ID"]
    cfg.model.class_weights = cache_data["metadata"]["class_weights"]

    loaders = create_stage_dataloaders(
        cache_data,
        batch_size=2,
        num_workers=0,
        dynamic_train_masking=True,
    )
    model = LitEdgeClassifier(cfg)

    class DummyTrainer:
        pass

    trainer = DummyTrainer()
    trainer.train_dataloader = loaders["train"]
    model._trainer = trainer

    model._build_dynamic_train_pool()
    model._sample_epoch_targets(0)
    epoch0 = model._epoch_target_edge_ids_cpu.clone()
    model._sample_epoch_targets(1)
    epoch1 = model._epoch_target_edge_ids_cpu.clone()

    assert epoch0.numel() == 2
    assert epoch1.numel() == 2
    assert not torch.equal(epoch0, epoch1)

    val_test_ids = torch.tensor([4, 5], dtype=torch.long)
    assert not torch.isin(epoch0, val_test_ids).any()
    assert not torch.isin(epoch1, val_test_ids).any()

    # Iterate all batches: at least one must have dynamic targets (avoids dependence on
    # random shuffle putting the right walk in the first batch).
    total_target_count = 0
    last_dynamic_input_ids, last_labels = None, None
    for batch in loaders["train"]:
        dynamic_input_ids, labels = model._build_dynamic_targets_for_batch(batch[0], batch[3])
        count = int((labels != model.ignore_index).sum().item())
        if count > 0:
            total_target_count += count
            last_dynamic_input_ids, last_labels = dynamic_input_ids, labels
    assert total_target_count >= 1
    assert last_dynamic_input_ids is not None
    assert torch.equal(last_dynamic_input_ids[last_labels != model.ignore_index], torch.full_like(last_dynamic_input_ids[last_labels != model.ignore_index], cfg.model.mask_id))