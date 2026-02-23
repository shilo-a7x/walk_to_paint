import pytest
from omegaconf import OmegaConf

from src.utils.config import validate_config


def _base_cfg():
    return OmegaConf.create(
        {
            "reproducibility": {"seed": 42},
            "dataset": {
                "name": "wiki-rfa",
                "data_dir": "data/wiki-rfa",
                "max_walk_length": 80,
                "num_walks": 1000,
                "train_ratio": 0.48,
                "mask_ratio": 0.32,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
            },
            "preprocess": {"use_cache": True, "save": True},
            "training": {
                "epochs": 3,
                "batch_size": 64,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "use_cuda": False,
            },
            "model": {
                "embedding_dim": 32,
                "hidden_dim": 16,
                "nhead": 8,
                "nlayers": 2,
                "dropout": 0.2,
            },
        }
    )


def test_validate_config_valid_passes():
    cfg = _base_cfg()
    validate_config(cfg, context="train")


def test_validate_config_missing_required_key_fails():
    cfg = _base_cfg()
    del cfg.dataset.max_walk_length

    with pytest.raises(ValueError, match="dataset.max_walk_length"):
        validate_config(cfg, context="train")


def test_validate_config_bad_dropout_range_fails():
    cfg = _base_cfg()
    cfg.model.dropout = 1.5

    with pytest.raises(ValueError, match="model.dropout"):
        validate_config(cfg, context="train")


def test_validate_config_split_ratio_mismatch_fails():
    cfg = _base_cfg()
    cfg.dataset.test_ratio = 0.2

    with pytest.raises(ValueError, match="split_sum"):
        validate_config(cfg, context="train")


def test_validate_config_embedding_not_divisible_by_nhead_fails():
    cfg = _base_cfg()
    cfg.model.embedding_dim = 30
    cfg.model.nhead = 8

    with pytest.raises(ValueError, match="divisible"):
        validate_config(cfg, context="train")
