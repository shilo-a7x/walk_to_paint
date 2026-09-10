"""Edge-identity-token variant of src/data/prepare_data.py's cache-loading path.

Deliberately NOT a full copy of prepare_data.py: this experiment doesn't build a
cache from scratch (get_edge_list/split_edges/get_walks/get_tokenizer/encode_walks/
build_ragged_arrays), it loads a cache already produced two ways in sequence, both
using real, unmodified production code:

  1. `src/data/prepare_data.py::prepare_data(cfg)` (called by build_eid_cache.py,
     imported directly, not copied) -- the exact real pipeline used for every
     production training run, walk-sampler and all -- builds/loads the normal
     production dataset_cache__*.pt for the dataset (2-token edge vocabulary).
  2. `experiments/edge_identity_tokens/build_cache.py::build` -- a pure
     post-processing transform (no resampling) that rewrites that cache's edge
     tokens into per-edge identity tokens + a parallel sign array (see
     MECHANISM.md).

This function is the thin loader for the RESULT of step 2 -- it plays the same
role prepare_data.py's `if load_path is not None:` branch plays for a normal
cache (setting cfg.model.* metadata fields, calling the stage-dataloader
builder), just pointed at the EID cache and eid_src's own
create_eid_stage_dataloaders instead of production's.
"""

import os

from src.data.dataset_cache import load_dataset_cache
from src.data.prepare_data import _dataloader_kwargs
from src.utils.config import get_seed

from experiments.edge_identity_tokens.eid_src.data.stage_dataset import create_eid_stage_dataloaders


def prepare_eid_data(cfg, eid_cache_path: str):
    if not os.path.isfile(eid_cache_path):
        raise FileNotFoundError(
            f"EID cache not found at {eid_cache_path} -- build it first via "
            "experiments/edge_identity_tokens/build_cache.py"
        )

    print(f"Loading EID dataset cache from {eid_cache_path}...")
    cache_data = load_dataset_cache(eid_cache_path, use_mmap=False)
    tok = cache_data["tokenizer"]

    cfg.model.vocab_size = int(tok["vocab_size"])
    cfg.model.old_vocab_size = int(tok["old_vocab_size"])
    cfg.model.eid_cache_path = str(eid_cache_path)  # for model.py's edge_residual_baseline
    cfg.model.num_classes = cache_data["metadata"]["num_classes"]
    cfg.model.pad_id = cache_data["metadata"]["pad_id"]
    cfg.model.ignore_index = cache_data["metadata"]["ignore_index"]
    cfg.model.unk_id = int(tok["UNK_ID"])
    cfg.model.mask_id = int(tok["MASK_ID"])
    if "class_weights" in cache_data["metadata"]:
        cfg.model.class_weights = cache_data["metadata"]["class_weights"]

    size_mb = os.path.getsize(eid_cache_path) / (1024 * 1024)
    print(
        f"Success! (loaded {size_mb:.1f} MB, vocab_size={cfg.model.vocab_size}, "
        f"old_vocab_size={cfg.model.old_vocab_size})"
    )

    return create_eid_stage_dataloaders(
        cache_data,
        dynamic_train_masking=bool(getattr(cfg.model, "dynamic_train_masking", False)),
        randomize_walk_direction=bool(getattr(cfg.model, "randomize_walk_direction", False)),
        reveal_holdout_identity=bool(getattr(cfg.model, "eid_reveal_holdout_identity", False)),
        reveal_holdout_attendable_only=bool(getattr(cfg.model, "eid_reveal_holdout_attendable_only", False)),
        **_dataloader_kwargs(cfg),
    )
