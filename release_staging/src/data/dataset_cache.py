"""
Dataset cache format for data pipeline.

v2.0 (current): Ragged/CSR format.  Single file stores tokenizer, CSR arrays
(offsets, flat_input_ids, flat_split_mask, flat_edge_ids), splits, and metadata.
No global padding — walks are stored at their natural length.  Dtype is chosen
by the caller to minimise memory (int16/int32/int64 for ids; int8 for split mask).

v1.x (legacy): Padded 2D tensors — triggers a rebuild warning on load.
"""

import os
import warnings
import torch
from typing import Dict, Set, Tuple, Any

from src.data.tokenizer import Tokenizer


def save_dataset_cache(
    cache_path: str,
    tokenizer: Tokenizer,
    offsets: torch.Tensor,
    flat_input_ids: torch.Tensor,
    flat_split_mask: torch.Tensor,
    flat_edge_ids: torch.Tensor,
    splits: Dict[str, Set[Tuple[int, int, int]]],
    metadata: Dict[str, Any],
) -> float:
    """Save ragged (CSR) dataset cache v2.0.

    No global padding.  Dtype is chosen by the caller via build_ragged_arrays:
      offsets:        int32 or int64
      flat_input_ids: int16 (vocab<=32767), int32, or int64
      flat_split_mask: int8
      flat_edge_ids:  int32 or int64
    """
    tokenizer_state = {
        "token2id": tokenizer.token2id,
        "id2token": tokenizer.id2token,
        "edge_label2id": tokenizer.edge_label2id,
        "id2edge_label": tokenizer.id2edge_label,
        "PAD_ID": tokenizer.PAD_ID,
        "MASK_ID": tokenizer.MASK_ID,
        "UNK_ID": tokenizer.UNK_ID,
        "UNK_LABEL_ID": tokenizer.UNK_LABEL_ID,
        "vocab_size": tokenizer.vocab_size,
        "num_edge_tokens": tokenizer.num_edge_tokens,
    }
    splits_serializable = {
        k: [tuple(int(x) for x in edge) for edge in v] for k, v in splits.items()
    }
    cache_data = {
        "version": _CURRENT_VERSION,
        "tokenizer": tokenizer_state,
        "encoded": {
            "offsets": offsets,
            "flat_input_ids": flat_input_ids,
            "flat_split_mask": flat_split_mask,
            "flat_edge_ids": flat_edge_ids,
        },
        "splits": splits_serializable,
        "metadata": metadata,
    }
    torch.save(cache_data, cache_path)
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    return size_mb


_CURRENT_VERSION = "2.0"
_LEGACY_VERSIONS = {"1.0", "1.1", "1.2"}


def load_dataset_cache(cache_path: str, use_mmap: bool = False) -> Dict[str, Any]:
    """Load dataset cache file.

    Args:
        cache_path: Path to the .pt cache file.
        use_mmap: When True, attempt to memory-map the file's tensors so the OS
            pages them in on demand rather than reading the whole file upfront.
            Falls back to a normal load if the mmap call fails (e.g. on some
            network file systems that do not support MAP_SHARED).
    """
    if use_mmap:
        try:
            cache_data = torch.load(cache_path, weights_only=False, mmap=True)
        except Exception as e:
            print(f"⚠️  mmap load failed ({e}), falling back to normal load.")
            cache_data = torch.load(cache_path, weights_only=False)
    else:
        cache_data = torch.load(cache_path, weights_only=False)
    version = cache_data.get("version", "unknown")
    if version in _LEGACY_VERSIONS:
        print(
            f"\n⚠️  Cache version {version} detected (current: {_CURRENT_VERSION}).\n"
            f"   The old format includes large serialized data (walks list and/or\n"
            f"   redundant tensors) that makes loading SLOW and wastes disk space.\n"
            f"   Delete {cache_path} and re-run to rebuild with the fast format.\n"
        )
    cache_data["splits"] = {
        k: {tuple(edge) for edge in v} for k, v in cache_data["splits"].items()
    }
    return cache_data


def cache_exists(cache_path: str) -> bool:
    """Check if dataset cache file exists."""
    return os.path.exists(cache_path)


def tokenizer_from_cache(cache_data: Dict[str, Any]) -> Tokenizer:
    """Rebuild a Tokenizer object from cached tokenizer state."""
    state = cache_data.get("tokenizer", {})
    tok = Tokenizer()
    tok.token2id = state.get("token2id", {})
    tok.id2token = {int(v): k for k, v in tok.token2id.items()}
    tok.edge_label2id = state.get("edge_label2id", {})
    tok.id2edge_label = {int(v): k for k, v in tok.edge_label2id.items()}

    # Rebuild cached lookup structures
    tok._edge_tokens = set()
    tok._node_tokens = set()
    tok._token_to_node_id = {}
    tok._token_to_edge_label = {}

    for token in tok.token2id.keys():
        if token.startswith(tok.EDGE_PREFIX):
            tok._edge_tokens.add(token)
            try:
                label = int(token.split(tok.DELIMITER, 1)[1])
                tok._token_to_edge_label[token] = label
            except (IndexError, ValueError):
                warnings.warn(
                    f"dataset_cache: cannot parse integer label from edge token {token!r} "
                    "during tokenizer reconstruction from cache; token will get sort key 0 "
                    "in _build_edge_label_map (may cause incorrect class assignment).",
                    stacklevel=2,
                )
        elif token.startswith(tok.NODE_PREFIX):
            tok._node_tokens.add(token)
            try:
                node_id = int(token.split(tok.DELIMITER, 1)[1])
                tok._token_to_node_id[token] = node_id
            except (IndexError, ValueError):
                warnings.warn(
                    f"dataset_cache: cannot parse integer node id from node token {token!r} "
                    "during tokenizer reconstruction from cache.",
                    stacklevel=2,
                )

    return tok
