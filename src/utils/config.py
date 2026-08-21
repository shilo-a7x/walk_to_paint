from pathlib import Path
from typing import List, Optional
from omegaconf import ListConfig, OmegaConf
import torch


def get_seed(cfg) -> int:
    """
    Get the canonical seed from config.

    This is the ONLY way to access the seed value in the codebase.
    Fails loudly if seed is not configured - no silent defaults.

    Args:
        cfg: OmegaConf configuration object

    Returns:
        int: The seed value

    Raises:
        ValueError: If reproducibility.seed is not set in config
    """
    try:
        seed = cfg.reproducibility.seed
        if seed is None:
            raise ValueError(
                "reproducibility.seed is None. "
                "Please set a valid integer seed in your config file."
            )
        return int(seed)
    except (AttributeError, KeyError):
        raise ValueError(
            "reproducibility.seed is not set in config. "
            "Please add 'reproducibility:\\n  seed: 42' to your config.yaml "
            "or set it via CLI: --reproducibility.seed=42"
        )


def load_config(
    config_path: str = "config.yaml", overrides: Optional[List[str]] = None
):
    """
    Load a base config and optionally merge a dataset-specific config found in `configs/<dataset>.yaml`.

    - `config_path` can be a path to a base config (default `config.yaml`).
    - `overrides` is a dotlist (as provided by argparse) and will be merged after loading files.

    Behavior:
      1. Load base config from `config_path`.
      2. If `dataset.name` is set in the merged config (or via overrides), look for `configs/<dataset.name>.yaml`.
      3. If found, merge dataset config on top of base config.
      4. Finally, apply CLI overrides.

    Returns the merged OmegaConf config object.
    """

    base_cfg = OmegaConf.load(config_path)

    cli_cfg = OmegaConf.from_dotlist(overrides or [])
    merged = OmegaConf.merge(base_cfg, cli_cfg)

    # Try to locate a dataset-specific config in configs/<name>.yaml
    dataset_name = None
    try:
        dataset_name = merged.dataset.name
    except Exception:
        dataset_name = None

    if dataset_name:
        cfg_file = Path("configs") / f"{dataset_name}.yaml"
        if cfg_file.exists():
            ds_cfg = OmegaConf.load(str(cfg_file))
            merged = OmegaConf.merge(merged, ds_cfg)

    # Re-apply CLI overrides to ensure they take precedence
    merged = OmegaConf.merge(merged, cli_cfg)

    return merged


def validate_config(cfg, context: str = "train") -> None:
    """
    Validate config values and fail fast on invalid or inconsistent settings.

    Args:
        cfg: OmegaConf configuration object
        context: Validation context ("train", "optuna", "posthoc")

    Raises:
        ValueError: On missing required fields, invalid types/ranges, or inconsistent config.
    """

    def _fmt_value(value):
        if value is _MISSING:
            return "<missing>"
        return repr(value)

    def _invalid(path: str, value, expected: str) -> None:
        raise ValueError(
            f"Invalid config: {path}={_fmt_value(value)} (expected {expected})"
        )

    def _get(path: str, default=None):
        return OmegaConf.select(cfg, path, default=default)

    def _is_int(value) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    def _is_number(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _require(path: str, expected: str):
        value = _get(path, _MISSING)
        if value is _MISSING or value is None:
            _invalid(path, value, expected)
        return value

    def _check_int(path: str, min_value: int = None, allow_zero: bool = False):
        value = _require(path, "integer")
        if not _is_int(value):
            _invalid(path, value, "integer")
        if min_value is not None and value < min_value:
            op = f">= {min_value}" if allow_zero else f"> {min_value - 1}"
            _invalid(path, value, f"integer {op}")
        return value

    def _check_number(path: str, predicate, expected: str):
        value = _require(path, expected)
        if not _is_number(value):
            _invalid(path, value, expected)
        if not predicate(float(value)):
            _invalid(path, value, expected)
        return value

    _MISSING = object()

    normalized_context = (context or "train").strip().lower()

    required_common = [
        "reproducibility.seed",
        "dataset.name",
        "dataset.data_dir",
        "dataset.max_walk_length",
        "dataset.num_walks",
        "training.batch_size",
        "model.embedding_dim",
        "model.hidden_dim",
        "model.nhead",
        "model.nlayers",
        "model.dropout",
    ]
    required_train_only = [
        "training.epochs",
        "training.lr",
        "training.weight_decay",
    ]

    required_paths = list(required_common)
    if normalized_context in {"train", "optuna"}:
        required_paths.extend(required_train_only)

    for path in required_paths:
        _require(path, "field to be set")

    _check_int("reproducibility.seed")

    _check_number(
        "dataset.max_walk_length", lambda v: v > 0, "dataset.max_walk_length > 0"
    )
    _check_number("dataset.num_walks", lambda v: v > 0, "dataset.num_walks > 0")
    _check_number("training.batch_size", lambda v: v > 0, "training.batch_size > 0")

    if normalized_context in {"train", "optuna"}:
        _check_number("training.epochs", lambda v: v > 0, "training.epochs > 0")
        _check_number("training.lr", lambda v: v > 0, "training.lr > 0")
        _check_number(
            "training.weight_decay",
            lambda v: v >= 0,
            "training.weight_decay >= 0",
        )

    _check_number("model.dropout", lambda v: 0.0 <= v < 1.0, "0.0 <= dropout < 1.0")
    _check_number("model.embedding_dim", lambda v: v > 0, "embedding_dim > 0")
    _check_number("model.hidden_dim", lambda v: v > 0, "hidden_dim > 0")
    _check_number("model.nhead", lambda v: v > 0, "nhead > 0")
    _check_number("model.nlayers", lambda v: v > 0, "nlayers > 0")

    node_context_mode = _get("model.node_context_mode", _MISSING)
    if node_context_mode is not _MISSING and node_context_mode is not None:
        allowed_modes = {"none", "mask_unscaled", "noise", "replace"}
        if str(node_context_mode) not in allowed_modes:
            _invalid(
                "model.node_context_mode",
                node_context_mode,
                "one of {'none','mask_unscaled','noise','replace'}",
            )

    node_mask_prob = _get("model.node_mask_prob", _MISSING)
    if node_mask_prob is not _MISSING and node_mask_prob is not None:
        if not _is_number(node_mask_prob) or not (0.0 <= float(node_mask_prob) <= 1.0):
            _invalid("model.node_mask_prob", node_mask_prob, "0.0 <= node_mask_prob <= 1.0")

    node_noise_sigma = _get("model.node_noise_sigma", _MISSING)
    if node_noise_sigma is not _MISSING and node_noise_sigma is not None:
        if not _is_number(node_noise_sigma) or float(node_noise_sigma) < 0.0:
            _invalid("model.node_noise_sigma", node_noise_sigma, "node_noise_sigma >= 0.0")

    local_attention_window = _get("model.local_attention_window", _MISSING)
    if local_attention_window is not _MISSING and local_attention_window is not None:
        if not _is_int(local_attention_window) or int(local_attention_window) < 0:
            _invalid(
                "model.local_attention_window",
                local_attention_window,
                "null or non-negative integer",
            )

    node_replace_prob = _get("model.node_replace_prob", _MISSING)
    if node_replace_prob is not _MISSING and node_replace_prob is not None:
        if not _is_number(node_replace_prob) or not (0.0 <= float(node_replace_prob) <= 1.0):
            _invalid(
                "model.node_replace_prob",
                node_replace_prob,
                "0.0 <= node_replace_prob <= 1.0",
            )

    node_replace_unk_ratio = _get("model.node_replace_unk_ratio", _MISSING)
    if node_replace_unk_ratio is not _MISSING and node_replace_unk_ratio is not None:
        if not _is_number(node_replace_unk_ratio) or not (0.0 <= float(node_replace_unk_ratio) <= 1.0):
            _invalid(
                "model.node_replace_unk_ratio",
                node_replace_unk_ratio,
                "0.0 <= node_replace_unk_ratio <= 1.0",
            )

    dynamic_train_masking = _get("model.dynamic_train_masking", _MISSING)
    if dynamic_train_masking is not _MISSING and not isinstance(dynamic_train_masking, bool):
        _invalid("model.dynamic_train_masking", dynamic_train_masking, "boolean")

    for _ablation_flag in (
        "zero_node_tokens",
        "zero_edge_tokens",
        "mask_node_tokens",
        "mask_edge_tokens",
        "randomize_walk_direction",
    ):
        _val = _get(f"model.{_ablation_flag}", _MISSING)
        if _val is not _MISSING and not isinstance(_val, bool):
            _invalid(f"model.{_ablation_flag}", _val, "boolean")

    dynamic_train_mask_seed_offset = _get("model.dynamic_train_mask_seed_offset", _MISSING)
    if dynamic_train_mask_seed_offset is not _MISSING and dynamic_train_mask_seed_offset is not None:
        if not _is_int(dynamic_train_mask_seed_offset):
            _invalid(
                "model.dynamic_train_mask_seed_offset",
                dynamic_train_mask_seed_offset,
                "integer",
            )

    train_ratio = _get("dataset.train_ratio", _MISSING)
    mask_ratio = _get("dataset.mask_ratio", _MISSING)
    val_ratio = _get("dataset.val_ratio", _MISSING)
    test_ratio = _get("dataset.test_ratio", _MISSING)
    ratios = {
        "dataset.train_ratio": train_ratio,
        "dataset.mask_ratio": mask_ratio,
        "dataset.val_ratio": val_ratio,
        "dataset.test_ratio": test_ratio,
    }
    if any(v is not _MISSING for v in ratios.values()):
        total = 0.0
        for path, value in ratios.items():
            if value is _MISSING:
                _invalid(
                    path,
                    value,
                    "split ratio field to be set when any ratio is provided",
                )
            if not _is_number(value):
                _invalid(path, value, "numeric split ratio")
            total += float(value)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Invalid config: "
                f"dataset.split_sum={total} "
                "(expected train_ratio + mask_ratio + val_ratio + test_ratio == 1.0 ± 1e-6)"
            )

    embedding_dim = int(_get("model.embedding_dim"))
    nhead = int(_get("model.nhead"))
    if nhead <= 0 or embedding_dim % nhead != 0:
        _invalid(
            "model.embedding_dim",
            embedding_dim,
            f"embedding_dim divisible by model.nhead ({nhead})",
        )

    use_cuda = _get("training.use_cuda", _MISSING)
    if use_cuda is not _MISSING:
        if not isinstance(use_cuda, bool):
            _invalid("training.use_cuda", use_cuda, "boolean")
        if use_cuda and not torch.cuda.is_available():
            print(
                "⚠ Config warning: training.use_cuda=True but CUDA is not available. "
                "Execution will fall back to CPU."
            )

    preprocess_use_cache = _get("preprocess.use_cache", _MISSING)
    if preprocess_use_cache is not _MISSING and not isinstance(
        preprocess_use_cache, bool
    ):
        _invalid("preprocess.use_cache", preprocess_use_cache, "boolean")

    preprocess_save = _get("preprocess.save", _MISSING)
    if preprocess_save is not _MISSING and not isinstance(preprocess_save, bool):
        _invalid("preprocess.save", preprocess_save, "boolean")

    walk_strategy = _get("dataset.walk_strategy", _MISSING)
    if walk_strategy is not _MISSING and walk_strategy is not None:
        _allowed_walk_strategies = {
            "uniform", "guaranteed", "neg_emphasis", "inv_degree", "node2vec",
            "edge_seeded", "neg_traversal", "set_cover", "cov_restart",
            "sign_alt", "smart", "k_cover", "k_cover_bp", "edge_cover",
        }
        if str(walk_strategy) not in _allowed_walk_strategies:
            _invalid(
                "dataset.walk_strategy",
                walk_strategy,
                f"one of {sorted(_allowed_walk_strategies)}",
            )

    walk_neg_emphasis_fraction = _get("dataset.walk_neg_emphasis_fraction", _MISSING)
    if walk_neg_emphasis_fraction is not _MISSING and walk_neg_emphasis_fraction is not None:
        if not _is_number(walk_neg_emphasis_fraction) or not (
            0.0 <= float(walk_neg_emphasis_fraction) <= 1.0
        ):
            _invalid(
                "dataset.walk_neg_emphasis_fraction",
                walk_neg_emphasis_fraction,
                "0.0 <= walk_neg_emphasis_fraction <= 1.0",
            )

    walk_set_cover_multiplier = _get("dataset.walk_set_cover_multiplier", _MISSING)
    if walk_set_cover_multiplier is not _MISSING and walk_set_cover_multiplier is not None:
        if not _is_number(walk_set_cover_multiplier) or float(walk_set_cover_multiplier) < 1.0:
            _invalid(
                "dataset.walk_set_cover_multiplier",
                walk_set_cover_multiplier,
                "walk_set_cover_multiplier >= 1.0",
            )

    for param in ("walk_p", "walk_q"):
        val = _get(f"dataset.{param}", _MISSING)
        if val is not _MISSING and val is not None:
            if not _is_number(val) or float(val) <= 0.0:
                _invalid(f"dataset.{param}", val, f"{param} > 0.0")

    class_weights = None
    class_weights_path = None
    for candidate in ("training.class_weights", "model.class_weights"):
        value = _get(candidate, _MISSING)
        if value is not _MISSING and value is not None:
            class_weights = value
            class_weights_path = candidate
            break

    num_classes = None
    num_classes_path = None
    for candidate in (
        "dataset.num_classes",
        "model.num_classes",
        "training.num_classes",
    ):
        value = _get(candidate, _MISSING)
        if value is not _MISSING and value is not None:
            num_classes = value
            num_classes_path = candidate
            break

    if class_weights is not None and num_classes is not None:
        if not _is_int(num_classes) or int(num_classes) <= 0:
            _invalid(num_classes_path, num_classes, "positive integer")
        if not isinstance(class_weights, (list, tuple, ListConfig)):
            _invalid(class_weights_path, class_weights, "list/tuple of class weights")
        if len(class_weights) != int(num_classes):
            raise ValueError(
                "Invalid config: "
                f"{class_weights_path}=len({len(class_weights)}) "
                f"(expected length == {num_classes_path}={int(num_classes)})"
            )
