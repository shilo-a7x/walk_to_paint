from pathlib import Path
from typing import List, Optional
from omegaconf import OmegaConf


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
