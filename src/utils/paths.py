from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


def resolve_outputs_dirs(
    cfg: Any, make_dirs: bool = True, base_outputs_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Resolve and create (optional) dataset-scoped output directories.

    Layout (default):
      <base_outputs_dir>/<dataset.name>/<exp_name>[_<timestamp>]/
         checkpoints/
         logs/
         optuna/
         plots/

    This updates `cfg.training.checkpoint_dir` and `cfg.training.log_dir`
    so downstream code will use the new locations.

    Returns a dict with the resolved paths.
    """

    base = (
        base_outputs_dir
        or (getattr(cfg, "paths", {}) or {}).get("base_outputs_dir")
        or "outputs"
    )

    # Defensive access for OmegaConf objects
    try:
        use_dataset_outputs = cfg.paths.get("use_dataset_outputs", True)
    except Exception:
        use_dataset_outputs = True

    try:
        append_ts = cfg.paths.get("append_timestamp", True)
    except Exception:
        append_ts = True

    dataset_name = getattr(getattr(cfg, "dataset", {}), "name", "default")
    exp_name = getattr(getattr(cfg, "training", {}), "exp_name", "experiment")

    if append_ts:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_folder = f"{exp_name}_{ts}"
    else:
        exp_folder = exp_name

    if use_dataset_outputs and dataset_name:
        exp_dir = Path(base) / dataset_name / exp_folder
    else:
        exp_dir = Path(base) / exp_folder

    checkpoint_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"
    optuna_dir = exp_dir / "optuna"
    plots_dir = exp_dir / "plots"

    if make_dirs:
        for p in (checkpoint_dir, log_dir, optuna_dir, plots_dir):
            p.mkdir(parents=True, exist_ok=True)

    # Update cfg so training and downstream code use the new dirs
    try:
        cfg.training.checkpoint_dir = str(checkpoint_dir)
    except Exception:
        pass
    try:
        cfg.training.log_dir = str(log_dir)
    except Exception:
        pass

    return {
        "exp_dir": str(exp_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "log_dir": str(log_dir),
        "optuna_dir": str(optuna_dir),
        "plots_dir": str(plots_dir),
    }
