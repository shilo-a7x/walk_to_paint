#!/usr/bin/env bash
set -euo pipefail

# Sync optuna.initial_params to match dataset/model/training values
# for the dataset-specific configs.

ROOT_DIR="/home/dsi/shilo_avital/yolo_lab/walk_to_paint"
CONFIG_DIR="$ROOT_DIR/configs"

datasets=("wiki-rfa" "epinions" "slashdot090221")

for ds in "${datasets[@]}"; do
  cfg="$CONFIG_DIR/$ds.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "Missing config: $cfg" >&2
    exit 1
  fi

  python - <<'PY'
import sys
import yaml
from pathlib import Path

cfg_path = Path(sys.argv[1])

with cfg_path.open("r") as f:
    data = yaml.safe_load(f)

if data is None:
    raise SystemExit(f"Empty config: {cfg_path}")

# Ensure sections exist
for key in ("dataset", "model", "training"):
    if key not in data:
        raise SystemExit(f"Missing '{key}' section in {cfg_path}")

data.setdefault("optuna", {})
optuna = data["optuna"]
optuna.setdefault("initial_params", {})
init = optuna["initial_params"]

# Sync dataset params
init["dataset"] = {
    "max_walk_length": data["dataset"].get("max_walk_length"),
    "num_walks": data["dataset"].get("num_walks"),
}

# Sync training params
init["training"] = {
    "lr": data["training"].get("lr"),
    "weight_decay": data["training"].get("weight_decay"),
    "batch_size": data["training"].get("batch_size"),
    "gradient_clip_val": data["training"].get("gradient_clip_val"),
    "early_stopping_patience": data["training"].get("early_stopping_patience"),
    "epochs": data["training"].get("epochs"),
}

# Sync model params
init["model"] = {
    "embedding_dim": data["model"].get("embedding_dim"),
    "hidden_dim": data["model"].get("hidden_dim"),
    "nhead": data["model"].get("nhead"),
    "nlayers": data["model"].get("nlayers"),
    "dropout": data["model"].get("dropout"),
}

with cfg_path.open("w") as f:
    yaml.safe_dump(data, f, sort_keys=False)

print(f"Synced optuna.initial_params in {cfg_path}")
PY
"$cfg"

done

echo "Done. All dataset configs updated."
