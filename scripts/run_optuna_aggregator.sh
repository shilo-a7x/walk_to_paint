#!/bin/bash
set -e

source /home/dsi/shilo_avital/yolo_lab/walk_to_paint/.venv/bin/activate

BASE_DIR="/home/dsi/shilo_avital/yolo_lab/walk_to_paint"
FEATURE_DIR="${BASE_DIR}/aggregator_features_new"
RESULTS_DIR="${BASE_DIR}/aggregator_optuna_results"

mkdir -p "${RESULTS_DIR}"

declare -A EXP_DIRS=(
  ["wiki-rfa"]="outputs/wiki-rfa/wiki-rfa-run_20260208-145124"
  ["epinions"]="outputs/epinions/epinions-run_20260208-145126"
  ["slashdot090221"]="outputs/slashdot090221/slashdot090221-run_20260208-145128"
)

for DATASET in "wiki-rfa" "epinions" "slashdot090221"; do
  echo "=============================="
  echo "DATASET: ${DATASET} (Optuna Tuning)"
  echo "=============================="

  EXP_DIR="${BASE_DIR}/${EXP_DIRS[$DATASET]}"
  OUT_FEATURES="${FEATURE_DIR}/${DATASET}_epoch000.pkl"
  OUT_RESULTS="${RESULTS_DIR}/${DATASET}"

  mkdir -p "${OUT_RESULTS}"

  python scripts/agg_optuna_tune.py \
    --input "${OUT_FEATURES}" \
    --output-dir "${OUT_RESULTS}" \
    --n-trials 30
done

echo ""
echo "=============================="
echo "BEST MODELS PER DATASET"
echo "=============================="
python scripts/compare_optuna_results.py
