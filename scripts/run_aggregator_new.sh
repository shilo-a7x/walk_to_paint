#!/bin/bash
set -e

BASE_DIR="/home/dsi/shilo_avital/yolo_lab/walk_to_paint"
FEATURE_DIR="${BASE_DIR}/aggregator_features_new"
RESULTS_DIR="${BASE_DIR}/aggregator_results_new"

mkdir -p "${FEATURE_DIR}" "${RESULTS_DIR}"

declare -A EXP_DIRS=(
  ["wiki-rfa"]="outputs/wiki-rfa/wiki-rfa-run_20260208-145124"
  ["epinions"]="outputs/epinions/epinions-run_20260208-145126"
  ["slashdot090221"]="outputs/slashdot090221/slashdot090221-run_20260208-145128"
)

for DATASET in "wiki-rfa" "epinions" "slashdot090221"; do
  echo "=============================="
  echo "DATASET: ${DATASET}"
  echo "=============================="

  EXP_DIR="${BASE_DIR}/${EXP_DIRS[$DATASET]}"
  OUT_FEATURES="${FEATURE_DIR}/${DATASET}_epoch000.pkl"
  OUT_RESULTS="${RESULTS_DIR}/${DATASET}"

  python scripts/agg_build_features.py \
    --dataset "${DATASET}" \
    --exp-dir "${EXP_DIR}" \
    --epoch 0 \
    --output "${OUT_FEATURES}"

  python scripts/agg_train_edge.py \
    --input "${OUT_FEATURES}" \
    --output-dir "${OUT_RESULTS}/logistic" \
    --model logistic

  python scripts/agg_train_edge.py \
    --input "${OUT_FEATURES}" \
    --output-dir "${OUT_RESULTS}/xgboost" \
    --model xgboost

  python scripts/agg_train_edge.py \
    --input "${OUT_FEATURES}" \
    --output-dir "${OUT_RESULTS}/lgbm" \
    --model lgbm
done

echo ""
echo "=============================="
echo "COMPARISON: Aggregator vs Transformer"
echo "=============================="
python scripts/compare_aggregator.py