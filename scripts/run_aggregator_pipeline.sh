#!/bin/bash
# Run the full aggregator pipeline for all datasets

set -e

# Configuration
BASE_DIR="/home/dsi/shilo_avital/yolo_lab/walk_to_paint"
FEATURES_DIR="${BASE_DIR}/aggregator_features"
RESULTS_DIR="${BASE_DIR}/aggregator_results"

mkdir -p ${FEATURES_DIR}
mkdir -p ${RESULTS_DIR}

# Dataset configurations
declare -A DATASETS=(
    ["epinions"]="outputs/epinions/epinions-run_20251224-214709/checkpoints/epinions-epinions-run-epoch=15-val_loss=1.25.ckpt"
    ["slashdot090221"]="outputs/slashdot090221/slashdot090221-run_20251224-214709/checkpoints/slashdot090221-slashdot090221-run-epoch=06-val_loss=2.22.ckpt"
    ["wiki-rfa"]="checkpoints/bitcoin_alpha-walk_to_paint_experiment-epoch=00-val_loss=2.09.ckpt"
)

# Device assignment
DEVICE=1

echo "======================================"
echo "Aggregator Pipeline"
echo "======================================"
echo ""

# Process each dataset
for DATASET in "${!DATASETS[@]}"; do
    echo "Processing dataset: ${DATASET}"
    echo "--------------------------------------"
    
    CHECKPOINT="${DATASETS[$DATASET]}"
    FEATURES_FILE="${FEATURES_DIR}/${DATASET}_features.pkl"
    OUTPUT_DIR="${RESULTS_DIR}/${DATASET}"
    
    # Step 1: Extract edge scores
    echo "[1/3] Extracting edge scores..."
    python scripts/extract_edge_scores.py \
        --config config.yaml \
        --checkpoint "${CHECKPOINT}" \
        --output "${FEATURES_FILE}" \
        --device ${DEVICE} \
        dataset.name=${DATASET}
    
    echo "Features saved to: ${FEATURES_FILE}"
    echo ""
    
    # Step 2: Train logistic regression aggregator
    echo "[2/3] Training logistic regression aggregator..."
    python scripts/train_aggregator.py \
        --features "${FEATURES_FILE}" \
        --output_dir "${OUTPUT_DIR}/logistic" \
        --model logistic
    
    echo ""
    
    # Step 3: Train MLP aggregator
    echo "[3/3] Training MLP aggregator..."
    python scripts/train_aggregator.py \
        --features "${FEATURES_FILE}" \
        --output_dir "${OUTPUT_DIR}/mlp" \
        --model mlp
    
    echo ""
    echo "======================================"
    echo "Completed: ${DATASET}"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "======================================"
    echo ""
done

echo ""
echo "======================================"
echo "All datasets processed!"
echo "======================================"
echo ""
echo "Summary of results:"
for DATASET in "${!DATASETS[@]}"; do
    LOGISTIC_SUMMARY="${RESULTS_DIR}/${DATASET}/logistic/summary.txt"
    MLP_SUMMARY="${RESULTS_DIR}/${DATASET}/mlp/summary.txt"
    
    echo ""
    echo "--- ${DATASET} ---"
    if [ -f "${LOGISTIC_SUMMARY}" ]; then
        echo "Logistic Regression:"
        grep "Test AUC" "${LOGISTIC_SUMMARY}" || echo "  (results not found)"
    fi
    if [ -f "${MLP_SUMMARY}" ]; then
        echo "MLP:"
        grep "Test AUC" "${MLP_SUMMARY}" || echo "  (results not found)"
    fi
done

echo ""
echo "Full results available in: ${RESULTS_DIR}"
