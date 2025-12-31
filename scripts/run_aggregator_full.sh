#!/bin/bash
# Master aggregator pipeline script with reproducibility and comparison
# Logs everything and produces a summary comparing transformer vs aggregator AUCs

set -e

# Activate virtual environment
source /home/dsi/shilo_avital/yolo_lab/walk_to_paint/.venv/bin/activate

# Configuration
BASE_DIR="/home/dsi/shilo_avital/yolo_lab/walk_to_paint"
FEATURES_DIR="${BASE_DIR}/aggregator_features"
RESULTS_DIR="${BASE_DIR}/aggregator_results"
LOGS_DIR="${BASE_DIR}/aggregator_logs"
RUN_LOG="${BASE_DIR}/aggregator_run.log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="aggregator_run_${TIMESTAMP}"

mkdir -p "${FEATURES_DIR}" "${RESULTS_DIR}" "${LOGS_DIR}"

# Transformer checkpoint paths (from 20251224 runs)
declare -A TRANSFORMER_CKPTS=(
    ["epinions"]="outputs/epinions/epinions-run_20251224-214709/checkpoints/epinions-epinions-run-epoch=15-val_loss=1.25.ckpt"
    ["slashdot090221"]="outputs/slashdot090221/slashdot090221-run_20251224-214709/checkpoints/slashdot090221-slashdot090221-run-epoch=06-val_loss=2.22.ckpt"
    ["wiki-rfa"]="outputs/wiki-rfa/wiki-rfa-run_20251225-*/checkpoints/wiki-rfa-wiki-rfa-run-epoch=*.ckpt"
)

# Transformer metrics (from eval-only runs)
# Data gathered from retrain logs:
#   epinions: val=0.8933, test=0.8852 (from conversation history)
#   slashdot: val_best=0.786 (epoch 6), test=0.8029
#   wiki-rfa: val_best=0.809, test=0.8198 (binary mode retrain)
declare -A TRANSFORMER_VAL_AUC=(
    ["epinions"]="0.8933"
    ["slashdot090221"]="0.7860"
    ["wiki-rfa"]="0.8090"
)

declare -A TRANSFORMER_TEST_AUC=(
    ["epinions"]="0.8852"
    ["slashdot090221"]="0.8029"
    ["wiki-rfa"]="0.8198"
)

# Per-dataset GPU assignment (fallback to 0 if missing)
declare -A DATASET_DEVICES=(
    ["wiki-rfa"]=0
    ["epinions"]=1
    ["slashdot090221"]=2
)

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         AGGREGATOR PIPELINE - Full Run                    ║"
echo "║         Run ID: ${RUN_ID}                   ║"
echo "║         Seed: 42 (reproducible)                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Log file for this run
MASTER_LOG="${LOGS_DIR}/${RUN_ID}_master.log"
touch "${MASTER_LOG}"

echo "Master log: ${MASTER_LOG}"
echo ""

# Store all results for final comparison
COMPARISON_FILE="${LOGS_DIR}/${RUN_ID}_comparison.csv"
cat > "${COMPARISON_FILE}" << 'EOF'
Dataset,Model,Agg_Train_AUC,Agg_Val_AUC,Agg_Test_AUC,Transformer_Val_AUC,Transformer_Test_AUC,Delta_Val,Delta_Test
EOF

# Process each dataset in parallel with per-dataset GPU assignment
pids=()
for DATASET in "epinions" "slashdot090221" "wiki-rfa"; do
  (
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  Processing: ${DATASET}"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    DATASET_START=$(date +%s)

    DEVICE=${DATASET_DEVICES[$DATASET]:-0}

    # Find actual checkpoint path (wiki-rfa may have wildcard)
    if [[ "${DATASET}" == "wiki-rfa" ]]; then
        CHECKPOINT=$(ls -t ${BASE_DIR}/${TRANSFORMER_CKPTS[$DATASET]} 2>/dev/null | head -1)
    else
        CHECKPOINT="${BASE_DIR}/${TRANSFORMER_CKPTS[$DATASET]}"
    fi

    if [ ! -f "${CHECKPOINT}" ]; then
        echo "⚠️  Checkpoint not found: ${CHECKPOINT}"
        echo "    Skipping ${DATASET}"
        echo ""
        exit 0
    fi

    echo "Checkpoint: ${CHECKPOINT}"
    echo "Using device: ${DEVICE}"
    echo ""

    FEATURES_FILE="${FEATURES_DIR}/${DATASET}_features.pkl"
    DATASET_RESULTS_DIR="${RESULTS_DIR}/${DATASET}"
    DATASET_LOG="${LOGS_DIR}/${RUN_ID}_${DATASET}.log"

    mkdir -p "${DATASET_RESULTS_DIR}"

    {
        echo "[1/3] Extracting edge scores for ${DATASET}..."
        python scripts/extract_edge_scores.py \
            --config config.yaml \
            --checkpoint "${CHECKPOINT}" \
            --output "${FEATURES_FILE}" \
            --device ${DEVICE} \
            dataset.name=${DATASET}

        echo "✓ Features saved to: ${FEATURES_FILE}"
        echo ""

        # Train aggregators (logistic + MLP)
        for MODEL in "logistic" "mlp"; do
            echo "[2/3] Training ${MODEL} aggregator for ${DATASET}..."
            MODEL_OUTPUT_DIR="${DATASET_RESULTS_DIR}/${MODEL}"
            mkdir -p "${MODEL_OUTPUT_DIR}"

            python scripts/train_aggregator.py \
                --features "${FEATURES_FILE}" \
                --output_dir "${MODEL_OUTPUT_DIR}" \
                --model ${MODEL}

            echo "✓ ${MODEL} aggregator saved to: ${MODEL_OUTPUT_DIR}"
            echo ""

            # Extract metrics from summary
            if [ -f "${MODEL_OUTPUT_DIR}/summary.txt" ]; then
                AGG_TRAIN_AUC=$(grep "Agg train set" -A 3 "${MODEL_OUTPUT_DIR}/summary.txt" | grep "AUC:" | awk '{print $2}')
                AGG_VAL_AUC=$(grep "Agg val set" -A 3 "${MODEL_OUTPUT_DIR}/summary.txt" | grep "AUC:" | awk '{print $2}')
                AGG_TEST_AUC=$(grep "Agg test set" -A 3 "${MODEL_OUTPUT_DIR}/summary.txt" | grep "AUC:" | awk '{print $2}')

                TRANSFORMER_VAL=${TRANSFORMER_VAL_AUC[$DATASET]}
                TRANSFORMER_TEST=${TRANSFORMER_TEST_AUC[$DATASET]}

                # Calculate deltas
                if [[ "${TRANSFORMER_VAL}" != "TBD" ]] && [[ -n "${AGG_VAL_AUC}" ]]; then
                    DELTA_VAL=$(echo "${AGG_VAL_AUC} - ${TRANSFORMER_VAL}" | bc 2>/dev/null || echo "N/A")
                else
                    DELTA_VAL="N/A"
                fi

                if [[ "${TRANSFORMER_TEST}" != "TBD" ]] && [[ -n "${AGG_TEST_AUC}" ]]; then
                    DELTA_TEST=$(echo "${AGG_TEST_AUC} - ${TRANSFORMER_TEST}" | bc 2>/dev/null || echo "N/A")
                else
                    DELTA_TEST="N/A"
                fi

                # Append to comparison file
                echo "${DATASET},${MODEL},${AGG_TRAIN_AUC:-N/A},${AGG_VAL_AUC:-N/A},${AGG_TEST_AUC:-N/A},${TRANSFORMER_VAL},${TRANSFORMER_TEST},${DELTA_VAL},${DELTA_TEST}" >> "${COMPARISON_FILE}"
            fi
        done

        DATASET_END=$(date +%s)
        DATASET_TIME=$((DATASET_END - DATASET_START))

        echo "╔════════════════════════════════════════════════════════════╗"
        echo "║  Completed: ${DATASET} (${DATASET_TIME}s)"
        echo "╚════════════════════════════════════════════════════════════╝"
        echo ""
    } | tee -a "${DATASET_LOG}"
  ) &
  pids+=($!)
done

# Wait for all datasets
for pid in "${pids[@]}"; do
  wait "$pid"
done

# ============================================================
# Final Summary
# ============================================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║             FINAL COMPARISON: Transformer vs Aggregator    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "${COMPARISON_FILE}" ]; then
    echo "Results saved to: ${COMPARISON_FILE}"
    echo ""
    cat "${COMPARISON_FILE}"
    echo ""
fi

echo "Full logs:"
echo "  Master log:  ${MASTER_LOG}"
echo "  Dataset logs: ${LOGS_DIR}/${RUN_ID}_*.log"
echo ""
echo "Results:"
echo "  Features:    ${FEATURES_DIR}/"
echo "  Models:      ${RESULTS_DIR}/"
echo "  Logs:        ${LOGS_DIR}/"
echo ""

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Run ID: ${RUN_ID}"
echo "║  Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "║  Status: ✓ COMPLETE"
echo "╚════════════════════════════════════════════════════════════╝"
