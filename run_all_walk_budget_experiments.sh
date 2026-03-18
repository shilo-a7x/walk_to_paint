#!/bin/bash
# Run walk-budget AUC experiment on all datasets sequentially

cd /home/dsi/shilo_avital/yolo_lab/walk_to_paint
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=1

DATASETS=("slashdot090221" "epinions" "wiki-rfa")

for dataset in "${DATASETS[@]}"; do
    echo "====================================================================="
    echo "Starting experiment for dataset: $dataset"
    echo "Time: $(date)"
    echo "====================================================================="
    
    .venv/bin/python scripts/run_walk_budget_auc_experiment.py \
        --dataset "$dataset" \
        --seed 42 \
        --epochs 5 \
        --num-walks "500000,1000000,2000000,3500000,5000000" \
        --max-walk-lengths "80" \
        --top-k 2 \
        --output-root outputs/walk_budget_auc \
        --tmp-root tmp/walk_budget_auc
    
    exit_code=$?
    echo "====================================================================="
    echo "Completed experiment for dataset: $dataset (exit code: $exit_code)"
    echo "Time: $(date)"
    echo "====================================================================="
    echo ""
done

echo "====================================================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "Time: $(date)"
echo "====================================================================="
