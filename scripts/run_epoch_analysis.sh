#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dsi/shilo_avital/yolo_lab/walk_to_paint"
PY="$ROOT/.venv/bin/python"
RUN_ID="epoch_analysis_$(date +%Y%m%d_%H%M%S)"

# Best runs found by test AUC
WIKI_RUN="outputs/wiki-rfa/wiki-rfa-run_20251225-094820"
SLASH_RUN="outputs/slashdot090221/slashdot090221-run_20251224-214709"
EPI_RUN="outputs/epinions/epinions-run_20251224-214709"

LOG_DIR="$ROOT/outputs/epoch_analysis_logs"
mkdir -p "$LOG_DIR"

echo "Starting comprehensive epoch analysis..."
echo "RUN_ID: $RUN_ID"
echo ""

# Function to process a single dataset (all checkpoints sequentially)
process_dataset() {
  local dataset=$1
  local run_dir=$2
  local device=$3
  
  echo "============================================================"
  echo "Processing $dataset (GPU $device)"
  echo "============================================================"
  
  # Find all checkpoints
  ckpt_dir="$run_dir/checkpoints"
  if [ ! -d "$ckpt_dir" ]; then
    echo "WARNING: No checkpoints found for $dataset"
    return
  fi
  
  # Process each checkpoint SEQUENTIALLY for this dataset
  for ckpt in "$ckpt_dir"/*.ckpt; do
    if [ ! -f "$ckpt" ]; then
      continue
    fi
    
    ckpt_name=$(basename "$ckpt" .ckpt)
    echo "  Processing: $ckpt_name"
    
    # Create output directories
    mkdir -p "outputs/predictions/$dataset/raw_scores/${RUN_ID}"
    mkdir -p "outputs/$dataset/epoch_analysis/${RUN_ID}/${ckpt_name}"
    
    # Extract predictions (GPU-bound)
    "$PY" scripts/extract_edge_scores.py \
      --checkpoint "$ckpt" \
      --output "outputs/$dataset/epoch_analysis/${RUN_ID}/${ckpt_name}_predictions.pkl" \
      --save-predictions --preds-only --run-id "${RUN_ID}/${ckpt_name}" \
      --device "$device" --batch-size 1024 \
      training.num_workers=16 training.prefetch_factor=4 \
      training.persistent_workers=true training.pin_memory=true \
      reproducibility.seed=42 \
      > "$LOG_DIR/${dataset}_${ckpt_name}.log" 2>&1
    
    echo "    ✓ Extracted predictions"
    
    # Generate triplets and heatmaps (CPU-bound, fast)
    "$PY" scripts/save_triplets.py \
      --dataset "$dataset" \
      --run-id "${RUN_ID}/${ckpt_name}" \
      --splits val,test
    
    echo "    ✓ Generated triplets"
    
    "$PY" scripts/plot_triplet_heatmap.py \
      --dataset "$dataset" \
      --split val \
      --run-id "${RUN_ID}/${ckpt_name}"
    
    "$PY" scripts/plot_triplet_heatmap.py \
      --dataset "$dataset" \
      --split test \
      --run-id "${RUN_ID}/${ckpt_name}"
    
    echo "    ✓ Generated heatmaps"
  done
  
  echo "  ✓ COMPLETE: All checkpoints processed for $dataset"
  echo ""
}

# Launch all 3 datasets in PARALLEL (in background)
echo "Launching all 3 datasets in parallel..."
echo ""

# Dataset 1: wiki-rfa on GPU 0 (in background)
{
  process_dataset "wiki-rfa" "$WIKI_RUN" "0"
} > "$LOG_DIR/wiki-rfa_full.log" 2>&1 &
PID_WIKI=$!
echo "  PID $PID_WIKI: wiki-rfa (GPU 0)"

# Dataset 2: slashdot on GPU 1 (in background)
{
  process_dataset "slashdot090221" "$SLASH_RUN" "1"
} > "$LOG_DIR/slashdot090221_full.log" 2>&1 &
PID_SLASH=$!
echo "  PID $PID_SLASH: slashdot090221 (GPU 1)"

# Dataset 3: epinions on GPU 2 (in background)
{
  process_dataset "epinions" "$EPI_RUN" "2"
} > "$LOG_DIR/epinions_full.log" 2>&1 &
PID_EPI=$!
echo "  PID $PID_EPI: epinions (GPU 2)"

echo ""
echo "Waiting for all datasets to complete..."
echo ""

# Wait for all to finish
wait $PID_WIKI
echo "✓ wiki-rfa complete"

wait $PID_SLASH
echo "✓ slashdot090221 complete"

wait $PID_EPI
echo "✓ epinions complete"

echo ""
echo "============================================================"
echo "All datasets extracted! Now generating evolution visualizations..."
echo "============================================================"
echo ""

# Generate evolution plots (loss/AUC curves, ROC evolution, heatmap evolution)
"$PY" scripts/plot_epoch_evolution.py --run-id "$RUN_ID"

echo ""
echo "============================================================"
echo "Now running comprehensive heatmap analysis..."
echo "============================================================"
echo ""

# Analyze all heatmaps
"$PY" scripts/analyze_all_heatmaps.py --run-id "$RUN_ID"

echo ""
echo "============================================================"
echo "Done! RUN_ID=${RUN_ID}"
echo "============================================================"
echo "Results in: outputs/{dataset}/epoch_analysis/${RUN_ID}/"
echo ""
echo "Generated outputs:"
echo "  • Heatmaps (PNG): outputs/{dataset}/epoch_analysis/${RUN_ID}/<ckpt>/heatmap_*.png"
echo "  • Raw data (NPZ): outputs/{dataset}/epoch_analysis/${RUN_ID}/<ckpt>/heatmap_*_data.npz"
echo "  • Statistics (JSON): outputs/{dataset}/epoch_analysis/${RUN_ID}/<ckpt>/heatmap_*_stats.json"
echo "  • Evolution plots: outputs/epoch_evolution_${RUN_ID}/"
echo "  • Analysis report: outputs/analysis_report_${RUN_ID}.txt"
echo ""
echo "To replay/customize plots from raw data:"
echo "  python scripts/replay_heatmap.py --data-path <path>_data.npz --output-dir outputs/custom_plots/ --all"
echo ""
echo "Log files:"
echo "  - $LOG_DIR/wiki-rfa_full.log"
echo "  - $LOG_DIR/slashdot090221_full.log"
echo "  - $LOG_DIR/epinions_full.log"
