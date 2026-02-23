#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dsi/shilo_avital/yolo_lab/walk_to_paint"
PY="$ROOT/.venv/bin/python"
RUN_ID="t3_full_$(date +%Y%m%d_%H%M%S)"

# Checkpoints (from outputs dirs)
WIKI_CKPT="outputs/wiki-rfa/wiki-rfa-run_20251225-094820/checkpoints/wiki-rfa-wiki-rfa-run-epoch=00-val_loss=0.76.ckpt"
SLASH_CKPT="outputs/slashdot090221/slashdot090221-run_20251224-214709/checkpoints/slashdot090221-slashdot090221-run-epoch=00-val_loss=0.83.ckpt"
EPI_CKPT="outputs/epinions/epinions-run_20251225-101336/checkpoints/epinions-epinions-run-epoch=00-val_loss=0.59.ckpt"

LOG_DIR="$ROOT/outputs/t3_full_logs"
mkdir -p "$LOG_DIR"

# 1) Save per-occurrence predictions (val/test) on separate GPUs
nohup "$PY" scripts/extract_edge_scores.py \
  --checkpoint "$WIKI_CKPT" \
  --output outputs/wiki-rfa/edge_scores/${RUN_ID}.pkl \
  --save-predictions --preds-only --run-id "$RUN_ID" \
  --device 0 --batch-size 1024 \
  training.num_workers=16 training.prefetch_factor=4 \
  training.persistent_workers=true training.pin_memory=true \
  reproducibility.seed=42 \
  > "$LOG_DIR/wiki-rfa_${RUN_ID}.log" 2>&1 &

nohup "$PY" scripts/extract_edge_scores.py \
  --checkpoint "$SLASH_CKPT" \
  --output outputs/slashdot090221/edge_scores/${RUN_ID}.pkl \
  --save-predictions --preds-only --run-id "$RUN_ID" \
  --device 1 --batch-size 1024 \
  training.num_workers=16 training.prefetch_factor=4 \
  training.persistent_workers=true training.pin_memory=true \
  reproducibility.seed=42 \
  > "$LOG_DIR/slashdot090221_${RUN_ID}.log" 2>&1 &

nohup "$PY" scripts/extract_edge_scores.py \
  --checkpoint "$EPI_CKPT" \
  --output outputs/epinions/edge_scores/${RUN_ID}.pkl \
  --save-predictions --preds-only --run-id "$RUN_ID" \
  --device 2 --batch-size 1024 \
  training.num_workers=16 training.prefetch_factor=4 \
  training.persistent_workers=true training.pin_memory=true \
  reproducibility.seed=42 \
  > "$LOG_DIR/epinions_${RUN_ID}.log" 2>&1 &

wait

# 2) Save triplets
"$PY" scripts/save_triplets.py --dataset wiki-rfa --run-id "$RUN_ID" --splits val,test
"$PY" scripts/save_triplets.py --dataset slashdot090221 --run-id "$RUN_ID" --splits val,test
"$PY" scripts/save_triplets.py --dataset epinions --run-id "$RUN_ID" --splits val,test

# 3) Plot heatmaps
"$PY" scripts/plot_triplet_heatmap.py --dataset wiki-rfa --split val
"$PY" scripts/plot_triplet_heatmap.py --dataset wiki-rfa --split test
"$PY" scripts/plot_triplet_heatmap.py --dataset slashdot090221 --split val
"$PY" scripts/plot_triplet_heatmap.py --dataset slashdot090221 --split test
"$PY" scripts/plot_triplet_heatmap.py --dataset epinions --split val
"$PY" scripts/plot_triplet_heatmap.py --dataset epinions --split test

echo "Done. RUN_ID=${RUN_ID}"
