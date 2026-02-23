#!/bin/bash
# Task 3 Production Retraining Script
# Retrain all 3 datasets with best Optuna hyperparameters + Task 3 callbacks
# Per-epoch prediction saving enabled for all splits (train/val/test)

set -e  # Exit on error

cd /home/dsi/shilo_avital/yolo_lab/walk_to_paint

PYTHON=python
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "════════════════════════════════════════════════════════════════════"
echo "  Task 3 Production Retraining - Started at $(date)"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  - Seed: 42 (from config.yaml)"
echo "  - Callbacks: PerEpochPredictionSaver + PerEpochTestRunner"
echo "  - Predictions saved: All epochs, all splits"
echo "  - TensorBoard: Full logging (metrics + plots)"
echo ""

# Create nohup output directory
mkdir -p nohup_logs

# Device 3: wiki-rfa (Binary, 20 epochs, Trial #87: AUC=0.7779)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Device 3: wiki-rfa"
echo "  - Dataset: Binary classification (2 classes)"
echo "  - Epochs: 20"
echo "  - Best trial: #87 (AUC=0.7779)"
echo "  - Multiedge: most_recent"
echo "  - Self-loops: removed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

nohup $PYTHON run.py \
  --config config.yaml \
  --device 3 \
  dataset.name=wiki-rfa \
  preprocess.save=true \
  preprocess.use_cache=true \
  > nohup_logs/wiki-rfa_${TIMESTAMP}.log 2>&1 &

WIKI_PID=$!
echo "✓ Started with PID $WIKI_PID"
echo "  Log: nohup_logs/wiki-rfa_${TIMESTAMP}.log"
echo ""

# Give it a moment to initialize
sleep 2

# Device 1: epinions (3-class, 25 epochs, Trial #31: AUC=0.9133)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Device 1: epinions"
echo "  - Dataset: 3-class classification"
echo "  - Epochs: 25"
echo "  - Best trial: #31 (AUC=0.9133)"
echo "  - Multiedge: keep"
echo "  - Self-loops: removed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

nohup $PYTHON run.py \
  --config config.yaml \
  --device 1 \
  dataset.name=epinions \
  preprocess.save=true \
  preprocess.use_cache=true \
  > nohup_logs/epinions_${TIMESTAMP}.log 2>&1 &

EPINIONS_PID=$!
echo "✓ Started with PID $EPINIONS_PID"
echo "  Log: nohup_logs/epinions_${TIMESTAMP}.log"
echo ""

# Give it a moment to initialize
sleep 2

# Device 2: slashdot090221 (3-class, 7 epochs, Trial #12: AUC=0.8529)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Device 2: slashdot090221"
echo "  - Dataset: 3-class classification"
echo "  - Epochs: 7"
echo "  - Best trial: #12 (AUC=0.8529)"
echo "  - Multiedge: keep"
echo "  - Self-loops: removed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

nohup $PYTHON run.py \
  --config config.yaml \
  --device 2 \
  dataset.name=slashdot090221 \
  preprocess.save=true \
  preprocess.use_cache=true \
  > nohup_logs/slashdot090221_${TIMESTAMP}.log 2>&1 &

SLASHDOT_PID=$!
echo "✓ Started with PID $SLASHDOT_PID"
echo "  Log: nohup_logs/slashdot090221_${TIMESTAMP}.log"
echo ""

echo "════════════════════════════════════════════════════════════════════"
echo "✅ All retraining jobs launched!"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "PIDs:"
echo "  wiki-rfa:        $WIKI_PID (device 3)"
echo "  epinions:        $EPINIONS_PID (device 1)"
echo "  slashdot090221:  $SLASHDOT_PID (device 2)"
echo ""
echo "Monitoring Commands:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "# Watch training progress (last 30 lines, auto-refresh):"
echo "  watch -n 5 'tail -30 nohup_logs/wiki-rfa_${TIMESTAMP}.log'"
echo "  watch -n 5 'tail -30 nohup_logs/epinions_${TIMESTAMP}.log'"
echo "  watch -n 5 'tail -30 nohup_logs/slashdot090221_${TIMESTAMP}.log'"
echo ""
echo "# Tail logs continuously:"
echo "  tail -f nohup_logs/wiki-rfa_${TIMESTAMP}.log"
echo "  tail -f nohup_logs/epinions_${TIMESTAMP}.log"
echo "  tail -f nohup_logs/slashdot090221_${TIMESTAMP}.log"
echo ""
echo "# Check GPU usage:"
echo "  nvidia-smi -l 1"
echo "  nvtop"
echo ""
echo "# Check process status:"
echo "  ps aux | grep run.py"
echo ""
echo "# Check disk space (predictions can be large):"
echo "  du -sh outputs/*/"
echo "  df -h ."
echo ""
echo "# TensorBoard (after some epochs):"
echo "  tensorboard --logdir outputs/wiki-rfa/ --port 6006"
echo "  tensorboard --logdir outputs/epinions/ --port 6007"
echo "  tensorboard --logdir outputs/slashdot090221/ --port 6008"
echo ""
echo "Estimated Completion Times:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  wiki-rfa:       ~6-8 hours (20 epochs, binary)"
echo "  epinions:       ~10-12 hours (25 epochs, 3-class, large)"
echo "  slashdot090221: ~4-6 hours (7 epochs, 3-class)"
echo ""
echo "Expected Outputs (per dataset):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  outputs/{dataset}/{exp_name}_{timestamp}/"
echo "    ├── checkpoints/"
echo "    │   ├── {dataset}_predictions/"
echo "    │   │   ├── epoch_000/{train,val,test}_predictions.pkl"
echo "    │   │   ├── epoch_001/..."
echo "    │   │   └── epoch_N/..."
echo "    │   └── {dataset}-{exp}-epoch={best}-val_loss={X.XX}.ckpt"
echo "    └── logs/"
echo "        └── {dataset}-{exp}/version_0/events.out.tfevents.*"
echo ""
echo "════════════════════════════════════════════════════════════════════"
