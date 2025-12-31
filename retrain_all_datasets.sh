#!/bin/bash
# Retrain all 3 datasets with best Optuna hyperparameters, one per GPU device
# Each uses cached preprocessing and optimized walk sampling (8 workers)

cd /home/dsi/shilo_avital/yolo_lab/walk_to_paint

PYTHON=./.venv/bin/python

echo "Starting retraining on 3 devices..."
echo ""

# Device 0: wiki-rfa (Trial #87: AUC=0.7779)
echo "📊 Device 0: wiki-rfa (Trial #87)"
nohup $PYTHON run.py \
  --device 0 \
  dataset.name=wiki-rfa \
  preprocess.save=true \
  preprocess.use_cache=true \
  > outputs/wiki-rfa/retrain_device0_nohup.log 2>&1 &
WIKI_PID=$!
echo "  Started with PID $WIKI_PID"

# Device 1: epinions (Using best available from last 10 trials; best trial #31 checkpoint deleted)
echo "📊 Device 1: epinions (Using approximate best params)"
nohup $PYTHON run.py \
  --device 1 \
  dataset.name=epinions \
  preprocess.save=true \
  preprocess.use_cache=true \
  > outputs/epinions/retrain_device1_nohup.log 2>&1 &
EPINIONS_PID=$!
echo "  Started with PID $EPINIONS_PID"

# Device 2: slashdot090221 (Trial #0: AUC=0.6853)
echo "📊 Device 2: slashdot090221 (Trial #0)"
nohup $PYTHON run.py \
  --device 2 \
  dataset.name=slashdot090221 \
  preprocess.save=true \
  preprocess.use_cache=true \
  > outputs/slashdot090221/retrain_device2_nohup.log 2>&1 &
SLASHDOT_PID=$!
echo "  Started with PID $SLASHDOT_PID"

echo ""
echo "✅ All retraining jobs started!"
echo ""
echo "Monitor logs with:"
echo "  tail -f outputs/wiki-rfa/retrain_device0_nohup.log"
echo "  tail -f outputs/epinions/retrain_device1_nohup.log"
echo "  tail -f outputs/slashdot090221/retrain_device2_nohup.log"
echo ""
echo "Or watch GPU usage:"
echo "  nvidia-smi -l 1"
echo "  or nvtop"
