#!/bin/bash
# Quick launch script for Optuna hyperparameter optimization

# Default values
DEVICE=${1:-2}
N_TRIALS=${2:-50}
EXP_NAME=${3:-"optuna_search"}

echo "🚀 Launching Optuna hyperparameter optimization"
echo "Device: $DEVICE"
echo "Trials: $N_TRIALS" 
echo "Experiment: $EXP_NAME"

# Activate virtual environment
source .venv/bin/activate

# Run Optuna with comprehensive hyperparameter search
nohup python -u optuna_run.py \
    --device $DEVICE \
    --n-trials $N_TRIALS \
    training.exp_name=$EXP_NAME \
    training.epochs=30 \
    > optuna_${EXP_NAME}.out 2>&1 &

echo "📝 Output will be saved to: optuna_${EXP_NAME}.out"
echo "🔍 Monitor progress with: tail -f optuna_${EXP_NAME}.out"
echo "📊 View TensorBoard: tensorboard --logdir logs/ --port 6006"
echo "✅ Optuna optimization started in background!"
