#!/bin/bash

###############################################################################
# Script: launch_optuna_all.sh
# Purpose: Launch Optuna hyperparameter search on all 3 datasets in parallel
#
# Usage: ./scripts/launch_optuna_all.sh [num_trials]
#   - num_trials: Number of trials per dataset (default: 30)
#
# Design:
#   - Uses base config.yaml with CLI overrides (dataset.name=<name>)
#   - Assigns one GPU per dataset (wiki=0, epinions=1, slashdot=2)
#   - All runs in parallel with nohup logs
###############################################################################

set -e

# Configuration
NUM_TRIALS=${1:-30}
DEVICE_WIKI=0
DEVICE_EPINIONS=1
DEVICE_SLASHDOT=2

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Launching Optuna Hyperparameter Search: All 3 Datasets"
echo "  Trials per dataset: $NUM_TRIALS"
echo "════════════════════════════════════════════════════════════"
echo ""

cd /home/dsi/shilo_avital/yolo_lab/walk_to_paint
mkdir -p nohup_logs

# Activate venv to ensure dependencies (e.g., omegaconf) are available
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "⚠️  .venv not found; using system Python"
fi

# Wiki-RfA on GPU 0
echo -e "${BLUE}🚀 Starting wiki-rfa on GPU ${DEVICE_WIKI}...${NC}"
nohup python optuna_run.py \
    --config config.yaml \
    --device $DEVICE_WIKI \
    --n-trials $NUM_TRIALS \
    dataset.name=wiki-rfa \
    > nohup_logs/optuna_wiki-rfa_${NUM_TRIALS}trials.log 2>&1 &
PID_WIKI=$!
echo -e "${GREEN}✅ Wiki-RfA launched (PID: $PID_WIKI)${NC}"

# Epinions on GPU 1
echo -e "${BLUE}🚀 Starting epinions on GPU ${DEVICE_EPINIONS}...${NC}"
nohup python optuna_run.py \
    --config config.yaml \
    --device $DEVICE_EPINIONS \
    --n-trials $NUM_TRIALS \
    dataset.name=epinions \
    > nohup_logs/optuna_epinions_${NUM_TRIALS}trials.log 2>&1 &
PID_EPINIONS=$!
echo -e "${GREEN}✅ Epinions launched (PID: $PID_EPINIONS)${NC}"

# Slashdot090221 on GPU 2
echo -e "${BLUE}🚀 Starting slashdot090221 on GPU ${DEVICE_SLASHDOT}...${NC}"
nohup python optuna_run.py \
    --config config.yaml \
    --device $DEVICE_SLASHDOT \
    --n-trials $NUM_TRIALS \
    dataset.name=slashdot090221 \
    > nohup_logs/optuna_slashdot090221_${NUM_TRIALS}trials.log 2>&1 &
PID_SLASHDOT=$!
echo -e "${GREEN}✅ Slashdot090221 launched (PID: $PID_SLASHDOT)${NC}"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All trials running in parallel!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Monitor progress:"
echo "  tail -f nohup_logs/optuna_wiki-rfa_${NUM_TRIALS}trials.log"
echo "  tail -f nohup_logs/optuna_epinions_${NUM_TRIALS}trials.log"
echo "  tail -f nohup_logs/optuna_slashdot090221_${NUM_TRIALS}trials.log"
echo ""
echo "PIDs: wiki=$PID_WIKI, epinions=$PID_EPINIONS, slashdot=$PID_SLASHDOT"
echo ""
