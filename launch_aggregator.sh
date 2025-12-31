#!/bin/bash
# Quick launcher for aggregator pipeline
cd /home/dsi/shilo_avital/yolo_lab/walk_to_paint
nohup bash scripts/run_aggregator_full.sh > aggregator_run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "Aggregator pipeline launched in background"
echo "Check logs in: aggregator_logs/ and aggregator_run_*.log"
