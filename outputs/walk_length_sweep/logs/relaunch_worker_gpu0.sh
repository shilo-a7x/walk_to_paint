#!/bin/bash
set -euo pipefail
cd "/home/dsi/shilo_avital/yolo_lab/walk_to_paint"
source .venv/bin/activate
export PYTHONPATH="/home/dsi/shilo_avital/yolo_lab/walk_to_paint"
echo "Relaunch worker GPU=0 started at $(date)"
echo ""
echo "================================================================"
echo "  [slashdot mw=5] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=0 python -u run.py \
    "dataset.name=slashdot090221" \
    "dataset.max_walk_length=5" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/slashdot/mw5" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/slashdot090221/soc-sign-Slashdot090221.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/slashdot090221_seed42_nw5000000_mw80_bs1024_ep50_20260420-134149/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw5" \
    2>&1 | tee "outputs/walk_length_sweep/logs/slashdot_mw5.log"
echo "  [slashdot mw=5] done at $(date)"
echo ""
echo "================================================================"
echo "  [slashdot mw=10] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=0 python -u run.py \
    "dataset.name=slashdot090221" \
    "dataset.max_walk_length=10" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/slashdot/mw10" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/slashdot090221/soc-sign-Slashdot090221.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/slashdot090221_seed42_nw5000000_mw80_bs1024_ep50_20260420-134149/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw10" \
    2>&1 | tee "outputs/walk_length_sweep/logs/slashdot_mw10.log"
echo "  [slashdot mw=10] done at $(date)"
echo ""
echo "================================================================"
echo "  [slashdot mw=20] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=0 python -u run.py \
    "dataset.name=slashdot090221" \
    "dataset.max_walk_length=20" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/slashdot/mw20" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/slashdot090221/soc-sign-Slashdot090221.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/slashdot090221_seed42_nw5000000_mw80_bs1024_ep50_20260420-134149/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw20" \
    2>&1 | tee "outputs/walk_length_sweep/logs/slashdot_mw20.log"
echo "  [slashdot mw=20] done at $(date)"
echo ""
echo "================================================================"
echo "  [slashdot mw=40] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=0 python -u run.py \
    "dataset.name=slashdot090221" \
    "dataset.max_walk_length=40" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/slashdot/mw40" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/slashdot090221/soc-sign-Slashdot090221.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/slashdot090221_seed42_nw5000000_mw80_bs1024_ep50_20260420-134149/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw40" \
    2>&1 | tee "outputs/walk_length_sweep/logs/slashdot_mw40.log"
echo "  [slashdot mw=40] done at $(date)"
echo ""
echo "================================================================"
echo "  [slashdot mw=80] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=0 python -u run.py \
    "dataset.name=slashdot090221" \
    "dataset.max_walk_length=80" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/slashdot/mw80" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/slashdot090221/soc-sign-Slashdot090221.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/slashdot090221_seed42_nw5000000_mw80_bs1024_ep50_20260420-134149/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw80" \
    2>&1 | tee "outputs/walk_length_sweep/logs/slashdot_mw80.log"
echo "  [slashdot mw=80] done at $(date)"
echo 'Relaunch worker GPU=0 finished at $(date)'
