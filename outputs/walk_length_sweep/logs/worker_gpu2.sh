#!/bin/bash
set -euo pipefail
cd "/home/dsi/shilo_avital/yolo_lab/walk_to_paint"
source .venv/bin/activate
export PYTHONPATH="/home/dsi/shilo_avital/yolo_lab/walk_to_paint"
echo "Worker GPU=2 started at $(date)"
echo ""
echo "================================================================"
echo "  [bitcoin-otc mw=20] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=2 python -u run.py \
    "configs/bitcoin-otc.yaml" \
    "dataset.max_walk_length=20" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/bitcoin-otc/mw20" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/bitcoin-otc/soc-sign-bitcoinotc.csv.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/bitcoin-otc_seed42_nw500000_mw80_bs1024_ep50_20260419-133621/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw20" \
    --device "2" \
    2>&1 | tee "outputs/walk_length_sweep/logs/bitcoin-otc_mw20.log"
echo "  [bitcoin-otc mw=20] done at $(date)"
echo ""
echo "================================================================"
echo "  [bitcoin-otc mw=40] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=2 python -u run.py \
    "configs/bitcoin-otc.yaml" \
    "dataset.max_walk_length=40" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/bitcoin-otc/mw40" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/bitcoin-otc/soc-sign-bitcoinotc.csv.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/bitcoin-otc_seed42_nw500000_mw80_bs1024_ep50_20260419-133621/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw40" \
    --device "2" \
    2>&1 | tee "outputs/walk_length_sweep/logs/bitcoin-otc_mw40.log"
echo "  [bitcoin-otc mw=40] done at $(date)"
echo ""
echo "================================================================"
echo "  [bitcoin-otc mw=80] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=2 python -u run.py \
    "configs/bitcoin-otc.yaml" \
    "dataset.max_walk_length=80" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/bitcoin-otc/mw80" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/bitcoin-otc/soc-sign-bitcoinotc.csv.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/bitcoin-otc_seed42_nw500000_mw80_bs1024_ep50_20260419-133621/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw80" \
    --device "2" \
    2>&1 | tee "outputs/walk_length_sweep/logs/bitcoin-otc_mw80.log"
echo "  [bitcoin-otc mw=80] done at $(date)"
echo ""
echo "================================================================"
echo "  [epinions mw=5] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=2 python -u run.py \
    "configs/epinions.yaml" \
    "dataset.max_walk_length=5" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/epinions/mw5" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/epinions/soc-sign-epinions.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/epinions_seed42_nw500000_mw80_bs1024_ep50_20260416-125419/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw5" \
    --device "2" \
    2>&1 | tee "outputs/walk_length_sweep/logs/epinions_mw5.log"
echo "  [epinions mw=5] done at $(date)"
echo ""
echo "================================================================"
echo "  [epinions mw=10] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=2 python -u run.py \
    "configs/epinions.yaml" \
    "dataset.max_walk_length=10" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/epinions/mw10" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/epinions/soc-sign-epinions.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/epinions_seed42_nw500000_mw80_bs1024_ep50_20260416-125419/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw10" \
    --device "2" \
    2>&1 | tee "outputs/walk_length_sweep/logs/epinions_mw10.log"
echo "  [epinions mw=10] done at $(date)"
echo ""
echo "================================================================"
echo "  [epinions mw=20] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=2 python -u run.py \
    "configs/epinions.yaml" \
    "dataset.max_walk_length=20" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/epinions/mw20" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/epinions/soc-sign-epinions.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/epinions_seed42_nw500000_mw80_bs1024_ep50_20260416-125419/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw20" \
    --device "2" \
    2>&1 | tee "outputs/walk_length_sweep/logs/epinions_mw20.log"
echo "  [epinions mw=20] done at $(date)"
echo ""
echo "================================================================"
echo "  [epinions mw=40] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=2 python -u run.py \
    "configs/epinions.yaml" \
    "dataset.max_walk_length=40" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/epinions/mw40" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/epinions/soc-sign-epinions.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/epinions_seed42_nw500000_mw80_bs1024_ep50_20260416-125419/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw40" \
    --device "2" \
    2>&1 | tee "outputs/walk_length_sweep/logs/epinions_mw40.log"
echo "  [epinions mw=40] done at $(date)"
echo ""
echo "================================================================"
echo "  [epinions mw=80] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=2 python -u run.py \
    "configs/epinions.yaml" \
    "dataset.max_walk_length=80" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/epinions/mw80" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/epinions/soc-sign-epinions.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/epinions_seed42_nw500000_mw80_bs1024_ep50_20260416-125419/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw80" \
    --device "2" \
    2>&1 | tee "outputs/walk_length_sweep/logs/epinions_mw80.log"
echo "  [epinions mw=80] done at $(date)"
echo "Worker GPU=2 finished at $(date)"
