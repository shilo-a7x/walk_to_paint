#!/bin/bash
set -euo pipefail
cd "/home/dsi/shilo_avital/yolo_lab/walk_to_paint"
source .venv/bin/activate
export PYTHONPATH="/home/dsi/shilo_avital/yolo_lab/walk_to_paint"
echo "Relaunch worker GPU=3 started at $(date)"
echo ""
echo "================================================================"
echo "  [wiki-elec mw=5] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=3 python -u run.py \
    "dataset.name=wiki-elec" \
    "dataset.max_walk_length=5" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/wiki-elec/mw5" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/wiki-Elec/wikiElec.ElecBs3.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/wiki-elec_seed42_nw500000_mw80_bs1024_ep50_20260419-133621/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw5" \
    2>&1 | tee "outputs/walk_length_sweep/logs/wiki-elec_mw5.log"
echo "  [wiki-elec mw=5] done at $(date)"
echo ""
echo "================================================================"
echo "  [wiki-elec mw=10] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=3 python -u run.py \
    "dataset.name=wiki-elec" \
    "dataset.max_walk_length=10" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/wiki-elec/mw10" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/wiki-Elec/wikiElec.ElecBs3.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/wiki-elec_seed42_nw500000_mw80_bs1024_ep50_20260419-133621/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw10" \
    2>&1 | tee "outputs/walk_length_sweep/logs/wiki-elec_mw10.log"
echo "  [wiki-elec mw=10] done at $(date)"
echo ""
echo "================================================================"
echo "  [wiki-elec mw=20] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=3 python -u run.py \
    "dataset.name=wiki-elec" \
    "dataset.max_walk_length=20" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/wiki-elec/mw20" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/wiki-Elec/wikiElec.ElecBs3.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/wiki-elec_seed42_nw500000_mw80_bs1024_ep50_20260419-133621/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw20" \
    2>&1 | tee "outputs/walk_length_sweep/logs/wiki-elec_mw20.log"
echo "  [wiki-elec mw=20] done at $(date)"
echo ""
echo "================================================================"
echo "  [wiki-elec mw=40] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=3 python -u run.py \
    "dataset.name=wiki-elec" \
    "dataset.max_walk_length=40" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/wiki-elec/mw40" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/wiki-Elec/wikiElec.ElecBs3.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/wiki-elec_seed42_nw500000_mw80_bs1024_ep50_20260419-133621/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw40" \
    2>&1 | tee "outputs/walk_length_sweep/logs/wiki-elec_mw40.log"
echo "  [wiki-elec mw=40] done at $(date)"
echo ""
echo "================================================================"
echo "  [wiki-elec mw=80] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=3 python -u run.py \
    "dataset.name=wiki-elec" \
    "dataset.max_walk_length=80" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/wiki-elec/mw80" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/wiki-Elec/wikiElec.ElecBs3.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/wiki-elec_seed42_nw500000_mw80_bs1024_ep50_20260419-133621/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw80" \
    2>&1 | tee "outputs/walk_length_sweep/logs/wiki-elec_mw80.log"
echo "  [wiki-elec mw=80] done at $(date)"
echo ""
echo "================================================================"
echo "  [wiki-rfa mw=5] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=3 python -u run.py \
    "dataset.name=wiki-rfa" \
    "dataset.max_walk_length=5" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/wiki-rfa/mw5" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/wiki-RfA/wiki-RfA.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/wiki-rfa_seed42_nw500000_mw80_bs1024_ep50_20260324-141920/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw5" \
    2>&1 | tee "outputs/walk_length_sweep/logs/wiki-rfa_mw5.log"
echo "  [wiki-rfa mw=5] done at $(date)"
echo ""
echo "================================================================"
echo "  [wiki-rfa mw=10] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=3 python -u run.py \
    "dataset.name=wiki-rfa" \
    "dataset.max_walk_length=10" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/wiki-rfa/mw10" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/wiki-RfA/wiki-RfA.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/wiki-rfa_seed42_nw500000_mw80_bs1024_ep50_20260324-141920/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw10" \
    2>&1 | tee "outputs/walk_length_sweep/logs/wiki-rfa_mw10.log"
echo "  [wiki-rfa mw=10] done at $(date)"
echo ""
echo "================================================================"
echo "  [wiki-rfa mw=20] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=3 python -u run.py \
    "dataset.name=wiki-rfa" \
    "dataset.max_walk_length=20" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/wiki-rfa/mw20" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/wiki-RfA/wiki-RfA.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/wiki-rfa_seed42_nw500000_mw80_bs1024_ep50_20260324-141920/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw20" \
    2>&1 | tee "outputs/walk_length_sweep/logs/wiki-rfa_mw20.log"
echo "  [wiki-rfa mw=20] done at $(date)"
echo ""
echo "================================================================"
echo "  [wiki-rfa mw=40] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=3 python -u run.py \
    "dataset.name=wiki-rfa" \
    "dataset.max_walk_length=40" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/wiki-rfa/mw40" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/wiki-RfA/wiki-RfA.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/wiki-rfa_seed42_nw500000_mw80_bs1024_ep50_20260324-141920/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw40" \
    2>&1 | tee "outputs/walk_length_sweep/logs/wiki-rfa_mw40.log"
echo "  [wiki-rfa mw=40] done at $(date)"
echo ""
echo "================================================================"
echo "  [wiki-rfa mw=80] started at $(date)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=3 python -u run.py \
    "dataset.name=wiki-rfa" \
    "dataset.max_walk_length=80" \
    "dataset.num_walks=500000" \
    "dataset.data_dir=outputs/walk_length_sweep/data/wiki-rfa/mw80" \
    "dataset.edge_list_file=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/data/wiki-RfA/wiki-RfA.txt.gz" \
    "preprocess.use_cache=false" \
    "preprocess.save=true" \
    "preprocess.use_mmap=false" \
    "model.dynamic_train_masking=true" \
    "model.dynamic_train_mask_seed_offset=0" \
    "model.node_context_mode=replace" \
    "model.node_replace_prob=0.2" \
    "model.node_replace_unk_ratio=0.7" \
    "model.hardness_lambda=1.0" \
    "model.hardness_map_path=/home/dsi/shilo_avital/yolo_lab/walk_to_paint/outputs/transformer_incremental/wiki-rfa_seed42_nw500000_mw80_bs1024_ep50_20260324-141920/artifacts/E14_HARDNODE_L10/hardness_map.pt" \
    "training.exp_name=WL_SWEEP_mw80" \
    2>&1 | tee "outputs/walk_length_sweep/logs/wiki-rfa_mw80.log"
echo "  [wiki-rfa mw=80] done at $(date)"
echo 'Relaunch worker GPU=3 finished at $(date)'
