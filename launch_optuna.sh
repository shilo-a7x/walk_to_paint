#!/bin/bash
# Launch script to run three Optuna experiments concurrently on specified GPUs.

# Usage: ./launch_optuna.sh [N_TRIALS]
# Default: N_TRIALS=200

N_TRIALS=${1:-200}

echo "🚀 Launching 3 Optuna experiments (epinions, wiki-rfa, slashdot090221)"
echo "Trials per experiment: ${N_TRIALS}"

# Activate virtual environment (if present)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Ensure PYTHONPATH so local imports work
export PYTHONPATH=.

# Experiment definitions: dataset -> device -> config file
declare -A EXP_CONFIGS
EXP_CONFIGS[epinions]=configs/epinions.yaml
EXP_CONFIGS[wiki-rfa]=configs/wiki-rfa.yaml
EXP_CONFIGS[slashdot090221]=configs/slashdot090221.yaml

declare -A EXP_DEVICE
EXP_DEVICE[epinions]=0
EXP_DEVICE[wiki-rfa]=1
EXP_DEVICE[slashdot090221]=3

for dataset in "epinions" "wiki-rfa" "slashdot090221"; do
    # use base config so model/training defaults are loaded, dataset-specific file will be merged
    base_cfg=config.yaml
    dev=${EXP_DEVICE[$dataset]}
    # create a deterministic experiment folder name (include timestamp) so optuna_run will use it
    ts=$(date +%Y%m%d-%H%M%S)
    exp_name="${dataset}-optuna_${ts}"
    exp_dir="outputs/${dataset}/${exp_name}"
    mkdir -p "${exp_dir}/optuna" "${exp_dir}/checkpoints" "${exp_dir}/logs" "${exp_dir}/plots"
    # put nohup log inside the experiment optuna dir
    logfile="${exp_dir}/optuna/optuna_nohup_${dataset}.log"

    echo "Starting ${dataset} on GPU ${dev} (exp=${exp_name}) -> log: ${logfile}"

    # Launch each experiment with CUDA_VISIBLE_DEVICES set for process-isolation.
    # We pass --device for internal selection too; optuna_run.py will place outputs under outputs/<dataset>/...
    CUDA_VISIBLE_DEVICES=${dev} nohup python -u optuna_run.py \
        --config ${base_cfg} \
        --device ${dev} \
        --n-trials ${N_TRIALS} \
        dataset.name=${dataset} \
        training.exp_name=${exp_name} \
        paths.append_timestamp=false \
        > ${logfile} 2>&1 &
done

echo "All jobs started. Logs:"
for dataset in "epinions" "wiki-rfa" "slashdot090221"; do
    echo " - outputs/${dataset}/optuna_nohup_${dataset}.log"
done

echo "Use 'tail -f outputs/<dataset>/optuna_nohup_<dataset>.log' to follow a job." 
echo "TensorBoard logs will be written to the per-experiment log directories under outputs/."
