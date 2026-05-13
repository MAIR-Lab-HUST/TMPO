#!/bin/bash
#   bash scripts/train_flux.sh
#   bash scripts/train_flux.sh config/flux_lora.yaml --max_steps 2 --grad_accum 1 --is_num_updates 1 --debug_grad_diag

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.8"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export PADDLE_VISIBLE_DEVICES=""

CONFIG_PATH="${1:-config/flux_lora_pickscore.yaml}"
if [ "$#" -gt 0 ]; then
    shift
fi

accelerate launch --config_file accelerate_configs/fsdp_small.yaml \
        --num_processes 8 \
        tmpo/train.py --config "$CONFIG_PATH" "$@"
