#!/bin/bash
# TreeMatch-RL: Flux 训练脚本
# 用法:
#   bash scripts/train_flux.sh
#   bash scripts/train_flux.sh config/flux_lora.yaml --max_steps 2 --grad_accum 1 --is_num_updates 1 --debug_grad_diag

set -euo pipefail

# 避免 PyTorch expandable_segments 在部分版本触发内部断言。
# 强制覆写, 防止继承到旧环境变量中的 expandable_segments:True。
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.8"

CONFIG_PATH="${1:-config/flux_lora.yaml}"
if [ "$#" -gt 0 ]; then
    shift
fi

accelerate launch --config_file accelerate_configs/fsdp_small.yaml \
        --num_processes 8 \
        treematch/train.py --config "$CONFIG_PATH" "$@"
