#!/bin/bash
# TreeMatch-RL: 8×H200 训练脚本
# FSDP SHARD_GRAD_OP (ZeRO-2): 梯度和优化器状态分片, 模型参数全量保留
#
# 用法:
#   bash scripts/train_8xH200.sh                           # 默认 SD3.5-Medium
#   bash scripts/train_8xH200.sh config/flux_lora.yaml     # 指定配置文件
#   bash scripts/train_8xH200.sh config/sd35m_lora.yaml --lr 2e-5 --max_steps 500

CONFIG=${1:-config/sd35m_lora.yaml}
shift 2>/dev/null  # 移除第一个参数，剩余传给 train.py

accelerate launch \
    --config_file accelerate_configs/fsdp_8gpu.yaml \
    --num_processes 8 \
    treematch/train.py \
    --config "$CONFIG" \
    "$@"
