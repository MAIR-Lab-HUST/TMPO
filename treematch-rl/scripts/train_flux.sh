#!/bin/bash
# TreeMatch-RL: Flux 训练脚本

# ═══════════════════════════════════════
# 4 卡 FSDP 训练
# ═══════════════════════════════════════
accelerate launch --config_file accelerate_configs/fsdp_small.yaml \
    --num_processes 4 \
    treematch/train.py --config config/flux_lora.yaml
