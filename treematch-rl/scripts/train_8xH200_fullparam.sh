#!/bin/bash
# ═══════════════════════════════════════
# TreeMatch-RL 8×H200 全参数训练启动脚本
# ═══════════════════════════════════════
set -e

CONFIG="${1:-config/sd35m_fullparam.yaml}"
shift 2>/dev/null || true

accelerate launch \
    --config_file accelerate_configs/fsdp_8gpu_fullparam.yaml \
    --num_processes 8 \
    treematch/train.py \
    --config "$CONFIG" \
    "$@"
