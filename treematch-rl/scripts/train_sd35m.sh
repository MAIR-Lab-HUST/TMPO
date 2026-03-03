#!/bin/bash
# TreeMatch-RL: SD3.5-Medium 训练脚本

# ═══════════════════════════════════════
# 单卡训练
# ═══════════════════════════════════════
# accelerate launch --config_file accelerate_configs/single_gpu.yaml \
#     treematch/train.py --config config/sd35m_lora.yaml

# ═══════════════════════════════════════
# 4 卡 FSDP 训练 (推荐)
# ═══════════════════════════════════════
accelerate launch --config_file accelerate_configs/fsdp_small.yaml \
    --num_processes 4 \
    treematch/train.py --config config/sd35m_lora.yaml

# ═══════════════════════════════════════
# 2 卡 DDP 训练
# ═══════════════════════════════════════
# accelerate launch --config_file accelerate_configs/ddp_offload.yaml \
#     --num_processes 2 \
#     treematch/train.py --config config/sd35m_lora.yaml
