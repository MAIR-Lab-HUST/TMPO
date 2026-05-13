#!/bin/bash

# ═══════════════════════════════════════
# ═══════════════════════════════════════
# accelerate launch --config_file accelerate_configs/single_gpu.yaml \
#     tmpo/train.py --config config/sd35m_lora.yaml

# ═══════════════════════════════════════
# ═══════════════════════════════════════
accelerate launch --config_file accelerate_configs/fsdp_small.yaml \
    --num_processes 4 \
    tmpo/train.py --config config/sd35m_lora.yaml

# ═══════════════════════════════════════
# ═══════════════════════════════════════
# accelerate launch --config_file accelerate_configs/ddp_offload.yaml \
#     --num_processes 2 \
#     tmpo/train.py --config config/sd35m_lora.yaml
