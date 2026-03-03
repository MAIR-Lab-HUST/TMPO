#!/bin/bash

# NCCL environment variables
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export NCCL_NVLS_ENABLE=0

# Python environment variables
export PYTHONPATH=\$PYTHONPATH:$(pwd)

# Input parameters 
# TODO: Modify these parameters as needed
flux_baseline_model_dir="./data/flux"
ckpt_dir="./mix_grpo_ckpt"
mix_sampling_steps=30
total_sampling_steps=50 
prompt_type="test"
prompt_file="./data/prompts_${prompt_type}.txt"

torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    fastvideo/sample/sample_flux.py \
    --model_path "${ckpt_dir}/diffusion_pytorch_model.safetensors" \
    --prompts_file $prompt_file \
    --output_dir "${ckpt_dir}/sample_${prompt_type}_mix_${mix_sampling_steps}_${total_sampling_steps}" \
    --output_json "${ckpt_dir}/prompt_${prompt_type}_mix_${mix_sampling_steps}_${total_sampling_steps}.json" \
    --seed 617 \
    --mix_sampling_steps $mix_sampling_steps \
    --total_sampling_steps $total_sampling_steps \
    --flux_baseline_model_dir $flux_baseline_model_dir \
    # --baseline
