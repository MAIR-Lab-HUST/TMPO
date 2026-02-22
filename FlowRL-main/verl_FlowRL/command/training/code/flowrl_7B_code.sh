#!/bin/bash

PRETRAINED_MODEL=../pre_trained_model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
n_nodes=1
n_gpus_per_node=8
tensor_model_parallel_size=2
save_freq=25

train_files="['../data/code_data/deepcoder_train-00000-of-00005.parquet','../data/code_data/deepcoder_train-00001-of-00005.parquet','../data/code_data/deepcoder_train-00002-of-00005.parquet','../data/code_data/deepcoder_train-00003-of-00005.parquet','../data/code_data/deepcoder_train-00004-of-00005.parquet']"

test_files="['../data/code_data/test_livecodebench-00000-of-00005.parquet','../data/code_data/test_livecodebench-00001-of-00005.parquet','../data/code_data/test_livecodebench-00002-of-00005.parquet','../data/code_data/test_livecodebench-00003-of-00005.parquet','../data/code_data/test_livecodebench-00004-of-00005.parquet']"

experiment_name="flowrl_qwen_7b_code"
max_prompt_length=2048
max_response_length=$((1024 * 8))
OUTPUT_DIR=../checkpoints/FlowRL/code/7B/$experiment_name

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='left' \
    +actor_rollout_ref.actor.tb_type=tempered_important_sampling \
    +actor_rollout_ref.actor.porj_layer=3 \
    actor_rollout_ref.model.path=$PRETRAINED_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='FlowRL' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.resume_mode=auto \
    trainer.val_before_train=True \
    trainer.save_freq=$save_freq \
    reward_model.reward_manager=naive_code \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.test_freq=10 \
    trainer.total_epochs=20 \
    $@