"""分布式通信工具

封装 Accelerate 的 gather/reduce 操作, 用于:
- 跨 GPU 汇聚奖励 (AllGather)
- Per-prompt 全局归一化
- 损失平均 (AllReduce)
"""

import torch
from typing import Optional


def gather_rewards(
    accelerator,
    rewards: torch.Tensor,
) -> torch.Tensor:
    """跨所有 GPU 汇聚奖励值

    Args:
        accelerator: HuggingFace Accelerator 实例
        rewards: (K,) 本 GPU 的奖励

    Returns:
        all_rewards: (K * num_gpus,) 所有 GPU 的奖励
    """
    if accelerator.num_processes <= 1:
        return rewards

    return accelerator.gather(rewards.contiguous())


def per_prompt_normalize(
    rewards: torch.Tensor,
    num_per_prompt: int = 27,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-prompt 组内归一化

    每 num_per_prompt 个样本为一组 (来自同一 prompt),
    组内独立计算 mean 和 std 进行归一化。

    Args:
        rewards: (K,) 奖励值
        num_per_prompt: 每 prompt 的样本数 (27)
        eps: 数值稳定性

    Returns:
        advantages: (K,) 归一化后的优势值
    """
    K = rewards.shape[0]
    advantages = torch.zeros_like(rewards)

    n_prompts = K // num_per_prompt
    for i in range(n_prompts):
        start = i * num_per_prompt
        end = (i + 1) * num_per_prompt
        group = rewards[start:end]
        advantages[start:end] = (group - group.mean()) / (group.std() + eps)

    # 处理余数
    remainder = K % num_per_prompt
    if remainder > 0:
        group = rewards[-remainder:]
        advantages[-remainder:] = (group - group.mean()) / (group.std() + eps)

    return advantages
