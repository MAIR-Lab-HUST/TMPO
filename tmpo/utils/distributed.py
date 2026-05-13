"""Distributed communication utilities (AllGather, per-prompt normalization)."""

import torch
from typing import Optional


def gather_rewards(
    accelerator,
    rewards: torch.Tensor,
) -> torch.Tensor:
    """Gather rewards across all GPUs."""
    if accelerator.num_processes <= 1:
        return rewards

    return accelerator.gather(rewards.contiguous())


def per_prompt_normalize(
    rewards: torch.Tensor,
    num_per_prompt: int = 27,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-prompt group normalization of rewards."""
    K = rewards.shape[0]
    advantages = torch.zeros_like(rewards)

    n_prompts = K // num_per_prompt
    for i in range(n_prompts):
        start = i * num_per_prompt
        end = (i + 1) * num_per_prompt
        group = rewards[start:end]
        advantages[start:end] = (group - group.mean()) / (group.std() + eps)

    remainder = K % num_per_prompt
    if remainder > 0:
        group = rewards[-remainder:]
        advantages[-remainder:] = (group - group.mean()) / (group.std() + eps)

    return advantages
