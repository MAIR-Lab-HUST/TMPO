"""TreeMatch-RL 总损失函数

论文 §4.3:
    L_total = (1/K) Σ_i clip(ŵ_i, 1-ε, 1+ε) · L_SoftTB^(i)
              + λ₁ · L_Entropy
              + λ₂ · L_Ref
"""

import torch
import torch.nn as nn
from typing import Dict

from .softmax_tb import SoftmaxTBLoss
from .ratio_norm import RatioNormIS
from .entropy import ParticleEntropyLoss
from .reference import ReferenceConstraintLoss


class TreeMatchRLLoss(nn.Module):
    """TreeMatch-RL 完整损失函数"""

    def __init__(
        self,
        beta: float = 15.0,
        lambda_entropy: float = 0.01,
        lambda_ref: float = 0.1,
        is_clip_range: float = 0.2,
        rbf_bandwidth: float = 1.0,
    ):
        super().__init__()
        self.soft_tb = SoftmaxTBLoss(beta=beta)
        self.is_module = RatioNormIS(clip_range=is_clip_range)
        self.entropy_loss = ParticleEntropyLoss(bandwidth=rbf_bandwidth)
        self.ref_loss = ReferenceConstraintLoss()
        self.lambda_entropy = lambda_entropy
        self.lambda_ref = lambda_ref

    def forward(
        self,
        current_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        ref_log_probs: torch.Tensor,
        path_features: torch.Tensor,
        num_sde_steps: int = 3,
    ) -> tuple:
        """
        Args:
            current_log_probs: (K,) 当前策略路径 log_prob
            old_log_probs: (K,) 旧策略路径 log_prob
            rewards: (K,) 终端奖励
            ref_log_probs: (K,) 参考模型路径 log_prob
            path_features: (K, D) 各路径的 latent 特征
            num_sde_steps: SDE 步数

        Returns:
            total_loss: scalar
            metrics: Dict 各项指标
        """
        # ① Softmax-TB 逐路径损失
        per_path_tb = self.soft_tb.forward_per_path(current_log_probs, rewards)

        # ② IS 权重
        weights = self.is_module.compute_weights(current_log_probs, old_log_probs)

        # ③ 加权 Soft-TB 损失
        weighted_tb = (weights * per_path_tb).mean()

        # ④ 粒子熵正则
        loss_entropy = self.entropy_loss(path_features)

        # ⑤ 参考约束
        loss_ref = self.ref_loss(current_log_probs, ref_log_probs, num_sde_steps)

        # ⑥ 总损失
        total_loss = (
            weighted_tb
            + self.lambda_entropy * loss_entropy
            + self.lambda_ref * loss_ref
        )

        metrics = {
            "loss_total": total_loss.item(),
            "loss_soft_tb": weighted_tb.item(),
            "loss_entropy": loss_entropy.item(),
            "loss_ref": loss_ref.item(),
            "is_weight_mean": weights.mean().item(),
            "is_weight_std": weights.std().item(),
            "rewards_mean": rewards.mean().item(),
            "rewards_std": rewards.std().item(),
        }

        return total_loss, metrics
