"""Softmax-Trajectory Balance 损失函数

论文 §4.1: 通过组内比例匹配消除配分函数 Z
    L_SoftTB = Σ_i (log P_θ(τ_i)/Σ_j P_θ(τ_j) - log exp(βR_i)/Σ_j exp(βR_j))²
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxTBLoss(nn.Module):
    """Softmax-Trajectory Balance 损失

    强制模型路径概率分布与指数化奖励分布匹配:
        P_θ(τ_i) / Σ P_θ(τ_j) ≈ exp(βR_i) / Σ exp(βR_j)

    Args:
        beta: 温度参数
            β=0  → 均匀分布 (所有路径概率相等)
            β→∞ → 贪心模式 (退化为奖励最大化)
            β=15 → 推荐值 (适度偏好高奖励, 保持多样性)
    """

    def __init__(self, beta: float = 15.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        path_log_probs: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            path_log_probs: (K,) 每条路径的累积 SDE log_prob
            rewards: (K,) 终端奖励

        Returns:
            loss: scalar, Softmax-TB 损失
        """
        # 路径概率的 softmax 归一化
        # log_softmax 比手动计算更数值稳定
        log_p_normalized = F.log_softmax(path_log_probs, dim=0)

        # 奖励的指数化 softmax
        log_r_normalized = F.log_softmax(self.beta * rewards, dim=0)

        # 逐路径残差平方和
        residuals = log_p_normalized - log_r_normalized
        loss = (residuals ** 2).sum()

        return loss

    def forward_per_path(
        self,
        path_log_probs: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """返回逐路径的残差平方（用于 IS 加权）

        Returns:
            per_path_loss: (K,) 每条路径的 (log_p - log_r)²
        """
        log_p = F.log_softmax(path_log_probs, dim=0)
        log_r = F.log_softmax(self.beta * rewards, dim=0)
        return (log_p - log_r) ** 2
