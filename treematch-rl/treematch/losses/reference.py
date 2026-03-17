"""参考模型约束损失

论文 §4.3:
    L_Ref = Σ_i (log π_θ(τ_i) / T - log π_ref(τ_i) / T)²

防止策略过度偏离预训练分布, 维持生成质量。
长度归一化 (1/T) 平衡不同复杂程度路径的梯度贡献。
"""

import torch
import torch.nn as nn


class ReferenceConstraintLoss(nn.Module):
    """参考模型约束损失"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        current_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        num_sde_steps: int = 3,
    ) -> torch.Tensor:
        """
        Args:
            current_log_probs: (K,) 当前策略的路径 log_prob 总和
            ref_log_probs: (K,) 参考模型的路径 log_prob 总和
            num_sde_steps: SDE 步数 (用于长度归一化)

        Returns:
            loss: scalar
        """
        # 长度归一化
        norm_current = current_log_probs / max(num_sde_steps, 1)
        norm_ref = ref_log_probs / max(num_sde_steps, 1)

        # L2 残差 (Bug6 修复: .sum() → .mean(), 使损失与路径数无关)
        loss = ((norm_current - norm_ref) ** 2).mean()

        return loss
