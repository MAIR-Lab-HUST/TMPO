"""RatioNorm 重要性采样

论文 §4.3: 解决标准 IS 在 Flow Matching 中的分布偏移 + 方差不一致
    log ŵ_{i,t} = σ_t√(Δt) · (log w_{i,t} + ||Δμ_θ||² / (2σ_t²Δt))
                = Δμ_θ · ε

    log ŵ_i = (1/T) Σ_t log ŵ_{i,t}    (均值聚合, 消除长度依赖)
"""

import torch


class RatioNormIS:
    """RatioNorm 标准化的重要性采样"""

    def __init__(self, clip_range: float = 0.2):
        """
        Args:
            clip_range: 裁剪范围 ε, 即 clip(w, 1-ε, 1+ε)
        """
        self.clip_range = clip_range

    def compute_weights(
        self,
        current_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """计算 RatioNorm 标准化后的轨迹级 IS 权重

        Args:
            current_log_probs: (K,) 当前策略的路径 log_prob
            old_log_probs: (K,) 旧策略的路径 log_prob

        Returns:
            weights: (K,) 裁剪后的 IS 权重 (detached)
        """
        # 对数比率
        log_ratio = current_log_probs - old_log_probs  # (K,)

        # RatioNorm 标准化: 零均值化
        # 消除 E[log w] < 0 的负偏置
        log_ratio_normalized = log_ratio - log_ratio.mean()

        # 指数化 + 对称裁剪
        weights = torch.exp(log_ratio_normalized)
        weights = torch.clamp(weights, 1.0 - self.clip_range, 1.0 + self.clip_range)

        return weights.detach()

    def compute_step_weights(
        self,
        current_step_log_probs: torch.Tensor,
        old_step_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """逐步 RatioNorm 后聚合（更精细的版本）

        Args:
            current_step_log_probs: (K, T) 各步 log_prob
            old_step_log_probs: (K, T) 旧策略各步 log_prob

        Returns:
            weights: (K,) 裁剪后的 IS 权重
        """
        # 逐步对数比率
        log_w = current_step_log_probs - old_step_log_probs  # (K, T)

        # 逐步零均值标准化 (跨路径维度)
        log_w_norm = log_w - log_w.mean(dim=0, keepdim=True)

        # 轨迹级均值聚合 (消除长度依赖)
        log_w_traj = log_w_norm.mean(dim=1)  # (K,)

        # 指数化 + 裁剪
        weights = torch.exp(log_w_traj)
        weights = torch.clamp(weights, 1.0 - self.clip_range, 1.0 + self.clip_range)

        return weights.detach()
