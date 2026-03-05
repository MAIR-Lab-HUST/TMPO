"""RatioNorm 重要性采样

论文 §4.3 / flow_grpo GRPO-Guard:
    逐步标准化:
        log ŵ_{i,t} = (log w_{i,t} + ||Δμ_θ||² / (2(σ_t·√Δt)²)) · σ_t · √Δt
                    = Δμ_θ · ε

    轨迹级聚合:
        log ŵ_i = (1/T) Σ_t log ŵ_{i,t}

参考实现: flow_grpo-main/scripts/train_sd3_GRPO_Guard.py L915-932
"""

import torch
from typing import List, Optional


class RatioNormIS:
    """RatioNorm 标准化的重要性采样

    实现完整的逐步 RatioNorm 标准化:
        1. 加回偏置: ||Δμ||² / (2(σ_t·√Δt)²) → 消除 E[log w] < 0 的负偏置
        2. σ_t·√Δt 缩放 → 消除方差对去噪调度器参数的依赖
        3. 轨迹级均值聚合 + 对称裁剪
    """

    def __init__(self, clip_range: float = 0.2):
        """
        Args:
            clip_range: 裁剪范围 ε, 即 clip(w, 1-ε, 1+ε)
        """
        self.clip_range = clip_range

    def compute_weights(
        self,
        current_step_log_probs: List[torch.Tensor],
        old_step_log_probs: List[torch.Tensor],
        current_step_means: List[torch.Tensor],
        old_step_means: List[torch.Tensor],
        std_dev_ts: List[float],
        sqrt_dts: List[float],
    ) -> tuple:
        """计算逐步 RatioNorm 标准化后的轨迹级 IS 权重

        参照 flow_grpo GRPO-Guard 实现:
            ratio_mean_bias = ||Δμ||² / (2(σ_t·√Δt)²)
            ratio = exp((log_w + bias) · σ_t · √Δt)

        Args:
            current_step_log_probs: T 个 (K,) tensor, 当前策略各步 log_prob
            old_step_log_probs: T 个 (K,) tensor, 旧策略各步 log_prob
            current_step_means: T 个 (K,C,H,W) tensor, 当前策略各步 SDE 均值
            old_step_means: T 个 (K,C,H,W) tensor, 旧策略各步 SDE 均值
            std_dev_ts: T 个 float, 各步的噪声系数 σ_t
            sqrt_dts: T 个 float, 各步的 √(-dt)

        Returns:
            weights: (K,) 裁剪后的 IS 权重 (detached)
            sqrt_dt_sq_mean: float, 各步 sqrt_dt² 的均值 (用于 loss 归一化)
        """
        T = len(current_step_log_probs)
        if T == 0:
            K = 1
            return torch.ones(K, device=current_step_log_probs[0].device if current_step_log_probs else "cpu"), 1.0

        K = current_step_log_probs[0].shape[0]
        device = current_step_log_probs[0].device

        normalized_ratios = []
        sqrt_dt_sq_sum = 0.0

        for t in range(T):
            sigma_t = std_dev_ts[t]
            sqrt_dt = sqrt_dts[t]
            noise_product = sigma_t * sqrt_dt  # σ_t · √Δt

            if noise_product < 1e-8:
                # 噪声极小时跳过该步
                continue

            # 逐步对数比率: log w_{i,t} = log π_θ - log π_old
            log_w = current_step_log_probs[t] - old_step_log_probs[t]  # (K,)

            # 偏置修正: ||Δμ||² / (2(σ_t·√Δt)²)
            delta_mu = current_step_means[t] - old_step_means[t]  # (K,C,H,W)
            # 沿空间维度求和 (与 sde_step.py 中 log_prob 使用 .sum() 一致)
            bias = delta_mu.pow(2).sum(
                dim=tuple(range(1, delta_mu.ndim))
            )  # (K,)
            bias = bias / (2.0 * noise_product ** 2)

            # RatioNorm 标准化: (log_w + bias) · σ_t · √Δt
            log_w_normalized = (log_w + bias) * noise_product  # (K,)

            normalized_ratios.append(log_w_normalized)
            sqrt_dt_sq_sum += sqrt_dt ** 2

        if len(normalized_ratios) == 0:
            return torch.ones(K, device=device), 1.0

        # 轨迹级均值聚合: (1/T) Σ_t log ŵ_{i,t}
        log_w_traj = torch.stack(normalized_ratios, dim=1).mean(dim=1)  # (K,)

        # 指数化 + 对称裁剪
        weights = torch.exp(log_w_traj)
        weights = torch.clamp(weights, 1.0 - self.clip_range, 1.0 + self.clip_range)

        # sqrt_dt² 均值 (用于 loss 归一化)
        sqrt_dt_sq_mean = sqrt_dt_sq_sum / len(normalized_ratios)

        return weights.detach(), sqrt_dt_sq_mean
