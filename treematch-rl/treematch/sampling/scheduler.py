"""Beta 分布自适应调度器

根据在线奖励均值 R̄ 驱动的 Beta 分布, 动态决定三个 SDE 分叉步的位置:
- 低奖励 (难题): Beta 右倾 → 早分叉 → 全局结构重塑
- 高奖励 (易题): Beta 左倾 → 晚分叉 → 保护语义, 微调细节

论文 §4.2:
    t_split ~ Beta(1 + (1-α)κ, 1 + ακ)
    α = clip((R̄ - R_min) / (R_max - R_min), 0, 1)
"""

import torch
import math
from typing import List, Optional


class AdaptiveScheduler:
    """自适应分叉位置调度器"""

    def __init__(
        self,
        num_inference_steps: int = 28,
        num_splits: int = 3,
        kappa: float = 4.0,
        base_noise_levels: List[float] = None,
        r_min: float = 0.2,
        r_max: float = 0.35,
        ema_decay: float = 0.99,
        min_gap: int = 3,
    ):
        """
        Args:
            num_inference_steps: 总采样步数
            num_splits: 分叉点数量 (3)
            kappa: Beta 分布集中度 (κ=0 退化为均匀分布)
            base_noise_levels: 各分叉层基础噪声系数
            r_min, r_max: 奖励边界初始值 (用 EMA 在线更新)
            ema_decay: EMA 衰减系数
            min_gap: 相邻分叉步的最小间隔
        """
        self.num_inference_steps = num_inference_steps
        self.num_splits = num_splits
        self.kappa = kappa
        self.base_noise_levels = base_noise_levels or [0.4, 0.7, 1.0]
        self.r_min = r_min
        self.r_max = r_max
        self.ema_decay = ema_decay
        self.min_gap = min_gap

        # 默认分叉位置 (均匀分布)
        spacing = num_inference_steps // (num_splits + 1)
        self.default_splits = [spacing * (i + 1) for i in range(num_splits)]

    def update_reward_bounds(self, rewards: torch.Tensor):
        """用 EMA 更新奖励边界

        Args:
            rewards: (K,) 本批样本的奖励值
        """
        batch_min = rewards.min().item()
        batch_max = rewards.max().item()
        self.r_min = self.ema_decay * self.r_min + (1 - self.ema_decay) * batch_min
        self.r_max = self.ema_decay * self.r_max + (1 - self.ema_decay) * batch_max

    def compute_alpha(self, mean_reward: float) -> float:
        """计算归一化奖励水平 α

        α = clip((R̄ - R_min) / (R_max - R_min), 0, 1)
        """
        denom = self.r_max - self.r_min
        if denom < 1e-8:
            return 0.5  # 边界相同时返回中间值
        alpha = (mean_reward - self.r_min) / denom
        return max(0.0, min(1.0, alpha))

    def compute_split_steps(self, mean_reward: float) -> List[int]:
        """通过 Beta 分布计算自适应分叉位置

        Args:
            mean_reward: 组内奖励均值

        Returns:
            split_steps: 排序后的分叉步索引列表
        """
        if self.kappa <= 0:
            return self.default_splits

        alpha = self.compute_alpha(mean_reward)

        # Beta 分布参数
        a = 1.0 + (1.0 - alpha) * self.kappa
        b = 1.0 + alpha * self.kappa

        # 采样 3 个分叉点
        beta_dist = torch.distributions.Beta(a, b)
        fractions = beta_dist.sample((self.num_splits,)).sort().values

        # 映射到步骤索引: [2, num_steps - 3]
        margin = 2
        effective_range = self.num_inference_steps - 2 * margin - 1
        split_steps = (fractions * effective_range + margin).long().tolist()

        # 确保最小间隔
        for i in range(1, len(split_steps)):
            if split_steps[i] - split_steps[i - 1] < self.min_gap:
                split_steps[i] = split_steps[i - 1] + self.min_gap

        # 确保不超出范围
        split_steps = [min(s, self.num_inference_steps - margin) for s in split_steps]

        return split_steps

    def compute_noise_levels(self, mean_reward: float) -> List[float]:
        """根据难度自适应调整噪声系数

        低奖励 (难题): scale ≈ 1.3 → 更多探索
        高奖励 (易题): scale ≈ 0.8 → 更精细调整

        Args:
            mean_reward: 组内奖励均值

        Returns:
            noise_levels: 调整后的噪声系数列表
        """
        alpha = self.compute_alpha(mean_reward)
        scale = 1.0 + (1.0 - alpha) * 0.3 - alpha * 0.2

        return [max(0.2, min(eta * scale, 1.5)) for eta in self.base_noise_levels]

    def get_schedule(self, mean_reward: Optional[float] = None):
        """获取完整调度方案

        Args:
            mean_reward: 组内奖励均值 (None 则使用默认)

        Returns:
            split_steps: 分叉步索引
            noise_levels: 噪声系数
            alpha: 归一化奖励水平
        """
        if mean_reward is None:
            return self.default_splits, self.base_noise_levels, 0.5

        split_steps = self.compute_split_steps(mean_reward)
        noise_levels = self.compute_noise_levels(mean_reward)
        alpha = self.compute_alpha(mean_reward)

        return split_steps, noise_levels, alpha


def build_sigma_schedule(num_steps: int, shift: float = 3.0, device: str = "cpu") -> torch.Tensor:
    """构建 SD3/Flux 的 sigma schedule

    SD3 使用 shift 参数调整时间步分布:
        σ = shift * t / (1 + (shift - 1) * t)

    Args:
        num_steps: 总步数
        shift: 时间步偏移量 (SD3 默认 3.0)
        device: 设备

    Returns:
        sigmas: (num_steps + 1,) sigma 值, 从 σ_max 到 0
    """
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    if shift != 1.0:
        sigmas = shift * timesteps / (1.0 + (shift - 1.0) * timesteps)
    else:
        sigmas = timesteps

    return sigmas
