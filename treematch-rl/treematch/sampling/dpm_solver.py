"""DPM-Solver++ 二阶求解器

实现 Flow Matching → DPM-Solver++ 的翻译层和多步求解器。
借鉴 MixGRPO Flash 的 post 压缩策略，用于加速 ODE 尾段。

核心公式 (论文 §4.4):
    x_θ(x_t, t, c) = x_t - v_θ · t          # 速度 → x₀ 预测
    λ_t = ln((1-t) / t)                       # log-SNR
    D_i = (1 + r/2) x₀^(i) - (r/2) x₀^(i-1)  # 二阶修正 (midpoint)
    x_{t_next} = (t_next/t) x_t - (1-t_next)(e^h - 1) D_i  # 指数积分
"""

import torch
import math
from typing import List, Optional


class DPMSolverPP:
    """DPM-Solver++ 多步求解器"""

    def __init__(self, order: int = 2, solver_type: str = "midpoint"):
        """
        Args:
            order: 求解器阶数 (1 或 2)
            solver_type: 二阶类型, "midpoint" (推荐) 或 "heun"
        """
        assert order in (1, 2), f"order must be 1 or 2, got {order}"
        assert solver_type in ("midpoint", "heun"), f"Unknown solver_type: {solver_type}"
        self.order = order
        self.solver_type = solver_type
        self.reset()

    def reset(self):
        """重置求解器状态 (新路径开始时调用)"""
        self.model_outputs: List[torch.Tensor] = []
        self.lambdas: List[float] = []

    @staticmethod
    def velocity_to_x0(v_pred: torch.Tensor, x_t: torch.Tensor, sigma: float) -> torch.Tensor:
        """Flow Matching 速度预测 → 干净数据预测

        论文 §4.4: x_θ(x_t, t, c) = x_t - v_θ · t
        """
        return x_t - v_pred * sigma

    @staticmethod
    def sigma_to_lambda(sigma: float) -> float:
        """σ → log-SNR: λ = ln((1-σ) / σ)"""
        sigma = max(min(sigma, 1.0 - 1e-6), 1e-6)
        return math.log((1.0 - sigma) / sigma)

    def step(
        self,
        v_pred: torch.Tensor,
        x_t: torch.Tensor,
        sigma: float,
        sigma_next: float,
    ) -> torch.Tensor:
        """执行一步 DPM-Solver++ 步进

        Args:
            v_pred: (B, C, H, W) 模型预测的速度场
            x_t: (B, C, H, W) 当前 latent
            sigma: 当前 σ 值
            sigma_next: 下一步 σ 值

        Returns:
            x_next: (B, C, H, W) 下一步 latent
        """
        # 转换为 x₀ 预测
        x0_pred = self.velocity_to_x0(v_pred, x_t, sigma)

        # log-SNR
        lambda_t = self.sigma_to_lambda(sigma)
        lambda_next = self.sigma_to_lambda(sigma_next)
        h = lambda_next - lambda_t

        # 存储历史
        self.model_outputs.append(x0_pred)
        self.lambdas.append(lambda_t)

        if len(self.model_outputs) < 2 or self.order == 1:
            # ════════════════════════════════════
            # 一阶更新 (DDIM 等价)
            # x_next = (σ_next / σ) x_t - (1-σ_next)(e^h - 1) x₀
            # ════════════════════════════════════
            x_next = (
                (sigma_next / (sigma + 1e-8)) * x_t
                - (1.0 - sigma_next) * (math.exp(h) - 1.0) * x0_pred
            )
        else:
            # ════════════════════════════════════
            # 二阶多步修正
            # D_i = (1 + r/2) x₀^(i) - (r/2) x₀^(i-1)   (midpoint)
            # D_i = (1 + 1/(2r)) x₀^(i) - 1/(2r) x₀^(i-1) (heun)
            # ════════════════════════════════════
            x0_prev = self.model_outputs[-2]
            lambda_prev = self.lambdas[-2]
            h_prev = lambda_t - lambda_prev

            if self.solver_type == "midpoint":
                r = h / (h_prev + 1e-8)
                D = (1.0 + r / 2.0) * x0_pred - (r / 2.0) * x0_prev
            else:  # heun
                r = h / (h_prev + 1e-8)
                D = (1.0 + 1.0 / (2.0 * r + 1e-8)) * x0_pred - 1.0 / (2.0 * r + 1e-8) * x0_prev

            # 指数积分状态转移
            x_next = (
                (sigma_next / (sigma + 1e-8)) * x_t
                - (1.0 - sigma_next) * (math.exp(h) - 1.0) * D
            )

        return x_next


def build_flash_sigma_schedule(
    original_sigmas: torch.Tensor,
    last_split_step: int,
    compress_ratio: float = 0.4,
) -> tuple:
    """构建 DPM-Solver++ Flash 的压缩 sigma schedule

    MixGRPO Flash 核心思想: 最后分叉步之后的 ODE 尾段用更少的步完成

    Args:
        original_sigmas: (N+1,) 原始 sigma schedule (从 σ_max 到 0)
        last_split_step: 最后一个 SDE 分叉步的索引
        compress_ratio: 压缩比, 0.4 = 尾段步数 × 0.4

    Returns:
        new_sigmas: 重建后的 sigma schedule
        dpm_start_idx: DPM 段开始的步索引
        dpm_steps: DPM 段的步数
    """
    total_steps = len(original_sigmas) - 1
    remaining_steps = total_steps - last_split_step - 1

    if remaining_steps <= 2:
        # 尾段太短, 不压缩
        return original_sigmas, total_steps, 0

    # 压缩后步数
    dpm_steps = max(int(remaining_steps * compress_ratio), 1)

    # 前段保持不变
    front_sigmas = original_sigmas[: last_split_step + 2]  # 包含 last_split 和 last_split+1

    # 尾段重新线性插值
    start_sigma = original_sigmas[last_split_step + 1].item()
    end_sigma = 0.0
    tail_sigmas = torch.linspace(start_sigma, end_sigma, dpm_steps + 1, device=original_sigmas.device)

    # 拼接 (去掉 tail 的第一个值, 因为和 front 最后一个重复)
    new_sigmas = torch.cat([front_sigmas, tail_sigmas[1:]])

    dpm_start_idx = last_split_step + 1

    return new_sigmas, dpm_start_idx, dpm_steps
