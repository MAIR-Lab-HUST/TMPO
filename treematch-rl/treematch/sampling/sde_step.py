"""SDE 噪声步进模块

实现 Flow Matching 框架下的 ODE→SDE 转换：
- flow_sde_step: 核心 SDE 步进函数，支持确定性 ODE 和随机 SDE 两种模式
- 公式来源: MixGRPO / flow_grpo 的 sde_step_with_logprob
"""

import torch
import math


def flow_sde_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    sigma: float,
    sigma_next: float,
    eta: float = 0.7,
    determistic: bool = False,
    generator: torch.Generator = None,
):
    """Flow Matching SDE 步进（含 log_prob 计算）

    将确定性 ODE dx = v_θ dt 转换为等价 SDE:
        dx = [v_θ + f(x,t)] dt + g(t) dw

    Args:
        model_output: (B, C, H, W) 模型预测的速度场 v_θ
        latents: (B, C, H, W) 当前 latent x_t
        sigma: 当前时间步的 σ 值 (对应 t, 范围 [0,1])
        sigma_next: 下一时间步的 σ 值
        eta: 噪声系数 (控制 SDE 噪声强度)
        determistic: True=使用 ODE (无噪声), False=使用 SDE (注入噪声)
        generator: 随机数生成器 (用于可复现性)

    Returns:
        prev_sample: (B, C, H, W) 下一步的 latent x_{t-Δt}
        log_prob: scalar, 该步的 log π_θ(x_{t-Δt} | x_t)
        mean: (B, C, H, W) SDE 均值 μ_θ
        std: scalar, SDE 标准差 σ_t √(-dt)
    """
    dt = sigma_next - sigma  # 负值 (时间反向)

    # 噪声标准差: std_dev_t = √(σ / (1-σ)) · η
    # σ/(1-σ) 是信噪比的倒数
    snr_inv = sigma / (1.0 - sigma + 1e-8)
    std_dev_t = math.sqrt(max(snr_inv, 0.0)) * eta

    # 原始样本预测: x₀ = x_t - σ · v_θ
    pred_original = latents - sigma * model_output

    if determistic:
        # ════════════════════════════════════════
        # ODE 模式: Euler 步进 (无噪声)
        # x_{t-Δt} = x_t + dt · v_θ
        # ════════════════════════════════════════
        prev_sample = latents + dt * model_output
        log_prob = torch.tensor(0.0, device=latents.device)
        mean = prev_sample
        std = torch.tensor(0.0, device=latents.device)
    else:
        # ════════════════════════════════════════
        # SDE 模式: 含噪声步进
        # mean = x_t(1 + std²/(2σ)·dt) + v_θ(1 + std²(1-σ)/(2σ))·dt
        # x_{t-Δt} = mean + std·√(-dt)·ε
        # ════════════════════════════════════════
        std_sq = std_dev_t ** 2

        # Drift 修正系数
        drift_coeff_x = 1.0 + std_sq / (2.0 * sigma + 1e-8) * dt
        drift_coeff_v = (1.0 + std_sq * (1.0 - sigma) / (2.0 * sigma + 1e-8)) * dt

        mean = latents * drift_coeff_x + model_output * drift_coeff_v

        # 噪声项: std · √(-dt) · ε
        noise_scale = std_dev_t * math.sqrt(max(-dt, 0.0))

        noise = torch.randn(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=generator,
        )
        prev_sample = mean + noise_scale * noise

        # 对数概率: log N(x_{t-Δt} | mean, noise_scale²)
        if noise_scale > 1e-8:
            d = latents.numel() // latents.shape[0]  # 特征维度
            log_prob = (
                -((prev_sample - mean) ** 2).sum() / (2.0 * noise_scale ** 2)
                - d * math.log(noise_scale)
                - 0.5 * d * math.log(2.0 * math.pi)
            )
        else:
            log_prob = torch.tensor(0.0, device=latents.device)

        std = torch.tensor(noise_scale, device=latents.device)

    return prev_sample, log_prob, mean, std


def recompute_log_prob(
    latent_in: torch.Tensor,
    latent_out: torch.Tensor,
    model_output: torch.Tensor,
    sigma: float,
    sigma_next: float,
    eta: float = 0.7,
):
    """用当前策略重新计算已有轨迹的 log_prob（训练阶段使用）

    给定 (x_t, x_{t-Δt}) 对, 用当前模型的 v_θ 重新计算:
        log π_θ(x_{t-Δt} | x_t)

    同时返回 RatioNorm 所需的中间量 (mean, std_dev_t, sqrt_dt)。

    Args:
        latent_in: (B, C, H, W) x_t
        latent_out: (B, C, H, W) x_{t-Δt} (来自旧策略采样)
        model_output: (B, C, H, W) 当前模型的 v_θ(x_t, t)
        sigma, sigma_next: 时间步的 σ 值
        eta: 噪声系数

    Returns:
        log_prob: scalar, log π_θ(x_{t-Δt} | x_t)
        mean: (B, C, H, W) 当前策略的 SDE 均值 μ_θ
        std_dev_t: float, 噪声系数 √(σ/(1-σ))·η
        sqrt_dt: float, √(-dt)
    """
    dt = sigma_next - sigma
    snr_inv = sigma / (1.0 - sigma + 1e-8)
    std_dev_t = math.sqrt(max(snr_inv, 0.0)) * eta

    std_sq = std_dev_t ** 2
    drift_coeff_x = 1.0 + std_sq / (2.0 * sigma + 1e-8) * dt
    drift_coeff_v = (1.0 + std_sq * (1.0 - sigma) / (2.0 * sigma + 1e-8)) * dt

    mean = latent_in * drift_coeff_x + model_output * drift_coeff_v
    sqrt_dt = math.sqrt(max(-dt, 0.0))
    noise_scale = std_dev_t * sqrt_dt

    if noise_scale > 1e-8:
        d = latent_in.numel() // latent_in.shape[0]
        log_prob = (
            -((latent_out - mean) ** 2).sum() / (2.0 * noise_scale ** 2)
            - d * math.log(noise_scale)
            - 0.5 * d * math.log(2.0 * math.pi)
        )
    else:
        log_prob = torch.tensor(0.0, device=latent_in.device)

    return log_prob, mean, std_dev_t, sqrt_dt
