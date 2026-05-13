"""SDE noise stepping for Flow Matching (ODE / SDE / CPS modes)."""

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
    sde_type: str = "cps",
):
    """Flow Matching SDE step with log_prob computation.

    Returns:
        prev_sample, log_prob, mean, std.
    """
    dt = sigma_next - sigma
    min_logprob_noise_scale = 1e-2

    if determistic:
        # ════════════════════════════════════════
        # x_{t-Δt} = x_t + dt · v_θ
        # ════════════════════════════════════════
        prev_sample = latents + dt * model_output
        log_prob = torch.tensor(0.0, device=latents.device)
        mean = prev_sample
        std = torch.tensor(0.0, device=latents.device)
    elif sde_type == "cps":
        # ════════════════════════════════════════
        # ════════════════════════════════════════
        std_dev_t = sigma_next * math.sin(eta * math.pi / 2.0)
        pred_original_sample = latents - sigma * model_output      # x̂₀
        noise_estimate = latents + model_output * (1.0 - sigma)    # x̂₁
        alpha = math.sqrt(max(sigma_next ** 2 - std_dev_t ** 2, 0.0))
        mean = pred_original_sample * (1.0 - sigma_next) + noise_estimate * alpha

        noise = torch.randn(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=generator,
        )
        prev_sample = mean + std_dev_t * noise

        diff = prev_sample - mean
        log_prob = -diff.detach().pow(2).mean(dim=tuple(range(1, diff.ndim)))  # (B,)

        std = torch.tensor(std_dev_t, device=latents.device)
    else:
        # ════════════════════════════════════════
        # mean = x_t(1 + std²/(2σ)·dt) + v_θ(1 + std²(1-σ)/(2σ))·dt
        # x_{t-Δt} = mean + std·√(-dt)·ε
        # ════════════════════════════════════════
        snr_inv = sigma / (1.0 - sigma + 1e-8)
        std_dev_t = math.sqrt(max(snr_inv, 0.0)) * eta
        std_sq = std_dev_t ** 2

        drift_coeff_x = 1.0 + std_sq / (2.0 * sigma + 1e-8) * dt
        drift_coeff_v = (1.0 + std_sq * (1.0 - sigma) / (2.0 * sigma + 1e-8)) * dt

        mean = latents * drift_coeff_x + model_output * drift_coeff_v

        noise_scale = std_dev_t * math.sqrt(max(-dt, 0.0))

        noise = torch.randn(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=generator,
        )
        prev_sample = mean + noise_scale * noise

        if noise_scale > 1e-8:
            noise_scale_eff = max(noise_scale, min_logprob_noise_scale)
            diff = prev_sample - mean
            log_prob = (
                -diff.pow(2).mean(dim=tuple(range(1, diff.ndim))) / (2.0 * noise_scale_eff ** 2)
                - math.log(noise_scale_eff)
                - 0.5 * math.log(2.0 * math.pi)
            )
        else:
            log_prob = torch.zeros(latents.shape[0], device=latents.device)

        std = torch.tensor(noise_scale, device=latents.device)

    return prev_sample, log_prob, mean, std


def recompute_log_prob(
    latent_in: torch.Tensor,
    latent_out: torch.Tensor,
    model_output: torch.Tensor,
    sigma: float,
    sigma_next: float,
    eta: float = 0.7,
    sde_type: str = "cps",
):
    """Recompute log_prob for an existing trajectory under the current policy.

    Returns:
        log_prob, mean, std_dev_t, sqrt_dt.
    """
    dt = sigma_next - sigma
    min_logprob_noise_scale = 1e-2

    if sde_type == "cps":
        std_dev_t = sigma_next * math.sin(eta * math.pi / 2.0)
        pred_original_sample = latent_in - sigma * model_output      # x̂₀
        noise_estimate = latent_in + model_output * (1.0 - sigma)    # x̂₁
        alpha = math.sqrt(max(sigma_next ** 2 - std_dev_t ** 2, 0.0))
        mean = pred_original_sample * (1.0 - sigma_next) + noise_estimate * alpha
        sqrt_dt = 1.0

        diff = latent_out - mean
        log_prob = -diff.pow(2).mean(dim=tuple(range(1, diff.ndim)))  # (B,)
    else:
        snr_inv = sigma / (1.0 - sigma + 1e-8)
        std_dev_t = math.sqrt(max(snr_inv, 0.0)) * eta

        std_sq = std_dev_t ** 2
        drift_coeff_x = 1.0 + std_sq / (2.0 * sigma + 1e-8) * dt
        drift_coeff_v = (1.0 + std_sq * (1.0 - sigma) / (2.0 * sigma + 1e-8)) * dt

        mean = latent_in * drift_coeff_x + model_output * drift_coeff_v
        sqrt_dt = math.sqrt(max(-dt, 0.0))
        noise_scale = std_dev_t * sqrt_dt

        if noise_scale > 1e-8:
            noise_scale_eff = max(noise_scale, min_logprob_noise_scale)
            diff = latent_out - mean
            diff = torch.clamp(diff, min=-10.0 * noise_scale_eff, max=10.0 * noise_scale_eff)
            log_prob = (
                -diff.pow(2).mean(dim=tuple(range(1, diff.ndim))) / (2.0 * noise_scale_eff ** 2)
                - math.log(noise_scale_eff)
                - 0.5 * math.log(2.0 * math.pi)
            )  # (B,)
        else:
            log_prob = torch.zeros(latent_in.shape[0], device=latent_in.device)

    return log_prob, mean, std_dev_t, sqrt_dt
