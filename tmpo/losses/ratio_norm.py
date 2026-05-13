"""RatioNorm importance sampling with per-step normalization."""

import torch
from typing import List, Optional


class RatioNormIS:
    """RatioNorm-normalized importance sampling with bias correction and symmetric clipping."""

    def __init__(self, clip_range: float = 0.2):
        """
        Args:
            clip_range: clipping range epsilon for clip(w, 1-eps, 1+eps).
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
        """Compute per-step RatioNorm-normalized trajectory-level IS weights.

        Args:
            current_step_log_probs: T tensors of shape (K,), current policy log_probs.
            old_step_log_probs: T tensors of shape (K,), old policy log_probs.
            current_step_means: T tensors of shape (K,C,H,W), current SDE means.
            old_step_means: T tensors of shape (K,C,H,W), old SDE means.
            std_dev_ts: T floats, noise coefficients sigma_t per step.
            sqrt_dts: T floats, sqrt(-dt) per step.
        Returns:
            weights: (K,) clipped IS weights (detached).
            sqrt_dt_sq_mean: float, mean of sqrt_dt^2 for loss normalization.
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
                continue

            log_w = current_step_log_probs[t] - old_step_log_probs[t]  # (K,)

            delta_mu = current_step_means[t] - old_step_means[t]  # (K,C,H,W)
            bias = delta_mu.pow(2).mean(
                dim=tuple(range(1, delta_mu.ndim))
            )  # (K,)
            bias = bias / (2.0 * noise_product ** 2)

            log_w_normalized = (log_w + bias) * noise_product  # (K,)

            normalized_ratios.append(log_w_normalized)
            sqrt_dt_sq_sum += sqrt_dt ** 2

        if len(normalized_ratios) == 0:
            return torch.ones(K, device=device), 1.0

        log_w_traj = torch.stack(normalized_ratios, dim=1).mean(dim=1)  # (K,)

        weights = torch.exp(log_w_traj)
        weights = torch.clamp(weights, 1.0 - self.clip_range, 1.0 + self.clip_range)

        sqrt_dt_sq_mean = sqrt_dt_sq_sum / len(normalized_ratios)

        return weights.detach(), sqrt_dt_sq_mean
