"""TMPO total loss: clipped IS ratio * detached Softmax-TB advantage."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .softmax_tb import SoftmaxTBLoss
from .ratio_norm import RatioNormIS
from .reference import ReferenceConstraintLoss


class TMPOLoss(nn.Module):
    """Complete TMPO loss with Softmax-TB advantage, IS ratio, and reference constraint."""

    def __init__(
        self,
        beta: float = 15.0,
        lambda_entropy: float = 0.01,
        lambda_ref: float = 0.1,
        is_clip_range: float = 0.2,
        rbf_bandwidth: float = 1.0,
        ref_scale: float = 1.0,
    ):
        super().__init__()
        self.soft_tb = SoftmaxTBLoss(beta=beta)
        self.is_module = RatioNormIS(clip_range=is_clip_range)
        self.ref_loss = ReferenceConstraintLoss()
        self.lambda_ref = lambda_ref
        self.ref_scale = float(ref_scale)
        self.clip_range = is_clip_range

    def forward(
        self,
        current_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        ref_log_probs: torch.Tensor,
        path_features: torch.Tensor,
        num_sde_steps: int = 3,
        step_log_probs: Optional[List[torch.Tensor]] = None,
        old_step_log_probs: Optional[List[torch.Tensor]] = None,
        step_means: Optional[List[torch.Tensor]] = None,
        old_step_means: Optional[List[torch.Tensor]] = None,
        std_dev_ts: Optional[List[float]] = None,
        sqrt_dts: Optional[List[float]] = None,
    ) -> tuple:
        advantage = self.soft_tb.compute_advantage(current_log_probs, rewards)

        if (step_log_probs is not None and old_step_log_probs is not None
                and step_means is not None and old_step_means is not None
                and std_dev_ts is not None and sqrt_dts is not None):
            # detached IS weights for magnitude
            weights_detached, _ = self.is_module.compute_weights(
                current_step_log_probs=step_log_probs,
                old_step_log_probs=old_step_log_probs,
                current_step_means=step_means,
                old_step_means=old_step_means,
                std_dev_ts=std_dev_ts,
                sqrt_dts=sqrt_dts,
            )
            log_ratio = (current_log_probs - old_log_probs)
            log_ratio_centered = log_ratio - log_ratio.detach().mean()
            log_ratio_centered = torch.clamp(log_ratio_centered, -10.0, 10.0)
            ratio = torch.exp(log_ratio_centered)
            ratio_clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        else:
            weights_detached = torch.ones_like(current_log_probs)
            log_ratio = current_log_probs - old_log_probs
            log_ratio_centered = log_ratio - log_ratio.detach().mean()
            log_ratio_centered = torch.clamp(log_ratio_centered, -10.0, 10.0)
            ratio = torch.exp(log_ratio_centered)
            ratio_clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)

        unclipped = -advantage * ratio * weights_detached
        clipped = -advantage * ratio_clipped * weights_detached
        loss_tb = torch.maximum(unclipped, clipped).mean()

        loss_ref_raw = self.ref_loss(current_log_probs, ref_log_probs, num_sde_steps)
        if not isinstance(loss_ref_raw, torch.Tensor):
            loss_ref_raw = torch.as_tensor(
                loss_ref_raw, device=current_log_probs.device, dtype=current_log_probs.dtype,
            )
        if loss_ref_raw.ndim > 0:
            loss_ref_raw = loss_ref_raw.mean()
        loss_ref = loss_ref_raw * float(self.ref_scale)
        weighted_ref = self.lambda_ref * loss_ref

        total_loss = loss_tb + weighted_ref

        # ⑥ Metrics
        with torch.no_grad():
            tb_mse_monitor = self.soft_tb.forward(current_log_probs, rewards).item()

        metrics = {
            "loss_total": total_loss.item(),
            "loss_soft_tb": tb_mse_monitor,
            "loss_entropy": 0.0,
            "loss_ref": loss_ref.item(),
            "loss_ref_weighted": weighted_ref.item(),
            "loss_ref_raw": loss_ref_raw.item(),
            "loss_ref_scale": float(self.ref_scale),
            "is_weight_mean": weights_detached.mean().item(),
            "is_weight_std": weights_detached.std().item(),
            "advantage_mean": advantage.mean().item(),
            "advantage_std": advantage.std().item(),
            "rewards_mean": rewards.mean().item(),
            "rewards_std": rewards.std().item(),
            "approx_kl": (0.5 * ((current_log_probs - old_log_probs).detach() ** 2).mean()).item(),
            "log_prob_mean": current_log_probs.mean().item(),
            "log_prob_old_mean": old_log_probs.mean().item(),
        }

        if step_log_probs is not None and old_step_log_probs is not None:
            all_log_ratios = []
            for t in range(len(step_log_probs)):
                all_log_ratios.append(step_log_probs[t] - old_step_log_probs[t])
            log_ratio_all = torch.cat(all_log_ratios)
            ratio_all = torch.exp(log_ratio_all)
            metrics["ratio_mean"] = ratio_all.mean().item()
            metrics["ratio_std"] = ratio_all.std().item()
            metrics["clipfrac"] = (
                (torch.abs(ratio_all - 1.0) > self.clip_range).float().mean().item()
            )
        else:
            metrics["ratio_mean"] = ratio.detach().mean().item()
            metrics["ratio_std"] = ratio.detach().std().item()
            metrics["clipfrac"] = 0.0

        return total_loss, metrics
