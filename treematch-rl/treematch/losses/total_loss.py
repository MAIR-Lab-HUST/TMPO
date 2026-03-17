"""TreeMatch-RL 总损失函数

论文 §4.3:
    L_total = (1/K) Σ_i clip(ŵ_i, 1-ε, 1+ε) · L_SoftTB^(i)
              + λ₁ · L_Entropy
              + λ₂ · L_Ref

    其中 ŵ_i 为经 RatioNorm 标准化后的轨迹级 IS 权重,
    loss 需除以 sqrt_dt² 以补偿 RatioNorm 缩放。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .softmax_tb import SoftmaxTBLoss
from .ratio_norm import RatioNormIS
from .entropy import ParticleEntropyLoss
from .reference import ReferenceConstraintLoss


class TreeMatchRLLoss(nn.Module):
    """TreeMatch-RL 完整损失函数"""

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
        self.entropy_loss = ParticleEntropyLoss(bandwidth=rbf_bandwidth)
        self.ref_loss = ReferenceConstraintLoss()
        self.lambda_entropy = lambda_entropy
        self.lambda_ref = lambda_ref
        self.ref_scale = float(ref_scale)

    def forward(
        self,
        current_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        ref_log_probs: torch.Tensor,
        path_features: torch.Tensor,
        num_sde_steps: int = 3,
        # ── RatioNorm 逐步数据 ──
        step_log_probs: Optional[List[torch.Tensor]] = None,
        old_step_log_probs: Optional[List[torch.Tensor]] = None,
        step_means: Optional[List[torch.Tensor]] = None,
        old_step_means: Optional[List[torch.Tensor]] = None,
        std_dev_ts: Optional[List[float]] = None,
        sqrt_dts: Optional[List[float]] = None,
    ) -> tuple:
        """
        Args:
            current_log_probs: (K,) 当前策略路径 log_prob
            old_log_probs: (K,) 旧策略路径 log_prob
            rewards: (K,) 终端奖励
            ref_log_probs: (K,) 参考模型路径 log_prob
            path_features: (K, D) 各路径的 latent 特征
            num_sde_steps: SDE 步数
            step_log_probs: T 个 (K,) tensor, 当前策略各步 log_prob
            old_step_log_probs: T 个 (K,) tensor, 旧策略各步 log_prob
            step_means: T 个 (K,C,H,W) tensor, 当前策略各步 SDE 均值
            old_step_means: T 个 (K,C,H,W) tensor, 旧策略各步 SDE 均值
            std_dev_ts: T 个 float, 各步的 σ_t
            sqrt_dts: T 个 float, 各步的 √(-dt)

        Returns:
            total_loss: scalar
            metrics: Dict 各项指标
        """
        # ① Softmax-TB 逐路径损失 (fp32 in softmax_tb)
        per_path_tb = self.soft_tb.forward_per_path(current_log_probs, rewards)

        # ② IS 权重 (逐步 RatioNorm)
        sqrt_dt_sq_mean = 1.0
        if (step_log_probs is not None and old_step_log_probs is not None
                and step_means is not None and old_step_means is not None
                and std_dev_ts is not None and sqrt_dts is not None):
            weights, sqrt_dt_sq_mean = self.is_module.compute_weights(
                current_step_log_probs=step_log_probs,
                old_step_log_probs=old_step_log_probs,
                current_step_means=step_means,
                old_step_means=old_step_means,
                std_dev_ts=std_dev_ts,
                sqrt_dts=sqrt_dts,
            )
        else:
            # fallback: 无逐步数据时用简化版 (首次迭代兼容)
            log_ratio = current_log_probs - old_log_probs
            log_ratio_normalized = log_ratio - log_ratio.mean()
            weights = torch.exp(log_ratio_normalized)
            weights = torch.clamp(
                weights,
                1.0 - self.is_module.clip_range,
                1.0 + self.is_module.clip_range,
            ).detach()

        # ③ 加权 Soft-TB 损失
        # 不除以 sqrt_dt_sq_mean: 该值 = mean(-dt) ≈ 0.02-0.04 (Flux 28步)
        # 若除之相当于 ×25-50 倍放大损失 → Loss>1000 → iter-2 必然 grad NaN
        # RatioNorm IS 权重已通过 noise_product 压缩为 ≈1.0, 不需额外补偿
        # sqrt_dt_sq_mean 仍保留在 metrics 中供诊断用
        weighted_tb = (weights * per_path_tb).mean()

        # ④ 粒子熵正则
        loss_entropy = self.entropy_loss(path_features)

        # ⑤ 参考约束
        loss_ref_raw = self.ref_loss(current_log_probs, ref_log_probs, num_sde_steps)
        if not isinstance(loss_ref_raw, torch.Tensor):
            loss_ref_raw = torch.as_tensor(
                loss_ref_raw,
                device=current_log_probs.device,
                dtype=current_log_probs.dtype,
            )
        if loss_ref_raw.ndim > 0:
            loss_ref_raw = loss_ref_raw.mean()
        loss_ref = loss_ref_raw * float(self.ref_scale)

        # ⑥ 总损失
        weighted_entropy = self.lambda_entropy * loss_entropy
        weighted_ref = self.lambda_ref * loss_ref
        total_loss = (
            weighted_tb
            + weighted_entropy
            + weighted_ref
        )

        # 记录三项在总损失中的相对贡献, 便于判断是否失衡
        denom = total_loss.detach().abs() + 1e-12
        tb_ratio = (weighted_tb.detach().abs() / denom).item()
        entropy_ratio = (weighted_entropy.detach().abs() / denom).item()
        ref_ratio = (weighted_ref.detach().abs() / denom).item()

        metrics = {
            "loss_total": total_loss.item(),
            "loss_soft_tb": weighted_tb.item(),
            "loss_entropy": loss_entropy.item(),
            "loss_ref": loss_ref.item(),
            "loss_entropy_weighted": weighted_entropy.item(),
            "loss_ref_weighted": weighted_ref.item(),
            "loss_ref_raw": loss_ref_raw.item(),
            "loss_ref_scale": float(self.ref_scale),
            "loss_tb_ratio": tb_ratio,
            "loss_entropy_ratio": entropy_ratio,
            "loss_ref_ratio": ref_ratio,
            "is_weight_mean": weights.mean().item(),
            "is_weight_std": weights.std().item(),
            "rewards_mean": rewards.mean().item(),
            "rewards_std": rewards.std().item(),
            "sqrt_dt_sq_mean": sqrt_dt_sq_mean,
            # ── GRPO-Guard 风格训练诊断指标 ──
            "approx_kl": (0.5 * ((current_log_probs - old_log_probs) ** 2).mean()).item(),
            "log_prob_mean": current_log_probs.mean().item(),
            "log_prob_old_mean": old_log_probs.mean().item(),
        }

        # IS ratio 统计 (需要逐步数据)
        if step_log_probs is not None and old_step_log_probs is not None:
            all_log_ratios = []
            for t in range(len(step_log_probs)):
                all_log_ratios.append(step_log_probs[t] - old_step_log_probs[t])
            log_ratio_all = torch.cat(all_log_ratios)
            ratio_all = torch.exp(log_ratio_all)
            metrics["ratio_mean"] = ratio_all.mean().item()
            metrics["ratio_std"] = ratio_all.std().item()
            metrics["clipfrac"] = (
                (torch.abs(ratio_all - 1.0) > self.is_module.clip_range).float().mean().item()
            )
        else:
            metrics["ratio_mean"] = 1.0
            metrics["ratio_std"] = 0.0
            metrics["clipfrac"] = 0.0

        return total_loss, metrics
