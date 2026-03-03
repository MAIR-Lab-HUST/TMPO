"""三阶 27 分支树状采样器

核心模块：构建树状采样拓扑，在 3 个分叉步执行 SDE 分叉（1→3→9→27），
其余步使用 ODE Euler 或 DPM-Solver++ 步进。

流程:
    Root → ODE → SDE(η₁) → ODE/DPM → SDE(η₂) → ODE/DPM → SDE(η₃) → DPM Flash → 27 leaves
"""

import torch
import copy
from typing import Dict, List, Optional

from .sde_step import flow_sde_step, recompute_log_prob
from .dpm_solver import DPMSolverPP, build_flash_sigma_schedule
from .scheduler import AdaptiveScheduler, build_sigma_schedule


class Branch:
    """单条采样分支的状态"""

    __slots__ = ["latent", "log_prob_sum", "step_log_probs", "latent_history",
                 "sigma_history", "dpm_solver"]

    def __init__(self, latent: torch.Tensor):
        self.latent = latent
        self.log_prob_sum = 0.0
        self.step_log_probs: List[float] = []         # 每个 SDE 步的 log_prob
        self.latent_history: List[torch.Tensor] = []   # 关键步的 latent 快照
        self.sigma_history: List[float] = []           # 对应的 sigma 值
        self.dpm_solver = DPMSolverPP(order=2, solver_type="midpoint")

    def clone_for_branch(self) -> "Branch":
        """创建分叉副本（共享历史，独立未来）"""
        new = Branch(self.latent.clone())
        new.log_prob_sum = self.log_prob_sum
        new.step_log_probs = list(self.step_log_probs)
        new.latent_history = list(self.latent_history)
        new.sigma_history = list(self.sigma_history)
        new.dpm_solver = DPMSolverPP(
            order=self.dpm_solver.order,
            solver_type=self.dpm_solver.solver_type,
        )
        # 复制 DPM 历史
        new.dpm_solver.model_outputs = list(self.dpm_solver.model_outputs)
        new.dpm_solver.lambdas = list(self.dpm_solver.lambdas)
        return new


class TreeSampler:
    """三阶 27 分支树状采样器"""

    def __init__(
        self,
        num_inference_steps: int = 28,
        k: int = 3,
        shift: float = 3.0,
        dpm_flash_enabled: bool = True,
        dpm_compress_ratio: float = 0.4,
        dpm_order: int = 2,
        dpm_solver_type: str = "midpoint",
    ):
        self.num_inference_steps = num_inference_steps
        self.k = k
        self.shift = shift
        self.dpm_flash_enabled = dpm_flash_enabled
        self.dpm_compress_ratio = dpm_compress_ratio
        self.dpm_order = dpm_order
        self.dpm_solver_type = dpm_solver_type

    @torch.no_grad()
    def sample(
        self,
        transformer,
        latent_shape: tuple,
        encoder_hidden_states: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        split_steps: List[int],
        noise_levels: List[float],
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        generator: torch.Generator = None,
        guidance_scale: float = 0.0,
    ) -> List[Dict]:
        """执行完整的树状采样

        Args:
            transformer: 扩散模型 (SD3 JointTransformer 或 Flux)
            latent_shape: (B, C, H, W) latent 尺寸
            encoder_hidden_states: (B, seq, D) 文本编码
            pooled_prompt_embeds: (B, D) 池化文本嵌入
            text_ids: (B, seq) 文本 token ids
            latent_image_ids: (B, H*W) 图像位置 ids
            split_steps: [s1, s2, s3] 分叉步索引
            noise_levels: [η1, η2, η3] 噪声系数
            device: 计算设备
            dtype: 推理精度
            generator: 随机数生成器
            guidance_scale: CFG 引导尺度 (0.0 = 无 CFG)

        Returns:
            branches: List[Dict], 每个包含:
                - "latent": 最终 latent (C,H,W)
                - "log_prob_sum": 累积 log_prob
                - "step_log_probs": 各 SDE 步的 log_prob
                - "latent_history": 关键 latent 快照
                - "sigma_history": 对应 sigma 值
        """
        # ═══ 构建 sigma schedule ═══
        sigmas = build_sigma_schedule(self.num_inference_steps, self.shift, device)

        # ═══ DPM Flash: 压缩尾段 ═══
        last_split = max(split_steps)
        use_dpm = [False] * self.num_inference_steps
        if self.dpm_flash_enabled:
            sigmas, dpm_start, dpm_steps = build_flash_sigma_schedule(
                sigmas, last_split, self.dpm_compress_ratio,
            )
            total_steps = len(sigmas) - 1
            for i in range(dpm_start, min(dpm_start + dpm_steps, total_steps)):
                use_dpm[i] = True
        else:
            total_steps = self.num_inference_steps

        # ═══ 构建 determistic 标志 ═══
        determistic = [True] * total_steps
        for s in split_steps:
            if s < total_steps:
                determistic[s] = False

        # ═══ 初始化: 共享噪声起点 ═══
        z_init = torch.randn(
            latent_shape, device=device, dtype=dtype, generator=generator,
        )
        active_branches = [Branch(z_init)]

        # ═══ 逐步采样 ═══
        for step_idx in range(total_steps):
            sigma = sigmas[step_idx].item()
            sigma_next = sigmas[step_idx + 1].item()
            timestep = torch.tensor([sigma], device=device, dtype=dtype)

            new_branches = []

            for branch in active_branches:
                z = branch.latent

                # ── 模型前向推理 ──
                with torch.autocast("cuda", dtype):
                    model_pred = transformer(
                        hidden_states=z.unsqueeze(0) if z.dim() == 3 else z,
                        timestep=timestep,
                        encoder_hidden_states=encoder_hidden_states,
                        pooled_projections=pooled_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]

                if model_pred.dim() == 4 and model_pred.shape[0] == 1:
                    model_pred = model_pred.squeeze(0)
                if z.dim() == 4 and z.shape[0] == 1:
                    z = z.squeeze(0)

                if not determistic[step_idx]:
                    # ══════════════════════════════
                    # SDE 分叉步: 1 → k 分支
                    # ══════════════════════════════
                    split_idx = split_steps.index(step_idx)
                    eta = noise_levels[split_idx]

                    for _ in range(self.k):
                        new_b = branch.clone_for_branch()
                        z_new, log_prob, mean, std = flow_sde_step(
                            model_output=model_pred,
                            latents=z,
                            sigma=sigma,
                            sigma_next=sigma_next,
                            eta=eta,
                            determistic=False,
                            generator=None,  # 每分支独立噪声
                        )
                        new_b.latent = z_new
                        new_b.log_prob_sum += log_prob.item()
                        new_b.step_log_probs.append(log_prob.item())
                        new_b.latent_history.append(z.clone())
                        new_b.sigma_history.append(sigma)
                        new_branches.append(new_b)

                elif use_dpm[step_idx] if step_idx < len(use_dpm) else False:
                    # ══════════════════════════════
                    # DPM-Solver++ 加速步
                    # ══════════════════════════════
                    z_new = branch.dpm_solver.step(model_pred, z, sigma, sigma_next)
                    branch.latent = z_new
                    new_branches.append(branch)

                else:
                    # ══════════════════════════════
                    # ODE Euler 步进
                    # ══════════════════════════════
                    dt = sigma_next - sigma
                    z_new = z + dt * model_pred
                    branch.latent = z_new
                    new_branches.append(branch)

            active_branches = new_branches

        # ═══ 构建返回结果 ═══
        expected = self.k ** len(split_steps)
        assert len(active_branches) == expected, (
            f"Expected {expected} branches, got {len(active_branches)}"
        )

        results = []
        for b in active_branches:
            results.append({
                "latent": b.latent,
                "log_prob_sum": b.log_prob_sum,
                "step_log_probs": b.step_log_probs,
                "latent_history": b.latent_history,
                "sigma_history": b.sigma_history,
            })

        return results

    def recompute_path_log_probs(
        self,
        transformer,
        branches: List[Dict],
        split_steps: List[int],
        noise_levels: List[float],
        sigmas: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """用当前策略重新计算各路径的 log_prob（训练阶段）

        Args:
            transformer: 当前模型（需要梯度）
            branches: sample() 返回的分支列表
            其他: 同 sample()

        Returns:
            path_log_probs: (K,) 各路径的累积 SDE log_prob
        """
        all_log_probs = []
        device = next(transformer.parameters()).device

        for branch in branches:
            total_log_prob = torch.tensor(0.0, device=device)

            for i, (latent_in, sigma_val) in enumerate(
                zip(branch["latent_history"], branch["sigma_history"])
            ):
                if i >= len(split_steps):
                    break

                step_idx = split_steps[i]
                eta = noise_levels[i]
                sigma = sigma_val
                sigma_next = sigmas[step_idx + 1].item()

                # 当前模型的前向推理（需要梯度）
                latent_in_4d = latent_in.unsqueeze(0) if latent_in.dim() == 3 else latent_in
                timestep = torch.tensor([sigma], device=device, dtype=dtype)

                with torch.autocast("cuda", dtype):
                    v_pred = transformer(
                        hidden_states=latent_in_4d,
                        timestep=timestep,
                        encoder_hidden_states=encoder_hidden_states,
                        pooled_projections=pooled_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]

                if v_pred.dim() == 4:
                    v_pred = v_pred.squeeze(0)

                # 下一步 latent（从历史中获取，或最终 latent）
                if i + 1 < len(branch["latent_history"]):
                    latent_out = branch["latent_history"][i + 1]
                else:
                    latent_out = branch["latent"]

                log_prob = recompute_log_prob(
                    latent_in=latent_in,
                    latent_out=latent_out,
                    model_output=v_pred,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    eta=eta,
                )
                total_log_prob = total_log_prob + log_prob

            all_log_probs.append(total_log_prob)

        return torch.stack(all_log_probs)
