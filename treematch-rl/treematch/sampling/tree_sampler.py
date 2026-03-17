"""三阶 27 分支树状采样器

核心模块：构建树状采样拓扑，在 3 个分叉步执行 SDE 分叉（1→3→9→27），
其余步使用 ODE Euler 或 DPM-Solver++ 步进。

流程:
    Root → ODE → SDE(η₁) → ODE/DPM → SDE(η₂) → ODE/DPM → SDE(η₃) → DPM Flash → 27 leaves
"""

import torch
import copy
import contextlib
from typing import Dict, List, Optional

from .sde_step import flow_sde_step, recompute_log_prob
from .dpm_solver import DPMSolverPP, build_flash_sigma_schedule
from .scheduler import AdaptiveScheduler, build_sigma_schedule


# ─── Flux latent packing helpers ──────────────────────────────────────────────
# Flux Transformer forward() 接收 packed 格式 (B, L, C*4) 而非原始 (B, C, H, W)
# _pack_latents / _unpack_latents 与 FluxPipeline 中的逻辑完全一致

def _pack_latents(z: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H//2 * W//2, C*4)  [Flux 2×2 patch]"""
    B, C, H, W = z.shape
    z = z.view(B, C, H // 2, 2, W // 2, 2)
    z = z.permute(0, 2, 4, 1, 3, 5)
    z = z.reshape(B, (H // 2) * (W // 2), C * 4)
    return z


def _unpack_latents(z: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """(B, H//2 * W//2, C*4) -> (B, C, H, W)"""
    B, _L, D = z.shape
    C = D // 4
    z = z.view(B, H // 2, W // 2, C, 2, 2)
    z = z.permute(0, 3, 1, 4, 2, 5)
    z = z.reshape(B, C, H, W)
    return z


def _is_flux(transformer) -> bool:
    """检测是否是 Flux Transformer (需要 latent packing)"""
    m = transformer
    for _ in range(6):  # 逐层 unwrap FSDP / LoRA / DDP
        if hasattr(m, 'module'):
            m = m.module
        elif hasattr(m, 'base_model'):
            m = m.base_model
        elif hasattr(m, 'model'):
            m = m.model
        else:
            break
    cls_name = m.__class__.__name__.lower()
    if 'flux' in cls_name:
        return True
    cfg = getattr(m, 'config', None)
    return cfg is not None and 'flux' in str(getattr(cfg, 'model_type', '')).lower()


def _is_cuda_oom(exc: BaseException) -> bool:
    """兼容 torch.cuda.OutOfMemoryError 与部分 RuntimeError 文案。"""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "cuda" in msg and "out of memory" in msg


def _sanitize_tensor(
    x: torch.Tensor,
    clamp_value: Optional[float] = None,
    nan: float = 0.0,
):
    """将 NaN/Inf 清理为有限值，并可选做幅值截断。"""
    if not torch.isfinite(x).all():
        pos = clamp_value if clamp_value is not None else 0.0
        neg = -pos
        x = torch.nan_to_num(x, nan=nan, posinf=pos, neginf=neg)
    if clamp_value is not None:
        x = torch.clamp(x, min=-clamp_value, max=clamp_value)
    return x


def _sdpa_stable_ctx(enabled: bool):
    """保留兼容接口；默认不强制切换 SDPA kernel。"""
    # 近期在 H200 + Flux 训练态观察到 math-only SDPA 可触发全量非有限输出，
    # 因此此处回退到默认 kernel 选择策略。
    _ = enabled
    return contextlib.nullcontext()
# ──────────────────────────────────────────────────────────────────────────────


class Branch:
    """单条采样分支的状态"""

    __slots__ = ["latent", "log_prob_sum", "step_log_probs", "step_means",
                 "step_outputs", "latent_history", "sigma_history", "dpm_solver"]

    def __init__(self, latent: torch.Tensor):
        self.latent = latent
        self.log_prob_sum = 0.0
        self.step_log_probs: List[float] = []         # 每个 SDE 步的 log_prob
        self.step_means: List[torch.Tensor] = []      # 每个 SDE 步的均值 μ_θ (用于 RatioNorm)
        self.step_outputs: List[torch.Tensor] = []    # 每个 SDE 步的输出 latent (Bug1 修复)
        self.latent_history: List[torch.Tensor] = []   # 关键步的 latent 快照 (SDE 输入)
        self.sigma_history: List[float] = []           # 对应的 sigma 值
        self.dpm_solver = DPMSolverPP(order=2, solver_type="midpoint")

    def clone_for_branch(self) -> "Branch":
        """创建分叉副本（共享历史，独立未来）"""
        new = Branch(self.latent.clone())
        new.log_prob_sum = self.log_prob_sum
        new.step_log_probs = list(self.step_log_probs)
        new.step_means = list(self.step_means)
        new.step_outputs = list(self.step_outputs)
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
        guidance_scale: Optional[float] = None,
        is_flux: bool = False,
        debug_validate_finite: bool = False,
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

        # ═══ 初始化: 共享噪声起点 (统一 4D) ═══
        z_init = torch.randn(
            latent_shape, device=device, dtype=dtype, generator=generator,
        )
        # Bug3 修复: 统一使用 4D tensor (B,C,H,W)
        if z_init.dim() == 3:
            z_init = z_init.unsqueeze(0)
        active_branches = [Branch(z_init)]

        # Flux 需要 pack latents 再送入 transformer
        use_pack = is_flux
        _, _, H_lat, W_lat = latent_shape if len(latent_shape) == 4 else (1, *latent_shape)

        # ═══ 逐步采样 ═══
        for step_idx in range(total_steps):
            sigma = sigmas[step_idx].item()
            sigma_next = sigmas[step_idx + 1].item()
            timestep = torch.tensor([sigma], device=device, dtype=torch.float32)

            new_branches = []

            for branch in active_branches:
                z = _sanitize_tensor(branch.latent, clamp_value=50.0)  # 始终 4D (1,C,H,W)
                branch.latent = z

                # ── 模型前向推理 ──
                # Flux 需要 pack latents; Flux-dev 还需要额外传 guidance
                _needs_guidance = getattr(getattr(transformer, 'module', transformer), 'config', None) and \
                    getattr(getattr(getattr(transformer, 'module', transformer), 'config', None), 'guidance_embeds', False)
                _extra = {}
                if _needs_guidance:
                    _guidance = 3.5 if guidance_scale is None else float(guidance_scale)
                    _extra['guidance'] = torch.tensor(
                        [_guidance], device=device, dtype=torch.float32
                    )
                z_in = _pack_latents(z) if use_pack else z
                with _sdpa_stable_ctx(use_pack):
                    with torch.autocast("cuda", dtype):
                        model_pred_raw = transformer(
                            hidden_states=z_in,
                            timestep=timestep,
                            encoder_hidden_states=encoder_hidden_states,
                            pooled_projections=pooled_prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            return_dict=False,
                            **_extra,
                        )[0]

                # 训练态 bf16 在少数 kernel/输入组合下可能瞬时产出 NaN/Inf。
                # 安全重试仅切换执行路径, 不升到 fp32, 避免无谓放大显存占用。
                if not torch.isfinite(model_pred_raw).all():
                    z_safe = z
                    z_in_safe = _pack_latents(z_safe) if use_pack else z_safe
                    b_enc_safe = encoder_hidden_states.to(dtype=z_safe.dtype)
                    b_pool_safe = pooled_prompt_embeds.to(dtype=z_safe.dtype) if pooled_prompt_embeds is not None else None

                    _extra_safe = {}
                    if _needs_guidance:
                        _guidance = 3.5 if guidance_scale is None else float(guidance_scale)
                        _extra_safe['guidance'] = torch.tensor([_guidance], device=device, dtype=torch.float32)

                    with _sdpa_stable_ctx(use_pack):
                        model_pred_raw = transformer(
                            hidden_states=z_in_safe,
                            timestep=timestep,
                            encoder_hidden_states=b_enc_safe,
                            pooled_projections=b_pool_safe,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            return_dict=False,
                            **_extra_safe,
                        )[0]

                if debug_validate_finite and (not torch.isfinite(model_pred_raw).all()):
                    _z_min = z.min().item()
                    _z_max = z.max().item()
                    _raw_fin = torch.isfinite(model_pred_raw).float().mean().item()
                    raise RuntimeError(
                        f"Non-finite model_pred_raw in sample: step_idx={step_idx}, "
                        f"sigma={sigma:.6f}, sigma_next={sigma_next:.6f}, "
                        f"z[min,max]=[{_z_min:.4f},{_z_max:.4f}], finite_ratio={_raw_fin:.6f}"
                    )
                model_pred_raw = _sanitize_tensor(model_pred_raw, clamp_value=20.0)
                # unpack: velocity field 需要与 latent 同维 (B,C,H,W)
                model_pred = _unpack_latents(model_pred_raw, H_lat, W_lat) if use_pack else model_pred_raw

                # 防御: 采样阶段 v_pred 截断, 防止异常值传播
                model_pred = _sanitize_tensor(model_pred, clamp_value=20.0)

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
                        z_new = _sanitize_tensor(z_new, clamp_value=50.0)
                        mean = _sanitize_tensor(mean, clamp_value=50.0)
                        log_prob = _sanitize_tensor(log_prob, clamp_value=20.0)
                        new_b.latent = z_new
                        # Bug8 修复后 log_prob 是 (B,) tensor, B=1 时取 item()
                        new_b.log_prob_sum += log_prob.squeeze().item()
                        new_b.step_log_probs.append(log_prob.squeeze().item())
                        new_b.step_means.append(mean.detach().clone())
                        # Bug1 修复: 记录 SDE 步的输出
                        new_b.step_outputs.append(z_new.detach().clone())
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
                    z_new = _sanitize_tensor(z_new, clamp_value=50.0)
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
                "step_means": b.step_means,
                "step_outputs": b.step_outputs,        # Bug1: SDE 步输出
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
        is_flux: bool = False,
        guidance_scale: Optional[float] = None,
        recompute_sub_batch: int = 9,
        use_autocast: bool = True,
        debug_validate_finite: bool = False,
    ) -> Dict:
        """用当前策略重新计算各路径的 log_prob（训练阶段）

        recompute_sub_batch: 每次送入 transformer 的最大分支数 (防 OOM)
        """
        device = next(transformer.parameters()).device
        K = len(branches)
        recompute_sub_batch = max(1, int(recompute_sub_batch))
        num_sde_steps = min(len(split_steps), len(branches[0]["latent_history"]))
        use_pack = is_flux

        # 检测是否 Flux-dev (guidance-distilled)
        _m = transformer
        for _ in range(6):
            _m = getattr(_m, 'module', None) or getattr(_m, 'base_model', None) or \
                 getattr(_m, 'model', None) or _m
            if not (hasattr(transformer, 'module') or hasattr(transformer, 'base_model')):
                break
        _cfg = getattr(_m, 'config', None)
        _needs_guidance = bool(_cfg and getattr(_cfg, 'guidance_embeds', False))

        per_step_log_probs = []      # List[Tensor(K,)]
        per_step_means = []          # List[Tensor(K,...)]  当前策略
        per_step_old_log_probs = []  # List[Tensor(K,)]
        per_step_old_means = []      # List[Tensor(K,...)]  旧策略
        per_step_std_dev_ts = []     # List[float]
        per_step_sqrt_dts = []       # List[float]

        for sde_idx in range(num_sde_steps):
            step_idx = split_steps[sde_idx]
            eta = noise_levels[sde_idx]
            sigma = branches[0]["sigma_history"][sde_idx]
            sigma_next = sigmas[step_idx + 1].item()
            timestep = torch.tensor([sigma], device=device, dtype=torch.float32)

            latent_ins_all = torch.cat(
                [b["latent_history"][sde_idx] for b in branches], dim=0
            )  # (K, C, H, W)
            latent_outs_all = torch.cat(
                [b["step_outputs"][sde_idx] for b in branches], dim=0
            )  # (K, C, H, W)
            _, _, H_lat, W_lat = latent_ins_all.shape

            # 分子批次前向推理, 防止 27 分支一次性占满显存
            sub_log_probs: List[torch.Tensor] = []
            sub_means: List[torch.Tensor] = []
            std_dev_t = sqrt_dt = None

            def _run_recompute_chunk(start: int, end: int):
                B = end - start

                z_in = latent_ins_all[start:end]
                z_out = latent_outs_all[start:end]

                if debug_validate_finite:
                    if not torch.isfinite(z_in).all():
                        raise RuntimeError(
                            f"Non-finite latent_in before transformer: sde_idx={sde_idx}, "
                            f"sub_batch=({start},{end}), sigma={sigma:.6f}, sigma_next={sigma_next:.6f}, eta={eta:.4f}"
                        )
                    if not torch.isfinite(z_out).all():
                        raise RuntimeError(
                            f"Non-finite latent_out before transformer: sde_idx={sde_idx}, "
                            f"sub_batch=({start},{end}), sigma={sigma:.6f}, sigma_next={sigma_next:.6f}, eta={eta:.4f}"
                        )

                b_enc = encoder_hidden_states.expand(B, -1, -1)
                b_pool = pooled_prompt_embeds.expand(B, -1) if pooled_prompt_embeds is not None else None

                if text_ids.dim() == 2:
                    b_txt_ids = text_ids.unsqueeze(0).expand(B, -1, -1)
                    b_img_ids = latent_image_ids.unsqueeze(0).expand(B, -1, -1)
                else:
                    b_txt_ids = text_ids.expand(B, -1, -1)
                    b_img_ids = latent_image_ids.expand(B, -1, -1)

                _extra = {}
                if _needs_guidance:
                    _guidance = 3.5 if guidance_scale is None else float(guidance_scale)
                    _extra["guidance"] = torch.tensor(
                        [_guidance] * B, device=device, dtype=torch.float32
                    )

                z_in_packed = _pack_latents(z_in) if use_pack else z_in
                _amp_ctx = torch.autocast("cuda", dtype=dtype) if use_autocast else contextlib.nullcontext()
                with _sdpa_stable_ctx(use_pack):
                    with _amp_ctx:
                        v_pred_raw = transformer(
                            hidden_states=z_in_packed,
                            timestep=timestep.expand(B),
                            encoder_hidden_states=b_enc,
                            pooled_projections=b_pool,
                            txt_ids=b_txt_ids,
                            img_ids=b_img_ids,
                            return_dict=False,
                            **_extra,
                        )[0]

                if not torch.isfinite(v_pred_raw).all():
                    can_retry_safe_forward = use_autocast and (dtype != torch.float32)
                    if can_retry_safe_forward:
                        _safe_extra = {}
                        if _needs_guidance:
                            _guidance = 3.5 if guidance_scale is None else float(guidance_scale)
                            _safe_extra["guidance"] = torch.tensor(
                                [_guidance] * B, device=device, dtype=torch.float32
                            )

                        z_in_safe = z_in
                        z_in_safe_packed = _pack_latents(z_in_safe) if use_pack else z_in_safe
                        b_enc_safe = b_enc.to(dtype=z_in_safe.dtype)
                        b_pool_safe = b_pool.to(dtype=z_in_safe.dtype) if b_pool is not None else None

                        with _sdpa_stable_ctx(use_pack):
                            v_pred_raw = transformer(
                                hidden_states=z_in_safe_packed,
                                timestep=timestep.expand(B),
                                encoder_hidden_states=b_enc_safe,
                                pooled_projections=b_pool_safe,
                                txt_ids=b_txt_ids,
                                img_ids=b_img_ids,
                                return_dict=False,
                                **_safe_extra,
                            )[0]

                if debug_validate_finite and (not torch.isfinite(v_pred_raw).all()):
                    raise RuntimeError(
                        f"Non-finite v_pred_raw in recompute: sde_idx={sde_idx}, "
                        f"sub_batch=({start},{end}), sigma={sigma:.6f}, sigma_next={sigma_next:.6f}, eta={eta:.4f}"
                    )
                v_pred = _unpack_latents(v_pred_raw, H_lat, W_lat) if use_pack else v_pred_raw

                # 防御: NaN 清除 + clamp
                # clamp 不处理 NaN (clamp(NaN)=NaN), 必须先 nan_to_num
                # 8 GPU 中任一卡的 v_pred 含 NaN → reduce-scatter 后全卡梯度 NaN
                if torch.isnan(v_pred).any() or torch.isinf(v_pred).any():
                    v_pred = torch.nan_to_num(v_pred, nan=0.0, posinf=20.0, neginf=-20.0)
                v_pred = torch.clamp(v_pred, min=-20.0, max=20.0)

                if debug_validate_finite and (not torch.isfinite(v_pred).all()):
                    raise RuntimeError(
                        f"Non-finite v_pred after sanitize/clamp: sde_idx={sde_idx}, "
                        f"sub_batch=({start},{end}), sigma={sigma:.6f}, sigma_next={sigma_next:.6f}, eta={eta:.4f}"
                    )

                log_p, mean_b, _std_dev_t, _sqrt_dt = recompute_log_prob(
                    latent_in=z_in.float(),    # float32: 避免 bf16 量化误差在 diff^2 累加中累积
                    latent_out=z_out.float(),
                    model_output=v_pred.float(),
                    sigma=sigma,
                    sigma_next=sigma_next,
                    eta=eta,
                )  # log_p: (B,), mean_b: (B,C,H,W)

                if debug_validate_finite:
                    if not torch.isfinite(log_p).all():
                        raise RuntimeError(
                            f"Non-finite log_p in recompute_log_prob: sde_idx={sde_idx}, "
                            f"sub_batch=({start},{end}), sigma={sigma:.6f}, sigma_next={sigma_next:.6f}, eta={eta:.4f}, "
                            f"std_dev_t={_std_dev_t:.6f}, sqrt_dt={_sqrt_dt:.6f}"
                        )
                    if not torch.isfinite(mean_b).all():
                        raise RuntimeError(
                            f"Non-finite mean_b in recompute_log_prob: sde_idx={sde_idx}, "
                            f"sub_batch=({start},{end}), sigma={sigma:.6f}, sigma_next={sigma_next:.6f}, eta={eta:.4f}, "
                            f"std_dev_t={_std_dev_t:.6f}, sqrt_dt={_sqrt_dt:.6f}"
                        )

                return log_p, mean_b, _std_dev_t, _sqrt_dt

            chunk_size = min(recompute_sub_batch, K)
            start = 0
            while start < K:
                end = min(start + chunk_size, K)
                chunk_result = None
                oom_exc = None
                try:
                    chunk_result = _run_recompute_chunk(start, end)
                except RuntimeError as exc:
                    if not _is_cuda_oom(exc):
                        raise

                    oom_exc = exc

                oom_flag = 1 if oom_exc is not None else 0
                if oom_flag and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    oom_flag_tensor = torch.tensor([oom_flag], device=device, dtype=torch.int32)
                    torch.distributed.all_reduce(
                        oom_flag_tensor,
                        op=torch.distributed.ReduceOp.MAX,
                    )
                    oom_flag = int(oom_flag_tensor.item())

                if oom_flag:
                    chunk_result = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if chunk_size <= 1:
                        msg = (
                            f"CUDA OOM in recompute even with recompute_sub_batch=1: "
                            f"sde_idx={sde_idx}, sigma={sigma:.6f}, sigma_next={sigma_next:.6f}, eta={eta:.4f}"
                        )
                        if oom_exc is not None:
                            raise RuntimeError(msg) from oom_exc
                        raise RuntimeError(msg)
                    chunk_size = max(1, chunk_size // 2)
                    continue

                log_p, mean_b, std_dev_t, sqrt_dt = chunk_result
                sub_log_probs.append(log_p)
                sub_means.append(mean_b)
                start = end

            log_prob = torch.cat(sub_log_probs, dim=0)   # (K,)
            mean     = torch.cat(sub_means,     dim=0)   # (K,C,H,W)

            per_step_log_probs.append(log_prob)
            per_step_means.append(mean)
            per_step_old_log_probs.append(
                torch.tensor(
                    [b["step_log_probs"][sde_idx] for b in branches],
                    device=device, dtype=torch.float32,
                )
            )  # (K,)
            per_step_old_means.append(
                torch.cat([b["step_means"][sde_idx] for b in branches], dim=0)
            )  # (K,C,H,W)
            per_step_std_dev_ts.append(std_dev_t)
            per_step_sqrt_dts.append(sqrt_dt)

        path_log_probs = sum(per_step_log_probs)  # (K,)

        return {
            "path_log_probs": path_log_probs,
            "step_log_probs": per_step_log_probs,
            "step_means": per_step_means,
            "old_step_log_probs": per_step_old_log_probs,
            "old_step_means": per_step_old_means,
            "std_dev_ts": per_step_std_dev_ts,
            "sqrt_dts": per_step_sqrt_dts,
        }
