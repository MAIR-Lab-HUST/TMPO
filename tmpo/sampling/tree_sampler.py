"""Multi-root tree sampler with SDE branching and ODE stepping."""

import torch
import copy
import contextlib
from typing import Dict, List, Optional

from .sde_step import flow_sde_step, recompute_log_prob
from .scheduler import AdaptiveScheduler, build_sigma_schedule


# ─── Flux latent packing helpers ──────────────────────────────────────────────

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


def prepare_flux_latent_image_ids(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create Flux image position ids matching diffusers' FluxPipeline.

    The last dimension is (image_index, row, column). Flux uses these ids for
    image-token positional information; setting them all to zero collapses the
    2D patch positions and hurts generation quality.
    """
    latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = torch.arange(height, device=device, dtype=dtype)[:, None]
    latent_image_ids[..., 2] = torch.arange(width, device=device, dtype=dtype)[None, :]
    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    return latent_image_ids.unsqueeze(0).expand(batch_size, -1, -1)


def _is_flux(transformer) -> bool:
    """Detect whether the transformer is a Flux model (requires latent packing)."""
    m = transformer
    for _ in range(6):
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
    """Check if exception is a CUDA OOM error."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "cuda" in msg and "out of memory" in msg


def _sanitize_tensor(
    x: torch.Tensor,
    clamp_value: Optional[float] = None,
    nan: float = 0.0,
):
    """Sanitize NaN/Inf to finite values with optional clamping."""
    if not torch.isfinite(x).all():
        pos = clamp_value if clamp_value is not None else 0.0
        neg = -pos
        x = torch.nan_to_num(x, nan=nan, posinf=pos, neginf=neg)
    if clamp_value is not None:
        x = torch.clamp(x, min=-clamp_value, max=clamp_value)
    return x


def _sdpa_stable_ctx(enabled: bool):
    """Compatibility wrapper; no-op by default."""
    _ = enabled
    return contextlib.nullcontext()
# ──────────────────────────────────────────────────────────────────────────────


class Branch:
    """State of a single sampling branch."""

    __slots__ = ["latent", "log_prob_sum", "step_log_probs", "step_means",
                 "step_outputs", "latent_history", "sigma_history"]

    def __init__(self, latent: torch.Tensor):
        self.latent = latent
        self.log_prob_sum = 0.0
        self.step_log_probs: List[float] = []
        self.step_means: List[torch.Tensor] = []
        self.step_outputs: List[torch.Tensor] = []
        self.latent_history: List[torch.Tensor] = []
        self.sigma_history: List[float] = []

    def clone_for_branch(self) -> "Branch":
        """Create a fork copy (shared history, independent future)."""
        new = Branch(self.latent.clone())
        new.log_prob_sum = self.log_prob_sum
        new.step_log_probs = list(self.step_log_probs)
        new.step_means = list(self.step_means)
        new.step_outputs = list(self.step_outputs)
        new.latent_history = list(self.latent_history)
        new.sigma_history = list(self.sigma_history)
        return new


class TreeSampler:
    """Multi-root tree sampler."""

    def __init__(
        self,
        num_inference_steps: int = 28,
        k: int = 3,
        num_roots: int = 1,
        branch_levels: Optional[int] = None,
        max_leaves: Optional[int] = None,
        shift: float = 3.0,
        use_flux_dynamic_shift: bool = True,
        flux_base_seq_len: int = 256,
        flux_max_seq_len: int = 4096,
        flux_base_shift: float = 0.5,
        flux_max_shift: float = 1.15,
        sde_type: str = "cps",
    ):
        self.num_inference_steps = num_inference_steps
        self.k = k
        self.num_roots = max(1, int(num_roots))
        self.branch_levels = None if branch_levels is None else max(0, int(branch_levels))
        self.max_leaves = None if max_leaves is None else max(1, int(max_leaves))
        self.shift = shift
        self.use_flux_dynamic_shift = use_flux_dynamic_shift
        self.flux_base_seq_len = flux_base_seq_len
        self.flux_max_seq_len = flux_max_seq_len
        self.flux_base_shift = flux_base_shift
        self.flux_max_shift = flux_max_shift
        self.sde_type = sde_type

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
        """Run full tree sampling and return branch results."""
        _, _, H_lat_sched, W_lat_sched = latent_shape if len(latent_shape) == 4 else (1, *latent_shape)
        flux_image_seq_len = (H_lat_sched // 2) * (W_lat_sched // 2) if is_flux else None
        sigmas = build_sigma_schedule(
            self.num_inference_steps,
            self.shift,
            device,
            is_flux=is_flux,
            image_seq_len=flux_image_seq_len,
            use_flux_dynamic_shift=self.use_flux_dynamic_shift,
            flux_base_seq_len=self.flux_base_seq_len,
            flux_max_seq_len=self.flux_max_seq_len,
            flux_base_shift=self.flux_base_shift,
            flux_max_shift=self.flux_max_shift,
        )

        total_steps = self.num_inference_steps

        max_branch_levels = len(split_steps)
        branch_levels = max_branch_levels if self.branch_levels is None else min(self.branch_levels, max_branch_levels)
        if self.max_leaves is not None and self.k > 1:
            while branch_levels > 0 and self.num_roots * (self.k ** branch_levels) > self.max_leaves:
                branch_levels -= 1

        promoted_step = None
        effective_roots = self.num_roots
        if (split_steps and split_steps[0] <= 1
                and branch_levels > 0 and self.k > 1):
            promoted_step = split_steps[0]
            effective_roots = self.num_roots * self.k
            branch_levels -= 1

        remaining_splits = [s for s in split_steps if s != promoted_step]
        branching_steps = set(remaining_splits[:branch_levels]) if branch_levels > 0 else set()

        determistic = [True] * total_steps
        for s in split_steps:
            if s < total_steps:
                determistic[s] = False
        if promoted_step is not None and promoted_step < total_steps:
            determistic[promoted_step] = True

        active_branches = []
        for _ in range(effective_roots):
            z_init = torch.randn(
                latent_shape, device=device, dtype=dtype, generator=generator,
            )
            if z_init.dim() == 3:
                z_init = z_init.unsqueeze(0)
            active_branches.append(Branch(z_init))

        use_pack = is_flux
        _, _, H_lat, W_lat = latent_shape if len(latent_shape) == 4 else (1, *latent_shape)

        for step_idx in range(total_steps):
            sigma = sigmas[step_idx].item()
            sigma_next = sigmas[step_idx + 1].item()
            timestep = torch.tensor([sigma], device=device, dtype=torch.float32)

            new_branches = []

            for branch in active_branches:
                z = _sanitize_tensor(branch.latent, clamp_value=50.0)
                branch.latent = z

                _needs_guidance = getattr(getattr(transformer, 'module', transformer), 'config', None) and \
                    getattr(getattr(getattr(transformer, 'module', transformer), 'config', None), 'guidance_embeds', False)
                _extra = {}
                if _needs_guidance:
                    _guidance = 3.5 if guidance_scale is None else float(guidance_scale)
                    _bs = z.shape[0]
                    _extra['guidance'] = torch.tensor(
                        [_guidance] * _bs, device=device, dtype=torch.float32
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

                if not torch.isfinite(model_pred_raw).all():
                    z_safe = z
                    z_in_safe = _pack_latents(z_safe) if use_pack else z_safe
                    b_enc_safe = encoder_hidden_states.to(dtype=z_safe.dtype)
                    b_pool_safe = pooled_prompt_embeds.to(dtype=z_safe.dtype) if pooled_prompt_embeds is not None else None

                    _extra_safe = {}
                    if _needs_guidance:
                        _guidance = 3.5 if guidance_scale is None else float(guidance_scale)
                        _bs_safe = z_safe.shape[0]
                        _extra_safe['guidance'] = torch.tensor([_guidance] * _bs_safe, device=device, dtype=torch.float32)

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
                model_pred = _unpack_latents(model_pred_raw, H_lat, W_lat) if use_pack else model_pred_raw

                model_pred = _sanitize_tensor(model_pred, clamp_value=20.0)

                if not determistic[step_idx]:
                    # ══════════════════════════════
                    # ══════════════════════════════
                    split_idx = split_steps.index(step_idx)
                    eta = noise_levels[split_idx]
                    num_children = self.k if step_idx in branching_steps else 1

                    for _ in range(num_children):
                        new_b = branch.clone_for_branch() if num_children > 1 else branch
                        z_new, log_prob, mean, std = flow_sde_step(
                            model_output=model_pred,
                            latents=z,
                            sigma=sigma,
                            sigma_next=sigma_next,
                            eta=eta,
                            determistic=False,
                            generator=generator,
                            sde_type=self.sde_type,
                        )
                        z_new = _sanitize_tensor(z_new, clamp_value=50.0)
                        mean = _sanitize_tensor(mean, clamp_value=50.0)
                        log_prob = _sanitize_tensor(log_prob, clamp_value=20.0)
                        new_b.latent = z_new
                        new_b.log_prob_sum += log_prob.squeeze().item()
                        new_b.step_log_probs.append(log_prob.squeeze().item())
                        new_b.step_means.append(mean.detach().clone())
                        new_b.step_outputs.append(z_new.detach().clone())
                        new_b.latent_history.append(z.clone())
                        new_b.sigma_history.append(sigma)
                        new_branches.append(new_b)

                else:
                    # ══════════════════════════════
                    # ══════════════════════════════
                    dt = sigma_next - sigma
                    z_new = z + dt * model_pred
                    z_new = _sanitize_tensor(z_new, clamp_value=50.0)
                    branch.latent = z_new
                    new_branches.append(branch)

            active_branches = new_branches

            # if not determistic[step_idx] and len(active_branches) > 1:
            #     _lats = torch.cat([b.latent for b in active_branches], dim=0)  # (K, C, H, W)
            #     _pair_dist = torch.cdist(
            #         _lats.view(len(active_branches), -1).float(),
            #         _lats.view(len(active_branches), -1).float(),
            #     ).mean().item()
            #     _std = _lats.std(dim=0).mean().item()
            #     print(
            #         f"[TREE-DIAG] step={step_idx}, branches={len(active_branches)}, "
            #         f"sigma={sigma:.4f}→{sigma_next:.4f}, eta={noise_levels[split_steps.index(step_idx)]:.2f}, "
            #         f"std_dev_t={sigma_next * __import__('math').sin(noise_levels[split_steps.index(step_idx)] * __import__('math').pi / 2):.4f}, "
            #         f"latent_pair_dist={_pair_dist:.6f}, latent_cross_std={_std:.6f}"
            #     )

            # if len(active_branches) > 1:
            #     _final_lats = torch.cat([b.latent for b in active_branches], dim=0)
            #     _final_pair_dist = torch.cdist(
            #         _final_lats.view(len(active_branches), -1).float(),
            #         _final_lats.view(len(active_branches), -1).float(),
            #     ).mean().item()
            #     _final_std = _final_lats.std(dim=0).mean().item()
            #     print(
            #         f"[TREE-DIAG] FINAL after ODE: branches={len(active_branches)}, "
            #         f"latent_pair_dist={_final_pair_dist:.6f}, latent_cross_std={_final_std:.6f}"
            #     )

        expected = effective_roots * (self.k ** len(branching_steps))
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
                "step_outputs": b.step_outputs,
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
        """Recompute path log-probs under current policy (training phase)."""
        device = next(transformer.parameters()).device
        K = len(branches)
        recompute_sub_batch = max(1, int(recompute_sub_batch))
        num_sde_steps = min(len(split_steps), len(branches[0]["latent_history"]))
        use_pack = is_flux

        _m = transformer
        for _ in range(6):
            _m = getattr(_m, 'module', None) or getattr(_m, 'base_model', None) or \
                 getattr(_m, 'model', None) or _m
            if not (hasattr(transformer, 'module') or hasattr(transformer, 'base_model')):
                break
        _cfg = getattr(_m, 'config', None)
        _needs_guidance = bool(_cfg and getattr(_cfg, 'guidance_embeds', False))

        per_step_log_probs = []      # List[Tensor(K,)]
        per_step_means = []
        per_step_old_log_probs = []  # List[Tensor(K,)]
        per_step_old_means = []
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

                if torch.isnan(v_pred).any() or torch.isinf(v_pred).any():
                    v_pred = torch.nan_to_num(v_pred, nan=0.0, posinf=20.0, neginf=-20.0)
                v_pred = torch.clamp(v_pred, min=-20.0, max=20.0)

                if debug_validate_finite and (not torch.isfinite(v_pred).all()):
                    raise RuntimeError(
                        f"Non-finite v_pred after sanitize/clamp: sde_idx={sde_idx}, "
                        f"sub_batch=({start},{end}), sigma={sigma:.6f}, sigma_next={sigma_next:.6f}, eta={eta:.4f}"
                    )

                log_p, mean_b, _std_dev_t, _sqrt_dt = recompute_log_prob(
                    latent_in=z_in.float(),
                    latent_out=z_out.float(),
                    model_output=v_pred.float(),
                    sigma=sigma,
                    sigma_next=sigma_next,
                    eta=eta,
                    sde_type=self.sde_type,
                )  # log_p: (B,), mean_b: (B,C,H,W)

                log_p = torch.nan_to_num(log_p, nan=0.0, posinf=40.0, neginf=-40.0)
                log_p = torch.clamp(log_p, min=-40.0, max=40.0)
                mean_b = torch.nan_to_num(mean_b, nan=0.0, posinf=20.0, neginf=-20.0)
                mean_b = torch.clamp(mean_b, min=-20.0, max=20.0)

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

        safe_step_log_probs = [
            torch.clamp(torch.nan_to_num(lp, nan=0.0, posinf=40.0, neginf=-40.0), min=-40.0, max=40.0)
            for lp in per_step_log_probs
        ]
        path_log_probs = sum(safe_step_log_probs)  # (K,)
        path_log_probs = torch.clamp(
            torch.nan_to_num(path_log_probs, nan=0.0, posinf=80.0, neginf=-80.0),
            min=-80.0,
            max=80.0,
        )

        return {
            "path_log_probs": path_log_probs,
            "step_log_probs": safe_step_log_probs,
            "step_means": per_step_means,
            "old_step_log_probs": per_step_old_log_probs,
            "old_step_means": per_step_old_means,
            "std_dev_ts": per_step_std_dev_ts,
            "sqrt_dts": per_step_sqrt_dts,
        }
