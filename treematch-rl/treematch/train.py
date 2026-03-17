"""TreeMatch-RL 主训练入口

完整训练循环:
    1. 加载模型 (SD3.5/Flux + LoRA + VAE)
    2. 自适应调度: Beta 分布 → 分叉位置 + 噪声系数
    3. 树状采样: 3 阶 27 分支 + DPM Flash 加速
    4. VAE 解码 + 多奖励评分
    5. Softmax-TB 损失 + IS 多次更新
    6. FSDP 分布式参数更新
"""

import os
import sys
# 自动将项目根目录加入 Python path (treematch-rl/)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import contextlib
import json
import argparse
import math
import re
import warnings
import torch
import yaml
try:
    import wandb
except ImportError:
    wandb = None
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, StableDiffusion3Pipeline
try:
    from diffusers import FluxPipeline
except ImportError:
    FluxPipeline = None
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset

from treematch.sampling import TreeSampler, AdaptiveScheduler
from treematch.sampling.scheduler import build_sigma_schedule
from treematch.losses import TreeMatchRLLoss
from treematch.losses.entropy import ParticleEntropyLoss
from treematch.rewards.compute import build_reward_models, decode_and_compute_rewards
from treematch.utils.logging_ import setup_logging, main_print
from treematch.utils.checkpoint import save_checkpoint
from treematch.utils.distributed import gather_rewards


def _parse_int_list(csv_text: str):
    if csv_text is None:
        return None
    parts = [x.strip() for x in csv_text.split(",") if x.strip()]
    return [int(x) for x in parts]


def _parse_float_list(csv_text: str):
    if csv_text is None:
        return None
    parts = [x.strip() for x in csv_text.split(",") if x.strip()]
    return [float(x) for x in parts]


def print_config_table(config: dict, accelerator):
    """格式化打印配置摘要表"""
    model_cfg = config.get("model", {})
    tree_cfg = config.get("tree", {})
    loss_cfg = config.get("loss", {})
    train_cfg = config.get("training", {})
    reward_cfg = config.get("reward", {})
    dpm_cfg = config.get("dpm_flash", {})
    ds_cfg = config.get("dataset", {})
    lora_cfg = model_cfg.get("lora", {})

    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║             TreeMatch-RL Configuration                  ║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║ Model         │ {model_cfg.get('pretrained_path', 'N/A'):<39}║",
        f"║ Resolution    │ {str(ds_cfg.get('resolution', 'N/A')):<39}║",
    ]
    if lora_cfg:
        lines.append(f"║ LoRA          │ rank={lora_cfg.get('rank')}, alpha={lora_cfg.get('alpha')}, dropout={lora_cfg.get('dropout', 0.0):<8}║")
    lines += [
        "╠══════════════════════════════════════════════════════════╣",
        f"║ Tree k        │ {tree_cfg.get('k', 3):<39}║",
        f"║ Infer Steps   │ {tree_cfg.get('num_inference_steps', 28):<39}║",
        f"║ κ (kappa)     │ {tree_cfg.get('kappa', 4.0):<39}║",
        f"║ Noise Levels  │ {str(tree_cfg.get('base_noise_levels', [])):<39}║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║ DPM Flash     │ {str(dpm_cfg.get('enabled', False)):<39}║",
        f"║ Compress Ratio│ {dpm_cfg.get('compress_ratio', 0.4):<39}║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║ β (Softmax-TB)│ {loss_cfg.get('beta', 15.0):<39}║",
        f"║ λ_entropy     │ {loss_cfg.get('lambda_entropy', 0.01):<39}║",
        f"║ λ_ref         │ {loss_cfg.get('lambda_ref', 0.1):<39}║",
        f"║ IS clip range │ {loss_cfg.get('is_clip_range', 0.2):<39}║",
        f"║ IS updates    │ {loss_cfg.get('is_num_updates', 4):<39}║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║ Reward Models │ {str(reward_cfg.get('models', [])):<39}║",
        f"║ Reward Weights│ {str(reward_cfg.get('weights', [])):<39}║",
        f"║ Mix Strategy  │ {reward_cfg.get('mix_strategy', 'advantage_aggr'):<39}║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║ LR            │ {train_cfg.get('learning_rate', 1e-5):<39}║",
        f"║ Max Steps     │ {train_cfg.get('max_train_steps', 300):<39}║",
        f"║ Grad Accum    │ {train_cfg.get('gradient_accumulation_steps', 1):<39}║",
        f"║ Mixed Prec    │ {train_cfg.get('mixed_precision', 'bf16'):<39}║",
        f"║ Seed          │ {train_cfg.get('seed', 42):<39}║",
        f"║ Num GPUs      │ {accelerator.num_processes:<39}║",
        "╚══════════════════════════════════════════════════════════╝",
    ]
    for line in lines:
        main_print(line)


def _set_param_debug_names(module):
    """为原始参数打上稳定名字，便于 FSDP/optimizer 诊断。"""
    for name, param in module.named_parameters():
        try:
            setattr(param, "_debug_name", name)
        except Exception:
            pass


def _param_debug_name(param: torch.nn.Parameter, fallback: str) -> str:
    name = getattr(param, "_debug_name", None)
    if name:
        return str(name)

    fqns = getattr(param, "_fqns", None)
    if fqns:
        fqns = list(fqns)
        if len(fqns) == 1:
            return str(fqns[0])
        return f"{fqns[0]} (+{len(fqns) - 1} more)"

    return fallback


def _iter_unique_optimizer_named_params(optimizer):
    """遍历 optimizer 当前真实持有的参数，去重并附带可读名字。"""
    seen = set()
    for group_idx, group in enumerate(optimizer.param_groups):
        for param_idx, param in enumerate(group.get("params", [])):
            if param is None:
                continue
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            yield _param_debug_name(param, f"optim_g{group_idx}_p{param_idx}"), param


# ═══════════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════════
class PromptDataset(Dataset):
    """Prompt 数据集 (支持 JSON / JSONL 格式)"""

    def __init__(self, json_path: str):
        self.prompts = []

        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # 自动检测格式: JSONL (每行一个 JSON) vs JSON (单个 JSON)
        if content.startswith("[") or content.startswith("{"):
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    # {"key": "prompt", ...}
                    self.prompts = list(data.values())
                elif isinstance(data, list):
                    # [{"prompt": "..."}, ...] 或 ["prompt1", ...]
                    self.prompts = [
                        item["prompt"] if isinstance(item, dict) else str(item)
                        for item in data
                    ]
                else:
                    self.prompts = [str(data)]
            except json.JSONDecodeError:
                # JSON 解析失败, 尝试 JSONL
                pass

        # JSONL 格式: 每行一个 JSON 对象
        if not self.prompts:
            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "prompt" in obj:
                        self.prompts.append(obj["prompt"])
                    elif isinstance(obj, str):
                        self.prompts.append(obj)
                    else:
                        self.prompts.append(str(obj))
                except json.JSONDecodeError:
                    # 纯文本行
                    self.prompts.append(line)

        if not self.prompts:
            raise ValueError(f"No prompts loaded from {json_path}")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}


# ═══════════════════════════════════════════════════
# 训练一步
# ═══════════════════════════════════════════════════
def train_one_step(
    accelerator,
    transformer,
    vae,
    pipeline,
    tree_sampler,
    scheduler,
    loss_fn,
    optimizer,
    lr_scheduler,
    prompt: str,
    reward_models,
    reward_weights,
    config,
    step: int,
    prev_mean_reward: float = None,
    is_flux: bool = False,
    debug_cfg: dict = None,
):
    """执行完整的一步训练

    Returns:
        metrics: 训练指标字典
        mean_reward: 本步平均奖励 (用于下一步自适应调度)
    """
    device = accelerator.device
    dtype = torch.bfloat16

    tree_cfg = config["tree"]
    loss_cfg = config["loss"]
    train_cfg = config["training"]
    model_cfg = config.get("model", {})
    debug_cfg = debug_cfg or {}
    guidance_scale = float(model_cfg.get("guidance_scale", 3.5 if is_flux else 0.0))

    debug_grad_diag = bool(debug_cfg.get("grad_diag", False))
    debug_tensor_stats = bool(debug_cfg.get("tensor_stats", False))
    debug_all_ranks = bool(debug_cfg.get("all_ranks", False))
    debug_fail_fast = bool(debug_cfg.get("fail_fast", False))
    debug_mid_stats = bool(debug_cfg.get("mid_stats", False))
    debug_step_stats = bool(debug_cfg.get("step_stats", False))
    debug_safe_grad = bool(debug_cfg.get("safe_grad", False))
    debug_recompute_fp32 = bool(debug_cfg.get("recompute_fp32", False))
    debug_nan_trace = bool(debug_cfg.get("nan_trace", False))
    debug_nan_trace_raise = bool(debug_cfg.get("nan_trace_raise", False))
    backprop_scale = float(debug_cfg.get("backprop_scale", 1.0))
    debug_every = max(1, int(debug_cfg.get("every", 1)))
    debug_nan_param_limit = max(1, int(debug_cfg.get("nan_param_print_limit", 3)))

    def rank_debug_print(msg: str):
        if debug_all_ranks:
            print(f"[RANK {accelerator.process_index}] {msg}", flush=True)

    # ═══ 1. 调度: 固定调度 or 自适应调度 ═══
    if tree_cfg.get("force_fixed_schedule", False):
        split_steps = list(tree_cfg.get("fixed_split_steps", [7, 14, 21]))
        noise_levels = list(tree_cfg.get("fixed_noise_levels", [0.4, 0.7, 1.0]))
        alpha = float(tree_cfg.get("fixed_alpha", 0.5))
    else:
        split_steps, noise_levels, alpha = scheduler.get_schedule(prev_mean_reward)

    if len(split_steps) != len(noise_levels):
        raise ValueError(
            f"split_steps/noise_levels length mismatch: {len(split_steps)} vs {len(noise_levels)}"
        )
    main_print(
        f"[Step {step}] alpha={alpha:.2f}, "
        f"splits={split_steps}, noise={[f'{n:.2f}' for n in noise_levels]}"
    )

    # ═══ 2. 文本编码 ═══
    with torch.no_grad():
        # 临时把 text encoder 搬到 GPU 做编码, 完成后立刻移回 CPU 释放显存
        _enc_attrs = ["text_encoder", "text_encoder_2", "text_encoder_3"]
        for _attr in _enc_attrs:
            _enc = getattr(pipeline, _attr, None)
            if _enc is not None:
                _enc.to(device)

        text_inputs = pipeline.tokenizer(
            [prompt],
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # 获取文本嵌入 (Flux vs SD3 接口不同)
        is_flux = "flux" in pipeline.__class__.__name__.lower()
        if is_flux:
            prompt_embeds = pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )
        else:
            prompt_embeds = pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                prompt_3=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

        if isinstance(prompt_embeds, tuple):
            encoder_hidden_states = prompt_embeds[0]
            pooled_prompt_embeds = prompt_embeds[1] if len(prompt_embeds) > 1 else None
        else:
            encoder_hidden_states = prompt_embeds
            pooled_prompt_embeds = None

        # 防御性清洗: 上游 embedding 一旦含 NaN/Inf 会在 step0 直接放大为全量非有限输出。
        if not torch.isfinite(encoder_hidden_states).all():
            encoder_hidden_states = torch.nan_to_num(
                encoder_hidden_states, nan=0.0, posinf=1e3, neginf=-1e3
            )
            encoder_hidden_states = torch.clamp(encoder_hidden_states, min=-1e3, max=1e3)
            main_print("[WARN] Non-finite encoder_hidden_states detected and sanitized")
        if pooled_prompt_embeds is not None and (not torch.isfinite(pooled_prompt_embeds).all()):
            pooled_prompt_embeds = torch.nan_to_num(
                pooled_prompt_embeds, nan=0.0, posinf=1e3, neginf=-1e3
            )
            pooled_prompt_embeds = torch.clamp(pooled_prompt_embeds, min=-1e3, max=1e3)
            main_print("[WARN] Non-finite pooled_prompt_embeds detected and sanitized")

        # 文本编码完成, 立刻把 text encoder 卸回 CPU 释放 ~10 GB 显存
        for _attr in _enc_attrs:
            _enc = getattr(pipeline, _attr, None)
            if _enc is not None:
                _enc.to("cpu")
        torch.cuda.empty_cache()

    # Latent 尺寸
    H, W = config["dataset"]["resolution"]
    latent_h, latent_w = H // 8, W // 8
    latent_channels = 16  # SD3/Flux latent channels
    latent_shape = (1, latent_channels, latent_h, latent_w)

    # Position ids: Flux 用 2x2 patch (token 数 = latent//2), SD3 直接展平
    _txt_seq = encoder_hidden_states.shape[1]
    if is_flux:
        _img_tokens = (latent_h // 2) * (latent_w // 2)
    else:
        _img_tokens = latent_h * latent_w
    # 带 batch 维 (1, N, 3), Flux transformer 要求 txt_ids/img_ids 有 batch 维
    text_ids = torch.zeros(1, _txt_seq, 3, device=device)
    latent_image_ids = torch.zeros(1, _img_tokens, 3, device=device)

    # ═══ 3. 树状采样 (no_grad) ═══
    # FSDP: 必须传 wrapped model, 否则 params 处于 shard 状态 (1D) 导致 matmul 报错
    branches = tree_sampler.sample(
        transformer=transformer,
        is_flux=is_flux,
        guidance_scale=guidance_scale,
        debug_validate_finite=debug_nan_trace,
        latent_shape=latent_shape,
        encoder_hidden_states=encoder_hidden_states,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
        latent_image_ids=latent_image_ids,
        split_steps=split_steps,
        noise_levels=noise_levels,
        device=device,
        dtype=dtype,
    )

    K = len(branches)
    main_print(f"[Step {step}] Sampled {K} branches")

    # ═══ 4. VAE 解码 + 奖励计算 ═══
    # Gap2 修复: torch.cat 而非 torch.stack (采样后 latent 已是 4D)
    all_latents = torch.cat([b["latent"] for b in branches], dim=0).to(device)  # (K,C,H,W)
    prompts_list = [prompt] * K

    rewards, rewards_dict = decode_and_compute_rewards(
        latents=all_latents,
        vae=vae,
        prompts=prompts_list,
        reward_models=reward_models,
        reward_weights=reward_weights,
        mix_strategy=config["reward"].get("mix_strategy", "advantage_aggr"),
        batch_size=train_cfg.get("vae_decode_batch_size", 4),
    )
    rewards = rewards.to(device)
    # 每个奖励模型的平均分 (用于 wandb 诊断)
    per_model_rewards = {name: scores.float().mean().item()
                        for name, scores in rewards_dict.items()}

    # 每张卡只持有本地 K 个分支的奖励, log_probs 也是本地 K 个——不要 gather
    # 但调度器需要全局奖励平均来更新边界, 单独算一个全局均值
    if accelerator.num_processes > 1:
        rewards_global = gather_rewards(accelerator, rewards)
    else:
        rewards_global = rewards

    # 更新调度器的奖励边界
    scheduler.update_reward_bounds(rewards_global)
    mean_reward = rewards_global.mean().item()

    # ═══ 5. 旧策略 log_prob (detach) ═══
    old_log_probs = torch.tensor(
        [b["log_prob_sum"] for b in branches],
        device=device, dtype=torch.float32,
    )

    # ═══ 6. 参考模型 log_prob ═══
    # 注意: 这里用的是本轮采样时的 log_prob, 不是预训练参考模型的。
    # 功能上等同于信赖域约束 (限制策略不偏离本轮采样时太远),
    # 而非严格的 KL(π_θ || π_ref) 参考约束。
    ref_log_probs = old_log_probs.clone().detach()

    # ═══ 7. 路径特征 (for 粒子熵) ═══
    path_features = ParticleEntropyLoss.compute_latent_features(all_latents.float())

    # ═══ 8. IS 多次更新 ═══
    all_metrics = []
    sigmas = build_sigma_schedule(
        tree_cfg["num_inference_steps"], tree_cfg.get("shift", 3.0), device
    )

    # Bug4 修复: 动态对齐 is_num_updates 为 grad_accum 的倍数
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    raw_updates = loss_cfg.get("is_num_updates", 4)
    is_num_updates = math.ceil(raw_updates / grad_accum) * grad_accum

    for update_iter in range(is_num_updates):
        _recompute_sub_batch = int(train_cfg.get("recompute_sub_batch", 9))
        _recompute_forward_dtype = dtype
        _recompute_use_autocast = True
        if debug_recompute_fp32:
            _recompute_sub_batch = min(_recompute_sub_batch, 2)
            if is_flux:
                if debug_grad_diag and (step % debug_every == 0):
                    main_print(
                        f"[DEBUG-CFG] Step {step} iter {update_iter}: "
                        "recompute_fp32 requested, but Flux transformer forward stays in bf16 autocast "
                        "for numerical stability; fp32 is still used in log-prob math."
                    )
            else:
                _recompute_forward_dtype = torch.float32
                _recompute_use_autocast = False
            if debug_grad_diag and (step % debug_every == 0):
                main_print(
                    f"[DEBUG-CFG] Step {step} iter {update_iter}: "
                    f"recompute_fp32=True, recompute_sub_batch={_recompute_sub_batch}, "
                    f"forward_dtype={_recompute_forward_dtype}, autocast={_recompute_use_autocast}"
                )

        # 重新计算当前策略的 log_prob (需要梯度) + RatioNorm 逐步数据
        recompute_result = tree_sampler.recompute_path_log_probs(
            transformer=transformer,
            branches=branches,
            split_steps=split_steps,
            noise_levels=noise_levels,
            sigmas=sigmas,
            encoder_hidden_states=encoder_hidden_states,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            latent_image_ids=latent_image_ids,
            dtype=_recompute_forward_dtype,
            is_flux=is_flux,
            guidance_scale=guidance_scale,
            recompute_sub_batch=_recompute_sub_batch,
            use_autocast=_recompute_use_autocast,
            debug_validate_finite=debug_nan_trace,
        )

        current_log_probs = recompute_result["path_log_probs"]

        # autograd 连通性检查: 若 current_log_probs 不可导, 本步必然无梯度
        if debug_grad_diag and (step % debug_every == 0):
            _req = torch.tensor(
                1.0 if current_log_probs.requires_grad else 0.0,
                device=device,
            )
            _req_all = accelerator.gather(_req).view(-1)
            _req_min = _req_all.min().item()
            _req_max = _req_all.max().item()
            main_print(
                f"[AUTOGRAD-DIAG] Step {step} iter {update_iter}: "
                f"path_log_probs.requires_grad(local={bool(_req.item())}, "
                f"global_min={_req_min:.0f}, global_max={_req_max:.0f})"
            )
            if (not current_log_probs.requires_grad) and debug_fail_fast:
                raise RuntimeError(
                    f"path_log_probs detached at step={step}, iter={update_iter}."
                )

        if debug_tensor_stats and (step % debug_every == 0):
            _lp = current_log_probs.detach().float()
            main_print(
                f"[DEBUG-TENSOR] Step {step} iter {update_iter}: "
                f"log_prob min={_lp.min().item():.4f} max={_lp.max().item():.4f} "
                f"mean={_lp.mean().item():.4f} finite={torch.isfinite(_lp).all().item()}"
            )

        # 计算总损失 (传入 RatioNorm 逐步数据)
        loss, metrics = loss_fn(
            current_log_probs=current_log_probs,
            old_log_probs=old_log_probs,
            rewards=rewards,
            ref_log_probs=ref_log_probs,
            path_features=path_features.detach(),
            num_sde_steps=len(split_steps),
            step_log_probs=recompute_result["step_log_probs"],
            old_step_log_probs=recompute_result["old_step_log_probs"],
            step_means=recompute_result["step_means"],
            old_step_means=recompute_result["old_step_means"],
            std_dev_ts=recompute_result["std_dev_ts"],
            sqrt_dts=recompute_result["sqrt_dts"],
        )

        if debug_mid_stats and (step % debug_every == 0):
            _old_lp = old_log_probs.detach().float()
            _new_lp = current_log_probs.detach().float()
            _delta_lp = _new_lp - _old_lp
            main_print(
                f"[MID-DIAG] Step {step} iter {update_iter}: "
                f"old_lp(mean={_old_lp.mean().item():.4f},std={_old_lp.std().item():.4f}) "
                f"new_lp(mean={_new_lp.mean().item():.4f},std={_new_lp.std().item():.4f}) "
                f"delta_lp(mean={_delta_lp.mean().item():.4f},std={_delta_lp.std().item():.4f})"
            )

            _rw = rewards.detach().float()
            main_print(
                f"[MID-DIAG] Step {step} iter {update_iter}: "
                f"reward(mean={_rw.mean().item():.4f},std={_rw.std().item():.4f},"
                f"min={_rw.min().item():.4f},max={_rw.max().item():.4f}) "
                f"weights(mean={metrics.get('is_weight_mean', 0):.4f},std={metrics.get('is_weight_std', 0):.4f})"
            )

        if debug_step_stats and (step % debug_every == 0):
            _stds = recompute_result.get("std_dev_ts", [])
            _sqrts = recompute_result.get("sqrt_dts", [])
            for _i in range(min(len(split_steps), len(_stds), len(_sqrts))):
                _sigma = sigmas[split_steps[_i]].item()
                _sigma_next = sigmas[split_steps[_i] + 1].item()
                _noise_scale = float(_stds[_i]) * float(_sqrts[_i])
                main_print(
                    f"[STEP-DIAG] Step {step} iter {update_iter} sde#{_i}: "
                    f"split={split_steps[_i]} sigma={_sigma:.6f} sigma_next={_sigma_next:.6f} "
                    f"eta={noise_levels[_i]:.4f} std_dev_t={float(_stds[_i]):.6f} "
                    f"sqrt_dt={float(_sqrts[_i]):.6f} noise_scale={_noise_scale:.6f}"
                )

        if debug_grad_diag and (step % debug_every == 0):
            _loss_req = torch.tensor(1.0 if loss.requires_grad else 0.0, device=device)
            _loss_req_all = accelerator.gather(_loss_req).view(-1)
            _loss_req_min = _loss_req_all.min().item()
            _loss_req_max = _loss_req_all.max().item()
            main_print(
                f"[AUTOGRAD-DIAG] Step {step} iter {update_iter}: "
                f"loss.requires_grad(local={bool(_loss_req.item())}, "
                f"global_min={_loss_req_min:.0f}, global_max={_loss_req_max:.0f})"
            )
            if (not loss.requires_grad) and debug_fail_fast:
                raise RuntimeError(
                    f"loss detached at step={step}, iter={update_iter}."
                )

        if debug_tensor_stats and (step % debug_every == 0):
            _loss_finite = torch.isfinite(loss.detach()).item()
            main_print(
                f"[DEBUG-TENSOR] Step {step} iter {update_iter}: "
                f"loss={loss.detach().item():.6f} finite={_loss_finite} "
                f"ratio_mean={metrics.get('ratio_mean', 0):.4f} "
                f"ratio_std={metrics.get('ratio_std', 0):.4f}"
            )

        # ══ FSDP 梯度累积: 非最后一步用 no_sync() 压制 reduce-scatter ══
        # 根因: FSDP SHARD_GRAD_OP 每次 backward() 默认触发一次 reduce-scatter。
        # 不加 no_sync() 时, 3 次 backward 会产生 3 次 reduce-scatter:
        #   accum iter 0: reduce-scatter(g0) → rank_i 得到 8*g0 (8 GPU sum)
        #   accum iter 1: reduce-scatter(8*g0 + g1) → rank_i 得到 8*(8*g0+g1) = 64*g0 + 8*g1
        #   accum iter 2: reduce-scatter(64*g0 + 8*g1 + g2) → rank_i 得到 512*g0 + ...
        # g0 被放大 512×(8^3) → bf16 溢出 → NaN → nan_to_num_ 掩盖为 0 → grad_norm=0.000
        # 正确做法: 前 (grad_accum-1) 次 backward 用 no_sync() 只做本地累积,
        # 最后一次 backward 不加 no_sync() → 触发一次 reduce-scatter → 梯度正确同步
        is_last_accum = (update_iter + 1) % grad_accum == 0
        sync_ctx = contextlib.nullcontext() if is_last_accum else transformer.no_sync()
        scaled_loss = (loss * backprop_scale) / grad_accum

        # 定位“第一个变 NaN 的梯度”:
        # 在 backward 期间给参数挂临时 hook, 一旦出现非有限梯度立即打印参数名和统计。
        _hook_handles = []
        _first_nonfinite = {"hit": False, "name": ""}
        if debug_nan_trace and (step % debug_every == 0):
            def _make_grad_hook(_name: str):
                def _grad_hook(_g):
                    if _g is None:
                        return _g
                    if (not _first_nonfinite["hit"]) and (not torch.isfinite(_g).all()):
                        _first_nonfinite["hit"] = True
                        _first_nonfinite["name"] = _name
                        _msg = (
                            f"[NAN-TRACE] Step {step} iter {update_iter}: "
                            f"first_nonfinite_grad_param={_name} "
                            f"has_nan={torch.isnan(_g).sum().item()} has_inf={torch.isinf(_g).sum().item()} "
                            f"grad_abs_max={_g.abs().max().item()}"
                        )
                        main_print(_msg)
                        rank_debug_print(_msg)
                        if debug_nan_trace_raise:
                            raise RuntimeError(_msg)
                    return _g
                return _grad_hook

            for _n, _p in _iter_unique_optimizer_named_params(optimizer):
                if _p.requires_grad:
                    _hook_handles.append(_p.register_hook(_make_grad_hook(_n)))

        with sync_ctx:
            accelerator.backward(scaled_loss)

        for _h in _hook_handles:
            _h.remove()

        if debug_nan_trace and (step % debug_every == 0) and (not _first_nonfinite["hit"]):
            main_print(f"[NAN-TRACE] Step {step} iter {update_iter}: no non-finite grad observed during backward")

        # 每 grad_accum 步更新一次 (保证整除)
        if is_last_accum:
            # 可选: 先清理非有限梯度, 避免 clip_grad_norm_ 直接返回 NaN 导致每步都跳过
            _sanitized_param_count = 0
            _pre_nonfinite_param_count = 0
            _pre_nonfinite_printed = 0
            _optim_named_params = list(_iter_unique_optimizer_named_params(optimizer))
            _optim_params = [p for _, p in _optim_named_params]
            if debug_safe_grad:
                for _pname, _p in _optim_named_params:
                    if _p.grad is None:
                        continue
                    if not torch.isfinite(_p.grad).all():
                        _pre_nonfinite_param_count += 1
                        if _pre_nonfinite_printed < debug_nan_param_limit:
                            _pre_nonfinite_printed += 1
                            _msg = (
                                f"[PRE-NONFINITE] Step {step} iter {update_iter}: "
                                f"param={_pname} has_nan={torch.isnan(_p.grad).sum().item()} "
                                f"has_inf={torch.isinf(_p.grad).sum().item()} "
                                f"abs_max={_p.grad.abs().max().item()}"
                            )
                            main_print(_msg)
                            rank_debug_print(_msg)
                        _p.grad = torch.nan_to_num(_p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                        _sanitized_param_count += 1

            _max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
            grad_norm = accelerator.clip_grad_norm_(
                _optim_params, _max_grad_norm
            )
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
            grad_clip_coef = min(1.0, _max_grad_norm / (abs(float(grad_norm)) + 1e-12))

            # 无论是否开启 debug，都统一汇总全局 grad_norm，避免主进程本地 shard 误导为 0。
            _gn_t = torch.tensor([float(grad_norm)], device=device, dtype=torch.float32)
            _gn_all = accelerator.gather(_gn_t).view(-1)
            global_gnorm_min = float(_gn_all.min().item())
            global_gnorm_max = float(_gn_all.max().item())

            _need_verbose_diag = debug_grad_diag and (step % debug_every == 0)
            _need_nan_scan = _need_verbose_diag or math.isnan(grad_norm) or math.isinf(grad_norm)

            _nan_count = 0
            _has_grad_count = 0
            _total_count = 0
            global_has = float(_has_grad_count)
            global_nan = float(_nan_count)
            if _need_nan_scan:
                for _pname, _p in _optim_named_params:
                    if not _p.requires_grad:
                        continue
                    _total_count += 1
                    if _p.grad is None:
                        continue
                    _has_grad_count += 1
                    if torch.isnan(_p.grad).any() or torch.isinf(_p.grad).any():
                        _nan_count += 1
                        if _nan_count <= debug_nan_param_limit:
                            _msg = (
                                f"[NaN-DIAG] Step {step} iter {update_iter}: "
                                f"param={_pname} grad_shape={tuple(_p.grad.shape)} "
                                f"has_nan={torch.isnan(_p.grad).sum().item()} "
                                f"has_inf={torch.isinf(_p.grad).sum().item()} "
                                f"grad_abs_max={_p.grad.abs().max().item()}"
                            )
                            main_print(_msg)
                            rank_debug_print(_msg)

                _rank_vec = torch.tensor(
                    [float(_total_count), float(_has_grad_count), float(_nan_count), float(grad_norm)],
                    device=device,
                    dtype=torch.float32,
                )
                _rank_all = accelerator.gather(_rank_vec).view(-1, 4).cpu()
                global_total = float(_rank_all[:, 0].sum().item())
                global_has = float(_rank_all[:, 1].sum().item())
                global_nan = float(_rank_all[:, 2].sum().item())
                global_has_min = float(_rank_all[:, 1].min().item())
                global_has_max = float(_rank_all[:, 1].max().item())
                global_nan_max = float(_rank_all[:, 2].max().item())
                global_gnorm_min = float(_rank_all[:, 3].min().item())
                global_gnorm_max = float(_rank_all[:, 3].max().item())

                main_print(
                    f"[GRAD-DIAG] Step {step} iter {update_iter}: "
                    f"local(trainable={_total_count}, has_grad={_has_grad_count}, nan_params={_nan_count}, grad_norm={grad_norm}) "
                    f"global(sum_trainable={int(global_total)}, sum_has_grad={int(global_has)}, sum_nan_params={int(global_nan)}, "
                    f"has_grad_min={int(global_has_min)}, has_grad_max={int(global_has_max)}, nan_params_max={int(global_nan_max)}, "
                    f"grad_norm_min={global_gnorm_min}, grad_norm_max={global_gnorm_max})"
                )

                # 每个 rank 的 has_grad/nan_params 状态向量, 直接定位哪张卡先坏
                _rank_vec2 = torch.tensor(
                    [float(_has_grad_count), float(_nan_count)],
                    device=device,
                    dtype=torch.float32,
                )
                _rank_all2 = accelerator.gather(_rank_vec2).view(-1, 2).cpu().tolist()
                if accelerator.is_main_process:
                    _rank_msg = ", ".join(
                        [f"r{i}(has={int(v[0])},nan={int(v[1])})" for i, v in enumerate(_rank_all2)]
                    )
                    main_print(f"[GRAD-RANKS] Step {step} iter {update_iter}: {_rank_msg}")

                if debug_safe_grad:
                    _san_t = torch.tensor(float(_sanitized_param_count), device=device)
                    _pre_t = torch.tensor(float(_pre_nonfinite_param_count), device=device)
                    _san_sum = accelerator.reduce(_san_t.clone(), reduction="sum").item()
                    _pre_sum = accelerator.reduce(_pre_t.clone(), reduction="sum").item()
                    main_print(
                        f"[SAFE-GRAD] Step {step} iter {update_iter}: "
                        f"pre_nonfinite_local={_pre_nonfinite_param_count}, pre_nonfinite_global={int(_pre_sum)}, "
                        f"sanitized_local={_sanitized_param_count}, sanitized_global={int(_san_sum)}"
                    )

                rank_debug_print(
                    f"[GRAD-DIAG-LOCAL] Step {step} iter {update_iter}: "
                    f"trainable={_total_count}, has_grad={_has_grad_count}, nan_params={_nan_count}, grad_norm={grad_norm}"
                )

                # FSDP SHARD_GRAD_OP 下局部 rank 可能无梯度(该 rank 不持有对应 shard)，
                # 以全局 sum_has_grad 判定是否“真正无梯度”。
                if int(global_total) > 0 and int(global_has) == 0:
                    main_print(
                        f"[WARNING] Step {step} iter {update_iter}: "
                        "loss has graph but optimizer params received no gradients. "
                        "Check [AUTOGRAD-DIAG] and [OPTIM-PARAMS] lines above."
                    )

            metrics["debug_grad_trainable"] = _total_count
            metrics["debug_grad_has_grad"] = _has_grad_count
            metrics["debug_grad_nan_params"] = _nan_count
            metrics["debug_grad_has_grad_global"] = int(global_has)
            metrics["debug_grad_nan_params_global"] = int(global_nan)
            metrics["grad_clip_coef"] = float(grad_clip_coef)
            metrics["max_grad_norm"] = _max_grad_norm

            # grad_norm 已通过 FSDP all-reduce 全卡统一 → NaN 则全卡一致跳过, 无死锁
            grad_norm_for_log = float(global_gnorm_max)
            if math.isnan(grad_norm_for_log) or math.isinf(grad_norm_for_log):
                main_print(f"[WARNING] Step {step} iter {update_iter}: grad_norm={grad_norm}, 跳过 optimizer.step()")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                grad_norm = 0.0
            else:
                metrics["grad_norm"] = grad_norm_for_log
                accelerator.wait_for_everyone()  # 确保全卡在同一 barrier 后再 step
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

        # 不在 inner update 中覆盖 old_log_probs：
        # old_log_probs 应保持为 rollout 时的旧策略基线，
        # 否则 approx_kl 会被“当前对当前”比较压成 0。

        metrics["update_iter"] = update_iter
        metrics["mean_reward"] = mean_reward
        metrics["alpha"] = alpha
        metrics["per_model_rewards"] = per_model_rewards
        metrics["reward_max"] = rewards_global.max().item()
        metrics["reward_min"] = rewards_global.min().item()
        metrics["num_branches"] = K
        all_metrics.append(metrics)

        # 及时释放本轮 recompute 相关大张量/计算图，避免下一轮 update_iter 重算时峰值叠加。
        if isinstance(recompute_result, dict):
            for _k in [
                "step_log_probs", "old_step_log_probs", "step_means",
                "old_step_means", "path_log_probs",
            ]:
                if _k in recompute_result:
                    recompute_result[_k] = None
        del recompute_result
        del current_log_probs
        del loss
        del scaled_loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_metrics[-1], mean_reward


# ═══════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="TreeMatch-RL Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="YAML config file path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint dir")

    # ═══ CLI 参数覆盖 (优先级 > YAML) ═══
    # Training
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides YAML)")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="VAE decode batch size")
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--recompute_sub_batch", type=int, default=None, help="Max branch count per recompute forward chunk")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Max gradient norm")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    # Loss
    parser.add_argument("--beta", type=float, default=None, help="Softmax-TB temperature β")
    parser.add_argument("--lambda_entropy", type=float, default=None, help="Entropy regularization weight")
    parser.add_argument("--lambda_ref", type=float, default=None, help="Reference constraint weight")
    parser.add_argument("--is_clip_range", type=float, default=None, help="IS weight clip range")
    parser.add_argument("--is_num_updates", type=int, default=None, help="IS update iterations per step")
    # Tree
    parser.add_argument("--kappa", type=float, default=None, help="Beta distribution concentration κ")
    parser.add_argument("--num_inference_steps", type=int, default=None, help="Total inference steps")
    parser.add_argument("--tree_k", type=int, default=None, help="Branching factor k")
    # Model
    parser.add_argument("--model_path", type=str, default=None, help="Override pretrained model path (absolute or relative)")
    # Wandb
    parser.add_argument("--wandb_project", type=str, default="treematch-rl", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name (auto-generated if None)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb_log_images_every", type=int, default=10, help="Log sample images every N steps")
    # Debug
    parser.add_argument("--debug_grad_diag", action="store_true", help="Enable gradient diagnostics output")
    parser.add_argument("--debug_tensor_stats", action="store_true", help="Enable tensor finite/min/max debug output")
    parser.add_argument("--debug_every", type=int, default=None, help="Print debug output every N global steps")
    parser.add_argument("--debug_nan_param_print_limit", type=int, default=None, help="Max NaN gradient params to print per step")
    parser.add_argument("--debug_all_ranks", action="store_true", help="Print gradient diagnostics on every rank")
    parser.add_argument("--debug_fail_fast", action="store_true", help="Raise error when autograd graph is detached")
    parser.add_argument("--debug_mid_stats", action="store_true", help="Print intermediate old/new log_prob and reward/weight stats")
    parser.add_argument("--debug_step_stats", action="store_true", help="Print per-SDE-step sigma/std/noise_scale stats")
    parser.add_argument("--debug_safe_grad", action="store_true", help="Sanitize NaN/Inf gradients before clip_grad_norm")
    parser.add_argument("--debug_recompute_fp32", action="store_true", help="Disable autocast in recompute forward and run it in fp32")
    parser.add_argument("--debug_nan_trace", action="store_true", help="Trace first non-finite tensor/grad in recompute and backward")
    parser.add_argument("--debug_nan_trace_raise", action="store_true", help="Raise immediately when first non-finite grad is observed")
    parser.add_argument("--backprop_scale", type=float, default=None, help="Scale factor applied to loss before backward (e.g., 0.1)")
    # Fixed schedule
    parser.add_argument("--force_fixed_schedule", action="store_true", help="Force fixed split steps and noise levels")
    parser.add_argument("--fixed_split_steps", type=str, default=None, help="CSV split steps, e.g. 7,14,21")
    parser.add_argument("--fixed_noise_levels", type=str, default=None, help="CSV noise levels, e.g. 0.4,0.7,1.0")

    args = parser.parse_args()

    # 加载 YAML 配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # CLI 参数覆盖 YAML
    train_cfg = config["training"]
    tree_cfg = config["tree"]
    loss_cfg = config["loss"]
    debug_cfg = config.setdefault("debug", {})

    _override = [
        (args.lr,               train_cfg, "learning_rate"),
        (args.max_steps,        train_cfg, "max_train_steps"),
        (args.batch_size,       train_cfg, "vae_decode_batch_size"),
        (args.grad_accum,       train_cfg, "gradient_accumulation_steps"),
        (args.recompute_sub_batch, train_cfg, "recompute_sub_batch"),
        (args.seed,             train_cfg, "seed"),
        (args.max_grad_norm,    train_cfg, "max_grad_norm"),
        (args.output_dir,       train_cfg, "output_dir"),
        (args.beta,             loss_cfg,  "beta"),
        (args.lambda_entropy,   loss_cfg,  "lambda_entropy"),
        (args.lambda_ref,       loss_cfg,  "lambda_ref"),
        (args.is_clip_range,    loss_cfg,  "is_clip_range"),
        (args.is_num_updates,   loss_cfg,  "is_num_updates"),
        (args.kappa,            tree_cfg,  "kappa"),
        (args.num_inference_steps, tree_cfg, "num_inference_steps"),
        (args.tree_k,           tree_cfg,  "k"),
    ]
    for val, cfg_dict, key in _override:
        if val is not None:
            cfg_dict[key] = val

    # 固定调度配置
    if args.force_fixed_schedule:
        tree_cfg["force_fixed_schedule"] = True
    if args.fixed_split_steps is not None:
        tree_cfg["fixed_split_steps"] = _parse_int_list(args.fixed_split_steps)
    if args.fixed_noise_levels is not None:
        tree_cfg["fixed_noise_levels"] = _parse_float_list(args.fixed_noise_levels)

    tree_cfg.setdefault("force_fixed_schedule", False)
    tree_cfg.setdefault("fixed_split_steps", [7, 14, 21])
    tree_cfg.setdefault("fixed_noise_levels", [0.4, 0.7, 1.0])

    # Debug 配置: YAML 可配, CLI 可覆盖
    if args.debug_grad_diag:
        debug_cfg["grad_diag"] = True
    if args.debug_tensor_stats:
        debug_cfg["tensor_stats"] = True
    if args.debug_every is not None:
        debug_cfg["every"] = args.debug_every
    if args.debug_nan_param_print_limit is not None:
        debug_cfg["nan_param_print_limit"] = args.debug_nan_param_print_limit
    if args.debug_all_ranks:
        debug_cfg["all_ranks"] = True
    if args.debug_fail_fast:
        debug_cfg["fail_fast"] = True
    if args.debug_mid_stats:
        debug_cfg["mid_stats"] = True
    if args.debug_step_stats:
        debug_cfg["step_stats"] = True
    if args.debug_safe_grad:
        debug_cfg["safe_grad"] = True
    if args.debug_recompute_fp32:
        debug_cfg["recompute_fp32"] = True
    if args.debug_nan_trace:
        debug_cfg["nan_trace"] = True
    if args.debug_nan_trace_raise:
        debug_cfg["nan_trace_raise"] = True
    if args.backprop_scale is not None:
        debug_cfg["backprop_scale"] = args.backprop_scale

    debug_cfg.setdefault("grad_diag", False)
    debug_cfg.setdefault("tensor_stats", False)
    debug_cfg.setdefault("every", 1)
    debug_cfg.setdefault("nan_param_print_limit", 3)
    debug_cfg.setdefault("all_ranks", False)
    debug_cfg.setdefault("fail_fast", False)
    debug_cfg.setdefault("mid_stats", False)
    debug_cfg.setdefault("step_stats", False)
    debug_cfg.setdefault("safe_grad", False)
    debug_cfg.setdefault("recompute_fp32", False)
    debug_cfg.setdefault("nan_trace", False)
    debug_cfg.setdefault("nan_trace_raise", False)
    debug_cfg.setdefault("backprop_scale", 1.0)

    train_cfg["recompute_sub_batch"] = max(1, int(train_cfg.get("recompute_sub_batch", 9)))

    # 模型路径覆盖
    model_cfg = config["model"]
    if args.model_path is not None:
        model_cfg["pretrained_path"] = args.model_path

    # ═══ Accelerator ═══
    # 注意: 不传 gradient_accumulation_steps 给 Accelerator, 因为训练循环手动
    # 用 transformer.no_sync() 控制 FSDP 梯度同步时机。如果传了 >1 的值,
    # Accelerator 的 sync_gradients 内部计数器不会被 accumulate() 推进,
    # 导致 clip_grad_norm_ 等操作可能行为异常。
    accelerator = Accelerator(
        mixed_precision=train_cfg.get("mixed_precision", "bf16"),
    )

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    # 每个 seed 独立输出目录: <output_dir>/seed_<seed>
    base_output_dir = str(train_cfg.get("output_dir", "outputs"))
    run_output_dir = os.path.join(base_output_dir, f"seed_{seed}")
    train_cfg["output_dir"] = run_output_dir
    os.makedirs(run_output_dir, exist_ok=True)

    # 日志
    logger = setup_logging(
        log_dir=run_output_dir,
        rank=accelerator.process_index,
    )
    main_print("=" * 60)
    main_print("TreeMatch-RL Training")
    main_print("=" * 60)
    main_print(f"Config: {args.config}")
    main_print(f"Output dir: {run_output_dir}")
    main_print(f"Num GPUs: {accelerator.num_processes}")
    main_print(
        f"Debug: grad_diag={debug_cfg['grad_diag']} "
        f"tensor_stats={debug_cfg['tensor_stats']} "
        f"every={debug_cfg['every']} nan_limit={debug_cfg['nan_param_print_limit']} "
        f"all_ranks={debug_cfg['all_ranks']} fail_fast={debug_cfg['fail_fast']} "
        f"mid_stats={debug_cfg['mid_stats']} step_stats={debug_cfg['step_stats']} "
        f"safe_grad={debug_cfg['safe_grad']} recompute_fp32={debug_cfg['recompute_fp32']} "
        f"nan_trace={debug_cfg['nan_trace']} nan_trace_raise={debug_cfg['nan_trace_raise']} "
        f"backprop_scale={debug_cfg['backprop_scale']}"
    )
    main_print(
        f"Schedule: force_fixed={tree_cfg['force_fixed_schedule']} "
        f"splits={tree_cfg['fixed_split_steps']} noise={tree_cfg['fixed_noise_levels']}"
    )
    main_print(
        f"Training: grad_accum={train_cfg.get('gradient_accumulation_steps', 1)} "
        f"vae_decode_batch={train_cfg.get('vae_decode_batch_size', 4)} "
        f"recompute_sub_batch={train_cfg['recompute_sub_batch']}"
    )
    if "flux" in str(model_cfg.get("pretrained_path", "")).lower():
        main_print(f"Flux guidance_scale={float(model_cfg.get('guidance_scale', 3.5))}")

    # 打印完整配置摘要
    print_config_table(config, accelerator)

    # ═══ Wandb 初始化 ═══
    use_wandb = (wandb is not None) and (not args.no_wandb) and accelerator.is_main_process
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"{train_cfg.get('experiment_name', 'treematch')}_{seed}",
            config=config,
            tags=["treematch-rl", model_cfg.get("pretrained_path", "unknown").rstrip("/").split("/")[-1]],
        )
        main_print(f"[Wandb] Project: {args.wandb_project}, Run: {wandb.run.name}")
    else:
        main_print("[Wandb] Disabled")

    if train_cfg.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 某些 batch/shape 组合下 cudnn v8 plan 会回退并打印 warning；
    # 这是已知的可恢复路径，不影响正确性，但会刷屏。
    if train_cfg.get("cudnn_safe_mode", True):
        torch.backends.cudnn.benchmark = False
        warnings.filterwarnings(
            "ignore",
            message=r"Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR",
            category=UserWarning,
        )

    # ═══ 加载模型 ═══
    model_cfg = config["model"]
    main_print(f"Loading model: {model_cfg['pretrained_path']}")

    # 自动检测 Flux vs SD3
    model_path = model_cfg["pretrained_path"].lower()
    is_flux = "flux" in model_path

    # 根据 mixed_precision 配置决定加载 dtype
    mp = train_cfg.get("mixed_precision", "no")
    load_dtype = torch.bfloat16 if mp == "bf16" else torch.float16 if mp == "fp16" else torch.float32

    if is_flux:
        if FluxPipeline is None:
            raise ImportError("FluxPipeline not found. Please upgrade diffusers: pip install diffusers>=0.30.0")
        pipeline = FluxPipeline.from_pretrained(
            model_cfg["pretrained_path"],
            torch_dtype=load_dtype,
        )
    else:
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_cfg["pretrained_path"],
            torch_dtype=load_dtype,
        )

    transformer = pipeline.transformer
    vae = pipeline.vae
    vae.requires_grad_(False)
    vae.to(accelerator.device)

    # text encoder(s) 暂放 CPU; 文本编码时临时搬 GPU, 编码后立即移回 CPU
    # 目的: T5-XXL ~9.4 GB + CLIP ~0.8 GB = 10 GB 常驻 VRAM 会与 SHARD_GRAD_OP 的
    # 全量参数 (24 GB) + 计算图激活 (~25 GB) 叠加导致 140 GB 卡 OOM
    for _enc_attr in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        _enc = getattr(pipeline, _enc_attr, None)
        if _enc is not None:
            _enc.requires_grad_(False)
            _enc.to("cpu")  # 先放 CPU, encode_prompt 调用前会临时移到 GPU

    # ═══ LoRA ═══
    lora_cfg = model_cfg.get("lora", {})
    if lora_cfg:
        main_print(f"Applying LoRA: rank={lora_cfg.get('rank', 32)}")
        lora_config = LoraConfig(
            r=lora_cfg.get("rank", 32),
            lora_alpha=lora_cfg.get("alpha", 32),
            target_modules=lora_cfg.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"]),
            lora_dropout=lora_cfg.get("dropout", 0.0),
            bias="none",
        )
        transformer = get_peft_model(transformer, lora_config)
        transformer.print_trainable_parameters()
        # FSDP 要求同一个 flatten 组内参数 dtype 一致。
        # 若 LoRA 保持 fp32 而 base 为 bf16, 会在 accelerator.prepare 阶段直接报错。
        transformer = transformer.to(torch.bfloat16)
        main_print("[INFO] Cast transformer(+LoRA) to bf16 for FSDP dtype uniformity")

        # ——— LoRA 权重自检: 确保初始化后无 NaN/Inf ———
        bad_params = []
        for pname, p in transformer.named_parameters():
            if p.requires_grad and (torch.isnan(p).any() or torch.isinf(p).any()):
                bad_params.append(pname)
        if bad_params:
            main_print(f"[WARNING] NaN/Inf detected in {len(bad_params)} LoRA params after init, re-init to zero:")
            for pname in bad_params:
                main_print(f"  {pname}")
                # 符合碞八则: LoRA A ~ N(0, std), B = 0 —— 重置 B 为 0 可安全恢复
                param = dict(transformer.named_parameters())[pname]
                torch.nn.init.zeros_(param.data)
            main_print("[INFO] LoRA weight sanitization complete.")
        else:
            main_print("[INFO] LoRA weight check passed: no NaN/Inf detected.")
    else:
        transformer.requires_grad_(True)

    _set_param_debug_names(transformer)

    # 梯度检查点: 由 FSDP activation checkpointing 统一管理
    # (fsdp_small.yaml: fsdp_activation_checkpointing=true)
    # 不要同时调用 transformer.enable_gradient_checkpointing(),
    # 否则同一个 FluxTransformerBlock 被两层 checkpoint 嵌套包裹:
    #   FSDP checkpoint_wrapper → HF gradient_checkpoint → forward
    # 双重 recompute 的 backward hooks 在 FSDP reduce-scatter 时序上冲突,
    # 可能导致梯度 corruption → NaN。
    # if train_cfg.get("gradient_checkpointing", True):
    #     transformer.enable_gradient_checkpointing()

    # ═══ 优化器 ═══
    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=float(train_cfg.get("learning_rate", 1e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.get("max_train_steps", 300),
        eta_min=float(train_cfg.get("learning_rate", 1e-5)) * 0.1,
    )

    # ═══ 数据集 ═══
    dataset = PromptDataset(config["dataset"]["data_json_path"])
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0,
    )
    main_print(f"Dataset: {len(dataset)} prompts")

    # ═══ Accelerator.prepare ═══
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler,
    )

    _model_trainable = sum(1 for _, p in transformer.named_parameters() if p.requires_grad)
    _optim_trainable = sum(1 for _ in _iter_unique_optimizer_named_params(optimizer))
    main_print(
        f"[OPTIM-PARAMS] model_trainable={_model_trainable}, optimizer_trainable={_optim_trainable}"
    )

    # 可选: LoRA 梯度清洗 hook (在参数梯度产生时执行)
    # 作用: 仅清理/截断 LoRA A/B 梯度, 防止单步全 NaN 扩散到优化器更新
    if train_cfg.get("sanitize_lora_grad", True):
        lora_grad_clip = float(train_cfg.get("lora_grad_clip", 1.0))

        def _make_sanitize_hook(clip_val: float):
            def _hook(grad):
                if grad is None:
                    return grad
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                if clip_val > 0:
                    grad = torch.clamp(grad, min=-clip_val, max=clip_val)
                return grad
            return _hook

        _hook = _make_sanitize_hook(lora_grad_clip)
        _hooked = 0
        for _n, _p in transformer.named_parameters():
            if _p.requires_grad and ("lora_A" in _n or "lora_B" in _n):
                _p.register_hook(_hook)
                _hooked += 1
        main_print(
            f"[INFO] sanitize_lora_grad=True: registered hooks on {_hooked} LoRA params, clip={lora_grad_clip}"
        )

    # ═══ 奖励模型 ═══
    reward_models, reward_weights = build_reward_models(
        config["reward"], accelerator.device,
    )
    main_print(f"Reward models: {list(reward_models.keys())}")

    # ═══ 采样器 & 调度器 & 损失 ═══
    tree_sampler = TreeSampler(
        num_inference_steps=tree_cfg.get("num_inference_steps", 28),
        k=tree_cfg.get("k", 3),
        shift=tree_cfg.get("shift", 3.0),
        dpm_flash_enabled=config.get("dpm_flash", {}).get("enabled", True),
        dpm_compress_ratio=config.get("dpm_flash", {}).get("compress_ratio", 0.4),
        dpm_order=config.get("dpm_flash", {}).get("solver_order", 2),
        dpm_solver_type=config.get("dpm_flash", {}).get("solver_type", "midpoint"),
    )

    scheduler = AdaptiveScheduler(
        num_inference_steps=tree_cfg.get("num_inference_steps", 28),
        num_splits=3,
        kappa=tree_cfg.get("kappa", 4.0),
        base_noise_levels=tree_cfg.get("base_noise_levels", [0.4, 0.7, 1.0]),
        tail_guard_steps=tree_cfg.get("tail_guard_steps", 4),
    )

    loss_fn = TreeMatchRLLoss(
        beta=loss_cfg.get("beta", 15.0),
        lambda_entropy=loss_cfg.get("lambda_entropy", 0.01),
        lambda_ref=loss_cfg.get("lambda_ref", 0.1),
        is_clip_range=loss_cfg.get("is_clip_range", 0.2),
        rbf_bandwidth=loss_cfg.get("rbf_bandwidth", 1.0),
        ref_scale=loss_cfg.get("ref_scale", 1.0),
    )

    # ═══ 训练循环 ═══
    max_steps = train_cfg.get("max_train_steps", 300)
    checkpoint_steps = int(train_cfg.get("checkpointing_steps", 50))
    save_state_enabled = bool(train_cfg.get("save_state", False))
    # save_state 体积更大, 默认频率应低于 LoRA checkpoint
    save_state_steps = int(train_cfg.get("save_state_steps", max(checkpoint_steps * 4, 200)))
    output_dir = train_cfg.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    if args.resume is not None:
        # 分布式恢复必须所有 rank 同步调用
        accelerator.wait_for_everyone()
        accelerator.load_state(args.resume)
        accelerator.wait_for_everyone()

        _resume_name = os.path.basename(os.path.normpath(args.resume))
        _m = re.match(r"state-(\d+)$", _resume_name)
        global_step = int(_m.group(1)) if _m else 0
        main_print(f"[Resume] Loaded distributed state from {args.resume}, start_step={global_step}")
    else:
        global_step = 0

    main_print(f"Starting training: {max_steps} steps")
    main_print("-" * 60)

    prev_mean_reward = None
    data_iter = iter(dataloader)

    while global_step < max_steps:
        # 取下一个 prompt
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        prompt = batch["prompt"][0] if isinstance(batch["prompt"], list) else batch["prompt"]

        # ── β 线性 warm-up: beta → beta_target (前 beta_warmup_steps 步) ──
        _beta_init = loss_cfg.get("beta", 3.0)
        _beta_target = loss_cfg.get("beta_target", _beta_init)
        _beta_warmup = loss_cfg.get("beta_warmup_steps", 0)
        if _beta_warmup > 0 and global_step < _beta_warmup:
            loss_fn.soft_tb.beta = _beta_init + (_beta_target - _beta_init) * global_step / _beta_warmup
        else:
            loss_fn.soft_tb.beta = _beta_target

        # 训练一步
        metrics, prev_mean_reward = train_one_step(
            accelerator=accelerator,
            transformer=transformer,
            vae=vae,
            pipeline=pipeline,
            tree_sampler=tree_sampler,
            scheduler=scheduler,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            prompt=prompt,
            reward_models=reward_models,
            reward_weights=reward_weights,
            config=config,
            step=global_step,
            prev_mean_reward=prev_mean_reward,
            is_flux=is_flux,
            debug_cfg=debug_cfg,
        )

        global_step += 1

        # 日志
        if global_step % 1 == 0:
            main_print(
                f"[Step {global_step}/{max_steps}] "
                f"loss={metrics['loss_total']:.6e} "
                f"tb={metrics['loss_soft_tb']:.6e} "
                f"entropy={metrics['loss_entropy']:.6e} "
                f"ref={metrics['loss_ref']:.6e} "
                f"w_ent={metrics.get('loss_entropy_weighted', 0.0):.6e} "
                f"w_ref={metrics.get('loss_ref_weighted', 0.0):.6e} "
                f"reward={metrics['mean_reward']:.6e} "
                f"α={metrics.get('alpha', 0):.2f}"
            )
            main_print(
                f"           "
                f"approx_kl={metrics.get('approx_kl', 0):.6e} "
                f"clipfrac={metrics.get('clipfrac', 0):.2f} "
                f"ratio={metrics.get('ratio_mean', 1):.6f}±{metrics.get('ratio_std', 0):.6f} "
                f"log_prob={metrics.get('log_prob_mean', 0):.1f} "
                f"grad_norm={metrics.get('grad_norm', 0):.6e} "
                f"grad_clip_coef={metrics.get('grad_clip_coef', 1.0):.3f} "
                f"reward_range=[{metrics.get('reward_min', 0):.6e},{metrics.get('reward_max', 0):.6e}]"
            )
            main_print(
                f"           "
                f"loss_ratio(tb/ent/ref)="
                f"{metrics.get('loss_tb_ratio', 0.0):.4f}/"
                f"{metrics.get('loss_entropy_ratio', 0.0):.4f}/"
                f"{metrics.get('loss_ref_ratio', 0.0):.4f}"
            )

        # ═══ Wandb 指标记录 ═══
        if use_wandb:
            wandb_log = {
                # ── 损失组 ──
                "train/loss_total": metrics["loss_total"],
                "train/loss_soft_tb": metrics["loss_soft_tb"],
                "train/loss_entropy": metrics["loss_entropy"],
                "train/loss_ref": metrics["loss_ref"],
                "train/grad_norm": metrics.get("grad_norm", 0),
                "debug/grad_trainable": metrics.get("debug_grad_trainable", 0),
                "debug/grad_has_grad": metrics.get("debug_grad_has_grad", 0),
                "debug/grad_nan_params": metrics.get("debug_grad_nan_params", 0),
                "train/lr": optimizer.param_groups[0]["lr"],
                # ── 奖励组 ──
                "reward/mean": metrics["mean_reward"],
                "reward/std": metrics.get("rewards_std", 0),
                "reward/max": metrics.get("reward_max", 0),
                "reward/min": metrics.get("reward_min", 0),
                # ── IS 诊断组 ──
                "is/approx_kl": metrics.get("approx_kl", 0),
                "is/clipfrac": metrics.get("clipfrac", 0),
                "is/ratio_mean": metrics.get("ratio_mean", 1),
                "is/ratio_std": metrics.get("ratio_std", 0),
                "is/weight_mean": metrics.get("is_weight_mean", 1),
                "is/weight_std": metrics.get("is_weight_std", 0),
                # ── 采样组 ──
                "tree/alpha": metrics.get("alpha", 0),
                "tree/log_prob_mean": metrics.get("log_prob_mean", 0),
                "tree/log_prob_old_mean": metrics.get("log_prob_old_mean", 0),
                "tree/sqrt_dt_sq_mean": metrics.get("sqrt_dt_sq_mean", 1),
                "tree/num_branches": metrics.get("num_branches", 8),
            }
            # 逐个奖励模型的分数
            for model_name, score in metrics.get("per_model_rewards", {}).items():
                wandb_log[f"reward/{model_name}_mean"] = score
            wandb.log(wandb_log, step=global_step)

        # 检查点
        if global_step % checkpoint_steps == 0:
            # 关键: rank0 保存 checkpoint 时其他 rank 必须等待，
            # 否则会先进入下一步 all_gather 导致 NCCL timeout。
            accelerator.wait_for_everyone()
            save_checkpoint(
                accelerator=accelerator,
                transformer=transformer,
                optimizer=optimizer,
                step=global_step,
                epoch=0,
                output_dir=output_dir,
                is_lora=bool(lora_cfg),
                pipeline=pipeline,
                save_optimizer=bool(train_cfg.get("save_optimizer", not bool(lora_cfg))),
            )
            accelerator.wait_for_everyone()

        # 分布式 state 保存: 全 rank 同步执行, 使用 FSDP SHARDED_STATE_DICT 避免全量汇总。
        if save_state_enabled and (global_step % max(1, save_state_steps) == 0):
            state_dir = os.path.join(output_dir, f"state-{global_step}")
            accelerator.wait_for_everyone()
            accelerator.save_state(output_dir=state_dir)
            accelerator.wait_for_everyone()
            main_print(f"[State] Saved distributed state to {state_dir}")

    # 最终保存
    accelerator.wait_for_everyone()
    save_checkpoint(
        accelerator=accelerator,
        transformer=transformer,
        optimizer=optimizer,
        step=global_step,
        epoch=0,
        output_dir=output_dir,
        is_lora=bool(lora_cfg),
        pipeline=pipeline,
        save_optimizer=bool(train_cfg.get("save_optimizer", not bool(lora_cfg))),
    )
    accelerator.wait_for_everyone()

    if save_state_enabled:
        final_state_dir = os.path.join(output_dir, f"state-{global_step}")
        accelerator.wait_for_everyone()
        accelerator.save_state(output_dir=final_state_dir)
        accelerator.wait_for_everyone()
        main_print(f"[State] Saved distributed state to {final_state_dir}")

    main_print("=" * 60)
    main_print("Training complete!")
    main_print("=" * 60)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
