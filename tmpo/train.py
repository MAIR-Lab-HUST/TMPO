"""TMPO main training entry point."""

import os
import sys
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
import torch.distributed as dist
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

from tmpo.sampling import TreeSampler, AdaptiveScheduler, ProgressAwareSplitScheduler, prepare_flux_latent_image_ids
from tmpo.sampling.scheduler import build_sigma_schedule
from tmpo.losses import TMPOLoss
from tmpo.rewards.compute import build_reward_models, decode_and_compute_rewards
from tmpo.eval.diversity import compute_lgmd, CLIPDiversityScorer, DINOv2DiversityScorer
from tmpo.eval.evaluator import InlineEvaluator
from tmpo.utils.logging_ import setup_logging, main_print
from tmpo.utils.checkpoint import save_checkpoint
from tmpo.utils.distributed import gather_rewards
from tmpo.utils.ema import EMAModuleWrapper


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
    """Print formatted configuration summary table."""
    model_cfg = config.get("model", {})
    tree_cfg = config.get("tree", {})
    loss_cfg = config.get("loss", {})
    train_cfg = config.get("training", {})
    reward_cfg = config.get("reward", {})
    ds_cfg = config.get("dataset", {})
    lora_cfg = model_cfg.get("lora", {})

    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║             TMPO Configuration                           ║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║ Model         │ {model_cfg.get('pretrained_path', 'N/A'):<39}║",
        f"║ Resolution    │ {str(ds_cfg.get('resolution', 'N/A')):<39}║",
    ]
    if lora_cfg:
        lines.append(f"║ LoRA          │ rank={lora_cfg.get('rank')}, alpha={lora_cfg.get('alpha')}, dropout={lora_cfg.get('dropout', 0.0):<8}║")
    lines += [
        "╠══════════════════════════════════════════════════════════╣",
        f"║ Tree k        │ {tree_cfg.get('k', 3):<39}║",
        f"║ Tree Roots    │ {tree_cfg.get('num_roots', 1)} roots / {tree_cfg.get('max_leaves', 'N/A')} leaves{'':<21}║",
        f"║ Infer Steps   │ {tree_cfg.get('num_inference_steps', 28):<39}║",
        f"║ Schedule Mode │ {tree_cfg.get('schedule_mode', 'adaptive'):<39}║",
        f"║ Noise Level   │ {tree_cfg.get('noise_level', tree_cfg.get('base_noise_levels', 'N/A'))!s:<39}║",
        f"║ SDE Type      │ {tree_cfg.get('sde_type', 'cps'):<39}║",
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
    """Assign stable debug names to parameters for FSDP/optimizer diagnostics."""
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
    """Iterate unique optimizer params with readable names."""
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


def _broadcast_trainable_state_from_rank0(
    transformer,
    optimizer,
    accelerator,
):
    """Broadcast trainable params and buffers from rank0 to all ranks."""
    if accelerator.num_processes <= 1:
        return
    if not (dist.is_available() and dist.is_initialized()):
        return

    for _name, _param in _iter_unique_optimizer_named_params(optimizer):
        if _param is None:
            continue
        dist.broadcast(_param.data, src=0)

    for _buf in transformer.buffers():
        if _buf is None:
            continue
        dist.broadcast(_buf.data, src=0)


# ═══════════════════════════════════════════════════
# ═══════════════════════════════════════════════════
class PromptDataset(Dataset):
    """Prompt dataset (supports JSON / JSONL / plain text formats)."""

    _OCR_FIELDS    = ("ocr_text", "text", "target_text")
    _DQUOTE_RE     = re.compile(r'"([^"]+)"')

    def __init__(self, json_path: str, auto_extract_ocr: bool = False):
        self.prompts   = []
        self.ocr_texts = []
        self._auto_extract_ocr = auto_extract_ocr

        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        raw_items = []

        if content.startswith("[") or content.startswith("{"):
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    raw_items = list(data.values())
                elif isinstance(data, list):
                    raw_items = data
                else:
                    raw_items = [str(data)]
            except json.JSONDecodeError:
                pass

        if not raw_items:
            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_items.append(json.loads(line))
                except json.JSONDecodeError:
                    raw_items.append(line)

        for item in raw_items:
            if isinstance(item, dict):
                self.prompts.append(item.get("prompt", str(item)))
                ocr = next((item[f] for f in self._OCR_FIELDS if f in item), None)
                if ocr is None and self._auto_extract_ocr:
                    m = self._DQUOTE_RE.search(item.get("prompt", ""))
                    ocr = m.group(1) if m else None
                self.ocr_texts.append(ocr)
            else:
                prompt_str = str(item)
                self.prompts.append(prompt_str)
                if self._auto_extract_ocr:
                    m = self._DQUOTE_RE.search(prompt_str)
                    self.ocr_texts.append(m.group(1) if m else None)
                else:
                    self.ocr_texts.append(None)

        if not self.prompts:
            raise ValueError(f"No prompts loaded from {json_path}")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "ocr_text": self.ocr_texts[idx] or ""}


# ═══════════════════════════════════════════════════
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
    ema = None,
    ocr_text: str = None,
    clip_div_scorer = None,
):
    """Execute one full training step. Returns (metrics, mean_reward)."""
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

    if tree_cfg.get("force_fixed_schedule", False):
        split_steps = list(tree_cfg.get("fixed_split_steps", [7, 14, 21]))
        noise_levels = list(tree_cfg.get("fixed_noise_levels", [0.4, 0.7, 1.0]))
        alpha = float(tree_cfg.get("fixed_alpha", 0.5))
    elif isinstance(scheduler, ProgressAwareSplitScheduler):
        split_steps, noise_levels, alpha = scheduler.get_schedule(step)
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

    with torch.no_grad():
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

        for _attr in _enc_attrs:
            _enc = getattr(pipeline, _attr, None)
            if _enc is not None:
                _enc.to("cpu")
        torch.cuda.empty_cache()

    H, W = config["dataset"]["resolution"]
    latent_h, latent_w = H // 8, W // 8
    latent_channels = 16  # SD3/Flux latent channels
    latent_shape = (1, latent_channels, latent_h, latent_w)

    _txt_seq = encoder_hidden_states.shape[1]
    text_ids = torch.zeros(1, _txt_seq, 3, device=device, dtype=encoder_hidden_states.dtype)
    if is_flux:
        latent_image_ids = prepare_flux_latent_image_ids(
            batch_size=1,
            height=latent_h // 2,
            width=latent_w // 2,
            device=device,
            dtype=encoder_hidden_states.dtype,
        )
    else:
        _img_tokens = latent_h * latent_w
        latent_image_ids = torch.zeros(1, _img_tokens, 3, device=device, dtype=encoder_hidden_states.dtype)

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

    all_latents = torch.cat([b["latent"] for b in branches], dim=0).to(device)  # (K,C,H,W)
    reward_prompt = f"{prompt} [OCR_TARGET: {ocr_text}]" if ocr_text else prompt
    prompts_list = [reward_prompt] * K

    log_images_every = int(train_cfg.get("wandb_log_images_every", 0) or 0)
    collect_wandb_images = (
        accelerator.is_main_process
        and log_images_every > 0
        and ((step + 1) % log_images_every == 0)
    )
    _need_images = collect_wandb_images or (clip_div_scorer is not None)
    reward_result = decode_and_compute_rewards(
        latents=all_latents,
        vae=vae,
        prompts=prompts_list,
        reward_models=reward_models,
        reward_weights=reward_weights,
        mix_strategy=config["reward"].get("mix_strategy", "advantage_aggr"),
        batch_size=train_cfg.get("vae_decode_batch_size", 4),
        return_images=_need_images,
    )
    if _need_images:
        rewards, rewards_dict, reward_images = reward_result
    else:
        rewards, rewards_dict = reward_result
        reward_images = None
    rewards = rewards.to(device)
    rewards = torch.nan_to_num(rewards.float(), nan=0.0, posinf=20.0, neginf=-20.0)
    rewards = torch.clamp(rewards, min=-20.0, max=20.0)

    safe_rewards_dict = {}
    for _name, _scores in rewards_dict.items():
        _safe = torch.nan_to_num(_scores.float(), nan=0.0, posinf=20.0, neginf=-20.0)
        _safe = torch.clamp(_safe, min=-20.0, max=20.0)
        safe_rewards_dict[_name] = _safe
    rewards_dict = safe_rewards_dict

    per_model_rewards = {}
    per_model_rewards_by_rank = {}
    for name, scores in rewards_dict.items():
        scores_local = scores.to(device=device, dtype=torch.float32)
        local_mean = torch.nan_to_num(scores_local.mean(), nan=0.0, posinf=0.0, neginf=0.0)
        if accelerator.num_processes > 1:
            rank_means = accelerator.gather(local_mean.reshape(1)).detach().cpu().tolist()
            scores_global = gather_rewards(accelerator, scores_local)
        else:
            rank_means = [float(local_mean.item())]
            scores_global = scores_local
        per_model_rewards_by_rank[name] = [float(x) for x in rank_means]
        per_model_rewards[name] = float(
            torch.nan_to_num(scores_global.mean(), nan=0.0, posinf=0.0, neginf=0.0).item()
        )

    train_lgmd = compute_lgmd(all_latents)
    train_cosine_div = 0.0
    if clip_div_scorer is not None and reward_images is not None:
        try:
            train_cosine_div = clip_div_scorer.score(reward_images)
        except Exception as _e:
            main_print(f"[Diversity] WARN: cosine diversity failed: {_e}")
    if reward_images is not None and not collect_wandb_images:
        del reward_images
        reward_images = None

    if accelerator.num_processes > 1:
        rewards_global = gather_rewards(accelerator, rewards)
    else:
        rewards_global = rewards
    rewards_global = torch.nan_to_num(rewards_global.float(), nan=0.0, posinf=20.0, neginf=-20.0)
    rewards_global = torch.clamp(rewards_global, min=-20.0, max=20.0)

    scheduler.update_reward_bounds(rewards_global)
    mean_reward_t = torch.nan_to_num(rewards_global.mean(), nan=0.0, posinf=0.0, neginf=0.0)
    mean_reward = float(mean_reward_t.item())
    reward_norm_std = float(
        torch.nan_to_num(rewards_global.std(unbiased=False), nan=0.0, posinf=0.0, neginf=0.0).item()
    )
    reward_raw_primary_name = next(iter(per_model_rewards), "raw")
    for _preferred_name in ("pickscore", "hpsv2", "imagereward", "clipscore"):
        if _preferred_name in per_model_rewards:
            reward_raw_primary_name = _preferred_name
            break
    reward_raw_primary_mean = float(per_model_rewards.get(reward_raw_primary_name, 0.0))
    reward_raw_avg = 0.0
    for _name, _score in per_model_rewards.items():
        reward_raw_avg += float(reward_weights.get(_name, 0.0)) * float(_score)
    num_rank_means = max((len(v) for v in per_model_rewards_by_rank.values()), default=0)
    reward_raw_avg_by_rank = []
    for _rank_idx in range(num_rank_means):
        _rank_avg = 0.0
        for _name, _rank_scores in per_model_rewards_by_rank.items():
            if _rank_idx < len(_rank_scores):
                _rank_avg += float(reward_weights.get(_name, 0.0)) * float(_rank_scores[_rank_idx])
        reward_raw_avg_by_rank.append(float(_rank_avg))

    wandb_sample_images = []
    if reward_images is not None:
        max_log_images = int(train_cfg.get("wandb_log_images_max", 8) or 8)
        for _idx, _img in enumerate(reward_images[:max_log_images]):
            _parts = []
            _raw_avg_i = 0.0
            for _name, _scores in rewards_dict.items():
                if _idx < _scores.numel():
                    _score_i = float(_scores[_idx].detach().cpu().item())
                    _raw_avg_i += float(reward_weights.get(_name, 0.0)) * _score_i
                    _parts.append(f"{_name}: {_score_i:.3f}")
            _parts.insert(0, f"avg: {_raw_avg_i:.3f}")
            wandb_sample_images.append({
                "image": _img,
                "caption": f"step {step + 1} | {prompt[:160]} | " + " | ".join(_parts),
            })

    old_log_probs = torch.tensor(
        [b["log_prob_sum"] for b in branches],
        device=device, dtype=torch.float32,
    )

    ref_log_probs_fallback = old_log_probs.clone().detach()
    use_real_ref = bool(loss_cfg.get("use_real_ref", True))

    # ═══ 7. (entropy removed — Softmax-TB forward KL is inherently mode-covering) ═══

    all_metrics = []
    sigmas = build_sigma_schedule(
        tree_cfg["num_inference_steps"],
        tree_cfg.get("shift", 3.0),
        device,
        is_flux=is_flux,
        image_seq_len=(latent_h // 2) * (latent_w // 2) if is_flux else None,
        use_flux_dynamic_shift=tree_cfg.get("use_flux_dynamic_shift", True),
        flux_base_seq_len=tree_cfg.get("flux_base_seq_len", 256),
        flux_max_seq_len=tree_cfg.get("flux_max_seq_len", 4096),
        flux_base_shift=tree_cfg.get("flux_base_shift", 0.5),
        flux_max_shift=tree_cfg.get("flux_max_shift", 1.15),
    )

    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    raw_updates = loss_cfg.get("is_num_updates", 4)
    is_num_updates = math.ceil(raw_updates / grad_accum) * grad_accum

    for update_iter in range(is_num_updates):
        _recompute_sub_batch = int(train_cfg.get("recompute_sub_batch", 9))
        _recompute_forward_dtype = dtype
        _recompute_use_autocast = True
        if debug_recompute_fp32:
            _recompute_sub_batch = min(_recompute_sub_batch, 2)
            _recompute_forward_dtype = torch.float32
            _recompute_use_autocast = False
            if debug_grad_diag and (step % debug_every == 0):
                main_print(
                    f"[DEBUG-CFG] Step {step} iter {update_iter}: "
                    f"recompute_fp32=True, recompute_sub_batch={_recompute_sub_batch}, "
                    f"forward_dtype={_recompute_forward_dtype}, autocast={_recompute_use_autocast}"
                )

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

        # (entropy removed — path_features no longer needed)

        if use_real_ref and update_iter == 0:
            main_print("[WARN] use_real_ref=true is not supported under FSDP sharding. "
                       "Set use_real_ref: false in config. Using fallback.")
            ref_log_probs = ref_log_probs_fallback
        elif not use_real_ref:
            ref_log_probs = ref_log_probs_fallback


        loss, metrics = loss_fn(
            current_log_probs=current_log_probs,
            old_log_probs=old_log_probs,
            rewards=rewards,
            ref_log_probs=ref_log_probs,
            path_features=None,
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

        is_last_accum = (update_iter + 1) % grad_accum == 0
        sync_ctx = contextlib.nullcontext() if is_last_accum else transformer.no_sync()
        scaled_loss = (loss * backprop_scale) / grad_accum

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

        if is_last_accum:
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

            grad_norm_for_log = float(global_gnorm_max)
            nonfinite_local = 1.0 if (math.isnan(grad_norm_for_log) or math.isinf(grad_norm_for_log)) else 0.0
            nonfinite_global = accelerator.reduce(
                torch.tensor(nonfinite_local, device=device, dtype=torch.float32), reduction="max"
            ).item()
            should_skip_step = bool(nonfinite_global > 0.0)

            if should_skip_step:
                main_print(
                    f"[WARNING] Step {step} iter {update_iter}: "
                    f"grad_norm(local={grad_norm}, global={grad_norm_for_log}) non-finite, "
                    "zeroing abnormal gradients and skipping optimizer.step()"
                )
                for _, _p in _optim_named_params:
                    if _p.grad is not None:
                        _p.grad = torch.nan_to_num(_p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                grad_norm_for_log = 0.0

            metrics["grad_norm"] = grad_norm_for_log
            metrics["optimizer_step_skipped_nonfinite"] = 1 if should_skip_step else 0
            accelerator.wait_for_everyone()
            if not should_skip_step:
                optimizer.step()
                if ema is not None:
                    _ema_params = [p for _, p in _optim_named_params]
                    ema.step(_ema_params, step)
            optimizer.zero_grad()
            torch.cuda.empty_cache()



        metrics["update_iter"] = update_iter
        metrics["mean_reward"] = mean_reward
        metrics["reward_norm_mean"] = mean_reward
        metrics["reward_norm_std"] = reward_norm_std
        metrics["reward_raw_primary_name"] = reward_raw_primary_name
        metrics["reward_raw_primary_mean"] = reward_raw_primary_mean
        metrics["reward_raw_avg"] = reward_raw_avg
        metrics["reward_raw_avg_by_rank"] = reward_raw_avg_by_rank
        metrics["alpha"] = alpha
        metrics["per_model_rewards"] = per_model_rewards
        metrics["per_model_rewards_by_rank"] = per_model_rewards_by_rank
        metrics["reward_max"] = float(torch.nan_to_num(rewards_global.max(), nan=0.0, posinf=20.0, neginf=-20.0).item())
        metrics["reward_min"] = float(torch.nan_to_num(rewards_global.min(), nan=0.0, posinf=20.0, neginf=-20.0).item())
        metrics["num_branches"] = K
        metrics["diversity_lgmd"] = train_lgmd
        metrics["diversity_cosine"] = train_cosine_div
        if wandb_sample_images:
            metrics["wandb_sample_images"] = wandb_sample_images
        all_metrics.append(metrics)

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

    if lr_scheduler is not None:
        lr_scheduler.step()

    return all_metrics[-1], mean_reward



# ═══════════════════════════════════════════════════
# ═══════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="TMPO Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="YAML config file path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint dir")

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
    parser.add_argument("--wandb_project", type=str, default="tmpo", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name (auto-generated if None)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    # Eval
    parser.add_argument("--eval_every", type=int, default=None, help="Evaluate every N steps")
    parser.add_argument("--eval_num_prompts", type=int, default=None, help="Number of prompts for evaluation")
    parser.add_argument("--no_eval", action="store_true", help="Disable inline evaluation")
    parser.add_argument("--wandb_log_images_every", type=int, default=10, help="Log sample images every N steps")
    parser.add_argument("--wandb_log_images_max", type=int, default=8, help="Max sample images to log each time")
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

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

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

    if args.force_fixed_schedule:
        tree_cfg["force_fixed_schedule"] = True
    if args.fixed_split_steps is not None:
        tree_cfg["fixed_split_steps"] = _parse_int_list(args.fixed_split_steps)
    if args.fixed_noise_levels is not None:
        tree_cfg["fixed_noise_levels"] = _parse_float_list(args.fixed_noise_levels)

    tree_cfg.setdefault("force_fixed_schedule", False)
    tree_cfg.setdefault("fixed_split_steps", [7, 14, 21])
    tree_cfg.setdefault("fixed_noise_levels", [1.0, 1.0, 1.0])
    tree_cfg.setdefault("num_roots", 1)
    tree_cfg.setdefault("branch_levels", None)
    tree_cfg.setdefault("max_leaves", None)

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

    eval_cfg = config.setdefault("eval", {})
    if args.no_eval:
        eval_cfg["enabled"] = False
    if args.eval_every is not None:
        eval_cfg["eval_every"] = args.eval_every
    if args.eval_num_prompts is not None:
        eval_cfg["num_prompts"] = args.eval_num_prompts
    eval_cfg.setdefault("enabled", False)
    eval_cfg.setdefault("eval_every", 50)
    eval_cfg.setdefault("num_prompts", 10)
    eval_cfg.setdefault("num_images_per_prompt", 10)
    eval_cfg.setdefault("output_file", "eval_results.jsonl")
    eval_cfg.setdefault("reward_mix_strategy", "raw_aggr")

    train_cfg["recompute_sub_batch"] = max(1, int(train_cfg.get("recompute_sub_batch", 9)))
    train_cfg["wandb_log_images_every"] = int(args.wandb_log_images_every)
    train_cfg["wandb_log_images_max"] = int(args.wandb_log_images_max)

    model_cfg = config["model"]
    if args.model_path is not None:
        model_cfg["pretrained_path"] = args.model_path

    # ═══ Accelerator ═══
    accelerator = Accelerator(
        mixed_precision=train_cfg.get("mixed_precision", "bf16"),
    )

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    base_output_dir = str(train_cfg.get("output_dir", "outputs"))
    run_output_dir = os.path.join(base_output_dir, f"seed_{seed}")
    train_cfg["output_dir"] = run_output_dir
    os.makedirs(run_output_dir, exist_ok=True)

    logger = setup_logging(
        log_dir=run_output_dir,
        rank=accelerator.process_index,
    )
    main_print("=" * 60)
    main_print("TMPO Training")
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
        f"Schedule: mode={tree_cfg.get('schedule_mode', 'adaptive')} "
        f"force_fixed={tree_cfg['force_fixed_schedule']} "
        f"noise_level={tree_cfg.get('noise_level', 'N/A')} "
        f"early={tree_cfg.get('early_splits', 'N/A')} late={tree_cfg.get('late_splits', 'N/A')}"
    )
    main_print(
        f"Training: grad_accum={train_cfg.get('gradient_accumulation_steps', 1)} "
        f"vae_decode_batch={train_cfg.get('vae_decode_batch_size', 4)} "
        f"recompute_sub_batch={train_cfg['recompute_sub_batch']}"
    )
    if "flux" in str(model_cfg.get("pretrained_path", "")).lower():
        main_print(f"Flux guidance_scale={float(model_cfg.get('guidance_scale', 3.5))}")

    print_config_table(config, accelerator)

    use_wandb = (wandb is not None) and (not args.no_wandb) and accelerator.is_main_process
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"{train_cfg.get('experiment_name', 'tmpo')}_{seed}",
            config=config,
            tags=["tmpo", model_cfg.get("pretrained_path", "unknown").rstrip("/").split("/")[-1]],
        )
        main_print(f"[Wandb] Project: {args.wandb_project}, Run: {wandb.run.name}")
    else:
        main_print("[Wandb] Disabled")

    if train_cfg.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if train_cfg.get("cudnn_safe_mode", True):
        torch.backends.cudnn.benchmark = False
        warnings.filterwarnings(
            "ignore",
            message=r"Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR",
            category=UserWarning,
        )

    model_cfg = config["model"]
    main_print(f"Loading model: {model_cfg['pretrained_path']}")

    model_path = model_cfg["pretrained_path"].lower()
    is_flux = "flux" in model_path

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

    for _enc_attr in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        _enc = getattr(pipeline, _enc_attr, None)
        if _enc is not None:
            _enc.requires_grad_(False)
            _enc.to("cpu")

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
        transformer = transformer.to(torch.bfloat16)
        main_print("[INFO] Cast transformer(+LoRA) to bf16 for FSDP dtype uniformity")

        bad_params = []
        for pname, p in transformer.named_parameters():
            if p.requires_grad and (torch.isnan(p).any() or torch.isinf(p).any()):
                bad_params.append(pname)
        if bad_params:
            main_print(f"[WARNING] NaN/Inf detected in {len(bad_params)} LoRA params after init, re-init to zero:")
            for pname in bad_params:
                main_print(f"  {pname}")
                param = dict(transformer.named_parameters())[pname]
                torch.nn.init.zeros_(param.data)
            main_print("[INFO] LoRA weight sanitization complete.")
        else:
            main_print("[INFO] LoRA weight check passed: no NaN/Inf detected.")
    else:
        transformer.requires_grad_(True)

    _set_param_debug_names(transformer)

    # (fsdp_small.yaml: fsdp_activation_checkpointing=true)
    #   FSDP checkpoint_wrapper → HF gradient_checkpoint → forward
    # if train_cfg.get("gradient_checkpointing", True):
    #     transformer.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=float(train_cfg.get("learning_rate", 1e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.get("max_train_steps", 300),
        eta_min=float(train_cfg.get("learning_rate", 1e-5)) * 0.1,
    )

    dataset = PromptDataset(
        config["dataset"]["data_json_path"],
        auto_extract_ocr=bool(config["dataset"].get("auto_extract_ocr", False)),
    )
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

    _ema_enabled = bool(train_cfg.get("ema_enabled", True))
    _ema_decay = float(train_cfg.get("ema_decay", 0.9))
    _ema_interval = int(train_cfg.get("ema_update_interval", 8))
    _trainable_params = [p for _, p in _iter_unique_optimizer_named_params(optimizer)]
    if _ema_enabled:
        ema = EMAModuleWrapper(
            _trainable_params,
            decay=_ema_decay,
            update_step_interval=_ema_interval,
            device=accelerator.device,
        )
        main_print(f"[EMA] Enabled: decay={_ema_decay}, interval={_ema_interval}")
    else:
        ema = None
        main_print("[EMA] Disabled")

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

    reward_models, reward_weights = build_reward_models(
        config["reward"], accelerator.device,
    )
    main_print(f"Reward models: {list(reward_models.keys())}")

    clip_div_scorer = None
    div_cfg = config.get("diversity", {})
    div_backend = div_cfg.get("backend", "dinov2")  # "dinov2" | "clip"
    if div_cfg.get("clip_cosine", True):
        try:
            _local = config["reward"].get("pickscore_local_files_only", False)
            if div_backend == "dinov2":
                _dino_path = div_cfg.get("dinov2_model", "facebook/dinov2-large")
                clip_div_scorer = DINOv2DiversityScorer(
                    device=accelerator.device,
                    model_name=_dino_path,
                    local_files_only=_local,
                )
                main_print(f"[Diversity] DINOv2 cosine diversity scorer loaded ({_dino_path})")
            else:
                _clip_path = config["reward"].get(
                    "clip_score_model_path", "openai/clip-vit-large-patch14"
                )
                clip_div_scorer = CLIPDiversityScorer(
                    device=accelerator.device,
                    model_name=_clip_path,
                    local_files_only=_local,
                )
                main_print(f"[Diversity] CLIP cosine diversity scorer loaded ({_clip_path})")
        except Exception as e:
            main_print(f"[Diversity] WARN: Failed to load diversity scorer: {e}")

    evaluator = InlineEvaluator(
        eval_config=eval_cfg,
        reward_config=config["reward"],
        device=accelerator.device,
        clip_div_scorer=clip_div_scorer,
    )
    main_print(
        f"[Eval] enabled={eval_cfg.get('enabled')}, every={eval_cfg.get('eval_every')}, "
        f"num_prompts={eval_cfg.get('num_prompts')}, "
        f"num_images_per_prompt={eval_cfg.get('num_images_per_prompt')}, "
        f"reward_mix_strategy={eval_cfg.get('reward_mix_strategy')}"
    )

    tree_sampler = TreeSampler(
        num_inference_steps=tree_cfg.get("num_inference_steps", 28),
        k=tree_cfg.get("k", 3),
        num_roots=tree_cfg.get("num_roots", 1),
        branch_levels=tree_cfg.get("branch_levels", None),
        max_leaves=tree_cfg.get("max_leaves", None),
        shift=tree_cfg.get("shift", 3.0),
        use_flux_dynamic_shift=tree_cfg.get("use_flux_dynamic_shift", True),
        flux_base_seq_len=tree_cfg.get("flux_base_seq_len", 256),
        flux_max_seq_len=tree_cfg.get("flux_max_seq_len", 4096),
        flux_base_shift=tree_cfg.get("flux_base_shift", 0.5),
        flux_max_shift=tree_cfg.get("flux_max_shift", 1.15),
        sde_type=tree_cfg.get("sde_type", "cps"),
    )

    _schedule_mode = tree_cfg.get("schedule_mode", "adaptive")
    if _schedule_mode == "progress":
        scheduler = ProgressAwareSplitScheduler(
            num_inference_steps=tree_cfg.get("num_inference_steps", 28),
            early_splits=tree_cfg.get("early_splits", [4, 7, 12]),
            late_splits=tree_cfg.get("late_splits", [6, 12, 20]),
            noise_level=float(tree_cfg.get("noise_level", 0.8)),
            noise_levels=tree_cfg.get("noise_levels", None),
            tail_guard_steps=tree_cfg.get("tail_guard_steps", 4),
            min_gap=tree_cfg.get("min_gap", 3),
            total_train_steps=train_cfg.get("max_train_steps", 500),
            beta_kappa=float(tree_cfg.get("beta_kappa", 0.0)),
        )
        main_print(
            f"[Scheduler] ProgressAware: early={scheduler.early_splits}, "
            f"late={scheduler.late_splits}, "
            f"η_per_split={scheduler._noise_levels}, "
            f"κ={scheduler.beta_kappa}, "
            f"total_steps={scheduler.total_train_steps}"
        )
    else:
        scheduler = AdaptiveScheduler(
            num_inference_steps=tree_cfg.get("num_inference_steps", 28),
            num_splits=3,
            kappa=tree_cfg.get("kappa", 4.0),
            base_noise_levels=tree_cfg.get("base_noise_levels", [0.4, 0.7, 1.0]),
            tail_guard_steps=tree_cfg.get("tail_guard_steps", 4),
        )
        main_print(f"[Scheduler] Adaptive: kappa={scheduler.kappa}")

    loss_fn = TMPOLoss(
        beta=loss_cfg.get("beta", 15.0),
        lambda_entropy=loss_cfg.get("lambda_entropy", 0.01),
        lambda_ref=loss_cfg.get("lambda_ref", 0.1),
        is_clip_range=loss_cfg.get("is_clip_range", 0.2),
        rbf_bandwidth=loss_cfg.get("rbf_bandwidth", 1.0),
        ref_scale=loss_cfg.get("ref_scale", 1.0),
    )

    max_steps = train_cfg.get("max_train_steps", 300)
    checkpoint_steps = int(train_cfg.get("checkpointing_steps", 50))
    save_state_enabled = bool(train_cfg.get("save_state", False))
    save_state_steps = int(train_cfg.get("save_state_steps", max(checkpoint_steps * 4, 200)))
    output_dir = train_cfg.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    if args.resume is not None:
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

    reward_log_path = os.path.join(output_dir, "reward_log.json")
    reward_detail_log_path = os.path.join(output_dir, "reward_detail_log.json")
    reward_log = {}
    reward_detail_log = {}
    if os.path.exists(reward_log_path):
        try:
            with open(reward_log_path, "r") as f:
                reward_log = json.load(f)
            main_print(f"[Reward Log] Loaded {len(reward_log)} historical entries")
        except (json.JSONDecodeError, IOError):
            reward_log = {}
    if os.path.exists(reward_detail_log_path):
        try:
            with open(reward_detail_log_path, "r") as f:
                reward_detail_log = json.load(f)
        except (json.JSONDecodeError, IOError):
            reward_detail_log = {}

    prev_mean_reward = None
    data_iter = iter(dataloader)

    if evaluator.should_eval(0):
        accelerator.wait_for_everyone()
        if ema is not None:
            _ema_params_init = [p for _, p in _iter_unique_optimizer_named_params(optimizer)]
            ema.copy_ema_to(_ema_params_init, store_temp=True)
        eval_metrics_init = evaluator.evaluate(
            step=0,
            transformer=transformer,
            vae=vae,
            pipeline=pipeline,
            tree_sampler=tree_sampler,
            scheduler=scheduler,
            prompts=[dataset[i]["prompt"] for i in range(min(eval_cfg.get("num_prompts", 100), len(dataset)))],
            config=config,
            accelerator=accelerator,
            is_flux=is_flux,
        )
        if ema is not None:
            ema.copy_temp_to(_ema_params_init)
        transformer.train()
        if use_wandb and eval_metrics_init:
            wandb.log(eval_metrics_init, step=0)
        accelerator.wait_for_everyone()

    while global_step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        prompt = batch["prompt"][0] if isinstance(batch["prompt"], list) else batch["prompt"]
        _ocr = batch.get("ocr_text", None)
        ocr_text = (_ocr[0] if isinstance(_ocr, (list, tuple)) else _ocr) or None

        _beta_init   = float(loss_cfg.get("beta", 15.0))
        _beta_target = float(loss_cfg.get("beta_target", _beta_init))
        _beta_warmup = int(loss_cfg.get("beta_warmup_steps", 0))
        if _beta_warmup > 0 and global_step < _beta_warmup:
            loss_fn.soft_tb.beta = _beta_init + (_beta_target - _beta_init) * global_step / _beta_warmup
        else:
            loss_fn.soft_tb.beta = _beta_target

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
            ema=ema,
            ocr_text=ocr_text,
            clip_div_scorer=clip_div_scorer,
        )

        global_step += 1

        if global_step % 1 == 0:
            main_print(
                f"[Step {global_step}/{max_steps}] "
                f"loss={metrics['loss_total']:.6e} "
                f"tb={metrics['loss_soft_tb']:.6e} "
                f"entropy={metrics['loss_entropy']:.6e} "
                f"ref={metrics['loss_ref']:.6e} "
                f"w_ent={metrics.get('loss_entropy_weighted', 0.0):.6e} "
                f"w_ref={metrics.get('loss_ref_weighted', 0.0):.6e} "
                f"reward_norm={metrics.get('reward_norm_mean', metrics['mean_reward']):.6e} "
                f"reward_norm_std={metrics.get('reward_norm_std', 0.0):.6e} "
                f"{metrics.get('reward_raw_primary_name', 'raw')}_raw={metrics.get('reward_raw_primary_mean', 0.0):.6e} "
                f"α={metrics.get('alpha', 0):.2f} "
                f"lgmd={metrics.get('diversity_lgmd', 0):.4f} "
                f"cos_div={metrics.get('diversity_cosine', 0):.4f}"
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

        if accelerator.is_main_process:
            reward_log[str(global_step)] = metrics.get("reward_raw_avg", metrics["mean_reward"])
            reward_detail_log[str(global_step)] = {
                "reward_avg": metrics.get("reward_raw_avg", metrics["mean_reward"]),
                "reward_ori_avg": metrics.get("reward_raw_avg", metrics["mean_reward"]),
                "reward_compute": metrics.get("reward_raw_avg", metrics["mean_reward"]),
                "per_model": metrics.get("per_model_rewards", {}),
                "per_model_by_rank": metrics.get("per_model_rewards_by_rank", {}),
                "avg_by_rank": metrics.get("reward_raw_avg_by_rank", []),
            }
            try:
                with open(reward_log_path, "w", encoding="utf-8") as f:
                    json.dump(reward_log, f, indent=2, ensure_ascii=False)
                with open(reward_detail_log_path, "w", encoding="utf-8") as f:
                    json.dump(reward_detail_log, f, indent=2, ensure_ascii=False)
            except IOError as e:
                main_print(f"[WARNING] Failed to write reward_log.json: {e}")

        if use_wandb:
            wandb_log = {
                "train/loss_total": metrics["loss_total"],
                "train/loss_soft_tb": metrics["loss_soft_tb"],
                "train/loss_entropy": metrics["loss_entropy"],
                "train/loss_ref": metrics["loss_ref"],
                "train/grad_norm": metrics.get("grad_norm", 0),
                "debug/grad_trainable": metrics.get("debug_grad_trainable", 0),
                "debug/grad_has_grad": metrics.get("debug_grad_has_grad", 0),
                "debug/grad_nan_params": metrics.get("debug_grad_nan_params", 0),
                "train/lr": optimizer.param_groups[0]["lr"],
                "reward_avg": metrics.get("reward_raw_avg", 0.0),
                "reward_ori_avg": metrics.get("reward_raw_avg", 0.0),
                "reward_compute": metrics.get("reward_raw_avg", 0.0),
                "is/approx_kl": metrics.get("approx_kl", 0),
                "is/clipfrac": metrics.get("clipfrac", 0),
                "is/ratio_mean": metrics.get("ratio_mean", 1),
                "is/ratio_std": metrics.get("ratio_std", 0),
                "is/weight_mean": metrics.get("is_weight_mean", 1),
                "is/weight_std": metrics.get("is_weight_std", 0),
                "tree/alpha": metrics.get("alpha", 0),
                "tree/log_prob_mean": metrics.get("log_prob_mean", 0),
                "tree/log_prob_old_mean": metrics.get("log_prob_old_mean", 0),
                "tree/sqrt_dt_sq_mean": metrics.get("sqrt_dt_sq_mean", 1),
                "tree/num_branches": metrics.get("num_branches", 8),
                "train/diversity_lgmd": metrics.get("diversity_lgmd", 0),
                "train/diversity_cosine": metrics.get("diversity_cosine", 0),
            }
            for model_name, score in metrics.get("per_model_rewards", {}).items():
                wandb_log[f"reward_{model_name}"] = score
            # Raw reward by rank: these keys let W&B draw one chart with 8 rank lines.
            for rank_idx, score in enumerate(metrics.get("reward_raw_avg_by_rank", [])):
                wandb_log[f"reward_rank/avg_rank{rank_idx}"] = score
                wandb_log[f"reward_rank/compute_rank{rank_idx}"] = score
            for model_name, rank_scores in metrics.get("per_model_rewards_by_rank", {}).items():
                for rank_idx, score in enumerate(rank_scores):
                    wandb_log[f"reward_rank/{model_name}_rank{rank_idx}"] = score
            sample_images = metrics.get("wandb_sample_images", [])
            if sample_images:
                wandb_log["train_images"] = [
                    wandb.Image(item["image"], caption=item["caption"])
                    for item in sample_images
                ]
            wandb.log(wandb_log, step=global_step)

        if global_step % checkpoint_steps == 0:
            accelerator.wait_for_everyone()
            if ema is not None:
                _ema_params = [p for _, p in _iter_unique_optimizer_named_params(optimizer)]
                ema.copy_ema_to(_ema_params, store_temp=True)
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
            if ema is not None:
                ema.copy_temp_to(_ema_params)
            accelerator.wait_for_everyone()

        if evaluator.should_eval(global_step):
            accelerator.wait_for_everyone()
            if ema is not None:
                _ema_params = [p for _, p in _iter_unique_optimizer_named_params(optimizer)]
                ema.copy_ema_to(_ema_params, store_temp=True)
            eval_metrics = evaluator.evaluate(
                step=global_step,
                transformer=transformer,
                vae=vae,
                pipeline=pipeline,
                tree_sampler=tree_sampler,
                scheduler=scheduler,
                prompts=[dataset[i]["prompt"] for i in range(min(eval_cfg.get("num_prompts", 100), len(dataset)))],
                config=config,
                accelerator=accelerator,
                is_flux=is_flux,
            )
            if ema is not None:
                ema.copy_temp_to(_ema_params)
            transformer.train()
            if use_wandb and eval_metrics:
                wandb.log(eval_metrics, step=global_step)
            accelerator.wait_for_everyone()

        if save_state_enabled and (global_step % max(1, save_state_steps) == 0):
            state_dir = os.path.join(output_dir, f"state-{global_step}")
            accelerator.wait_for_everyone()
            accelerator.save_state(output_dir=state_dir)
            accelerator.wait_for_everyone()
            main_print(f"[State] Saved distributed state to {state_dir}")

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
