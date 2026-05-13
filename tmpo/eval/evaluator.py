"""Inline evaluation module for periodic assessment during training."""

import os
import json
import math
import re
import torch
import torch.distributed as dist
import numpy as np
from typing import Dict, List, Optional

from tmpo.rewards.compute import build_reward_models, decode_and_compute_rewards
from tmpo.eval.diversity import compute_lgmd, compute_quality_score, CLIPDiversityScorer
from tmpo.sampling import prepare_flux_latent_image_ids
from tmpo.utils.logging_ import main_print


_DQUOTE_RE      = re.compile(r'"([^"]+)"')
_OCR_FIELDS     = ("ocr_text", "text", "target_text")


def _load_test_data(json_path: str, auto_extract_ocr: bool = False):
    """Load test data from JSON/JSONL/txt file, returning (prompts, ocr_texts)."""
    prompts   = []
    ocr_texts = []
    with open(json_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    raw_items = []
    if content.startswith("[") or content.startswith("{"):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                raw_items = data
            elif isinstance(data, dict):
                raw_items = list(data.values())
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
            prompt = item.get("prompt", str(item))
            ocr = next((item[f] for f in _OCR_FIELDS if f in item), None)
            if ocr is None and auto_extract_ocr:
                m = _DQUOTE_RE.search(prompt)
                ocr = m.group(1) if m else None
        else:
            prompt = str(item)
            ocr = None
            if auto_extract_ocr:
                m = _DQUOTE_RE.search(prompt)
                ocr = m.group(1) if m else None
        prompts.append(prompt)
        ocr_texts.append(ocr)

    return prompts, ocr_texts


def _sanitize_prompt_for_path(prompt: str, max_len: int = 80) -> str:
    """Sanitize prompt into a short directory-safe string."""
    prompt = re.sub(r"\s+", "_", str(prompt).strip())
    prompt = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "", prompt)
    prompt = prompt.strip("._-")
    if not prompt:
        prompt = "prompt"
    return prompt[:max_len]


class InlineEvaluator:
    """Periodic inline evaluator during training."""

    def __init__(
        self,
        eval_config: dict,
        reward_config: dict,
        device: torch.device,
        clip_div_scorer: CLIPDiversityScorer = None,
    ):
        """
        Args:
            eval_config: evaluation config dict.
            reward_config: reward config dict for building eval reward models.
            device: compute device.
        """
        self.enabled = eval_config.get("enabled", False)
        self.eval_every = eval_config.get("eval_every", 50)
        self.num_prompts = eval_config.get("num_prompts", 10)
        self.num_images_per_prompt = eval_config.get("num_images_per_prompt", 10)
        self.output_file = eval_config.get("output_file", "eval_results.jsonl")
        self.save_images = bool(eval_config.get("save_images", True))
        self.image_dir_name = str(eval_config.get("image_dir_name", "eval_images"))
        self.eval_num_inference_steps = int(eval_config.get("num_inference_steps", 28))
        self.sample_batch_size = eval_config.get("eval_sample_batch_size", 5)
        # VAE decode batch
        self.decode_batch_size = eval_config.get("eval_decode_batch_size", 10)
        self.device = device

        self.test_prompts   = None
        self.test_ocr_texts = None
        auto_extract_ocr = bool(eval_config.get("auto_extract_ocr", False))
        test_data_path = eval_config.get("test_data_path")
        if test_data_path and os.path.exists(test_data_path):
            self.test_prompts, self.test_ocr_texts = _load_test_data(
                test_data_path, auto_extract_ocr=auto_extract_ocr
            )
            main_print(f"[Eval] Loaded test set: {test_data_path} ({len(self.test_prompts)} samples)")
        elif test_data_path:
            main_print(f"[Eval] WARNING: test data path not found: {test_data_path}, using first N training prompts")

        self.reward_names = eval_config.get(
            "eval_reward_models", ["imagereward", "pickscore", "hpsv2"]
        )
        self.mix_strategy = eval_config.get("reward_mix_strategy", "raw_aggr")
        self.eval_at_start = bool(eval_config.get("eval_at_start", False))
        self.clip_div_scorer = clip_div_scorer

        if self.enabled:
            eval_reward_config = dict(reward_config)
            eval_reward_config["models"] = list(self.reward_names)
            orig_weights = list(reward_config.get("weights", []))
            if len(orig_weights) == len(self.reward_names):
                eval_reward_config["weights"] = orig_weights
            else:
                eval_reward_config["weights"] = [1.0] * len(self.reward_names)
            eval_reward_config["geneval_only_strict"] = False
            self.reward_models, self.reward_weights = build_reward_models(eval_reward_config, device)
            main_print(
                "[Eval] Reward eval models loaded: "
                f"models={self.reward_names}, eval_mix={self.mix_strategy}"
            )
        else:
            self.reward_models = {}
            self.reward_weights = {}

    def should_eval(self, step: int) -> bool:
        """Check if evaluation should run at current step."""
        if not self.enabled:
            return False
        if step == 0:
            return self.eval_at_start
        return step % self.eval_every == 0

    def _get_eval_prompts(self, fallback_prompts: List[str]):
        """Get evaluation prompts, preferring independent test set."""
        if self.test_prompts is not None:
            ps = self.test_prompts[: self.num_prompts]
            os_ = (self.test_ocr_texts or [None] * len(self.test_prompts))[: self.num_prompts]
            return ps, os_
        return fallback_prompts[: self.num_prompts], [None] * min(self.num_prompts, len(fallback_prompts))

    @torch.no_grad()
    def _encode_prompt_for_eval(
        self,
        pipeline,
        prompt: str,
        device: torch.device,
        is_flux: bool,
    ):
        """Encode prompt using the training-consistent text encoding pipeline."""
        enc_attrs = ["text_encoder", "text_encoder_2", "text_encoder_3"]
        for attr in enc_attrs:
            enc = getattr(pipeline, attr, None)
            if enc is not None:
                enc.to(device)

        try:
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
                print(f"[Eval RANK {device}] WARN: non-finite encoder_hidden_states, sanitized", flush=True)

            if pooled_prompt_embeds is not None and (not torch.isfinite(pooled_prompt_embeds).all()):
                pooled_prompt_embeds = torch.nan_to_num(
                    pooled_prompt_embeds, nan=0.0, posinf=1e3, neginf=-1e3
                )
                pooled_prompt_embeds = torch.clamp(pooled_prompt_embeds, min=-1e3, max=1e3)
                print(f"[Eval RANK {device}] WARN: non-finite pooled_prompt_embeds, sanitized", flush=True)

            return encoder_hidden_states, pooled_prompt_embeds
        finally:
            for attr in enc_attrs:
                enc = getattr(pipeline, attr, None)
                if enc is not None:
                    enc.to("cpu")
            torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate(
        self,
        step: int,
        transformer,
        vae,
        pipeline,
        tree_sampler,
        scheduler,
        prompts: List[str],
        config: dict,
        accelerator,
        is_flux: bool = False,
    ) -> Dict:
        """Run one evaluation round (FSDP-safe: all ranks iterate all prompts in sync)."""
        if not self.enabled:
            return {}

        eval_prompts, eval_ocr_texts = self._get_eval_prompts(prompts)
        device = accelerator.device
        dtype = torch.bfloat16
        model_cfg = config.get("model", {})
        train_cfg = config["training"]
        guidance_scale = float(model_cfg.get("guidance_scale", 3.5 if is_flux else 0.0))
        base_seed = int(train_cfg.get("seed", 42))
        eval_num_inference_steps = int(
            self.eval_num_inference_steps or config["tree"].get("num_inference_steps", 28)
        )

        transformer.eval()
        if hasattr(vae, "eval"):
            vae.eval()

        H, W = config["dataset"]["resolution"]
        latent_h, latent_w = H // 8, W // 8
        latent_channels = 16
        latent_shape = (1, latent_channels, latent_h, latent_w)

        num_samples = max(1, int(self.num_images_per_prompt))

        output_dir = train_cfg.get("output_dir", "outputs")
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        _base, _ext = os.path.splitext(self.output_file)
        eval_output_path = os.path.join(output_dir, f"{_base}_rank{rank}{_ext}")
        eval_image_root = os.path.join(output_dir, self.image_dir_name)

        orig_num_prompts = len(eval_prompts)

        local_reward_means = []
        local_lgmd_scores = []
        local_cosine_div_scores = []
        local_per_model_means = {name: [] for name in self.reward_names}
        local_eval_results = []

        main_print(
            f"[Eval Step {step}] Evaluating {orig_num_prompts} prompts, "
            f"world_size={world_size}, {num_samples} images per prompt..."
        )

        orig_num_roots = getattr(tree_sampler, "num_roots", 1)
        orig_num_inference_steps = getattr(tree_sampler, "num_inference_steps", eval_num_inference_steps)
        tree_sampler.num_roots = 1
        tree_sampler.num_inference_steps = eval_num_inference_steps

        # Flux: T5=[1,512,4096], CLIP=[1,768]; SD3: T5=[1,154,4096], CLIP=[1,2048]
        if is_flux:
            _enc_hs_shape = (1, 512, 4096)
            _pooled_shape = (1, 768)
        else:
            _enc_hs_shape = (1, 154, 4096)
            _pooled_shape = (1, 2048)

        remainder = orig_num_prompts % world_size
        if remainder != 0:
            eval_prompts = eval_prompts + [eval_prompts[-1]] * (world_size - remainder)
        num_slots = len(eval_prompts) // world_size   # = ceil(orig/world_size)

        try:
            for slot in range(num_slots):
                slot_base = slot * world_size
                prompt_idx = slot_base + rank
                prompt     = eval_prompts[prompt_idx]
                ocr_target = eval_ocr_texts[prompt_idx] if prompt_idx < len(eval_ocr_texts) else None
                is_padding = (prompt_idx >= orig_num_prompts)

                if world_size > 1 and dist.is_initialized():
                    if rank == 0:
                        enc_hs_list, pooled_list = [], []
                        for r in range(world_size):
                            _p = eval_prompts[slot_base + r]
                            _hs, _pe = self._encode_prompt_for_eval(
                                pipeline=pipeline, prompt=_p,
                                device=device, is_flux=is_flux,
                            )
                            enc_hs_list.append(_hs.to(dtype=dtype, device=device))
                            pooled_list.append(_pe.to(dtype=dtype, device=device) if _pe is not None else None)
                        all_enc_hs = torch.cat(enc_hs_list, dim=0)
                        all_pooled = torch.cat(pooled_list, dim=0) if pooled_list[0] is not None else None
                    else:
                        all_enc_hs = torch.empty(
                            world_size, *_enc_hs_shape[1:], device=device, dtype=dtype
                        )
                        all_pooled = (
                            torch.empty(world_size, *_pooled_shape[1:], device=device, dtype=dtype)
                            if _pooled_shape is not None else None
                        )
                    dist.broadcast(all_enc_hs, src=0)
                    if all_pooled is not None:
                        dist.broadcast(all_pooled, src=0)
                    encoder_hidden_states = all_enc_hs[rank : rank + 1]
                    pooled_prompt_embeds = all_pooled[rank : rank + 1] if all_pooled is not None else None
                else:
                    encoder_hidden_states, pooled_prompt_embeds = self._encode_prompt_for_eval(
                        pipeline=pipeline, prompt=prompt, device=device, is_flux=is_flux,
                    )

                sample_latents = []
                txt_seq = encoder_hidden_states.shape[1]
                text_ids = torch.zeros(1, txt_seq, 3, device=device, dtype=encoder_hidden_states.dtype)
                if is_flux:
                    latent_image_ids = prepare_flux_latent_image_ids(
                        batch_size=1,
                        height=latent_h // 2,
                        width=latent_w // 2,
                        device=device,
                        dtype=encoder_hidden_states.dtype,
                    )
                else:
                    latent_image_ids = torch.zeros(
                        1,
                        latent_h * latent_w,
                        3,
                        device=device,
                        dtype=encoder_hidden_states.dtype,
                    )

                for sample_offset in range(num_samples):
                    eval_seed = (
                        base_seed
                        + step * 1000003
                        + prompt_idx * 10007
                        + rank * 997
                        + sample_offset
                    ) % (2**31)
                    generator = torch.Generator(device=device).manual_seed(int(eval_seed))
                    sample_result = tree_sampler.sample(
                        transformer=transformer,
                        latent_shape=latent_shape,
                        encoder_hidden_states=encoder_hidden_states,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        text_ids=text_ids,
                        latent_image_ids=latent_image_ids,
                        split_steps=[],
                        noise_levels=[],
                        device=device,
                        dtype=dtype,
                        generator=generator,
                        guidance_scale=guidance_scale,
                        is_flux=is_flux,
                    )
                    sample_latents.append(sample_result[0]["latent"].detach())
                    torch.cuda.empty_cache()

                latents_stack = torch.cat(sample_latents, dim=0)
                K = latents_stack.shape[0]
                reward_prompt = f"{prompt} [OCR_TARGET: {ocr_target}]" if ocr_target else prompt
                prompts_list = [reward_prompt] * K
                fused_rewards, rewards_dict, all_images = decode_and_compute_rewards(
                    latents=latents_stack,
                    vae=vae,
                    prompts=prompts_list,
                    reward_models=self.reward_models,
                    reward_weights=self.reward_weights,
                    mix_strategy=self.mix_strategy,
                    batch_size=self.decode_batch_size,
                    return_images=True,
                )

                if not is_padding:
                    reward_scores = fused_rewards.detach().cpu().tolist()
                    per_model_means = {
                        name: float(scores.mean().item()) for name, scores in rewards_dict.items()
                    }
                    for name, v in per_model_means.items():
                        local_per_model_means.setdefault(name, []).append(float(v))

                    lgmd_score = compute_lgmd(latents_stack)
                    mean_reward = compute_quality_score(fused_rewards.detach().cpu())

                    # GARDO-style cosine diversity
                    cosine_div_score = 0.0
                    if self.clip_div_scorer is not None and all_images:
                        try:
                            cosine_div_score = self.clip_div_scorer.score(all_images)
                        except Exception:
                            pass

                    local_reward_means.append(mean_reward)
                    local_lgmd_scores.append(lgmd_score)
                    local_cosine_div_scores.append(cosine_div_score)

                    result = {
                        "step": step,
                        "prompt_idx": prompt_idx,
                        "prompt": prompt,
                        "rank": rank,
                        "num_samples": K,
                        "eval_mix_strategy": self.mix_strategy,
                        "sample_rewards": reward_scores,
                        "mean_reward": mean_reward,
                        "per_model_means": per_model_means,
                        "lgmd_diversity": lgmd_score,
                        "cosine_diversity": cosine_div_score,
                    }
                    local_eval_results.append(result)

                    print(
                        f"  [Eval rank {rank} | prompt {prompt_idx + 1}/{orig_num_prompts}] "
                        f"reward={mean_reward:.6f} L-GMD={lgmd_score:.4f} cos_div={cosine_div_score:.4f} K={K}",
                        flush=True,
                    )

                    if self.save_images:
                        prompt_slug = _sanitize_prompt_for_path(prompt)
                        prompt_dir = os.path.join(
                            eval_image_root,
                            f"step_{step:06d}",
                            f"prompt_{prompt_idx:04d}_{prompt_slug}",
                        )
                        os.makedirs(prompt_dir, exist_ok=True)
                        with open(os.path.join(prompt_dir, "prompt.txt"), "w", encoding="utf-8") as f:
                            f.write(prompt)
                        with open(os.path.join(prompt_dir, "metrics.json"), "w", encoding="utf-8") as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                        for image_idx, image in enumerate(all_images):
                            image.save(os.path.join(prompt_dir, f"image_{image_idx:02d}.png"))

                del all_images
                del latents_stack
                torch.cuda.empty_cache()
        finally:
            tree_sampler.num_roots = orig_num_roots
            tree_sampler.num_inference_steps = orig_num_inference_steps
            transformer.train()

        if not local_reward_means:
            return {}

        eval_results = list(local_eval_results)
        eval_results.sort(key=lambda x: x.get("prompt_idx", 0))
        all_reward_means = list(local_reward_means)
        all_lgmd_scores = list(local_lgmd_scores)
        all_cosine_div_scores = list(local_cosine_div_scores)
        aggregated_per_model_means = {
            name: list(values) for name, values in local_per_model_means.items()
        }

        os.makedirs(os.path.dirname(eval_output_path) or ".", exist_ok=True)
        with open(eval_output_path, "a", encoding="utf-8") as f:
            for result in eval_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        main_print(f"[Eval Step {step}] Results saved to {eval_output_path}")
        if self.save_images:
            main_print(
                f"[Eval Step {step}] Images saved to "
                f"{os.path.join(eval_image_root, f'step_{step:06d}')}"
            )

        eval_metrics = {
            "eval_reward_avg": float(np.mean(all_reward_means)),
            "eval_reward_compute": float(np.mean(all_reward_means)),
            "eval_lgmd": float(np.mean(all_lgmd_scores)),
            "eval_cosine_div": float(np.mean(all_cosine_div_scores)) if all_cosine_div_scores else 0.0,
            "eval_num_prompts": len(eval_results),
        }
        for name, values in aggregated_per_model_means.items():
            if values:
                eval_metrics[f"eval_reward_{name}"] = float(np.mean(values))

        main_print(
            f"[Eval Step {step}] Summary: "
            f"reward_avg={eval_metrics['eval_reward_avg']:.6f} "
            f"L-GMD={eval_metrics['eval_lgmd']:.4f} "
            f"cos_div={eval_metrics['eval_cosine_div']:.4f}"
        )

        return eval_metrics
