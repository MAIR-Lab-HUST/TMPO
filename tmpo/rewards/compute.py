"""Multi-reward model computation and aggregation."""

import torch
from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np


def build_reward_models(config: dict, device: torch.device) -> Dict:
    """Build reward models from config."""
    reward_models = {}
    reward_weights = {}
    model_names = config.get("models", ["hpsv2"])
    weights = config.get("weights", [1.0])

    for name, weight in zip(model_names, weights):
        if name == "hpsv2":
            from .hpsv2 import HPSv2RewardModel
            model = HPSv2RewardModel(
                ckpt_path=config.get("hps_path"),
                clip_path=config.get("hps_clip_path"),
                device=device,
            )
        elif name == "clipscore":
            from .clipscore import CLIPScoreRewardModel
            clip_model_path = config.get(
                "clip_score_model_path",
                config.get("clip_score_path", "openai/clip-vit-large-patch14"),
            )
            model = CLIPScoreRewardModel(
                model_path=clip_model_path,
                device=device,
            )
        elif name == "aesthetic":
            from .aesthetic import AestheticRewardModel
            model = AestheticRewardModel(
                device=device,
                clip_model_name=config.get("aesthetic_clip_model_name", "openai/clip-vit-large-patch14"),
                predictor_ckpt=config.get("aesthetic_predictor_ckpt", None),
            )
        elif name == "pickscore":
            from .pickscore import PickScoreRewardModel
            model = PickScoreRewardModel(
                device=torch.device("cpu"),
                processor_path=config.get("pickscore_processor_path", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"),
                model_path=config.get("pickscore_model_path", "yuvalkirstain/PickScore_v1"),
                mean=config.get("pickscore_mean", 18.0),
                std=config.get("pickscore_std", 8.0),
                score_mode=config.get("pickscore_score_mode", "flow_scaled"),
                strict=config.get("pickscore_strict", True),
                local_files_only=config.get("pickscore_local_files_only", True),
                batch_size=config.get("pickscore_batch_size", 8),
            )
        elif name == "imagereward":
            from .imagereward import ImageRewardRewardModel
            model = ImageRewardRewardModel(
                model_name=config.get("imagereward_model_name", "ImageReward-v1.0"),
                med_config=config.get("imagereward_med_config", None),
                device=device,
            )
        elif name in ("geneval", "geneval_http"):
            from .geneval_http import GenEvalHTTPRewardModel
            model = GenEvalHTTPRewardModel(
                url=config.get("geneval_url", ""),
                batch_size=config.get("geneval_batch_size", 64),
                timeout=config.get("geneval_timeout", 120),
                only_strict=config.get("geneval_only_strict", True),
                retries=config.get("geneval_retries", 6),
            )
        elif name == "paddleocr":
            from .paddleocr import PaddleOCRRewardModel
            model = PaddleOCRRewardModel(
                lang=config.get("paddleocr_lang", "ch"),
                use_angle_cls=config.get("paddleocr_use_angle_cls", True),
                score_mode=config.get("paddleocr_score_mode", "ned"),
                det_db_thresh=config.get("paddleocr_det_db_thresh", 0.3),
            )
        else:
            raise ValueError(f"Unknown reward model: {name}")

        reward_models[name] = model
        reward_weights[name] = weight

    return reward_models, reward_weights


def compute_reward(
    images: List,
    prompts: List[str],
    reward_models: Dict,
    reward_weights: Dict,
    mix_strategy: str = "advantage_aggr",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute and aggregate rewards from multiple models.

    Args:
        images: K PIL.Images or Tensors.
        prompts: K prompt strings.
        reward_models: {name: model} dict.
        reward_weights: {name: weight} dict.
        mix_strategy: "advantage_aggr", "reward_aggr", or "raw_aggr".
    Returns:
        final_rewards: (K,) aggregated rewards.
        rewards_dict: {name: (K,)} per-model raw rewards.
    """
    K = len(images)
    rewards_dict = {}

    for name, model in reward_models.items():
        try:
            scores = model(images, prompts)
            r = torch.tensor(scores, dtype=torch.float32)
            nonfinite_count = (~torch.isfinite(r)).sum().item()
            if nonfinite_count > 0:
                print(
                    f"[WARNING] Reward model {name} returned {int(nonfinite_count)} non-finite values; sanitizing to finite range."
                )
            r = torch.nan_to_num(r, nan=0.0, posinf=20.0, neginf=-20.0)
            r = torch.clamp(r, min=-20.0, max=20.0)
            rewards_dict[name] = r
        except Exception as e:
            raise RuntimeError(f"Reward model {name} failed during scoring: {e}") from e


    reward_std_eps = 1e-4

    if mix_strategy == "advantage_aggr":
        total = torch.zeros(K)
        for name, weight in reward_weights.items():
            r = rewards_dict.get(name, torch.zeros(K))
            r_mean = torch.nan_to_num(r.mean(), nan=0.0, posinf=0.0, neginf=0.0)
            r_std = torch.nan_to_num(r.std(unbiased=False), nan=0.0, posinf=0.0, neginf=0.0)
            r_norm = (r - r_mean) / r_std.clamp_min(reward_std_eps)
            r_norm = torch.nan_to_num(r_norm, nan=0.0, posinf=20.0, neginf=-20.0)
            total += weight * r_norm
        final_rewards = total

    elif mix_strategy == "reward_aggr":
        total = torch.zeros(K)
        for name, weight in reward_weights.items():
            r = rewards_dict.get(name, torch.zeros(K))
            total += weight * r
        total_mean = torch.nan_to_num(total.mean(), nan=0.0, posinf=0.0, neginf=0.0)
        total_std = torch.nan_to_num(total.std(unbiased=False), nan=0.0, posinf=0.0, neginf=0.0)
        final_rewards = (total - total_mean) / total_std.clamp_min(reward_std_eps)

    elif mix_strategy == "raw_aggr":
        total = torch.zeros(K)
        for name, weight in reward_weights.items():
            r = rewards_dict.get(name, torch.zeros(K))
            total += weight * r
        final_rewards = total

    else:
        raise ValueError(f"Unknown mix_strategy: {mix_strategy}")

    final_rewards = torch.nan_to_num(final_rewards, nan=0.0, posinf=20.0, neginf=-20.0)
    final_rewards = torch.clamp(final_rewards, min=-20.0, max=20.0)
    return final_rewards, rewards_dict


def decode_and_compute_rewards(
    latents: torch.Tensor,
    vae,
    prompts: List[str],
    reward_models: Dict,
    reward_weights: Dict,
    mix_strategy: str = "advantage_aggr",
    batch_size: int = 4,
    return_images: bool = False,
):
    """Batch VAE decode + reward computation.

    Args:
        latents: (K, C, H, W) terminal latents.
        vae: VAE decoder.
        prompts: K prompt strings.
        return_images: if True, also return decoded PIL images.
    Returns:
        (rewards, rewards_dict) or (rewards, rewards_dict, all_images).
    """
    K = latents.shape[0]
    all_images = []

    for i in range(0, K, batch_size):
        batch = latents[i : i + batch_size]
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
            decoded = vae.decode(batch / vae.config.scaling_factor + shift).sample

        decoded = ((decoded + 1.0) / 2.0).clamp(0, 1)
        for img_tensor in decoded:
            img_tensor = torch.nan_to_num(img_tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
            img_np = (img_tensor.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
            all_images.append(Image.fromarray(img_np))

        del decoded
        torch.cuda.empty_cache()

    rewards, rewards_dict = compute_reward(
        images=all_images,
        prompts=prompts,
        reward_models=reward_models,
        reward_weights=reward_weights,
        mix_strategy=mix_strategy,
    )
    if return_images:
        return rewards, rewards_dict, all_images
    return rewards, rewards_dict
