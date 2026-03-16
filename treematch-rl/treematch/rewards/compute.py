"""多奖励模型并行计算与融合

支持:
- HPSv2: 人类偏好评分
- CLIP Score: 文图一致性
- Aesthetic Score: 美学评分

融合策略:
- advantage_aggr: 各模型独立归一化后加权合并 (推荐)
- reward_aggr: 先加权合并奖励再归一化
- raw_aggr: 直接加权求和, 不做任何归一化
"""

import torch
import concurrent.futures
from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np


def build_reward_models(config: dict, device: torch.device) -> Dict:
    """根据配置构建奖励模型

    Args:
        config: reward 配置字典
        device: 计算设备

    Returns:
        reward_models: {name: model} 字典
        reward_weights: {name: weight} 字典
    """
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
            model = CLIPScoreRewardModel(
                model_path=config.get("clip_score_path"),
                device=device,
            )
        elif name == "aesthetic":
            from .aesthetic import AestheticRewardModel
            model = AestheticRewardModel(device=device)
        else:
            raise ValueError(f"Unknown reward model: {name}")

        reward_models[name] = model
        reward_weights[name] = weight

    return reward_models, reward_weights


def _compute_single(model, images: List, prompts: List[str]) -> List[float]:
    """单个奖励模型计算（用于线程池并行）"""
    return model(images, prompts)


def compute_reward(
    images: List,
    prompts: List[str],
    reward_models: Dict,
    reward_weights: Dict,
    mix_strategy: str = "advantage_aggr",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """多奖励模型并行计算与融合

    Args:
        images: K 张 PIL.Image 或 Tensor
        prompts: K 个 prompt 文本
        reward_models: {name: model} 字典
        reward_weights: {name: weight} 字典
        mix_strategy: 融合策略
            "advantage_aggr" → 各模型独立归一化后加权
            "reward_aggr" → 先加权合并再归一化
            "raw_aggr" → 直接加权求和(不中心化/不标准化)

    Returns:
        final_rewards: (K,) 融合后的奖励
        rewards_dict: {name: (K,)} 各模型的原始奖励
    """
    K = len(images)
    rewards_dict = {}

    # 线程池并行计算各奖励模型
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(reward_models)) as executor:
        futures = {
            executor.submit(_compute_single, model, images, prompts): name
            for name, model in reward_models.items()
        }

        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                scores = future.result()
                rewards_dict[name] = torch.tensor(scores, dtype=torch.float32)
            except Exception as e:
                print(f"[WARNING] Reward model {name} failed: {e}")
                rewards_dict[name] = torch.zeros(K)

    # ═══ 融合策略 ═══
    if mix_strategy == "advantage_aggr":
        # 各模型独立 per-prompt 归一化后加权合并
        total = torch.zeros(K)
        for name, weight in reward_weights.items():
            r = rewards_dict.get(name, torch.zeros(K))
            # 独立归一化
            r_norm = (r - r.mean()) / (r.std() + 1e-8)
            total += weight * r_norm
        final_rewards = total

    elif mix_strategy == "reward_aggr":
        # 先加权合并再归一化
        total = torch.zeros(K)
        for name, weight in reward_weights.items():
            r = rewards_dict.get(name, torch.zeros(K))
            total += weight * r
        final_rewards = (total - total.mean()) / (total.std() + 1e-8)

    elif mix_strategy == "raw_aggr":
        # 直接加权聚合, 保留奖励原始尺度
        total = torch.zeros(K)
        for name, weight in reward_weights.items():
            r = rewards_dict.get(name, torch.zeros(K))
            total += weight * r
        final_rewards = total

    else:
        raise ValueError(f"Unknown mix_strategy: {mix_strategy}")

    return final_rewards, rewards_dict


def decode_and_compute_rewards(
    latents: torch.Tensor,
    vae,
    prompts: List[str],
    reward_models: Dict,
    reward_weights: Dict,
    mix_strategy: str = "advantage_aggr",
    batch_size: int = 4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """分批 VAE 解码 + 奖励计算 (防 OOM)

    Args:
        latents: (K, C, H, W) 各路径的最终 latent
        vae: VAE 解码器
        prompts: K 个 prompt
        其他: 同 compute_reward

    Returns:
        rewards: (K,) 融合奖励
        rewards_dict: {name: (K,)} 原始奖励
    """
    K = latents.shape[0]
    all_images = []

    # 分批 VAE 解码
    for i in range(0, K, batch_size):
        batch = latents[i : i + batch_size]
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            # Flux VAE decode 需要加 shift_factor; SD3/其他为 0
            shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
            decoded = vae.decode(batch / vae.config.scaling_factor + shift).sample

        # 转为 PIL Image
        decoded = ((decoded + 1.0) / 2.0).clamp(0, 1)
        for img_tensor in decoded:
            # NaN/Inf 诎为 0, 防止 uint8 转换溢出
            img_tensor = torch.nan_to_num(img_tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
            img_np = (img_tensor.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
            all_images.append(Image.fromarray(img_np))

        del decoded
        torch.cuda.empty_cache()

    # 向所有奖励模型传入所有图像
    return compute_reward(
        images=all_images,
        prompts=prompts,
        reward_models=reward_models,
        reward_weights=reward_weights,
        mix_strategy=mix_strategy,
    )
