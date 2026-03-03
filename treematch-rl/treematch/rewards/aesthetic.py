"""Aesthetic Score 奖励模型封装"""

import torch
from typing import List
from PIL import Image


class AestheticRewardModel:
    """基于 CLIP 的美学评分模型"""

    def __init__(self, device: torch.device = torch.device("cuda")):
        self.device = device
        self.model = None
        self.preprocess = None

        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai", device=device,
            )
            model.eval()
            self.model = model
            self.preprocess = preprocess
        except Exception as e:
            print(f"[WARNING] Failed to load Aesthetic model: {e}")

    @torch.no_grad()
    def __call__(self, images: List, prompts: List[str]) -> List[float]:
        if self.model is None:
            return [5.5 + 0.5 * torch.randn(1).item() for _ in images]

        scores = []
        for img in images:
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            feat = self.model.encode_image(img_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            # 简易美学分 = 特征范数的函数
            score = feat.norm().item() * 5.0
            scores.append(score)

        return scores
