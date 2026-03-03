"""HPSv2 奖励模型封装

Human Preference Score v2.1 — 评估图像与人类审美偏好的一致性。
"""

import torch
import torch.nn as nn
from typing import List, Optional
from PIL import Image


class HPSv2RewardModel:
    """HPSv2 奖励模型"""

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        clip_path: Optional[str] = None,
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
        self.model = None
        self.preprocess = None

        if ckpt_path is not None:
            self._load_model(ckpt_path, clip_path)

    def _load_model(self, ckpt_path: str, clip_path: Optional[str] = None):
        """加载 HPS v2.1 模型"""
        try:
            import open_clip

            # 加载 CLIP 模型
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14",
                pretrained=clip_path or "laion2b_s32b_b79k",
                device=self.device,
            )

            # 加载 HPS 检查点
            state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            self.model = model
            self.preprocess = preprocess
            self.tokenizer = open_clip.get_tokenizer("ViT-H-14")

        except ImportError:
            print("[WARNING] open_clip not installed, using dummy HPSv2")
            self.model = None

    @torch.no_grad()
    def __call__(self, images: List, prompts: List[str]) -> List[float]:
        """计算 HPS 评分

        Args:
            images: K 张 PIL.Image
            prompts: K 个 prompt

        Returns:
            scores: K 个 HPS 评分
        """
        if self.model is None:
            # Dummy fallback
            return [0.25 + 0.05 * torch.randn(1).item() for _ in images]

        scores = []
        for img, prompt in zip(images, prompts):
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            text_tokens = self.tokenizer([prompt]).to(self.device)

            img_features = self.model.encode_image(img_tensor)
            text_features = self.model.encode_text(text_tokens)

            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            score = (img_features * text_features).sum().item()
            scores.append(score)

        return scores
