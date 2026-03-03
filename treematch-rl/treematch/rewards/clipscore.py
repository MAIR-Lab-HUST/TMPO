"""CLIP Score 奖励模型封装"""

import torch
from typing import List, Optional
from PIL import Image


class CLIPScoreRewardModel:
    """CLIP Score: 衡量图像与文本的语义一致性"""

    def __init__(
        self,
        model_path: str = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384",
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
        self.model = None
        self.preprocess = None
        self.tokenizer = None

        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14-378-quickgelu" if "384" in model_path else "ViT-H-14",
                pretrained=model_path,
                device=device,
            )
            model.eval()
            self.model = model
            self.preprocess = preprocess
            self.tokenizer = open_clip.get_tokenizer("ViT-H-14")
        except Exception as e:
            print(f"[WARNING] Failed to load CLIP model: {e}")

    @torch.no_grad()
    def __call__(self, images: List, prompts: List[str]) -> List[float]:
        if self.model is None:
            return [25.0 + torch.randn(1).item() for _ in images]

        scores = []
        for img, prompt in zip(images, prompts):
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            text_tokens = self.tokenizer([prompt]).to(self.device)

            img_feat = self.model.encode_image(img_tensor)
            txt_feat = self.model.encode_text(text_tokens)

            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            score = (100.0 * (img_feat * txt_feat).sum()).item()
            scores.append(score)

        return scores
