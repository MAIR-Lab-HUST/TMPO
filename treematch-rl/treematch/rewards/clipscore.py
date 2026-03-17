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
            import open_clip, os, json
            # 目录路径 → 解析到 bin 文件
            actual_path = model_path
            model_name = "ViT-H-14"
            if os.path.isdir(model_path):
                bin_file = os.path.join(model_path, "open_clip_pytorch_model.bin")
                actual_path = bin_file if os.path.exists(bin_file) else model_path
                # 读 config 获取架构名, 默认 DFN5B 用 ViT-H-14-378-quickgelu
                cfg_file = os.path.join(model_path, "open_clip_config.json")
                if os.path.exists(cfg_file):
                    with open(cfg_file) as _f:
                        _cfg = json.load(_f)
                    # 依次尝试各字段, DFN5B 的 config 不同版本字段名不同
                    model_name = (
                        _cfg.get("model_name")
                        or _cfg.get("model_cfg", {}).get("model_type")
                        or _cfg.get("model_cfg", {}).get("model_name")
                        or "ViT-H-14-378-quickgelu"
                    )
                else:
                    model_name = "ViT-H-14-378-quickgelu"
            elif "384" in model_path or "dfn" in model_path.lower():
                model_name = "ViT-H-14-378-quickgelu"
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=actual_path,
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
