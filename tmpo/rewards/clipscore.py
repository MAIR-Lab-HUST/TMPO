"""CLIP Score reward model (aligned with Flow-GRPO)."""

import torch
from typing import List
from PIL import Image
import numpy as np


class CLIPScoreRewardModel:
    """CLIP Score: image-text semantic alignment."""

    _loaded_logged = False

    def __init__(
        self,
        model_path: str = "openai/clip-vit-large-patch14",
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
        self.model = None
        self.processor = None

        try:
            from transformers import CLIPModel, CLIPProcessor

            self.model = CLIPModel.from_pretrained(model_path).eval().to(device)
            self.processor = CLIPProcessor.from_pretrained(model_path)

            if not CLIPScoreRewardModel._loaded_logged:
                print(f"[CLIPScore] Model loaded: {model_path}")
                CLIPScoreRewardModel._loaded_logged = True

        except Exception as e:
            raise RuntimeError(f"Failed to load CLIPScore model from {model_path}: {e}") from e

    @torch.no_grad()
    def __call__(self, images: List, prompts: List[str]) -> List[float]:
        """Compute CLIP Score: logits_per_image / 30."""
        if self.model is None or self.processor is None:
            raise RuntimeError("CLIPScore model is not loaded")

        if isinstance(images[0], Image.Image):
            np_images = [np.array(img) for img in images]
        else:
            np_images = images

        image_inputs = self.processor(
            images=np_images,
            return_tensors="pt",
        )
        pixel_values = image_inputs["pixel_values"].to(
            device=self.device, dtype=torch.float32
        )

        text_inputs = self.processor(
            text=prompts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {
            k: v.to(self.device) for k, v in text_inputs.items()
            if k != "pixel_values"
        }

        outputs = self.model(pixel_values=pixel_values, **text_inputs)

        scores = outputs.logits_per_image.diagonal() / 30.0

        return scores.float().cpu().tolist()
