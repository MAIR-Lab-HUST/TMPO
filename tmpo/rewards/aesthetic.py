"""Aesthetic score reward model (standard CLIP-L/14 + MLP predictor)."""

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn


class _AestheticMLP(nn.Module):
    """MLP head from improved-aesthetic-predictor (CLIP-L/14 embedding -> score)."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AestheticRewardModel:
    """Standard aesthetic reward model used in GRPO-style pipelines."""

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        clip_model_name: str = "openai/clip-vit-large-patch14",
        predictor_ckpt: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype = dtype
        self.clip = None
        self.processor = None
        self.mlp = None

        try:
            from transformers import CLIPModel, CLIPProcessor

            self.clip = CLIPModel.from_pretrained(clip_model_name).eval().to(device)
            self.processor = CLIPProcessor.from_pretrained(clip_model_name)

            self.mlp = _AestheticMLP().to(device=device, dtype=dtype).eval()
            ckpt = self._resolve_predictor_ckpt(predictor_ckpt)
            if ckpt is None:
                raise FileNotFoundError(
                    "Aesthetic predictor checkpoint not found. "
                    "Set reward.aesthetic_predictor_ckpt in your config."
                )
            state_dict = torch.load(ckpt, map_location=device, weights_only=True)
            self.mlp.load_state_dict(state_dict)
        except Exception as e:
            print(f"[WARNING] Failed to load standard Aesthetic model: {e}")
            self.clip = None
            self.processor = None
            self.mlp = None

    def _resolve_predictor_ckpt(self, predictor_ckpt: Optional[str]) -> Optional[str]:
        if predictor_ckpt and Path(predictor_ckpt).exists():
            return predictor_ckpt

        # Workspace-friendly fallback: reuse FlowGRPO asset if available.
        candidate = (
            Path(__file__).resolve().parents[3]
            / "flow_grpo-main"
            / "flow_grpo"
            / "assets"
            / "sac+logos+ava1-l14-linearMSE.pth"
        )
        if candidate.exists():
            return str(candidate)
        return None

    @torch.no_grad()
    def __call__(self, images: List, prompts: List[str]) -> List[float]:
        del prompts  # Aesthetic score is image-only.
        if self.clip is None or self.processor is None or self.mlp is None:
            return [0.0 for _ in images]

        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)

        image_embeds = self.clip.get_image_features(pixel_values=pixel_values)
        image_embeds = image_embeds / torch.linalg.vector_norm(
            image_embeds, dim=-1, keepdim=True
        )
        scores = self.mlp(image_embeds).squeeze(1)
        return scores.float().cpu().tolist()
