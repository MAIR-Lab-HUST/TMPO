"""PickScore reward model wrapper with strict loading."""

from pathlib import Path
from typing import List, Optional
import importlib

import torch

_original_torch_empty = torch.empty

def _patched_torch_empty(*args, **kwargs):
    if not args:
        return _original_torch_empty((), **kwargs)
    return _original_torch_empty(*args, **kwargs)

torch.empty = _patched_torch_empty
# ─────────────────────────────────────────────────────────────────────────────


class PickScoreRewardModel:
    """PickScore: text-image preference score from CLIP-style embeddings."""

    _loaded_logged = False
    _MODE_DANCE = "dance_raw_cosine"
    _MODE_MIX = "mix_zscore"
    _MODE_FLOW = "flow_scaled"

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        processor_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_path: str = "yuvalkirstain/PickScore_v1",
        mean: Optional[float] = 18.0,
        std: Optional[float] = 8.0,
        score_mode: str = _MODE_FLOW,
        strict: bool = True,
        local_files_only: bool = True,
        batch_size: int = 8,
    ):
        self.device = device
        self.mean = mean
        self.std = std
        self.score_mode = score_mode
        self.strict = strict
        self.batch_size = max(1, int(batch_size))

        self.processor = None
        self.model = None

        try:
            transformers = importlib.import_module("transformers")
            CLIPModel = getattr(transformers, "CLIPModel")
            AutoProcessor = getattr(transformers, "AutoProcessor")

            processor_source = self._resolve_source(
                configured_path=processor_path,
                default_repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            )
            model_source = self._resolve_source(
                configured_path=model_path,
                default_repo_id="yuvalkirstain/PickScore_v1",
            )

            self.processor = AutoProcessor.from_pretrained(
                processor_source,
                local_files_only=local_files_only,
            )
            self.model = CLIPModel.from_pretrained(
                model_source,
                local_files_only=local_files_only,
            ).eval().to(device)

            if not PickScoreRewardModel._loaded_logged:
                print(
                    "[PickScore] Model loaded: "
                    f"{processor_source} + {model_source} "
                    f"(AutoProcessor/CLIPModel, mode={self.score_mode}, batch={self.batch_size})"
                )
                PickScoreRewardModel._loaded_logged = True
        except Exception as e:
            message = f"PickScore load failed: {e}"
            if strict:
                raise RuntimeError(message) from e
            print(f"[WARNING] {message}")

    @staticmethod
    def _resolve_source(configured_path: str, default_repo_id: str) -> str:
        if configured_path:
            if configured_path.startswith("hf://"):
                return configured_path[len("hf://") :]
            if Path(configured_path).exists():
                return configured_path
            if "/" in configured_path and not configured_path.startswith("/"):
                return configured_path
        return default_repo_id

    @torch.no_grad()
    def __call__(self, images: List, prompts: List[str]) -> List[float]:
        if self.model is None or self.processor is None:
            raise RuntimeError("PickScore model is not loaded")

        if len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length")

        all_scores: List[float] = []
        for start in range(0, len(images), self.batch_size):
            end = min(start + self.batch_size, len(images))
            image_chunk = images[start:end]
            prompt_chunk = prompts[start:end]

            image_inputs = self.processor(
                images=image_chunk,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            image_inputs = {
                key: value.to(device=self.device) for key, value in image_inputs.items()
            }

            text_inputs = self.processor(
                text=prompt_chunk,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            text_inputs = {
                key: value.to(device=self.device) for key, value in text_inputs.items()
            }

            with torch.autocast("cuda", enabled=False):
                image_embs = self.model.get_image_features(**image_inputs).float()
                text_embs = self.model.get_text_features(**text_inputs).float()

            image_norm = image_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
            text_norm = text_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
            image_embs = image_embs / image_norm
            text_embs = text_embs / text_norm

            cosine_scores = (text_embs @ image_embs.T).diag()

            if self.score_mode == self._MODE_DANCE:
                scores = cosine_scores
            elif self.score_mode == self._MODE_MIX:
                if self.mean is None or self.std is None:
                    raise ValueError("pickscore mean/std must be provided for mix_zscore mode")
                scores = (self.model.logit_scale.exp() * cosine_scores - self.mean) / self.std
            elif self.score_mode == self._MODE_FLOW:
                scores = self.model.logit_scale.float().clamp(-10, 10).exp() * cosine_scores / 26.0
            else:
                raise ValueError(f"Unknown pickscore score_mode: {self.score_mode}")

            safe_scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=20.0, neginf=-20.0)
            safe_scores = torch.clamp(safe_scores, min=-20.0, max=20.0)
            all_scores.extend(safe_scores.cpu().tolist())

        return all_scores
