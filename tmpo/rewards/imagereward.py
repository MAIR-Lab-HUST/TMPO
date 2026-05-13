"""ImageReward model wrapper."""

from typing import List, Optional

import torch


class ImageRewardRewardModel:
    """ImageReward: general purpose text-image reward model."""

    def __init__(
        self,
        model_name: str = "ImageReward-v1.0",
        med_config: Optional[str] = None,
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
        self.model = None

        try:
            import ImageReward as RM
        except Exception as e:
            raise RuntimeError(f"ImageReward import failed: {e}") from e

        load_errors = []
        load_attempts = []
        if med_config:
            load_attempts.append(("with_med_config", {"device": str(device), "med_config": med_config}))
        load_attempts.append(("without_med_config", {"device": str(device)}))

        for tag, kwargs in load_attempts:
            try:
                self.model = RM.load(model_name, **kwargs)
                break
            except Exception as e:
                load_errors.append(f"{tag}: {type(e).__name__}: {e}")

        if self.model is None:
            raise RuntimeError(
                "ImageReward load failed.\n"
                f"  model_name={model_name}\n"
                f"  med_config={med_config}\n"
                "  attempts:\n    - "
                + "\n    - ".join(load_errors)
            )

        if hasattr(self.model, "eval"):
            self.model = self.model.eval()
        if hasattr(self.model, "requires_grad_"):
            self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images: List, prompts: List[str]) -> List[float]:
        if self.model is None:
            raise RuntimeError("ImageReward model is not loaded")

        if len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length")

        rewards = []
        for image, prompt in zip(images, prompts):
            # ImageReward uses single prompt + candidate images for rank and score.
            _, reward = self.model.inference_rank(prompt, [image])
            if isinstance(reward, (list, tuple)):
                reward = reward[0]
            rewards.append(float(reward))

        return rewards
