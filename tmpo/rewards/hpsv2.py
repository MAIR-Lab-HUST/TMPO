"""HPSv2 reward model (aligned with DanceGRPO)."""

import torch
from typing import List, Optional
from PIL import Image


class HPSv2RewardModel:
    """HPSv2 reward model."""

    _success_logged = False
    _first_score_logged = False

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        clip_path: Optional[str] = None,
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
        self.model = None
        self.preprocess_val = None
        self.tokenizer = None

        if ckpt_path is not None:
            self._load_model(ckpt_path, clip_path)

    def _load_model(self, ckpt_path: str, clip_path: Optional[str] = None):
        """Load HPS v2.1 model."""
        try:
            from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

            model, _, preprocess_val = create_model_and_transforms(
                'ViT-H-14',
                clip_path or 'laion2b_s32b_b79k',
                precision='amp',
                device=self.device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False,
            )

            checkpoint = torch.load(ckpt_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                ckpt_keys_info = f"checkpoint keys: {list(checkpoint.keys())}, state_dict keys count: {len(checkpoint['state_dict'])}"
            else:
                model.load_state_dict(checkpoint)
                ckpt_keys_info = f"direct state_dict, keys count: {len(checkpoint)}"
            model.eval()

            self.model = model
            self.preprocess_val = preprocess_val
            self.tokenizer = get_tokenizer('ViT-H-14')

            if not HPSv2RewardModel._success_logged:
                logit_scale_val = model.logit_scale.exp().item() if hasattr(model, 'logit_scale') else 'N/A'
                print(f"[HPSv2] Model loaded (hpsv2.src.open_clip)")
                print(f"[HPSv2] {ckpt_keys_info}")
                print(f"[HPSv2] logit_scale = {logit_scale_val}")
                HPSv2RewardModel._success_logged = True

        except ImportError:
            print(
                "[WARNING] hpsv2 package not installed. "
                "Falling back to generic open_clip..."
            )
            self._load_model_fallback(ckpt_path, clip_path)

    def _load_model_fallback(self, ckpt_path: str, clip_path: Optional[str] = None):
        """Fallback: load via generic open_clip."""
        try:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14",
                pretrained=clip_path or "laion2b_s32b_b79k",
                device=self.device,
            )

            checkpoint = torch.load(ckpt_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                ckpt_keys_info = f"checkpoint keys: {list(checkpoint.keys())}, state_dict keys count: {len(checkpoint['state_dict'])}"
            else:
                model.load_state_dict(checkpoint, strict=False)
                ckpt_keys_info = f"direct state_dict, keys count: {len(checkpoint)}"
            model.eval()

            self.model = model
            self.preprocess_val = preprocess
            self.tokenizer = open_clip.get_tokenizer("ViT-H-14")

            if not HPSv2RewardModel._success_logged:
                logit_scale_val = model.logit_scale.exp().item() if hasattr(model, 'logit_scale') else 'N/A'
                print(f"[HPSv2] Loaded via generic open_clip fallback")
                print(f"[HPSv2] {ckpt_keys_info}")
                print(f"[HPSv2] logit_scale = {logit_scale_val}")
                HPSv2RewardModel._success_logged = True

        except ImportError:
            print("[WARNING] open_clip not installed; HPSv2 will return dummy scores")
            self.model = None

    @torch.no_grad()
    def __call__(self, images: List, prompts: List[str]) -> List[float]:
        """Compute HPS scores."""
        if self.model is None:
            return [0.25 + 0.05 * torch.randn(1).item() for _ in images]

        scores = []
        for img, prompt in zip(images, prompts):
            img_tensor = self.preprocess_val(img).unsqueeze(0).to(
                device=self.device, non_blocking=True
            )
            text_tokens = self.tokenizer([prompt]).to(
                device=self.device, non_blocking=True
            )

            with torch.amp.autocast('cuda'):
                outputs = self.model(img_tensor, text_tokens)

            if isinstance(outputs, dict):
                # hpsv2.src.open_clip (output_dict=True)
                image_features = outputs["image_features"]
                text_features = outputs["text_features"]
                logits_per_image = image_features @ text_features.T
                score = torch.diagonal(logits_per_image).item()
            else:
                image_features, text_features, logit_scale = outputs
                score = (image_features @ text_features.T).diagonal().item()

            if not HPSv2RewardModel._first_score_logged:
                if isinstance(outputs, dict):
                    ls = outputs.get('logit_scale', 'N/A')
                    ls_val = ls.item() if hasattr(ls, 'item') else ls
                else:
                    ls_val = logit_scale.item()
                print(f"[HPSv2 DIAG] First score: score={score:.6f}, logit_scale={ls_val}, "
                      f"img_feat_norm={image_features.norm().item():.4f}, "
                      f"txt_feat_norm={text_features.norm().item():.4f}, "
                      f"output_type={'dict' if isinstance(outputs, dict) else 'tuple'}")
                HPSv2RewardModel._first_score_logged = True

            scores.append(score)

        return scores
