import contextlib
import os

import torch
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

class HPS_v2:
    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model, _, self.preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
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
            with_region_predictor=False
        )
        cp = "./hps_ckpt/HPS_v2.1_compressed.pt"
        if not os.path.exists(cp):
            raise FileNotFoundError(
                "HPSv2 checkpoint not found. Download `HPS_v2.1_compressed.pt` and place it at "
                f"`{cp}`."
            )
        checkpoint = torch.load(cp, map_location="cpu")
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        self.tokenizer = get_tokenizer('ViT-H-14')

    def __call__(self, img, text):

        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        autocast_ctx = (
            torch.amp.autocast(device_type="cuda") if device_type == "cuda" else contextlib.nullcontext()
        )
        with torch.no_grad():
            # Process the image
            image = self.preprocess_val(img).unsqueeze(0).to(device=self.device, non_blocking=True)
            # Process the prompt
            text = self.tokenizer([text]).to(device=self.device, non_blocking=True)
            # Calculate the HPS
            with autocast_ctx:
                outputs = self.model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image)
        return hps_score