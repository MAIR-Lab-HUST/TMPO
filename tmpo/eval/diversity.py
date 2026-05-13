"""Diversity metrics: L-GMD and GARDO-style Cosine Diversity."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
from PIL import Image


def compute_lgmd(
    features: torch.Tensor,
    eps: float = 1e-8,
    normalize_by_dim: bool = True,
) -> float:
    """Compute L-GMD (Log Geometric Mean Distance) diversity score.

    Args:
        features: (N, D) tensor, N samples' feature vectors
                  For latent space, usually (27, C*H*W) flattened latent
        eps: Minimum distance threshold to prevent log(0)
        normalize_by_dim: Whether to normalize by dimension (divide by √D), making scores comparable across different dimensions

    Returns:
        lgmd: L-GMD score (scalar). Higher values indicate more diversity, extremely small/negative values indicate mode collapse
    """
    N = features.shape[0]
    if N < 2:
        return 0.0

    features_flat = features.reshape(N, -1).float()
    D = features_flat.shape[1]

    dists = torch.cdist(features_flat, features_flat, p=2)  # (N, N)

    if normalize_by_dim and D > 0:
        dists = dists / (D ** 0.5)

    mask = torch.triu(torch.ones(N, N, device=features.device, dtype=torch.bool), diagonal=1)
    pairwise_dists = dists[mask]

    pairwise_dists = torch.clamp(pairwise_dists, min=eps)

    # L-GMD = (2 / N(N-1)) * Σ log(d_ij)
    num_pairs = N * (N - 1) // 2
    lgmd = (pairwise_dists.log().sum() / num_pairs).item()

    return lgmd


def compute_quality_score(rewards: torch.Tensor) -> float:
    """Compute quality score as mean reward."""
    return rewards.float().mean().item()


def compute_cosine_diversity(features: torch.Tensor) -> float:
    """GARDO-style cosine diversity: mean pairwise cosine distance

    Div = mean_{i,j∈[1,G], i≠j} (1 - cos_sim(e_i, e_j))

    Args:
        features: (N, D) L2-normalized CLIP image embeddings

    Returns:
        diversity: scalar, higher means more diverse
    """
    N = features.shape[0]
    if N < 2:
        return 0.0

    features = features.float()
    features = F.normalize(features, p=2, dim=-1)
    # (N, N) cosine similarity matrix
    sim_matrix = features @ features.T
    # exclude diagonal (self-similarity = 1)
    mask = ~torch.eye(N, device=features.device, dtype=torch.bool)
    pairwise_cos = sim_matrix[mask]
    diversity = (1.0 - pairwise_cos).mean().item()
    return diversity


class CLIPDiversityScorer:
    """CLIP-based diversity scorer (GARDO-style pairwise cosine distance)."""

    def __init__(self, device: torch.device, model_name: str = "openai/clip-vit-large-patch14",
                 local_files_only: bool = False):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(
            model_name, local_files_only=local_files_only
        ).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def extract_features(self, images: List[Image.Image], batch_size: int = 16) -> torch.Tensor:
        """Extract L2-normalized CLIP image features from PIL images.

        Args:
            images: list of PIL.Image
            batch_size: encode batch size

        Returns:
            features: (N, D) L2-normalized CLIP image embeddings
        """
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            inputs = self.processor(images=batch_imgs, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)  # (B, D)
            feats = F.normalize(feats, p=2, dim=-1)
            all_feats.append(feats.cpu())
        return torch.cat(all_feats, dim=0)  # (N, D)

    def score(self, images: List[Image.Image], batch_size: int = 16) -> float:
        """Compute cosine diversity for a group of images.

        Args:
            images: list of PIL.Image (same prompt)
            batch_size: CLIP encode batch size

        Returns:
            diversity: scalar
        """
        if len(images) < 2:
            return 0.0
        features = self.extract_features(images, batch_size=batch_size)
        return compute_cosine_diversity(features)


class DINOv2DiversityScorer:
    """DINOv2-based diversity scorer (GARDO-style, captures fine-grained visual differences)."""

    def __init__(self, device: torch.device, model_name: str = "facebook/dinov2-large",
                 local_files_only: bool = False):
        from transformers import AutoModel, AutoImageProcessor
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name, local_files_only=local_files_only
        ).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def extract_features(self, images: List[Image.Image], batch_size: int = 16) -> torch.Tensor:
        """Extract DINOv2 CLS token features from PIL images.

        Args:
            images: list of PIL.Image
            batch_size: encode batch size

        Returns:
            features: (N, D) DINOv2 image embeddings (L2-normalized)
        """
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            inputs = self.processor(images=batch_imgs, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            feats = outputs.last_hidden_state[:, 0]  # CLS token (B, D)
            feats = F.normalize(feats, p=2, dim=-1)
            all_feats.append(feats.cpu())
        return torch.cat(all_feats, dim=0)  # (N, D)

    def score(self, images: List[Image.Image], batch_size: int = 16) -> float:
        """Compute cosine diversity for a group of images.

        Args:
            images: list of PIL.Image (same prompt)
            batch_size: DINOv2 encode batch size

        Returns:
            diversity: scalar
        """
        if len(images) < 2:
            return 0.0
        features = self.extract_features(images, batch_size=batch_size)
        return compute_cosine_diversity(features)
