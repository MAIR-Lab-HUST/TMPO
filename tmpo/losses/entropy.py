"""RBF kernel particle entropy regularization."""

import torch
import torch.nn as nn


class ParticleEntropyLoss(nn.Module):
    """RBF kernel repulsion loss to prevent trajectory collapse."""

    def __init__(self, bandwidth: float = 1.0, feature_space: str = "latent"):
        """
        Args:
            bandwidth: RBF kernel bandwidth h.
            feature_space: "latent" or "clip".
        """
        super().__init__()
        self.bandwidth = bandwidth
        self.feature_space = feature_space

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (K, D) feature representations of each trajectory.
        Returns:
            loss: scalar RBF similarity (to be minimized).
        """
        features = features.float()
        K = features.shape[0]
        if K < 2:
            return torch.tensor(0.0, device=features.device)

        feat_mean = features.mean(dim=0, keepdim=True)
        feat_std = features.std(dim=0, keepdim=True, unbiased=False)
        features = (features - feat_mean) / (feat_std + 1e-6)

        dists_sq = torch.cdist(features, features, p=2.0).pow(2)

        exponent = -dists_sq / (self.bandwidth + 1e-8)
        exponent = torch.clamp(exponent, min=-60.0, max=0.0)
        rbf = torch.exp(exponent)

        mask = ~torch.eye(K, dtype=torch.bool, device=features.device)
        loss = rbf[mask].mean()

        return loss

    @staticmethod
    def compute_latent_features(latents: torch.Tensor) -> torch.Tensor:
        """Convert latents to feature vectors via spatial mean pooling."""
        K = latents.shape[0]
        features = latents.mean(dim=(-2, -1))
        return features
