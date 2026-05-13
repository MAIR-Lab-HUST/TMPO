"""Softmax-Trajectory Balance loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxTBLoss(nn.Module):
    """Softmax-TB loss: matches policy trajectory distribution to Boltzmann reward distribution."""

    def __init__(self, beta: float = 15.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        path_log_probs: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            path_log_probs: (K,) cumulative SDE log-prob per trajectory.
            rewards: (K,) terminal rewards.
        Returns:
            loss: scalar Softmax-TB loss.
        """
        lp = path_log_probs.float()
        rw = rewards.float()
        log_p_normalized = F.log_softmax(lp, dim=0)
        log_r_normalized = F.log_softmax(self.beta * rw, dim=0)
        residuals = log_p_normalized - log_r_normalized
        loss = (residuals ** 2).sum()
        return loss

    def compute_advantage(
        self,
        path_log_probs: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute detached advantage: A_i = log_softmax(beta*R) - log_softmax(log_P)."""
        lp = path_log_probs.detach().float()
        rw = rewards.detach().float()
        log_p = F.log_softmax(lp, dim=0)
        log_r = F.log_softmax(self.beta * rw, dim=0)
        return log_r - log_p

    def forward_per_path(
        self,
        path_log_probs: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-path residual squares (log_p - log_r)^2 in fp32."""
        lp = path_log_probs.float()
        rw = rewards.float()
        log_p = F.log_softmax(lp, dim=0)
        log_r = F.log_softmax(self.beta * rw, dim=0)
        return (log_p - log_r) ** 2
