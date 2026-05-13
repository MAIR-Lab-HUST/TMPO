"""Reference model constraint loss to prevent policy drift."""

import torch
import torch.nn as nn


class ReferenceConstraintLoss(nn.Module):
    """Length-normalized MSE between current and reference log-probs."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        current_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        num_sde_steps: int = 3,
    ) -> torch.Tensor:
        """
        Args:
            current_log_probs: (K,) trajectory log-prob sums from current policy.
            ref_log_probs: (K,) trajectory log-prob sums from reference model.
            num_sde_steps: number of SDE steps for length normalization.
        Returns:
            loss: scalar.
        """
        norm_current = current_log_probs / max(num_sde_steps, 1)
        norm_ref = ref_log_probs / max(num_sde_steps, 1)

        loss = ((norm_current - norm_ref) ** 2).mean()

        return loss
