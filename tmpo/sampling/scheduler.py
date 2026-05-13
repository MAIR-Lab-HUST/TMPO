"""Adaptive split scheduling via Beta distribution and progress-aware curriculum."""

import torch
import math
from typing import List, Optional


class AdaptiveScheduler:
    """Reward-driven adaptive split scheduler using Beta distribution."""

    def __init__(
        self,
        num_inference_steps: int = 28,
        num_splits: int = 3,
        kappa: float = 4.0,
        base_noise_levels: List[float] = None,
        r_min: float = 0.2,
        r_max: float = 0.35,
        ema_decay: float = 0.99,
        min_gap: int = 3,
        tail_guard_steps: int = 4,
        alpha_ema: float = 0.85,
        alpha_min: float = 0.10,
    ):
        """
        Args:
            num_inference_steps: total sampling steps.
            num_splits: number of split points.
            kappa: Beta distribution concentration.
            base_noise_levels: per-split base noise coefficients.
            r_min, r_max: initial reward bounds (EMA-updated online).
            ema_decay: EMA decay for reward bounds.
            min_gap: minimum gap between adjacent split steps.
            tail_guard_steps: safety margin from the end.
            alpha_ema: EMA smoothing for alpha.
            alpha_min: lower bound for alpha.
        """
        self.num_inference_steps = num_inference_steps
        self.num_splits = num_splits
        self.kappa = kappa
        self.base_noise_levels = base_noise_levels or [0.4, 0.7, 1.0]
        self.r_min = r_min
        self.r_max = r_max
        self.ema_decay = ema_decay
        self.min_gap = min_gap
        self.tail_guard_steps = max(2, int(tail_guard_steps))
        self.alpha_ema = alpha_ema
        self.alpha_min = alpha_min
        self._alpha_smoothed: Optional[float] = None

        spacing = num_inference_steps // (num_splits + 1)
        self.default_splits = [spacing * (i + 1) for i in range(num_splits)]

    def update_reward_bounds(self, rewards: torch.Tensor):
        """Update reward bounds via EMA."""
        batch_min = rewards.min().item()
        batch_max = rewards.max().item()
        self.r_min = self.ema_decay * self.r_min + (1 - self.ema_decay) * batch_min
        self.r_max = self.ema_decay * self.r_max + (1 - self.ema_decay) * batch_max

    def compute_alpha(self, mean_reward: float) -> float:
        """Compute EMA-smoothed normalized reward level alpha."""
        denom = self.r_max - self.r_min
        if denom < 1e-8:
            alpha_raw = 0.5
        else:
            alpha_raw = max(self.alpha_min, min(1.0, (mean_reward - self.r_min) / denom))

        if self._alpha_smoothed is None:
            self._alpha_smoothed = alpha_raw
        else:
            self._alpha_smoothed = (
                self.alpha_ema * self._alpha_smoothed
                + (1.0 - self.alpha_ema) * alpha_raw
            )
        return self._alpha_smoothed

    def compute_split_steps(self, mean_reward: float) -> List[int]:
        """Compute adaptive split positions via Beta distribution."""
        if self.kappa <= 0:
            return self.default_splits

        alpha = self.compute_alpha(mean_reward)

        a = 1.0 + (1.0 - alpha) * self.kappa
        b = 1.0 + alpha * self.kappa

        beta_dist = torch.distributions.Beta(a, b)
        fractions = beta_dist.sample((self.num_splits,)).sort().values

        margin = 2
        upper = self.num_inference_steps - self.tail_guard_steps
        if upper <= margin:
            upper = margin + 1
        effective_range = upper - margin
        split_steps = (fractions * effective_range + margin).long().tolist()

        for i in range(1, len(split_steps)):
            if split_steps[i] - split_steps[i - 1] < self.min_gap:
                split_steps[i] = split_steps[i - 1] + self.min_gap

        split_steps = [min(s, upper) for s in split_steps]

        return split_steps

    def compute_noise_levels(self, mean_reward: float) -> List[float]:
        """Adaptively scale noise levels based on reward difficulty."""
        alpha = self.compute_alpha(mean_reward)
        scale = 1.0 + (1.0 - alpha) * 0.3 - alpha * 0.2

        return [max(0.2, min(eta * scale, 1.0)) for eta in self.base_noise_levels]

    def get_schedule(self, mean_reward: Optional[float] = None):
        """Get full schedule (split_steps, noise_levels, alpha)."""
        if mean_reward is None:
            return self.default_splits, self.base_noise_levels, 0.5

        split_steps = self.compute_split_steps(mean_reward)
        noise_levels = self.compute_noise_levels(mean_reward)
        alpha = self.compute_alpha(mean_reward)

        return split_steps, noise_levels, alpha


class ProgressAwareSplitScheduler:
    """Training-progress-driven split scheduler with curriculum-guided Beta sampling."""

    def __init__(
        self,
        num_inference_steps: int = 28,
        early_splits: List[int] = None,
        late_splits: List[int] = None,
        noise_level: float = 0.8,
        noise_levels: List[float] = None,
        tail_guard_steps: int = 4,
        min_gap: int = 3,
        total_train_steps: int = 500,
        beta_kappa: float = 0.0,
    ):
        """
        Args:
            num_inference_steps: total sampling steps.
            early_splits: split positions at early training.
            late_splits: split positions at late training.
            noise_level: uniform noise coefficient (used if noise_levels is None).
            noise_levels: per-split noise coefficients.
            tail_guard_steps: safety margin from the end.
            min_gap: minimum gap between adjacent splits.
            total_train_steps: total training steps for progress calculation.
            beta_kappa: Beta sampling concentration (0 = deterministic curriculum).
        """
        self.num_inference_steps = num_inference_steps
        self.early_splits = early_splits or [4, 7, 12]
        self.late_splits  = late_splits  or [6, 12, 20]
        self.tail_guard_steps = max(2, int(tail_guard_steps))
        self.min_gap = min_gap
        self.total_train_steps = max(1, total_train_steps)
        self.beta_kappa = float(beta_kappa)

        assert len(self.early_splits) == len(self.late_splits), (
            f"early_splits/late_splits length mismatch: "
            f"{len(self.early_splits)} vs {len(self.late_splits)}"
        )
        self.num_splits = len(self.early_splits)

        if noise_levels is not None:
            assert len(noise_levels) == self.num_splits, (
                f"noise_levels length ({len(noise_levels)}) != num_splits ({self.num_splits})"
            )
            self._noise_levels = list(noise_levels)
        else:
            self._noise_levels = [float(noise_level)] * self.num_splits
        self.noise_level = self._noise_levels[0] if self._noise_levels else float(noise_level)

    def get_schedule(self, current_step: int) -> tuple:
        """Compute split steps and noise levels based on training progress."""
        progress = min(float(current_step) / float(self.total_train_steps), 1.0)
        s_min = 1
        upper = self.num_inference_steps - self.tail_guard_steps
        s_max = upper
        eff_range = max(s_max - s_min, 1)

        split_steps = []
        for early, late in zip(self.early_splits, self.late_splits):
            mu_cont = early + (late - early) * progress
            mu_cont = max(float(s_min), min(float(s_max), mu_cont))

            if self.beta_kappa > 0:
                # E[s_i] = mu_cont; Var ∝ mū(1-mū)/(κ+1)
                mu_bar = (mu_cont - s_min) / eff_range
                mu_bar = max(1e-3, min(1.0 - 1e-3, mu_bar))
                a = mu_bar * self.beta_kappa
                b = (1.0 - mu_bar) * self.beta_kappa
                xi = torch.distributions.Beta(a, b).sample().item()
                s = int(round(s_min + xi * eff_range))
            else:
                s = int(round(mu_cont))

            s = max(s_min, min(s, s_max))
            split_steps.append(s)

        split_steps.sort()

        for i in range(1, len(split_steps)):
            if split_steps[i] - split_steps[i - 1] < self.min_gap:
                split_steps[i] = split_steps[i - 1] + self.min_gap

        split_steps = [min(s, upper) for s in split_steps]

        noise_levels = list(self._noise_levels)

        return split_steps, noise_levels, progress

    def update_reward_bounds(self, rewards):
        """Compatibility no-op (ProgressAware does not use reward bounds)."""
        pass


def calculate_flux_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Compute Flux dynamic shift mu (linear interpolation by image sequence length)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def build_sigma_schedule(
    num_steps: int,
    shift: float = 3.0,
    device: str = "cpu",
    *,
    is_flux: bool = False,
    image_seq_len: int = None,
    use_flux_dynamic_shift: bool = True,
    flux_base_seq_len: int = 256,
    flux_max_seq_len: int = 4096,
    flux_base_shift: float = 0.5,
    flux_max_shift: float = 1.15,
) -> torch.Tensor:
    """Build flow-matching sigma schedule (SD3 static shift or Flux dynamic shift)."""
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    if is_flux and use_flux_dynamic_shift:
        if image_seq_len is None:
            raise ValueError("image_seq_len is required when using Flux dynamic sigma shift")
        mu = calculate_flux_shift(
            int(image_seq_len),
            base_seq_len=int(flux_base_seq_len),
            max_seq_len=int(flux_max_seq_len),
            base_shift=float(flux_base_shift),
            max_shift=float(flux_max_shift),
        )
        shift = math.exp(mu)

    if shift != 1.0:
        sigmas = shift * timesteps / (1.0 + (shift - 1.0) * timesteps)
    else:
        sigmas = timesteps

    return sigmas
