import math
from typing import List, Optional, Union

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

def flow_match_euler_discrete_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    per_token_timesteps: Optional[torch.Tensor] = None,
    prev_sample: Optional[torch.Tensor] = None,
    noise_level: float = 0.7,
    log_std_min: float = -5,
    log_std_max: float = 2,
    tree_k: int = 1,
    use_ode: bool = False,
):

    if (
        isinstance(timestep, int)
        or isinstance(timestep, torch.IntTensor)
        or isinstance(timestep, torch.LongTensor)
    ):
        raise ValueError(
            (
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                " `FlowMatchEulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                " one of the `scheduler.timesteps` as a timestep."
            ),
        )

    # if self.step_index is None:
    self._init_step_index(timestep)

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)

    if per_token_timesteps is not None:
        per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps

        sigmas = self.sigmas[:, None, None]
        lower_mask = sigmas < per_token_sigmas[None] - 1e-6
        lower_sigmas = lower_mask * sigmas
        lower_sigmas, _ = lower_sigmas.max(dim=0)

        current_sigma = per_token_sigmas[..., None]
        next_sigma = lower_sigmas[..., None]
        dt = current_sigma - next_sigma
    else:
        sigma_idx = self.step_index
        sigma = self.sigmas[sigma_idx]
        sigma_next = self.sigmas[sigma_idx + 1]

        current_sigma = sigma
        next_sigma = sigma_next
        dt = sigma_next - sigma
    
    if use_ode:
        return [sample + dt * model_output], [None]

    # ODE -> SDE
    sigma_max = self.sigmas[1].item()
    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level
    
    # our sde
    prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
    std = std_dev_t * torch.sqrt(-1*dt)
    std = torch.clamp(std, math.exp(log_std_min), math.exp(log_std_max))
    if prev_sample is None:
        prev_samples = []
        for _ in range(tree_k):
            variance_noise = torch.randn_like(sample)
            prev_sample = prev_sample_mean + std * variance_noise
            prev_samples.append(prev_sample)
    elif not isinstance(prev_sample, List):
        prev_samples = [prev_sample]
    else:
        prev_samples = prev_sample

    log_probs = []
    for prev_sample in prev_samples:
        assert not isinstance(prev_sample, List)
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std**2))
            - torch.log(std)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )


        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        log_probs.append(log_prob)
    return prev_samples, log_probs
