"""Diffusion noise schedule + DDIM sampler.

Per [PROJECT_PLAN.md §4.2 + §4.3]
(../../docs/references/audio-blendshape-diffusion/PROJECT_PLAN.md):

  - Cosine noise schedule (Nichol & Dhariwal 2021).
  - 1000 train timesteps; 50-step DDIM at inference.
  - ε-prediction MSE loss on Stage-1 VAE latents (B, T_lat, d_lat).

Conventions:
  - `t` is a 1-D LongTensor of shape `(B,)`, one timestep per batch sample.
  - All schedule tensors are 1-D over T.
  - The model predicts ε given (x_t, t, conditioning).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    n_timesteps: int = 1000
    cosine_s: float = 0.008      # Nichol & Dhariwal default
    cfg_drop_prob: float = 0.10  # CFG: drop conditioning 10% of training steps


class CosineSchedule:
    """Nichol & Dhariwal 2021 cosine alpha-bar schedule."""

    def __init__(self, cfg: Optional[DiffusionConfig] = None) -> None:
        self.cfg = cfg or DiffusionConfig()
        T = self.cfg.n_timesteps
        s = self.cfg.cosine_s

        # alpha_bar_t = cos^2((t/T + s) / (1 + s) * pi/2)
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        f = torch.cos((steps + s) / (1 + s) * torch.pi / 2) ** 2
        alpha_bar = f / f[0]
        # betas_t = clip(1 - alpha_bar_t / alpha_bar_{t-1}, 0, 0.999)
        betas = torch.clamp(1.0 - alpha_bar[1:] / alpha_bar[:-1], 0.0, 0.999)
        alphas = 1.0 - betas

        self.betas = betas.to(torch.float32)
        self.alphas = alphas.to(torch.float32)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def to(self, device: torch.device) -> "CosineSchedule":
        for name in (
            "betas", "alphas", "alphas_cumprod",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self


def _gather(s: torch.Tensor, t: torch.Tensor, broadcast_shape: tuple) -> torch.Tensor:
    """Index 1-D `s` by `t` and reshape so it broadcasts to `broadcast_shape`."""
    out = s.gather(0, t)
    while out.dim() < len(broadcast_shape):
        out = out[..., None]
    return out


def q_sample(
    schedule: CosineSchedule,
    x_0: torch.Tensor,
    t: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward process: x_0 -> x_t = sqrt(α_t) x_0 + sqrt(1-α_t) ε.

    Returns `(x_t, noise)`. Noise is sampled if not provided.
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    sa = _gather(schedule.sqrt_alphas_cumprod, t, x_0.shape)
    s1 = _gather(schedule.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return sa * x_0 + s1 * noise, noise


def diffusion_loss(
    schedule: CosineSchedule,
    pred_noise: torch.Tensor,
    true_noise: torch.Tensor,
) -> torch.Tensor:
    """ε-prediction MSE."""
    return F.mse_loss(pred_noise, true_noise)


@torch.no_grad()
def ddim_sample(
    schedule: CosineSchedule,
    model_fn,                          # callable(x_t, t, *cond) -> ε_hat
    shape: tuple,
    *,
    n_steps: int = 50,
    device: torch.device | str = "cpu",
    cond: Optional[tuple] = None,
    eta: float = 0.0,                  # 0 = deterministic DDIM
    cfg_scale: float = 1.0,            # 1.0 = no CFG
    null_cond: Optional[tuple] = None,
) -> torch.Tensor:
    """DDIM sampler.

    `model_fn(x_t, t, *cond) -> ε_hat`. With CFG, calls also with
    `null_cond` and blends:
        ε = (1-w)·ε_uncond + w·ε_cond,  with w = cfg_scale.
    """

    T = schedule.cfg.n_timesteps
    # Evenly-spaced timestep indices, descending from T-1 to 0.
    step_indices = torch.linspace(T - 1, 0, n_steps + 1, dtype=torch.long, device=device)
    x = torch.randn(shape, device=device)

    for i in range(n_steps):
        t_cur = step_indices[i].expand(shape[0])
        t_nxt = step_indices[i + 1]

        eps = model_fn(x, t_cur, *cond) if cond else model_fn(x, t_cur)
        if cfg_scale != 1.0 and null_cond is not None:
            eps_uncond = model_fn(x, t_cur, *null_cond)
            eps = (1.0 - cfg_scale) * eps_uncond + cfg_scale * eps

        a_cur = _gather(schedule.alphas_cumprod, t_cur, x.shape)
        # x_0_hat = (x_t - sqrt(1-α_t)·ε) / sqrt(α_t)
        x0 = (x - torch.sqrt(1 - a_cur) * eps) / torch.sqrt(a_cur)
        if t_nxt < 0:
            x = x0
            break
        a_nxt = schedule.alphas_cumprod[t_nxt]
        sigma = eta * torch.sqrt((1 - a_nxt) / (1 - a_cur)) * torch.sqrt(1 - a_cur / a_nxt)
        noise = torch.randn_like(x) if eta > 0 else 0.0
        # DDIM update.
        x = (
            torch.sqrt(a_nxt) * x0
            + torch.sqrt(torch.clamp(1 - a_nxt - sigma ** 2, min=0)) * eps
            + sigma * noise
        )
    return x
