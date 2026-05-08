"""Stage-1 blendshape VAE.

Per [PROJECT_PLAN.md §4.1](../../docs/references/audio-blendshape-diffusion/PROJECT_PLAN.md):

  - Encoder: 4 Conv1d blocks (2 stride-2, 2 stride-1) → (T/4, d_lat=16).
  - Decoder: 4 ConvTranspose1d mirror; sigmoid output.
  - Loss: MSE + 0.001·KL.
  - Target param count: ~500k.

Operates on `(B, T, K)` blendshape tensors with K=54 (MEAD_3D native
vocabulary). Internal representation is `(B, K, T)` per Conv1d
convention; the public `encode`/`decode`/`forward` methods accept and
return `(B, T, K)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEConfig:
    k_dim: int = 54
    d_lat: int = 16
    hidden: int = 128
    kl_weight: float = 1e-3


class _ConvBlock(nn.Module):
    """Conv1d → GroupNorm → GELU. Optional stride-2 for temporal halving."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        # kernel_size=4 / padding=1 for stride-2 (halves T cleanly);
        # kernel_size=3 / padding=1 for stride-1 (preserves T).
        if stride == 2:
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        # `min(8, out_ch)` so we never request more groups than channels.
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(x)))


class _DeconvBlock(nn.Module):
    """ConvTranspose1d → GroupNorm → GELU. Mirror of `_ConvBlock`."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        if stride == 2:
            self.conv = nn.ConvTranspose1d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1
            )
        else:
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(x)))


class BlendshapeVAE(nn.Module):
    def __init__(self, cfg: Optional[VAEConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or VAEConfig()
        K = self.cfg.k_dim
        H = self.cfg.hidden
        D = self.cfg.d_lat

        # Encoder: K → H, T → T (stride-1 entry block)
        #          H → H, T → T/2 (stride-2)
        #          H → H, T → T/2 (stride-1)
        #          H → 2D, T → T/4 (stride-2; outputs concatenated mu+logvar)
        self.enc = nn.Sequential(
            _ConvBlock(K, H, stride=1),
            _ConvBlock(H, H, stride=2),
            _ConvBlock(H, H, stride=1),
            _ConvBlock(H, 2 * D, stride=2),
        )
        # Decoder: D → H, T*2 (stride-2)
        #          H → H, T  (stride-1)
        #          H → H, T*2 (stride-2)
        #          H → K, T  (stride-1, no activation; final sigmoid in `decode`)
        self.dec = nn.Sequential(
            _DeconvBlock(D, H, stride=2),
            _DeconvBlock(H, H, stride=1),
            _DeconvBlock(H, H, stride=2),
            nn.Conv1d(H, K, kernel_size=3, stride=1, padding=1),
        )

    # ------------------------------------------------------------------
    # Public API — operates on `(B, T, K)`
    # ------------------------------------------------------------------

    def encode(self, x_btk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(B, T, K) -> (mu, logvar)` of shape `(B, T/4, D)` each."""
        x_bkt = x_btk.transpose(1, 2)             # (B, K, T)
        h = self.enc(x_bkt)                       # (B, 2D, T/4)
        mu, logvar = h.chunk(2, dim=1)
        return mu.transpose(1, 2), logvar.transpose(1, 2)

    def decode(self, z_btd: torch.Tensor) -> torch.Tensor:
        """`(B, T/4, D) -> (B, T, K)` reconstruction in [0, 1]."""
        z_bdt = z_btd.transpose(1, 2)
        out = self.dec(z_bdt)
        return torch.sigmoid(out).transpose(1, 2)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x_btk: torch.Tensor) -> dict:
        mu, logvar = self.encode(x_btk)
        z = self.reparam(mu, logvar)
        recon = self.decode(z)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def vae_loss(self, x_btk: torch.Tensor, out: dict) -> dict:
        recon = out["recon"]
        if recon.shape != x_btk.shape:
            # Stride-2 + stride-2 down + ×2 ×2 up = identity *only* if
            # T is divisible by 4. Crop to min length to be safe.
            T = min(recon.shape[1], x_btk.shape[1])
            recon = recon[:, :T]
            x_btk = x_btk[:, :T]
        recon_mse = F.mse_loss(recon, x_btk)
        # KL(N(mu, sigma) || N(0, 1)) per element.
        kl = -0.5 * (1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
        kl = kl.mean()
        total = recon_mse + self.cfg.kl_weight * kl
        return {"loss": total, "recon_mse": recon_mse.detach(), "kl": kl.detach()}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
