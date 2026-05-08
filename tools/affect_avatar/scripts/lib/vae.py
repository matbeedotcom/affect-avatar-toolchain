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
    # Per [STAGE1_VAE_PLAN.md §4 Exp 1](../../STAGE1_VAE_PLAN.md):
    # - `deterministic=True` skips the reparam noise and uses z=mu always
    #   (training and inference). Pairs with `kl_weight=0` to make this
    #   a pure autoencoder. F1+F3 in the plan motivate the mode: KL was
    #   collapsing rare channels at d_lat=16, which has 2× the capacity
    #   needed for 95% variance.
    # - `output_activation="linear"` removes the final sigmoid in
    #   `decode`. The plan's reasoning: sigmoid compresses peaks
    #   (gradient → 0 at saturation), so high-amplitude channels like
    #   eyeWide and jawOpen can never reach their GT max. Linear gives
    #   the model freedom to push hard at peaks; the recon may
    #   occasionally drift outside [0, 1] but the loss pulls it back.
    deterministic: bool = False
    output_activation: str = "sigmoid"   # "sigmoid" | "linear"


# Channel groupings used by the deterministic-AE training loss.
# Indices into MEAD_3D's 54-channel actions in MediaPipe alphabetical
# (no leading `_neutral`) order. Channels 51..53 are unknown extras
# (omitted). `jawOpen=24` is in `mouth_speech` not `jaw` — it's
# perceptually lip-sync-relevant; the `jaw` group covers translational
# jaw motion only.
GROUP_INDICES: dict[str, tuple[int, ...]] = {
    "mouth_speech": (24, 26, 31, 32, 37, 38, 39, 40),
    "mouth_affect": (27, 28, 29, 30, 33, 34, 35, 36,
                     41, 42, 43, 44, 45, 46, 47, 48),
    "jaw":          (22, 23, 25),
    "eyes_brows":   (0, 1, 2, 3, 4,
                     8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                     18, 19, 20, 21),
    "cheeks_nose":  (5, 6, 7, 49, 50),
}


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
        """`(B, T/4, D) -> (B, T, K)` reconstruction.

        With `output_activation="sigmoid"` the result is bounded to
        [0, 1] (matches MEAD's blendshape range). With "linear" the
        result is unbounded — the model is free to overshoot at peaks
        and the loss is responsible for keeping it in range.
        """
        z_bdt = z_btd.transpose(1, 2)
        out = self.dec(z_bdt)
        if self.cfg.output_activation == "linear":
            return out.transpose(1, 2)
        return torch.sigmoid(out).transpose(1, 2)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.cfg.deterministic:
            return mu
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

    def vae_loss(
        self, x_btk: torch.Tensor, out: dict,
        channel_weights: Optional[torch.Tensor] = None,
    ) -> dict:
        recon = out["recon"]
        if recon.shape != x_btk.shape:
            # Stride-2 + stride-2 down + ×2 ×2 up = identity *only* if
            # T is divisible by 4. Crop to min length to be safe.
            T = min(recon.shape[1], x_btk.shape[1])
            recon = recon[:, :T]
            x_btk = x_btk[:, :T]
        sq_err = (recon - x_btk).pow(2)
        if channel_weights is None:
            recon_mse = sq_err.mean()
        else:
            # `channel_weights: (K,)` is normalized to mean-1 by the
            # caller, so the magnitude of `recon_mse` stays comparable
            # to the uniform-weighted version. Channels with low stdev
            # (eyeWide, cheekPuff) get boosted weight; high-stdev
            # channels (jawOpen, mouthSmile) get attenuated weight.
            recon_mse = (sq_err * channel_weights).mean()
        # KL(N(mu, sigma) || N(0, 1)) per element.
        kl = -0.5 * (1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
        kl = kl.mean()
        total = recon_mse + self.cfg.kl_weight * kl
        return {"loss": total, "recon_mse": recon_mse.detach(), "kl": kl.detach()}

    # ------------------------------------------------------------------
    # Deterministic-AE loss (Exp 1 per STAGE1_VAE_PLAN.md §4)
    # ------------------------------------------------------------------

    def ae_loss(
        self, x_btk: torch.Tensor, out: dict, *,
        alpha_value: float = 1.0,
        alpha_velocity: float = 0.5,
        alpha_peak: float = 0.5,
        group_weights: Optional[dict[str, float]] = None,
    ) -> dict:
        """Grouped MSE + velocity MSE + peak MSE — no KL.

        - **Grouped MSE**: per-group MSE (mean over channels, time,
          batch), then averaged across groups. Each of the 5 groups
          contributes equally regardless of channel count, so
          `mouth_speech` (8 channels) doesn't drown out `cheeks_nose`
          (5 channels). Per-group weights override this if supplied.
        - **Velocity MSE**: MSE on adjacent-frame deltas. Encourages
          the recon to track motion timing, not just static value.
        - **Peak MSE**: MSE on max-over-time per (batch, channel).
          Directly addresses F1's encoder-collapse-on-peaks finding —
          if the recon's per-channel max diverges from GT's max, this
          loss has gradient even when the value MSE is averaged out.
        """
        recon = out["recon"]
        if recon.shape != x_btk.shape:
            T = min(recon.shape[1], x_btk.shape[1])
            recon = recon[:, :T]
            x_btk = x_btk[:, :T]

        # --- Value MSE, grouped ---
        sq_err = (recon - x_btk).pow(2)               # (B, T, K)
        group_mses = []
        for name, indices in GROUP_INDICES.items():
            w = (group_weights or {}).get(name, 1.0)
            cols = sq_err[..., list(indices)]
            group_mses.append(w * cols.mean())
        grouped_mse = torch.stack(group_mses).mean()

        # --- Velocity MSE ---
        if recon.shape[1] >= 2:
            v_pred = recon[:, 1:] - recon[:, :-1]
            v_true = x_btk[:, 1:] - x_btk[:, :-1]
            velocity_mse = (v_pred - v_true).pow(2).mean()
        else:
            velocity_mse = recon.new_tensor(0.0)

        # --- Peak MSE ---
        peak_pred = recon.amax(dim=1)                 # (B, K)
        peak_true = x_btk.amax(dim=1)
        peak_mse = (peak_pred - peak_true).pow(2).mean()

        total = (alpha_value * grouped_mse
                 + alpha_velocity * velocity_mse
                 + alpha_peak * peak_mse)
        return {
            "loss": total,
            "grouped_mse": grouped_mse.detach(),
            "velocity_mse": velocity_mse.detach(),
            "peak_mse": peak_mse.detach(),
            # Plain element-wise MSE so logging is comparable across
            # AE and VAE training runs.
            "recon_mse": sq_err.mean().detach(),
        }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
