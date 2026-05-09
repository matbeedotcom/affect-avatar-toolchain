"""DiT-1D denoiser conditioned on Whisper hidden states + V/A/D coords.

Per [PROJECT_PLAN.md §4.2]
(../../docs/references/audio-blendshape-diffusion/PROJECT_PLAN.md):

  - 8 transformer blocks; d_model=384; 6 heads.
  - Per block: AdaLN-modulated self-attention over latents +
    cross-attention into Whisper hidden states + AdaLN-modulated MLP.
  - Conditioning: timestep (sinusoidal embedding) + V/A/D coords (16-d
    one-hot-style projection) summed and broadcast across blocks.
  - Classifier-free guidance: caller drops V/A/D embedding 10% of
    training steps and at inference scales between conditioned and
    unconditioned ε predictions.

Operates on `(B, T_lat, d_lat=16)` latents (output of Stage-1 VAE
encoder). Output is `(B, T_lat, d_lat)` ε prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiTConfig:
    d_lat: int = 16
    d_model: int = 384                 # ~37M params @ n_blocks=8 — pre-VAE-retrain baseline
    n_heads: int = 6
    n_blocks: int = 8
    d_whisper: int = 1280
    # Conditioning vector layout. d_vad=3 is the legacy "valence, arousal,
    # dominance" set. d_vad=4 adds a normalized intensity scalar in [0, 1]:
    # 0 = no emotional intensity (e.g. neutral clips); 0.33/0.67/1.0 =
    # MEAD intensity 1/2/3. Lets the model differentiate "subtle happy"
    # from "peak happy" — without it the V/A/D signal is identical
    # across MEAD intensities, and the model averages them.
    d_vad: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    n_timesteps: int = 1000


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer sinusoidal embedding (Vaswani 2017)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeMLP(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_timestep_embedding(t, self.d_model)
        return self.net(emb)


class VadMLP(nn.Module):
    """V/A/D coords (B, 3) → conditioning vector (B, d_model).

    The training loop is responsible for zeroing out the input on the
    10% of steps that drop V/A/D conditioning for classifier-free
    guidance — this module is a pure projector.
    """

    def __init__(self, d_vad: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_vad, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, vad: torch.Tensor) -> torch.Tensor:
        return self.net(vad)


# ---------------------------------------------------------------------------
# Attention primitives
# ---------------------------------------------------------------------------

class MultiheadAttention(nn.Module):
    """Standard MHA with separate q / k-v projections."""

    def __init__(self, d_q: int, d_kv: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_q % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_q // n_heads
        self.q = nn.Linear(d_q, d_q, bias=False)
        self.k = nn.Linear(d_kv, d_q, bias=False)
        self.v = nn.Linear(d_kv, d_q, bias=False)
        self.out = nn.Linear(d_q, d_q, bias=False)
        self.dropout = dropout

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        B, Tq, _ = x_q.shape
        Tk = x_kv.shape[1]
        H, D = self.n_heads, self.head_dim
        q = self.q(x_q).view(B, Tq, H, D).transpose(1, 2)
        k = self.k(x_kv).view(B, Tk, H, D).transpose(1, 2)
        v = self.v(x_kv).view(B, Tk, H, D).transpose(1, 2)
        # Scaled dot-product attention; dropout only in training.
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
        )
        out = attn.transpose(1, 2).reshape(B, Tq, H * D)
        return self.out(out)


# ---------------------------------------------------------------------------
# AdaLN-modulated transformer block
# ---------------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN: x -> (1+scale) * x + shift.

    `shift` / `scale` are `(B, d)` and broadcast across the time axis.
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Self-attn + cross-attn + MLP, each AdaLN-modulated by `cond`."""

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = MultiheadAttention(d, d, cfg.n_heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.cross_attn = MultiheadAttention(d, cfg.d_whisper, cfg.n_heads, cfg.dropout)
        self.norm3 = nn.LayerNorm(d, elementwise_affine=False)
        d_ff = int(d * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d),
        )
        # AdaLN-Zero (Peebles & Xie 2022): one Linear emits 9 parameters per block —
        # (shift1, scale1, gate1) for self-attn, (shift2, scale2, gate2) for
        # cross-attn, (shift3, scale3, gate3) for MLP. Initialized to zero
        # so block contributes nothing at the start of training.
        self.ada = nn.Linear(d, 9 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        whisper: torch.Tensor,
    ) -> torch.Tensor:
        s1, sc1, g1, s2, sc2, g2, s3, sc3, g3 = self.ada(cond).chunk(9, dim=-1)
        x = x + g1.unsqueeze(1) * self.self_attn(
            modulate(self.norm1(x), s1, sc1),
            modulate(self.norm1(x), s1, sc1),
        )
        x = x + g2.unsqueeze(1) * self.cross_attn(
            modulate(self.norm2(x), s2, sc2),
            whisper,
        )
        x = x + g3.unsqueeze(1) * self.mlp(modulate(self.norm3(x), s3, sc3))
        return x


# ---------------------------------------------------------------------------
# Full DiT-1D
# ---------------------------------------------------------------------------

class DiT1D(nn.Module):
    """ε-prediction denoiser: `(x_t, t, whisper, vad) -> ε̂` of shape `x_t`."""

    def __init__(self, cfg: Optional[DiTConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or DiTConfig()
        d = self.cfg.d_model

        self.x_proj = nn.Linear(self.cfg.d_lat, d)
        self.time_mlp = TimeMLP(d)
        self.vad_mlp = VadMLP(self.cfg.d_vad, d)
        # Whisper cross-attention K/V: LayerNorm first to centre/scale the
        # input. Layer −2 (pre-LN) hidden states from Whisper-large-v3-turbo
        # are heavily biased (mean ≈ +0.57, std ≈ 0.09); without this
        # normalisation, cross-attention scores blow up within ~30 training
        # steps on MPS. The Linear is a learnable channel-mixer on top.
        self.whisper_norm = nn.LayerNorm(self.cfg.d_whisper)
        self.whisper_proj = nn.Linear(self.cfg.d_whisper, self.cfg.d_whisper)

        self.blocks = nn.ModuleList(
            DiTBlock(self.cfg) for _ in range(self.cfg.n_blocks)
        )

        self.final_norm = nn.LayerNorm(d, elementwise_affine=False)
        # AdaLN for the final layer too (Peebles & Xie 2022 §3.4).
        self.final_ada = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.final_ada.weight)
        nn.init.zeros_(self.final_ada.bias)
        self.final_proj = nn.Linear(d, self.cfg.d_lat)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,        # (B, T_lat, d_lat)
        t: torch.Tensor,          # (B,) long
        whisper: torch.Tensor,    # (B, T_audio, d_whisper)
        vad: torch.Tensor,        # (B, d_vad)
    ) -> torch.Tensor:
        h = self.x_proj(x_t)
        cond = self.time_mlp(t) + self.vad_mlp(vad)
        whisper = self.whisper_proj(self.whisper_norm(whisper))
        for blk in self.blocks:
            h = blk(h, cond, whisper)

        s, sc = self.final_ada(cond).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), s, sc)
        return self.final_proj(h)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
