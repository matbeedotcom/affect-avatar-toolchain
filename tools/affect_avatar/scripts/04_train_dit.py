#!/usr/bin/env python3
"""Train Stage-2 DiT-1D denoiser on cached Whisper + VAE latents.

Per [PROJECT_PLAN.md §5 Stage 2]
(../../docs/references/audio-blendshape-diffusion/PROJECT_PLAN.md):

  audio (Whisper hidden state) + V/A/D coords → blendshape latent
  trajectory, via 1000-step cosine-noise diffusion + classifier-free
  guidance.

Three preconditions, all produced earlier in B1-B3 scaffolding:

  1. `artifacts/blendshape_vae.pt` — frozen Stage-1 VAE.
  2. `artifacts/whisper_cache/{i:04d}.npz` — pre-computed Whisper
     hidden states per clip.
  3. `data/mead_3d/train-00000-of-00001.parquet` — for the blendshape
     trajectories (encoded once via the frozen VAE at startup).

Outputs:
  artifacts/blendshape_dit.pt           — final EMA state dict
  artifacts/blendshape_dit_loss.csv     — per-epoch train+val ε-loss
  artifacts/blendshape_dit.report.md    — auto-generated summary

Usage:
  ./04_train_dit.py \\
      --vae artifacts/blendshape_vae.pt \\
      --whisper-cache artifacts/whisper_cache \\
      --parquet data/mead_3d/train-00000-of-00001.parquet \\
      --epochs 50 --batch-size 32 --crops-per-clip 8 \\
      --output artifacts/blendshape_dit.pt

  ./04_train_dit.py --epochs 5 --crops-per-clip 2 \\
      --output /tmp/dit_smoke.pt
      # 5-epoch smoke; ~5 min on M1 Pro MPS.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.diffusion import (  # noqa: E402
    CosineSchedule, DiffusionConfig, diffusion_loss, q_sample,
)
from lib.dit import DiT1D, DiTConfig, count_params  # noqa: E402
from lib.emotion_to_vad import emotion_to_vad  # noqa: E402
from lib.mead_3d_loader import (  # noqa: E402
    ACTION_DIM,
    MeadParquet,
    discover_parquets,
    parse_path,
)
from lib.vae import BlendshapeVAE, VAEConfig  # noqa: E402


# ------------------------------------------------------------------
# Constants tuned for MEAD_3D
# ------------------------------------------------------------------

# Audio frames per latent frame ≈ 8 for MEAD_3D (50 fps audio /
# 12.5 fps latent = 4 from VAE downsample × 2 from Whisper rate vs
# blendshape rate). Per-clip exact ratio is computed at clip-load time;
# this is just the smoke-test default.
DEFAULT_T_LAT = 16
DEFAULT_T_AUD = 128


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--parquet", type=Path, default=None,
        help="Single MEAD_3D parquet. Mutually exclusive with --data-dir.",
    )
    p.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory of `mead_3d*` subdirectories. Loads every parquet "
             "discovered. Cache lookup uses `{source_id}__{local_idx:04d}.npz`. "
             "Mutually exclusive with --parquet.",
    )
    p.add_argument(
        "--val-actors", type=str, default="video28",
        help="Comma-separated actor IDs held out for validation. "
             "(Only used in multi-actor mode.)",
    )
    p.add_argument(
        "--test-actors", type=str, default="M003",
        help="Comma-separated actor IDs held out for test. "
             "(Only used in multi-actor mode.)",
    )
    p.add_argument("--vae", type=Path, default=Path("artifacts/blendshape_vae.pt"))
    p.add_argument("--whisper-cache", type=Path,
                   default=Path("artifacts/whisper_cache"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--crops-per-clip", type=int, default=8)
    p.add_argument("--t-lat", type=int, default=DEFAULT_T_LAT,
                   help="Latent crop length (frames after VAE 4x downsample).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--cfg-drop-prob", type=float, default=0.0)
    p.add_argument(
        "--zscore-latents", action=argparse.BooleanOptionalAction, default=True,
        help="Per-channel z-score latents (computed on train split). "
             "Stats are saved with the checkpoint for inference. "
             "Default: enabled.",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/blendshape_dit.pt"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pick_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------
# Pre-loading: encode all blendshape clips through the frozen VAE
# and pull each clip's Whisper hidden states + V/A/D coords into RAM.
# ------------------------------------------------------------------

_VALID_EMOTIONS = (
    "neutral", "happy", "sad", "angry", "fear",
    "disgusted", "surprised", "contempt",
)


def load_clips_from_parquet(
    pq: MeadParquet,
    vae: BlendshapeVAE,
    whisper_cache_dir: Path,
    device: torch.device,
    t_lat: int,
) -> tuple[list[dict], int]:
    """Pre-compute and stash per-clip latents + Whisper + V/A/D for one source.

    Returns `(clips, skipped)`. Cache filename convention:
    `{pq.source_id}__{local_idx:04d}.npz` (matches
    `03_cache_whisper_embeddings.py`'s output).
    """

    paths = pq.all_paths()
    clips: list[dict] = []
    skipped = 0
    vae.eval()
    for row_idx in range(pq.num_rows):
        cache = whisper_cache_dir / f"{pq.source_id}__{row_idx:04d}.npz"
        if not cache.exists():
            skipped += 1
            continue
        path = paths[row_idx]
        meta = parse_path(path, actor_override=pq.actor_override) or {}
        emotion = meta.get("emotion", "")
        if emotion not in _VALID_EMOTIONS:
            skipped += 1
            continue

        rg, idx = _row_to_rg(pq, row_idx)
        bs_arr = np.asarray(
            pq._pf.read_row_group(rg, columns=["actions"])["actions"][idx].as_py(),
            dtype=np.float32,
        )
        if bs_arr.shape[0] < 4 * t_lat or bs_arr.shape[1] != ACTION_DIM:
            skipped += 1
            continue

        with torch.no_grad():
            x = torch.from_numpy(bs_arr).unsqueeze(0).to(device)
            T_crop = (x.shape[1] // 4) * 4
            x = x[:, :T_crop]
            mu, _logvar = vae.encode(x)
            latent = mu[0].cpu().numpy().astype(np.float32)

        whisper = np.load(cache)["hidden"].astype(np.float16)
        vad = np.array(emotion_to_vad(emotion), dtype=np.float32)

        clips.append({
            "source_id": pq.source_id,
            "local_idx": row_idx,
            "actor": meta.get("actor", ""),
            "emotion": emotion,
            "intensity": meta.get("intensity", 0),
            "latent": latent,
            "whisper": whisper,
            "vad": vad,
        })
    return clips, skipped


def load_clips(
    data_dir: Path | None,
    parquet_path: Path | None,
    vae: BlendshapeVAE,
    whisper_cache_dir: Path,
    device: torch.device,
    t_lat: int,
) -> list[dict]:
    """Load clips from either a multi-source data dir or a single parquet."""

    t0 = time.time()
    if data_dir is not None:
        triples = discover_parquets(data_dir)
        if not triples:
            raise RuntimeError(f"no `mead_3d*` parquets discovered under {data_dir}")
        all_clips: list[dict] = []
        total_skipped = 0
        for path, actor, sid in triples:
            pq = MeadParquet(path, actor_override=actor, source_id=sid)
            clips, skipped = load_clips_from_parquet(
                pq, vae, whisper_cache_dir, device, t_lat,
            )
            print(
                f"  {pq.source_id:30s}  kept={len(clips):4d}  "
                f"skipped={skipped:4d}",
                file=sys.stderr,
            )
            all_clips.extend(clips)
            total_skipped += skipped
        print(
            f"  total {len(all_clips)} clips, {total_skipped} skipped "
            f"({time.time() - t0:.1f}s)",
            file=sys.stderr,
        )
        return all_clips

    pq = MeadParquet(parquet_path)
    clips, skipped = load_clips_from_parquet(
        pq, vae, whisper_cache_dir, device, t_lat,
    )
    print(
        f"  {len(clips)} clips kept, {skipped} skipped "
        f"({time.time() - t0:.1f}s)",
        file=sys.stderr,
    )
    return clips


def _row_to_rg(pq: MeadParquet, row_idx: int) -> tuple[int, int]:
    """Resolve a flat row index to `(row_group, idx_in_group)`."""
    cur = 0
    for rg in range(pq.num_row_groups):
        n = pq._pf.metadata.row_group(rg).num_rows
        if row_idx < cur + n:
            return rg, row_idx - cur
        cur += n
    raise IndexError(row_idx)


def split_clips(
    clips: list[dict],
    *,
    val_actors: set[str] | None = None,
    test_actors: set[str] | None = None,
) -> dict[str, list[dict]]:
    """Speaker-disjoint split.

    - Multi-actor mode (val_actors / test_actors specified): train is the
      complement of (val ∪ test). Designed for the 11-actor MEAD_3D
      union — caller supplies actor IDs to hold out. Default suggested
      split: val={video28}, test={M003}, train=everyone else.
    - Single-actor legacy mode (no val/test_actors): falls back to the
      original 2-speaker layout for backward compat with the
      pre-multi-source baseline.
    """
    if val_actors or test_actors:
        val_set = val_actors or set()
        test_set = test_actors or set()
        train = [c for c in clips if c["actor"] not in val_set | test_set]
        val = sorted(
            (c for c in clips if c["actor"] in val_set),
            key=lambda c: (c["actor"], c["source_id"], c["local_idx"]),
        )
        test = sorted(
            (c for c in clips if c["actor"] in test_set),
            key=lambda c: (c["actor"], c["source_id"], c["local_idx"]),
        )
        return {"train": train, "val": val, "test": test}

    # Legacy 2-speaker fallback.
    train = [c for c in clips if c["actor"] == "W009"]
    other = sorted(
        (c for c in clips if c["actor"] == "video28"),
        key=lambda c: (c["source_id"], c["local_idx"]),
    )
    half = len(other) // 2
    return {"train": train, "val": other[:half], "test": other[half:]}


# ------------------------------------------------------------------
# Random-crop dataset
# ------------------------------------------------------------------

class DiTDataset(Dataset):
    """Yields aligned `(latent_crop, whisper_crop, vad)` tuples.

    Latent crop is `(t_lat, d_lat)`; whisper crop is `(t_aud, 1280)`
    where `t_aud` is computed per clip to span the same temporal
    extent as the latent crop. Variable `t_aud` would break batching,
    so we *resample* whisper crops to a fixed length `t_aud_target`
    via linear interpolation along the time axis.
    """

    def __init__(
        self,
        clips: list[dict],
        t_lat: int,
        t_aud_target: int,
        crops_per_clip: int,
        seed: int = 0,
    ) -> None:
        self.clips = clips
        self.t_lat = t_lat
        self.t_aud_target = t_aud_target
        self.crops_per_clip = crops_per_clip
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.clips) * self.crops_per_clip

    def __getitem__(self, idx: int):
        clip = self.clips[idx % len(self.clips)]
        T_lat_full = clip["latent"].shape[0]
        T_aud_full = clip["whisper"].shape[0]

        max_lat_start = max(0, T_lat_full - self.t_lat)
        s_lat = int(self._rng.integers(0, max_lat_start + 1)) if max_lat_start > 0 else 0
        latent_crop = clip["latent"][s_lat : s_lat + self.t_lat]

        # Time-aligned audio span: (s_lat / T_lat_full) ... ((s_lat+t_lat) / T_lat_full)
        # of the audio. Round to nearest integer audio frame, clip to bounds.
        t0 = int(round(s_lat / T_lat_full * T_aud_full))
        t1 = int(round((s_lat + self.t_lat) / T_lat_full * T_aud_full))
        t1 = max(t1, t0 + 1)
        t1 = min(t1, T_aud_full)
        whisper_span = clip["whisper"][t0:t1].astype(np.float32)
        whisper_crop = _resample_time(whisper_span, self.t_aud_target)

        return (
            torch.from_numpy(latent_crop),
            torch.from_numpy(whisper_crop),
            torch.from_numpy(clip["vad"]),
        )


def _resample_time(x: np.ndarray, target_T: int) -> np.ndarray:
    """Linear-interp resample `(T, D) -> (target_T, D)` along time axis."""
    T = x.shape[0]
    if T == target_T:
        return x
    # Map target indices [0, target_T) to [0, T-1] in source.
    src_idx = np.linspace(0, T - 1, target_T)
    lo = np.floor(src_idx).astype(np.int64)
    hi = np.minimum(lo + 1, T - 1)
    w = (src_idx - lo).astype(np.float32)[:, None]
    return ((1 - w) * x[lo] + w * x[hi]).astype(np.float32)


# ------------------------------------------------------------------
# EMA helper (manual; AveragedModel is tied to a sequence of callbacks
# we don't want here)
# ------------------------------------------------------------------

class EMA:
    """Exponential moving average with PyTorch-standard warmup.

    Effective decay at step `t` is `min(target_decay, (1+t)/(10+t))`,
    so the shadow weights match the online weights for the first ~10
    steps and asymptote toward `target_decay` afterwards. This is
    important for short training runs — without warmup, decay=0.9999
    leaves the EMA still ~50% random-init after several thousand
    steps, making val/test loss meaningless.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.target_decay = decay
        self.step = 0
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def _decay_at(self, step: int) -> float:
        return min(self.target_decay, (1.0 + step) / (10.0 + step))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step += 1
        d = self._decay_at(self.step)
        for sp, p in zip(self.shadow.parameters(), model.parameters()):
            sp.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for sb, b in zip(self.shadow.buffers(), model.buffers()):
            sb.copy_(b)


# ------------------------------------------------------------------
# Train / eval loops
# ------------------------------------------------------------------

def train_one_epoch(
    *,
    model: DiT1D,
    schedule: CosineSchedule,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    ema: EMA,
    device: torch.device,
    cfg_drop_prob: float,
) -> dict:
    model.train()
    total_loss = 0.0
    n = 0
    for latent, whisper, vad in loader:
        latent = latent.to(device, non_blocking=False)
        whisper = whisper.to(device, non_blocking=False)
        vad = vad.to(device, non_blocking=False)

        # Random t per sample.
        t = torch.randint(0, schedule.cfg.n_timesteps, (latent.shape[0],),
                          device=device)
        x_t, true_noise = q_sample(schedule, latent, t)

        # CFG drop on V/A/D: zero out for cfg_drop_prob of samples.
        if cfg_drop_prob > 0:
            drop_mask = (torch.rand(vad.shape[0], device=device) < cfg_drop_prob)
            vad = vad.clone()
            vad[drop_mask] = 0.0

        pred = model(x_t, t, whisper, vad)
        loss = diffusion_loss(schedule, pred, true_noise)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        ema.update(model)

        bs = latent.shape[0]
        total_loss += loss.item() * bs
        n += bs
    return {"loss": total_loss / max(n, 1)}


@torch.no_grad()
def eval_loss(
    *, model, schedule, loader, device,
) -> dict:
    model.eval()
    total = 0.0
    n = 0
    for latent, whisper, vad in loader:
        latent = latent.to(device)
        whisper = whisper.to(device)
        vad = vad.to(device)
        t = torch.randint(0, schedule.cfg.n_timesteps, (latent.shape[0],),
                          device=device)
        x_t, true_noise = q_sample(schedule, latent, t)
        pred = model(x_t, t, whisper, vad)
        loss = diffusion_loss(schedule, pred, true_noise)
        bs = latent.shape[0]
        total += loss.item() * bs
        n += bs
    return {"loss": total / max(n, 1)}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    print(f"device: {device}", file=sys.stderr)

    # --- Load frozen VAE ---
    print(f"loading VAE from {args.vae}...", file=sys.stderr)
    ckpt = torch.load(args.vae, map_location=device, weights_only=False)
    vae_cfg = VAEConfig(**ckpt["config"])
    vae = BlendshapeVAE(vae_cfg).to(device).eval()
    vae.load_state_dict(ckpt["state_dict"])
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"  VAE params: {sum(p.numel() for p in vae.parameters()):,} (frozen)",
          file=sys.stderr)

    # --- Pre-load every clip ---
    print(f"\npre-loading clips...", file=sys.stderr)
    t_aud_target = args.t_lat * 8   # ~8 audio frames per latent frame in MEAD_3D
    if args.parquet and args.data_dir:
        print("error: --parquet and --data-dir are mutually exclusive",
              file=sys.stderr)
        return 2
    if not args.parquet and not args.data_dir:
        # Default: legacy single-parquet mode.
        args.parquet = Path("data/mead_3d/train-00000-of-00001.parquet")

    print(f"\npre-loading clips...", file=sys.stderr)
    clips = load_clips(
        data_dir=args.data_dir,
        parquet_path=args.parquet,
        vae=vae,
        whisper_cache_dir=args.whisper_cache,
        device=device,
        t_lat=args.t_lat,
    )

    if args.data_dir:
        val_actors = {a.strip() for a in args.val_actors.split(",") if a.strip()}
        test_actors = {a.strip() for a in args.test_actors.split(",") if a.strip()}
        splits = split_clips(clips, val_actors=val_actors,
                             test_actors=test_actors)
    else:
        splits = split_clips(clips)
    for k, v in splits.items():
        print(f"  {k:5s}: {len(v):4d} clips", file=sys.stderr)
    if not splits["train"] or not splits["val"]:
        print("error: empty split", file=sys.stderr)
        return 2

    # Per-channel latent z-scoring. Stats computed on training latents only;
    # applied in-place to all splits. Saved with the checkpoint so inference
    # can undo: x = z * std + mu before VAE decode. Diagnostic showed the
    # baseline VAE produces latents with per-channel std ranging 0.05..1.5;
    # without z-scoring, low-amplitude channels contribute almost no signal
    # to the noisy latent and the DiT learns nothing about them.
    latent_mu = np.zeros((vae_cfg.d_lat,), dtype=np.float32)
    latent_std = np.ones((vae_cfg.d_lat,), dtype=np.float32)
    if args.zscore_latents:
        train_latents = np.concatenate(
            [c["latent"] for c in splits["train"]], axis=0,
        )
        latent_mu = train_latents.mean(axis=0).astype(np.float32)
        latent_std = train_latents.std(axis=0).astype(np.float32)
        # Floor std to avoid amplifying pure-noise dead channels into the
        # signal range. Channels with std < 0.05 are typically uninformative;
        # leave them near-zero post-normalization rather than blowing them up.
        latent_std = np.maximum(latent_std, 0.05).astype(np.float32)
        print(
            f"latent stats (train, per-channel): "
            f"mu range [{latent_mu.min():+.3f}, {latent_mu.max():+.3f}], "
            f"std range [{latent_std.min():.3f}, {latent_std.max():.3f}]",
            file=sys.stderr,
        )
        for subset in splits.values():
            for c in subset:
                c["latent"] = ((c["latent"] - latent_mu) / latent_std).astype(np.float32)

    train_ds = DiTDataset(splits["train"], args.t_lat, t_aud_target,
                          crops_per_clip=args.crops_per_clip, seed=args.seed)
    val_ds = DiTDataset(splits["val"], args.t_lat, t_aud_target,
                        crops_per_clip=max(args.crops_per_clip // 2, 1),
                        seed=args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, num_workers=0)

    # --- Build DiT + schedule + EMA ---
    dit_cfg = DiTConfig(d_lat=vae_cfg.d_lat, n_timesteps=1000)
    model = DiT1D(dit_cfg).to(device)
    print(f"\nDiT params: {count_params(model):,}", file=sys.stderr)

    schedule = CosineSchedule(DiffusionConfig(
        n_timesteps=dit_cfg.n_timesteps,
        cfg_drop_prob=args.cfg_drop_prob,
    )).to(device)

    ema = EMA(model, decay=args.ema_decay)
    ema.shadow.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr * 1e-2,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    loss_csv = args.output.with_suffix(".csv")
    csv_fh = loss_csv.open("w", newline="")
    csv_w = csv.writer(csv_fh)
    csv_w.writerow(["epoch", "lr", "train_loss", "val_online", "val_ema"])

    print(f"\ntraining {args.epochs} epochs...", file=sys.stderr)
    t_train = time.time()
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    for ep in range(args.epochs):
        t0 = time.time()
        tr = train_one_epoch(
            model=model, schedule=schedule, loader=train_loader,
            optim=optim, ema=ema, device=device,
            cfg_drop_prob=args.cfg_drop_prob,
        )
        va_online = eval_loss(model=model, schedule=schedule,
                              loader=val_loader, device=device)
        va_ema = eval_loss(model=ema.shadow, schedule=schedule,
                           loader=val_loader, device=device)
        sched.step()
        cur_lr = sched.get_last_lr()[0]
        ep_dt = time.time() - t0
        marker = ""
        if va_ema["loss"] < best_val:
            best_val = va_ema["loss"]
            best_epoch = ep + 1
            # CPU-offload the snapshot so we don't hold two MPS-resident
            # copies of the model while training continues.
            best_state = {k: v.detach().cpu().clone()
                          for k, v in ema.shadow.state_dict().items()}
            marker = "  <-- best"
        print(
            f"  ep{ep + 1:3d}/{args.epochs}  "
            f"train_eps={tr['loss']:.5f}  "
            f"val_online={va_online['loss']:.5f}  "
            f"val_ema={va_ema['loss']:.5f}  "
            f"lr={cur_lr:.2e}  ({ep_dt:.1f}s){marker}",
            file=sys.stderr,
        )
        csv_w.writerow([
            ep + 1, f"{cur_lr:.4e}",
            f"{tr['loss']:.6f}",
            f"{va_online['loss']:.6f}",
            f"{va_ema['loss']:.6f}",
        ])
        csv_fh.flush()
    va = va_ema  # report-friendly alias

    csv_fh.close()
    print(
        f"\ntraining done; total {time.time() - t_train:.0f}s  "
        f"best val_eps={best_val:.5f}",
        file=sys.stderr,
    )

    # Restore the best-EMA snapshot before saving — final-epoch EMA
    # is typically overfit (val-loss climbs ~2× from minimum on a 2-speaker
    # split). The on-disk checkpoint should be the model the test
    # numbers actually reflect.
    if best_state is not None:
        ema.shadow.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(
            f"restored best-EMA snapshot from epoch {best_epoch} "
            f"(val={best_val:.5f})",
            file=sys.stderr,
        )

    print(f"saving best-EMA -> {args.output}", file=sys.stderr)
    torch.save({
        "state_dict": ema.shadow.state_dict(),
        "config": dit_cfg.__dict__,
        "best_val_eps": float(best_val),
        "best_epoch": int(best_epoch),
        "epochs_trained": args.epochs,
        "final_val_eps": float(va_ema["loss"]),
        "n_train_clips": len(splits["train"]),
        "n_val_clips": len(splits["val"]),
        "n_test_clips": len(splits["test"]),
        "zscore_latents": bool(args.zscore_latents),
        "latent_mu": latent_mu,
        "latent_std": latent_std,
    }, args.output)

    # Final test eval on the best-EMA weights.
    test_ds = DiTDataset(splits["test"], args.t_lat, t_aud_target,
                         crops_per_clip=4, seed=args.seed + 2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, drop_last=False)
    test_metrics = eval_loss(model=ema.shadow, schedule=schedule,
                             loader=test_loader, device=device)

    _write_report(
        path=args.output.with_suffix(".report.md"),
        cfg=dit_cfg,
        n_params=count_params(model),
        n_clips=len(clips),
        splits=splits,
        train_eps_final=tr["loss"],
        val_eps_final=va["loss"],
        val_eps_best=best_val,
        best_epoch=best_epoch,
        test_eps_best=test_metrics["loss"],
        epochs=args.epochs, device=str(device),
    )
    print(f"test_eps(best-EMA)={test_metrics['loss']:.5f}", file=sys.stderr)
    return 0


def _write_report(*, path, cfg, n_params, n_clips, splits,
                  train_eps_final, val_eps_final, val_eps_best,
                  best_epoch, test_eps_best, epochs, device) -> None:
    lines: list[str] = []
    lines.append("# Stage-2 DiT-1D training report\n\n")
    lines.append(f"- Device: `{device}`\n")
    lines.append(f"- Epochs trained: {epochs}\n")
    lines.append(f"- Params: {n_params:,}\n")
    lines.append(f"- d_lat: {cfg.d_lat}  d_model: {cfg.d_model}  "
                 f"n_blocks: {cfg.n_blocks}  n_heads: {cfg.n_heads}\n")
    lines.append(f"- Diffusion timesteps: {cfg.n_timesteps}  "
                 f"d_whisper: {cfg.d_whisper}\n\n")
    lines.append("## Splits\n\n")
    for k in ("train", "val", "test"):
        lines.append(f"- {k}: {len(splits[k])} clips\n")

    lines.append("\n## Final ε-loss snapshot (saved checkpoint = best-EMA from "
                 f"epoch {best_epoch})\n\n")
    lines.append("| split | ε-MSE | weights | source |\n|---|---:|---|---|\n")
    lines.append(f"| train | {train_eps_final:.5f} | online | last epoch |\n")
    lines.append(f"| val   | {val_eps_best:.5f} | EMA | "
                 f"best (epoch {best_epoch}) |\n")
    lines.append(f"| test  | {test_eps_best:.5f} | EMA | "
                 f"evaluated on best-EMA |\n")
    lines.append(
        f"\n- Last-epoch val_ema: {val_eps_final:.5f} "
        f"(diverged from best by {val_eps_final - val_eps_best:+.3f}; "
        f"clear overfit signal — see Notes).\n"
    )

    lines.append("\n## Notes\n\n")
    lines.append("- ε-MSE around `1.0` indicates no learning (zero is a "
                 "competitive baseline). Below `0.5` is real denoising "
                 "signal. Below `0.3` is decent for a first cycle.\n")
    lines.append("- The val curve typically bottoms out then climbs as the "
                 "model overfits the training-speaker crops. We export the "
                 "best-EMA snapshot — the saved checkpoint is the model "
                 "from the val-minimum epoch, not the final epoch.\n")
    lines.append("- Cross-speaker generalization gap is the dominant cost "
                 "per RB7 (only 2 speakers in the main parquet). Mitigation: "
                 "fan out to per-actor companion parquets (W040 / W026 / "
                 "W035 / M003) before the next training cycle.\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


if __name__ == "__main__":
    sys.exit(main())
