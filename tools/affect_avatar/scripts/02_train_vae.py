#!/usr/bin/env python3
"""Train Stage-1 blendshape VAE on MEAD_3D.

Per [PROJECT_PLAN.md §4.1 + §5 Stage 1]
(../../docs/references/audio-blendshape-diffusion/PROJECT_PLAN.md).

Reads the BarryFutureman/MEAD_3D parquet, random-crops 64-frame
windows from each clip, and trains the VAE in `lib/vae.py` to
reconstruct them. Speaker-disjoint train/val/test split:

    train: W009 (654 clips)
    val:   first half of video28 (329 clips)
    test:  second half of video28 (329 clips)

Outputs:
    artifacts/blendshape_vae.pt           — final state dict
    artifacts/blendshape_vae_loss.csv     — per-epoch train+val loss
    artifacts/blendshape_vae_recon_report.md — final stats summary

Usage:
  ./02_train_vae.py \\
      --parquet data/mead_3d/train-00000-of-00001.parquet \\
      --epochs 50 \\
      --batch-size 64 \\
      --crops-per-clip 8 \\
      --output artifacts/blendshape_vae.pt

  ./02_train_vae.py --epochs 5 --crops-per-clip 2 \\
      --output /tmp/vae_smoke.pt
      # 5-epoch smoke run; useful for verifying gradient flow before
      # committing to a full 50-epoch run.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.mead_3d_loader import (  # noqa: E402
    ACTION_DIM,
    MeadParquet,
    MultiMeadParquet,
    discover_parquets,
    parse_path,
)
from lib.vae import BlendshapeVAE, VAEConfig, count_params  # noqa: E402


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
             "discovered and dedupes by (actor, emotion, intensity, utt). "
             "Mutually exclusive with --parquet.",
    )
    p.add_argument(
        "--val-actors", type=str, default="video28",
        help="Comma-separated actor IDs held out for validation. "
             "Used in multi-source mode; legacy 2-speaker fallback "
             "ignores this.",
    )
    p.add_argument(
        "--test-actors", type=str, default="M003",
        help="Comma-separated actor IDs held out for test. "
             "Used in multi-source mode.",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--crops-per-clip", type=int, default=8,
                   help="Random crops sampled per clip per epoch.")
    p.add_argument("--crop-len", type=int, default=64,
                   help="Crop length in frames (default ~2 s at 30 fps).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--kl-weight", type=float, default=1e-3)
    p.add_argument(
        "--channel-weight-power", type=float, default=0.0,
        help="Power applied to per-channel inverse-stdev weights in the "
             "reconstruction loss. 0.0 = uniform (legacy). 1.0 = pure "
             "inverse-stdev (rare/extreme channels boosted hardest). "
             "0.5 = sqrt-inverse-stdev (gentler). Diagnostic round-trip "
             "showed that low-variance eye-region and cheek channels "
             "(eyeWide, cheekPuff, browDn) collapse to near-zero in "
             "VAE recon at power=0; values in [0.5, 1.0] should rebalance.",
    )
    p.add_argument("--d-lat", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--device", default=None,
                   help="cpu / mps / cuda. Auto-pick if unset.")
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/blendshape_vae.pt"))
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
# Data — pre-load all blendshape arrays into RAM
# ------------------------------------------------------------------

def load_blendshape_clips_from_parquet(
    pq: MeadParquet, crop_len: int,
) -> list[dict]:
    """Stream a single parquet → list of clip dicts (actions + meta).

    Filters out clips shorter than `crop_len`. Each yielded dict
    carries `(actions, actor, emotion, intensity, source_id)`.
    """

    paths = pq.all_paths()
    clips_by_idx: dict[int, dict] = {}
    for row_idx, arr in pq.iter_actions():
        if arr.shape[0] < crop_len or arr.shape[1] != ACTION_DIM:
            continue
        clips_by_idx[row_idx] = {"row_idx": row_idx, "actions": arr}

    for i, p in enumerate(paths):
        if i not in clips_by_idx:
            continue
        m = parse_path(p, actor_override=pq.actor_override) or {}
        clips_by_idx[i].update({
            "path": p,
            "actor": m.get("actor", ""),
            "emotion": m.get("emotion", ""),
            "intensity": m.get("intensity", 0),
            "source_id": pq.source_id,
        })
    return list(clips_by_idx.values())


def load_blendshape_clips(
    *,
    data_dir: Path | None,
    parquet_path: Path | None,
    crop_len: int,
) -> list[dict]:
    """Load clips from a single parquet OR a multi-source data dir.

    In multi-source mode, deduplicates clips across overlapping
    sources by `(actor, emotion, intensity, utt)`; the first
    occurrence wins. Total RAM footprint scales linearly with frame
    count (~5 MB per 1k clips at K=54, T~120, fp32).
    """

    if data_dir is not None:
        triples = discover_parquets(data_dir)
        if not triples:
            raise RuntimeError(f"no `mead_3d*` parquets discovered under {data_dir}")
        all_clips: list[dict] = []
        seen: set[tuple] = set()
        n_dup = 0
        for path, actor, sid in triples:
            pq = MeadParquet(path, actor_override=actor, source_id=sid)
            clips = load_blendshape_clips_from_parquet(pq, crop_len)
            kept = 0
            for c in clips:
                key = (c["actor"], c["emotion"], c["intensity"],
                       c.get("path", "").split("/")[-1])
                if key in seen:
                    n_dup += 1
                    continue
                seen.add(key)
                all_clips.append(c)
                kept += 1
            print(f"  {pq.source_id:30s}  kept={kept:5d}  dup_skipped={len(clips)-kept:4d}",
                  file=sys.stderr)
        print(f"  total {len(all_clips)} unique clips, {n_dup} duplicates skipped",
              file=sys.stderr)
        return all_clips

    pq = MeadParquet(parquet_path)
    clips = load_blendshape_clips_from_parquet(pq, crop_len)
    # Backwards-compat: stamp source_id="mead_3d" if missing.
    for c in clips:
        c.setdefault("source_id", pq.source_id)
    return clips


def split_clips(
    clips: list[dict],
    *,
    val_actors: set[str] | None = None,
    test_actors: set[str] | None = None,
) -> dict[str, list[dict]]:
    """Speaker-disjoint split.

    - Multi-actor mode (val_actors / test_actors specified): train is
      the complement of (val ∪ test). Designed for the 26-actor
      MEAD_3D union.
    - Single-actor legacy mode (no val/test_actors): train=W009,
      val/test=video28 split. Backward compat with the 2-speaker
      pre-RB7 baseline.
    """
    if val_actors or test_actors:
        val_set = val_actors or set()
        test_set = test_actors or set()
        train = [c for c in clips if c["actor"] not in val_set | test_set]
        val = sorted(
            (c for c in clips if c["actor"] in val_set),
            key=lambda c: (c["actor"], c.get("source_id", ""), c.get("path", "")),
        )
        test = sorted(
            (c for c in clips if c["actor"] in test_set),
            key=lambda c: (c["actor"], c.get("source_id", ""), c.get("path", "")),
        )
        return {"train": train, "val": val, "test": test}

    # Legacy 2-speaker fallback.
    train = [c for c in clips if c["actor"] == "W009"]
    other = sorted(
        (c for c in clips if c["actor"] == "video28"),
        key=lambda c: c.get("path", ""),
    )
    half = len(other) // 2
    return {"train": train, "val": other[:half], "test": other[half:]}


class CropDataset(Dataset):
    """Yield random `(crop_len, K)` windows from a list of clips.

    Length is `len(clips) * crops_per_clip` so each epoch sees
    `crops_per_clip` distinct crops per clip on average.
    """

    def __init__(self, clips: list[dict], crop_len: int, crops_per_clip: int) -> None:
        self.clips = clips
        self.crop_len = crop_len
        self.crops_per_clip = crops_per_clip
        self._rng = np.random.default_rng(0)

    def __len__(self) -> int:
        return len(self.clips) * self.crops_per_clip

    def __getitem__(self, idx: int) -> torch.Tensor:
        clip = self.clips[idx % len(self.clips)]
        T = clip["actions"].shape[0]
        if T == self.crop_len:
            start = 0
        else:
            start = int(self._rng.integers(0, T - self.crop_len + 1))
        crop = clip["actions"][start : start + self.crop_len]
        return torch.from_numpy(crop)


# ------------------------------------------------------------------
# Train / eval loops
# ------------------------------------------------------------------

def train_one_epoch(model, loader, optim, device, kl_weight,
                    channel_weights=None) -> dict:
    model.train()
    losses = {"total": 0.0, "recon_mse": 0.0, "kl": 0.0}
    n = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out = model(batch)
        ld = model.vae_loss(batch, out, channel_weights=channel_weights)
        optim.zero_grad(set_to_none=True)
        ld["loss"].backward()
        optim.step()
        bs = batch.shape[0]
        losses["total"] += ld["loss"].item() * bs
        losses["recon_mse"] += ld["recon_mse"].item() * bs
        losses["kl"] += ld["kl"].item() * bs
        n += bs
    return {k: v / max(n, 1) for k, v in losses.items()}


@torch.no_grad()
def eval_recon(model, loader, device, channel_weights=None) -> dict:
    model.eval()
    losses = {"total": 0.0, "recon_mse": 0.0, "kl": 0.0}
    n = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out = model(batch)
        ld = model.vae_loss(batch, out, channel_weights=channel_weights)
        bs = batch.shape[0]
        losses["total"] += ld["loss"].item() * bs
        losses["recon_mse"] += ld["recon_mse"].item() * bs
        losses["kl"] += ld["kl"].item() * bs
        n += bs
    return {k: v / max(n, 1) for k, v in losses.items()}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    print(f"device: {device}", file=sys.stderr)

    if args.parquet and args.data_dir:
        print("error: --parquet and --data-dir are mutually exclusive",
              file=sys.stderr)
        return 2
    if not args.parquet and not args.data_dir:
        # Default: legacy single-parquet mode.
        args.parquet = Path("data/mead_3d/train-00000-of-00001.parquet")

    src = args.data_dir if args.data_dir else args.parquet
    print(f"loading clips from {src}...", file=sys.stderr)
    t0 = time.time()
    clips = load_blendshape_clips(
        data_dir=args.data_dir,
        parquet_path=args.parquet,
        crop_len=args.crop_len,
    )
    print(
        f"  {len(clips)} clips kept (≥{args.crop_len} frames)  "
        f"({time.time() - t0:.1f}s)",
        file=sys.stderr,
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

    train_ds = CropDataset(splits["train"], args.crop_len, args.crops_per_clip)
    val_ds = CropDataset(splits["val"], args.crop_len, max(args.crops_per_clip // 2, 1))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    cfg = VAEConfig(k_dim=ACTION_DIM, d_lat=args.d_lat, hidden=args.hidden,
                    kl_weight=args.kl_weight)
    model = BlendshapeVAE(cfg).to(device)
    print(f"params: {count_params(model):,}", file=sys.stderr)

    # Per-channel reconstruction weights. Two safeguards on top of the
    # inverse-stdev formula:
    #
    #   1. **Dead-channel exclusion**: channels with stdev < DEAD_FLOOR
    #      (≈always-zero in training data) get uniform weight 1.0
    #      regardless of `channel_weight_power`. Boosting always-zero
    #      channels by 12× (as raw inverse-stdev would for stdev=1e-4)
    #      pulls capacity off perceptually meaningful channels — the
    #      smoke run on this dataset showed 4 always-zero channels each
    #      getting weight ≈12, consuming ~half the total loss attention.
    #   2. **Dynamic-range clip**: after normalization to mean-1, clip
    #      weights to [1/CLIP, CLIP] then re-normalize. Bounds the
    #      max-vs-min weight ratio, preventing one rare-but-active
    #      channel from dominating.
    DEAD_FLOOR = 0.005
    CLIP = 4.0
    channel_weights_t = None
    channel_weights_np = None
    channel_stds_np = None
    if args.channel_weight_power > 0.0:
        sample = splits["train"][:200]
        all_frames = np.concatenate([c["actions"] for c in sample], axis=0)
        channel_stds_np = all_frames.std(axis=0).astype(np.float64)
        is_dead = channel_stds_np < DEAD_FLOOR
        # Floor stdev for weight computation only (so the formula doesn't
        # divide by zero); dead channels are then forced to weight 1.
        std_for_w = np.maximum(channel_stds_np, 1e-3)
        raw_w = (1.0 / std_for_w) ** float(args.channel_weight_power)
        raw_w[is_dead] = 1.0
        w = raw_w / raw_w.mean()
        w = np.clip(w, 1.0 / CLIP, CLIP)
        w = w / w.mean()
        channel_weights_np = w
        channel_weights_t = torch.from_numpy(
            channel_weights_np.astype(np.float32)
        ).to(device)
        # Show what got boosted vs attenuated for traceability.
        order = np.argsort(channel_weights_np)
        boosted = order[-5:][::-1]
        attenuated = order[:5]
        n_dead = int(is_dead.sum())
        print(f"channel weighting: power={args.channel_weight_power} "
              f"clip={CLIP}  dead_floor={DEAD_FLOOR}  "
              f"({n_dead}/{len(channel_weights_np)} dead channels uniform-weighted)",
              file=sys.stderr)
        print(f"  weights∈[{channel_weights_np.min():.2f}, "
              f"{channel_weights_np.max():.2f}]  mean={channel_weights_np.mean():.2f}",
              file=sys.stderr)
        print(f"  most boosted   chs: "
              f"{[(int(k), round(float(channel_weights_np[k]), 2), round(float(channel_stds_np[k]), 4)) for k in boosted]}",
              file=sys.stderr)
        print(f"  most attenuated chs: "
              f"{[(int(k), round(float(channel_weights_np[k]), 2), round(float(channel_stds_np[k]), 4)) for k in attenuated]}",
              file=sys.stderr)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr * 1e-2,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    loss_csv = args.output.with_suffix(".csv")
    csv_fh = loss_csv.open("w", newline="")
    csv_w = csv.writer(csv_fh)
    csv_w.writerow(["epoch", "lr", "train_total", "train_recon_mse", "train_kl",
                    "val_total", "val_recon_mse", "val_kl"])

    print(f"\ntraining {args.epochs} epochs...", file=sys.stderr)
    t_train = time.time()
    best_val_recon = float("inf")
    for ep in range(args.epochs):
        ep_t0 = time.time()
        tr = train_one_epoch(model, train_loader, optim, device, args.kl_weight,
                             channel_weights=channel_weights_t)
        va = eval_recon(model, val_loader, device,
                        channel_weights=channel_weights_t)
        sched.step()
        cur_lr = sched.get_last_lr()[0]
        ep_dt = time.time() - ep_t0
        print(
            f"  ep{ep + 1:3d}/{args.epochs}  "
            f"train_recon={tr['recon_mse']:.5f}  "
            f"val_recon={va['recon_mse']:.5f}  "
            f"kl={tr['kl']:.4f}  "
            f"lr={cur_lr:.2e}  ({ep_dt:.1f}s)",
            file=sys.stderr,
        )
        csv_w.writerow([
            ep + 1, f"{cur_lr:.4e}",
            f"{tr['total']:.6f}", f"{tr['recon_mse']:.6f}", f"{tr['kl']:.6f}",
            f"{va['total']:.6f}", f"{va['recon_mse']:.6f}", f"{va['kl']:.6f}",
        ])
        csv_fh.flush()
        if va["recon_mse"] < best_val_recon:
            best_val_recon = va["recon_mse"]

    csv_fh.close()
    print(
        f"\ntraining done; total {time.time() - t_train:.0f}s  "
        f"best val_recon={best_val_recon:.5f}",
        file=sys.stderr,
    )

    print(f"saving model -> {args.output}", file=sys.stderr)
    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "best_val_recon_mse": float(best_val_recon),
        "epochs": args.epochs,
        "n_train_clips": len(splits["train"]),
        "n_val_clips": len(splits["val"]),
        "n_test_clips": len(splits["test"]),
        "channel_weight_power": float(args.channel_weight_power),
        "channel_weights": (channel_weights_np.astype(np.float32)
                            if channel_weights_np is not None else None),
        "channel_stds": (channel_stds_np.astype(np.float32)
                         if channel_stds_np is not None else None),
    }, args.output)

    # Final test eval + per-channel breakdown.
    test_ds = CropDataset(splits["test"], args.crop_len, 4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    test_metrics = eval_recon(model, test_loader, device,
                              channel_weights=channel_weights_t)
    per_ch = _per_channel_recon(model, splits["test"], args.crop_len, device)
    _write_recon_report(
        path=args.output.with_suffix(".report.md"),
        cfg=cfg,
        n_params=count_params(model),
        n_clips=len(clips),
        splits=splits,
        train_metrics=tr, val_metrics=va, test_metrics=test_metrics,
        per_ch_mse=per_ch,
        epochs=args.epochs, device=str(device),
    )
    print(
        f"test_recon={test_metrics['recon_mse']:.5f}  "
        f"per-channel mean MSE={per_ch.mean():.5f}",
        file=sys.stderr,
    )
    return 0


@torch.no_grad()
def _per_channel_recon(model, clips, crop_len, device) -> np.ndarray:
    """Compute per-channel MSE on full-length clips (no cropping)."""

    model.eval()
    sums = np.zeros(ACTION_DIM, dtype=np.float64)
    counts = np.zeros(ACTION_DIM, dtype=np.float64)
    for c in clips[:50]:  # sample 50 clips for speed
        x = torch.from_numpy(c["actions"]).unsqueeze(0).to(device)
        T = x.shape[1]
        T_crop = (T // 4) * 4
        if T_crop < crop_len:
            continue
        x = x[:, :T_crop]
        out = model(x)
        recon = out["recon"]
        diff = (recon - x).pow(2).mean(dim=(0, 1)).cpu().numpy()  # (K,)
        sums += diff * T_crop
        counts += T_crop
    return sums / np.where(counts > 0, counts, 1)


def _write_recon_report(*, path, cfg, n_params, n_clips, splits,
                        train_metrics, val_metrics, test_metrics,
                        per_ch_mse, epochs, device) -> None:
    lines: list[str] = []
    lines.append("# Stage-1 VAE training report\n\n")
    lines.append(f"- Device: `{device}`\n")
    lines.append(f"- Epochs: {epochs}\n")
    lines.append(f"- Params: {n_params:,}\n")
    lines.append(f"- Latent dim: {cfg.d_lat}  hidden: {cfg.hidden}\n")
    lines.append(f"- KL weight: {cfg.kl_weight}\n\n")
    lines.append("## Splits\n\n")
    for k in ("train", "val", "test"):
        lines.append(f"- {k}: {len(splits[k])} clips\n")
    lines.append("\n## Final-epoch metrics\n\n")
    lines.append("| split | recon_mse | kl |\n|---|---:|---:|\n")
    for name, m in (("train", train_metrics), ("val", val_metrics),
                    ("test", test_metrics)):
        lines.append(f"| {name} | {m['recon_mse']:.5f} | {m['kl']:.4f} |\n")
    lines.append("\n## Per-channel MSE (test, sampled 50 clips)\n\n")
    lines.append(f"- mean: {per_ch_mse.mean():.5f}  "
                 f"min: {per_ch_mse.min():.5f}  "
                 f"max: {per_ch_mse.max():.5f}\n")
    worst = np.argsort(per_ch_mse)[-5:][::-1]
    lines.append("- worst 5 channels (by MSE):\n")
    for k in worst:
        lines.append(f"  - k={int(k):2d}  mse={per_ch_mse[k]:.5f}\n")
    target = 0.005
    lines.append(f"\n## Pass criterion (PROJECT_PLAN §8 row B2)\n\n")
    lines.append(f"- recon MSE < {target} target.  "
                 f"Achieved (test): **{test_metrics['recon_mse']:.5f}** — "
                 f"{'✅ PASS' if test_metrics['recon_mse'] < target else '❌ FAIL'}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


if __name__ == "__main__":
    sys.exit(main())
