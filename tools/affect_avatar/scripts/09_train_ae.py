#!/usr/bin/env python3
"""Train Stage-1 deterministic AE on MEAD_3D — Exp 1 per STAGE1_VAE_PLAN §4.

Recipe:
  - Same Conv1d encoder/decoder as the VAE (`lib/vae.py:BlendshapeVAE`).
  - **Deterministic** (`z = mu` always; no reparam noise).
  - **Linear output** (no final sigmoid; peaks aren't compressed).
  - **`kl_weight = 0`** (no KL regularizer; encoder is free to use the
    full latent dimension).
  - **Loss**: `α_value · grouped_MSE + α_velocity · velocity_MSE + α_peak · peak_MSE`.
    See [`lib/vae.py:BlendshapeVAE.ae_loss`](lib/vae.py).

Pass condition (per STAGE1_VAE_PLAN.md §4 Exp 1):
  All 8 round-trip gates pass on the dashboard. Partial pass (≥5/8)
  triggers Exp 1b (loss-weight retune). Fail (<5/8) branches to Exp 4
  (VQ-VAE).

Outputs:
    artifacts/blendshape_ae_v3.pt        — final state dict
    artifacts/blendshape_ae_v3.csv       — per-epoch loss components
    artifacts/blendshape_ae_v3.report.md — final stats summary

Usage:
  ./09_train_ae.py \\
      --data-dir data \\
      --epochs 50 \\
      --batch-size 64 \\
      --crops-per-clip 8 \\
      --output artifacts/blendshape_ae_v3.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the VAE-trainer's data loading + crop dataset — those are
# blendshape-only and don't depend on the loss formulation.
from lib.mead_3d_loader import ACTION_DIM  # noqa: E402
from lib.vae import BlendshapeVAE, VAEConfig, count_params  # noqa: E402

# scripts/02_train_vae.py is sibling — direct relative import via the
# numbered filename isn't possible, so we re-export the helpers we need.
import importlib.util as _importutil

_train_vae_spec = _importutil.spec_from_file_location(
    "_train_vae_helpers",
    str(Path(__file__).resolve().parent / "02_train_vae.py"),
)
_train_vae = _importutil.module_from_spec(_train_vae_spec)  # type: ignore[arg-type]
_train_vae_spec.loader.exec_module(_train_vae)  # type: ignore[union-attr]

CropDataset = _train_vae.CropDataset
load_blendshape_clips = _train_vae.load_blendshape_clips
split_clips = _train_vae.split_clips
pick_device = _train_vae.pick_device


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
        help="Directory of `mead_3d*` subdirectories.",
    )
    p.add_argument(
        "--val-actors", type=str, default="video28",
        help="Comma-separated actor IDs held out for validation.",
    )
    p.add_argument(
        "--test-actors", type=str, default="M003",
        help="Comma-separated actor IDs held out for test.",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--crops-per-clip", type=int, default=8)
    p.add_argument("--crop-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--alpha-value", type=float, default=1.0,
                   help="Weight on grouped MSE.")
    p.add_argument("--alpha-velocity", type=float, default=0.5,
                   help="Weight on velocity (adjacent-frame Δ) MSE.")
    p.add_argument("--alpha-peak", type=float, default=0.5,
                   help="Weight on per-channel max-over-time MSE.")
    p.add_argument("--d-lat", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--device", default=None,
                   help="cpu / mps / cuda. Auto-pick if unset.")
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/blendshape_ae_v3.pt"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ------------------------------------------------------------------
# Train / eval loops — distinct from 02_train_vae.py because the
# loss returns different keys (grouped_mse / velocity_mse / peak_mse
# instead of recon_mse / kl).
# ------------------------------------------------------------------

def train_one_epoch(model, loader, optim, device, *,
                    alpha_value, alpha_velocity, alpha_peak) -> dict:
    model.train()
    losses = {"total": 0.0, "grouped_mse": 0.0,
              "velocity_mse": 0.0, "peak_mse": 0.0, "recon_mse": 0.0}
    n = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out = model(batch)
        ld = model.ae_loss(
            batch, out,
            alpha_value=alpha_value,
            alpha_velocity=alpha_velocity,
            alpha_peak=alpha_peak,
        )
        optim.zero_grad(set_to_none=True)
        ld["loss"].backward()
        optim.step()
        bs = batch.shape[0]
        for k in losses:
            v = ld["loss"] if k == "total" else ld[k]
            losses[k] += float(v.item()) * bs
        n += bs
    return {k: v / max(n, 1) for k, v in losses.items()}


@torch.no_grad()
def eval_recon(model, loader, device, *,
               alpha_value, alpha_velocity, alpha_peak) -> dict:
    model.eval()
    losses = {"total": 0.0, "grouped_mse": 0.0,
              "velocity_mse": 0.0, "peak_mse": 0.0, "recon_mse": 0.0}
    n = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out = model(batch)
        ld = model.ae_loss(
            batch, out,
            alpha_value=alpha_value,
            alpha_velocity=alpha_velocity,
            alpha_peak=alpha_peak,
        )
        bs = batch.shape[0]
        for k in losses:
            v = ld["loss"] if k == "total" else ld[k]
            losses[k] += float(v.item()) * bs
        n += bs
    return {k: v / max(n, 1) for k, v in losses.items()}


@torch.no_grad()
def per_channel_recon(model, clips, crop_len, device) -> np.ndarray:
    model.eval()
    sums = np.zeros(ACTION_DIM, dtype=np.float64)
    counts = np.zeros(ACTION_DIM, dtype=np.float64)
    for c in clips[:50]:
        x = torch.from_numpy(c["actions"]).unsqueeze(0).to(device)
        T = x.shape[1]
        T_crop = (T // 4) * 4
        if T_crop < crop_len:
            continue
        x = x[:, :T_crop]
        out = model(x)
        recon = out["recon"]
        diff = (recon - x).pow(2).mean(dim=(0, 1)).cpu().numpy()
        sums += diff * T_crop
        counts += T_crop
    return sums / np.where(counts > 0, counts, 1)


def write_report(*, path, cfg, n_params, splits,
                 train_metrics, val_metrics, test_metrics,
                 per_ch_mse, epochs, device,
                 alpha_value, alpha_velocity, alpha_peak,
                 best_val_total) -> None:
    lines: list[str] = []
    lines.append("# Stage-1 deterministic AE training report (Exp 1)\n\n")
    lines.append(f"- Device: `{device}`\n")
    lines.append(f"- Epochs: {epochs}\n")
    lines.append(f"- Params: {n_params:,}\n")
    lines.append(f"- Latent dim: {cfg.d_lat}  hidden: {cfg.hidden}\n")
    lines.append(f"- KL weight: {cfg.kl_weight} (deterministic)\n")
    lines.append(f"- Output activation: `{cfg.output_activation}`\n")
    lines.append(f"- Loss weights: α_value={alpha_value}  "
                 f"α_velocity={alpha_velocity}  α_peak={alpha_peak}\n")
    lines.append(f"- Best val total loss: {best_val_total:.5f}\n\n")
    lines.append("## Splits\n\n")
    for k in ("train", "val", "test"):
        lines.append(f"- {k}: {len(splits[k])} clips\n")
    lines.append("\n## Final-epoch metrics\n\n")
    lines.append("| split | total | grouped_mse | velocity_mse | peak_mse | recon_mse |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for name, m in (("train", train_metrics), ("val", val_metrics),
                    ("test", test_metrics)):
        lines.append(
            f"| {name} | {m['total']:.5f} | {m['grouped_mse']:.5f} | "
            f"{m['velocity_mse']:.5f} | {m['peak_mse']:.5f} | "
            f"{m['recon_mse']:.5f} |\n"
        )
    lines.append("\n## Per-channel MSE (test, 50-clip sample)\n\n")
    lines.append(f"- mean: {per_ch_mse.mean():.5f}  "
                 f"min: {per_ch_mse.min():.5f}  "
                 f"max: {per_ch_mse.max():.5f}\n")
    worst = np.argsort(per_ch_mse)[-5:][::-1]
    lines.append("- worst 5 channels (by MSE):\n")
    for k in worst:
        lines.append(f"  - k={int(k):2d}  mse={per_ch_mse[k]:.5f}\n")
    lines.append("\n## Next step\n\n")
    lines.append("- Run `08_diagnose.py --vae <this ckpt>` to score the "
                 "8 round-trip gates from STAGE1_VAE_PLAN §4.\n")
    lines.append("- 8/8 → admit to Exp 5 (DiT retrain).\n")
    lines.append("- 5–7/8 → branch to Exp 1b (loss-weight retune).\n")
    lines.append("- <5/8 → branch to Exp 4 (VQ-VAE).\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


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
        args.parquet = Path("data/mead_3d/train-00000-of-00001.parquet")

    src = args.data_dir if args.data_dir else args.parquet
    print(f"loading clips from {src}...", file=sys.stderr)
    t0 = time.time()
    clips = load_blendshape_clips(
        data_dir=args.data_dir,
        parquet_path=args.parquet,
        crop_len=args.crop_len,
    )
    print(f"  {len(clips)} clips kept (≥{args.crop_len} frames)  "
          f"({time.time() - t0:.1f}s)", file=sys.stderr)

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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, drop_last=False)

    cfg = VAEConfig(
        k_dim=ACTION_DIM, d_lat=args.d_lat, hidden=args.hidden,
        kl_weight=0.0,
        deterministic=True,
        output_activation="linear",
    )
    model = BlendshapeVAE(cfg).to(device)
    print(f"params: {count_params(model):,}", file=sys.stderr)
    print(f"AE mode: deterministic={cfg.deterministic} "
          f"output_activation={cfg.output_activation} "
          f"kl_weight={cfg.kl_weight}", file=sys.stderr)
    print(f"loss: α_value={args.alpha_value}  "
          f"α_velocity={args.alpha_velocity}  "
          f"α_peak={args.alpha_peak}", file=sys.stderr)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr * 1e-2,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    loss_csv = args.output.with_suffix(".csv")
    csv_fh = loss_csv.open("w", newline="")
    csv_w = csv.writer(csv_fh)
    csv_w.writerow([
        "epoch", "lr",
        "train_total", "train_grouped", "train_velocity", "train_peak", "train_recon",
        "val_total",   "val_grouped",   "val_velocity",   "val_peak",   "val_recon",
    ])

    print(f"\ntraining {args.epochs} epochs...", file=sys.stderr)
    t_train = time.time()
    best_val_total = float("inf")
    for ep in range(args.epochs):
        ep_t0 = time.time()
        tr = train_one_epoch(
            model, train_loader, optim, device,
            alpha_value=args.alpha_value,
            alpha_velocity=args.alpha_velocity,
            alpha_peak=args.alpha_peak,
        )
        va = eval_recon(
            model, val_loader, device,
            alpha_value=args.alpha_value,
            alpha_velocity=args.alpha_velocity,
            alpha_peak=args.alpha_peak,
        )
        sched.step()
        cur_lr = sched.get_last_lr()[0]
        ep_dt = time.time() - ep_t0
        print(
            f"  ep{ep + 1:3d}/{args.epochs}  "
            f"train_total={tr['total']:.5f}  val_total={va['total']:.5f}  "
            f"val_recon={va['recon_mse']:.5f}  "
            f"val_peak={va['peak_mse']:.5f}  "
            f"lr={cur_lr:.2e}  ({ep_dt:.1f}s)",
            file=sys.stderr,
        )
        csv_w.writerow([
            ep + 1, f"{cur_lr:.4e}",
            f"{tr['total']:.6f}", f"{tr['grouped_mse']:.6f}",
            f"{tr['velocity_mse']:.6f}", f"{tr['peak_mse']:.6f}",
            f"{tr['recon_mse']:.6f}",
            f"{va['total']:.6f}", f"{va['grouped_mse']:.6f}",
            f"{va['velocity_mse']:.6f}", f"{va['peak_mse']:.6f}",
            f"{va['recon_mse']:.6f}",
        ])
        csv_fh.flush()
        if va["total"] < best_val_total:
            best_val_total = va["total"]

    csv_fh.close()
    print(
        f"\ntraining done; total {time.time() - t_train:.0f}s  "
        f"best val_total={best_val_total:.5f}",
        file=sys.stderr,
    )

    print(f"saving model -> {args.output}", file=sys.stderr)
    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "best_val_total": float(best_val_total),
        "epochs": args.epochs,
        "n_train_clips": len(splits["train"]),
        "n_val_clips": len(splits["val"]),
        "n_test_clips": len(splits["test"]),
        "alpha_value": float(args.alpha_value),
        "alpha_velocity": float(args.alpha_velocity),
        "alpha_peak": float(args.alpha_peak),
        # Sentinel so 08_diagnose / 06_validate can identify AE ckpts.
        "stage1_kind": "deterministic_ae_v3",
    }, args.output)

    test_ds = CropDataset(splits["test"], args.crop_len, 4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    test_metrics = eval_recon(
        model, test_loader, device,
        alpha_value=args.alpha_value,
        alpha_velocity=args.alpha_velocity,
        alpha_peak=args.alpha_peak,
    )
    per_ch = per_channel_recon(model, splits["test"], args.crop_len, device)
    write_report(
        path=args.output.with_suffix(".report.md"),
        cfg=cfg, n_params=count_params(model), splits=splits,
        train_metrics=tr, val_metrics=va, test_metrics=test_metrics,
        per_ch_mse=per_ch,
        epochs=args.epochs, device=str(device),
        alpha_value=args.alpha_value,
        alpha_velocity=args.alpha_velocity,
        alpha_peak=args.alpha_peak,
        best_val_total=best_val_total,
    )
    print(f"test_total={test_metrics['total']:.5f}  "
          f"test_recon_mse={test_metrics['recon_mse']:.5f}  "
          f"per-channel mean MSE={per_ch.mean():.5f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
