#!/usr/bin/env python3
"""Inference-time classifier-free-guidance sweep.

Reuses the §6.2 controllability protocol from `06_validate.py` but
loops over cfg_scale values at inference. Loads VAE + DiT + face-VAD
probe once; samples are produced per (clip, cfg_scale, axis, level)
combination. Reports a `cfg_scale → gap_pos / gap_neg / cross-axis`
table.

CFG amplifies the difference between conditioned and unconditioned ε
predictions:

    ε = (1 - w) · ε_uncond + w · ε_cond

Without CFG-drop training the unconditioned path is just "V/A/D = 0",
so this is a partial test. If a model trained with cfg_drop_prob=0.0
already responds to CFG > 1, great; if not, we know we need to retrain
with CFG-drop.

Usage:
  ./07_cfg_sweep.py \\
      --dit artifacts/blendshape_dit.pt \\
      --vae artifacts/blendshape_vae.pt \\
      --whisper-cache artifacts/whisper_cache \\
      --data-dir data --n-clips 12 \\
      --cfg-scales 1.0 1.5 3.0 5.0 \\
      --output artifacts/cfg_sweep.report.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the heavy lifting from 06_validate.py.
import importlib.util  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "validate06", str(Path(__file__).resolve().parent / "06_validate.py"),
)
_M = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_M)

DitSampler = _M.DitSampler
FaceVADProbe = _M.FaceVADProbe
load_full_clips = _M.load_full_clips
split_clips = _M.split_clips
train_probe = _M.train_probe
select_test_clips = _M.select_test_clips
prep_whisper = _M.prep_whisper
pool_action = _M.pool_action

from lib.dit import DiT1D, DiTConfig  # noqa: E402
from lib.vae import BlendshapeVAE, VAEConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dit", type=Path, default=Path("artifacts/blendshape_dit.pt"))
    p.add_argument("--vae", type=Path, default=Path("artifacts/blendshape_vae.pt"))
    p.add_argument("--whisper-cache", type=Path,
                   default=Path("artifacts/whisper_cache"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--val-actors", type=str, default="video28,video26")
    p.add_argument("--test-actors", type=str, default="M003,W040")
    p.add_argument("--n-clips", type=int, default=12)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument(
        "--cfg-scales", type=float, nargs="+",
        default=[1.0, 1.5, 3.0, 5.0],
        help="cfg_scale values to sweep. 1.0 = no guidance.",
    )
    p.add_argument("--probe-epochs", type=int, default=15)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--probe-hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/cfg_sweep.report.md"))
    p.add_argument("--summary-json", type=Path,
                   default=Path("artifacts/cfg_sweep.summary.json"))
    return p.parse_args()


def pick_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def eval_controllability_at_scale(
    *, sampler: DitSampler, probe: FaceVADProbe,
    clips: list[dict], device: torch.device,
) -> dict:
    """Same as 06_validate.eval_controllability but uses the sampler's
    current cfg_scale (set by caller)."""
    AXES = ("V", "A", "D")
    preds: dict[tuple[int, int], list[np.ndarray]] = {
        (ax, lvl): [] for ax in range(3) for lvl in (-1, 0, 1)
    }

    d_vad = sampler.cfg.d_vad
    for c in clips:
        actions = c["actions"]
        n_frames = (actions.shape[0] // 4) * 4
        if n_frames < 16:
            continue
        t_lat = n_frames // 4
        whisper = prep_whisper(c["whisper"], t_lat * 8, device)
        for ax in range(3):
            for level in (-1, 0, 1):
                vad = np.zeros(d_vad, dtype=np.float32)
                vad[ax] = float(level)
                if d_vad == 4:
                    vad[3] = 1.0   # peak intensity
                vad_t = torch.from_numpy(vad).unsqueeze(0).to(device)
                sample = sampler.sample(whisper, vad_t, t_lat)
                pooled = torch.from_numpy(
                    pool_action(sample[:n_frames]).astype(np.float32)
                ).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = probe(pooled).cpu().numpy()[0]
                preds[(ax, level)].append(pred)

    means = {
        k: np.stack(v).mean(axis=0) if v else np.zeros(3)
        for k, v in preds.items()
    }
    gap_pos = np.zeros((3, 3))
    gap_neg = np.zeros((3, 3))
    for ax in range(3):
        gap_pos[ax] = means[(ax, 1)] - means[(ax, 0)]
        gap_neg[ax] = means[(ax, -1)] - means[(ax, 0)]
    pos_diag = [gap_pos[ax, ax] for ax in range(3)]
    neg_diag = [gap_neg[ax, ax] for ax in range(3)]
    cross = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                cross[i, j] = float("nan")
            else:
                cross[i, j] = 0.5 * (abs(gap_pos[i, j]) + abs(gap_neg[i, j]))
    return {
        "axes": AXES,
        "gap_pos_diag": pos_diag,
        "gap_neg_diag": neg_diag,
        "gap_pos_full": gap_pos.tolist(),
        "gap_neg_full": gap_neg.tolist(),
        "cross_axis": cross.tolist(),
        "n": len(preds[(0, 0)]),
    }


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}", file=sys.stderr)

    # --- Load VAE + DiT once ---
    print(f"loading VAE: {args.vae}", file=sys.stderr)
    vae_ckpt = torch.load(args.vae, map_location=device, weights_only=False)
    vae_cfg = VAEConfig(**vae_ckpt["config"])
    vae = BlendshapeVAE(vae_cfg).to(device).eval()
    vae.load_state_dict(vae_ckpt["state_dict"])

    print(f"loading DiT: {args.dit}", file=sys.stderr)
    dit_ckpt = torch.load(args.dit, map_location=device, weights_only=False)
    dit_cfg = DiTConfig(**dit_ckpt["config"])
    dit = DiT1D(dit_cfg).to(device).eval()
    dit.load_state_dict(dit_ckpt["state_dict"])
    if dit_ckpt.get("zscore_latents", False):
        latent_mu = torch.from_numpy(dit_ckpt["latent_mu"]).to(device)
        latent_std = torch.from_numpy(dit_ckpt["latent_std"]).to(device)
    else:
        latent_mu = torch.zeros(vae_cfg.d_lat, device=device)
        latent_std = torch.ones(vae_cfg.d_lat, device=device)
    print(f"  best_val_eps={dit_ckpt.get('best_val_eps', float('nan')):.5f}",
          file=sys.stderr)

    # --- Load clips, train probe (once for all scales) ---
    print(f"loading clips...", file=sys.stderr)
    t0 = time.time()
    clips = load_full_clips(args.data_dir, args.whisper_cache, set())
    print(f"  {len(clips)} clips ({time.time() - t0:.1f}s)", file=sys.stderr)

    val_actors = {a.strip() for a in args.val_actors.split(",") if a.strip()}
    test_actors = {a.strip() for a in args.test_actors.split(",") if a.strip()}
    splits = split_clips(clips, val_actors, test_actors)
    print(f"  train={len(splits['train'])}, val={len(splits['val'])}, "
          f"test={len(splits['test'])}", file=sys.stderr)

    print(f"\ntraining face→VAD probe ({args.probe_epochs} epochs)...",
          file=sys.stderr)
    probe, probe_metrics = train_probe(
        splits["train"], splits["val"],
        epochs=args.probe_epochs, lr=args.probe_lr, hidden=args.probe_hidden,
        device=device, seed=args.seed,
    )

    eval_clips = select_test_clips(splits["test"], args.n_clips, args.seed)
    print(f"\nselected {len(eval_clips)} test clips", file=sys.stderr)

    # --- Sweep ---
    results: list[tuple[float, dict]] = []
    for scale in args.cfg_scales:
        print(f"\n=== cfg_scale={scale} ===", file=sys.stderr)
        sampler = DitSampler(
            dit=dit, dit_cfg=dit_cfg, vae=vae,
            latent_mu=latent_mu, latent_std=latent_std,
            n_steps=args.n_steps, cfg_scale=scale, device=device,
        )
        t0 = time.time()
        ctrl = eval_controllability_at_scale(
            sampler=sampler, probe=probe, clips=eval_clips, device=device,
        )
        dt = time.time() - t0
        print(
            f"  done ({dt:.0f}s)  gap_pos diag="
            f"{[f'{v:+.3f}' for v in ctrl['gap_pos_diag']]}  "
            f"gap_neg diag={[f'{v:+.3f}' for v in ctrl['gap_neg_diag']]}",
            file=sys.stderr,
        )
        results.append((scale, ctrl))

    # --- Report ---
    lines: list[str] = []
    lines.append(f"# CFG sweep — controllability vs cfg_scale\n\n")
    lines.append(f"- DiT: `{args.dit}` (best_val_eps="
                 f"{dit_ckpt.get('best_val_eps', 0):.5f})\n")
    lines.append(f"- VAE: `{args.vae}`\n")
    lines.append(f"- n test clips: {len(eval_clips)}\n")
    lines.append(f"- n DDIM steps: {args.n_steps}\n\n")

    lines.append("## §6.2 Controllability vs cfg_scale\n\n")
    lines.append("Higher gap_pos / more negative gap_neg = stronger response "
                 "to V/A/D conditioning. Pass thresholds: gap_pos ≥ +0.30, "
                 "gap_neg ≤ -0.30.\n\n")
    lines.append(
        "| cfg_scale | gap_pos V | gap_pos A | gap_pos D | "
        "gap_neg V | gap_neg A | gap_neg D |\n"
        "|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for scale, ctrl in results:
        gp = ctrl["gap_pos_diag"]
        gn = ctrl["gap_neg_diag"]
        lines.append(
            f"| {scale} | {gp[0]:+.3f} | {gp[1]:+.3f} | {gp[2]:+.3f} | "
            f"{gn[0]:+.3f} | {gn[1]:+.3f} | {gn[2]:+.3f} |\n"
        )

    lines.append("\n## §6.3 Cross-axis bleed vs cfg_scale\n\n")
    lines.append("Mean off-diagonal bleed `(|gap_pos[i≠j]| + |gap_neg[i≠j]|) / 2`. "
                 "Pass: ≤ 0.15.\n\n")
    lines.append("| cfg_scale | mean off-diag bleed |\n|---:|---:|\n")
    for scale, ctrl in results:
        cross = np.array(ctrl["cross_axis"], dtype=float)
        valid = ~np.isnan(cross)
        mean_bleed = float(cross[valid].mean()) if valid.any() else float("nan")
        lines.append(f"| {scale} | {mean_bleed:.3f} |\n")

    lines.append("\n## Summary\n\n")
    # Pick best scale by V-axis gap_pos.
    best_scale, best_ctrl = max(
        results, key=lambda x: x[1]["gap_pos_diag"][0],
    )
    best_v = best_ctrl["gap_pos_diag"][0]
    lines.append(f"- Best cfg_scale by gap_pos V: **{best_scale}** "
                 f"(gap_pos V = {best_v:+.3f})\n")
    pass_axes = sum(
        1 for v in best_ctrl["gap_pos_diag"] if v >= 0.30
    )
    lines.append(f"- At best scale: {pass_axes}/3 axes pass gap_pos ≥ +0.30\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(lines))
    summary = {
        "dit": str(args.dit),
        "vae": str(args.vae),
        "n_clips": len(eval_clips),
        "scales": [
            {"cfg_scale": s, **ctrl} for s, ctrl in results
        ],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.output}", file=sys.stderr)
    print(f"wrote {args.summary_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
