#!/usr/bin/env python3
"""Sample a single held-out clip and compare to ground truth.

Visual gut-check that the DiT is learning real denoising vs.
gaming a degenerate (cheap-task) loss surface — independent of the
proxy ε-MSE metric.

Pipeline:
  1. Load DiT + frozen VAE.
  2. Pick a held-out clip (by source + index or by emotion).
  3. DDIM-sample a latent conditioned on the clip's Whisper + V/A/D.
  4. Undo per-channel z-score (if `zscore_latents=True` in ckpt).
  5. VAE-decode the sampled latent → ARKit-52 blendshape trajectory.
  6. Plot 8 representative channels alongside ground truth and dump
     summary stats (per-channel correlation, RMS error).

Usage:
  ./05_sample_smoke.py \\
      --dit artifacts/blendshape_dit.pt \\
      --vae artifacts/blendshape_vae.pt \\
      --whisper-cache artifacts/whisper_cache \\
      --data-dir data \\
      --actor M003 --emotion happy --row 0 \\
      --output artifacts/smoke_sample.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.diffusion import CosineSchedule, DiffusionConfig, ddim_sample  # noqa: E402
from lib.dit import DiT1D, DiTConfig  # noqa: E402
from lib.emotion_to_vad import emotion_to_vad  # noqa: E402
from lib.mead_3d_loader import MeadParquet, discover_parquets, parse_path  # noqa: E402
from lib.vae import BlendshapeVAE, VAEConfig  # noqa: E402


# Number of channels to plot. Picked at runtime as the top-K by GT
# variance — MEAD_3D has 54 channels and the order is not documented,
# so naming-by-index is unreliable. The high-variance channels are
# the ones that actually move during a clip and are the meaningful
# qualitative comparison.
N_PLOT_CHANNELS = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dit", type=Path, default=Path("artifacts/blendshape_dit.pt"))
    p.add_argument("--vae", type=Path, default=Path("artifacts/blendshape_vae.pt"))
    p.add_argument("--whisper-cache", type=Path,
                   default=Path("artifacts/whisper_cache"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--source-id", type=str, default=None,
                   help="MEAD_3D source-id (e.g., `mead_3d_M003`). If unset, "
                        "the first source containing --actor is used.")
    p.add_argument("--actor", type=str, default="M003")
    p.add_argument("--emotion", type=str, default="happy",
                   help="Pick first matching clip with this emotion.")
    p.add_argument("--row", type=int, default=None,
                   help="Specific row index in the parquet. Overrides --emotion.")
    p.add_argument("--intensity", type=int, default=None,
                   help="MEAD intensity filter (1=subtle, 2=mid, 3=peak). "
                        "Combined with --actor + --emotion to pick a specific clip.")
    p.add_argument(
        "--vad-scale", type=float, default=1.0,
        help="Multiplier on the V/A/D vector at inference. 1.0 = use the "
             "emotion table's value as-is. >1 amplifies emotion request "
             "(e.g. 1.5 = 'extra happy'). Compounds with --cfg-scale.",
    )
    p.add_argument(
        "--intensity-cond", type=float, default=1.0,
        help="Intensity dim for d_vad=4 ckpts. 0=flat, 0.33/0.67/1.0 "
             "match MEAD intensity 1/2/3, >1.0 extrapolates. Ignored "
             "for legacy d_vad=3 ckpts.",
    )
    p.add_argument(
        "--cfg-scale", type=float, default=3.0,
        help="Classifier-free-guidance scale. 1.0 = conditional only "
             "(weakest emotion expression). Per the cfg_sweep_canonical "
             "report, cfg=3.0 gives the best balance of V/A/D "
             "controllability without the D-axis collapse seen at cfg=5.",
    )
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/smoke_sample.png"))
    p.add_argument(
        "--render-mp4", action="store_true",
        help="Also render GT and sampled blendshapes as MP4 via the "
             "Bevy `affect_face_smoke` binary. Output paths derived from "
             "--output (replacing .png with _gt.mp4 and _sample.mp4).",
    )
    p.add_argument(
        "--render-glb", type=Path,
        default=Path("avatars/processed/beautiful_assistant.smpl_aligned.glb"),
        help="GLB avatar to render against (must have ARKit blendshapes baked).",
    )
    p.add_argument(
        "--render-map", type=Path,
        default=Path("avatars/beautiful_assistant.arkit_map.resolved.json"),
        help="ARKit-name → GLB-morph-target map JSON. Default points to "
             "the beautiful_assistant map which uses CC5 mouth morph "
             "names; pair it with a GLB onboarded via "
             "`scripts/avatars/onboard_avatar.sh --arkit-map ...` so the "
             "mouth morphs survive the 64-slot Bevy cap.",
    )
    p.add_argument(
        "--smooth-sigma", type=float, default=1.5,
        help="Gaussian σ (in frames) applied to the DiT sample's "
             "blendshape trajectory before rendering. The deterministic "
             "AE has no temporal-smoothness prior on its latent space, "
             "so DiT-sampled (non-GT-encoded) latents decode to per-frame "
             "Δ ~10× higher than GT — visible as 'spazzing'. σ=1.5 frames "
             "(~50ms at 30fps) brings sample velocity within ~1.5× of GT "
             "without blurring expression peaks. Set to 0 to disable.",
    )
    p.add_argument(
        "--listener-mode", action="store_true",
        help="Phase L0 of LISTENER_MODE_PLAN: zero out 19 speech-coupled "
             "/ lip-sync-prone channels on the DiT sample before "
             "rendering. Includes all jaw + mouth-shape channels "
             "(close/funnel/left/right/pucker/rollLower/rollUpper) plus "
             "mouthLowerDown / mouthUpperUp / mouthStretch / mouthPress "
             "— all viseme-correlated even though some have semantic "
             "affect uses. Surviving affect channels: mouthSmile, "
             "mouthFrown, mouthDimple, mouthShrug + all upper-face. "
             "No retrain required.",
    )
    p.add_argument(
        "--render-binary", type=Path,
        default=Path("target/release/examples/affect_face_smoke"),
        help="Path to the compiled `affect_face_smoke` binary.",
    )
    p.add_argument(
        "--render-fps", type=int, default=30,
        help="Output MP4 frame rate. Should match the dataset's blendshape rate.",
    )
    return p.parse_args()


def pick_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_parquet(data_dir: Path, source_id: str | None, actor: str) -> tuple[MeadParquet, str]:
    triples = discover_parquets(data_dir)
    if source_id:
        for path, actor_override, sid in triples:
            if sid == source_id:
                return MeadParquet(path, actor_override=actor_override,
                                   source_id=sid), sid
        raise SystemExit(f"source-id {source_id!r} not found under {data_dir}")
    for path, actor_override, sid in triples:
        pq = MeadParquet(path, actor_override=actor_override, source_id=sid)
        for p in pq.all_paths():
            meta = parse_path(p, actor_override=actor_override) or {}
            if meta.get("actor") == actor:
                return pq, sid
    raise SystemExit(f"no parquet contains actor {actor!r} under {data_dir}")


def pick_row(
    pq: MeadParquet, actor: str, emotion: str, row: int | None,
    intensity: int | None = None,
) -> int:
    paths = pq.all_paths()
    if row is not None:
        if row < 0 or row >= len(paths):
            raise SystemExit(f"row {row} out of range [0, {len(paths)})")
        return row
    for i, p in enumerate(paths):
        meta = parse_path(p, actor_override=pq.actor_override) or {}
        if meta.get("actor") != actor or meta.get("emotion") != emotion:
            continue
        if intensity is not None and meta.get("intensity") != intensity:
            continue
        return i
    raise SystemExit(
        f"no clip with actor={actor} emotion={emotion} "
        f"intensity={intensity} in {pq.source_id}"
    )


def fetch_clip(
    pq: MeadParquet, row_idx: int,
) -> tuple[np.ndarray, dict, bytes | None]:
    """Returns `(actions, meta, audio_bytes)`. `audio_bytes` is the
    raw WAV payload from the parquet's `audio` struct (16 kHz mono),
    or `None` if missing."""
    cur = 0
    for rg in range(pq.num_row_groups):
        n = pq._pf.metadata.row_group(rg).num_rows
        if row_idx < cur + n:
            tbl = pq._pf.read_row_group(
                rg, columns=["path", "actions", "audio"],
            )
            local = row_idx - cur
            actions = np.asarray(tbl["actions"][local].as_py(), dtype=np.float32)
            path = tbl["path"][local].as_py()
            meta = parse_path(path, actor_override=pq.actor_override) or {}
            audio = (tbl["audio"][local].as_py() or {}).get("bytes")
            return actions, meta, audio
        cur += n
    raise IndexError(row_idx)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    print(f"device: {device}", file=sys.stderr)

    # --- Load VAE ---
    print(f"loading VAE from {args.vae}", file=sys.stderr)
    vae_ckpt = torch.load(args.vae, map_location=device, weights_only=False)
    vae_cfg = VAEConfig(**vae_ckpt["config"])
    vae = BlendshapeVAE(vae_cfg).to(device).eval()
    vae.load_state_dict(vae_ckpt["state_dict"])

    # --- Load DiT ---
    print(f"loading DiT from {args.dit}", file=sys.stderr)
    dit_ckpt = torch.load(args.dit, map_location=device, weights_only=False)
    dit_cfg = DiTConfig(**dit_ckpt["config"])
    dit = DiT1D(dit_cfg).to(device).eval()
    dit.load_state_dict(dit_ckpt["state_dict"])
    zscore = bool(dit_ckpt.get("zscore_latents", False))
    if zscore:
        latent_mu = torch.from_numpy(dit_ckpt["latent_mu"]).to(device)
        latent_std = torch.from_numpy(dit_ckpt["latent_std"]).to(device)
        print(
            f"  z-score active: mu range [{latent_mu.min():+.3f}, "
            f"{latent_mu.max():+.3f}], std range "
            f"[{latent_std.min():.3f}, {latent_std.max():.3f}]",
            file=sys.stderr,
        )
    else:
        latent_mu = torch.zeros(vae_cfg.d_lat, device=device)
        latent_std = torch.ones(vae_cfg.d_lat, device=device)
        print("  z-score: not present in ckpt (legacy run)", file=sys.stderr)

    print(
        f"  DiT: best_val_eps={dit_ckpt.get('best_val_eps', float('nan')):.5f}  "
        f"best_epoch={dit_ckpt.get('best_epoch', '?')}",
        file=sys.stderr,
    )

    # --- Pick a clip ---
    pq, sid = find_parquet(args.data_dir, args.source_id, args.actor)
    row_idx = pick_row(pq, args.actor, args.emotion, args.row, args.intensity)
    actions_gt, meta, audio_bytes = fetch_clip(pq, row_idx)
    print(
        f"clip: source={sid} row={row_idx} actor={meta.get('actor')} "
        f"emotion={meta.get('emotion')} intensity={meta.get('intensity')} "
        f"frames={actions_gt.shape[0]}",
        file=sys.stderr,
    )

    # --- Whisper cache lookup ---
    cache = args.whisper_cache / f"{sid}__{row_idx:04d}.npz"
    if not cache.exists():
        raise SystemExit(f"no Whisper cache at {cache}")
    whisper_np = np.load(cache)["hidden"].astype(np.float32)
    whisper = torch.from_numpy(whisper_np).unsqueeze(0).to(device)

    # --- Crop alignment: VAE downsamples 4x, so crop blendshapes to a
    #    multiple of 4. Sample the same temporal extent from Whisper.
    T_actions = (actions_gt.shape[0] // 4) * 4
    actions_gt = actions_gt[:T_actions]
    T_lat = T_actions // 4
    # Resample whisper to a fixed 8x audio-frames-per-latent ratio so the
    # cross-attention K/V length matches the training crop.
    t_aud_target = T_lat * 8
    whisper_resampled = torch.nn.functional.interpolate(
        whisper.transpose(1, 2), size=t_aud_target, mode="linear",
        align_corners=False,
    ).transpose(1, 2)

    vad_tuple = emotion_to_vad(meta["emotion"])
    vad_arr = [v * args.vad_scale for v in vad_tuple]
    if dit_cfg.d_vad == 4:
        vad_arr.append(float(args.intensity_cond))
    elif dit_cfg.d_vad != 3:
        raise SystemExit(f"unsupported dit_cfg.d_vad={dit_cfg.d_vad}")
    vad = torch.tensor(vad_arr, dtype=torch.float32, device=device).unsqueeze(0)
    if args.vad_scale != 1.0 or dit_cfg.d_vad == 4:
        scale_msg = f" vad×{args.vad_scale}" if args.vad_scale != 1.0 else ""
        intensity_msg = (f" intensity_cond={args.intensity_cond}"
                         if dit_cfg.d_vad == 4 else "")
        print(f"  conditioning:{scale_msg}{intensity_msg} -> "
              + ", ".join(f"{v:+.2f}" for v in vad_arr),
              file=sys.stderr)

    # --- DDIM sample ---
    print(
        f"sampling: t_lat={T_lat}, t_aud={t_aud_target}, "
        f"cfg_scale={args.cfg_scale}, steps={args.n_steps}",
        file=sys.stderr,
    )
    schedule = CosineSchedule(DiffusionConfig(
        n_timesteps=dit_cfg.n_timesteps, cfg_drop_prob=0.0,
    )).to(device)

    def model_fn(x_t, t, w, v):
        return dit(x_t, t, w, v)

    null_v = torch.zeros_like(vad)
    z = ddim_sample(
        schedule=schedule, model_fn=model_fn,
        shape=(1, T_lat, vae_cfg.d_lat),
        n_steps=args.n_steps, device=device,
        cond=(whisper_resampled, vad),
        null_cond=(whisper_resampled, null_v) if args.cfg_scale != 1.0 else None,
        cfg_scale=args.cfg_scale,
    )
    # --- Undo z-score and VAE-decode ---
    latent = z * latent_std + latent_mu
    with torch.no_grad():
        decoded = vae.decode(latent)
    sample = decoded[0].cpu().numpy()  # (T, 52)

    # --- Stats ---
    sample = sample[: actions_gt.shape[0]]
    actions_gt = actions_gt[: sample.shape[0]]
    rms = float(np.sqrt(((sample - actions_gt) ** 2).mean()))
    per_ch_corr = []
    for ch in range(actions_gt.shape[1]):
        gt = actions_gt[:, ch]
        sm = sample[:, ch]
        if gt.std() < 1e-6 or sm.std() < 1e-6:
            per_ch_corr.append(0.0)
        else:
            per_ch_corr.append(float(np.corrcoef(gt, sm)[0, 1]))
    per_ch_corr = np.array(per_ch_corr)
    print(
        f"\nresults: rms={rms:.4f}  "
        f"mean_corr={per_ch_corr.mean():+.3f}  "
        f"top10_corr_mean={np.sort(per_ch_corr)[-10:].mean():+.3f}  "
        f"frac_corr_above_0.3={(per_ch_corr > 0.3).mean():.2f}",
        file=sys.stderr,
    )

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping plot", file=sys.stderr)
        return 0

    # Pick the K highest-variance GT channels — the channels that actually
    # move during this clip. Naming by index isn't reliable for MEAD_3D's
    # 54-channel output, so we just label by index + show per-channel r.
    variances = actions_gt.var(axis=0)
    top_chs = np.argsort(variances)[::-1][:N_PLOT_CHANNELS]

    t_axis = np.arange(actions_gt.shape[0]) / 30.0  # 30 fps
    title_str = (
        f"{meta.get('actor')} / {meta.get('emotion')} / "
        f"int={meta.get('intensity')} — best_val="
        f"{dit_ckpt.get('best_val_eps', 0):.4f}, "
        f"rms={rms:.4f}, mean_r={per_ch_corr.mean():+.2f}, "
        f"top10_r={np.sort(per_ch_corr)[-10:].mean():+.2f}"
    )

    # Stack the per-channel curves (top) over the all-channel heatmap
    # triplet (bottom). The heatmap rows are sorted by GT variance so
    # the active channels cluster near the top — flat regions at the
    # bottom are the channels that don't move in this clip and aren't
    # diagnostic.
    n_curve = len(top_chs)
    fig = plt.figure(figsize=(11, 2 + 1.6 * n_curve + 6))
    gs = fig.add_gridspec(
        nrows=n_curve + 3, ncols=1,
        height_ratios=[1.0] * n_curve + [2.5, 2.5, 2.5],
        hspace=0.35,
    )

    for i, ch in enumerate(top_chs):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(t_axis, actions_gt[:, ch], label="ground truth", lw=1.5, color="black")
        ax.plot(t_axis, sample[:, ch], label="sampled", lw=1.2,
                color="tab:red", alpha=0.85)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f"ch {ch}\nvar={variances[ch]:.3f}\nr={per_ch_corr[ch]:+.2f}",
                      fontsize=8)
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
        if i < n_curve - 1:
            ax.set_xticklabels([])

    # Sort all 54 channels by GT variance so the heatmaps are
    # interpretable at a glance — high-activity channels at the top.
    order = np.argsort(variances)[::-1]
    gt_sorted = actions_gt[:, order].T          # (54, T)
    sm_sorted = sample[:, order].T
    diff = sm_sorted - gt_sorted

    extent = [t_axis[0], t_axis[-1], len(order) - 0.5, -0.5]

    ax_gt = fig.add_subplot(gs[n_curve, 0])
    ax_gt.imshow(gt_sorted, aspect="auto", cmap="viridis",
                 vmin=0.0, vmax=1.0, extent=extent, interpolation="nearest")
    ax_gt.set_title("ground truth — channels sorted by variance", fontsize=9)
    ax_gt.set_ylabel("ch (sorted)", fontsize=8)

    ax_sm = fig.add_subplot(gs[n_curve + 1, 0])
    ax_sm.imshow(sm_sorted, aspect="auto", cmap="viridis",
                 vmin=0.0, vmax=1.0, extent=extent, interpolation="nearest")
    ax_sm.set_title("sampled", fontsize=9)
    ax_sm.set_ylabel("ch (sorted)", fontsize=8)

    ax_diff = fig.add_subplot(gs[n_curve + 2, 0])
    im = ax_diff.imshow(diff, aspect="auto", cmap="RdBu_r",
                        vmin=-0.5, vmax=0.5, extent=extent,
                        interpolation="nearest")
    ax_diff.set_title("sample − ground truth (red=over, blue=under)", fontsize=9)
    ax_diff.set_xlabel("time (s)")
    ax_diff.set_ylabel("ch (sorted)", fontsize=8)
    fig.colorbar(im, ax=ax_diff, fraction=0.02, pad=0.01)

    fig.suptitle(title_str, fontsize=10, y=0.995)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"wrote {args.output}", file=sys.stderr)

    if args.render_mp4:
        _render_mp4_pair(
            actions_gt=actions_gt, sample=sample,
            args=args, meta=meta, audio_bytes=audio_bytes,
        )
    return 0


def _render_mp4_pair(*, actions_gt, sample, args, meta, audio_bytes) -> None:
    """Map 54-channel MEAD_3D actions → ARKit-52, write JSONL, render MP4,
    and mux the original clip audio into the resulting MP4 (so the user
    can hear what the speaker is saying while watching the face)."""
    import shutil
    import subprocess

    from lib.mead3d_to_arkit import map_mead_to_arkit, write_arkit_jsonl  # noqa: E402

    if not args.render_binary.exists():
        print(f"render binary not found: {args.render_binary}", file=sys.stderr)
        print("  build with:", file=sys.stderr)
        print("  LIVE2D_CUBISM_CORE_DIR=$PWD/sdk/CubismSdkForNative-5-r.5 \\",
              file=sys.stderr)
        print("    cargo build --release --example affect_face_smoke "
              "-p remotemedia-core --features avatar-render-cc,avatar-lipsync",
              file=sys.stderr)
        return
    if not args.render_glb.exists():
        print(f"render GLB not found: {args.render_glb}", file=sys.stderr)
        return
    if not args.render_map.exists():
        print(f"ARKit map not found: {args.render_map}", file=sys.stderr)
        return

    base = args.output.with_suffix("")
    jsonl_gt = base.with_name(base.name + "_gt.jsonl")
    jsonl_sm = base.with_name(base.name + "_sample.jsonl")
    mp4_gt = base.with_name(base.name + "_gt.mp4")
    mp4_sm = base.with_name(base.name + "_sample.mp4")

    # Smooth the DiT sample only — GT is already temporally coherent
    # (from MEAD's MediaPipe extraction). σ=0 disables.
    sample_for_render = sample
    if args.smooth_sigma and args.smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        import numpy as np
        sample_for_render = gaussian_filter1d(
            np.asarray(sample, dtype=np.float32),
            sigma=float(args.smooth_sigma), axis=0,
        )
        print(f"smoothed sample: gaussian σ={args.smooth_sigma} frames",
              file=sys.stderr)

    # Listener-mode mask: zero out speech-coupled blendshape channels
    # so the assistant doesn't lip-sync to the user's audio. Applied to
    # the DiT sample only — GT is left untouched as a control. See
    # LISTENER_MODE_PLAN.md §2 for the channel split rationale.
    if args.listener_mode:
        import numpy as np
        from lib.vae import LISTENER_SPEECH_ONLY_CHANNELS
        sample_for_render = np.asarray(sample_for_render, dtype=np.float32).copy()
        sample_for_render[:, list(LISTENER_SPEECH_ONLY_CHANNELS)] = 0.0
        print(f"listener mode: zeroed {len(LISTENER_SPEECH_ONLY_CHANNELS)} "
              f"speech channels on sample (indices "
              f"{list(LISTENER_SPEECH_ONLY_CHANNELS)})", file=sys.stderr)

    write_arkit_jsonl(map_mead_to_arkit(actions_gt), args.render_fps, jsonl_gt)
    write_arkit_jsonl(map_mead_to_arkit(sample_for_render), args.render_fps, jsonl_sm)
    print(f"wrote JSONL: {jsonl_gt}, {jsonl_sm}", file=sys.stderr)

    # Persist the clip's original audio (16 kHz mono WAV from the parquet)
    # next to the MP4s, so the muxed videos carry an audio track and the
    # user can verify lip-sync by ear.
    audio_wav: Path | None = None
    if audio_bytes:
        audio_wav = base.with_name(base.name + "_audio.wav")
        audio_wav.write_bytes(audio_bytes)
        print(f"wrote audio: {audio_wav}", file=sys.stderr)

    for label, jsonl_path, mp4_path in (
        ("GT", jsonl_gt, mp4_gt),
        ("SAMPLE", jsonl_sm, mp4_sm),
    ):
        # Render the silent video first.
        silent_mp4 = mp4_path.with_name(mp4_path.stem + "_silent.mp4")
        cmd = [
            str(args.render_binary),
            "--glb", str(args.render_glb),
            "--map", str(args.render_map),
            "--face", str(jsonl_path),
            "--out", str(silent_mp4),
            "--fps", str(args.render_fps),
        ]
        print(f"\nrendering {label}: {mp4_path}", file=sys.stderr)
        env = {**__import__("os").environ}
        env.setdefault(
            "LIVE2D_CUBISM_CORE_DIR",
            str(Path.cwd().parent.parent / "sdk" / "CubismSdkForNative-5-r.5"),
        )
        # face-only render: focus on face, fit-to-frame.
        env.setdefault("CC_AVATAR_FOCUS", "face")
        env.setdefault("CC_AVATAR_FIT_FRAME", "1")
        env.setdefault("CC_RENDER_FAST", "1")
        try:
            subprocess.run(cmd, env=env, check=False, capture_output=False)
        except Exception as exc:
            print(f"  render failed: {exc}", file=sys.stderr)
            continue
        if not silent_mp4.exists():
            print(f"  ✗ {silent_mp4} not produced", file=sys.stderr)
            continue

        # Mux the audio. If audio is missing or ffmpeg not on PATH, just
        # rename silent → final and continue.
        ffmpeg = shutil.which("ffmpeg")
        if audio_wav is None or ffmpeg is None:
            silent_mp4.replace(mp4_path)
            print(f"  ✓ {mp4_path} (silent — "
                  f"{'no audio in parquet' if audio_wav is None else 'ffmpeg missing'})",
                  file=sys.stderr)
            continue
        mux_cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(silent_mp4), "-i", str(audio_wav),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(mp4_path),
        ]
        try:
            subprocess.run(mux_cmd, check=True)
            silent_mp4.unlink()
            print(f"  ✓ {mp4_path} (with audio)", file=sys.stderr)
        except subprocess.CalledProcessError as exc:
            print(f"  mux failed ({exc}); keeping silent {silent_mp4}",
                  file=sys.stderr)
            silent_mp4.replace(mp4_path)


if __name__ == "__main__":
    sys.exit(main())
