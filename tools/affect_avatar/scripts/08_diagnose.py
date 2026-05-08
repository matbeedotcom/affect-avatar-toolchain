#!/usr/bin/env python3
"""Stage-1 metrics dashboard — round-trip + (optionally) DiT diagnostic.

Per [STAGE1_VAE_PLAN.md §4 Experiment 0](../STAGE1_VAE_PLAN.md):

  Single script that produces the per-channel / per-emotion peak-recall
  table, smile-leakage score, and round-trip-vs-DiT gap, for any
  (VAE_ckpt, optional DiT_ckpt) pair.

The "round-trip" path is `vae(GT)["recon"]` with no DiT, no diffusion.
This isolates Stage-1 (encoder/decoder) from Stage-2 (denoiser).
The "DiT-vs-roundtrip gap" is the per-channel difference between
sampled output and round-trip output on the same clip — separates
Stage-1 cap from Stage-2 navigation error.

Pass condition: the 8 gates from STAGE1_VAE_PLAN §4 are evaluated and
a binary x/8 score is reported.

Usage:
  ./08_diagnose.py \\
      --vae artifacts/blendshape_vae.pt \\
      --output-dir artifacts/diag_v1 \\
      --actors M003 --intensity 3 --clips-per-emotion 5

  # With DiT (requires whisper cache):
  ./08_diagnose.py \\
      --vae artifacts/blendshape_vae.pt \\
      --dit artifacts/blendshape_dit.pt \\
      --whisper-cache artifacts/whisper_cache \\
      --output-dir artifacts/diag_v1_with_dit
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.diffusion import CosineSchedule, DiffusionConfig, ddim_sample  # noqa: E402
from lib.dit import DiT1D, DiTConfig  # noqa: E402
from lib.emotion_to_vad import vad_with_intensity  # noqa: E402
from lib.mead_3d_loader import (  # noqa: E402
    MeadParquet, discover_parquets, parse_path,
)
from lib.vae import BlendshapeVAE, VAEConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Channel layout — MEAD_3D's 54-channel actions in MediaPipe alphabetical
# (no leading `_neutral`) order; channels 51..53 are unknown extras.
# ---------------------------------------------------------------------------

CHANNEL_NAMES: tuple[str, ...] = (
    "browDownLeft", "browDownRight", "browInnerUp",        # 0..2
    "browOuterUpLeft", "browOuterUpRight",                 # 3..4
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",    # 5..7
    "eyeBlinkLeft", "eyeBlinkRight",                       # 8..9
    "eyeLookDownLeft", "eyeLookDownRight",                 # 10..11
    "eyeLookInLeft", "eyeLookInRight",                     # 12..13
    "eyeLookOutLeft", "eyeLookOutRight",                   # 14..15
    "eyeLookUpLeft", "eyeLookUpRight",                     # 16..17
    "eyeSquintLeft", "eyeSquintRight",                     # 18..19
    "eyeWideLeft", "eyeWideRight",                         # 20..21
    "jawForward", "jawLeft", "jawOpen", "jawRight",        # 22..25
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",   # 26..28
    "mouthFrownLeft", "mouthFrownRight",                   # 29..30
    "mouthFunnel", "mouthLeft",                            # 31..32
    "mouthLowerDownLeft", "mouthLowerDownRight",           # 33..34
    "mouthPressLeft", "mouthPressRight",                   # 35..36
    "mouthPucker", "mouthRight",                           # 37..38
    "mouthRollLower", "mouthRollUpper",                    # 39..40
    "mouthShrugLower", "mouthShrugUpper",                  # 41..42
    "mouthSmileLeft", "mouthSmileRight",                   # 43..44
    "mouthStretchLeft", "mouthStretchRight",               # 45..46
    "mouthUpperUpLeft", "mouthUpperUpRight",               # 47..48
    "noseSneerLeft", "noseSneerRight",                     # 49..50
)


# Named channels for gate scoring. Each maps to a tuple of source indices
# whose **max** is taken (covers L/R-paired channels: a peak on either
# side counts).
NAMED: dict[str, tuple[int, ...]] = {
    "browDn":      (0, 1),
    "browInnerUp": (2,),
    "browOuterUp": (3, 4),
    "cheekPuff":   (5,),
    "cheekSquint": (6, 7),
    "eyeBlink":    (8, 9),
    "eyeSquint":   (18, 19),
    "eyeWide":     (20, 21),
    "jawOpen":     (24,),
    "mouthClose":  (26,),
    "mouthFrown":  (29, 30),
    "mouthShrug":  (41, 42),
    "mouthSmile":  (43, 44),
    "mouthDimple": (27, 28),
    "noseSneer":   (49, 50),
}


_VALID_EMOTIONS = (
    "neutral", "happy", "sad", "angry", "fear",
    "disgusted", "surprised", "contempt",
)

# Negative-valence emotions used for smile-leak checks.
SMILE_LEAK_EMOTIONS = ("angry", "fear", "sad", "disgusted")
# Eye-wide-prone emotions for the disjunctive eyeWide gate.
EYEWIDE_EMOTIONS = ("angry", "fear", "disgusted", "surprised")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--vae", type=Path, required=True,
                   help="Stage-1 ckpt to evaluate.")
    p.add_argument("--dit", type=Path, default=None,
                   help="Optional Stage-2 ckpt. When set, also samples DiT "
                        "for the round-trip-vs-DiT gap.")
    p.add_argument("--whisper-cache", type=Path,
                   default=Path("artifacts/whisper_cache"),
                   help="Required if --dit is set.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--actors", type=str, default="M003",
                   help="Comma-separated actor IDs (default: M003 — the "
                        "test-actor used in the original round-trip "
                        "diagnostic).")
    p.add_argument("--intensity", type=int, default=3,
                   help="MEAD intensity to filter to (1, 2, or 3). 3 = peak.")
    p.add_argument("--emotions", type=str,
                   default="angry,fear,disgusted,sad,surprised,happy,contempt,neutral")
    p.add_argument("--clips-per-emotion", type=int, default=10)
    p.add_argument("--peak-agg", choices=("median", "max", "p75"),
                   default="median",
                   help="How to aggregate per-clip channel peaks across "
                        "the per-emotion clip pool. `median` is robust to "
                        "outlier clips; `max` mimics the cherry-picked "
                        "protocol of the original ad-hoc round-trip "
                        "diagnostic (high-amplitude clip per channel). "
                        "`p75` is a middle ground.")
    p.add_argument("--n-steps", type=int, default=50,
                   help="DDIM steps for DiT sampling (ignored if no --dit).")
    p.add_argument("--cfg-scale", type=float, default=3.0,
                   help="Classifier-free-guidance scale (DiT only).")
    p.add_argument("--intensity-cond", type=float, default=1.0,
                   help="Intensity conditioning for d_vad=4 DiT (DiT only).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="`artifacts/diag_<exp_id>/` per STAGE1_VAE_PLAN §8.")
    p.add_argument("--run-name", type=str, default=None,
                   help="Free-text label embedded in report header. "
                        "Defaults to the output dir name.")
    return p.parse_args()


def pick_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Clip selection — filter parquets by (actor, emotion, intensity).
# ---------------------------------------------------------------------------

def load_filtered_clips(
    data_dir: Path,
    whisper_cache_dir: Path,
    *,
    actors: set[str],
    emotions: set[str],
    intensity: int,
    clips_per_emotion: int,
    need_audio: bool,
    seed: int,
) -> dict[str, list[dict]]:
    """Returns `{emotion: [clip, ...]}`. Each clip has actions, vad,
    optionally whisper. Caps at `clips_per_emotion` per emotion.

    `need_audio=True` skips clips without a whisper cache hit.
    """
    rng = np.random.default_rng(seed)
    triples = discover_parquets(data_dir)
    if not triples:
        raise SystemExit(f"no `mead_3d*` parquets discovered under {data_dir}")

    by_emotion: dict[str, list[dict]] = {e: [] for e in emotions}
    seen: set = set()

    for parquet_path, actor_override, sid in triples:
        pq = MeadParquet(parquet_path, actor_override=actor_override, source_id=sid)
        cur = 0
        for rg in range(pq.num_row_groups):
            tbl = pq._pf.read_row_group(rg, columns=["path", "actions"])
            for i in range(tbl.num_rows):
                row_idx = cur + i
                path = tbl["path"][i].as_py()
                meta = parse_path(path, actor_override=actor_override) or {}
                emotion = meta.get("emotion", "")
                actor = meta.get("actor", "")
                intens = int(meta.get("intensity", 0))
                if emotion not in emotions:
                    continue
                if actor not in actors:
                    continue
                # Neutral has no real intensity in MEAD — accept any.
                if emotion != "neutral" and intens != intensity:
                    continue
                key = (actor, emotion, intens, meta.get("utt", ""))
                if key in seen:
                    continue
                if len(by_emotion[emotion]) >= clips_per_emotion:
                    continue
                seen.add(key)

                actions = np.asarray(
                    tbl["actions"][i].as_py(), dtype=np.float32,
                )
                if actions.shape[0] < 16:
                    continue

                whisper = None
                if need_audio:
                    cache = whisper_cache_dir / f"{sid}__{row_idx:04d}.npz"
                    if not cache.exists():
                        continue
                    whisper = np.load(cache)["hidden"].astype(np.float16)

                vad = np.array(
                    vad_with_intensity(emotion, intens),
                    dtype=np.float32,
                )
                by_emotion[emotion].append({
                    "actor": actor,
                    "emotion": emotion,
                    "intensity": intens,
                    "utt": meta.get("utt", ""),
                    "actions": actions,
                    "vad": vad,
                    "whisper": whisper,
                })
            cur += tbl.num_rows

    # Stable shuffle so reruns with the same seed pick the same clips.
    for emo in by_emotion:
        rng.shuffle(by_emotion[emo])
        by_emotion[emo] = by_emotion[emo][:clips_per_emotion]
    return by_emotion


# ---------------------------------------------------------------------------
# Round-trip + sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def vae_roundtrip(vae: BlendshapeVAE, x_tk: np.ndarray, device: torch.device) -> np.ndarray:
    """`(T, K) -> (T, K)` reconstruction via the VAE."""
    n_frames = (x_tk.shape[0] // 4) * 4
    if n_frames == 0:
        return x_tk
    x_tk = x_tk[:n_frames]
    x_btk = torch.from_numpy(x_tk).unsqueeze(0).to(device)
    out = vae(x_btk)
    recon = out["recon"][0].cpu().numpy()
    # vae.decode crops via min(recon.T, x.T) inside vae_loss; the forward
    # path may emit recon.T == x.T already, but we crop defensively.
    T = min(recon.shape[0], x_tk.shape[0])
    return recon[:T]


class DitSampler:
    def __init__(
        self, dit: DiT1D, dit_cfg: DiTConfig, vae: BlendshapeVAE,
        latent_mu: torch.Tensor, latent_std: torch.Tensor,
        n_steps: int, cfg_scale: float, device: torch.device,
    ) -> None:
        self.dit = dit
        self.vae = vae
        self.cfg = dit_cfg
        self.latent_mu = latent_mu
        self.latent_std = latent_std
        self.n_steps = n_steps
        self.cfg_scale = cfg_scale
        self.device = device
        self.schedule = CosineSchedule(DiffusionConfig(
            n_timesteps=dit_cfg.n_timesteps, cfg_drop_prob=0.0,
        )).to(device)

    @torch.no_grad()
    def sample(
        self, whisper_resampled: torch.Tensor, vad: torch.Tensor, t_lat: int,
    ) -> np.ndarray:
        def model_fn(x_t, t, w, v):
            return self.dit(x_t, t, w, v)

        null_v = torch.zeros_like(vad)
        z = ddim_sample(
            schedule=self.schedule, model_fn=model_fn,
            shape=(1, t_lat, self.cfg.d_lat),
            n_steps=self.n_steps, device=self.device,
            cond=(whisper_resampled, vad),
            null_cond=(whisper_resampled, null_v) if self.cfg_scale != 1.0 else None,
            cfg_scale=self.cfg_scale,
        )
        latent = z * self.latent_std + self.latent_mu
        decoded = self.vae.decode(latent)[0].cpu().numpy()
        return decoded


def prep_whisper(whisper_np: np.ndarray, t_aud_target: int,
                 device: torch.device) -> torch.Tensor:
    w = torch.from_numpy(whisper_np.astype(np.float32)).unsqueeze(0).to(device)
    w = torch.nn.functional.interpolate(
        w.transpose(1, 2), size=t_aud_target, mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    return w


# ---------------------------------------------------------------------------
# Per-emotion peak aggregation
# ---------------------------------------------------------------------------

def named_peak(traj: np.ndarray, indices: tuple[int, ...]) -> float:
    """Max over time, max over the given (paired) channels."""
    if traj.shape[0] == 0:
        return float("nan")
    cols = traj[:, list(indices)]
    return float(cols.max())


def aggregate_emotion_peaks(
    clip_results: list[dict], source: str, agg: str = "median",
) -> dict[str, float]:
    """Aggregate over clips of `named_peak(clip[source], indices)` per channel.

    `source` ∈ {"gt", "recon", "sample"}. Returns a dict[name -> peak].
    Skips entries where the source is missing (sample on no-DiT runs).
    `agg` ∈ {"median", "max", "p75"}.
    """
    out: dict[str, float] = {}
    for name, idx in NAMED.items():
        peaks = []
        for c in clip_results:
            arr = c.get(source)
            if arr is None:
                continue
            peaks.append(named_peak(arr, idx))
        if not peaks:
            out[name] = float("nan")
        elif agg == "max":
            out[name] = float(np.max(peaks))
        elif agg == "p75":
            out[name] = float(np.percentile(peaks, 75))
        else:
            out[name] = float(np.median(peaks))
    return out


# ---------------------------------------------------------------------------
# Gate evaluation — STAGE1_VAE_PLAN §4
# ---------------------------------------------------------------------------

def evaluate_gates(peaks: dict[str, dict[str, dict[str, float]]]) -> dict:
    """`peaks[emotion][source][name] -> peak`. Returns gate pass/fail.

    Gates are evaluated on the round-trip ("recon") source, since the
    plan defines the round-trip pass gate as the Stage-1 admission test.
    """
    def get(em: str, src: str, ch: str) -> float:
        return peaks.get(em, {}).get(src, {}).get(ch, float("nan"))

    gates: list[dict] = []

    # Gate 1 — angry eyeWide ≥ 0.70
    v = get("angry", "recon", "eyeWide")
    gates.append({
        "id": 1, "name": "angry eyeWide peak ≥ 0.70",
        "value": v, "target": "≥ 0.70",
        "pass": (not np.isnan(v)) and v >= 0.70,
    })
    # Gate 2 — angry browDn ≥ 0.50
    v = get("angry", "recon", "browDn")
    gates.append({
        "id": 2, "name": "angry browDn peak ≥ 0.50",
        "value": v, "target": "≥ 0.50",
        "pass": (not np.isnan(v)) and v >= 0.50,
    })
    # Gate 3 — angry cheekPuff ≥ 0.50
    v = get("angry", "recon", "cheekPuff")
    gates.append({
        "id": 3, "name": "angry cheekPuff peak ≥ 0.50",
        "value": v, "target": "≥ 0.50",
        "pass": (not np.isnan(v)) and v >= 0.50,
    })
    # Gate 4 — angry mouthSmile ≤ 0.10 (no smile-leak on angry)
    v = get("angry", "recon", "mouthSmile")
    gates.append({
        "id": 4, "name": "angry mouthSmile peak ≤ 0.10 (no leak)",
        "value": v, "target": "≤ 0.10",
        "pass": (not np.isnan(v)) and v <= 0.10,
    })
    # Gate 5 — fear/disgust/sad smile leak ≤ 0.10 (all three)
    leak_vals = {em: get(em, "recon", "mouthSmile") for em in ("fear", "disgusted", "sad")}
    leak_pass = all(
        (not np.isnan(v)) and v <= 0.10 for v in leak_vals.values()
    )
    gates.append({
        "id": 5,
        "name": "fear/disgust/sad smile leak ≤ 0.10 (all)",
        "value": ", ".join(f"{em}={v:.2f}" for em, v in leak_vals.items()),
        "target": "≤ 0.10 each", "pass": leak_pass,
    })
    # Gate 6 — happy mouthSmile within ±0.10 of GT
    happy_recon = get("happy", "recon", "mouthSmile")
    happy_gt = get("happy", "gt", "mouthSmile")
    diff = abs(happy_recon - happy_gt) if not (np.isnan(happy_recon) or np.isnan(happy_gt)) else float("nan")
    gates.append({
        "id": 6,
        "name": "happy mouthSmile within ±0.10 of GT",
        "value": f"recon={happy_recon:.2f}, gt={happy_gt:.2f}, |Δ|={diff:.2f}",
        "target": "|Δ| ≤ 0.10",
        "pass": (not np.isnan(diff)) and diff <= 0.10,
    })
    # Gate 7 — happy jawOpen within ±0.15 of GT
    jaw_recon = get("happy", "recon", "jawOpen")
    jaw_gt = get("happy", "gt", "jawOpen")
    jdiff = abs(jaw_recon - jaw_gt) if not (np.isnan(jaw_recon) or np.isnan(jaw_gt)) else float("nan")
    gates.append({
        "id": 7,
        "name": "happy jawOpen within ±0.15 of GT",
        "value": f"recon={jaw_recon:.2f}, gt={jaw_gt:.2f}, |Δ|={jdiff:.2f}",
        "target": "|Δ| ≤ 0.15",
        "pass": (not np.isnan(jdiff)) and jdiff <= 0.15,
    })
    # Gate 8 — eyeWide ≥ 0.70 on at least one of {angry, fear, disgust, surprised}
    eyewide_per = {em: get(em, "recon", "eyeWide") for em in EYEWIDE_EMOTIONS}
    eye_pass = any(
        (not np.isnan(v)) and v >= 0.70 for v in eyewide_per.values()
    )
    gates.append({
        "id": 8,
        "name": "eyeWide ≥ 0.70 on ≥1 of {angry, fear, disgusted, surprised}",
        "value": ", ".join(f"{em}={v:.2f}" for em, v in eyewide_per.items()),
        "target": "≥ 0.70 (any)",
        "pass": eye_pass,
    })

    n_pass = sum(1 for g in gates if g["pass"])
    return {
        "gates": gates,
        "n_pass": n_pass,
        "n_total": len(gates),
        "verdict": (
            "pass" if n_pass == len(gates) else
            "partial" if n_pass >= 5 else
            "fail"
        ),
    }


# ---------------------------------------------------------------------------
# Preservation-ratio evaluation — per F9 in STAGE1_VAE_PLAN.md
# ---------------------------------------------------------------------------

# A channel/emotion pair is considered "active" if GT max ≥ this floor.
# Below the floor, asking for preservation is meaningless (numerator is
# in the noise floor). 0.30 is a reasonable threshold given MEAD's
# noise floor (~0.03 for chronically-zero channels like cheekPuff).
PRESERVATION_GT_FLOOR = 0.30
# Per-pair preservation ratio threshold. recon ≥ this fraction of GT max.
PRESERVATION_RATIO_TARGET = 0.70
# Pair is "preserved" if ratio ≥ target. Median ratio across pairs is
# the meta-metric.
MEDIAN_PRESERVATION_TARGET = 0.85
# Decoder-invention check: for "near-zero" channels (GT max ≤ this),
# recon should not exceed GT by more than INVENTION_TOLERANCE.
INVENTION_GT_CEILING = 0.10
INVENTION_TOLERANCE = 0.20


def evaluate_preservation_gates(
    peaks: dict[str, dict[str, dict[str, float]]],
) -> dict:
    """Per-(emotion, channel) preservation ratios.

    For each pair where `gt_peak ≥ PRESERVATION_GT_FLOOR`, compute
    `recon_peak / gt_peak` and check if ≥ `PRESERVATION_RATIO_TARGET`.

    Decoder-invention check: for pairs where `gt_peak ≤
    INVENTION_GT_CEILING`, check `recon_peak − gt_peak ≤
    INVENTION_TOLERANCE`.

    Per F9, this is the more honest formulation of "encoder doesn't
    wipe rare channels" than the absolute-threshold gates 1–8 — those
    were calibrated against cherry-picked clip data and are
    unreachable on random samples.
    """

    rows: list[dict] = []
    invention_rows: list[dict] = []
    for em, by_src in peaks.items():
        gt = by_src["gt"]
        rc = by_src["recon"]
        for ch_name in NAMED:
            gt_v = gt[ch_name]
            rc_v = rc[ch_name]
            if np.isnan(gt_v) or np.isnan(rc_v):
                continue
            if gt_v >= PRESERVATION_GT_FLOOR:
                ratio = rc_v / gt_v if gt_v > 0 else float("nan")
                rows.append({
                    "emotion": em, "channel": ch_name,
                    "gt": gt_v, "recon": rc_v, "ratio": ratio,
                    "pass": (not np.isnan(ratio))
                            and ratio >= PRESERVATION_RATIO_TARGET,
                })
            elif gt_v <= INVENTION_GT_CEILING:
                excess = rc_v - gt_v
                invention_rows.append({
                    "emotion": em, "channel": ch_name,
                    "gt": gt_v, "recon": rc_v, "excess": excess,
                    "pass": excess <= INVENTION_TOLERANCE,
                })

    if rows:
        ratios = np.array([r["ratio"] for r in rows], dtype=np.float64)
        ratios = ratios[~np.isnan(ratios)]
        median_ratio = float(np.median(ratios)) if ratios.size else float("nan")
        n_preserved = sum(1 for r in rows if r["pass"])
    else:
        median_ratio = float("nan")
        n_preserved = 0

    if invention_rows:
        n_invented = sum(1 for r in invention_rows if not r["pass"])
        max_excess = max((r["excess"] for r in invention_rows), default=0.0)
    else:
        n_invented = 0
        max_excess = 0.0

    # Meta-gates.
    meta_gates = [
        {
            "id": "P1",
            "name": f"Median preservation ratio ≥ {MEDIAN_PRESERVATION_TARGET}",
            "value": median_ratio,
            "target": f"≥ {MEDIAN_PRESERVATION_TARGET}",
            "pass": (not np.isnan(median_ratio))
                    and median_ratio >= MEDIAN_PRESERVATION_TARGET,
        },
        {
            "id": "P2",
            "name": (f"≥ 80% of (emotion, channel) pairs preserved "
                     f"(ratio ≥ {PRESERVATION_RATIO_TARGET}) where "
                     f"gt ≥ {PRESERVATION_GT_FLOOR}"),
            "value": (n_preserved / len(rows)) if rows else float("nan"),
            "target": "≥ 0.80",
            "pass": bool(rows) and (n_preserved / len(rows)) >= 0.80,
        },
        {
            "id": "P3",
            "name": (f"No decoder invention (max excess on near-zero GT "
                     f"channels ≤ {INVENTION_TOLERANCE})"),
            "value": max_excess,
            "target": f"≤ {INVENTION_TOLERANCE}",
            "pass": max_excess <= INVENTION_TOLERANCE,
        },
    ]
    n_pass = sum(1 for g in meta_gates if g["pass"])

    return {
        "rows": rows,
        "invention_rows": invention_rows,
        "median_ratio": median_ratio,
        "n_preserved": n_preserved,
        "n_total_active_pairs": len(rows),
        "n_invented": n_invented,
        "n_total_inactive_pairs": len(invention_rows),
        "max_excess": max_excess,
        "meta_gates": meta_gates,
        "n_meta_pass": n_pass,
        "n_meta_total": len(meta_gates),
        "verdict": (
            "pass" if n_pass == len(meta_gates) else
            "partial" if n_pass >= 2 else
            "fail"
        ),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_peak(v: float) -> str:
    return "—" if np.isnan(v) else f"{v:.2f}"


def render_per_channel_table(
    peaks: dict[str, dict[str, dict[str, float]]],
    emotions: list[str], with_sample: bool,
) -> list[str]:
    lines: list[str] = []
    cols = list(NAMED.keys())
    header = "| emotion | source | " + " | ".join(cols) + " |"
    sep = "|---|---|" + "|".join(["---:"] * len(cols)) + "|"
    lines.append(header + "\n")
    lines.append(sep + "\n")
    sources = ["gt", "recon"] + (["sample"] if with_sample else [])
    for em in emotions:
        for src in sources:
            row = f"| {em} | {src} |"
            for ch in cols:
                row += f" {fmt_peak(peaks[em][src][ch])} |"
            lines.append(row + "\n")
    return lines


def render_recon_vs_gt_delta(
    peaks: dict[str, dict[str, dict[str, float]]],
    emotions: list[str],
) -> list[str]:
    lines: list[str] = ["| emotion | channel | gt | recon | Δ |\n",
                        "|---|---|---:|---:|---:|\n"]
    for em in emotions:
        # Highlight the gate-relevant channels.
        for ch in ("eyeWide", "browDn", "cheekPuff", "mouthShrug",
                   "mouthSmile", "jawOpen", "mouthFrown", "browInnerUp"):
            gt_v = peaks[em]["gt"][ch]
            rc_v = peaks[em]["recon"][ch]
            if np.isnan(gt_v) and np.isnan(rc_v):
                continue
            delta = rc_v - gt_v if not (np.isnan(gt_v) or np.isnan(rc_v)) else float("nan")
            mark = ""
            if not np.isnan(delta):
                if delta < -0.30:
                    mark = " ↓"
                elif delta > 0.20 and gt_v < 0.10:
                    mark = " ↑ invented"
            lines.append(
                f"| {em} | {ch} | {fmt_peak(gt_v)} | {fmt_peak(rc_v)} | "
                f"{delta:+.2f}{mark} |\n"
            )
    return lines


def render_dit_vs_recon_gap(
    peaks: dict[str, dict[str, dict[str, float]]],
    emotions: list[str],
) -> list[str]:
    """Per-channel Δ between sample and recon. Captures Stage-2 navigation
    error on top of Stage-1 cap.
    """
    lines: list[str] = ["| emotion | channel | recon | sample | Δ (sample−recon) |\n",
                        "|---|---|---:|---:|---:|\n"]
    for em in emotions:
        for ch in ("eyeWide", "browDn", "cheekPuff", "mouthShrug",
                   "mouthSmile", "jawOpen", "mouthFrown"):
            rc_v = peaks[em]["recon"][ch]
            sm_v = peaks[em]["sample"][ch]
            if np.isnan(rc_v) and np.isnan(sm_v):
                continue
            delta = sm_v - rc_v if not (np.isnan(rc_v) or np.isnan(sm_v)) else float("nan")
            lines.append(
                f"| {em} | {ch} | {fmt_peak(rc_v)} | {fmt_peak(sm_v)} | "
                f"{delta:+.2f} |\n"
            )
    return lines


def write_report(
    out_path: Path, *,
    run_name: str,
    vae_path: Path, vae_meta: dict,
    dit_path: Path | None, dit_meta: dict | None,
    actors: list[str], intensity: int,
    emotions: list[str],
    n_clips_per_emotion: dict[str, int],
    peaks: dict[str, dict[str, dict[str, float]]],
    gate_result: dict,
    pres_result: dict,
    cfg_scale: float, intensity_cond: float, n_steps: int,
    peak_agg: str,
) -> None:
    lines: list[str] = []
    lines.append(f"# Stage-1 diagnostic — {run_name}\n\n")
    lines.append(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n\n")
    lines.append(f"## Configuration\n\n")
    lines.append(f"- VAE ckpt: `{vae_path}`\n")
    if vae_meta.get("config"):
        lines.append(f"- VAE config: `{vae_meta['config']}`\n")
    if vae_meta.get("channel_weight_power") is not None:
        lines.append(
            f"- VAE channel_weight_power: `{vae_meta['channel_weight_power']}`\n"
        )
    if dit_path:
        lines.append(f"- DiT ckpt: `{dit_path}`\n")
        if dit_meta and dit_meta.get("config"):
            lines.append(f"- DiT config: `{dit_meta['config']}`\n")
        lines.append(f"- DiT cfg_scale: {cfg_scale:.2f}, "
                     f"intensity_cond: {intensity_cond:.2f}, "
                     f"n_steps: {n_steps}\n")
    lines.append(f"- Actors: {', '.join(actors)}\n")
    lines.append(f"- Intensity filter (non-neutral): {intensity}\n")
    counts = ", ".join(f"{e}={n_clips_per_emotion[e]}" for e in emotions)
    lines.append(f"- Clips per emotion: {counts}\n")
    lines.append(f"- Peak aggregation across clips: `{peak_agg}`\n\n")

    # Gate summary up top.
    n_pass = gate_result["n_pass"]
    n_total = gate_result["n_total"]
    verdict = gate_result["verdict"].upper()
    lines.append(f"## Round-trip gate summary — {n_pass}/{n_total} ({verdict})\n\n")
    lines.append("| # | gate | value | target | pass |\n")
    lines.append("|---:|---|---|---|:---:|\n")
    for g in gate_result["gates"]:
        v = g["value"]
        v_str = f"{v:.2f}" if isinstance(v, float) else str(v)
        mark = "✅" if g["pass"] else "❌"
        lines.append(f"| {g['id']} | {g['name']} | {v_str} | {g['target']} | {mark} |\n")
    lines.append("\n")

    # Preservation-ratio gates (per F9 in STAGE1_VAE_PLAN.md).
    p_pass = pres_result["n_meta_pass"]
    p_total = pres_result["n_meta_total"]
    lines.append(f"## Preservation-ratio gates (per F9) — "
                 f"{p_pass}/{p_total} ({pres_result['verdict'].upper()})\n\n")
    lines.append("Preservation ratio = `recon_peak / gt_peak` for "
                 f"(emotion, channel) pairs where `gt_peak ≥ "
                 f"{PRESERVATION_GT_FLOOR}`. Decoder invention = "
                 f"`recon_peak − gt_peak` for pairs where `gt_peak ≤ "
                 f"{INVENTION_GT_CEILING}`. Strict gates 1–8 above are "
                 "calibrated against cherry-picked clips; these meta-gates "
                 "(P1–P3) test whether the encoder/decoder is faithful to "
                 "the random-sampled GT distribution.\n\n")
    lines.append("| # | gate | value | target | pass |\n")
    lines.append("|---|---|---|---|:---:|\n")
    for g in pres_result["meta_gates"]:
        v = g["value"]
        v_str = f"{v:.3f}" if isinstance(v, float) else str(v)
        mark = "✅" if g["pass"] else "❌"
        lines.append(f"| {g['id']} | {g['name']} | {v_str} | "
                     f"{g['target']} | {mark} |\n")
    lines.append(f"\nActive pairs (gt ≥ {PRESERVATION_GT_FLOOR}): "
                 f"**{pres_result['n_preserved']}/{pres_result['n_total_active_pairs']}** "
                 f"preserved (ratio ≥ {PRESERVATION_RATIO_TARGET}). "
                 f"Inactive pairs (gt ≤ {INVENTION_GT_CEILING}): "
                 f"{pres_result['n_total_inactive_pairs'] - pres_result['n_invented']}/"
                 f"{pres_result['n_total_inactive_pairs']} non-invented.\n\n")

    # Worst preservation cases (≤ 0.70 ratio) — actionable signal.
    bad = sorted(
        (r for r in pres_result["rows"] if not r["pass"]),
        key=lambda r: r["ratio"],
    )
    if bad:
        lines.append("### Worst preservation cases (ratio < target)\n\n")
        lines.append("| emotion | channel | gt | recon | ratio |\n"
                     "|---|---|---:|---:|---:|\n")
        for r in bad[:15]:
            lines.append(f"| {r['emotion']} | {r['channel']} | "
                         f"{r['gt']:.2f} | {r['recon']:.2f} | "
                         f"{r['ratio']:.2f} |\n")
        lines.append("\n")

    # Worst invention cases (excess > tolerance).
    invented = sorted(
        (r for r in pres_result["invention_rows"] if not r["pass"]),
        key=lambda r: r["excess"], reverse=True,
    )
    if invented:
        lines.append("### Decoder-invention cases (excess > tolerance)\n\n")
        lines.append("| emotion | channel | gt | recon | excess |\n"
                     "|---|---|---:|---:|---:|\n")
        for r in invented[:15]:
            lines.append(f"| {r['emotion']} | {r['channel']} | "
                         f"{r['gt']:.2f} | {r['recon']:.2f} | "
                         f"{r['excess']:+.2f} |\n")
        lines.append("\n")

    # Per-channel peak table.
    lines.append("## Per-emotion / per-channel peak table\n\n")
    agg_label = {
        "median": "Median",
        "max":    "Max-clip max-time",
        "p75":    "75th percentile",
    }.get(peak_agg, peak_agg)
    lines.append(f"{agg_label} across clips of channel-peak (max over time, "
                 "max across L/R-paired indices). `gt` = parquet ground "
                 "truth, `recon` = VAE round-trip, `sample` = DiT sampled.\n\n")
    with_sample = any(
        any(c.get("sample") is not None for c in [])  # placeholder
        for em in emotions
    )
    # Determine with_sample from peaks structure: a sample peak is present
    # iff it's not NaN for at least one (em, ch).
    with_sample = any(
        not np.isnan(peaks[em]["sample"][ch])
        for em in emotions for ch in NAMED
    )
    lines.extend(render_per_channel_table(peaks, emotions, with_sample))
    lines.append("\n")

    # Recon-vs-GT delta on the gate-relevant channels.
    lines.append("## Round-trip delta on key channels (recon − gt)\n\n")
    lines.append("Negative Δ on a channel that GT activates strongly = "
                 "encoder collapse. Positive Δ on a channel that GT keeps "
                 "near zero = decoder invention.\n\n")
    lines.extend(render_recon_vs_gt_delta(peaks, emotions))
    lines.append("\n")

    # DiT-vs-recon gap (only if DiT loaded).
    if with_sample:
        lines.append("## DiT-vs-round-trip gap on key channels (sample − recon)\n\n")
        lines.append("Per F8 in STAGE1_VAE_PLAN: this is the only reliable "
                     "separator between Stage-1 cap and Stage-2 navigation "
                     "error. Large |Δ| with bounded recon = Stage-2 issue. "
                     "Small Δ but bad recon = Stage-1 issue.\n\n")
        lines.extend(render_dit_vs_recon_gap(peaks, emotions))
        lines.append("\n")

    # Footer.
    lines.append("## Notes\n\n")
    lines.append("- Gate definitions and thresholds from "
                 "[STAGE1_VAE_PLAN.md §4](../STAGE1_VAE_PLAN.md). 8/8 = "
                 "Stage-1 admission to DiT retrain (Exp 5). 5–7/8 = "
                 "partial (retune in same family). <5/8 = branch decision.\n")
    lines.append("- Channel ordering follows MediaPipe alphabetical without "
                 "leading `_neutral`; channels 51..53 of MEAD_3D are "
                 "unknown extras and ignored.\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    print(f"device: {device}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or args.output_dir.name

    # --- Load VAE ---
    print(f"loading VAE: {args.vae}", file=sys.stderr)
    vae_ckpt = torch.load(args.vae, map_location=device, weights_only=False)
    vae_cfg = VAEConfig(**vae_ckpt["config"])
    vae = BlendshapeVAE(vae_cfg).to(device).eval()
    vae.load_state_dict(vae_ckpt["state_dict"])

    sampler: DitSampler | None = None
    dit_ckpt = None
    dit_cfg: DiTConfig | None = None
    if args.dit:
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
        sampler = DitSampler(
            dit=dit, dit_cfg=dit_cfg, vae=vae,
            latent_mu=latent_mu, latent_std=latent_std,
            n_steps=args.n_steps, cfg_scale=args.cfg_scale, device=device,
        )

    # --- Load clips ---
    actors = {a.strip() for a in args.actors.split(",") if a.strip()}
    emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]
    for em in emotions:
        if em not in _VALID_EMOTIONS:
            raise SystemExit(f"unknown emotion: {em}")

    print(f"loading clips for actors={sorted(actors)} "
          f"emotions={emotions} intensity={args.intensity}...", file=sys.stderr)
    t0 = time.time()
    by_emotion = load_filtered_clips(
        args.data_dir, args.whisper_cache,
        actors=actors, emotions=set(emotions),
        intensity=args.intensity,
        clips_per_emotion=args.clips_per_emotion,
        need_audio=sampler is not None,
        seed=args.seed,
    )
    counts = {em: len(by_emotion[em]) for em in emotions}
    print(f"  loaded ({time.time() - t0:.1f}s) per-emotion clips: {counts}",
          file=sys.stderr)

    missing = [em for em, n in counts.items() if n == 0]
    if missing:
        print(f"  ⚠ no clips for emotions: {missing} — gates referencing "
              f"these will be NaN", file=sys.stderr)

    # --- Run round-trip + (optionally) sample ---
    print("running VAE round-trip"
          + (" + DiT sampling" if sampler else "")
          + " on each clip...", file=sys.stderr)
    t0 = time.time()
    clip_results: dict[str, list[dict]] = {em: [] for em in emotions}
    for em in emotions:
        for c in by_emotion[em]:
            actions = c["actions"]
            recon = vae_roundtrip(vae, actions, device)
            sample_arr = None
            if sampler is not None and c["whisper"] is not None:
                n_frames = (actions.shape[0] // 4) * 4
                t_lat = n_frames // 4
                whisper = prep_whisper(c["whisper"], t_lat * 8, device)
                d_vad = sampler.cfg.d_vad
                vad = c["vad"][:d_vad].copy()
                if d_vad == 4:
                    vad[3] = float(args.intensity_cond)
                vad_t = torch.from_numpy(vad).unsqueeze(0).to(device)
                sample_arr = sampler.sample(whisper, vad_t, t_lat)[:n_frames]
            clip_results[em].append({
                "gt": actions[: (actions.shape[0] // 4) * 4],
                "recon": recon,
                "sample": sample_arr,
            })
    print(f"  done ({time.time() - t0:.0f}s)", file=sys.stderr)

    # --- Aggregate per-emotion peaks ---
    peaks: dict[str, dict[str, dict[str, float]]] = {}
    for em in emotions:
        peaks[em] = {
            "gt":     aggregate_emotion_peaks(clip_results[em], "gt", args.peak_agg),
            "recon":  aggregate_emotion_peaks(clip_results[em], "recon", args.peak_agg),
            "sample": aggregate_emotion_peaks(clip_results[em], "sample", args.peak_agg),
        }

    # --- Evaluate gates (round-trip only) ---
    gate_result = evaluate_gates(peaks)
    print(f"\nround-trip gates: {gate_result['n_pass']}/{gate_result['n_total']} "
          f"({gate_result['verdict']})", file=sys.stderr)
    for g in gate_result["gates"]:
        mark = "PASS" if g["pass"] else "FAIL"
        v = g["value"]
        v_str = f"{v:.2f}" if isinstance(v, float) else str(v)
        print(f"  [{g['id']}] {mark}  {g['name']}  →  {v_str}", file=sys.stderr)

    pres_result = evaluate_preservation_gates(peaks)
    print(f"\npreservation gates (per F9): "
          f"{pres_result['n_meta_pass']}/{pres_result['n_meta_total']} "
          f"({pres_result['verdict']})", file=sys.stderr)
    print(f"  median ratio={pres_result['median_ratio']:.3f} "
          f"({pres_result['n_preserved']}/{pres_result['n_total_active_pairs']} "
          f"active pairs preserved); "
          f"max invention={pres_result['max_excess']:+.3f}",
          file=sys.stderr)
    for g in pres_result["meta_gates"]:
        mark = "PASS" if g["pass"] else "FAIL"
        v = g["value"]
        v_str = f"{v:.3f}" if isinstance(v, float) else str(v)
        print(f"  [{g['id']}] {mark}  {g['name']}  →  {v_str}", file=sys.stderr)

    # --- Write outputs ---
    report_path = args.output_dir / "report.md"
    write_report(
        report_path,
        run_name=run_name,
        vae_path=args.vae, vae_meta=vae_ckpt,
        dit_path=args.dit, dit_meta=dit_ckpt,
        actors=sorted(actors), intensity=args.intensity,
        emotions=emotions, n_clips_per_emotion=counts,
        peaks=peaks, gate_result=gate_result, pres_result=pres_result,
        cfg_scale=args.cfg_scale, intensity_cond=args.intensity_cond,
        n_steps=args.n_steps, peak_agg=args.peak_agg,
    )
    print(f"wrote {report_path}", file=sys.stderr)

    summary = {
        "run_name": run_name,
        "vae_ckpt": str(args.vae),
        "dit_ckpt": str(args.dit) if args.dit else None,
        "actors": sorted(actors),
        "intensity": args.intensity,
        "emotions": emotions,
        "clips_per_emotion": counts,
        "peaks": peaks,
        "gates": gate_result,
        "preservation_gates": pres_result,
        "cfg_scale": args.cfg_scale,
        "intensity_cond": args.intensity_cond,
        "n_steps": args.n_steps,
        "peak_agg": args.peak_agg,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"wrote {summary_path}", file=sys.stderr)

    # Copy ckpt metadata (but not the weights — too large).
    ckpt_meta = {
        "vae": {k: v for k, v in vae_ckpt.items() if k not in ("state_dict",)},
    }
    if dit_ckpt is not None:
        ckpt_meta["dit"] = {
            k: v for k, v in dit_ckpt.items() if k not in ("state_dict",)
        }
    # Convert non-serializable pieces (numpy arrays, tensors) to lists.
    def _norm(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, dict):
            return {k: _norm(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_norm(v) for v in o]
        return o
    ckpt_meta_path = args.output_dir / "ckpt_meta.json"
    ckpt_meta_path.write_text(json.dumps(_norm(ckpt_meta), indent=2, default=str))
    print(f"wrote {ckpt_meta_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
