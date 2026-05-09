"""VAD → ARKit-52 blendshape interpolator backed by MEAD-derived anchors.

The listener-face pipeline doesn't need audio at all: given a VAD
trajectory (per-frame valence/arousal/dominance + intensity), it
interpolates between the 21 emotion×intensity anchors built from
MEAD by [`10_build_vad_anchors.py`](../10_build_vad_anchors.py) (or
the inline equivalent in `artifacts/vad_anchors_delta.npz`).

Anchors are stored as **deltas from neutral baseline**, peak-aggregated
over MEAD clips:
  - For each MEAD `(emotion, intensity)` group, take per-clip
    max-over-time blendshape, mean across clips.
  - Subtract neutral L1 anchor as a baseline (removes systematic
    boundary-frame artefacts where MediaPipe misclassifies closed
    mouths as `mouthPucker` etc.).
  - Clip to [0, 1]: a positive emotion-delta only.

Inference path:
  - User audio → affect_calibration's whisper_to_vad regressor
    → per-frame `(V, A, D)` ∈ [-1, +1]³.
  - Optional: smooth that trajectory with a small EMA (the heavy
    listener smoothing is applied AFTER blendshape lookup, since
    smoothing in VAD space distorts emotion identity less than
    smoothing on output).
  - Look up nearest emotion anchors via Gaussian RBF in VAD space;
    weighted-sum gives the 54-channel MEAD blendshape, mapped to
    52-channel ARKit, optionally listener-masked.

Output is always **MEAD-54 channels** (caller maps to ARKit-52 via
[`mead3d_to_arkit.py`](mead3d_to_arkit.py)).

Usage:
    interp = VadToBlendshape.load("artifacts/vad_anchors_delta.npz")
    bs54 = interp.lookup(np.array([0.6, 0.4, 0.5]))  # → (54,)
    bs54_seq = interp.lookup(vad_traj)                # → (T, 54) for (T, 3) input
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


# Per [`emotion_to_vad.py`](emotion_to_vad.py) — same VAD anchors used
# for DiT conditioning. Reproduced here so the interpolator can be
# imported standalone without that dependency.
MEAD_VAD: Mapping[str, tuple[float, float, float]] = {
    "neutral":   (0.00, 0.00, 0.00),
    "happy":     (+0.80, +0.50, +0.40),
    "sad":       (-0.70, -0.30, -0.40),
    "angry":     (-0.50, +0.70, +0.50),
    "fear":      (-0.60, +0.60, -0.50),
    "disgusted": (-0.60, +0.30, -0.20),
    "surprised": (+0.30, +0.70,  0.00),
    "contempt":  (-0.40, -0.10, +0.50),
}


@dataclass
class VadToBlendshapeConfig:
    """Tunables for the RBF interpolator.

    - `tau`: Gaussian bandwidth in VAD space. Smaller = sharper
      transitions between anchors (more single-anchor at a time);
      larger = smoother blends. 0.25 chosen so a query near one anchor
      gets ~80% of its weight from that anchor + the rest spread.
    - `intensity_floor`: anchors below this normalized intensity get
      their delta scaled down. Stops L1 anchors from looking
      indistinguishable from rest.
    - `add_baseline`: when True, the returned blendshape is
      `baseline + emotion_delta`. When False, it's the pure delta
      (better for listener mode — no baseline pucker / squint
      bleeding through, cleaner rest pose).
    """
    tau: float = 0.25
    intensity_floor: float = 0.20
    add_baseline: bool = False


class VadToBlendshape:
    """Stateless RBF interpolator over MEAD emotion×intensity anchors.

    Caller passes in a VAD vector (or batch of them); interpolator
    returns the blended 54-channel MEAD blendshape vector(s).
    """

    def __init__(
        self,
        anchors: dict[tuple[str, int], np.ndarray],
        baseline: np.ndarray,
        cfg: VadToBlendshapeConfig | None = None,
    ) -> None:
        """`anchors`: `{(emotion_name, intensity): (54,) delta}`.
        `baseline`: `(54,)` neutral-L1 anchor used when `add_baseline=True`.
        """
        self.cfg = cfg or VadToBlendshapeConfig()

        # Stack anchors into a `(K, 54)` matrix and a parallel
        # `(K, 4)` matrix of (V, A, D, intensity_norm) for distance
        # computation. K = number of anchors; typically 21 (7 non-neutral
        # emotions × 3 intensities).
        emotions = []
        intensities = []
        deltas = []
        for (emo, lvl), d in anchors.items():
            if emo == "neutral":
                # Neutral has only one intensity (L1) and lives in the
                # baseline; skip it in the anchor pool so it doesn't
                # dominate near-rest queries.
                continue
            emotions.append(emo)
            intensities.append(int(lvl))
            deltas.append(d)
        self.anchor_emotion = list(emotions)
        self.anchor_intensity = np.array(intensities, dtype=np.int32)
        self.deltas = np.stack(deltas, axis=0).astype(np.float32)   # (K, 54)
        self.baseline = baseline.astype(np.float32)                 # (54,)

        # Anchor coordinates in 4-D conditioning space:
        # (V, A, D, intensity_norm) where intensity_norm = lvl / 3.
        coords = np.zeros((len(emotions), 4), dtype=np.float32)
        for i, (emo, lvl) in enumerate(zip(emotions, self.anchor_intensity)):
            v, a, d = MEAD_VAD.get(emo, (0.0, 0.0, 0.0))
            coords[i] = (v, a, d, lvl / 3.0)
        self.anchor_coords = coords    # (K, 4)

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: Path | str,
        cfg: VadToBlendshapeConfig | None = None,
    ) -> "VadToBlendshape":
        """Load anchors from `vad_anchors_delta.npz`.

        Expected keys: `baseline` (54,), and one entry per anchor named
        `{emotion}_{intensity}` (e.g. `happy_3`, `surprised_2`).
        """
        data = np.load(str(path))
        baseline = data["baseline"]
        anchors: dict[tuple[str, int], np.ndarray] = {}
        for k in data.files:
            if k == "baseline":
                continue
            emo, lvl = k.rsplit("_", 1)
            anchors[(emo, int(lvl))] = data[k]
        return cls(anchors, baseline, cfg)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def lookup(self, vad: np.ndarray, intensity: float | np.ndarray = 1.0) -> np.ndarray:
        """`vad`: shape `(3,)` or `(T, 3)` — V/A/D in [-1, +1].
        `intensity`: scalar or `(T,)` — emotion intensity in [0, 1].
        Returns: `(54,)` or `(T, 54)` MEAD-54 blendshape.

        For batched input, intensity may be a scalar (broadcast) or a
        per-frame array.
        """
        vad = np.asarray(vad, dtype=np.float32)
        single = vad.ndim == 1
        if single:
            vad = vad[None, :]                          # (1, 3)

        T = vad.shape[0]
        if np.isscalar(intensity):
            intensity_arr = np.full((T,), float(intensity), dtype=np.float32)
        else:
            intensity_arr = np.asarray(intensity, dtype=np.float32).reshape(T)

        # Query coords: (T, 4)
        q = np.concatenate([vad, intensity_arr[:, None]], axis=1)

        # Pairwise squared distance to anchors: (T, K)
        d2 = ((q[:, None, :] - self.anchor_coords[None, :, :]) ** 2).sum(axis=2)

        # Gaussian RBF weights, normalized per query.
        w = np.exp(-d2 / (2 * self.cfg.tau * self.cfg.tau))
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-8)      # (T, K)

        # Weighted sum of anchor deltas → emotion delta per query.
        emo_delta = w @ self.deltas                                  # (T, 54)

        # Intensity floor: low-intensity queries get a scaled-down
        # emotion delta so the L1 anchors don't end up identical to
        # neutral.
        if self.cfg.intensity_floor > 0:
            scale = np.maximum(intensity_arr, self.cfg.intensity_floor)
            emo_delta = emo_delta * scale[:, None]

        if self.cfg.add_baseline:
            out = self.baseline[None, :] + emo_delta
        else:
            out = emo_delta

        # Clamp to [0, 1] — blendshapes are nonnegative by convention.
        out = np.clip(out, 0.0, 1.0)

        return out[0] if single else out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def anchor_summary(self) -> str:
        """Pretty-printed (anchor, top-3 channels) for debugging."""
        # Lazy import to keep core path dependency-free.
        try:
            from .mead3d_to_arkit import MEDIAPIPE_ORDER
        except ImportError:
            from mead3d_to_arkit import MEDIAPIPE_ORDER  # type: ignore[no-redef]
        lines = []
        for i, (emo, lvl) in enumerate(zip(self.anchor_emotion, self.anchor_intensity)):
            d = self.deltas[i, :51]
            top = np.argsort(d)[::-1][:3]
            sig = ", ".join(f"{MEDIAPIPE_ORDER[t]}={d[t]:+.2f}" for t in top)
            lines.append(f"  {emo:10s} L{int(lvl)} → {sig}")
        return "\n".join(lines)
