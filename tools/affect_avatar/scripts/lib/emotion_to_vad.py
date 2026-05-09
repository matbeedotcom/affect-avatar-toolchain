"""Categorical emotion → (valence, arousal, dominance) for MEAD_3D.

Coordinate values lifted from
[`tools/affect_calibration/scripts/lib/emotion_to_vad.py`](../../../affect_calibration/scripts/lib/emotion_to_vad.py)
to keep the affect_avatar V/A/D conditioning consistent with the
affect_calibration regressor (which also uses [-1, +1]³ per axis,
Russell's circumplex anchors).

MEAD-specific deltas vs the calibration table:
  - MEAD uses `fear` (we map to the calibration `fearful` anchor).
  - MEAD has `contempt`, which is *not* in the calibration table —
    new entry. Approximate based on Mehrabian's PAD model: low
    valence, low arousal, high dominance (a sneer is composed, not
    aroused, but socially superior in stance).
"""

from __future__ import annotations

from typing import Mapping


# Each tuple is (valence, arousal, dominance) ∈ [-1, +1]³.
MEAD_VAD: Mapping[str, tuple[float, float, float]] = {
    "neutral":   ( 0.00,  0.00,  0.00),
    "happy":     (+0.80, +0.50, +0.40),
    "sad":       (-0.70, -0.30, -0.40),
    "angry":     (-0.50, +0.70, +0.50),
    "fear":      (-0.60, +0.60, -0.50),
    "disgusted": (-0.60, +0.30, -0.20),
    "surprised": (+0.30, +0.70,  0.00),
    "contempt":  (-0.40, -0.10, +0.50),
}


def emotion_to_vad(emotion: str) -> tuple[float, float, float]:
    """`emotion -> (V, A, D)`. Raises KeyError on unknown labels."""
    return MEAD_VAD[emotion.strip().lower()]


def intensity_to_norm(emotion: str, intensity: int) -> float:
    """`(emotion, MEAD intensity ∈ {1, 2, 3}) -> normalized intensity ∈ [0, 1]`.

    Neutral clips have no emotional intensity, so they map to 0.0
    regardless of the integer value MEAD encodes. Other emotions:
    `i / 3` (i=1 → 0.33, i=2 → 0.67, i=3 → 1.0) — keeps the model's
    `intensity_cond ∈ [0, 1]` knob interpretable as "fraction of peak".
    """
    if emotion.strip().lower() == "neutral":
        return 0.0
    i = max(1, min(3, int(intensity)))
    return i / 3.0


def vad_with_intensity(
    emotion: str, intensity: int,
) -> tuple[float, float, float, float]:
    """`(emotion, intensity) -> (V, A, D, intensity_norm)`.

    The conditioning shape used by the d_vad=4 DiT during training. At
    inference time, callers can override the 4th dim independently
    (`--intensity-cond` knob in 05_sample_smoke.py) to produce
    "fraction of peak" expressions or extrapolate past 1.0.
    """
    v, a, d = emotion_to_vad(emotion)
    return (v, a, d, intensity_to_norm(emotion, intensity))
