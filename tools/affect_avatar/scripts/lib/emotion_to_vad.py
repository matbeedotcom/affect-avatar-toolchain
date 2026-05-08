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
