"""Categorical-emotion → VAD coordinate mapping.

Each calibration dataset (EmoVoice-DB, CREMA-D, JL-Corpus) ships only
categorical emotion labels. The Whisper-side regressor (02) targets a
continuous Valence/Arousal/Dominance vector, so we project categorical
labels into VAD using a fixed lookup before training.

Coordinate convention: each axis is normalized to [-1, +1] to match
steering-coefficient α conventions (see persona_pipeline.AXES). Values
are derived from Russell's circumplex (V/A) and Mehrabian's PAD model
extended with NRC-VAD-style empirical anchors. Treat them as priors,
not ground truth — Phase 5 may revise per dataset based on annotator
spot-checks.

Per-dataset alias tables map the on-disk label vocabulary onto a
canonical key in CANONICAL_VAD. CREMA-D ships 3-letter codes
(ANG/DIS/FEA/HAP/NEU/SAD) which alias onto the same anchors as
EmoVoice-DB's spelled-out forms. JL-Corpus has additional categories
(anxious, apologetic, assertive, concerned, encouraging, excited)
not present in the other two datasets.

Skeleton for Phase 5 D1.1+. The mapping table is authored by hand; if
Phase 5 spot-checks reveal mismatches, edit CANONICAL_VAD directly
rather than introducing per-dataset overrides.
"""

from __future__ import annotations

from typing import Mapping

from .datasets import DatasetName


# Canonical anchor table. Keys are normalized lowercase emotion names.
# Coordinates are (valence, arousal, dominance) ∈ [-1, +1]^3.
CANONICAL_VAD: Mapping[str, tuple[float, float, float]] = {
    # Shared across EmoVoice-DB / CREMA-D / JL-Corpus
    "happy":       (+0.80, +0.50, +0.40),
    "sad":         (-0.70, -0.30, -0.40),
    "angry":       (-0.50, +0.70, +0.50),
    "fearful":     (-0.60, +0.60, -0.50),
    "disgusted":   (-0.60, +0.30, -0.20),
    "neutral":     ( 0.00,  0.00,  0.00),
    "surprised":   (+0.30, +0.70,  0.00),
    # JL-Corpus extras
    "anxious":     (-0.50, +0.50, -0.40),
    "apologetic":  (-0.40, -0.20, -0.40),
    "assertive":   (+0.20, +0.40, +0.70),
    "concerned":   (-0.30, +0.20, -0.10),
    "encouraging": (+0.60, +0.30, +0.40),
    "excited":     (+0.70, +0.70, +0.40),
}


# Per-dataset label-vocabulary aliases onto CANONICAL_VAD keys.
_ALIASES: dict[DatasetName, Mapping[str, str]] = {
    "emovoice_db": {
        "angry":     "angry",
        "disgusted": "disgusted",
        "fearful":   "fearful",
        "happy":     "happy",
        "neutral":   "neutral",
        "sad":       "sad",
        "surprised": "surprised",
    },
    "crema_d": {
        "ANG": "angry",
        "DIS": "disgusted",
        "FEA": "fearful",
        "HAP": "happy",
        "NEU": "neutral",
        "SAD": "sad",
    },
    "jl_corpus": {
        "angry":       "angry",
        "anxious":     "anxious",
        "apologetic":  "apologetic",
        "assertive":   "assertive",
        "concerned":   "concerned",
        "encouraging": "encouraging",
        "excited":     "excited",
        "happy":       "happy",
        "neutral":     "neutral",
        "sad":         "sad",
    },
}


class UnknownEmotionError(KeyError):
    """Raised when a dataset emits a categorical label outside its alias table."""


def categorical_to_vad(dataset: DatasetName, label: str) -> tuple[float, float, float]:
    """Map a dataset-specific categorical label to (V, A, D) ∈ [-1, +1]^3.

    Raises UnknownEmotionError if the label is not in the dataset's alias
    table (typo, schema drift, or a new emotion that needs an entry in
    CANONICAL_VAD + _ALIASES).
    """
    aliases = _ALIASES.get(dataset)
    if aliases is None:
        raise UnknownEmotionError(f"no alias table for dataset '{dataset}'")
    canonical = aliases.get(label)
    if canonical is None:
        raise UnknownEmotionError(
            f"dataset '{dataset}' has no mapping for label '{label}'; "
            f"expected one of {sorted(aliases.keys())}"
        )
    return CANONICAL_VAD[canonical]


def known_labels(dataset: DatasetName) -> tuple[str, ...]:
    """Return the sorted tuple of categorical labels recognized for `dataset`."""
    aliases = _ALIASES.get(dataset)
    if aliases is None:
        raise UnknownEmotionError(f"no alias table for dataset '{dataset}'")
    return tuple(sorted(aliases.keys()))
