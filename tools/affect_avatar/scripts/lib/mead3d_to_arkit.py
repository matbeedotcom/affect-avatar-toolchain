"""Map MEAD_3D's 54-channel output → canonical ARKit-52 order.

MEAD_3D is extracted with MediaPipe FaceLandmarker. The dataset card
documents nothing about channel ordering, but
`tools/affect_avatar/scripts/diag_mead_channels.py` empirically shows:

  - "happy" peaks at idx 43, 44 → must be mouthSmileLeft/Right.
  - "surprised" peaks at idx 2, 3, 4 → browInnerUp, browOuterUpLeft/Right.
  - Indices 5, 6, 7, 49, 50 are constant-zero across emotions →
    cheekPuff, cheekSquintLeft/Right, noseSneerLeft/Right (chronically
    weak MediaPipe outputs).

These match MediaPipe alphabetical order **without** the leading
`_neutral` channel — i.e. 51 named blendshapes at indices 0..50, with
3 trailing unknown extras at 51, 52, 53 (MEAD_3D's pose-or-other
additions; ignored here).

The canonical ARKit-52 ordering (target) lives in
`crates/core/src/nodes/lip_sync/blendshape.rs`. MediaPipe doesn't emit
`tongueOut`, so the ARKit `tongueOut` channel is zero-filled on output.
"""

from __future__ import annotations

import numpy as np

# MediaPipe FaceLandmarker alphabetical names, *without* `_neutral`.
# Indices into this tuple match MEAD_3D's `actions` channel indices
# 0..50. Channels 51, 52, 53 in MEAD_3D are unknown extras (ignored).
MEDIAPIPE_ORDER: tuple[str, ...] = (
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
)
assert len(MEDIAPIPE_ORDER) == 51, f"expected 51, got {len(MEDIAPIPE_ORDER)}"


# Canonical ARKit-52 ordering, mirrored from
# `crates/core/src/nodes/lip_sync/blendshape.rs::ARKIT_BLENDSHAPE_NAMES`.
# Renderer expects this exact order in the JSONL `arkit_52` field.
ARKIT_ORDER: tuple[str, ...] = (
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker",
    "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",
    "tongueOut",
)
assert len(ARKIT_ORDER) == 52, f"expected 52, got {len(ARKIT_ORDER)}"


def build_mead_to_arkit_map() -> tuple[np.ndarray, list[str]]:
    """`(arkit_52_indices, missing_names)`.

    `arkit_52_indices[i]` = the source index into the MEAD_3D 54-channel
    array that should populate `arkit_52[i]`. A value of -1 means
    "no source" — the ARKit channel will be zero (e.g. `tongueOut`,
    which MediaPipe doesn't emit).

    Assumes MEAD_3D's first 52 channels follow MediaPipe's standard
    alphabetical-with-_neutral-prepended order; channels 52 and 53 are
    treated as unknown extras and ignored.
    """
    name_to_mp_idx = {name: i for i, name in enumerate(MEDIAPIPE_ORDER)}
    indices = np.full(52, -1, dtype=np.int32)
    missing: list[str] = []
    for arkit_i, name in enumerate(ARKIT_ORDER):
        mp_idx = name_to_mp_idx.get(name)
        if mp_idx is None:
            missing.append(name)
        else:
            indices[arkit_i] = mp_idx
    return indices, missing


def map_mead_to_arkit(actions: np.ndarray) -> np.ndarray:
    """`(T, 54) MEAD_3D actions -> (T, 52) ARKit-52` reordered.

    Channels that don't correspond to a MediaPipe-named blendshape
    (i.e. `tongueOut`) are zero-filled. The 2 trailing MEAD_3D extras
    (indices 52, 53) are ignored.
    """
    if actions.ndim != 2 or actions.shape[1] != 54:
        raise ValueError(f"expected (T, 54), got {actions.shape}")
    indices, _missing = build_mead_to_arkit_map()
    out = np.zeros((actions.shape[0], 52), dtype=np.float32)
    valid = indices >= 0
    src = indices[valid]
    out[:, valid] = actions[:, src]
    return out


def write_arkit_jsonl(actions_arkit_52: np.ndarray, fps: float, path) -> None:
    """Write `(T, 52) ARKit-52 actions` as the renderer's JSONL format.

    One line per frame: `{"kind": "blendshapes", "arkit_52": [...],
    "pts_ms": int}`, exactly the format `affect_face_smoke` reads.
    """
    import json

    if actions_arkit_52.ndim != 2 or actions_arkit_52.shape[1] != 52:
        raise ValueError(
            f"expected (T, 52), got {actions_arkit_52.shape}"
        )
    dt_ms = 1000.0 / fps
    with open(path, "w") as f:
        for t in range(actions_arkit_52.shape[0]):
            obj = {
                "kind": "blendshapes",
                "arkit_52": [
                    round(float(x), 4) for x in actions_arkit_52[t]
                ],
                "pts_ms": int(round(t * dt_ms)),
            }
            f.write(json.dumps(obj) + "\n")
