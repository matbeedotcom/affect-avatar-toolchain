"""MediaPipe FaceLandmarker wrapper — extracts ARKit-flavored blendshapes.

Per Spike B0 protocol
(docs/references/audio-blendshape-diffusion/notes/spike-b0-blendshape-extraction.md).

Wraps Google's MediaPipe Tasks `FaceLandmarker` so the rest of the
toolchain sees a simple `extract_clip(video_path) -> (T, K)` surface.
The K dimension is whatever set the FaceLandmarker emits (52 entries
for the standard `face_landmarker.task` model), in the model's native
order; callers do their own remapping into ARKit-52 order if they need
strict ARKit semantics.

Model file:
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Pass `--model-path` to the spike script, or set `MEDIAPIPE_FACE_LANDMARKER`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_MODEL_ENV = "MEDIAPIPE_FACE_LANDMARKER"
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class ClipExtraction:
    """Result of extracting blendshapes from one video clip.

    `blendshapes` is `(T, K)` float32 in `[0, 1]` (rows with no detected
    face are NaN — callers can either drop, interpolate, or treat as
    eyes-closed/mouth-closed depending on intent).

    `names` lists the K column names in column order, exactly as
    MediaPipe returned them on the first successful frame.
    """

    blendshapes: "object"            # np.ndarray (T, K)
    names: list[str]
    n_frames_total: int
    n_frames_with_face: int
    fps: float


class BlendshapeExtractor:
    """Stateful FaceLandmarker bound to a single video stream's timestamps.

    `__init__` loads the MediaPipe model once; reuse the same instance
    across many clips to amortize the model-load cost.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        # Lazy import — the library carries a few hundred MB of TFLite
        # runtime; we don't want it on the import path of, say,
        # `02_train_vae.py` which never extracts.
        import mediapipe as mp  # noqa: F401

        path = model_path or _resolve_model_path()
        if not path.exists():
            raise FileNotFoundError(
                f"FaceLandmarker model not found at {path}. "
                f"Download with: curl -L {MODEL_DOWNLOAD_URL} -o {path}"
            )

        self._model_path = path
        self._mp = mp
        self._landmarker = self._build_landmarker()

    def _build_landmarker(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=str(self._model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
        )
        return mp_vision.FaceLandmarker.create_from_options(options)

    def extract_clip(self, video_path: Path) -> ClipExtraction:
        """Run the model frame-by-frame; return per-frame blendshape rows.

        FaceLandmarker is created in `VIDEO` mode and re-built per clip
        so timestamps reset to zero (the running-mode contract requires
        monotonically increasing timestamps per stream).
        """

        import cv2
        import numpy as np

        # Re-create the landmarker per clip — `detect_for_video` requires
        # monotonically-increasing timestamps within a single instance,
        # and we want each clip to start at t=0.
        self._landmarker.close()
        self._landmarker = self._build_landmarker()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_period_ms = max(1, int(round(1000.0 / fps)))

        rows: list[list[float]] = []
        names: list[str] = []
        n_total = 0
        n_with_face = 0
        timestamp_ms = 0

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                n_total += 1
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = self._mp.Image(
                    image_format=self._mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )
                result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
                timestamp_ms += frame_period_ms

                if not result.face_blendshapes:
                    rows.append([])  # marker for "no face this frame"
                    continue

                cats = result.face_blendshapes[0]
                if not names:
                    names = [c.category_name for c in cats]
                rows.append([float(c.score) for c in cats])
                n_with_face += 1
        finally:
            cap.release()

        if not names:
            # No frame ever produced a detection. Return an empty (T, 0)
            # array; downstream stats can flag the clip as unusable.
            return ClipExtraction(
                blendshapes=np.zeros((n_total, 0), dtype=np.float32),
                names=[],
                n_frames_total=n_total,
                n_frames_with_face=0,
                fps=float(fps),
            )

        K = len(names)
        out = np.full((n_total, K), np.nan, dtype=np.float32)
        for i, row in enumerate(rows):
            if row:
                out[i] = row
        return ClipExtraction(
            blendshapes=out,
            names=names,
            n_frames_total=n_total,
            n_frames_with_face=n_with_face,
            fps=float(fps),
        )

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass


def _resolve_model_path() -> Path:
    env = os.environ.get(DEFAULT_MODEL_ENV)
    if env:
        return Path(env).expanduser().resolve()
    # Sibling to `data/`, alongside other downloaded model artifacts.
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "face_landmarker.task"
    )
