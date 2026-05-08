"""Audio loading helper used by 01_extract_whisper_embeddings.py.

Whisper's encoder expects 16 kHz mono float32 PCM. Most emotion datasets
ship at 16 kHz already (CREMA-D, EmoVoice-DB, JL-Corpus all comply per
spike-d-datasets memo); the resample branch only fires on outlier files.

soundfile is used for raw decode (no codec dependency stack); librosa
handles any required resampling because its `resampy` backend is already
in the project's dep set (see whisper_*.py nodes). Both imports are lazy
so importing this module is cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


TARGET_SR = 16_000


def load_audio_16k_mono(path: Union[str, Path]) -> np.ndarray:
    """Load `path` as a 16 kHz mono float32 1-D array.

    Side-effects:
      - mono-mixes multi-channel inputs by averaging.
      - resamples non-16 kHz inputs via librosa (kaiser_best by default).

    Raises FileNotFoundError if the file is missing, RuntimeError on
    decode failure.
    """
    import soundfile as sf  # lazy

    path = str(path)
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as exc:
        raise RuntimeError(f"audio decode failed for {path}: {exc}") from exc

    if wav.ndim > 1:
        # (T, channels) → mono by mean
        wav = wav.mean(axis=1).astype(np.float32, copy=False)
    else:
        wav = wav.astype(np.float32, copy=False)

    if sr != TARGET_SR:
        # Lazy librosa import — only loaded when resampling is actually needed.
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        wav = wav.astype(np.float32, copy=False)

    return np.ascontiguousarray(wav)
