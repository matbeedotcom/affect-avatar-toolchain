"""Decode MEAD_3D-style audio bytes → 16 kHz mono numpy.

The dataset stores the audio per row as `{bytes: <encoded payload>,
path: <str|None>}`. The bytes turn out to already be 16-kHz mono WAV
in the BarryFutureman/MEAD_3D parquet, so the decoder is a one-line
`soundfile.read(BytesIO(...))` plus a defensive resample if the
sampling rate doesn't match.
"""

from __future__ import annotations

import io


TARGET_SR = 16_000


def decode_audio_bytes(audio_bytes: bytes) -> tuple["object", int]:
    """`bytes -> (np.ndarray float32 mono, sample_rate)`.

    Resamples to 16 kHz if the source rate differs (uses librosa, lazily
    imported). Mixes stereo to mono via mean of channels.
    """
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(audio_bytes))
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != TARGET_SR:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    return data, sr
