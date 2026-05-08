#!/usr/bin/env python3
"""Spike C — Whisper encoder hidden-state extraction (HF transformers path).

Companion to ``spike-c-whisper-encoder.md``. Demonstrates the minimum viable
call sequence for accessing Whisper's encoder ``last_hidden_state`` while
keeping the existing transcription path unchanged.

This is a Phase 0 throwaway. It is not committed as a unit-tested module
and not invoked from any pipeline. Its purpose is to:

  1. Document the exact API call sequence for Phase 3 integration.
  2. Confirm that ``AutoModelForSpeechSeq2Seq`` and the ``pipeline()``
     wrapper can coexist on the same loaded weights.
  3. Record output shape (T_frames, d_whisper) per model size.

Usage::

    python3 spike-c-whisper-encoder.py /path/to/sample.wav

If no path is supplied, generates a 5-second 16 kHz silence buffer.

Requirements (NOT pinned in this script — Phase 3 integration uses the
project's existing dependency set per
``clients/python/remotemedia/nodes/ml/whisper_*.py``)::

    transformers >= 4.40
    torch >= 2.1
    librosa >= 0.10  (or soundfile)
    numpy

Skipped intentionally:

  - GPU/MPS device handling (Phase 3 inherits the existing nodes' device
    selection logic).
  - Multi-language config (Phase 3 inherits the existing nodes' language
    config).
  - Streaming audio chunking (the existing transcription path already
    handles chunking; Spike C only validates one chunk end-to-end).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _load_audio(path: Path | None) -> tuple[np.ndarray, int]:
    """Return (waveform, sample_rate). Falls back to 5s silence if path is None."""
    if path is None:
        sr = 16_000
        return np.zeros(sr * 5, dtype=np.float32), sr

    try:
        import librosa
        wav, sr = librosa.load(str(path), sr=16_000, mono=True)
        return wav.astype(np.float32), sr
    except ImportError:
        import soundfile as sf
        wav, sr = sf.read(str(path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16_000:
            raise RuntimeError(
                f"Audio sr={sr}; install librosa or pre-resample to 16 kHz."
            )
        return wav, sr


def _model_sizes() -> list[tuple[str, str]]:
    """Tuples of (label, HF model id) to test, in increasing size order.

    The plan calls for testing tiny, small, and large-v3-turbo. Adjust as
    needed if any are not locally cached and the network is slow.
    """
    return [
        ("tiny", "openai/whisper-tiny"),
        ("small", "openai/whisper-small"),
        ("large-v3-turbo", "openai/whisper-large-v3-turbo"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio",
        nargs="?",
        type=Path,
        default=None,
        help="Optional path to a .wav file. If omitted, uses 5s of silence.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Only test the given size label (tiny/small/large-v3-turbo).",
    )
    args = parser.parse_args()

    # Late import so that running the script with --help does not require torch.
    import torch
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        pipeline as hf_pipeline,
    )

    waveform, sr = _load_audio(args.audio)
    print(f"audio: {len(waveform)/sr:.2f}s @ {sr} Hz  (path={args.audio or 'silence'})")
    print()

    sizes = _model_sizes()
    if args.only:
        sizes = [pair for pair in sizes if pair[0] == args.only]
        if not sizes:
            print(f"Unknown size '{args.only}'. Choices: {[p[0] for p in _model_sizes()]}",
                  file=sys.stderr)
            return 2

    for label, model_id in sizes:
        print(f"=== {label}  ({model_id}) ===")
        t0 = time.perf_counter()
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
            model.eval()
        except Exception as e:
            print(f"  load FAILED: {e!r}")
            print()
            continue
        t_load = time.perf_counter() - t0
        print(f"  loaded in {t_load:.2f}s")

        # Step 1 — encoder hidden-state extraction (Phase 3 integration target).
        # The processor produces log-mel features expected by the encoder.
        inputs = processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
        )
        # Ensure features land on the same device as the model (CPU here).
        input_features = inputs["input_features"]

        with torch.inference_mode():
            t0 = time.perf_counter()
            encoder = model.get_encoder()
            encoder_out = encoder(input_features, return_dict=True)
            t_encode = time.perf_counter() - t0

        last_hidden = encoder_out.last_hidden_state  # (B, T_frames, d_whisper)
        T_frames, d_whisper = last_hidden.shape[1], last_hidden.shape[2]
        print(
            f"  encoder.last_hidden_state: shape={tuple(last_hidden.shape)} "
            f"dtype={last_hidden.dtype}  in {t_encode*1000:.1f}ms"
        )
        print(f"  -> T_frames={T_frames}  d_whisper={d_whisper}")

        # Step 2 — verify the existing transcription path still works
        # alongside the encoder call (Phase 3 will run both in one node).
        try:
            t0 = time.perf_counter()
            asr = hf_pipeline(
                task="automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch.float32,
                device=-1,                # CPU; existing nodes pick GPU/MPS at runtime
            )
            transcript = asr(
                {"raw": waveform, "sampling_rate": sr},
                return_timestamps=False,
                generate_kwargs={"max_new_tokens": 64},
            )["text"]
            t_asr = time.perf_counter() - t0
            print(f"  pipeline transcript ({t_asr:.2f}s): {transcript!r}")
        except Exception as e:
            print(f"  pipeline transcription FAILED: {e!r}")

        # Step 3 — show how to mean-pool to a (d_whisper,) vector for the
        # downstream prosody-VAD regressor (Phase 3 uses the planned
        # MeanPoolNode in Rust; here we mean-pool inline for demonstration).
        pooled = last_hidden.mean(dim=1).squeeze(0).cpu().numpy()  # (d_whisper,)
        print(f"  mean-pooled vector: shape={pooled.shape}  dtype={pooled.dtype}")
        print(f"    norm={np.linalg.norm(pooled):.3f}  "
              f"mean={pooled.mean():.4f}  std={pooled.std():.4f}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
