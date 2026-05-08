#!/usr/bin/env python3
"""Extract Whisper-encoder embeddings for an emotion-labeled audio dataset.

Per IMPLEMENTATION_PLAN §4.2 + Spike C memo
(docs/references/activation-steering-audio-llm/notes/spike-c-whisper-encoder.md).
For each clip in the dataset:

    audio (16kHz mono) → log-mel → Whisper encoder → layer L hidden states
    → mean-pool over time frames → [d_whisper] vector

Outputs an .npz suitable for consumption by 02_train_vad_regressor.py.

Usage:
  ./01_extract_whisper_embeddings.py \\
      --dataset crema_d \\
      --data-dir data/crema_d \\
      --whisper-model openai/whisper-large-v3-turbo \\
      --layer -1 \\
      --output data/crema_d/whisper_embeds.npz

  ./01_extract_whisper_embeddings.py \\
      --dataset crema_d --data-dir data/crema_d \\
      --dry-run --d-whisper 1280 --n-clips 200 \\
      --output /tmp/dryrun_embeds.npz
      # Synthesizes structurally valid embeddings without touching audio
      # or Whisper; useful for wiring 02_train_vad_regressor.py before
      # the dataset loaders + Whisper provider land.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Protocol

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.datasets import (  # noqa: E402
    DATASETS,
    DatasetClip,
    DatasetName,
    get_loader,
)
from lib.emotion_to_vad import known_labels  # noqa: E402


# Per Spike C memo: d_whisper varies by model size.
WHISPER_D_BY_SIZE: dict[str, int] = {
    "openai/whisper-tiny": 384,
    "openai/whisper-base": 512,
    "openai/whisper-small": 768,
    "openai/whisper-medium": 1024,
    "openai/whisper-large-v3": 1280,
    "openai/whisper-large-v3-turbo": 1280,
}


class WhisperEncoder(Protocol):
    """Encoder-only Whisper wrapper (HF AutoModelForSpeechSeq2Seq.get_encoder()).

    See spike-c-whisper-encoder.py for the working extraction snippet.
    """
    d_whisper: int

    def encode(self, audio_16k_mono: np.ndarray, layer: int) -> np.ndarray:
        """Return mean-pooled hidden state at `layer` for one clip.

        Input: audio_16k_mono float32 array, sample rate 16000 Hz, mono.
        Output: [d_whisper] float32 vector (mean over time frames).
        """
        ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        required=True,
        help="Which dataset loader to dispatch on.",
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to the dataset's local root.")
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="HF model id for the Whisper encoder.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Encoder layer to pool. -1 = final (default per Spike C). Phase 5 may sweep.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npz path. Stored arrays: embeddings, clip_ids, speaker_ids, emotion_categoricals, vad_labels.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Cap clip count (debugging / progressive extraction).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Synthesize embeddings without loading audio or Whisper. Requires --d-whisper and --n-clips.",
    )
    parser.add_argument(
        "--d-whisper",
        type=int,
        default=None,
        help="Embedding dim for --dry-run (look up via WHISPER_D_BY_SIZE for real-mode sanity).",
    )
    parser.add_argument(
        "--n-clips",
        type=int,
        default=200,
        help="Synthetic clip count for --dry-run.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _dry_run_clips(dataset: DatasetName, n: int) -> list[DatasetClip]:
    """Synthesize n DatasetClip records for --dry-run.

    Emits emotion labels drawn from the dataset's actual vocabulary (via
    emotion_to_vad.known_labels) so the dry-run mirrors the real loader
    contract — CREMA-D uppercase codes vs EmoVoice-DB / JL-Corpus
    lowercase names. Without this, 02's categorical → VAD lookup fails
    with UnknownEmotionError on the dry-run output.
    """
    emotions = known_labels(dataset)
    n_speakers = 8
    return [
        DatasetClip(
            clip_id=f"dryrun_{dataset}_{i:05d}",
            audio_path=Path(f"/dev/null/{dataset}/{i}.wav"),  # never read
            dataset=dataset,
            speaker_id=f"spk_{i % n_speakers:02d}",
            emotion_categorical=emotions[i % len(emotions)],
            vad_label=None,
        )
        for i in range(n)
    ]


def _save(output: Path, clips: list[DatasetClip], embeddings: np.ndarray, meta: dict) -> None:
    """Save NPZ with stacked arrays for fast 02-side loading."""
    n = len(clips)
    assert embeddings.shape == (n, embeddings.shape[1]), f"embeddings shape mismatch: {embeddings.shape}"

    clip_ids = np.array([c.clip_id for c in clips], dtype=object)
    speaker_ids = np.array([c.speaker_id for c in clips], dtype=object)
    emotion_categoricals = np.array(
        [c.emotion_categorical or "" for c in clips], dtype=object
    )
    vad_labels = np.full((n, 3), np.nan, dtype=np.float32)
    for i, c in enumerate(clips):
        if c.vad_label is not None:
            vad_labels[i] = c.vad_label

    # Single dataset per .npz file by construction (01 dispatches one
    # loader per run); stamp it explicitly so 02 doesn't have to guess
    # from clip_id substrings (most loaders' clip IDs don't contain the
    # dataset name — e.g. JL-Corpus emits 'female1_angry_10a_1').
    dataset_name = clips[0].dataset if clips else ""

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        embeddings=embeddings.astype(np.float32),
        clip_ids=clip_ids,
        speaker_ids=speaker_ids,
        emotion_categoricals=emotion_categoricals,
        vad_labels=vad_labels,
        whisper_model=np.array(meta["whisper_model"], dtype=object),
        layer=np.int32(meta["layer"]),
        d_whisper=np.int32(embeddings.shape[1]),
        dataset=np.array(dataset_name, dtype=object),
    )


def main() -> int:
    args = parse_args()

    if args.dry_run:
        if args.d_whisper is None:
            print("error: --d-whisper required for --dry-run", file=sys.stderr)
            return 2
        clips = _dry_run_clips(args.dataset, args.n_clips)
        rng = np.random.default_rng(args.seed)
        embeddings = rng.standard_normal((len(clips), args.d_whisper), dtype=np.float32)
        d_whisper = args.d_whisper
    else:
        if not args.data_dir.exists():
            print(f"error: data dir not found at {args.data_dir}", file=sys.stderr)
            return 2

        loader = get_loader(args.dataset)
        clips = list(loader(args.data_dir, max_clips=args.max_clips))
        if not clips:
            print(f"error: loader returned 0 clips for {args.dataset} at {args.data_dir}", file=sys.stderr)
            return 2

        encoder = _build_whisper_encoder(args.whisper_model)
        d_whisper = encoder.d_whisper
        print(f"loaded encoder: {args.whisper_model}  d_whisper={d_whisper}", file=sys.stderr)

        embeddings = np.zeros((len(clips), d_whisper), dtype=np.float32)
        # Checkpoint frequency: write a partial NPZ every CHECKPOINT_EVERY
        # clips so a kill / crash mid-run doesn't lose the whole batch.
        # Whisper-large-v3-turbo on CPU is ~12 s/clip; even on MPS we're
        # in the seconds-per-clip range, so 2000 clips is a meaningful
        # wall-clock window worth checkpointing.
        CHECKPOINT_EVERY = 200
        import time
        t0 = time.time()
        for i, clip in enumerate(clips):
            audio = clip.audio_16k_mono
            if audio is None:
                audio = _load_audio_16k_mono(clip.audio_path)
            embeddings[i] = encoder.encode(audio, layer=args.layer)
            done = i + 1
            if done % 100 == 0:
                rate = done / max(1e-6, time.time() - t0)
                eta = (len(clips) - done) / rate
                print(
                    f"  {done}/{len(clips)}  ({rate:.2f} clips/s, ETA {eta/60:.1f} min)",
                    file=sys.stderr,
                )
            if done % CHECKPOINT_EVERY == 0 and done < len(clips):
                _save(
                    args.output, clips[:done], embeddings[:done],
                    meta={"whisper_model": args.whisper_model, "layer": args.layer},
                )
                print(f"  checkpoint -> {args.output}  ({done} clips)", file=sys.stderr)

    print(
        f"  dataset={args.dataset}  n_clips={len(clips)}  d_whisper={d_whisper}  layer={args.layer}",
        file=sys.stderr,
    )
    _save(
        args.output,
        clips,
        embeddings,
        meta={"whisper_model": args.whisper_model, "layer": args.layer},
    )
    print(f"wrote {len(clips)} embeddings -> {args.output}", file=sys.stderr)
    return 0


def _build_whisper_encoder(model_id: str) -> WhisperEncoder:
    """Construct a WhisperEncoder over the configured HF model."""
    from lib.whisper_encoder import WhisperEncoderProvider  # noqa: E402  (lazy)
    return WhisperEncoderProvider(model_id)


def _load_audio_16k_mono(path: Path) -> np.ndarray:
    """Load a clip as a 16kHz mono float32 array."""
    from lib.audio import load_audio_16k_mono  # noqa: E402  (lazy)
    return load_audio_16k_mono(path)


if __name__ == "__main__":
    sys.exit(main())
