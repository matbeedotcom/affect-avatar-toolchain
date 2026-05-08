"""Dataset loaders for the Whisper-side calibration (Workstream D1).

Each dataset has a different on-disk layout (see
docs/references/activation-steering-audio-llm/notes/spike-d-datasets.md):

- EmoVoice-DB (yhaha/EmoVoice-DB) — MIT, synthetic, 7 emotion zips +
  train/val/test JSONL splits. 5 speaker timbres.
- CREMA-D (myleslinder/crema-d) — ODbL, human, 91 actors. HF parquet
  format with legacy crema-d.py script (trust_remote_code or read parquet
  directly).
- JL-Corpus (CLAPv2/JL-Corpus) — CC0, parquet-native, 32 batches keyed by
  <gender>{1,2}_<emotion>_<id>. 4 speakers, ~10 emotions.

Loaders return DatasetClip records uniformly; callers don't need to know
which dataset they're consuming. Per-dataset categorical-emotion-name
spaces differ; the categorical → VAD mapping is a separate concern handled
in lib.emotion_to_vad.

Skeleton for Phase 5 D1.1+. Loader bodies stub to NotImplementedError;
real implementations land alongside the first 01 driver run on real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Protocol


DatasetName = Literal["emovoice_db", "crema_d", "jl_corpus"]
DATASETS: tuple[DatasetName, ...] = ("emovoice_db", "crema_d", "jl_corpus")


@dataclass(frozen=True)
class DatasetClip:
    """One emotion-labeled audio clip from any of the supported datasets.

    `vad_label` is None when the source dataset ships only categorical
    labels (CREMA-D, EmoVoice-DB, JL-Corpus); the categorical → VAD
    mapping is applied later via lib.emotion_to_vad.

    `speaker_id` is required for speaker-disjoint splits in 02
    (PROJECT_PLAN §10 R5). For datasets that don't expose explicit
    speaker IDs, the loader synthesizes a stable id (e.g. "emovoice_spk_3").

    `audio_16k_mono` is an optional in-memory float32 16 kHz mono PCM
    array. HF-datasets-backed loaders populate it directly so the 01
    driver doesn't have to round-trip through the filesystem. When
    populated, `audio_path` is informational only (e.g. the original
    in-archive path).
    """
    clip_id: str
    audio_path: Path
    dataset: DatasetName
    speaker_id: str
    emotion_categorical: str | None
    vad_label: tuple[float, float, float] | None
    audio_16k_mono: Any = field(default=None, compare=False, repr=False)


class DatasetLoader(Protocol):
    def __call__(self, data_dir: Path, max_clips: int | None = None) -> Iterator[DatasetClip]: ...


_EMOVOICE_DB_REPO = "yhaha/EmoVoice-DB"
_CREMA_D_REPO = "myleslinder/crema-d"
_JL_CORPUS_REPO = "CLAPv2/JL-Corpus"


def _audio_to_16k_mono(audio_dict: dict) -> "Any":
    """Convert HF datasets `Audio` feature dict → 16 kHz mono float32 array.

    HF returns {"array": np.ndarray, "sampling_rate": int, "path": str?}.
    We resample as needed using lib.audio's helper (lazy import via the
    same dep stack as 01).
    """
    import numpy as np
    arr = np.asarray(audio_dict["array"], dtype=np.float32)
    sr = int(audio_dict.get("sampling_rate", 0))
    if arr.ndim > 1:
        arr = arr.mean(axis=1).astype(np.float32, copy=False)
    if sr != 16_000:
        import librosa  # lazy
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16_000)
        arr = arr.astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)


def load_jl_corpus(data_dir: Path, max_clips: int | None = None) -> Iterator[DatasetClip]:
    """Iterate JL-Corpus clips (CC0; held-out test set per spike-d).

    Schema (verified 2026-05-05): index, datasetname, audio (44.1 kHz),
    audio_len, text, raw_text. Filename keying lives in `index`:
        `<gender>{1,2}_<emotion>_<id>_<seq>` e.g. `female1_angry_10a_1`.
    Speakers (4 total): female1, female2, male1, male2. Emotions (10):
    angry, anxious, apologetic, assertive, concerned, encouraging,
    excited, happy, neutral, sad.

    Streams the dataset shuffled (buffer=10_000) so that small `max_clips`
    pulls span all 4 speakers — the on-disk shard order is
    speaker-then-emotion sorted, which would otherwise produce
    single-speaker single-emotion samples for any practical cap.

    `data_dir` is informational (HF datasets uses its own cache); kept in
    the signature for uniformity with the disk-layout-based loaders.
    """
    import re
    from datasets import load_dataset  # lazy

    ds = load_dataset(_JL_CORPUS_REPO, split="train", streaming=True)
    ds = ds.shuffle(seed=0, buffer_size=10_000)
    pat = re.compile(r"^(female|male)(\d+)_([a-z]+)_(.+)$")

    n = 0
    for row in ds:
        idx = str(row["index"])
        m = pat.match(idx)
        if not m:
            # Skip rows whose key doesn't match the documented convention;
            # the dataset has been clean to date but be defensive.
            continue
        gender, gnum, emotion, _tail = m.groups()
        speaker_id = f"{gender}{gnum}"
        audio_arr = _audio_to_16k_mono(row["audio"])
        yield DatasetClip(
            clip_id=idx,
            audio_path=Path(f"hf://{_JL_CORPUS_REPO}/{idx}"),
            dataset="jl_corpus",
            speaker_id=speaker_id,
            emotion_categorical=emotion,
            vad_label=None,
            audio_16k_mono=audio_arr,
        )
        n += 1
        if max_clips is not None and n >= max_clips:
            return


def load_crema_d(data_dir: Path, max_clips: int | None = None) -> Iterator[DatasetClip]:
    """Iterate CREMA-D clips from a local extracted tarball (ODbL).

    Expected layout (after one-time extraction):
        data_dir/crema_d/data/AudioWAV/
            ├── 1001_DFA_ANG_XX.wav
            ├── 1001_DFA_DIS_XX.wav
            └── ...

    One-time setup:
        huggingface-cli download myleslinder/crema-d --repo-type dataset \\
            --local-dir tools/affect_calibration/data/crema_d
        cd tools/affect_calibration/data/crema_d
        tar -xzf data/crema_d.tar.gz

    Filename convention: `{actor_id}_{sentence}_{emotion}_{intensity}.wav`
    where emotion ∈ {ANG, DIS, FEA, HAP, NEU, SAD} and actor_id ∈ 1001..1091
    (91 actors). actor_id maps directly to speaker_id for speaker-disjoint
    splits (PROJECT_PLAN R5).

    We bypass `datasets.load_dataset` because CREMA-D's HF loader script
    doesn't support `streaming=True` (the dl_manager returns None paths
    in streaming mode → FileNotFoundError on `None/data/...`). Reading
    from disk directly per spike-d's "avoid trust_remote_code"
    recommendation; this is also faster (no HF cache lookup per row)
    and works offline.
    """
    import random
    import re

    audio_root = data_dir / "crema_d" / "data" / "AudioWAV"
    if not audio_root.exists():
        # Some setups extract directly into data_dir/data/AudioWAV; tolerate that too.
        alt_root = data_dir / "data" / "AudioWAV"
        if alt_root.exists():
            audio_root = alt_root
        else:
            raise FileNotFoundError(
                f"CREMA-D AudioWAV directory not found at {audio_root} or {alt_root}; "
                f"see this loader's docstring for the one-time extraction recipe."
            )

    pat = re.compile(r"^(\d{4})_([A-Z]+)_([A-Z]+)_([A-Z]+)\.wav$")

    wavs = sorted(audio_root.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"no .wav files under {audio_root}")
    # Deterministic shuffle so small max_clips samples span many actors;
    # the on-disk order is actor-then-emotion sorted.
    random.Random(0).shuffle(wavs)

    n = 0
    for path in wavs:
        m = pat.match(path.name)
        if not m:
            continue
        actor_id, _sentence, emotion, _intensity = m.groups()
        yield DatasetClip(
            clip_id=path.stem,
            audio_path=path,
            dataset="crema_d",
            speaker_id=actor_id,
            emotion_categorical=emotion,
            vad_label=None,
            # audio_16k_mono left None — 01's _load_audio_16k_mono will
            # decode from disk via soundfile (CREMA-D is already 16 kHz mono).
        )
        n += 1
        if max_clips is not None and n >= max_clips:
            return


def load_emovoice_db(data_dir: Path, max_clips: int | None = None) -> Iterator[DatasetClip]:
    """Iterate EmoVoice-DB clips (MIT; synthetic, ~22k samples).

    HF dataset `yhaha/EmoVoice-DB` exposes the JSONL splits but ships
    audio in **separate per-emotion zip archives** (audio/{emotion}.zip).
    The `target_wav` field is a path INSIDE the corresponding zip
    (e.g. "audio/angry/gpt4o_16326_angry_ash.wav").

    For Phase 5 v1 we expect `data_dir` to point at a directory
    where the zips have already been extracted into:
        {data_dir}/audio/{emotion}/{key}.wav

    The user should run this once (per spike-d):
        huggingface-cli download yhaha/EmoVoice-DB --repo-type dataset \\
            --local-dir tools/affect_calibration/data/emovoice_db
        cd tools/affect_calibration/data/emovoice_db/audio
        for z in *.zip; do unzip -o "$z" -d "${z%.zip}"; done

    Speaker (timbre) is encoded in the filename suffix (e.g. "_ash" =
    OpenAI 'ash' voice; 5 timbres total: ash/coral/echo/sage/shimmer).
    """
    from datasets import load_dataset  # lazy

    audio_root = data_dir / "audio"
    if not audio_root.exists():
        raise FileNotFoundError(
            f"EmoVoice-DB audio root not found at {audio_root}; "
            f"see this loader's docstring for the one-time extraction recipe."
        )

    ds = load_dataset(_EMOVOICE_DB_REPO, split="train", streaming=True)
    ds = ds.shuffle(seed=0, buffer_size=10_000)

    # Extract speaker (timbre) suffix from key: "gpt4o_16326_angry_ash" → "ash".
    def _speaker_from_key(key: str) -> str:
        parts = key.rsplit("_", 1)
        return parts[1] if len(parts) == 2 else "unknown"

    n = 0
    for row in ds:
        key = str(row["key"])
        emotion = str(row.get("emotion", ""))
        rel = str(row["target_wav"])  # e.g. "audio/angry/gpt4o_16326_angry_ash.wav"
        wav_path = data_dir / rel
        if not wav_path.exists():
            # Skip rows whose audio hasn't been extracted; tolerate partial
            # downloads rather than aborting the whole iteration.
            continue
        yield DatasetClip(
            clip_id=key,
            audio_path=wav_path,
            dataset="emovoice_db",
            speaker_id=_speaker_from_key(key),
            emotion_categorical=emotion or None,
            vad_label=None,
            # audio_16k_mono left None — 01's _load_audio_16k_mono will
            # decode from disk because the file is local already.
        )
        n += 1
        if max_clips is not None and n >= max_clips:
            return


_LOADERS: dict[DatasetName, DatasetLoader] = {
    "emovoice_db": load_emovoice_db,
    "crema_d": load_crema_d,
    "jl_corpus": load_jl_corpus,
}


def get_loader(name: DatasetName) -> DatasetLoader:
    if name not in _LOADERS:
        raise ValueError(f"unknown dataset: {name}; expected one of {DATASETS}")
    return _LOADERS[name]
