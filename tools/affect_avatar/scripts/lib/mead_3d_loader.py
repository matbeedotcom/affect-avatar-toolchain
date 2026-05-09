"""Lightweight HF-parquet wrapper for the BarryFutureman/MEAD_3D family.

Reads parquets directly via pyarrow rather than going through
`datasets.load_dataset(...)` — that route streams via the Hub which has
been flaky for us, and the parquets are small enough to keep locally.

Two parquet dialects we encounter, both K=54:

  - **Main parquet** (`BarryFutureman/MEAD_3D`): paths have an actor
    prefix, e.g. ``W009_video/video/front/sad/level_3/021.mp4`` or
    ``video28/video/front/...``.
  - **Per-actor parquets** (``BarryFutureman/MEAD_3D_W040``,
    ``MEAD_3D_M003`` etc.): paths are stripped of the actor prefix
    (``video/front/angry/level_1/001.mp4``) and the actor is implied
    by the dataset name. We pass an `actor_override` to recover it.
    These also carry an extra `text` column with the spoken sentence
    (TIMIT-style); harmless, currently unused.

Use `MeadParquet` for a single source. Use `MultiMeadParquet` to read
from a union of parquets with globally-unique row indices and a
deterministic stable ordering across the union.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence


PARQUET_REL_PATH = "data/train-00000-of-00001.parquet"
HF_REPO = "BarryFutureman/MEAD_3D"
HF_PARQUET_URL = (
    f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{PARQUET_REL_PATH}"
)
ACTION_DIM = 54

# `{actor}_video/video/front/{emotion}/level_{intensity}/{utt}.mp4` (main parquet,
# named-actor variants), or
# `video28/video/front/{emotion}/level_{intensity}/{utt}.mp4` (one main-parquet
# actor that uses the generic-numbered convention), or
# `video_19/video/front/...` (underscored generic actor IDs in the
# multi-shard parquets).
_PATH_RE_WITH_ACTOR = re.compile(
    r"^(?P<actor>[A-Za-z0-9_]+?)(?:_video)?/video/front/"
    r"(?P<emotion>[a-z]+)/level_(?P<intensity>\d+)/(?P<utt>[^/]+)\.[^/.]+$"
)

# `video/front/{emotion}/level_{intensity}/{utt}.mp4` (per-actor parquets — the
# actor is implied by the parquet name, supplied via `actor_override`).
_PATH_RE_NO_ACTOR = re.compile(
    r"^video/front/(?P<emotion>[a-z]+)/level_(?P<intensity>\d+)/(?P<utt>[^/]+)\.[^/.]+$"
)


@dataclass
class ClipMeta:
    """Per-row metadata parsed from `path`.

    `row_idx` is unique within a single source. `MultiMeadParquet`
    re-indexes to globally-unique values across the union.

    `source_id` identifies the parquet the row came from — caller-supplied
    label, defaults to the parquet stem (e.g. ``train-00000-of-00001`` →
    ``MEAD_3D`` if `MultiMeadParquet` is used; otherwise the basename).
    Used to key downstream caches without collision when multiple
    parquets are loaded.
    """

    row_idx: int
    path: str
    actor: str
    emotion: str
    intensity: int
    utt: str
    n_frames: int
    n_audio_samples: int
    sample_rate: int
    source_id: str = ""


def parse_path(path: str, *, actor_override: Optional[str] = None) -> Optional[dict]:
    """Parse a MEAD_3D `path` field. Returns None if neither dialect matches.

    `actor_override` is used when the path lacks the actor prefix (the
    per-actor parquet dialect). When the path *has* an actor prefix,
    the override is ignored — the embedded actor wins.
    """
    m = _PATH_RE_WITH_ACTOR.match(path)
    if m:
        return {
            "actor": m["actor"],
            "emotion": m["emotion"],
            "intensity": int(m["intensity"]),
            "utt": m["utt"],
        }
    m = _PATH_RE_NO_ACTOR.match(path)
    if m and actor_override:
        return {
            "actor": actor_override,
            "emotion": m["emotion"],
            "intensity": int(m["intensity"]),
            "utt": m["utt"],
        }
    return None


class MeadParquet:
    """Read-only wrapper around a single MEAD_3D parquet."""

    def __init__(
        self,
        parquet_path: Path,
        *,
        actor_override: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> None:
        # Lazy import — pyarrow is the only hard dep, but keep it off the
        # module import path so tools without it can still import lib.
        import pyarrow.parquet as pq

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"MEAD_3D parquet not found at {parquet_path}. "
                f"Download from `{HF_REPO}` (or per-actor variant) under "
                f"`/data/train-*.parquet`."
            )
        self._pf = pq.ParquetFile(parquet_path)
        self._path = parquet_path
        self.actor_override = actor_override
        self.source_id = source_id or parquet_path.parent.name or parquet_path.stem

    @property
    def num_rows(self) -> int:
        return int(self._pf.metadata.num_rows)

    @property
    def num_row_groups(self) -> int:
        return int(self._pf.num_row_groups)

    def all_paths(self) -> list[str]:
        """Cheap full-column read: just `path`, ~kilobyte total."""
        return self._pf.read(columns=["path"])["path"].to_pylist()

    def iter_meta(self) -> Iterator[ClipMeta]:
        """Stream per-row metadata without decoding audio/actions payloads."""
        row_idx = 0
        for rg in range(self._pf.num_row_groups):
            tbl = self._pf.read_row_group(rg, columns=["path", "actions", "audio"])
            for i in range(tbl.num_rows):
                p = tbl["path"][i].as_py()
                meta = parse_path(p, actor_override=self.actor_override)
                if meta is None:
                    row_idx += 1
                    continue
                actions = tbl["actions"][i].as_py()
                n_frames = len(actions)
                audio_struct = tbl["audio"][i].as_py()
                n_audio_samples = len(audio_struct.get("bytes") or b"")
                yield ClipMeta(
                    row_idx=row_idx,
                    path=p,
                    actor=meta["actor"],
                    emotion=meta["emotion"],
                    intensity=meta["intensity"],
                    utt=meta["utt"],
                    n_frames=n_frames,
                    n_audio_samples=n_audio_samples,
                    sample_rate=16_000,
                    source_id=self.source_id,
                )
                row_idx += 1

    def iter_actions(
        self, indices: Optional[Iterable[int]] = None
    ) -> Iterator[tuple[int, "np.ndarray"]]:
        """Yield `(row_idx, (T, K) np.float32)` for selected rows.

        If `indices` is None, iterate all rows. Decodes one row group at a
        time — cheap for sequential scans, less so for random access.
        """
        import numpy as np

        wanted = None if indices is None else set(int(i) for i in indices)
        row_idx = 0
        for rg in range(self._pf.num_row_groups):
            tbl = self._pf.read_row_group(rg, columns=["actions"])
            for i in range(tbl.num_rows):
                if wanted is None or row_idx in wanted:
                    arr = np.asarray(tbl["actions"][i].as_py(), dtype=np.float32)
                    yield row_idx, arr
                row_idx += 1


class MultiMeadParquet:
    """Union of multiple `MeadParquet` sources with stable global indexing.

    Sources are iterated in the order supplied. Global `row_idx` is a
    flat 0-based index across the union; the source identifier travels
    on `ClipMeta.source_id` for callers that need to disambiguate
    cache files / per-source counts.

    Constructor takes a list of `(parquet_path, actor_override,
    source_id)` triples; `actor_override` and `source_id` may be None
    to fall back to the path-embedded actor and the parquet's parent
    directory name respectively.
    """

    def __init__(
        self,
        sources: Sequence[tuple[Path, Optional[str], Optional[str]]],
    ) -> None:
        self.parquets: list[MeadParquet] = []
        for path, actor, sid in sources:
            self.parquets.append(
                MeadParquet(path, actor_override=actor, source_id=sid)
            )

    @property
    def num_rows(self) -> int:
        return sum(p.num_rows for p in self.parquets)

    @property
    def num_row_groups(self) -> int:
        return sum(p.num_row_groups for p in self.parquets)

    def all_paths(self) -> list[str]:
        out: list[str] = []
        for pq in self.parquets:
            out.extend(pq.all_paths())
        return out

    def iter_meta(self) -> Iterator[ClipMeta]:
        """Stream per-row metadata across all parquets, with global row_idx."""
        global_idx = 0
        for pq in self.parquets:
            local_to_global: dict[int, int] = {}
            for meta in pq.iter_meta():
                # Re-key row_idx into the global namespace.
                yield ClipMeta(
                    row_idx=global_idx,
                    path=meta.path,
                    actor=meta.actor,
                    emotion=meta.emotion,
                    intensity=meta.intensity,
                    utt=meta.utt,
                    n_frames=meta.n_frames,
                    n_audio_samples=meta.n_audio_samples,
                    sample_rate=meta.sample_rate,
                    source_id=meta.source_id,
                )
                local_to_global[meta.row_idx] = global_idx
                global_idx += 1

    def iter_actions(
        self, indices: Optional[Iterable[int]] = None
    ) -> Iterator[tuple[int, "np.ndarray"]]:
        """Yield `(global_row_idx, (T, K))` for the union, sequential scan.

        Random access via `indices` is supported but pays a full per-source
        scan; if you need it often, build an index first.
        """
        wanted = None if indices is None else set(int(i) for i in indices)
        global_idx = 0
        for pq in self.parquets:
            # Walk both meta + actions in lock-step so we know which
            # local rows the parquet emits (some get filtered by
            # parse_path returning None — those advance row_idx but
            # produce no ClipMeta; iter_actions emits rows regardless).
            for local_idx, arr in pq.iter_actions():
                if wanted is None or global_idx in wanted:
                    yield global_idx, arr
                global_idx += 1


def discover_parquets(data_dir: Path) -> list[tuple[Path, Optional[str], Optional[str]]]:
    """Walk a `data/` directory and return loader-ready triples.

    Layout assumption (matches `huggingface-cli download` defaults):
        data_dir/
          mead_3d/                          # main parquet (W009 + video28)
            train-00000-of-00001.parquet
          mead_3d_W040/                     # per-actor variant
            data/train-00000-of-00001.parquet
            (or just) train-00000-of-00001.parquet
          mead_3d_M003/...

    For each subdirectory whose name starts with `mead_3d`, locates the
    parquet and infers the `actor_override` from the suffix
    (everything after `mead_3d_`). The main parquet's actor is encoded
    in the path field, so its override is None.
    """
    triples: list[tuple[Path, Optional[str], Optional[str]]] = []
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir() or not sub.name.lower().startswith("mead_3d"):
            continue
        parquets = _find_parquets(sub)
        if not parquets:
            continue
        suffix = sub.name[len("mead_3d"):].lstrip("_") or None
        # The "main" parquet has an empty suffix → actor_override=None,
        # so paths' embedded actor (W009 / video28) is used as-is.
        # Multi-actor dialects (`shard_NN`, `unified`) also encode actor
        # in the path string and must NOT use the directory name as override.
        if suffix and (suffix.startswith("shard_") or suffix == "unified"):
            actor_override: Optional[str] = None
        else:
            actor_override = suffix if suffix else None
        # Emit one triple per parquet file. Multi-file shards
        # (`train-00000-of-00002.parquet` + `train-00001-of-00002.parquet`)
        # would otherwise drop half the rows on the floor.
        for parquet in parquets:
            triples.append((parquet, actor_override, sub.name))
    return triples


def _find_parquet(d: Path) -> Optional[Path]:
    """Return the first `*.parquet` at `d` or `d/data/`, else None.

    Kept for backward compatibility; new code should prefer
    `_find_parquets` which returns all matches.
    """
    parquets = _find_parquets(d)
    return parquets[0] if parquets else None


def _find_parquets(d: Path) -> list[Path]:
    """Return every `*.parquet` at `d` or `d/data/`, sorted. The `data/`
    subdir is skipped if `d` itself contains parquets, to avoid emitting
    the same files twice when both layouts coexist."""
    for cand in (d, d / "data"):
        if cand.is_dir():
            files = sorted(cand.glob("*.parquet"))
            if files:
                return files
    return []
