#!/usr/bin/env python3
"""Load + validate the BarryFutureman/MEAD_3D parquet, emit a manifest.

Per [PROJECT_PLAN.md §2 Datasets — primary path: pre-extracted MEAD_3D]
(../../docs/references/audio-blendshape-diffusion/PROJECT_PLAN.md). This
replaces the original "01_extract_blendshapes.py" — MEAD_3D ships
already-extracted blendshape trajectories paired with audio, so the
B1 deliverable is a *validation + manifest* step, not an extraction
step.

What this script does:

  1. Confirms the parquet exists (and downloads it if `--download` is
     passed and it's missing).
  2. Reads each row's metadata + parses the `path` field into
     `(actor, emotion, intensity, utt)` tuples.
  3. Walks the `actions` column to summarize per-channel statistics
     across the full corpus (min/max/mean/std per K-channel) — feeds
     into Stage-1 VAE input normalization.
  4. Writes `artifacts/mead_3d_manifest.json` listing every clip's
     metadata + a corpus-level summary block.

Usage:
  ./01_load_mead_3d.py \\
      --parquet data/mead_3d/train-00000-of-00001.parquet \\
      --output artifacts/mead_3d_manifest.json

  ./01_load_mead_3d.py --download \\
      --parquet data/mead_3d/train-00000-of-00001.parquet \\
      --output artifacts/mead_3d_manifest.json
      # Will huggingface-CLI-style download the parquet first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.mead_3d_loader import (  # noqa: E402
    ACTION_DIM,
    HF_PARQUET_URL,
    MeadParquet,
    MultiMeadParquet,
    discover_parquets,
    parse_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--parquet", type=Path, default=None,
        help="Single MEAD_3D parquet to ingest. Mutually exclusive with "
             "--data-dir.",
    )
    p.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory of `mead_3d*` subdirectories (each containing a "
             "parquet). All discovered parquets are unioned, with the "
             "actor inferred from the subdir suffix for per-actor "
             "variants. Mutually exclusive with --parquet.",
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("artifacts/mead_3d_manifest.json"),
        help="Output manifest path.",
    )
    p.add_argument(
        "--download", action="store_true",
        help="If --parquet is given and missing, fetch it from "
             "HuggingFace before running.",
    )
    p.add_argument(
        "--per-channel-stats", action="store_true",
        help="Compute per-channel min/max/mean/std across the full corpus "
             "(adds ~30 s; the VAE training loop wants these).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.parquet and args.data_dir:
        print("error: --parquet and --data-dir are mutually exclusive",
              file=sys.stderr)
        return 2
    if not args.parquet and not args.data_dir:
        # Default: single main parquet under data/mead_3d/.
        args.parquet = Path("data/mead_3d/train-00000-of-00001.parquet")

    if args.data_dir:
        if not args.data_dir.is_dir():
            print(f"error: --data-dir not found at {args.data_dir}",
                  file=sys.stderr)
            return 2
        triples = discover_parquets(args.data_dir)
        if not triples:
            print(f"error: no `mead_3d*` parquets discovered under "
                  f"{args.data_dir}", file=sys.stderr)
            return 2
        print(
            f"discovered {len(triples)} parquet sources under "
            f"{args.data_dir}:",
            file=sys.stderr,
        )
        for path, actor, sid in triples:
            print(f"  {sid:30s}  actor_override={actor!r:>10}  {path}",
                  file=sys.stderr)
        pq = MultiMeadParquet(triples)
        print(
            f"\nunion: rows={pq.num_rows}  row_groups={pq.num_row_groups}",
            file=sys.stderr,
        )
    else:
        if not args.parquet.exists():
            if args.download:
                _download_parquet(args.parquet)
            else:
                print(
                    f"error: parquet not found at {args.parquet}.\n"
                    f"       pass --download or fetch manually:\n"
                    f"       curl -L {HF_PARQUET_URL} -o {args.parquet}",
                    file=sys.stderr,
                )
                return 2

        pq = MeadParquet(args.parquet)
        print(
            f"opened {args.parquet}  rows={pq.num_rows}  "
            f"row_groups={pq.num_row_groups}",
            file=sys.stderr,
        )

    # Pre-flight path-parse check is only meaningful for a single
    # parquet (no actor_override needed). For the union case the
    # actor_override is per-source, so iter_meta() handles it; we skip
    # the pre-check rather than print a misleading "unparseable" count.
    if isinstance(pq, MeadParquet):
        paths = pq.all_paths()
        parsed_count = sum(1 for p in paths if parse_path(p) is not None)
        print(
            f"  {parsed_count}/{len(paths)} paths parse cleanly "
            f"({len(paths) - parsed_count} unparseable)",
            file=sys.stderr,
        )

    print("\nstreaming row metadata...", file=sys.stderr)
    t0 = time.time()
    metas: list[dict] = []
    actor_counts: Counter[str] = Counter()
    emotion_counts: Counter[str] = Counter()
    level_counts: Counter[int] = Counter()
    actor_emotion: dict[tuple[str, str], int] = defaultdict(int)
    n_frames_total = 0
    for m in pq.iter_meta():
        metas.append({
            "row_idx": m.row_idx,
            "path": m.path,
            "actor": m.actor,
            "emotion": m.emotion,
            "intensity": m.intensity,
            "utt": m.utt,
            "n_frames": m.n_frames,
            "sample_rate": m.sample_rate,
        })
        actor_counts[m.actor] += 1
        emotion_counts[m.emotion] += 1
        level_counts[m.intensity] += 1
        actor_emotion[(m.actor, m.emotion)] += 1
        n_frames_total += m.n_frames
        if (m.row_idx + 1) % 200 == 0:
            print(
                f"  [{m.row_idx + 1}/{pq.num_rows}] "
                f"elapsed={time.time() - t0:.1f}s",
                file=sys.stderr,
            )
    print(
        f"  {len(metas)} clips parsed  "
        f"total frames={n_frames_total:,}  "
        f"({time.time() - t0:.1f}s)",
        file=sys.stderr,
    )

    summary: dict = {
        "parquet_path": str(args.parquet),
        "action_dim": ACTION_DIM,
        "n_clips": len(metas),
        "n_frames_total": int(n_frames_total),
        "actors": dict(actor_counts),
        "emotions": dict(emotion_counts),
        "intensities": {str(k): v for k, v in level_counts.items()},
        "actor_x_emotion": {
            f"{a}|{e}": n for (a, e), n in sorted(actor_emotion.items())
        },
    }

    if args.per_channel_stats:
        summary["per_channel_stats"] = _compute_per_channel_stats(pq)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "summary": summary,
        "clips": metas,
    }, indent=2))
    print(f"\nwrote -> {args.output}", file=sys.stderr)

    print("\n=== Summary ===", file=sys.stderr)
    print(f"clips:    {len(metas)}", file=sys.stderr)
    print(f"actors:   {dict(actor_counts)}", file=sys.stderr)
    print(f"emotions: {dict(emotion_counts)}", file=sys.stderr)
    print(f"levels:   {dict(level_counts)}", file=sys.stderr)
    print(f"frames:   {n_frames_total:,}", file=sys.stderr)
    if "per_channel_stats" in summary:
        s = summary["per_channel_stats"]
        print(
            f"channels: K={s['K']}  "
            f"global range=[{s['global_min']:+.3f}, {s['global_max']:+.3f}]  "
            f"mean={s['global_mean']:+.3f}  std={s['global_std']:.3f}",
            file=sys.stderr,
        )

    return 0 if metas else 1


def _download_parquet(target: Path) -> None:
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"downloading {HF_PARQUET_URL} -> {target} (~237 MB)...",
        file=sys.stderr,
    )
    urllib.request.urlretrieve(HF_PARQUET_URL, target)
    print(f"  done; {target.stat().st_size / 1e6:.1f} MB", file=sys.stderr)


def _compute_per_channel_stats(pq: MeadParquet) -> dict:
    """Streamed Welford pass over (T, K) action arrays."""

    # Welford (Wikipedia: online_algorithm) — running mean / M2 per channel.
    n = 0
    K = ACTION_DIM
    mean = np.zeros(K, dtype=np.float64)
    m2 = np.zeros(K, dtype=np.float64)
    cmin = np.full(K, np.inf, dtype=np.float64)
    cmax = np.full(K, -np.inf, dtype=np.float64)

    print("computing per-channel statistics...", file=sys.stderr)
    t0 = time.time()
    for row_idx, arr in pq.iter_actions():
        if arr.shape[1] != K:
            continue
        cmin = np.minimum(cmin, arr.min(axis=0))
        cmax = np.maximum(cmax, arr.max(axis=0))
        # Update running stats with each frame.
        for frame in arr:
            n += 1
            delta = frame - mean
            mean += delta / n
            delta2 = frame - mean
            m2 += delta * delta2
        if (row_idx + 1) % 200 == 0:
            print(
                f"  [{row_idx + 1}/{pq.num_rows}] "
                f"frames seen={n:,}  ({time.time() - t0:.1f}s)",
                file=sys.stderr,
            )
    var = m2 / max(n - 1, 1)
    std = np.sqrt(var)
    print(f"  done; {n:,} frames  ({time.time() - t0:.1f}s)", file=sys.stderr)
    return {
        "K": K,
        "n_frames": int(n),
        "global_min": float(cmin.min()),
        "global_max": float(cmax.max()),
        "global_mean": float(mean.mean()),
        "global_std": float(std.mean()),
        "per_channel": [
            {
                "k": int(k),
                "min": float(cmin[k]),
                "max": float(cmax[k]),
                "mean": float(mean[k]),
                "std": float(std[k]),
            }
            for k in range(K)
        ],
    }


if __name__ == "__main__":
    sys.exit(main())
