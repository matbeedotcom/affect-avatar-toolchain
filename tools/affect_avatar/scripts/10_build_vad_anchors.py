#!/usr/bin/env python3
"""Build per-(emotion, intensity) blendshape anchors from MEAD.

Per [INTEGRATION.md §6](../INTEGRATION.md): the listener-face
pipeline uses a 21-anchor RBF lookup over `(V, A, D, intensity)` →
54-channel MEAD blendshape, derived from MEAD by:

  1. Group MEAD clips by `(emotion, intensity)`.
  2. Per clip: take max-over-time (peak expression).
  3. Mean across clips of each group → per-(emotion, intensity)
     "peak" anchor.
  4. Subtract the neutral L1 peak as a baseline → emotion-delta.
     Removes systematic boundary-frame artefacts where MediaPipe
     misclassifies closed mouths as `mouthPucker` etc.
  5. Clip negative deltas to 0 — listener-mode anchors are pure
     positive emotion deltas, applied additively over a chosen
     baseline (or alone, for the cleanest rest pose).

Outputs `artifacts/vad_anchors_delta.npz` with keys:
  - `baseline`: (54,) neutral L1 peak.
  - `<emotion>_<intensity>`: (54,) emotion delta from baseline,
    clipped to [0, 1]. e.g. `happy_3`, `surprised_2`.

Usage:
    ./10_build_vad_anchors.py --data-dir data \\
        --output artifacts/vad_anchors_delta.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.mead_3d_loader import (  # noqa: E402
    MeadParquet, discover_parquets, parse_path,
)


_VALID_EMOTIONS = (
    "neutral", "happy", "sad", "angry", "fear",
    "disgusted", "surprised", "contempt",
)
_VALID_INTENSITIES = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Directory of `mead_3d*` subdirectories (per the loader's "
             "`discover_parquets` convention).",
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("artifacts/vad_anchors_delta.npz"),
        help="Output `.npz`. Default lives next to the other Stage-1 "
             "artefacts so the listener pipeline can find it.",
    )
    p.add_argument(
        "--also-save-peak", action="store_true",
        help="Also save the pre-subtraction peak anchors as "
             "`vad_anchors_peak.npz` (sibling of --output). Useful for "
             "debugging; not required for inference.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_dir.exists():
        raise SystemExit(f"data-dir not found: {args.data_dir}")

    triples = discover_parquets(args.data_dir)
    if not triples:
        raise SystemExit(f"no `mead_3d*` parquets discovered under {args.data_dir}")

    sums: dict[tuple[str, int], np.ndarray] = {}
    counts: dict[tuple[str, int], int] = {}
    seen: set = set()

    t0 = time.time()
    print(f"scanning {len(triples)} parquets...", file=sys.stderr)
    for parquet_path, actor_override, sid in triples:
        pq = MeadParquet(parquet_path, actor_override=actor_override, source_id=sid)
        for rg in range(pq.num_row_groups):
            tbl = pq._pf.read_row_group(rg, columns=["path", "actions"])
            for i in range(tbl.num_rows):
                path = tbl["path"][i].as_py()
                meta = parse_path(path, actor_override=actor_override) or {}
                emo = meta.get("emotion", "")
                lvl = int(meta.get("intensity", 0))
                if emo not in _VALID_EMOTIONS or lvl not in _VALID_INTENSITIES:
                    continue
                # Dedupe by (actor, emotion, intensity, utt) — the same
                # clip can appear in multiple shards.
                key_dedup = (
                    meta.get("actor", ""), emo, lvl, meta.get("utt", ""),
                )
                if key_dedup in seen:
                    continue
                seen.add(key_dedup)
                actions = np.asarray(
                    tbl["actions"][i].as_py(), dtype=np.float32,
                )
                if actions.shape[0] < 8 or actions.shape[1] != 54:
                    continue
                clip_peak = actions.max(axis=0)
                k = (emo, lvl)
                sums[k] = sums.get(k, np.zeros(54, dtype=np.float64)) + clip_peak
                counts[k] = counts.get(k, 0) + 1

    if (("neutral", 1)) not in sums:
        raise SystemExit("no neutral L1 clips found — cannot derive baseline")

    peak_anchors: dict[tuple[str, int], np.ndarray] = {
        k: (sums[k] / counts[k]).astype(np.float32)
        for k in sums if counts[k] > 0
    }
    print(f"  built {len(peak_anchors)} peak-anchors "
          f"({time.time() - t0:.1f}s)", file=sys.stderr)
    print(f"  per-anchor clip count: min="
          f"{min(counts.values())} max={max(counts.values())}",
          file=sys.stderr)

    baseline = peak_anchors[("neutral", 1)].copy()
    deltas: dict[tuple[str, int], np.ndarray] = {
        k: np.clip(v - baseline, 0.0, 1.0).astype(np.float32)
        for k, v in peak_anchors.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        baseline=baseline,
        **{f"{e}_{l}": v for (e, l), v in deltas.items()},
    )
    print(f"\nwrote {args.output}", file=sys.stderr)

    if args.also_save_peak:
        peak_path = args.output.with_name(
            args.output.stem.replace("_delta", "") + "_peak"
            + args.output.suffix
        )
        np.savez(peak_path, **{f"{e}_{l}": v for (e, l), v in peak_anchors.items()})
        print(f"wrote {peak_path}", file=sys.stderr)

    # Print a sanity table.
    print(f"\ndelta-anchor signatures (top-3 mapped channels):", file=sys.stderr)
    from lib.mead3d_to_arkit import MEDIAPIPE_ORDER
    for k in sorted(deltas.keys()):
        if k == ("neutral", 1):
            continue
        a = deltas[k][:51]
        top = np.argsort(a)[::-1][:3]
        sig = ", ".join(f"{MEDIAPIPE_ORDER[t]}={a[t]:+.2f}" for t in top)
        print(f"  {k[0]:10s} L{k[1]} → {sig}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
