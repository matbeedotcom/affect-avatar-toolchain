#!/usr/bin/env python3
"""Identify MEAD_3D's 54 channel slots by emotion-correlation signature.

Strategy: each emotion has known facial signatures (happy=smile,
sad=mouth-frown + brow-inner-up, surprised=eye-wide + jaw-open). Pool
each clip's actions over time, then for each channel measure its mean
activation per emotion. The channel that fires highest for happy is the
mouth-smile channel; for surprised, eye-wide; etc.

Cross-reference against the standard MediaPipe FaceLandmarker order
(`MEDIAPIPE_ORDER` in `lib/mead3d_to_arkit.py`) to verify whether the
hard-coded mapping is correct, or to author a corrected mapping.

Usage:
  ./diag_mead_channels.py --data-dir data --output artifacts/mead_channel_diag.md
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.mead_3d_loader import (  # noqa: E402
    MeadParquet, discover_parquets, parse_path,
)
from lib.mead3d_to_arkit import MEDIAPIPE_ORDER  # noqa: E402

# Expected emotion -> high-activation channel(s) mapping. Names from
# the standard MediaPipe FaceLandmarker output. If the diagnostic
# disagrees with these expectations, the assumed channel order is
# wrong.
EXPECTED_TOP_CHANNELS: dict[str, list[str]] = {
    "happy": ["mouthSmileLeft", "mouthSmileRight", "cheekSquintLeft", "cheekSquintRight"],
    "sad": ["mouthFrownLeft", "mouthFrownRight", "browInnerUp"],
    "angry": ["browDownLeft", "browDownRight"],
    "surprised": ["eyeWideLeft", "eyeWideRight", "jawOpen", "browInnerUp"],
    "fear": ["eyeWideLeft", "eyeWideRight", "browInnerUp"],
    "disgusted": ["noseSneerLeft", "noseSneerRight", "mouthShrugUpper"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--clips-per-emotion", type=int, default=50)
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/mead_channel_diag.md"))
    return p.parse_args()


def load_pooled_per_emotion(
    data_dir: Path, n_per: int,
) -> dict[str, list[np.ndarray]]:
    triples = discover_parquets(data_dir)
    out: dict[str, list[np.ndarray]] = defaultdict(list)
    seen = set()
    for path, actor_override, sid in triples:
        pq = MeadParquet(path, actor_override=actor_override, source_id=sid)
        cur = 0
        for rg in range(pq.num_row_groups):
            tbl = pq._pf.read_row_group(rg, columns=["path", "actions"])
            for i in range(tbl.num_rows):
                p = tbl["path"][i].as_py()
                meta = parse_path(p, actor_override=actor_override) or {}
                emo = meta.get("emotion", "")
                if not emo:
                    continue
                if len(out[emo]) >= n_per:
                    continue
                key = (meta.get("actor", ""), emo, meta.get("intensity", 0),
                       meta.get("utt", ""))
                if key in seen:
                    continue
                seen.add(key)
                actions = np.asarray(tbl["actions"][i].as_py(), dtype=np.float32)
                if actions.shape[0] < 16 or actions.shape[1] != 54:
                    continue
                out[emo].append(actions.mean(axis=0))
            cur += tbl.num_rows
        # Stop when we have enough clips for every emotion.
        if all(len(v) >= n_per for v in out.values()) and len(out) >= 6:
            break
    return out


def main() -> int:
    args = parse_args()
    print(f"loading clips from {args.data_dir} ({args.clips_per_emotion} per emotion)...",
          file=sys.stderr)
    t0 = time.time()
    pooled = load_pooled_per_emotion(args.data_dir, args.clips_per_emotion)
    print(f"  loaded ({time.time() - t0:.1f}s); per-emotion counts: "
          f"{ {k: len(v) for k, v in pooled.items()} }",
          file=sys.stderr)

    if not pooled:
        print("no clips loaded", file=sys.stderr)
        return 1

    # (emotion -> 54-D mean activation)
    emo_means = {
        emo: np.stack(arrs).mean(axis=0) if arrs else np.zeros(54)
        for emo, arrs in pooled.items()
    }
    neutral = emo_means.get("neutral")
    if neutral is None:
        print("no neutral baseline", file=sys.stderr)
        return 1

    lines: list[str] = []
    lines.append("# MEAD_3D channel diagnostic\n\n")
    lines.append(f"- Clips per emotion: {args.clips_per_emotion}\n")
    lines.append(f"- Channels: 54 (MEAD_3D actions)\n")
    lines.append(f"- Reference order assumed: MediaPipe FaceLandmarker "
                 f"alphabetical (52 names, _neutral at idx 0).\n\n")

    lines.append("## Per-channel emotion-vs-neutral delta\n\n")
    lines.append("For each emotion we compute "
                 "`channel_activation[emotion] - channel_activation[neutral]`. "
                 "The top-K channels per emotion are the ones whose firing is "
                 "most distinct from neutral.\n\n")

    # For each emotion, list top-5 channels by delta vs neutral.
    for emo in sorted(emo_means):
        if emo == "neutral":
            continue
        delta = emo_means[emo] - neutral
        top = np.argsort(delta)[::-1][:8]
        lines.append(f"### {emo}\n\n")
        lines.append("| rank | mead_idx | mediapipe_name (assumed) | "
                     "Δ from neutral | mean[emo] | mean[neutral] |\n"
                     "|---:|---:|---|---:|---:|---:|\n")
        for rank, idx in enumerate(top):
            mp_name = (
                MEDIAPIPE_ORDER[idx] if idx < len(MEDIAPIPE_ORDER) else f"(unknown #{idx})"
            )
            lines.append(
                f"| {rank + 1} | {idx} | {mp_name} | "
                f"{delta[idx]:+.4f} | {emo_means[emo][idx]:.4f} | "
                f"{neutral[idx]:.4f} |\n"
            )
        # Verdict: do the top-K channels match expectations?
        expected = EXPECTED_TOP_CHANNELS.get(emo)
        if expected:
            expected_indices = {
                MEDIAPIPE_ORDER.index(name) if name in MEDIAPIPE_ORDER else -1
                for name in expected
            }
            top_set = set(int(i) for i in top[:6])
            hits = top_set & expected_indices
            lines.append(f"\n*Expected high channels for {emo}*: "
                         f"{', '.join(expected)} "
                         f"→ MediaPipe indices {sorted(int(i) for i in expected_indices if i >= 0)}\n")
            lines.append(f"*Hit*: {len(hits)} of {len(expected_indices)} expected "
                         f"indices appear in top-6 actual deltas.\n\n")

    lines.append("\n## Verdict\n\n")
    lines.append("If `Hit` ≥ half the expected channels per emotion, the "
                 "MediaPipe-alphabetical mapping in `lib/mead3d_to_arkit.py` "
                 "is likely correct. If `Hit` is consistently 0, the channel "
                 "ordering is different and needs to be re-derived from this "
                 "table.\n\n")
    lines.append("Constant channels (idx 0 = `_neutral` should be near-constant; "
                 "non-constant idx-0 means the dataset doesn't include "
                 "_neutral and the alphabetical channels are shifted by one).\n\n")

    # Top-3 most-constant channels — likely candidates for _neutral or extras.
    spreads = np.stack(list(emo_means.values())).std(axis=0)
    most_constant = np.argsort(spreads)[:5]
    lines.append("| rank | mead_idx | mediapipe_name (assumed) | std across emotions |\n"
                 "|---:|---:|---|---:|\n")
    for rank, idx in enumerate(most_constant):
        mp_name = (
            MEDIAPIPE_ORDER[idx] if idx < len(MEDIAPIPE_ORDER) else f"(unknown #{idx})"
        )
        lines.append(f"| {rank + 1} | {idx} | {mp_name} | {spreads[idx]:.4f} |\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(lines))
    print(f"\nwrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
