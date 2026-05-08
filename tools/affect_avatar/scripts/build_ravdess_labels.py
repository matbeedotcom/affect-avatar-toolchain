#!/usr/bin/env python3
"""Generate the spike-B0 labels JSON from a RAVDESS Actor_NN/ tree.

RAVDESS filename convention (7 hyphen-separated fields):
    {modality}-{vocal}-{emotion}-{intensity}-{statement}-{repetition}-{actor}.mp4

Emotion codes: 01=neutral 02=calm 03=happy 04=sad 05=angry
               06=fearful 07=disgust 08=surprised
Intensity:     01=normal 02=strong (neutral is always 01)
Modality:      01=full-AV  02=video-only  03=audio-only

This script walks the RAVDESS data root, filters to modality=01 (full-AV
mp4), and writes a JSON of `{relative_path: {speaker, emotion,
intensity}}` consumable by `spike_b0_extract_smoke.py --labels-json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EMOTION_CODE: dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
INTENSITY_CODE: dict[str, str] = {"01": "normal", "02": "strong"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True,
                   help="RAVDESS data root containing Actor_NN/ subdirs.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output JSON path.")
    p.add_argument("--modality", default="01",
                   choices=("01", "02"),
                   help="01=full-AV (default), 02=video-only.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.root.is_dir():
        print(f"error: --root not a directory: {args.root}", file=sys.stderr)
        return 2

    labels: dict[str, dict[str, str]] = {}
    for path in sorted(args.root.rglob("*.mp4")):
        rel = path.relative_to(args.root)
        parts = path.stem.split("-")
        if len(parts) != 7:
            continue
        modality, _vocal, emotion, intensity, _statement, _rep, actor = parts
        if modality != args.modality:
            continue
        labels[str(rel)] = {
            "speaker": f"Actor_{actor}",
            "emotion": EMOTION_CODE.get(emotion, emotion),
            "intensity": INTENSITY_CODE.get(intensity, intensity),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(labels, indent=2, sort_keys=True))
    counts: dict[str, int] = {}
    for v in labels.values():
        counts[v["emotion"]] = counts.get(v["emotion"], 0) + 1
    print(f"wrote {len(labels)} labels -> {args.output}", file=sys.stderr)
    for em in sorted(counts):
        print(f"  {em:9s}  {counts[em]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
