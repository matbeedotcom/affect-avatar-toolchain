#!/usr/bin/env python3
"""Spike B0 — MediaPipe blendshape extraction smoke test.

Per the protocol in
docs/references/audio-blendshape-diffusion/notes/spike-b0-blendshape-extraction.md.

For each clip in `--input-dir` (recursive search of common video
extensions), run MediaPipe FaceLandmarker and emit:

  M-B0-1 — Coverage check: list of returned blendshape names + count;
           cross-check against the ARKit-52 reference list.
  M-B0-2 — Smoothness: per-clip per-frame first-difference RMS jitter,
           reported as a distribution + 95th percentile.
  M-B0-3 — Affect signal (optional): if `--labels-json` is provided
           (mapping {relative_path: {speaker, emotion, intensity}}),
           compute mean of diagnostic blendshapes per emotion class
           and tabulate the expected-direction matches.

Outputs:
  - <output>.npz: per-clip blendshape trajectories + metadata.
  - <report>.md: human-readable summary + decision recommendation.

Usage:
  ./spike_b0_extract_smoke.py \\
      --input-dir data/mead/clips \\
      --labels-json data/mead/labels_b0.json \\
      --max-clips 100 \\
      --model-path data/face_landmarker.task \\
      --output artifacts/spike_b0_results.npz \\
      --report ../../docs/references/audio-blendshape-diffusion/notes/spike-b0-blendshape-extraction.md.run

  ./spike_b0_extract_smoke.py --dry-run \\
      --output /tmp/spike_b0_dry.npz \\
      --report /tmp/spike_b0_dry.md
      # Synthesizes plausible blendshape trajectories without invoking
      # MediaPipe; useful for exercising the report code on CI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ARKit-52 reference order — Apple's canonical iOS ARKit blendshape list.
# We don't strictly *require* MediaPipe to return these names (the model
# returns its own set, with `_neutral` and possibly other deltas); the
# coverage check just reports presence/absence so we know what to map.
ARKIT_52: tuple[str, ...] = (
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
    "tongueOut",
)


# Diagnostic blendshapes used in M-B0-3 (mean per emotion, average across
# left/right). Each value is the *expected ordering* for the four scored
# emotions (happy, sad, angry, surprised) — "high"/"medium"/"low".
DIAGNOSTIC_BLENDSHAPES: dict[str, list[str]] = {
    "smile":     ["mouthSmileLeft", "mouthSmileRight"],
    "browInnerUp": ["browInnerUp"],
    "eyeWide":   ["eyeWideLeft", "eyeWideRight"],
    "browDown":  ["browDownLeft", "browDownRight"],
}

# Expected direction (rank order, highest=3, lowest=0). NaN means
# "not scored" (we don't grade against neutral/disgusted/contempt; their
# expected behavior on smile/brow/eye is more ambiguous).
EXPECTED_RANKS: dict[str, dict[str, float]] = {
    "smile":     {"happy": 3.0, "sad": 0.0, "angry": 0.0, "surprised": 2.0},
    "browInnerUp": {"happy": 0.0, "sad": 3.0, "angry": 0.0, "surprised": 2.0},
    "eyeWide":   {"happy": 1.5, "sad": 0.0, "angry": 0.0, "surprised": 3.0},
    "browDown":  {"happy": 0.0, "sad": 0.0, "angry": 3.0, "surprised": 0.0},
}

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}

JITTER_PASS_THRESHOLD = 0.05      # M-B0-2: 95th-percentile per-frame jitter
COVERAGE_PASS_NAMES = 52          # M-B0-1: at least 52 names returned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Directory to recursively search for video clips. Required "
             "unless --dry-run.",
    )
    parser.add_argument(
        "--labels-json", type=Path, default=None,
        help="Optional JSON mapping {relative_path: {speaker, emotion, "
             "intensity}}. If absent, M-B0-3 affect-signal scoring is "
             "skipped.",
    )
    parser.add_argument(
        "--model-path", type=Path, default=None,
        help="Path to MediaPipe face_landmarker.task. Defaults to "
             "$MEDIAPIPE_FACE_LANDMARKER, then "
             "tools/affect_avatar/data/face_landmarker.task.",
    )
    parser.add_argument(
        "--max-clips", type=int, default=100,
        help="Cap clip count (default 100, per spike protocol).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output NPZ with per-clip arrays + metadata.",
    )
    parser.add_argument(
        "--report", type=Path, required=True,
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Synthesize plausible-shape blendshape trajectories instead "
             "of running MediaPipe. Useful for testing the report writer.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.dry_run:
        clips = _dry_run_clips(rng, n=args.max_clips)
        names = list(ARKIT_52)
    else:
        if args.input_dir is None:
            print("error: --input-dir required (or pass --dry-run)", file=sys.stderr)
            return 2
        labels = _load_labels(args.labels_json) if args.labels_json else {}
        clips, names = _real_clips(
            input_dir=args.input_dir,
            labels=labels,
            model_path=args.model_path,
            max_clips=args.max_clips,
        )
        if not clips:
            print("error: no usable clips found", file=sys.stderr)
            return 2

    coverage = _evaluate_coverage(names)
    jitter = _evaluate_jitter(clips)
    affect = _evaluate_affect_signal(clips, names)

    _save_npz(args.output, clips, names)
    _write_report(
        args.report,
        clips=clips,
        names=names,
        coverage=coverage,
        jitter=jitter,
        affect=affect,
        dry_run=args.dry_run,
    )
    print(f"wrote -> {args.output}", file=sys.stderr)
    print(f"wrote -> {args.report}", file=sys.stderr)

    overall_pass = (
        coverage["pass"]
        and jitter["pass"]
        and (affect is None or affect["pass"])
    )
    return 0 if overall_pass else 1


# ---------------------------------------------------------------------------
# Clip loading & extraction
# ---------------------------------------------------------------------------

def _real_clips(
    *,
    input_dir: Path,
    labels: dict[str, dict],
    model_path: Optional[Path],
    max_clips: int,
):
    """Walk input_dir, run MediaPipe on each clip, collect results."""

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lib.blendshape_extractor import BlendshapeExtractor  # noqa: E402

    paths = sorted(
        p for p in input_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS
    )
    if not paths:
        return [], []
    if max_clips > 0:
        paths = paths[:max_clips]

    extractor = BlendshapeExtractor(model_path=model_path)
    names: list[str] = []
    clips: list[dict] = []
    t0 = time.time()

    for i, path in enumerate(paths, 1):
        try:
            result = extractor.extract_clip(path)
        except Exception as exc:
            print(f"  [{i}/{len(paths)}] FAILED {path.name}: {exc}", file=sys.stderr)
            continue

        if result.blendshapes.size == 0:
            print(f"  [{i}/{len(paths)}] NO_FACE   {path.name}", file=sys.stderr)
            continue

        if not names:
            names = result.names

        rel = str(path.relative_to(input_dir))
        meta = labels.get(rel, {})
        clips.append({
            "relative_path": rel,
            "blendshapes": result.blendshapes.astype(np.float32),
            "n_frames_total": result.n_frames_total,
            "n_frames_with_face": result.n_frames_with_face,
            "fps": result.fps,
            "speaker": meta.get("speaker", ""),
            "emotion": meta.get("emotion", ""),
            "intensity": meta.get("intensity", ""),
        })
        if i % 10 == 0 or i == len(paths):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-6)
            print(
                f"  [{i}/{len(paths)}] kept={len(clips)}  "
                f"{rate:.2f} clip/s  elapsed={elapsed:.0f}s",
                file=sys.stderr,
            )

    extractor.close()
    return clips, names


def _dry_run_clips(rng: np.random.Generator, *, n: int) -> list[dict]:
    """Synthesize plausible (T, 52) trajectories with ARKit-52 names."""

    # Five fake speakers x four emotions x intensities, capped at n.
    speakers = [f"M{i:02d}" for i in range(1, 4)] + [f"W{i:02d}" for i in range(1, 4)]
    emotions = ["neutral", "happy", "sad", "angry", "surprised"]
    intensities = ["low", "medium", "high"]

    clips: list[dict] = []
    while len(clips) < n:
        sp = speakers[rng.integers(len(speakers))]
        em = emotions[rng.integers(len(emotions))]
        it = intensities[rng.integers(len(intensities))]
        T = int(rng.integers(60, 180))   # 2-6 seconds at 30 fps

        # White noise → 11-tap moving-average → ~real-face-like smoothness.
        # Without smoothing the per-frame jitter sits ~0.07, far above the
        # 0.05 pass threshold; real trajectories are temporally correlated.
        raw = rng.normal(0.1, 0.05, size=(T, len(ARKIT_52))).astype(np.float32)
        kernel = np.ones(11, dtype=np.float32) / 11.0
        bs = np.stack(
            [np.convolve(raw[:, k], kernel, mode="same") for k in range(raw.shape[1])],
            axis=1,
        )
        bs = np.clip(bs, 0.0, 1.0)
        # Inject expected-direction signal so M-B0-3 reports a PASS in dry-run.
        idx = {n: i for i, n in enumerate(ARKIT_52)}
        if em == "happy":
            for k in ("mouthSmileLeft", "mouthSmileRight"):
                bs[:, idx[k]] += 0.45
        elif em == "sad":
            bs[:, idx["browInnerUp"]] += 0.55
        elif em == "angry":
            for k in ("browDownLeft", "browDownRight"):
                bs[:, idx[k]] += 0.50
        elif em == "surprised":
            for k in ("eyeWideLeft", "eyeWideRight", "browInnerUp"):
                bs[:, idx[k]] += 0.40
        bs = np.clip(bs, 0.0, 1.0)

        clips.append({
            "relative_path": f"dry_run/{len(clips):03d}.mp4",
            "blendshapes": bs,
            "n_frames_total": T,
            "n_frames_with_face": T,
            "fps": 30.0,
            "speaker": sp,
            "emotion": em,
            "intensity": it,
        })
    return clips


def _load_labels(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metric evaluation
# ---------------------------------------------------------------------------

def _evaluate_coverage(names: list[str]) -> dict:
    """M-B0-1 — does the model emit ≥52 names that include the ARKit set?"""

    arkit_set = set(ARKIT_52)
    name_set = set(names)
    missing = sorted(arkit_set - name_set)
    extras = sorted(name_set - arkit_set)
    n_returned = len(names)
    n_arkit_present = len(arkit_set & name_set)

    return {
        "names": names,
        "n_returned": n_returned,
        "n_arkit_present": n_arkit_present,
        "missing": missing,
        "extras": extras,
        "pass": n_returned >= COVERAGE_PASS_NAMES and not missing,
    }


def _evaluate_jitter(clips: list[dict]) -> dict:
    """M-B0-2 — 95th-percentile per-frame first-difference RMS."""

    per_clip: list[float] = []
    for c in clips:
        bs = c["blendshapes"]
        if bs.shape[0] < 2 or bs.shape[1] == 0:
            continue
        # NaN-safe: drop rows that contain NaNs from the diff.
        mask = ~np.isnan(bs).any(axis=1)
        clean = bs[mask]
        if clean.shape[0] < 2:
            continue
        diff = np.diff(clean, axis=0)
        rms = float(np.sqrt(np.mean(diff ** 2)))
        per_clip.append(rms)

    if not per_clip:
        return {
            "per_clip": [],
            "n_clips_scored": 0,
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "pass": False,
            "note": "no clips with enough valid frames to score",
        }

    arr = np.array(per_clip, dtype=np.float32)
    return {
        "per_clip": per_clip,
        "n_clips_scored": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "pass": float(np.percentile(arr, 95)) < JITTER_PASS_THRESHOLD,
    }


def _evaluate_affect_signal(clips: list[dict], names: list[str]) -> Optional[dict]:
    """M-B0-3 — mean of diagnostic blendshapes per emotion, ranked."""

    if not names or not clips:
        return None

    have_labels = any(c.get("emotion") for c in clips)
    if not have_labels:
        return None

    name_to_col = {n: i for i, n in enumerate(names)}
    by_emotion: dict[str, dict[str, list[float]]] = {}

    for c in clips:
        em = c.get("emotion") or ""
        if not em:
            continue
        bs = c["blendshapes"]
        if bs.shape[0] == 0:
            continue
        clean = bs[~np.isnan(bs).any(axis=1)]
        if clean.shape[0] == 0:
            continue
        clip_means = clean.mean(axis=0)
        slot = by_emotion.setdefault(em, {k: [] for k in DIAGNOSTIC_BLENDSHAPES})
        for diag, members in DIAGNOSTIC_BLENDSHAPES.items():
            cols = [name_to_col[m] for m in members if m in name_to_col]
            if not cols:
                continue
            slot[diag].append(float(np.mean(clip_means[cols])))

    table: dict[str, dict[str, float]] = {}
    for em, slots in by_emotion.items():
        table[em] = {
            k: float(np.mean(v)) if v else float("nan") for k, v in slots.items()
        }

    rank_correctness: dict[str, dict[str, bool]] = {}
    scored_emotions = {"happy", "sad", "angry", "surprised"}
    n_emotion_axes = 0
    n_correct = 0
    for diag, expected in EXPECTED_RANKS.items():
        ranking = sorted(
            ((em, table.get(em, {}).get(diag, float("nan"))) for em in scored_emotions),
            key=lambda kv: (np.nan_to_num(kv[1], nan=-1.0)),
            reverse=True,
        )
        observed_rank = {em: float(len(scored_emotions) - i - 1) for i, (em, _) in enumerate(ranking)}
        rank_correctness[diag] = {}
        for em in scored_emotions:
            exp = expected[em]
            obs = observed_rank.get(em, float("nan"))
            ok = abs(exp - obs) <= 1.0  # within one rank position
            rank_correctness[diag][em] = ok
            n_emotion_axes += 1
            if ok:
                n_correct += 1

    correct_diag = sum(
        1 for diag, marks in rank_correctness.items()
        if sum(marks.values()) >= 3
    )
    return {
        "table": table,
        "rank_correctness": rank_correctness,
        "n_correct_diag": correct_diag,
        "n_diag": len(DIAGNOSTIC_BLENDSHAPES),
        "n_correct_emotion_axes": n_correct,
        "n_emotion_axes": n_emotion_axes,
        "pass": correct_diag >= 3,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _save_npz(path: Path, clips: list[dict], names: list[str]) -> None:
    """Pack clips + metadata into a single NPZ.

    Per-clip arrays are stored as `bs_{i}` to keep them ragged-safe;
    metadata travels as parallel object arrays addressable by index.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "names": np.asarray(names, dtype=object),
        "relative_paths": np.asarray([c["relative_path"] for c in clips], dtype=object),
        "speakers":      np.asarray([c["speaker"] for c in clips], dtype=object),
        "emotions":      np.asarray([c["emotion"] for c in clips], dtype=object),
        "intensities":   np.asarray([c["intensity"] for c in clips], dtype=object),
        "fps":           np.asarray([c["fps"] for c in clips], dtype=np.float32),
        "n_frames_total": np.asarray([c["n_frames_total"] for c in clips], dtype=np.int32),
        "n_frames_with_face": np.asarray([c["n_frames_with_face"] for c in clips], dtype=np.int32),
    }
    for i, c in enumerate(clips):
        payload[f"bs_{i:04d}"] = c["blendshapes"]
    np.savez_compressed(path, **payload)


def _write_report(
    path: Path,
    *,
    clips: list[dict],
    names: list[str],
    coverage: dict,
    jitter: dict,
    affect: Optional[dict],
    dry_run: bool,
) -> None:
    lines: list[str] = []
    lines.append("# Spike B0 — MediaPipe blendshape extraction smoke results\n")
    if dry_run:
        lines.append(
            "\n> **DRY-RUN report** — no MediaPipe involved; trajectories "
            "are synthesized. Re-run without `--dry-run` for a real result.\n"
        )
    overall = coverage["pass"] and jitter["pass"] and (affect is None or affect["pass"])
    lines.append(f"\n**Overall**: {'PASS' if overall else 'FAIL'}\n")
    lines.append(f"\n**Clips scored**: {len(clips)}\n")

    # ---- M-B0-1 ----
    lines.append("\n## M-B0-1 — Coverage\n")
    lines.append(f"\n- Names returned: **{coverage['n_returned']}**\n")
    lines.append(f"- ARKit-52 present: **{coverage['n_arkit_present']}/52**\n")
    if coverage["missing"]:
        lines.append(f"- **Missing from MediaPipe output**: {coverage['missing']}\n")
    if coverage["extras"]:
        lines.append(f"- Extra (non-ARKit) names: {coverage['extras']}\n")
    lines.append(f"- Result: {'✅ PASS' if coverage['pass'] else '❌ FAIL'}\n")
    lines.append("\n<details><summary>Full name list</summary>\n\n")
    for i, n in enumerate(names):
        lines.append(f"  {i:>2d}. {n}\n")
    lines.append("\n</details>\n")

    # ---- M-B0-2 ----
    lines.append("\n## M-B0-2 — Per-frame jitter (RMS first-difference)\n")
    if jitter.get("note"):
        lines.append(f"\n> {jitter['note']}\n")
    lines.append(
        f"\n- Clips scored: **{jitter['n_clips_scored']}**\n"
        f"- p50 jitter: **{jitter['p50']:.4f}**\n"
        f"- p95 jitter: **{jitter['p95']:.4f}** (pass threshold < {JITTER_PASS_THRESHOLD})\n"
        f"- p99 jitter: **{jitter['p99']:.4f}**\n"
        f"- Result: {'✅ PASS' if jitter['pass'] else '❌ FAIL'}\n"
    )

    # ---- M-B0-3 ----
    lines.append("\n## M-B0-3 — Affect signal\n")
    if affect is None:
        lines.append(
            "\n> Skipped — no clip-level emotion labels were provided. "
            "Pass `--labels-json data/<dataset>/labels_b0.json` to score this gate.\n"
        )
    else:
        lines.append(
            "\n_Mean diagnostic-blendshape activation per emotion class. "
            "Higher = more activated. Per-clip mean over time, then averaged "
            "across clips with that label._\n\n"
        )
        emotions = sorted(affect["table"].keys())
        diags = list(DIAGNOSTIC_BLENDSHAPES.keys())
        head = "| emotion | " + " | ".join(diags) + " |\n"
        sep  = "|---|" + "|".join([":---:"] * len(diags)) + "|\n"
        lines.append(head)
        lines.append(sep)
        for em in emotions:
            cells = [f"{affect['table'][em][d]:+.3f}" for d in diags]
            lines.append(f"| {em} | " + " | ".join(cells) + " |\n")

        lines.append(
            f"\n- Diagnostic blendshapes correctly ordered (≥3 of 4 "
            f"scored emotions in expected position): "
            f"**{affect['n_correct_diag']}/{affect['n_diag']}**\n"
            f"- Per-emotion-axis correctness: "
            f"**{affect['n_correct_emotion_axes']}/{affect['n_emotion_axes']}**\n"
            f"- Result: {'✅ PASS' if affect['pass'] else '❌ FAIL'}\n"
        )

    # ---- Decision ----
    lines.append("\n## Decision\n\n")
    if overall:
        lines.append(
            "All three gates pass. **Recommend B1**: commit "
            "`01_extract_blendshapes.py`, run extraction at scale on "
            "MEAD + CREMA-D + RAVDESS, and proceed to Stage-1 VAE training.\n"
        )
    else:
        why = []
        if not coverage["pass"]:
            why.append("M-B0-1 (coverage)")
        if not jitter["pass"]:
            why.append("M-B0-2 (jitter)")
        if affect is not None and not affect["pass"]:
            why.append("M-B0-3 (affect signal)")
        lines.append(
            f"Failed gates: **{', '.join(why)}**. "
            f"See `notes/spike-b0-blendshape-extraction.md` decision-gate "
            f"table for the remediation path per failure mode.\n"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
