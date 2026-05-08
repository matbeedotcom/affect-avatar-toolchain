#!/usr/bin/env python3
"""Phase 2 Channels A + B smoke test.

End-to-end demonstration that ``LFM2AudioMlxNode`` honors the simulator's
trace output as a runtime steering directive across all three live
channels (A: decoding override, B: system augmentation, D: activation
steering — already proven in Phase 4).

Pipeline:
  1. Run the affect simulator on `repeated_tool_failure.json` to produce
     a trace JSON with per-frame `channel_a` / `channel_b` /
     `channel_d_target_vad` outputs.
  2. Pick a *baseline* frame (t=0, neutral) and a *stressed* frame (post
     third tool failure, frustration peaking).
  3. For each of two question audio clips, run the LLM through the same
     turn under both affect states. Drive all three channels via aux
     ports between the two runs.
  4. Verify the LLM responses differ across affect states. The pass
     criterion is qualitative — this is a wiring smoke test, not the
     full §6 Phase 2 acceptance gate.

Pass: the stressed-state response differs from the baseline-state
response on at least one of the two clips. The text differences are
printed to stdout for human review.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "clients/python"))

from remotemedia.core.multiprocessing.data import RuntimeData  # noqa: E402
from remotemedia.nodes.ml.lfm2_audio_mlx import LFM2AudioMlxNode  # noqa: E402

DEFAULT_DIRECTIONS = (
    REPO_ROOT
    / "tools/affect_calibration/artifacts/llm_directions/lfm2-audio-1.5b_layer9.npz"
)
DEFAULT_SCENARIO = (
    REPO_ROOT
    / "tools/affect_simulator/scenarios/repeated_tool_failure.json"
)
DEFAULT_AUDIO_DIR = REPO_ROOT / "tools/affect_calibration/data/crema_d/data/AudioWAV"
TARGET_SR = 24000

# Base values to which the simulator's `channel_a` scale factors apply.
BASE_TEMPERATURE = 1.0
BASE_MAX_NEW_TOKENS = 120

DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly conversational assistant. "
    "The user will speak to you; respond naturally in 2-3 short sentences "
    "as if continuing a conversation."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("phase2_smoke")


def _ensure_trace(scenario_path: Path) -> Path:
    """Run the simulator if a fresh trace isn't already next to scenario.json."""
    trace_path = scenario_path.with_suffix(".trace.json")
    if trace_path.exists() and trace_path.stat().st_mtime > scenario_path.stat().st_mtime:
        return trace_path
    logger.info("rebuilding trace via cargo run -p affect-simulator")
    subprocess.run(
        [
            "cargo", "run", "-p", "affect-simulator", "--quiet", "--",
            "run", str(scenario_path), "--out", str(trace_path),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    return trace_path


def _load_audio_24k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)
    if sr != TARGET_SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, sr // g).astype(np.float32, copy=False)
    return audio


def _pick_frames(trace: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Pick a baseline (t=0) frame and the most-stressed frame in the trace.

    Stress proxy: max channels.frustration. Repeated_tool_failure peaks
    frustration on the third failure; that's the frame we want.
    """
    frames = trace["frames"]
    baseline = frames[0]
    stressed = max(frames, key=lambda f: f["channels"]["frustration"])
    return {"baseline": baseline, "stressed": stressed}


async def _run_audio_turn(node: LFM2AudioMlxNode, audio_np: np.ndarray) -> str:
    rd = RuntimeData.audio(audio_np, TARGET_SR, channels=1)
    text_chunks: List[str] = []
    async for out in node.process(rd):
        if hasattr(out, "is_text") and out.is_text():
            try:
                t = out.as_text() if hasattr(out, "as_text") else str(out.payload)
            except Exception:  # noqa: BLE001
                t = str(getattr(out, "payload", ""))
            if t in ("<|text_end|>", "<|audio_end|>"):
                continue
            text_chunks.append(t)
    return "".join(text_chunks).strip()


def _apply_frame(node: LFM2AudioMlxNode, frame: Dict[str, Any]) -> None:
    """Drive Channels A, B, D from a single trace frame into the node."""
    # Channel A — multiplicative scale on base sampling values.
    ch_a = frame["channel_a"]
    node.set_sampling(
        temperature=BASE_TEMPERATURE * float(ch_a["temperature_scale"]),
        max_new_tokens=int(round(BASE_MAX_NEW_TOKENS * float(ch_a["max_tokens_scale"]))),
    )
    # Channel B — system-prompt augmentation. Empty string clears.
    node.set_system_augmentation(frame["channel_b"])
    # Channel D — activation steering target VAD.
    node.set_steering(target_vad=list(frame["channel_d_target_vad"]), alpha=1.0)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--directions", type=Path, default=DEFAULT_DIRECTIONS)
    parser.add_argument(
        "--clips", nargs="+", type=Path,
        default=[
            DEFAULT_AUDIO_DIR / "1001_DFA_NEU_XX.wav",  # "Don't forget a jacket"
            DEFAULT_AUDIO_DIR / "1001_IEO_NEU_XX.wav",  # "It's eleven o'clock"
        ],
        help="Audio clips to use as user turns; default = two CREMA-D neutrals.",
    )
    args = parser.parse_args()

    if not args.directions.exists():
        logger.error("steering artifact missing: %s", args.directions)
        return 2
    if not args.scenario.exists():
        logger.error("scenario missing: %s", args.scenario)
        return 2

    trace_path = _ensure_trace(args.scenario)
    trace = json.loads(trace_path.read_text())
    picks = _pick_frames(trace)
    logger.info(
        "frames picked — baseline t=%dms (frustration=%.2f, valence=%+.2f); "
        "stressed t=%dms (frustration=%.2f, valence=%+.2f)",
        picks["baseline"]["timestamp_ms"],
        picks["baseline"]["channels"]["frustration"],
        picks["baseline"]["core"]["valence"],
        picks["stressed"]["timestamp_ms"],
        picks["stressed"]["channels"]["frustration"],
        picks["stressed"]["core"]["valence"],
    )

    # Single LLM instance, exercised under both affect states. Steering
    # is enabled at construction so the layer-9 hook is installed; a
    # baseline frame with target_vad ≈ baseline-affect produces a
    # near-zero residual (small but nonzero — channels.calm is at its
    # baseline 0.4, so valence ≈ +0.4).
    node = LFM2AudioMlxNode(
        steering_enabled=True,
        steering_directions_path=str(args.directions),
        steering_alpha=1.0,
        text_only=True,
        max_new_tokens=BASE_MAX_NEW_TOKENS,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )
    await node.initialize()

    results: List[Dict[str, str]] = []
    diverged = 0
    for clip_path in args.clips:
        if not clip_path.exists():
            logger.warning("clip missing, skipping: %s", clip_path)
            continue
        audio = _load_audio_24k(clip_path)
        logger.info("=== clip %s (%.2fs) ===", clip_path.name, len(audio) / TARGET_SR)

        # Baseline run.
        _apply_frame(node, picks["baseline"])
        out_baseline = await _run_audio_turn(node, audio)

        # Stressed run. The augmentation change invalidates sessions, so
        # this turn rebuilds chat state with the stressed system message.
        _apply_frame(node, picks["stressed"])
        out_stressed = await _run_audio_turn(node, audio)

        results.append({
            "clip": clip_path.name,
            "baseline": out_baseline,
            "stressed": out_stressed,
        })
        if out_baseline != out_stressed:
            diverged += 1

        print(f"\n=== {clip_path.name} ===")
        print("BASELINE (neutral affect):\n  " + (out_baseline or "(empty)"))
        print("STRESSED (frustration peak):\n  " + (out_stressed or "(empty)"))

    await node.cleanup()

    print()
    if diverged == 0:
        print("FAIL: baseline and stressed responses identical on all clips.")
        return 1
    print(f"PASS: {diverged}/{len(results)} clips show response divergence "
          "between baseline and stressed affect states.")
    print(
        "\nNote: this is a *wiring* smoke test — it confirms Channels A/B/D "
        "carry signal end-to-end. The §6 Phase 2 acceptance gate (subjective "
        "review by 2+ humans) is a separate exercise."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
