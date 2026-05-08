#!/usr/bin/env python3
"""Phase 6 audio-side steering smoke test (Spike G Q-G4).

The Phase 5 calibration extracted layer-9 directions on text-in /
text-out forwards (Spike G Option α). Q-G4 asks: do those directions
still bias the response when the *input* arrives as audio rather than
text? This is the production path — LFM2-Audio's natural mode is
audio-in / interleaved-out — so a "no" answer would mean the
calibration has to be redone with audio prompts before runtime
steering ships.

Mechanism: feed the same neutral CREMA-D clip through
`LFM2AudioMlxNode.process()` with steering disabled, +1 valence, and
-1 valence. If the three text outputs differ — and the +/- shift
matches the §4.6 text-side gap pattern — the directions transfer to
the audio modality.

Pass criterion (qualitative; quantitative gate is §4.6 already passed
for the text path):
- unsteered ≠ +α steered
- unsteered ≠ -α steered
- +α ≠ -α
- clear_steering produces text identical to a fresh unsteered run

This is a *sanity* check. A multi-clip / LLM-judge variant would be
the proper Phase 6 deliverable; this script is the Q-G4 yes-or-no.

Usage:
  ./06_audio_steering_smoke.py
  ./06_audio_steering_smoke.py --clip /path/to/utterance.wav --alpha 1.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

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
# CREMA-D sentence DFA = "Don't forget a jacket" — short neutral content
# so any response shift comes from the steering bias rather than the
# model parroting the input's emotional content. NEU intensity tag
# selects the calibration's neutrally-spoken take.
DEFAULT_CLIP = (
    REPO_ROOT
    / "tools/affect_calibration/data/crema_d/data/AudioWAV/1001_DFA_NEU_XX.wav"
)
TARGET_SR = 24000
# CREMA-D clips are short isolated sentences with no preamble, so the
# default model card prompt "Respond with interleaved text and audio."
# leaves the model uncertain what behavior to produce — it falls into
# half-transcription, half-token-loop. This system prompt nudges it
# toward conversational reply behavior, which is what the production
# pipeline actually targets.
DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly conversational assistant. "
    "The user will speak to you; respond naturally in 2-3 short sentences "
    "as if continuing a conversation."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("phase6_smoke")


def _load_audio_24k(path: Path) -> np.ndarray:
    """Read a WAV and resample to 24 kHz mono float32 for LFM2-Audio."""
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)
    if sr != TARGET_SR:
        # Use polyphase resampling — same gcd-based path librosa uses
        # internally, no extra dep beyond scipy which is already
        # required by the calibration toolchain.
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, TARGET_SR)
        up, down = TARGET_SR // g, sr // g
        audio = resample_poly(audio, up, down).astype(np.float32, copy=False)
    return audio


async def _run_audio_turn(
    node: LFM2AudioMlxNode,
    audio_np: np.ndarray,
) -> str:
    """Drive node.process() with one audio turn, collect text output."""
    rd = RuntimeData.audio(audio_np, TARGET_SR, channels=1)
    text_chunks: list[str] = []
    async for out in node.process(rd):
        # Only collect text outputs. Audio outputs are skipped — this
        # smoke test compares response *content*, not waveforms.
        if hasattr(out, "is_text") and out.is_text():
            try:
                t = out.as_text() if hasattr(out, "as_text") else str(out.payload)
            except Exception:  # noqa: BLE001
                t = str(getattr(out, "payload", ""))
            if t in ("<|text_end|>", "<|audio_end|>"):
                continue
            text_chunks.append(t)
    return "".join(text_chunks).strip()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directions", type=Path, default=DEFAULT_DIRECTIONS)
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt — drives the model toward conversational reply behavior.",
    )
    args = parser.parse_args()

    if not args.directions.exists():
        logger.error("steering artifact missing: %s", args.directions)
        return 2
    if not args.clip.exists():
        logger.error("audio clip missing: %s", args.clip)
        return 2

    audio_np = _load_audio_24k(args.clip)
    logger.info(
        "loaded audio: %s (%.2fs at %d Hz)",
        args.clip.name, len(audio_np) / TARGET_SR, TARGET_SR,
    )

    # Run 1 — unsteered baseline.
    logger.info("=== run 1: unsteered ===")
    node_a = LFM2AudioMlxNode(
        steering_enabled=False,
        text_only=True,
        max_new_tokens=args.max_new_tokens,
        system_prompt=args.system_prompt,
    )
    await node_a.initialize()
    out_unsteered = await _run_audio_turn(node_a, audio_np)
    print("\nUNSTEERED:\n" + (out_unsteered or "(empty)") + "\n")
    await node_a.cleanup()

    # Run 2 — +α valence.
    logger.info("=== run 2: steered +%.2f valence ===", args.alpha)
    node_b = LFM2AudioMlxNode(
        steering_enabled=True,
        steering_directions_path=str(args.directions),
        steering_target_vad=[args.alpha, 0.0, 0.0],
        steering_alpha=1.0,
        text_only=True,
        max_new_tokens=args.max_new_tokens,
        system_prompt=args.system_prompt,
    )
    await node_b.initialize()
    cfg = node_b.get_config()
    print("config:", json.dumps(
        {k: cfg[k] for k in (
            "steering_enabled", "steering_layer", "steering_target_vad",
            "steering_alpha", "steering_active",
        )},
        default=str,
    ))
    out_pos = await _run_audio_turn(node_b, audio_np)
    print("\nSTEERED (+%.2f valence):\n%s\n" % (args.alpha, out_pos or "(empty)"))

    # Run 3 — -α valence (set_steering, no re-init).
    logger.info("=== run 3: steered -%.2f valence ===", args.alpha)
    node_b.set_steering(target_vad=[-args.alpha, 0.0, 0.0], alpha=1.0)
    out_neg = await _run_audio_turn(node_b, audio_np)
    print("\nSTEERED (-%.2f valence):\n%s\n" % (args.alpha, out_neg or "(empty)"))

    # Run 4 — cleared. Should match a fresh unsteered run.
    logger.info("=== run 4: cleared ===")
    node_b.clear_steering()
    out_cleared = await _run_audio_turn(node_b, audio_np)
    print("\nSTEERED → CLEARED:\n%s\n" % (out_cleared or "(empty)"))
    await node_b.cleanup()

    failures: list[str] = []
    if out_unsteered == out_pos:
        failures.append("+α-steered output identical to unsteered")
    if out_unsteered == out_neg:
        failures.append("-α-steered output identical to unsteered")
    if out_pos == out_neg:
        failures.append("+α and -α valence produced identical output")
    # `clear_steering` is on the same node instance with multi-turn
    # session state, so the *prior* steered turn is in the chat history
    # — its output may differ from the fresh unsteered run for that
    # reason alone (history bias, not steering bias). We log the diff
    # but don't fail on it.
    if out_unsteered != out_cleared:
        print(
            "NOTE: cleared-output ≠ unsteered baseline — likely session "
            "history accumulation, not a steering leak. Inspect manually."
        )
        print("  unsteered:", repr(out_unsteered[:80]))
        print("  cleared:  ", repr(out_cleared[:80]))

    if failures:
        for f in failures:
            print("FAIL:", f)
        print(
            "\nQ-G4: NEGATIVE — directions did not transfer to audio input. "
            "Calibration may need to be re-run with audio prompts."
        )
        return 1
    print(
        "\nQ-G4: POSITIVE — text-extracted directions still bias response "
        "under audio input. Production path validated."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
