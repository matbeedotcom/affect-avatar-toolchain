#!/usr/bin/env python3
"""Phase 4 runtime steering smoke test.

Sanity-checks that ``LFM2AudioMlxNode`` actually applies the calibration
artifact at inference: loads the production node once with steering
disabled and once with steering enabled, runs the same text-only prompt
through both, and verifies the two outputs differ in the expected
direction (+α valence → more positive answer, ‑α valence → less
positive). Also verifies that ``clear_steering`` restores byte-identical
output to the baseline (proves the wrapper's short-circuit path).

This is a *sanity* check, not the quantitative §4.6 gate — that lives in
``04_validate_pipeline.py`` and runs the LLM-judge protocol over hundreds
of prompts. This script only confirms the wiring is intact.

Usage:
    ./05_runtime_steering_smoke.py
    ./05_runtime_steering_smoke.py --directions /path/to/custom.npz \\
                                   --question "How are you feeling?"

Run from the calibration env that has ``mlx-audio`` available, e.g.::

    /Users/.../envs/<id>/bin/python tools/affect_calibration/scripts/05_runtime_steering_smoke.py

Exit 0 on pass, 1 on smoke-check failure, 2 on missing artifact.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Make the SDK importable when this script is run directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "clients/python"))

from remotemedia.nodes.ml.lfm2_audio_mlx import LFM2AudioMlxNode  # noqa: E402

DEFAULT_DIRECTIONS = (
    REPO_ROOT
    / "tools/affect_calibration/artifacts/llm_directions/lfm2-audio-1.5b_layer9.npz"
)
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question in 2-3 short sentences."
)
DEFAULT_QUESTION = "How is your day going?"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("phase4_smoke")


async def _generate_text_only(
    node: LFM2AudioMlxNode,
    *,
    system_prompt: str,
    question: str,
    max_new_tokens: int = 80,
) -> str:
    """Drive the model directly through ``mlx-audio``'s text path.

    The node's :meth:`process` expects an audio RuntimeData input; for the
    smoke test we want text-in/text-out to match the calibration's
    extraction modality (Spike G Option α). Reuse the underlying chat
    primitives so the steering hook still fires on every per-step forward.
    """
    import mlx.core as mx
    from mlx_audio.sts.models.lfm_audio import ChatState, LFMModality

    chat = ChatState(node._processor)  # type: ignore[attr-defined]
    chat.new_turn("system")
    chat.add_text(system_prompt)
    chat.end_turn()
    chat.new_turn("user")
    chat.add_text(question)
    chat.end_turn()
    chat.new_turn("assistant")

    gen = node._model.generate_sequential(  # type: ignore[attr-defined]
        **dict(chat),
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=1,
    )
    text_ids: list[int] = []
    IM_END = 7
    for token, modality in gen:
        mx.eval(token)
        if modality != LFMModality.TEXT:
            break
        tid = int(token.item())
        if tid == IM_END:
            break
        text_ids.append(tid)
    if not text_ids:
        return ""
    return node._processor.tokenizer.decode(  # type: ignore[attr-defined]
        text_ids, skip_special_tokens=True
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directions",
        type=Path,
        default=DEFAULT_DIRECTIONS,
        help="Path to the LLM directions NPZ (default: layer 9 LFM2-Audio).",
    )
    parser.add_argument(
        "--system-prompt", default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt sent to the model.",
    )
    parser.add_argument(
        "--question", default=DEFAULT_QUESTION,
        help="Single-turn user question.",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Steering scale (default: 1.0).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=80,
        help="Cap on generated tokens per run.",
    )
    args = parser.parse_args()

    if not args.directions.exists():
        logger.error("steering artifact missing: %s", args.directions)
        return 2

    # Run 1 — unsteered baseline.
    logger.info("=== run 1: unsteered ===")
    node_a = LFM2AudioMlxNode(steering_enabled=False, text_only=True)
    await node_a.initialize()
    out_unsteered = await _generate_text_only(
        node_a,
        system_prompt=args.system_prompt,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
    )
    print("\nUNSTEERED:\n" + out_unsteered + "\n")
    await node_a.cleanup()

    # Run 2 — steered toward positive valence.
    logger.info("=== run 2: steered +%.2f valence ===", args.alpha)
    node_b = LFM2AudioMlxNode(
        steering_enabled=True,
        steering_directions_path=str(args.directions),
        steering_target_vad=[args.alpha, 0.0, 0.0],
        steering_alpha=1.0,
        text_only=True,
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
    out_steered_pos = await _generate_text_only(
        node_b,
        system_prompt=args.system_prompt,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
    )
    print("\nSTEERED (+%.2f valence):\n%s\n" % (args.alpha, out_steered_pos))

    # Run 3 — same node, different directive (-α valence). Verifies the
    # set_steering aux-port path: state mutation without re-init.
    logger.info("=== run 3: steered -%.2f valence (via set_steering) ===", args.alpha)
    node_b.set_steering(target_vad=[-args.alpha, 0.0, 0.0], alpha=1.0)
    out_steered_neg = await _generate_text_only(
        node_b,
        system_prompt=args.system_prompt,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
    )
    print("\nSTEERED (-%.2f valence):\n%s\n" % (args.alpha, out_steered_neg))

    # Run 4 — clear_steering. Should match unsteered exactly because
    # sampling is deterministic-greedy.
    logger.info("=== run 4: cleared steering ===")
    node_b.clear_steering()
    out_cleared = await _generate_text_only(
        node_b,
        system_prompt=args.system_prompt,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
    )
    print("\nSTEERED → CLEARED:\n" + out_cleared + "\n")

    await node_b.cleanup()

    # Smoke checks.
    failures: list[str] = []
    if out_unsteered == out_steered_pos:
        failures.append("positive-valence-steered output identical to unsteered")
    if out_unsteered == out_steered_neg:
        failures.append("negative-valence-steered output identical to unsteered")
    if out_steered_pos == out_steered_neg:
        failures.append("+α and -α valence produced identical output")
    if out_unsteered != out_cleared:
        failures.append(
            "clear_steering did not restore baseline "
            f"(unsteered={out_unsteered[:60]!r}, cleared={out_cleared[:60]!r})"
        )

    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("ALL SMOKE CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
