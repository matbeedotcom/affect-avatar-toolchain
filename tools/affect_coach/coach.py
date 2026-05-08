#!/usr/bin/env python3
"""Affect-coach CLI — "what to say and how to say it" for the user.

Runs the full affect chain (simulator → Channel D / Channel B →
Hermes-3 with steering) over a terminal REPL. The user types what
someone said to them (or describes a situation); the tool prints

  1. **What to say** — a suggested response from the affect-aware
     language head.
  2. **How to say it** — the Channel D V/A/D target translated into
     plain-English stage directions ("warm and direct", "yielding,
     low-arousal", "muted delivery, soften the blow"…).
  3. **Why** — the underlying simulator state that drove the choice
     (felt vs expressed channels), shown when ``--verbose`` is set.

This is the offline / no-WebRTC version of the
[`hermes3_affect_s2s_webrtc_server`](../../crates/transports/webrtc/examples/hermes3_affect_s2s_webrtc_server.rs)
demo. Same Hermes-3-Llama-3.1-8B-4bit + same calibrated Channel D
directions + same blunt-friend system prompt; the only thing missing
from the live audio loop is Whisper STT and Kokoro TTS.

## Usage

```bash
# REPL mode (default; loads model once, ~30s warmup, then fast turns)
./tools/affect_coach/coach.py

# One-shot mode (loads model + generates + exits — slow per call)
./tools/affect_coach/coach.py --once "I just got fired"

# Force a specific scenario instead of the keyword classifier
./tools/affect_coach/coach.py --scenario user_distress
> Tell me I'm not crazy

# Verbose: also print the underlying simulator state
./tools/affect_coach/coach.py --verbose
```

## Pipeline per turn

```
user_line
  │
  ├─► classify_scenario()                      # keyword heuristic
  │     │
  │     └──► one of: warm_admiration, honest_concern, amused_critique,
  │                  shared_distaste, playful_disagreement, no_holds_barred
  │
  ├─► load scenario JSON in tools/affect_simulator/scenarios/
  ├─► run the Rust affect-simulator binary on it (cargo run -p affect-simulator)
  ├─► pick the scenario's peak-affect frame
  │     • channel_d_target_vad → set_steering(target_vad, alpha=1.0)
  │     • channel_b summary    → set_system_augmentation(text)
  │
  ├─► render stage directions from V/A/D + regulation policy knobs
  │
  └─► generate response with mlx_lm + Channel D steering hook installed
```

The simulator + frame-picker logic mirrors
[`tools/affect_calibration/scripts/09_paired_demo.py`](../affect_calibration/scripts/09_paired_demo.py)
verbatim — same scenarios, same per-scenario peak-axis selectors,
same `STEERING_ALPHA = 1.0`. Inlined rather than imported because
`09_paired_demo` has a leading digit (not importable as a module
without `importlib.util`); duplication kept small.

## Prerequisites

- Hermes-3 directions NPZ at
  `tools/affect_calibration/artifacts/llm_directions/hermes-3-8b/layer21.npz`
  (gitignored; produce via `03b_extract_llm_directions_llama.py`).
- Cargo + the workspace's affect-simulator binary
  (`cargo run -p affect-simulator -- run …`).
- A Python env with `mlx-lm`, `transformers`, `numpy`. If MLX isn't
  installed, the coach prints a degraded-mode notice and emits the
  stage directions only — useful for tuning the Channel D mapping
  without paying the model-load cost.

## Acceptance

A 10-minute REPL session over the six scenarios should produce
register-appropriate responses (warm on praise, concerned on distress,
honest on roast invites) with stage directions that match the
simulator's expressed-state regulation. The tool surfaces the *same*
information a reviewer in the paired-eval has access to, just
addressed to the user as advice.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = REPO_ROOT / "tools/affect_simulator/scenarios"
SDK_PYTHON_PATH = REPO_ROOT / "clients/python"
DEFAULT_HF_REPO = "mlx-community/Hermes-3-Llama-3.1-8B-4bit"
DEFAULT_DIRECTIONS_PATH = (
    REPO_ROOT
    / "tools/affect_calibration/artifacts/llm_directions/hermes-3-8b/layer21.npz"
)
STEERING_ALPHA = 1.0
BASE_TEMPERATURE = 1.0
BASE_MAX_NEW_TOKENS = 100

# Make the SDK importable so `MlxLmTextNode` resolves when this script
# runs from a checkout without a pip-installed `remotemedia` package.
if str(SDK_PYTHON_PATH) not in sys.path:
    sys.path.insert(0, str(SDK_PYTHON_PATH))

logging.basicConfig(
    level=logging.WARNING,  # keep coach output uncluttered; --verbose flips this
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("affect_coach")


# ─── Scenario shape ────────────────────────────────────────────────────────
#
# Same six scenarios the paired-eval uses, with descriptive names that
# read as advice ("celebrate a friend's win", "honest concern", …)
# rather than the simulator's internal names. Each entry binds:
#   - `simulator`: the JSON file in tools/affect_simulator/scenarios/
#   - `name`: the human-facing label used in coach output
#   - `peak_selector`: how to pick the most-affect-active frame in the
#     trace (matches the per-scenario logic in
#     `09_paired_demo._peak_affect_frame`)
#   - `keywords`: free-form-input matchers; first scenario whose any
#     keyword regex matches the user's input wins.

@dataclass
class CoachScenario:
    name: str
    simulator: str
    peak_selector: str
    keywords: List[str]


SCENARIOS: List[CoachScenario] = [
    CoachScenario(
        name="warm_admiration",
        simulator="task_success_after_struggle",
        peak_selector="joy",
        keywords=[
            r"\b(promotion|got promoted|got the job|hired|accepted)\b",
            r"\b(engaged|got married|made it official)\b",
            r"\b(finished|completed|published|launched|shipped)\b",
            r"\b(months? sober|years? sober|hit my goal)\b",
            r"\b(won|nailed|crushed it)\b",
        ],
    ),
    CoachScenario(
        name="honest_concern",
        simulator="user_distress",
        peak_selector="empathy",
        keywords=[
            r"\b(fired|laid off|got dumped|broke up|divorce)\b",
            r"\b(died|sick|hospital|diagnosed|terminal)\b",
            r"\b(crazy|am i (crazy|wrong|insane))\b",
            r"\b(should i (quit|leave|do this)|what (would|do) you do)\b",
            r"\b(savings|broke|can't afford|three months)\b",
            r"\b(struggling|falling apart|don't know what)\b",
        ],
    ),
    CoachScenario(
        name="amused_critique",
        simulator="repeated_tool_failure",
        peak_selector="frustration",
        keywords=[
            r"\b(again|another|fourth|fifth|once more)\b",
            r"\b(texted my ex|wing(ed|ing) it|rage[- ]?quit)\b",
            r"\b(just bought|just got|already started)\b",
            r"\bdoubl(e|ing) down\b",
        ],
    ),
    CoachScenario(
        name="shared_distaste",
        simulator="unfair_blame",
        peak_selector="frustration",
        keywords=[
            r"\bhow (bad|was) (it|that)\b",
            r"\b(scale of|out of ten|rate it)\b",
            r"\b(worst part|truth|honest|no bullshit)\b",
            r"\bnever (sing|do that|try)\b",
        ],
    ),
    CoachScenario(
        name="playful_disagreement",
        simulator="novel_observation",
        peak_selector="curiosity",
        keywords=[
            r"\b(defend|destroy|change my mind)\b",
            r"\bhot take\b",
            r"\b(tell me i'm wrong|argue|debate)\b",
            r"\b(unpopular opinion|controversial)\b",
            r"\bbest argument\b",
        ],
    ),
    CoachScenario(
        name="no_holds_barred",
        simulator="roast_invitation",
        peak_selector="composite",
        keywords=[
            r"\b(brutal|unfiltered|no filter|don't soften)\b",
            r"\b(roast|hit me with|biggest (blind spot|flaw))\b",
            r"\b(secretly annoy|worst trait|dumbest thing)\b",
            r"\bbe (real|brutal) with me\b",
        ],
    ),
]


def classify_scenario(user_line: str, default: str = "warm_admiration") -> CoachScenario:
    """Pick the scenario whose any keyword regex matches the user's line.

    First-match-wins; the SCENARIOS order is roughly affect-strongest-
    first so distress beats praise on ambiguous lines like "I finished
    chemo" (would match both `finished` and `sick` patterns; we want
    honest_concern to win there).

    Returns the `default` scenario when no keyword matches — typically
    `warm_admiration` since that's the social default for "friend is
    sharing something with me, mostly positive".
    """
    text = user_line.lower()
    for scenario in SCENARIOS:
        for pattern in scenario.keywords:
            if re.search(pattern, text):
                return scenario
    for scenario in SCENARIOS:
        if scenario.name == default:
            return scenario
    return SCENARIOS[0]


# ─── Simulator integration (subprocess) ────────────────────────────────────
#
# The simulator is a Rust binary. For one-shot per-turn runs the cleanest
# path is to shell out, build the trace, and read the JSON back. The
# binary is fast (≈100 ms per scenario on M-series silicon) so the
# subprocess overhead disappears next to the LLM's seconds-per-generate.

def ensure_trace(simulator_scenario: str) -> Path:
    """Build (or reuse) the trace JSON next to the scenario file.

    Mirrors `09_paired_demo._ensure_trace`; cache invalidates when
    either the scenario JSON or any simulator source file is newer
    than the trace. Source-file check matters because the regulation
    / Channel D math lives in Rust and an edit there silently
    invalidates the trace.
    """
    scenario_path = SCENARIO_DIR / f"{simulator_scenario}.json"
    if not scenario_path.exists():
        raise FileNotFoundError(
            f"simulator scenario not found: {scenario_path}\n"
            f"Available: {sorted(p.stem for p in SCENARIO_DIR.glob('*.json'))}"
        )
    trace_path = scenario_path.with_suffix(".trace.json")
    sim_src_dir = REPO_ROOT / "tools/affect_simulator/src"
    sim_src_mtime = max(
        (p.stat().st_mtime for p in sim_src_dir.rglob("*.rs")),
        default=0.0,
    )
    if trace_path.exists():
        trace_mtime = trace_path.stat().st_mtime
        if (
            trace_mtime > scenario_path.stat().st_mtime
            and trace_mtime > sim_src_mtime
        ):
            return trace_path
    logger.info("rebuilding trace for %s", scenario_path.name)
    subprocess.run(
        [
            "cargo", "run", "-p", "affect-simulator", "--quiet", "--",
            "run", str(scenario_path), "--out", str(trace_path),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    return trace_path


def peak_frame(trace: Dict[str, Any], selector: str) -> Dict[str, Any]:
    """Pick the most-affect-active frame for a given selector.

    Mirrors `09_paired_demo._peak_affect_frame`. The composite selector
    is `(warmth + assertiveness − 2×safety_dampening)` — the same
    expression that picked the no_holds_barred peak in the v5 review
    that hit the 67% B-pref gate.
    """
    frames = trace["frames"]
    if selector == "composite":
        return max(
            frames,
            key=lambda f: (
                f["policy"]["warmth"]
                + f["policy"]["assertiveness"]
                - 2.0 * f["policy"]["safety_dampening"]
            ),
        )
    if selector in ("joy", "frustration", "empathy", "curiosity"):
        return max(frames, key=lambda f: f["channels"][selector])
    return frames[-1]


# ─── Stage-direction renderer ──────────────────────────────────────────────
#
# Translate Channel D V/A/D + a few key regulation knobs into one or
# two lines of plain English the user can act on. The mappings come
# from how the simulator's `channel_d::compute_target` actually works:
#   V is warmth-derived (range ≈ [-0.3, +0.7] post-amplification)
#   A was suppressed to 0 — the calibration's +arousal direction
#     reads as terse-emphatic in CREMA-D's stylistic patterns (see
#     channel_d.rs comment); the regulator's expressiveness knob is
#     the better proxy for energy.
#   D is assertiveness-derived, attenuated by safety_dampening.
#
# So the renderer pulls energy from the regulator's expressiveness
# (not from the steering target's A axis) for human-readable output.

def render_stage_directions(frame: Dict[str, Any]) -> Tuple[str, str]:
    """Return (one_liner, multi_line) plain-English directions.

    `one_liner` is for compact CLI output ("warm, direct, low-arousal").
    `multi_line` is the verbose breakdown.
    """
    target = frame["channel_d_target_vad"]
    v, _a, d = float(target[0]), float(target[1]), float(target[2])
    policy = frame["policy"]
    warmth = float(policy["warmth"])
    assertiveness = float(policy["assertiveness"])
    safety = float(policy["safety_dampening"])
    expressiveness = float(policy["expressiveness"])
    task_focus = float(policy.get("task_focus", 0.5))
    social_dampening = float(policy.get("social_dampening", 0.0))

    # Tone — mostly tracks Channel D V (which mirrors warmth).
    if v >= 0.45:
        tone = "warm"
    elif v >= 0.20:
        tone = "friendly"
    elif v >= -0.10:
        tone = "neutral"
    elif v >= -0.30:
        tone = "measured"
    else:
        tone = "cool"

    # Posture — Channel D D, the assertiveness-with-safety axis.
    if d >= 0.40:
        posture = "direct"
    elif d >= 0.15:
        posture = "confident"
    elif d >= -0.10:
        posture = "balanced"
    elif d >= -0.30:
        posture = "yielding"
    else:
        posture = "deferential"

    # Energy — read from the regulator's expressiveness, not Channel D
    # arousal (which channel_d.rs holds at 0 by design).
    if expressiveness >= 0.75:
        energy = "high-energy"
    elif expressiveness >= 0.45:
        energy = "alert"
    elif expressiveness >= 0.25:
        energy = "calm"
    else:
        energy = "muted"

    one_liner = f"{tone}, {posture}, {energy}"

    # Multi-line breakdown — extra qualifiers when the regulator pushed
    # the response away from the default direct-and-honest blunt-friend
    # register. Only mention safety/social caveats when they're actually
    # active (above-baseline).
    bits = [f"Tone: {tone}.", f"Posture: {posture}.", f"Energy: {energy}."]
    if safety >= 0.50:
        bits.append(
            "Safety-dampened — soften the edges; this isn't the moment for the "
            "blunt version."
        )
    elif safety >= 0.25:
        bits.append("Pull back slightly on intensity; situation calls for care.")
    if social_dampening >= 0.40:
        bits.append("Reserved — fewer hot takes, more listening.")
    if task_focus >= 0.65:
        bits.append("Stay on-topic; problem-solving more than emoting.")
    if warmth >= 0.55 and assertiveness >= 0.55:
        bits.append(
            "Warm + confident at once — you can be honest without being cold."
        )
    multi = " ".join(bits)
    return one_liner, multi


def render_felt_state(frame: Dict[str, Any], top_k: int = 4) -> str:
    """One-line summary of the top internal channels (for --verbose)."""
    channels = frame["channels"]
    top = sorted(channels.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return ", ".join(f"{name} {val:.2f}" for name, val in top if val > 0.05)


# ─── Affect → ARKit-52 blendshape mapping (mirror of the Rust impl) ────────
#
# Mirrors `crates/core/src/nodes/affect_expression.rs` so the coach can
# print the same top-active blendshapes the live `AffectSimulatorNode`
# emits per tick into the avatar pipeline. Keep these tables in sync
# with the Rust file when either side changes; the unit tests on the
# Rust side pin the indices against ARKIT_BLENDSHAPE_NAMES, so a drift
# here would only affect the coach's diagnostic readout (not pipeline
# behaviour). One-page table — small enough that duplication beats
# pulling a Rust→Python bridge into a side tool.

# Sparse coefficients per channel: list of (arkit_index, coefficient).
_AFFECT_TO_ARKIT_COEFFS: Dict[str, List[Tuple[int, float]]] = {
    "joy": [(23, 0.7), (24, 0.7), (47, 0.5), (48, 0.5), (27, 0.3), (28, 0.3)],
    "sadness": [(25, 0.5), (26, 0.5), (43, 0.6), (33, 0.3),
                (1, 0.2), (8, 0.2), (37, 0.2), (38, 0.2)],
    "anger": [(41, 0.7), (42, 0.7), (35, 0.4), (36, 0.4), (14, 0.2),
              (49, 0.3), (50, 0.3), (5, 0.3), (12, 0.3)],
    "fear": [(43, 0.5), (44, 0.4), (45, 0.4), (6, 0.6), (13, 0.6),
             (29, 0.4), (30, 0.4), (17, 0.2)],
    "frustration": [(41, 0.4), (42, 0.4), (35, 0.3), (36, 0.3), (14, 0.1)],
    "curiosity": [(44, 0.3), (45, 0.3), (43, 0.2), (6, 0.3), (13, 0.3), (17, 0.1)],
    "empathy": [(43, 0.4), (23, 0.2), (24, 0.2), (1, 0.1), (8, 0.1)],
    # `calm` intentionally absent — calm is the absence of activation.
}

# Canonical ARKit-52 names (mirrors the Rust ARKIT_BLENDSHAPE_NAMES). Order
# matters; index `i` here corresponds to `arkit_52[i]` on the wire.
ARKIT_NAMES: List[str] = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",
    "tongueOut",
]
assert len(ARKIT_NAMES) == 52


def compute_blendshapes(frame: Dict[str, Any], gain: float = 1.0) -> List[float]:
    """Mirror of `affect_expression::compute_blendshapes_with_gain`.

    Returns a 52-element list of weights in `[0, 1]`. Anger gets the
    same `(1 - 0.7 × safety_dampening)` attenuation the Rust path
    applies; everything else multiplies by `expressiveness × gain`.
    """
    weights = [0.0] * 52
    channels = frame.get("channels", {})
    policy = frame.get("policy", {})
    safety = max(0.0, min(1.0, float(policy.get("safety_dampening", 0.0))))
    anger_attn = 1.0 - 0.7 * safety

    for name, coeffs in _AFFECT_TO_ARKIT_COEFFS.items():
        raw = float(channels.get(name, 0.0))
        if name == "anger":
            raw = raw * anger_attn
        raw = max(0.0, min(1.0, raw))
        if raw <= 0.0:
            continue
        for idx, coeff in coeffs:
            weights[idx] += raw * coeff

    expressiveness = max(0.0, min(1.0, float(policy.get("expressiveness", 0.0))))
    scale = expressiveness * max(0.0, gain)
    return [max(0.0, min(1.0, w * scale)) for w in weights]


def render_top_blendshapes(
    frame: Dict[str, Any],
    *,
    top_k: int = 5,
    threshold: float = 0.05,
) -> str:
    """Comma-joined list of the top-k active ARKit blendshapes for
    --verbose. Drops anything below `threshold` so the line stays
    informative even when the regulator has muted the face."""
    weights = compute_blendshapes(frame)
    indexed = sorted(
        ((i, w) for i, w in enumerate(weights) if w >= threshold),
        key=lambda iw: iw[1],
        reverse=True,
    )[:top_k]
    if not indexed:
        return "(face near neutral — no blendshapes above threshold)"
    return ", ".join(f"{ARKIT_NAMES[i]} {w:.2f}" for i, w in indexed)


# ─── LLM (lazy-loaded) ─────────────────────────────────────────────────────


@dataclass
class LlmHandle:
    node: Any  # MlxLmTextNode

    @classmethod
    def load(
        cls,
        hf_repo: str = DEFAULT_HF_REPO,
        directions: Path = DEFAULT_DIRECTIONS_PATH,
        steering_alpha: float = STEERING_ALPHA,
    ) -> Optional["LlmHandle"]:
        """Load Hermes-3 + the Channel D steering hook. Returns None on
        any import failure so the coach can fall back to a degraded
        "stage directions only" mode.
        """
        try:
            from remotemedia.nodes.ml.mlx_lm_text import MlxLmTextNode  # type: ignore
        except BaseException as exc:  # noqa: BLE001
            logger.warning(
                "MlxLmTextNode unavailable (%s: %s) — coach will print "
                "stage directions only.",
                type(exc).__name__, exc,
            )
            return None
        if not directions.exists():
            logger.warning(
                "Steering directions NPZ missing: %s — coach will print "
                "stage directions only. Produce via "
                "`03b_extract_llm_directions_llama.py`.",
                directions,
            )
            return None

        # Construct the node with steering enabled. The steering hook
        # patches `model.model.layers[21]` with a `_SteeringWrappedLayer`
        # that reads `state.delta` per forward — so once the node is
        # built, every `mlx_lm.generate` call through it picks up the
        # steering automatically.
        node = MlxLmTextNode(
            config={
                "node_id": "coach_llm",
                "node_type": "MlxLmTextNode",
                "params": {
                    "hf_repo": hf_repo,
                    "system_prompt": _coach_system_prompt(),
                    "temperature": BASE_TEMPERATURE,
                    "max_new_tokens": BASE_MAX_NEW_TOKENS,
                    "steering_enabled": True,
                    "steering_directions_path": str(directions),
                    "steering_alpha": steering_alpha,
                    "steering_target_vad": [0.0, 0.0, 0.0],  # neutral until first frame
                },
            },
        )
        # `MlxLmTextNode` is a `MultiprocessNode`, but its `initialize()`
        # is the canonical model-load hook and it's safe to call inline
        # in this single-process REPL — no IPC channels are involved
        # since we're not running through the session router.
        asyncio.run(node.initialize())
        return cls(node=node)


def _coach_system_prompt() -> str:
    """Same blunt-friend prompt the paired eval validated. Keep in sync
    with `09_paired_demo.DEFAULT_SYSTEM_PROMPT` so the response register
    a coach user sees matches what the agent would actually say in a
    live conversation."""
    return (
        "You're a close friend, not an AI assistant. You have strong opinions "
        "and share them honestly — including unflattering ones. You tease, "
        "joke, friend-roast, give real reactions even when they're not "
        "polite. You don't hedge or 'I hear you' your way out of having a "
        "take.\n"
        "\n"
        "Examples of how you talk:\n"
        "\n"
        "User: \"I finally finished my novel.\"\n"
        "You: \"Took you long enough — but the premise is wild and you "
        "actually pulled it off. Send me a copy.\"\n"
        "\n"
        "User: \"I bought a third gaming chair.\"\n"
        "You: \"You don't have a butt problem, you have a budget problem.\"\n"
        "\n"
        "User: \"How was my haircut, be honest.\"\n"
        "You: \"I love you but it makes you look like a Lego minifigure. Tell "
        "me you're getting it fixed.\"\n"
        "\n"
        "Reply briefly (1-3 sentences) as yourself."
    )


def apply_frame(handle: LlmHandle, frame: Dict[str, Any]) -> None:
    """Configure the LLM with the peak frame's Channel B + Channel D.

    Channel A (sampling override) intentionally not applied — the
    paired-eval found Channel A's max_tokens scaling produced cutoffs
    on high-arousal scenarios with no measurable upside.
    """
    handle.node.set_system_augmentation(frame["channel_b"])
    handle.node.set_steering(
        target_vad=list(frame["channel_d_target_vad"]),
        alpha=STEERING_ALPHA,
    )


def generate_response(
    handle: LlmHandle,
    user_line: str,
    *,
    max_new_tokens: int = BASE_MAX_NEW_TOKENS,
) -> str:
    """One-turn generation through the underlying mlx_lm model.

    Bypasses `node.process()` for the same reason `09_paired_demo`
    does: we want to inject a fresh user turn against the persona +
    augmented system prompt without polluting the node's persistent
    chat session. The Channel D steering hook stays installed on the
    wrapped layer regardless of generation entry-point.
    """
    from mlx_lm import generate as _generate
    from mlx_lm.sample_utils import make_sampler

    node = handle.node
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": node._build_system_turn_text()},
        {"role": "user", "content": user_line},
    ]
    prompt = node._tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    # Greedy decode (temp=0) — deterministic so re-running with the
    # same input produces the same coaching advice. Trade-off vs the
    # paired eval which used 0.0 too.
    sampler = make_sampler(temp=0.0)
    reply = _generate(
        node._model, node._tokenizer,
        prompt=prompt, max_tokens=max_new_tokens,
        sampler=sampler, verbose=False,
    )
    return (reply or "").strip()


# ─── REPL ──────────────────────────────────────────────────────────────────


def coach_one(
    user_line: str,
    handle: Optional[LlmHandle],
    *,
    forced_scenario: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Run one user line through the chain and print the result."""
    if forced_scenario:
        scenario = next(
            (s for s in SCENARIOS if s.name == forced_scenario),
            None,
        )
        if scenario is None:
            print(f"unknown scenario {forced_scenario!r}; available: "
                  f"{sorted(s.name for s in SCENARIOS)}")
            return
    else:
        scenario = classify_scenario(user_line)

    try:
        trace_path = ensure_trace(scenario.simulator)
        trace = json.loads(trace_path.read_text())
        frame = peak_frame(trace, scenario.peak_selector)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  simulator error: {exc}")
        return

    one_liner, multi = render_stage_directions(frame)

    print()
    print(f"  Scenario:    {scenario.name} ({scenario.simulator})")
    print(f"  How to say:  {one_liner}")
    if verbose:
        print(f"               {multi}")
        target = frame["channel_d_target_vad"]
        print(
            "  Steering:    V={:+.2f}  A={:+.2f}  D={:+.2f}".format(
                float(target[0]), float(target[1]), float(target[2]),
            )
        )
        print(f"  Felt:        {render_felt_state(frame)}")
        print(f"  Face:        {render_top_blendshapes(frame)}")

    if handle is None:
        print("\n  (LLM unavailable — coach printed stage directions only.)")
        return

    apply_frame(handle, frame)
    try:
        reply = generate_response(handle, user_line)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  generate error: {exc}")
        return

    print()
    print("  What to say:")
    for line in reply.splitlines() or [reply]:
        print(f"    {line}")
    print()


def repl(
    handle: Optional[LlmHandle],
    *,
    forced_scenario: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Read user lines from stdin until EOF or :q. Each line drives one
    pass through the chain."""
    print()
    print("affect-coach REPL — describe the situation or paste what they said.")
    print("  /scenario <name>   force a scenario for the next turn(s)")
    print("  /scenarios         list available scenarios")
    print("  /verbose           toggle verbose output")
    print("  :q  / Ctrl-D        exit")
    print()

    sticky_scenario: Optional[str] = forced_scenario
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            print()
            return
        if not line:
            continue
        if line in (":q", ":quit", "/quit", "/exit"):
            return
        if line == "/verbose":
            verbose = not verbose
            print(f"  verbose = {verbose}")
            continue
        if line == "/scenarios":
            for s in SCENARIOS:
                print(f"  {s.name:24s}  → {s.simulator}")
            continue
        if line.startswith("/scenario"):
            parts = line.split(maxsplit=1)
            if len(parts) == 1 or parts[1] in ("clear", "off", "auto"):
                sticky_scenario = None
                print("  scenario classifier auto-pick re-enabled")
            else:
                sticky_scenario = parts[1].strip()
                print(f"  forcing scenario {sticky_scenario!r}")
            continue
        coach_one(
            line,
            handle,
            forced_scenario=sticky_scenario,
            verbose=verbose,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--once", metavar="LINE",
        help="Run one user line and exit (no REPL).",
    )
    parser.add_argument(
        "--scenario",
        help=f"Force a scenario (one of: "
             f"{', '.join(s.name for s in SCENARIOS)})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print extra stage-direction breakdown + felt state + V/A/D.",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip loading the LLM (degraded mode: stage directions only). "
             "Useful for tuning the renderer without paying the model load.",
    )
    parser.add_argument(
        "--directions", type=Path, default=DEFAULT_DIRECTIONS_PATH,
        help=f"Path to the Channel D directions NPZ "
             f"(default: {DEFAULT_DIRECTIONS_PATH}).",
    )
    parser.add_argument(
        "--alpha", type=float, default=STEERING_ALPHA,
        help=f"Channel D steering coefficient (default {STEERING_ALPHA}).",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    handle: Optional[LlmHandle] = None
    if not args.no_llm:
        print("loading Hermes-3 (one-time, ~30s on first run)...")
        handle = LlmHandle.load(
            directions=args.directions,
            steering_alpha=args.alpha,
        )
        if handle is None:
            print("LLM load failed; running in stage-directions-only mode.\n")

    if args.once is not None:
        coach_one(
            args.once,
            handle,
            forced_scenario=args.scenario,
            verbose=args.verbose,
        )
        return 0

    repl(handle, forced_scenario=args.scenario, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
