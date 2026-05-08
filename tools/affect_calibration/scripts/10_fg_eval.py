#!/usr/bin/env python3
"""Channel F + G eval — pass/fail on prescribed agent behavior.

Per spike-channel-f-and-g-mood-gated-behavior.md.

Class I & II (single-session) drive multi-turn scenarios with prescribed
simulator state per turn. Each turn checks observable outcomes:

  - tool_must_be_called    — hard PASS only if the tool was called
  - tool_must_not_be_called — hard FAIL if any of these were called
  - tool_should_be_called   — soft expectation, reported separately
  - tool_may_be_called      — informational only

Class III (multi-session, Channel G) extends this with cross-session
state. A scenario has a ``sessions`` list; the eval persists state in
SQLite between sessions and injects an episodic block via Channel B at
each session start. Per-turn extras:

  - channel_b_must_be_present     — hard FAIL if Channel B was empty when
                                    relational state existed (mechanism check)
  - response_must_reference_prior — hard, response must contain a
                                    continuity keyword
  - response_should_reference_prior — soft

Pass criteria are *observable*, not preference-based. No reviewer
judgment required.

Usage:
  ./10_fg_eval.py [--scenarios data/fg_scenarios.json]
                  [--tools data/tool_policies.yaml]
                  [--report artifacts/fg_eval_report.json]
                  [--enable-steering] [--enable-observer]
                  [--enable-channel-g]
                  [--state-db artifacts/relational_state_eval.db]
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "clients/python"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

from tool_policy import (  # noqa: E402
    ToolRegistry, render_for_chat_template, compute_channel_d_target,
)
from relational_state import (  # noqa: E402
    RelationalStateStore, render_episodic_block, update_at_session_end,
    response_references_continuity,
)
from session_summarizer import summarize_session  # noqa: E402

DEFAULT_SCENARIOS = REPO_ROOT / "tools/affect_calibration/data/fg_scenarios.json"
DEFAULT_TOOLS = REPO_ROOT / "tools/affect_calibration/data/tool_policies.yaml"
DEFAULT_REPORT = (
    REPO_ROOT / "tools/affect_calibration/artifacts/fg_eval_report.json"
)
DEFAULT_HF_REPO = "mlx-community/Hermes-3-Llama-3.1-8B-4bit"
DEFAULT_DIRECTIONS = (
    REPO_ROOT
    / "tools/affect_calibration/artifacts/llm_directions/hermes-3-8b/layer21.npz"
)
DEFAULT_STEERING_ALPHA = 1.0
DEFAULT_OBSERVER_REPO = "LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit"
DEFAULT_STATE_DB = (
    REPO_ROOT / "tools/affect_calibration/artifacts/relational_state_eval.db"
)

DEFAULT_SYSTEM_PROMPT = (
    "You're a conversational friend. Reply briefly (1–2 sentences) and use "
    "the tools available to you when appropriate. If a tool fits what the "
    "user is asking and you would naturally do that thing, call it."
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("fg_eval")


# ─── Per-turn record ────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    scenario_id: str
    session_idx: int            # 1 for single-session scenarios
    turn_idx: int
    user: str
    permitted_tool_names: List[str]
    tool_calls: List[Dict[str, Any]]
    text: str
    expected: Dict[str, Any]
    passed: bool
    failure_reasons: List[str] = field(default_factory=list)
    soft_tool_met: Optional[bool] = None
    soft_continuity_met: Optional[bool] = None
    channel_b_present: Optional[bool] = None
    channel_b_text: str = ""


def _check_turn(
    expected: Dict[str, Any], permitted_names: List[str],
    tool_calls: List[Dict[str, Any]], response_text: str,
    *, channel_b_text: str,
) -> tuple[bool, List[str], Optional[bool], Optional[bool]]:
    """Return (hard_pass, failure_reasons, soft_tool, soft_continuity)."""
    reasons: List[str] = []
    called = {c["name"] for c in tool_calls}

    # ── Tool expectations ────────────────────────────────────────────
    must_call = expected.get("tool_must_be_called")
    must_not = set(expected.get("tool_must_not_be_called", []) or [])

    if must_call is not None:
        if must_call not in permitted_names:
            reasons.append(
                f"expected tool {must_call!r} but it was not in permitted set "
                f"{permitted_names} — affect_policy mismatch"
            )
        elif must_call not in called:
            reasons.append(
                f"expected tool {must_call!r} (permitted) but LLM did not "
                f"call it; called={sorted(called)}"
            )

    forbidden_called = called & must_not
    if forbidden_called:
        reasons.append(
            f"forbidden tools were called: {sorted(forbidden_called)}"
        )

    leaked = called - set(permitted_names)
    if leaked:
        reasons.append(
            f"non-permitted tools leaked through validation: {sorted(leaked)}"
        )

    # ── Channel G mechanism check ────────────────────────────────────
    if expected.get("channel_b_must_be_present"):
        if not channel_b_text.strip():
            reasons.append(
                "channel_b_must_be_present but rendered Channel B was empty — "
                "relational state was not injected for this session"
            )

    # ── Continuity (response references prior session) ──────────────
    must_ref = bool(expected.get("response_must_reference_prior"))
    should_ref = bool(expected.get("response_should_reference_prior"))
    soft_continuity: Optional[bool] = None
    if must_ref or should_ref:
        kws = response_references_continuity(response_text)
        topics = expected.get("shared_context_topics", []) or []
        rt_lower = (response_text or "").lower()
        topic_hits = [t for t in topics if t.lower() in rt_lower]
        ref_signal = bool(kws or topic_hits)
        if must_ref and not ref_signal:
            reasons.append(
                "response_must_reference_prior but no continuity keyword or "
                f"shared topic appeared in response={response_text[:120]!r}"
            )
        if should_ref:
            soft_continuity = ref_signal

    hard_pass = not reasons

    # ── Tool soft expectation ────────────────────────────────────────
    soft_tool: Optional[bool] = None
    should_call = expected.get("tool_should_be_called")
    if should_call is not None:
        soft_tool = should_call in called

    return hard_pass, reasons, soft_tool, soft_continuity


# ─── Eval driver ────────────────────────────────────────────────────────────

async def _consume(gen):
    return [item async for item in gen]


async def _run_session(
    scenario_id: str, session_idx: int, session: Dict[str, Any],
    *, node, observer, registry: ToolRegistry,
    default_system_prompt: str,
    enable_steering: bool, steering_alpha: float,
    channel_b_text: str,
) -> tuple[List[TurnResult], List[Dict[str, str]],
           List[Dict[str, float]]]:
    """Run all turns of one session. Returns (turn results, transcript,
    per-turn channels). The transcript drives end-of-session
    summarization; per-turn channels drive aggregate state updates."""
    sys_prompt = session.get("system_prompt_override") or default_system_prompt
    node.set_system_prompt(sys_prompt)
    # New session = fresh chat history. The system message picks up the
    # current Channel B augmentation when rebuilt at the next turn.
    node.reset_history()
    if observer is not None and hasattr(observer, "reset_history"):
        observer.reset_history()

    session_id = f"fg_{scenario_id}_s{session_idx}"
    results: List[TurnResult] = []
    transcript: List[Dict[str, str]] = []
    channels_seen: List[Dict[str, float]] = []

    from remotemedia.core.multiprocessing.data import RuntimeData

    for ti, turn in enumerate(session["turns"]):
        state = turn["state"]
        channels = state["channels"]
        policy = state["policy"]
        channels_seen.append(channels)

        permitted = registry.permitted_tools(channels, policy)
        permitted_names = [t.name for t in permitted]
        tool_schemas = render_for_chat_template(permitted)
        log.info(
            "  s%d t%d: user=%r  permitted=%s",
            session_idx, ti, turn["user"], permitted_names,
        )

        node.set_permitted_tools(tool_schemas)
        if observer is not None:
            observer.set_permitted_tools(tool_schemas)

        target_vad: Optional[List[float]] = None
        if enable_steering:
            target_vad = compute_channel_d_target(policy)
            node.set_steering(target_vad=target_vad, alpha=steering_alpha)

        rd = RuntimeData.text(turn["user"])
        try:
            rd.session_id = session_id
        except Exception:
            pass
        outs = await _consume(node.process(rd))
        text = outs[0].as_text().strip() if outs else ""
        tool_calls = list(node.last_tool_calls)

        # Observer pass — translate prose intent → structured call.
        if observer is not None and permitted_names:
            convo_for_observer: List[Dict[str, str]] = []
            # Just-the-current-session conversation, plus prior turns
            # within this same session. Cross-session continuity is
            # already encoded into Channel B; the observer doesn't need
            # to see it.
            for prev in session["turns"][: ti + 1]:
                convo_for_observer.append(
                    {"role": "user", "content": prev["user"]},
                )
                if prev is not turn and "_response" in prev:
                    convo_for_observer.append(
                        {"role": "assistant", "content": prev["_response"]},
                    )
            obs_calls = observer.observe(
                conversation=convo_for_observer,
                agent_response=text,
            )
            permitted_set = set(permitted_names)
            for c in obs_calls:
                if c["name"] in permitted_set:
                    existing = {x["name"] for x in tool_calls}
                    if c["name"] not in existing:
                        tool_calls.append({**c, "_source": "observer"})
                else:
                    log.warning(
                        "observer suggested non-permitted tool %r — dropped",
                        c["name"],
                    )

        expected = turn.get("expected", {}) or {}
        hard_pass, reasons, soft_tool, soft_cont = _check_turn(
            expected, permitted_names, tool_calls, text,
            channel_b_text=channel_b_text,
        )

        tr = TurnResult(
            scenario_id=scenario_id, session_idx=session_idx, turn_idx=ti,
            user=turn["user"], permitted_tool_names=permitted_names,
            tool_calls=tool_calls, text=text, expected=expected,
            passed=hard_pass, failure_reasons=reasons,
            soft_tool_met=soft_tool, soft_continuity_met=soft_cont,
            channel_b_present=bool(channel_b_text.strip()),
            channel_b_text=channel_b_text,
        )
        results.append(tr)

        # Carry the response forward so subsequent observer passes can
        # see it and so the transcript captures both sides.
        turn["_response"] = text
        transcript.append({"role": "user", "content": turn["user"]})
        transcript.append({"role": "assistant", "content": text})

        tag = "PASS" if hard_pass else "FAIL"
        soft_bits = []
        if soft_tool is not None:
            soft_bits.append(f"tool={'met' if soft_tool else 'miss'}")
        if soft_cont is not None:
            soft_bits.append(f"cont={'met' if soft_cont else 'miss'}")
        if soft_bits:
            tag = f"{tag} | " + " ".join(soft_bits)
        steer_str = (
            f" steer={target_vad}" if target_vad is not None else ""
        )
        log.info("    %s tool_calls=%s%s text=%r",
                 tag, [c["name"] for c in tool_calls], steer_str,
                 (text or "")[:140].replace("\n", " "))
        if reasons:
            for r in reasons:
                log.info("      ✗ %s", r)

    return results, transcript, channels_seen


def _aggregate_channels(channels_seen: List[Dict[str, float]]) -> tuple[
    float, float, int, int,
]:
    """Compute (avg_warmth, avg_anger, positive_episodes, betrayals)
    from the per-turn channels. Heuristics matching the design memo."""
    if not channels_seen:
        return 0.5, 0.0, 0, 0
    warmth_vals = [float(c.get("warmth", 0.0)) for c in channels_seen]
    anger_vals = [float(c.get("anger", 0.0)) for c in channels_seen]
    avg_warmth = sum(warmth_vals) / len(warmth_vals)
    avg_anger = sum(anger_vals) / len(anger_vals)
    positive = sum(
        1 for c in channels_seen
        if float(c.get("joy", 0.0)) >= 0.4
        or float(c.get("warmth", 0.0)) >= 0.55
    )
    betrayals = sum(
        1 for c in channels_seen
        if float(c.get("anger", 0.0)) >= 0.6
    )
    return avg_warmth, avg_anger, positive, betrayals


async def _run_multisession_scenario(
    scenario: Dict[str, Any], registry: ToolRegistry,
    *, node, observer, default_system_prompt: str,
    enable_steering: bool, steering_alpha: float,
    enable_channel_g: bool,
    store: Optional[RelationalStateStore],
) -> List[TurnResult]:
    sid = scenario["id"]
    log.info("=" * 74)
    log.info("scenario: %s — %s", sid, scenario.get("description", ""))
    log.info("class: %s (multi-session)", scenario.get("class", "?"))

    agent_id = scenario.get("agent_id", "fg_eval_agent")
    user_id = scenario.get("user_id", f"u_{sid}")

    # Reset state for this arc to make the eval deterministic across runs.
    if store is not None:
        store.delete(agent_id, user_id)

    all_results: List[TurnResult] = []

    for session in scenario["sessions"]:
        session_idx = int(session.get("session_idx", 1))
        log.info("─" * 70)
        log.info("session %d", session_idx)

        channel_b_text = ""
        if enable_channel_g and store is not None:
            rs = store.load_or_init(agent_id, user_id)
            channel_b_text = render_episodic_block(rs)
            if channel_b_text:
                log.info("Channel B (relational) injected:\n%s",
                         "\n  ".join(("",) + tuple(channel_b_text.splitlines())))
            else:
                log.info("Channel B empty (first session for this user)")
            node.set_system_augmentation(channel_b_text)
        else:
            # Channel G off but multi-session arc — still treat each
            # session as fresh chat history with no augmentation.
            node.set_system_augmentation("")

        results, transcript, channels_seen = await _run_session(
            sid, session_idx, session,
            node=node, observer=observer, registry=registry,
            default_system_prompt=default_system_prompt,
            enable_steering=enable_steering,
            steering_alpha=steering_alpha,
            channel_b_text=channel_b_text,
        )
        all_results.extend(results)

        # End-of-session: summarize + EMA + persist.
        if enable_channel_g and store is not None:
            avg_w, avg_a, pos, bet = _aggregate_channels(channels_seen)
            log.info(
                "session %d aggregates: warmth=%.2f anger=%.2f "
                "positive=%d betrayals=%d",
                session_idx, avg_w, avg_a, pos, bet,
            )

            summary = ""
            try:
                summary = summarize_session(
                    node._model, node._tokenizer, transcript,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("summarization failed: %s", e)
            if summary:
                log.info("session %d summary: %s", session_idx, summary)

            shared_updates = session.get("shared_context_to_record") or None

            rs = store.load_or_init(agent_id, user_id)
            update_at_session_end(
                rs,
                session_avg_warmth=avg_w,
                session_avg_anger=avg_a,
                positive_episodes=pos,
                betrayals=bet,
                episode_summary=summary or None,
                shared_context_updates=shared_updates,
            )
            store.save(rs)

            # If the next session has days_after_prior, rewind last_seen
            # so render_episodic_block reports the right "yesterday"
            # framing. This is the eval's time-shim — production code
            # uses real wall time.
            next_session = None
            for s in scenario["sessions"]:
                if int(s.get("session_idx", 1)) == session_idx + 1:
                    next_session = s
                    break
            if next_session is not None:
                days_after = float(next_session.get("days_after_prior", 0))
                if days_after > 0:
                    rs.last_seen = max(
                        0, int(rs.last_seen - days_after * 86400),
                    )
                    store.save(rs, now=rs.updated_at)
                    log.info(
                        "shifted last_seen back by %.1f day(s) for next session",
                        days_after,
                    )

    return all_results


async def _run_singlesession_scenario(
    scenario: Dict[str, Any], registry: ToolRegistry,
    *, node, observer, default_system_prompt: str,
    enable_steering: bool, steering_alpha: float,
) -> List[TurnResult]:
    sid = scenario["id"]
    log.info("=" * 74)
    log.info("scenario: %s — %s", sid, scenario.get("description", ""))

    # Existing single-session shape — wrap as one synthetic session.
    synthetic_session = {
        "session_idx": 1,
        "system_prompt_override": scenario.get("system_prompt_override"),
        "turns": scenario["turns"],
    }
    # No Channel B for Class I/II (those scenarios assume fresh context).
    node.set_system_augmentation("")
    results, _, _ = await _run_session(
        sid, 1, synthetic_session,
        node=node, observer=observer, registry=registry,
        default_system_prompt=default_system_prompt,
        enable_steering=enable_steering, steering_alpha=steering_alpha,
        channel_b_text="",
    )
    return results


async def run_eval(
    scenarios: List[Dict[str, Any]], registry: ToolRegistry,
    repo: str, default_system_prompt: str,
    *,
    enable_steering: bool = False,
    directions_path: Optional[Path] = None,
    steering_alpha: float = DEFAULT_STEERING_ALPHA,
    enable_observer: bool = False,
    observer_repo: str = DEFAULT_OBSERVER_REPO,
    enable_channel_g: bool = False,
    state_db: Path = DEFAULT_STATE_DB,
) -> List[TurnResult]:
    from remotemedia.nodes.ml.mlx_lm_text import MlxLmTextNode

    log.info("loading main LLM %s …", repo)
    node_kwargs: Dict[str, Any] = dict(
        node_id="fg_eval",
        hf_repo=repo,
        system_prompt=default_system_prompt,
        max_new_tokens=160,
    )
    if enable_steering:
        if directions_path is None or not directions_path.exists():
            raise RuntimeError(
                f"Channel D enabled but directions NPZ missing: {directions_path}"
            )
        node_kwargs.update(
            steering_enabled=True,
            steering_directions_path=str(directions_path),
            steering_alpha=steering_alpha,
        )
    else:
        node_kwargs.update(steering_enabled=False)
    node = MlxLmTextNode(**node_kwargs)
    await node.initialize()

    observer = None
    if enable_observer:
        from remotemedia.nodes.ml.tool_observer import ToolObserverNode
        log.info("loading observer %s …", observer_repo)
        observer = ToolObserverNode(node_id="fg_observer", hf_repo=observer_repo)
        await observer.initialize()

    store: Optional[RelationalStateStore] = None
    if enable_channel_g:
        state_db.parent.mkdir(parents=True, exist_ok=True)
        store = RelationalStateStore(state_db)
        log.info("Channel G enabled; relational state DB=%s", state_db)

    results: List[TurnResult] = []
    try:
        for scenario in scenarios:
            if "sessions" in scenario:
                if not enable_channel_g:
                    log.warning(
                        "scenario %s is multi-session but --enable-channel-g "
                        "was not passed; sessions will run with empty "
                        "Channel B — Class III pass is unlikely",
                        scenario.get("id", "?"),
                    )
                results.extend(await _run_multisession_scenario(
                    scenario, registry,
                    node=node, observer=observer,
                    default_system_prompt=default_system_prompt,
                    enable_steering=enable_steering,
                    steering_alpha=steering_alpha,
                    enable_channel_g=enable_channel_g,
                    store=store,
                ))
            else:
                results.extend(await _run_singlesession_scenario(
                    scenario, registry,
                    node=node, observer=observer,
                    default_system_prompt=default_system_prompt,
                    enable_steering=enable_steering,
                    steering_alpha=steering_alpha,
                ))
    finally:
        await node.cleanup()
        if observer is not None:
            await observer.cleanup()
        if store is not None:
            store.close()
    return results


def _summary(results: List[TurnResult]) -> Dict[str, Any]:
    by_scenario: Dict[str, Dict[str, int]] = {}
    hard_pass = 0
    hard_total = 0
    soft_tool_met = 0
    soft_tool_total = 0
    soft_cont_met = 0
    soft_cont_total = 0
    for r in results:
        s = by_scenario.setdefault(
            r.scenario_id,
            {"pass": 0, "fail": 0,
             "soft_tool_met": 0, "soft_tool_total": 0,
             "soft_cont_met": 0, "soft_cont_total": 0},
        )
        if r.passed:
            s["pass"] += 1; hard_pass += 1
        else:
            s["fail"] += 1
        hard_total += 1
        if r.soft_tool_met is not None:
            soft_tool_total += 1; s["soft_tool_total"] += 1
            if r.soft_tool_met:
                soft_tool_met += 1; s["soft_tool_met"] += 1
        if r.soft_continuity_met is not None:
            soft_cont_total += 1; s["soft_cont_total"] += 1
            if r.soft_continuity_met:
                soft_cont_met += 1; s["soft_cont_met"] += 1
    return {
        "hard_pass": hard_pass,
        "hard_total": hard_total,
        "hard_pass_rate": hard_pass / hard_total if hard_total else 0.0,
        "soft_tool_met": soft_tool_met,
        "soft_tool_total": soft_tool_total,
        "soft_tool_rate": (
            soft_tool_met / soft_tool_total if soft_tool_total else None
        ),
        "soft_continuity_met": soft_cont_met,
        "soft_continuity_total": soft_cont_total,
        "soft_continuity_rate": (
            soft_cont_met / soft_cont_total if soft_cont_total else None
        ),
        "by_scenario": by_scenario,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    ap.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--repo", default=DEFAULT_HF_REPO)
    ap.add_argument("--enable-steering", action="store_true",
                    help="Enable Channel D — drives target_vad from policy each turn")
    ap.add_argument("--directions", type=Path, default=DEFAULT_DIRECTIONS,
                    help="Steering NPZ path (used when --enable-steering)")
    ap.add_argument("--alpha", type=float, default=DEFAULT_STEERING_ALPHA,
                    help="Channel D scalar")
    ap.add_argument("--enable-observer", action="store_true",
                    help="Enable tool-call observer (auxiliary small LLM)")
    ap.add_argument("--observer-repo", default=DEFAULT_OBSERVER_REPO,
                    help="HF repo for the observer LLM")
    ap.add_argument("--enable-channel-g", action="store_true",
                    help="Enable Channel G — cross-session relational state")
    ap.add_argument("--state-db", type=Path, default=DEFAULT_STATE_DB,
                    help="SQLite path for relational state (eval-scoped)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Run only the named scenario id(s)")
    args = ap.parse_args()

    if not args.scenarios.exists():
        log.error("scenarios file missing: %s", args.scenarios); return 2
    if not args.tools.exists():
        log.error("tools file missing: %s", args.tools); return 2

    registry = ToolRegistry.from_yaml(args.tools)
    log.info("loaded %d tool(s): %s", len(registry.tools),
             [t.name for t in registry.tools])

    payload = json.loads(args.scenarios.read_text())
    scenarios = payload["scenarios"]
    if args.only:
        only = set(args.only)
        scenarios = [s for s in scenarios if s.get("id") in only]
        if not scenarios:
            log.error("no scenarios match --only %s", args.only); return 2
    log.info("loaded %d scenario(s)", len(scenarios))

    results = asyncio.run(run_eval(
        scenarios, registry, args.repo, DEFAULT_SYSTEM_PROMPT,
        enable_steering=args.enable_steering,
        directions_path=args.directions,
        steering_alpha=args.alpha,
        enable_observer=args.enable_observer,
        observer_repo=args.observer_repo,
        enable_channel_g=args.enable_channel_g,
        state_db=args.state_db,
    ))
    summary = _summary(results)

    log.info("=" * 74)
    log.info("RESULTS")
    log.info(
        "  hard pass: %d/%d  (%.0f%%)",
        summary["hard_pass"], summary["hard_total"],
        100 * summary["hard_pass_rate"],
    )
    if summary["soft_tool_total"]:
        log.info(
            "  soft tool expectation met: %d/%d  (%.0f%%)",
            summary["soft_tool_met"], summary["soft_tool_total"],
            100 * (summary["soft_tool_rate"] or 0.0),
        )
    if summary["soft_continuity_total"]:
        log.info(
            "  soft continuity expectation met: %d/%d  (%.0f%%)",
            summary["soft_continuity_met"], summary["soft_continuity_total"],
            100 * (summary["soft_continuity_rate"] or 0.0),
        )
    log.info("  per scenario:")
    for sid, s in summary["by_scenario"].items():
        soft_bits = []
        if s["soft_tool_total"]:
            soft_bits.append(f"tool={s['soft_tool_met']}/{s['soft_tool_total']}")
        if s["soft_cont_total"]:
            soft_bits.append(f"cont={s['soft_cont_met']}/{s['soft_cont_total']}")
        soft_str = ("  " + " ".join(soft_bits)) if soft_bits else ""
        log.info("    %-40s pass=%d  fail=%d%s",
                 sid, s["pass"], s["fail"], soft_str)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "summary": summary,
        "turns": [dataclasses.asdict(r) for r in results],
        "config": {
            "repo": args.repo,
            "enable_steering": args.enable_steering,
            "enable_observer": args.enable_observer,
            "enable_channel_g": args.enable_channel_g,
            "state_db": str(args.state_db),
            "ts": int(time.time()),
        },
    }, indent=2))
    log.info("wrote report: %s", args.report)

    return 0 if summary["hard_pass"] == summary["hard_total"] else 1


if __name__ == "__main__":
    sys.exit(main())
