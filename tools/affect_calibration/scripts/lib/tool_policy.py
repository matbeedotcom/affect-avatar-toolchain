"""Tool-policy registry + predicate evaluator for Channel F.

Per spike-channel-f-and-g-mood-gated-behavior.md. A tool's
``affect_policy`` declares preconditions (``requires``) and exclusions
(``forbids``) over the simulator's affect state. The evaluator returns
the subset of tools whose preconditions are satisfied for a given
``(EmotionChannels, RegulationPolicy)`` pair.

This is the load-bearing primitive for Channel F: simulator state →
permitted tool list → injected into the LLM's chat-template tools arg.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Affect-state field names the predicate evaluator can target.
# These must match the JSON keys produced by the Rust simulator's
# `EmotionChannels` and `RegulationPolicy`.
CHANNEL_FIELDS = {
    "anger", "sadness", "fear", "joy", "calm",
    "frustration", "curiosity", "empathy",
}
POLICY_FIELDS = {
    "expressiveness", "safety_dampening", "social_dampening",
    "task_focus", "warmth", "assertiveness",
}
CORE_FIELDS = {"valence", "arousal", "dominance"}
KNOWN_FIELDS = CHANNEL_FIELDS | POLICY_FIELDS | CORE_FIELDS


# Predicate parser — accepts simple comparison strings used in YAML:
#   ">0.7", ">=0.5", "<0.3", "<=0.2", "==0", "!=0"
_PRED_RE = re.compile(r"^\s*(>=|<=|==|!=|>|<)\s*(-?\d*\.?\d+)\s*$")


def _parse_predicate(expr: str):
    """Return a (op, threshold) tuple, or raise ValueError."""
    m = _PRED_RE.match(expr)
    if not m:
        raise ValueError(
            f"unparseable predicate {expr!r}; expected forms like "
            f"'>0.5', '<=0.2', '==0'"
        )
    op, num = m.group(1), float(m.group(2))
    return op, num


def _eval_predicate(value: float, op: str, threshold: float) -> bool:
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == "==":
        return value == threshold
    if op == "!=":
        return value != threshold
    raise ValueError(f"unknown op {op!r}")


def _resolve_field(field: str, channels: Dict[str, float],
                   policy: Dict[str, float],
                   core: Optional[Dict[str, float]] = None) -> float:
    """Look up the named field in the merged state. Channels win over
    policy on name collisions (none today, but defensive)."""
    if field in CHANNEL_FIELDS:
        return float(channels.get(field, 0.0))
    if field in POLICY_FIELDS:
        return float(policy.get(field, 0.0))
    if field in CORE_FIELDS:
        if core is None:
            raise ValueError(f"core field {field!r} requested but core not provided")
        return float(core.get(field, 0.0))
    raise ValueError(f"unknown affect field {field!r}; known: {sorted(KNOWN_FIELDS)}")


@dataclass
class AffectPolicy:
    """Preconditions / exclusions for one tool."""
    requires: Dict[str, str] = field(default_factory=dict)  # field -> predicate
    forbids: Dict[str, str] = field(default_factory=dict)
    description_for_llm: str = ""

    def evaluate(self, channels: Dict[str, float], policy: Dict[str, float],
                 core: Optional[Dict[str, float]] = None) -> Tuple[bool, str]:
        """Return (permitted, reason). Reason is empty when permitted."""
        for fld, expr in self.requires.items():
            op, thresh = _parse_predicate(expr)
            v = _resolve_field(fld, channels, policy, core)
            if not _eval_predicate(v, op, thresh):
                return False, f"requires {fld}{expr}; have {fld}={v:.2f}"
        for fld, expr in self.forbids.items():
            op, thresh = _parse_predicate(expr)
            v = _resolve_field(fld, channels, policy, core)
            if _eval_predicate(v, op, thresh):
                return False, f"forbids {fld}{expr}; have {fld}={v:.2f}"
        return True, ""


@dataclass
class ToolDef:
    """One entry in the registry. ``schema`` is the OpenAI/Llama 3.1
    function-calling JSON the LLM sees in its tools= field."""
    name: str
    description: str
    schema: Dict[str, Any]
    affect_policy: AffectPolicy


@dataclass
class ToolRegistry:
    tools: List[ToolDef] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "ToolRegistry":
        # Parse a small YAML subset rather than pull a yaml dep — keeps
        # the calibration env light.
        import json
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(path.read_text())
        except ImportError:
            # Fallback: scenarios author can ship JSON-formatted YAML.
            data = json.loads(path.read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolRegistry":
        tools: List[ToolDef] = []
        for name, spec in data.items():
            policy_dict = spec.get("affect_policy", {})
            policy = AffectPolicy(
                requires=policy_dict.get("requires", {}) or {},
                forbids=policy_dict.get("forbids", {}) or {},
                description_for_llm=policy_dict.get("description_for_llm", ""),
            )
            tools.append(ToolDef(
                name=name,
                description=spec.get("description", ""),
                schema=spec.get("schema", {
                    "name": name,
                    "description": spec.get("description", ""),
                    "parameters": {"type": "object", "properties": {}},
                }),
                affect_policy=policy,
            ))
        return cls(tools=tools)

    def permitted_tools(self, channels: Dict[str, float],
                        policy: Dict[str, float],
                        core: Optional[Dict[str, float]] = None
                        ) -> List[ToolDef]:
        """Return the subset of tools whose affect_policy is satisfied."""
        out: List[ToolDef] = []
        for t in self.tools:
            ok, _ = t.affect_policy.evaluate(channels, policy, core)
            if ok:
                out.append(t)
        return out

    def explain(self, channels: Dict[str, float], policy: Dict[str, float],
                core: Optional[Dict[str, float]] = None
                ) -> List[Tuple[str, bool, str]]:
        """For debugging: per-tool (name, permitted, reason)."""
        out = []
        for t in self.tools:
            ok, reason = t.affect_policy.evaluate(channels, policy, core)
            out.append((t.name, ok, reason))
        return out


def compute_channel_d_target(policy: Dict[str, float]) -> List[float]:
    """Python port of ``tools/affect_simulator/src/channel_d.rs::compute_target``.

    Derives the steering target VAD from the regulation policy:
        raw_v = warmth - 0.3
        raw_d = (assertiveness - 0.3) * (1 - 0.5 * safety_dampening)
        raw_a = 0.0       # per the regulation-mediated rewrite
        scale = 3.0
        return clamp(scale * raw, [-1, 1] for v/d, [0, 1] for a)

    Kept in Python here so the F-eval harness can drive Channel D without
    invoking the Rust simulator. Must stay in sync with channel_d.rs.
    """
    warmth = float(policy.get("warmth", 0.5))
    assertiveness = float(policy.get("assertiveness", 0.5))
    safety_dampening = float(policy.get("safety_dampening", 0.0))
    raw_v = warmth - 0.3
    raw_d = (assertiveness - 0.3) * (1.0 - 0.5 * safety_dampening)
    raw_a = 0.0
    scale = 3.0
    v = max(-1.0, min(1.0, scale * raw_v))
    a = max(0.0, min(1.0, scale * raw_a))
    d = max(-1.0, min(1.0, scale * raw_d))
    return [v, a, d]


def render_for_chat_template(tools: List[ToolDef]) -> List[Dict[str, Any]]:
    """Convert ToolDef list to the OpenAI-style schema list that
    ``tokenizer.apply_chat_template(tools=...)`` accepts."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.schema.get("name", t.name),
                "description": t.affect_policy.description_for_llm or t.description,
                "parameters": t.schema.get("parameters", {
                    "type": "object", "properties": {},
                }),
            },
        }
        for t in tools
    ]


# Tool-call parsing — Llama 3.1 / Hermes-3 emit calls in a few formats.
# Hermes-3 specifically uses XML-ish: <tool_call>{json}</tool_call>.
# Llama-3.1-Instruct uses python-block style or `<|python_tag|>` envelopes.
# This handles both.

_HERMES_TAG = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL,
)
_PY_TAG = re.compile(
    r"<\|python_tag\|>(.+?)(?:<\|eom_id\|>|<\|eot_id\|>|$)", re.DOTALL,
)


def parse_tool_calls(response_text: str) -> List[Dict[str, Any]]:
    """Best-effort parse. Returns list of {name, arguments} dicts.

    Empty list when the LLM didn't emit a tool call. The eval harness
    treats "no tool call" as a valid response (LLM chose to reply
    verbally).
    """
    import json as _json
    out: List[Dict[str, Any]] = []
    for m in _HERMES_TAG.finditer(response_text):
        try:
            obj = _json.loads(m.group(1))
            if isinstance(obj, dict) and "name" in obj:
                out.append({
                    "name": obj["name"],
                    "arguments": obj.get("arguments", {}),
                })
        except _json.JSONDecodeError:
            continue
    if not out:
        for m in _PY_TAG.finditer(response_text):
            blob = m.group(1).strip()
            try:
                obj = _json.loads(blob)
                if isinstance(obj, dict) and "name" in obj:
                    out.append({
                        "name": obj["name"],
                        "arguments": obj.get("arguments", {}),
                    })
            except _json.JSONDecodeError:
                continue
    return out


def strip_tool_calls(response_text: str) -> str:
    """Remove tool-call envelopes so the residual is the verbal reply."""
    s = _HERMES_TAG.sub("", response_text)
    s = _PY_TAG.sub("", s)
    return s.strip()
