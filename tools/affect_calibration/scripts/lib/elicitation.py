"""Elicitation-LLM providers for `02b_generate_contrast_pairs.py`.

Two implementations of the `ElicitationLLM` Protocol from
`lib.persona_pipeline`:

- `AnthropicElicitor` — production path. Uses the `anthropic` SDK; expects
  `ANTHROPIC_API_KEY` in the environment. Default model: a recent Sonnet.
  One-shot per call: takes (system, user) → string. The
  `elicit_contrast_prompts_for_axis` consumer makes 3 calls per axis
  (positive prompts, negative prompts, questions) — 9 calls total per
  trait_descriptions.json.

- `MockElicitor` — deterministic synthetic prompts for non-API smoke
  testing. Generates plausible-shape lines that the parser can split.
  Not useful for real direction extraction; only for wiring tests.

Factory `build_elicitor(backend, model_id)` selects the right one.
"""

from __future__ import annotations

import logging
import os
from typing import Optional


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock elicitor
# ---------------------------------------------------------------------------

class MockElicitor:
    """Deterministic synthetic elicitor for non-API wiring tests.

    Returns canned output keyed on a heuristic — if the user prompt asks
    for "system prompts" we emit numbered persona descriptions; if it
    asks for "questions" we emit a numbered question list. The line
    parser in `elicit_contrast_prompts_for_axis` strips numbering
    automatically.
    """

    def __init__(self) -> None:
        pass

    def complete(self, system: str, user: str) -> str:
        ul = user.lower()
        if "system prompt" in ul or "personas" in ul or "persona" in ul:
            n = self._extract_count(user, default=5)
            pole = "positive" if "positive" in ul or "high pole" in ul else "negative"
            return "\n".join(
                f"{i + 1}. You are a {pole}-pole assistant variant #{i + 1}; respond accordingly."
                for i in range(n)
            )
        if "question" in ul or "questions" in ul:
            n = self._extract_count(user, default=40)
            return "\n".join(
                f"{i + 1}. Mock evaluation question #{i + 1}."
                for i in range(n)
            )
        # Fallback: echo a single line so the parser doesn't choke.
        return "1. Mock elicitation response."

    @staticmethod
    def _extract_count(user_prompt: str, *, default: int) -> int:
        """Pull an integer like 'generate 5' out of the user message; fallback to default."""
        import re
        m = re.search(r"\b(\d+)\b", user_prompt)
        if m:
            return int(m.group(1))
        return default


# ---------------------------------------------------------------------------
# Anthropic elicitor
# ---------------------------------------------------------------------------

class AnthropicElicitor:
    """Production elicitation provider using the Anthropic API.

    Lazy-imports `anthropic` so `--elicitation-backend mock` paths avoid
    the SDK + key requirement. Reads `ANTHROPIC_API_KEY` from the
    environment.
    """

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

    def __init__(self, model_id: Optional[str] = None) -> None:
        try:
            import anthropic  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "AnthropicElicitor requires the `anthropic` package. "
                "Install with `pip install anthropic`."
            ) from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "AnthropicElicitor requires ANTHROPIC_API_KEY in the environment."
            )

        from anthropic import Anthropic  # type: ignore

        self.model_id = model_id or self.DEFAULT_MODEL
        self._client = Anthropic()

    def complete(self, system: str, user: str) -> str:
        try:
            msg = self._client.messages.create(
                model=self.model_id,
                max_tokens=2048,
                temperature=0.7,  # diversity matters for elicitation
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"AnthropicElicitor call failed (model={self.model_id}): {exc}"
            ) from exc
        return "".join(
            block.text for block in msg.content if getattr(block, "type", None) == "text"
        ).strip()


# ---------------------------------------------------------------------------
# OpenAI-compatible elicitor (local llama-cpp / vLLM / llama-swap / etc.)
# ---------------------------------------------------------------------------

class OpenAICompatibleElicitor:
    """Production elicitor against any OpenAI-compatible /v1 endpoint.

    Reads OPENAI_BASE_URL + OPENAI_API_KEY (sk-local when unset) — see
    `lib.openai_compat`. Lets Phase 5 calibration run against a local
    Qwen / Llama / Mistral server without spending Anthropic budget.

    Reasoning-model models (Qwen3, DeepSeek-R1) need generous max_tokens
    so chain-of-thought + the actual numbered list can both fit; the
    default of 1500 is sized for the "40 questions per axis" call which
    is the largest of the 9 calls per run.
    """

    def __init__(
        self,
        model_id: str,
        *,
        max_tokens: int = 1500,
        temperature: float = 0.7,
    ) -> None:
        if not model_id:
            raise ValueError("OpenAICompatibleElicitor requires --elicitation-model")
        from .openai_compat import OpenAICompatibleClient
        self.model_id = model_id
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self._client = OpenAICompatibleClient()

    def complete(self, system: str, user: str) -> str:
        rsp = self._client.chat(
            model=self.model_id,
            system=system,
            user=user,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return rsp.content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

ELICITOR_BACKENDS = ("anthropic", "openai", "mock")


def build_elicitor(backend: str, model_id: Optional[str] = None):
    """Build an elicitor by backend name."""
    if backend == "mock":
        return MockElicitor()
    if backend == "anthropic":
        return AnthropicElicitor(model_id=model_id)
    if backend == "openai":
        return OpenAICompatibleElicitor(model_id=model_id or "")
    raise ValueError(
        f"unknown elicitor backend: {backend!r}; expected one of {ELICITOR_BACKENDS}"
    )
