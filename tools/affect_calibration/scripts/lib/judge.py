"""Judge-LLM providers for `04_validate_pipeline.py`.

Two implementations of the `JudgeLLM` Protocol from `lib.persona_pipeline`:

- `AnthropicJudge` — production path. Cross-model judging (judge != target,
  per Spike F + paper App. B). Uses the `anthropic` Python SDK; expects
  `ANTHROPIC_API_KEY` in the environment. Default model: a recent Sonnet.

- `MockJudge` — deterministic-heuristic scorer for development. Scores by
  string heuristics over the response text; not a real evaluator, but
  produces stable, repeatable numbers so the validation pipeline's wiring
  can be exercised without burning API budget. Stable across runs given
  the same `(axis, pole, response)` tuple.

The factory `build_judge(backend, model_id)` selects the right one.

Per Spike F: the judge prompt is the 0-100 net rubric — score the
positive_pole and negative_pole independently, net = pos - neg. The score
function below returns a single 0-100 value for one (axis, pole, response);
`persona_pipeline.judge_steered_responses` aggregates net scores over a
batch.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Optional

from .persona_pipeline import Axis, Pole, TraitDescription


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock judge
# ---------------------------------------------------------------------------

class MockJudge:
    """Deterministic synthetic judge for non-API smoke testing.

    Scoring strategy:
        - Hash (axis, pole, response) → integer in [0, 100].
        - Add a trait-aware bias: count occurrences of pole-cue substrings
          in the response (e.g. "cheerful", "calm") and shift score up by
          a few points. Crude but enough that *behavioural* responses
          score above generic ones, so a steered response that adopts the
          target tone scores higher than its unsteered counterpart in
          most cases.

    NOT a substitute for the real Anthropic judge. Use only for wiring tests.
    """

    # Cheap pole-aware lexicon. Order matters only for the count.
    _POSITIVE_CUES: dict[str, tuple[str, ...]] = {
        Axis.VALENCE.value: ("happy", "joy", "love", "wonderful", "great", "delight", "excited"),
        Axis.AROUSAL.value: ("excited", "energetic", "intense", "thrilled", "vibrant", "active", "fast"),
        Axis.DOMINANCE.value: ("definitely", "certainly", "must", "should", "command", "decisive", "sure"),
    }
    _NEGATIVE_CUES: dict[str, tuple[str, ...]] = {
        Axis.VALENCE.value: ("sad", "unhappy", "miserable", "regret", "disappointing", "gloomy"),
        Axis.AROUSAL.value: ("calm", "quiet", "still", "slow", "relaxed", "tranquil", "peaceful"),
        Axis.DOMINANCE.value: ("perhaps", "maybe", "tentatively", "unsure", "defer", "could"),
    }

    def __init__(self) -> None:
        pass

    def score(
        self,
        axis: Axis,
        pole: Pole,
        response: str,
        trait_description: TraitDescription,  # noqa: ARG002 — kept for Protocol parity
    ) -> float:
        # Stable hash (md5; not cryptographic — we just want determinism
        # across Python runs, which built-in hash() does NOT provide).
        digest = hashlib.md5(
            f"{axis.value}|{pole.value}|{response}".encode("utf-8")
        ).digest()
        base = int.from_bytes(digest[:2], "big") % 100  # 0..99 baseline

        # Pole-aware lexical bump.
        cues = (
            self._POSITIVE_CUES[axis.value]
            if pole == Pole.POSITIVE
            else self._NEGATIVE_CUES[axis.value]
        )
        lower = response.lower()
        hits = sum(lower.count(cue) for cue in cues)
        bumped = min(100, max(0, base + 8 * hits))

        return float(bumped)


# ---------------------------------------------------------------------------
# Anthropic judge
# ---------------------------------------------------------------------------

class AnthropicJudge:
    """Cross-model judge using the Anthropic API.

    Lazy-imports the `anthropic` SDK so `--judge-backend mock` paths don't
    need it installed. Reads `ANTHROPIC_API_KEY` from the environment.
    """

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

    def __init__(self, model_id: Optional[str] = None) -> None:
        try:
            import anthropic  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "AnthropicJudge requires the `anthropic` package. "
                "Install with `pip install anthropic`."
            ) from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "AnthropicJudge requires ANTHROPIC_API_KEY in the environment."
            )

        from anthropic import Anthropic  # type: ignore

        self.model_id = model_id or self.DEFAULT_MODEL
        self._client = Anthropic()
        self._cache: dict[tuple[str, str, str], float] = {}

    def score(
        self,
        axis: Axis,
        pole: Pole,
        response: str,
        trait_description: TraitDescription,
    ) -> float:
        cache_key = (axis.value, pole.value, response)
        if cache_key in self._cache:
            return self._cache[cache_key]

        pole_text = (
            trait_description.positive_pole
            if pole == Pole.POSITIVE
            else trait_description.negative_pole
        )
        rubric_prompt = self._build_rubric(axis, pole, pole_text, response)

        try:
            msg = self._client.messages.create(
                model=self.model_id,
                max_tokens=16,
                temperature=0.0,
                system=(
                    "You are an evaluator that scores how strongly a response "
                    "expresses a target trait. Output only an integer between "
                    "0 and 100, on its own line. No commentary, no formatting."
                ),
                messages=[{"role": "user", "content": rubric_prompt}],
            )
        except Exception as exc:  # noqa: BLE001 — re-raise with context
            raise RuntimeError(
                f"AnthropicJudge call failed for axis={axis.value} pole={pole.value}: {exc}"
            ) from exc

        text = "".join(
            block.text for block in msg.content if getattr(block, "type", None) == "text"
        ).strip()
        score = self._parse_int_score(text)
        self._cache[cache_key] = score
        return score

    # ------------------------------------------------------------------

    @staticmethod
    def _build_rubric(axis: Axis, pole: Pole, pole_text: str, response: str) -> str:
        return (
            f"Evaluate the response below on a 0–100 scale.\n\n"
            f"Trait axis: {axis.value}\n"
            f"Pole: {pole.value}\n"
            f"Pole description:\n{pole_text}\n\n"
            f"Response to evaluate:\n\"\"\"\n{response}\n\"\"\"\n\n"
            f"Rate how strongly this response expresses the {pole.value} pole "
            f"of {axis.value}. 0 = not at all, 100 = strongly. Output only the integer."
        )

    @staticmethod
    def _parse_int_score(text: str) -> float:
        match = re.search(r"-?\d+", text)
        if not match:
            logger.warning("AnthropicJudge: no integer in response %r; returning 50", text)
            return 50.0
        score = int(match.group(0))
        return float(max(0, min(100, score)))


# ---------------------------------------------------------------------------
# OpenAI-compatible judge (local llama-cpp / vLLM / llama-swap / etc.)
# ---------------------------------------------------------------------------

class OpenAICompatibleJudge:
    """Cross-model judge against any OpenAI-compatible /v1 endpoint.

    Per Spike F: judge MUST be a different model than the target LLM
    (avoids self-grading bias). The local server picked here should NOT
    be the same model family/checkpoint as the target — e.g. judging
    LFM2-Audio-1.5B with a local Qwen3.6-27B is fine; judging
    Qwen3.6-27B with Qwen3.6-27B is NOT.

    Reasoning-model handling: max_tokens=512 is enough for chain-of-thought
    plus a single integer answer. We then parse the integer with the same
    regex as the Anthropic path. Caches (axis, pole, response) → score
    so the same response isn't re-judged across reruns.
    """

    def __init__(
        self,
        model_id: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        if not model_id:
            raise ValueError("OpenAICompatibleJudge requires --judge-model")
        from .openai_compat import OpenAICompatibleClient
        self.model_id = model_id
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self._client = OpenAICompatibleClient()
        self._cache: dict[tuple[str, str, str], float] = {}

    def score(
        self,
        axis: Axis,
        pole: Pole,
        response: str,
        trait_description: TraitDescription,
    ) -> float:
        cache_key = (axis.value, pole.value, response)
        if cache_key in self._cache:
            return self._cache[cache_key]

        pole_text = (
            trait_description.positive_pole
            if pole == Pole.POSITIVE
            else trait_description.negative_pole
        )
        rubric_prompt = AnthropicJudge._build_rubric(axis, pole, pole_text, response)

        rsp = self._client.chat(
            model=self.model_id,
            system=(
                "You are an evaluator that scores how strongly a response "
                "expresses a target trait. Output only an integer between "
                "0 and 100, on its own line. No commentary, no formatting."
            ),
            user=rubric_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        score = AnthropicJudge._parse_int_score(rsp.content)
        self._cache[cache_key] = score
        return score


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

JUDGE_BACKENDS = ("anthropic", "openai", "mock")


def build_judge(backend: str, model_id: Optional[str] = None):
    """Build a Judge by backend name. Used by `04_validate_pipeline.py`."""
    if backend == "mock":
        return MockJudge()
    if backend == "anthropic":
        return AnthropicJudge(model_id=model_id)
    if backend == "openai":
        return OpenAICompatibleJudge(model_id=model_id or "")
    raise ValueError(f"unknown judge backend: {backend!r}; expected one of {JUDGE_BACKENDS}")
