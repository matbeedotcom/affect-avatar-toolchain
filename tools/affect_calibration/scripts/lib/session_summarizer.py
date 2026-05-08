"""Channel G — end-of-session summarization.

A single ``mlx_lm.generate`` call against the same loaded model the main
LLM uses, with a short summarization prompt. The resulting one-line
summary gets appended to ``RelationalState.recent_episode_summaries``
and rendered into Channel B at the next session start.

Reusing the main model is intentional: it keeps the summary in the same
register as the persona ("you", "we", informal). A separate small model
would be cheaper but produces stylistic drift that the chat LLM then has
to override at session start. Latency is acceptable because summarization
runs after the user has left the session.
"""
from __future__ import annotations

import datetime as _dt
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DATE_PREFIX_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s*:\s*", re.MULTILINE)


def _format_transcript(transcript: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for m in transcript:
        if m.get("role") == "system":
            continue
        role = str(m.get("role", "")).upper() or "?"
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def summarize_session(
    model: Any,
    tokenizer: Any,
    transcript: List[Dict[str, str]],
    *,
    today: Optional[str] = None,
    max_tokens: int = 100,
) -> str:
    """Generate a 30-word session summary. Empty string on failure or
    on an empty transcript (no user/assistant turns).
    """
    body = _format_transcript(transcript)
    if not body:
        return ""

    today = today or _dt.date.today().isoformat()
    system = (
        "You write one-line conversation summaries for an agent's own "
        "memory. Capture: (1) the emotional arc, (2) any topics that "
        "might come up again. Maximum 30 words. Output one line in this "
        f"exact format and nothing else:\n{today}: <summary>"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Conversation:\n{body}"},
    ]
    try:
        from mlx_lm import generate as _gen
        from mlx_lm.sample_utils import make_sampler
    except ImportError as e:
        logger.warning("mlx_lm unavailable for summarization: %s", e)
        return ""

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("apply_chat_template failed for summarization: %s", e)
        return ""

    sampler = make_sampler(temp=0.0)
    try:
        out = _gen(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens,
            sampler=sampler, verbose=False,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("generation failed for summarization: %s", e)
        return ""

    text = (out or "").strip()
    if not text:
        return ""

    # Take only the first non-empty line. Models occasionally bolt on a
    # second-line gloss despite the "one line" instruction.
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Models sometimes prepend their own date — keep ours.
        line = _DATE_PREFIX_RE.sub("", line, count=1).strip()
        if not line:
            continue
        return f"{today}: {line}"
    return ""
