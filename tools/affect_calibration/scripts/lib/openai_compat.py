"""Minimal OpenAI-compatible chat client (no SDK dependency).

Used by the OpenAI-compatible variants of `lib.elicitation` and
`lib.judge` to talk to a local llama-cpp / llama-swap / vLLM / etc.
endpoint that exposes `/v1/chat/completions`. The official `openai`
Python SDK works too, but we'd rather not bring it in just for one
HTTP call shape — `urllib.request` from the stdlib is enough.

Reads two env vars (defaults match the openai SDK):
    OPENAI_BASE_URL  — e.g. "http://192.168.18.11:8888/v1"
                       or "https://api.openai.com/v1" (default).
    OPENAI_API_KEY   — bearer token. Many local servers ignore it but
                       require *some* non-empty value; we send "sk-local"
                       when unset to keep them happy.

Reasoning-model handling: Qwen3, DeepSeek-R1, and similar models
return chain-of-thought in a non-standard `reasoning_content` field
with `content=""` when `max_tokens` is too small. The caller is
responsible for setting `max_tokens` large enough for both reasoning
and answer to fit (1024+ is a reasonable default for Qwen3.6-27B).
This client returns the `content` field as-is.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


DEFAULT_BASE_URL = "https://api.openai.com/v1"
LOCAL_KEY_FALLBACK = "sk-local"

logger = logging.getLogger(__name__)


@dataclass
class ChatCompletion:
    """Minimal subset of the OpenAI chat-completion response shape."""
    content: str
    finish_reason: str
    model: str
    usage: dict


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 240.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", LOCAL_KEY_FALLBACK) or LOCAL_KEY_FALLBACK
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatCompletion:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        # Retry transient network failures with exponential backoff.
        # A single 60-second TCP timeout in the middle of a 900-call judge
        # batch was killing 1.5-hour validation runs; retrying recovers
        # cleanly when llama-swap is mid-model-swap or the network blips.
        last_exc: BaseException | None = None
        body: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                # 5xx is transient; 4xx is the caller's fault — bail.
                if 500 <= exc.code < 600 and attempt < self.max_retries:
                    last_exc = exc
                    sleep_s = self.retry_backoff ** attempt
                    logger.warning(
                        "OpenAI-compatible chat call HTTP %d on attempt %d/%d; "
                        "retrying in %.1fs",
                        exc.code, attempt + 1, self.max_retries + 1, sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                raise RuntimeError(
                    f"OpenAI-compatible chat call failed: HTTP {exc.code} {exc.reason} — {detail[:500]}"
                ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < self.max_retries:
                    last_exc = exc
                    sleep_s = self.retry_backoff ** attempt
                    logger.warning(
                        "OpenAI-compatible chat call %s on attempt %d/%d; "
                        "retrying in %.1fs",
                        type(exc).__name__, attempt + 1, self.max_retries + 1, sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(
                    f"OpenAI-compatible chat call failed after {self.max_retries + 1} attempts: "
                    f"{exc} (url={url})"
                ) from exc

        if body is None:
            # Defensive — should never reach here, but the type checker
            # doesn't know the loop always exits one of the three ways.
            raise RuntimeError(
                f"OpenAI-compatible chat call failed without raising; last_exc={last_exc}"
            )

        data = json.loads(body)
        if "error" in data:
            raise RuntimeError(f"OpenAI-compatible chat call returned error: {data['error']}")
        if not data.get("choices"):
            raise RuntimeError(f"OpenAI-compatible chat call returned no choices: {body[:500]}")

        choice = data["choices"][0]
        msg = choice.get("message", {})
        content = (msg.get("content") or "").strip()
        if not content and choice.get("finish_reason") == "length":
            # Reasoning-model pathology: chain-of-thought consumed the whole
            # budget and the actual answer never started. Surface this with
            # context so the caller knows to bump max_tokens.
            reasoning_len = len(msg.get("reasoning_content") or "")
            raise RuntimeError(
                f"chat call returned empty content with finish_reason=length "
                f"(reasoning_content={reasoning_len} chars). Increase max_tokens; "
                f"reasoning models like Qwen3 / DeepSeek-R1 need budget for both "
                f"reasoning and answer. model={data.get('model', model)}"
            )
        return ChatCompletion(
            content=content,
            finish_reason=choice.get("finish_reason", ""),
            model=str(data.get("model", model)),
            usage=data.get("usage", {}),
        )
