#!/usr/bin/env python3
"""Spike G — feasibility script for the MLX-target-LLM pivot.

Pairs with spike-g-mlx-target-llm.md. Runs three gates against the loaded
LFM2.5-Audio-1.5B-4bit model and prints a pass/fail line per gate plus the
data the memo needs (layer count, d_model, divergence rates, text-token
counts). The memo's Status moves from "Drafted" to "Accepted"/"Rejected"
based on this script's output.

Gates (all must pass to accept the pivot):

  G-F1  Capture residual-stream hidden state at a chosen trunk layer L
        without breaking generation. Verifies shape (B, T, d_model).

  G-F2  Inject alpha * v_random at layer L during generation. Verifies
        that text output diverges between alpha=+1 and alpha=-1 on >= 80%
        of a small prompt set. Random v means we don't expect *meaningful*
        divergence — we just need divergence at all (proves the hook is
        live).

  G-F3  Verify text-only contrast prompts elicit >= 20 text tokens of
        coherent response per prompt. Mirrors how 02b's pairs will be
        consumed by 03 and 04 (Option α from the memo: text-in, text-out).

Run:

    /Users/mathieugosbee/.config/remotemedia/envs/e56108523461b6be/bin/python3 \\
      docs/references/activation-steering-audio-llm/notes/spike-g-mlx-extraction.py

The hard-coded interpreter path points at the per-node sandbox env where
mlx-audio is already installed (created by `LFM2AudioMlxNode` initialization).
Pass `--python /path/to/other/python` if you've installed mlx-audio elsewhere.

Throwaway: this script is reference material, not production code. Do not
import from it.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.sts.models.lfm_audio import (
    ChatState,
    LFMModality,
    LFM2AudioModel,
    LFM2AudioProcessor,
)


class _WrappedLayer(nn.Module):
    """Substitutes for one Lfm2DecoderLayer to expose a post-forward hook.

    Monkey-patching `instance.__call__` does not work on MLX nn.Module
    because `obj(...)` dispatches via `type(obj).__call__`. Replacing the
    layer object in `model.lfm.layers` with this wrapper does work — the
    parent loop calls `layer(h, mask, c)` which goes through this class's
    `__call__`.
    """

    def __init__(self, inner: nn.Module, on_post):
        super().__init__()
        self.inner = inner
        self._on_post = on_post  # (h_post: mx.array) -> mx.array | None

    @property
    def is_attention_layer(self) -> bool:  # required by Lfm2Model.make_cache()
        return self.inner.is_attention_layer

    def __call__(self, x, mask=None, cache=None):
        out = self.inner(x, mask=mask, cache=cache)
        modified = self._on_post(out)
        return out if modified is None else modified


def _install_wrapper(model: LFM2AudioModel, layer_idx: int, on_post):
    """Splice _WrappedLayer at layer_idx; return a restorer callable."""
    original = model.lfm.layers[layer_idx]
    model.lfm.layers[layer_idx] = _WrappedLayer(original, on_post)

    def restore() -> None:
        model.lfm.layers[layer_idx] = original

    return restore


REPO = "mlx-community/LFM2.5-Audio-1.5B-4bit"

# Phase-5-relevant probe prompts. Mirror the kinds of contrast-pair
# questions 02b will emit: short, neutral, factual or conversational.
G_F2_PROMPTS = [
    "Tell me about your day.",
    "What's your opinion on remote work?",
    "Describe a place you'd like to visit.",
    "How do you feel about modern technology?",
    "What's a book you would recommend?",
    "Describe a perfect afternoon.",
    "What's the most important thing in life?",
    "How would you spend a free Saturday?",
    "What's something you've learned recently?",
    "Describe your ideal job.",
]

G_F3_SYSTEM_PROMPT = (
    "You are a thoughtful conversational assistant. "
    "Respond in plain text only — do not produce audio. "
    "Keep responses to 2-4 sentences."
)
G_F3_PROMPTS = [
    "How are you feeling today?",
    "What do you think about creative writing?",
    "Describe what makes a good meal.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=REPO, help="HF repo id for the LFM2-Audio MLX model.")
    p.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Trunk layer index L for capture/inject. Defaults to floor(0.6 * num_hidden_layers).",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Cap per-generation token count (G-F2 / G-F3).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=4.0,
        help="Steering coefficient magnitude for G-F2 (random v; large to force divergence).",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def banner(s: str) -> None:
    print(f"\n=== {s} ===", file=sys.stderr, flush=True)


def load_model(repo: str) -> Tuple[LFM2AudioProcessor, LFM2AudioModel]:
    banner(f"loading {repo}")
    t0 = time.time()
    processor = LFM2AudioProcessor.from_pretrained(repo)
    model = LFM2AudioModel.from_pretrained(repo)
    print(f"  loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    cfg = model.config.lfm
    print(f"  num_hidden_layers={cfg.num_hidden_layers}  hidden_size={cfg.hidden_size}",
          file=sys.stderr)
    return processor, model


def build_chat(processor: LFM2AudioProcessor, system: str, user: str) -> ChatState:
    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text(system)
    chat.end_turn()
    chat.new_turn("user")
    chat.add_text(user)
    chat.end_turn()
    chat.new_turn("assistant")
    return chat


# ──────────────────────────── G-F1 ─────────────────────────────────────

def gate_f1(model: LFM2AudioModel, processor: LFM2AudioProcessor, layer_idx: int) -> bool:
    banner(f"G-F1: capture hidden state at layer {layer_idx}")

    captured: List[mx.array] = []

    def on_post(h):
        captured.append(h)
        return None

    restore = _install_wrapper(model, layer_idx, on_post)

    chat = build_chat(processor, G_F3_SYSTEM_PROMPT, "Say hi.")
    try:
        gen = model.generate_sequential(
            **dict(chat), max_new_tokens=4, temperature=0.0, top_k=1,
        )
        # Consume a few tokens to drive the forward pass.
        for _ in range(4):
            try:
                next(gen)
            except StopIteration:
                break
    finally:
        restore()

    if not captured:
        print("  FAIL: no hidden states captured (hook never fired)", file=sys.stderr)
        return False

    h = captured[0]
    mx.eval(h)
    print(f"  captured {len(captured)} forward(s); first shape={tuple(h.shape)} dtype={h.dtype}",
          file=sys.stderr)

    expected_dim = model.config.lfm.hidden_size
    if h.ndim != 3 or h.shape[-1] != expected_dim:
        print(f"  FAIL: expected (B, T, {expected_dim}); got {tuple(h.shape)}", file=sys.stderr)
        return False

    print(f"  PASS  (B={h.shape[0]}, T={h.shape[1]}, d_model={h.shape[-1]})", file=sys.stderr)
    return True


# ──────────────────────────── G-F2 ─────────────────────────────────────

def _generate_text_tokens(
    model: LFM2AudioModel,
    processor: LFM2AudioProcessor,
    user_prompt: str,
    max_new_tokens: int,
) -> Tuple[List[int], int]:
    """Generate up to max_new_tokens; collect text-token IDs only.

    Returns (text_token_ids, total_tokens_seen). Stops at <|im_end|>, at
    audio-modality switch, or at max_new_tokens.
    """
    chat = build_chat(processor, G_F3_SYSTEM_PROMPT, user_prompt)
    gen = model.generate_sequential(
        **dict(chat), max_new_tokens=max_new_tokens, temperature=0.0, top_k=1,
    )
    text_ids: List[int] = []
    total = 0
    for token, modality in gen:
        total += 1
        mx.eval(token)
        if modality == LFMModality.TEXT:
            text_ids.append(int(token.item()))
        else:
            # Bail at first audio token; the spike only cares about text.
            break
    return text_ids, total


def gate_f2(
    model: LFM2AudioModel,
    processor: LFM2AudioProcessor,
    layer_idx: int,
    alpha: float,
    max_new_tokens: int,
    seed: int,
) -> bool:
    banner(f"G-F2: inject alpha*v at layer {layer_idx}, alpha=+/-{alpha}")

    d_model = model.config.lfm.hidden_size
    rng = np.random.default_rng(seed)
    v_np = rng.standard_normal(d_model).astype(np.float32)
    v_np = v_np / np.linalg.norm(v_np)  # unit-norm random direction
    v_mx = mx.array(v_np)

    state = {"alpha": 0.0}

    def on_post(h):
        a = state["alpha"]
        return None if a == 0.0 else h + (a * v_mx)

    restore = _install_wrapper(model, layer_idx, on_post)

    n_diverge = 0
    n_total = 0
    samples: List[Tuple[str, List[int], List[int]]] = []
    try:
        for prompt in G_F2_PROMPTS:
            state["alpha"] = +alpha
            ids_pos, _ = _generate_text_tokens(model, processor, prompt, max_new_tokens)
            state["alpha"] = -alpha
            ids_neg, _ = _generate_text_tokens(model, processor, prompt, max_new_tokens)
            n_total += 1
            if ids_pos != ids_neg:
                n_diverge += 1
            samples.append((prompt, ids_pos, ids_neg))
    finally:
        restore()

    rate = n_diverge / max(n_total, 1)
    print(f"  diverged on {n_diverge}/{n_total} prompts ({rate:.0%})", file=sys.stderr)

    # Print the first sample as a sanity check.
    if samples:
        prompt, ids_pos, ids_neg = samples[0]
        try:
            text_pos = processor.tokenizer.decode(ids_pos, skip_special_tokens=False)
            text_neg = processor.tokenizer.decode(ids_neg, skip_special_tokens=False)
        except Exception:
            text_pos = f"<{len(ids_pos)} tokens>"
            text_neg = f"<{len(ids_neg)} tokens>"
        print(f"  sample prompt: {prompt!r}", file=sys.stderr)
        print(f"    +alpha: {text_pos!r}", file=sys.stderr)
        print(f"    -alpha: {text_neg!r}", file=sys.stderr)

    passed = rate >= 0.8
    print(f"  {'PASS' if passed else 'FAIL'}  (threshold 80%)", file=sys.stderr)
    return passed


# ──────────────────────────── G-F3 ─────────────────────────────────────

def gate_f3(
    model: LFM2AudioModel,
    processor: LFM2AudioProcessor,
    max_new_tokens: int,
) -> bool:
    banner("G-F3: text-only contrast prompts elicit >= 20 text tokens")

    counts: List[int] = []
    samples: List[Tuple[str, str]] = []
    for prompt in G_F3_PROMPTS:
        ids, total = _generate_text_tokens(model, processor, prompt, max_new_tokens)
        counts.append(len(ids))
        try:
            text = processor.tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:
            text = f"<{len(ids)} text tokens, {total} total>"
        samples.append((prompt, text))
        print(f"  {len(ids):3d} text / {total:3d} total  prompt={prompt!r}", file=sys.stderr)

    print("  sample response:", file=sys.stderr)
    print(f"    {samples[0][0]!r} -> {samples[0][1]!r}", file=sys.stderr)

    passed = all(c >= 20 for c in counts)
    print(f"  {'PASS' if passed else 'FAIL'}  (min={min(counts)}, threshold=20 / prompt)",
          file=sys.stderr)
    return passed


# ──────────────────────────── main ─────────────────────────────────────

def main() -> int:
    args = parse_args()
    processor, model = load_model(args.repo)

    n_layers = model.config.lfm.num_hidden_layers
    layer_idx = args.layer if args.layer is not None else int(0.6 * n_layers)
    if not (0 <= layer_idx < n_layers):
        print(f"error: --layer {layer_idx} out of range [0, {n_layers})", file=sys.stderr)
        return 2

    print(f"\n→ probing trunk layer {layer_idx}/{n_layers}", file=sys.stderr)

    f1 = gate_f1(model, processor, layer_idx)
    f2 = gate_f2(model, processor, layer_idx, args.alpha, args.max_new_tokens, args.seed)
    f3 = gate_f3(model, processor, args.max_new_tokens)

    banner("summary")
    print(f"  G-F1 capture        : {'PASS' if f1 else 'FAIL'}", file=sys.stderr)
    print(f"  G-F2 inject + steer : {'PASS' if f2 else 'FAIL'}", file=sys.stderr)
    print(f"  G-F3 text-only resp : {'PASS' if f3 else 'FAIL'}", file=sys.stderr)
    overall = f1 and f2 and f3
    print(f"  OVERALL             : {'PASS — pivot ACCEPTED' if overall else 'FAIL — pivot REJECTED'}",
          file=sys.stderr)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
