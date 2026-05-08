"""MLX Llama / Hermes target-LLM provider for the persona-vectors pipeline.

Parallel to ``mlx_target.MLXTargetLLM`` but wraps any chat model loadable
via ``mlx_lm.load`` (Llama-3.x family, Hermes-3, Dolphin 3.0, etc.).
The protocol-conforming method ``generate_and_pool_response`` returns
the per-response-token mean-pooled hidden state at one chosen layer.

Why this exists separately: LFM2-Audio runs through the ``mlx_audio``
package's bespoke ``ChatState`` + ``generate_sequential`` API and lives
at ``model.lfm.layers``; pure Llama models go through ``mlx_lm`` and
live at ``model.model.layers`` with no audio modality concept. The
wrapper-substitution mechanism is identical, but the model-load and
generation calls differ enough that one provider per architecture
keeps each one readable.

Multi-layer harvest:
    ``harvest_all_layers(system_prompt, question)`` captures *every*
    layer's post-block residual on a single forward, returning a
    list of L mean-pooled vectors. Designed for one-pass extraction
    across all candidate layers in a sweep — much cheaper than
    re-running 1200 contrast pairs once per layer.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np


class _LlamaWrappedLayer:
    """Substitutes for one ``llama.TransformerBlock``; exposes a post-forward hook.

    Mirrors ``mlx_target._WrappedLayer``. Patching ``instance.__call__``
    is silently broken on MLX nn.Module (Python dispatches via type, not
    the instance attribute); only swapping the layer reference works.

    Forwards ``use_sliding`` because LlamaModel.__call__ reads it on each
    layer to pick the right attention mask.
    """

    def __init__(self, inner: Any, on_post):
        self.inner = inner
        self._on_post = on_post

    @property
    def use_sliding(self) -> bool:
        return bool(getattr(self.inner, "use_sliding", False))

    def __call__(self, x, mask=None, cache=None):
        out = self.inner(x, mask=mask, cache=cache)
        modified = self._on_post(out)
        return out if modified is None else modified


def _install_wrapper(model: Any, layer_idx: int, on_post):
    layers = model.model.layers
    original = layers[layer_idx]
    layers[layer_idx] = _LlamaWrappedLayer(original, on_post)

    def restore() -> None:
        layers[layer_idx] = original

    return restore


def _install_all_wrappers(model: Any, on_post_per_layer):
    """Wrap every layer; on_post_per_layer is called with (layer_idx, h)."""
    layers = model.model.layers
    originals = list(layers)

    def make_hook(idx):
        return lambda h: on_post_per_layer(idx, h)

    for i in range(len(originals)):
        layers[i] = _LlamaWrappedLayer(originals[i], make_hook(i))

    def restore() -> None:
        for i, orig in enumerate(originals):
            layers[i] = orig

    return restore


class MLXLlamaTargetLLM:
    """TargetLLM provider over an mlx_lm-loaded chat model (Llama family)."""

    def __init__(
        self,
        model_id_or_path: str,
        *,
        max_response_tokens: int = 64,
    ) -> None:
        # Lazy imports — mlx + mlx_lm + ~5 GB of weights only when extracting.
        import mlx.core as mx  # noqa: F401
        from mlx_lm import load as _load
        from mlx_lm.generate import stream_generate as _stream_generate

        self._mx = mx
        self._stream_generate = _stream_generate

        self.model_id = model_id_or_path
        self.model, self.tokenizer = _load(model_id_or_path)

        # Hermes-3 / Dolphin / Llama-3.1 all expose `args` on the inner Model.
        args = self.model.args
        self.n_embd: int = int(args.hidden_size)
        self.num_hidden_layers: int = int(args.num_hidden_layers)
        self.max_response_tokens = int(max_response_tokens)

    # ------------------------------------------------------------------
    # TargetLLM Protocol — single-layer pool
    # ------------------------------------------------------------------

    def generate_and_pool_response(
        self,
        system_prompt: str,
        question: str,
        layer: int,
        pooling: str = "response_mean",
    ) -> List[float]:
        if pooling != "response_mean":
            raise ValueError(
                f"MLXLlamaTargetLLM only supports pooling='response_mean'; got {pooling!r}"
            )
        if not (0 <= layer < self.num_hidden_layers):
            raise ValueError(
                f"layer {layer} out of range [0, {self.num_hidden_layers})"
            )

        captured: List[Any] = []

        def on_post(h):
            captured.append(h)
            return None

        restore = _install_wrapper(self.model, layer, on_post)
        try:
            self._drain_generation(system_prompt, question)
        finally:
            restore()

        # captured[0] = prefill (full prompt); captured[1:] = per-response-token.
        if len(captured) < 2:
            raise RuntimeError(
                f"MLXLlamaTargetLLM: response too short to pool "
                f"(captured={len(captured)} forward(s)); "
                f"system={system_prompt!r} question={question!r}"
            )
        per_token = []
        for h in captured[1:]:
            per_token.append(h[0, -1, :])
        stacked = self._mx.stack(per_token, axis=0)
        pooled = self._mx.mean(stacked, axis=0).astype(self._mx.float32)
        self._mx.eval(pooled)
        return np.asarray(pooled).reshape(-1).tolist()

    # ------------------------------------------------------------------
    # Multi-layer harvest — pool every layer in a single forward
    # ------------------------------------------------------------------

    def harvest_all_layers(
        self,
        system_prompt: str,
        question: str,
    ) -> np.ndarray:
        """Return (num_layers, n_embd) mean-pooled response activations.

        One forward through generate; all 32 layers' residuals captured
        at every step. Pool over response tokens (skip prefill).
        """
        # captured[layer_idx] is a list of post-block tensors per call.
        captured: List[List[Any]] = [[] for _ in range(self.num_hidden_layers)]

        def on_post(layer_idx: int, h):
            captured[layer_idx].append(h)
            return None

        restore = _install_all_wrappers(self.model, on_post)
        try:
            self._drain_generation(system_prompt, question)
        finally:
            restore()

        # Each list: [prefill_tensor, per_token_tensor_1, ..., per_token_tensor_N].
        # Drop prefill (index 0); pool the rest.
        out = np.zeros((self.num_hidden_layers, self.n_embd), dtype=np.float32)
        for li, hs in enumerate(captured):
            if len(hs) < 2:
                raise RuntimeError(
                    f"layer {li}: response too short to pool "
                    f"(captured={len(hs)} forward(s))"
                )
            per_token = self._mx.stack([h[0, -1, :] for h in hs[1:]], axis=0)
            pooled = self._mx.mean(per_token, axis=0).astype(self._mx.float32)
            self._mx.eval(pooled)
            out[li] = np.asarray(pooled).reshape(-1)
        return out

    # ------------------------------------------------------------------
    # Steered generation (used by 04_validate_pipeline.py)
    # ------------------------------------------------------------------

    def generate_text(
        self,
        system_prompt: str,
        question: str,
        *,
        max_new_tokens: int = 128,
        layer: Optional[int] = None,
        direction: Optional[Sequence[float]] = None,
        alpha: float = 0.0,
    ) -> str:
        if alpha != 0.0:
            if direction is None or layer is None:
                raise ValueError(
                    "steered generate_text() requires both `direction` and `layer`"
                )
            if not (0 <= layer < self.num_hidden_layers):
                raise ValueError(
                    f"layer {layer} out of range [0, {self.num_hidden_layers})"
                )
            v_arr = np.asarray(direction, dtype=np.float32).reshape(-1)
            if v_arr.size != self.n_embd:
                raise ValueError(
                    f"direction has {v_arr.size} dims; expected {self.n_embd}"
                )
            v_mx = self._mx.array(v_arr)

            def on_post(h):
                return h + (alpha * v_mx)

            restore = _install_wrapper(self.model, layer, on_post)
        else:
            def restore() -> None:
                return None

        try:
            chunks: List[str] = []
            prompt_str = self._build_prompt(system_prompt, question)
            for resp in self._stream_generate(
                self.model, self.tokenizer, prompt=prompt_str,
                max_tokens=max_new_tokens, sampler=self._greedy_sampler(),
            ):
                if resp.text:
                    chunks.append(resp.text)
            return "".join(chunks).strip()
        finally:
            restore()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, system_prompt: str, question: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def _greedy_sampler(self):
        from mlx_lm.sample_utils import make_sampler
        return make_sampler(temp=0.0)

    def _drain_generation(self, system_prompt: str, question: str) -> int:
        """Run greedy generation to completion or max_tokens, return token count."""
        prompt_str = self._build_prompt(system_prompt, question)
        n = 0
        for _ in self._stream_generate(
            self.model, self.tokenizer, prompt=prompt_str,
            max_tokens=self.max_response_tokens, sampler=self._greedy_sampler(),
        ):
            n += 1
        return n
