"""MLX TargetLLM provider for the persona-vectors pipeline.

Pairs with notes/spike-g-mlx-target-llm.md (memo) and
notes/spike-g-mlx-extraction.py (verified pattern). Wraps a loaded
LFM2-Audio MLX model so that 03_extract_llm_directions.py can compute
response-token mean-pooled hidden states at a chosen trunk layer.

Key design points (all verified by Spike G):

- Hook mechanism: substitute a wrapper nn.Module into
  `model.lfm.layers[L]`. Patching `instance.__call__` is silently
  broken on MLX nn.Module because Python dispatches via
  `type(obj).__call__`, not the instance attribute.

- Sampling: pass `temperature=0.0, top_k=1` (deterministic greedy)
  whenever the *direction extraction* path runs. Stochastic sampling
  introduces noise across pairs that gets absorbed into the
  mean-difference and dilutes the recovered direction.

- Pooling: mean over response-token hidden states only (paper App. A.3).
  The wrapper captures the per-step trunk forward; we discard the
  prefill capture (prompt context) and pool the per-token forwards.

- Modality: contrast prompts are text-in / text-out (Option α from the
  Spike G memo); we bail if the model switches to AUDIO_OUT modality
  before reaching `<|im_end|>`.

This module is imported lazily — it pulls mlx, mlx-audio, and the
~1.95 GB model snapshot. 03's --dry-run path never imports it.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np


# LFM2-Audio special token: end-of-turn marker; also used to terminate
# generation. See model.py:39 (IM_END_TOKEN = 7).
_IM_END_TOKEN = 7


class _WrappedLayer:
    """Substitutes for one `Lfm2DecoderLayer` to expose a post-forward hook.

    See spike-g-mlx-extraction.py:_WrappedLayer for the verified version
    of this pattern. We use a plain object (not nn.Module) here because
    we need only delegation: parameter discovery on the inner module
    still works because the parent (Lfm2Model) holds the original
    reference indirectly via cache materialization, and the inner module
    is what carries weights.
    """

    def __init__(self, inner: Any, on_post):
        self.inner = inner
        self._on_post = on_post

    @property
    def is_attention_layer(self) -> bool:
        return self.inner.is_attention_layer

    def __call__(self, x, mask=None, cache=None):
        out = self.inner(x, mask=mask, cache=cache)
        modified = self._on_post(out)
        return out if modified is None else modified


def _install_wrapper(model: Any, layer_idx: int, on_post):
    original = model.lfm.layers[layer_idx]
    model.lfm.layers[layer_idx] = _WrappedLayer(original, on_post)

    def restore() -> None:
        model.lfm.layers[layer_idx] = original

    return restore


class MLXTargetLLM:
    """Concrete `TargetLLM` provider over an MLX-loaded LFM2-Audio model.

    Conforms to the Protocol declared in `lib.persona_pipeline.TargetLLM`:
    a single method `generate_and_pool_response(system, question, layer,
    pooling) -> list[float]`. Layer is mandatory and per-call so a single
    loaded model can extract at multiple layers if the manifest sweeps.
    """

    def __init__(
        self,
        model_id_or_path: str,
        *,
        max_response_tokens: int = 64,
    ) -> None:
        # Lazy imports — keep `import lib.mlx_target` cheap so the dry-run
        # paths in 03/04 don't pull mlx + ~2 GB of weights when not needed.
        import mlx.core as mx  # noqa: F401  (used at call time)
        from mlx_audio.sts.models.lfm_audio import (  # type: ignore
            ChatState,
            LFMModality,
            LFM2AudioModel,
            LFM2AudioProcessor,
        )

        self._mx = mx
        self._ChatState = ChatState
        self._LFMModality = LFMModality

        self.model_id = model_id_or_path
        self.processor = LFM2AudioProcessor.from_pretrained(model_id_or_path)
        self.model = LFM2AudioModel.from_pretrained(model_id_or_path)

        cfg = self.model.config.lfm
        self.n_embd: int = int(cfg.hidden_size)
        self.num_hidden_layers: int = int(cfg.num_hidden_layers)
        self.max_response_tokens = int(max_response_tokens)

    # ------------------------------------------------------------------
    # TargetLLM Protocol
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
                f"MLXTargetLLM only supports pooling='response_mean'; got {pooling!r}"
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
            chat = self._build_chat(system_prompt, question)
            gen = self.model.generate_sequential(
                **dict(chat),
                max_new_tokens=self.max_response_tokens,
                temperature=0.0,
                top_k=1,
            )
            text_token_count = 0
            for token, modality in gen:
                self._mx.eval(token)
                if modality == self._LFMModality.TEXT:
                    text_token_count += 1
                else:
                    # Bail at first audio token (Option α: text-in/text-out).
                    break
        finally:
            restore()

        # captured[0] = prefill forward (full prompt). captured[1:] are
        # per-response-token forwards, each with shape (1, 1, d_model).
        # Pool over response tokens only.
        if len(captured) < 2:
            raise RuntimeError(
                f"MLXTargetLLM: response too short to pool "
                f"(captured={len(captured)} forward(s), text_tokens={text_token_count}); "
                f"system={system_prompt!r} question={question!r}"
            )

        # Stack the per-token forwards along a new axis, then mean.
        per_token = []
        for h in captured[1:]:
            # h shape: (B, T, D). For per-step forward T=1; take last position.
            per_token.append(h[0, -1, :])
        stacked = self._mx.stack(per_token, axis=0)  # (n_response, d_model)
        pooled = self._mx.mean(stacked, axis=0)      # (d_model,)
        self._mx.eval(pooled)

        arr = np.array(pooled.astype(self._mx.float32))
        return arr.reshape(-1).tolist()

    # ------------------------------------------------------------------
    # Generation (used by 04_validate_pipeline.py)
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
        """Generate a text-only response, optionally steered.

        When `alpha != 0`, both `layer` and `direction` are required and
        the wrapper hook adds `alpha * direction` to the post-block residual
        at `layer` (same site as `generate_and_pool_response`'s capture).

        Sampling is deterministic-greedy (`temperature=0, top_k=1`) so two
        runs with the same args produce identical strings — required for
        the §4.6 validation gap to be a stable signal rather than a
        sampling-noise echo.
        """
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

        text_ids: List[int] = []
        try:
            chat = self._build_chat(system_prompt, question)
            gen = self.model.generate_sequential(
                **dict(chat),
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_k=1,
            )
            for token, modality in gen:
                self._mx.eval(token)
                if modality != self._LFMModality.TEXT:
                    # Bail at first audio token (Option α: text-in/text-out).
                    break
                tid = int(token.item())
                if tid == _IM_END_TOKEN:
                    break
                text_ids.append(tid)
        finally:
            restore()

        if not text_ids:
            return ""
        return self.processor.tokenizer.decode(text_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_chat(self, system_prompt: str, question: str):
        chat = self._ChatState(self.processor)
        chat.new_turn("system")
        chat.add_text(system_prompt)
        chat.end_turn()
        chat.new_turn("user")
        chat.add_text(question)
        chat.end_turn()
        chat.new_turn("assistant")
        return chat
