"""Whisper encoder wrapper for hidden-state extraction.

Implements the verified pattern from
`docs/references/activation-steering-audio-llm/notes/spike-c-whisper-encoder.py`.
Wraps `transformers.AutoModelForSpeechSeq2Seq` so that 01 can pull the
encoder hidden state at any layer, mean-pool over time frames, and feed
the resulting `[d_whisper]` vector to the VAD regressor in 02.

Imports (torch / transformers) are lazy so the dry-run path stays free
of the ML stack.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class WhisperEncoderProvider:
    """Loaded Whisper model exposing layer-L mean-pooled encoder output.

    Conforms to the `WhisperEncoder` Protocol declared in
    01_extract_whisper_embeddings.py.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        dtype: str = "float32",
    ) -> None:
        import torch  # lazy
        from transformers import (  # lazy
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
        )

        self._torch = torch

        # Auto-detect device: CUDA > MPS (Apple Silicon) > CPU. On Apple
        # Silicon MPS gives a ~5x speedup over CPU for Whisper-large.
        # The 30-second processor padding makes per-clip CPU encode
        # painful (~12 sec/clip on M-series); MPS brings it to ~2 sec.
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.model_id = model_id
        self.device = device
        torch_dtype = getattr(torch, dtype)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
        self.model.eval()
        self.model.to(device)
        self.encoder = self.model.get_encoder()

        # `model.config.d_model` is the encoder hidden size for Whisper.
        cfg = self.model.config
        self.d_whisper: int = int(getattr(cfg, "d_model", 0))
        if self.d_whisper <= 0:
            raise RuntimeError(
                f"could not determine d_whisper from {model_id} "
                f"(model.config.d_model={getattr(cfg, 'd_model', None)!r})"
            )
        # Whisper has 4-32 encoder layers depending on size.
        self.num_layers: int = int(
            getattr(cfg, "encoder_layers", getattr(cfg, "num_hidden_layers", 0))
        )

        logger.info(
            "WhisperEncoderProvider: %s on %s  d_whisper=%d  encoder_layers=%d",
            model_id, self.device, self.d_whisper, self.num_layers,
        )

    # ------------------------------------------------------------------
    # WhisperEncoder Protocol (matches 01's class WhisperEncoder)
    # ------------------------------------------------------------------

    def encode(self, audio_16k_mono: np.ndarray, layer: int) -> np.ndarray:
        """Return mean-pooled hidden state at `layer` for one clip.

        `audio_16k_mono` is float32 PCM at 16 kHz, shape (T,) or (1, T).
        `layer` indexes encoder hidden states. -1 = final, 0 = embedding.
        Whisper exposes `output_hidden_states=True` only when the encoder
        is called with that flag; we pass it always to keep the API uniform.

        Returns: float32 (d_whisper,).
        """
        torch = self._torch
        wav = np.asarray(audio_16k_mono, dtype=np.float32).reshape(-1)

        # AutoProcessor handles log-mel + padding to fixed Whisper window
        # (30 s by default). Anything longer is truncated by the processor;
        # anything shorter is padded with silence. Phase 5 calibration
        # clips are <= 5 s typically, well within the window.
        inputs = self.processor(
            wav,
            sampling_rate=16_000,
            return_tensors="pt",
        )
        input_features = inputs["input_features"].to(self.device)

        with torch.inference_mode():
            out = self.encoder(
                input_features,
                output_hidden_states=True,
                return_dict=True,
            )

        # `out.hidden_states` is a tuple of length (num_layers + 1) — the
        # extra entry is the post-embedding pre-block residual. layer=-1
        # gives the final post-block residual; layer=0 gives the
        # post-embedding residual.
        hidden = out.hidden_states[layer]  # (1, T_frames, d_whisper)
        pooled = hidden.mean(dim=1).squeeze(0).float().cpu().numpy()
        return pooled.astype(np.float32, copy=False)
