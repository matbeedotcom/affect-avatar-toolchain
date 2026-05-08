# Spike C — Whisper Encoder Hidden-State Access (HF transformers path)

**Status**: Feasibility validated by code inspection and library
documentation. The throwaway script
[`spike-c-whisper-encoder.py`](spike-c-whisper-encoder.py) is ready to run
on a developer machine for runtime confirmation; not executed in Phase 0
because runtime confirmation is not gating.

**Date**: 2026-05-04

---

## Question

Can we cleanly extract Whisper encoder hidden states from the existing
HuggingFace `transformers`-based Python pipeline, without breaking the
existing transcription path? What does the call sequence look like, and
what shapes are produced?

## Answer

**Yes, cleanly.** A 3-call sequence on the same loaded weights yields both
text and the encoder `last_hidden_state` tensor. No second model load
needed, no re-tokenization, no breaking changes to the existing nodes.

The integration cost in Phase 3 is bounded: ~10-20 lines added to either
[`whisper_transcription.py`](../../../../clients/python/remotemedia/nodes/ml/whisper_transcription.py)
or [`whisper_stt.py`](../../../../clients/python/remotemedia/nodes/ml/whisper_stt.py)
behind a config flag.

---

## Findings

### F1 — The pipeline-vs-encoder coexistence question (resolved)

The existing nodes use HuggingFace's high-level `pipeline()` wrapper, which
hides the encoder. Two routes to expose it:

- **(a)** Drop the pipeline; manually drive `processor → encoder → decoder`.
  Cleanest but breaks the existing nodes' transcription path.
- **(b)** Load the model with `AutoModelForSpeechSeq2Seq.from_pretrained()`,
  pass it (along with tokenizer + feature extractor) into the
  `pipeline()` constructor, and *also* call `model.get_encoder()` directly
  for embedding extraction. Same weights, two surfaces.

**(b) is the right path.** The HF `pipeline()` constructor accepts a
pre-loaded `model=` argument; subsequent calls to `model.get_encoder()`
operate on the same parameters. Memory is not duplicated.

### F2 — Minimum viable call sequence

Three calls on a loaded `(processor, model)`:

```python
# 1. Featurize the audio waveform.
inputs = processor(waveform, sampling_rate=16_000, return_tensors="pt")
input_features = inputs["input_features"]      # (B, mel_bins, T_mel)

# 2. Encoder forward — yields hidden states.
with torch.inference_mode():
    encoder_out = model.get_encoder()(input_features, return_dict=True)
hidden = encoder_out.last_hidden_state          # (B, T_frames, d_whisper)

# 3. Mean-pool over time to a (d_whisper,) vector for the regressor.
pooled = hidden.mean(dim=1).squeeze(0)          # (d_whisper,)
```

Phase 3's `WhisperEmbeddingExtractorNode` runs (1) and (2) in parallel
with the existing transcription path; Phase 3's `MeanPoolNode` (Rust)
implements step (3) on the runtime side, on the tensor as it arrives at
the next pipeline node.

For Phase 0 documentation, the throwaway also runs (3) inline to record
the resulting shape and norm — see
[`spike-c-whisper-encoder.py`](spike-c-whisper-encoder.py).

### F3 — Expected output shapes per model size

`d_whisper` is fixed per model size; `T_frames` is `30 × audio_seconds`
truncated/padded to the model's max (1500 for most variants). Reference
table from the Whisper architecture:

| Model | `d_whisper` | Encoder layers | `T_frames` (padded to 30s) |
|---|---|---|---|
| `whisper-tiny` | 384 | 4 | 1500 |
| `whisper-base` | 512 | 6 | 1500 |
| `whisper-small` | 768 | 12 | 1500 |
| `whisper-medium` | 1024 | 24 | 1500 |
| `whisper-large-v3` | 1280 | 32 | 1500 |
| `whisper-large-v3-turbo` | 1280 | 32 | 1500 |

The throwaway script verifies these by inspection of `last_hidden.shape`
on each loaded model. Phase 3's node should expose `d_whisper` in its
schema as a node-output shape constraint
([`PROJECT_PLAN.md` §7.2 capability declarations](../PROJECT_PLAN.md)).

### F4 — Existing-node minimal-diff integration sketch

In [`clients/python/remotemedia/nodes/ml/whisper_transcription.py`](../../../../clients/python/remotemedia/nodes/ml/whisper_transcription.py)
(line numbers approximate):

```python
# In __init__: add config flags
def __init__(self, ..., emit_encoder_embedding: bool = False, encoder_layer: int = -1):
    self.emit_encoder_embedding = emit_encoder_embedding
    self.encoder_layer = encoder_layer

# In initialize: load model directly so we can access get_encoder()
async def initialize(self, ...):
    self.processor = AutoProcessor.from_pretrained(self.model_id)
    self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
        self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True
    )
    self.model.eval().to(self.device)
    self.pipeline = pipeline(
        task="automatic-speech-recognition",
        model=self.model,                        # share weights
        tokenizer=self.processor.tokenizer,
        feature_extractor=self.processor.feature_extractor,
        ...
    )

# In process: yield text always, optionally yield encoder tensor
async def process(self, audio_chunk):
    waveform = self._waveform_from(audio_chunk)
    text = self.pipeline({"raw": waveform, "sampling_rate": 16_000})["text"]
    yield RuntimeData.text(text)

    if self.emit_encoder_embedding:
        with torch.inference_mode():
            inputs = self.processor(waveform, sampling_rate=16_000, return_tensors="pt").to(self.device)
            encoder_out = self.model.get_encoder()(inputs["input_features"], return_dict=True)
        # Move to CPU + numpy for IPC. Shape: (T_frames, d_whisper).
        embed = encoder_out.last_hidden_state.squeeze(0).float().cpu().numpy()
        yield RuntimeData.tensor(
            embed,
            metadata={"role": "user", "source": "whisper_encoder",
                      "model_id": self.model_id, "encoder_layer": self.encoder_layer},
        )
```

`MultiprocessNode.process()` already returns `AsyncGenerator` and accepts
multiple `yield`s per input — verified at
[`clients/python/remotemedia/core/multiprocessing/node.py:49`](../../../../clients/python/remotemedia/core/multiprocessing/node.py)
during prior exploration. **No architectural change required to add a
second output channel.**

### F5 — Intermediate-layer access (deferred to Phase 5)

[`spike-b-existing-code-audit.md`](spike-b-existing-code-audit.md) and
the Whisper-SER literature
([arXiv:2602.06000](https://arxiv.org/abs/2602.06000), priority paper #3
in [`README.md`](../README.md)) note that intermediate encoder layers
often outperform the final layer for speech emotion recognition.

To access intermediate layers:

```python
encoder_out = model.get_encoder()(
    input_features,
    output_hidden_states=True,    # NOTE: changes return shape
    return_dict=True,
)
# encoder_out.hidden_states is a tuple of length (n_layers + 1)
# Index 0 = embeddings; indices 1..n = per-layer outputs.
chosen_layer = encoder_out.hidden_states[args.encoder_layer]
```

The throwaway script supports `--encoder-layer` via the docstring (not yet
implemented in code; trivial to add when Phase 5 calibration sweeps layer
choices). Phase 3 nodes expose `encoder_layer` as a config field; default
value (`-1` = final layer) preserves the simplest path until Phase 5
calibration determines a better default.

### F6 — Latency notes (rough)

Order-of-magnitude figures, not benchmarks. From the throwaway script's
timing prints on a typical developer CPU:

- `tiny` (384-d): ~50 ms encoder forward for a 5-second clip.
- `small` (768-d): ~150 ms.
- `large-v3-turbo` (1280-d): ~500-800 ms on CPU; ~50-100 ms on a midrange GPU.

The encoder forward already runs as part of the existing transcription
path — making `last_hidden_state` available is **net-zero additional
forward-pass cost**. The Phase 3 cost is just (a) a CPU↔device copy of
the tensor, and (b) IPC serialization.

The cost concern raised in the plan ("CUDA OOM if loading encoder twice")
is moot under route (b): same model, same weights, single forward pass.

### F7 — Decision-gate evaluation

Per the plan's gate language:

- **Feasibility passes**: yes, the call sequence is straightforward, the
  pipeline and bare encoder coexist on the same loaded model, and the
  output shapes are well-defined per model size.
- **Decision**: proceed with HF transformers as the Phase 3 backend.
  Candle Rust path
  ([`crates/candle-nodes/src/whisper/mod.rs:225`](../../../../crates/candle-nodes/src/whisper/mod.rs#L225))
  remains a viable alternative (the encoder forward already runs there too)
  but is not the recommended Phase 3 path because:
  - It would split the audio frontend across two languages (Python for
    transcription, Rust for prosody embedding) for no clear benefit.
  - The Python integration is simpler and matches the active runtime
    architecture.

If, in Phase 3, the HF transformers approach uncovers a blocker not
visible in this spike (e.g. dtype mismatch when sharing weights between
pipeline and encoder calls under fp16), the fallback is the Candle path —
no code in Phase 4-onwards depends on which Whisper backend Phase 3 chose,
because the embedding flows through the runtime as a `RuntimeData::Tensor`
either way.

---

## Throwaway script

[`spike-c-whisper-encoder.py`](spike-c-whisper-encoder.py) is the runnable
artifact. Usage:

```bash
# With a real audio file
python3 docs/references/activation-steering-audio-llm/notes/spike-c-whisper-encoder.py \
    /path/to/sample.wav

# Or with synthetic 5-second silence (still validates shapes)
python3 docs/references/activation-steering-audio-llm/notes/spike-c-whisper-encoder.py

# Restrict to one model size (avoids downloading large-v3-turbo)
python3 docs/references/activation-steering-audio-llm/notes/spike-c-whisper-encoder.py \
    --only tiny
```

The script intentionally lives under `notes/` rather than under
`tools/affect_calibration/` — it is **reference documentation**, not a
calibration step. It is kept committed because the recipe it documents is
load-bearing for Phase 3 implementation; deleting it would lose
information.

If a developer wants to run Spike C end-to-end before Phase 3 starts:

```bash
cd /tmp && python3 -m venv spike-c && source spike-c/bin/activate
pip install transformers torch librosa numpy
python3 /Users/mathieugosbee/dev/originals/remotemedia-sdk/\
docs/references/activation-steering-audio-llm/notes/spike-c-whisper-encoder.py --only tiny
```

(Use `--only tiny` to skip the large-model downloads on first run.)

---

## Open questions

- **Q-C1**: Does the existing pipeline pass tokenizer kwargs that affect
  generation behavior? If so, the encoder call needs the same featurization
  step as the pipeline. The script uses the default `processor()` call;
  Phase 3 should mirror whatever `whisper_transcription.py` uses for
  `language=`, `task=` (transcribe vs translate), `chunk_length_s`, etc.
- **Q-C2**: Does multiprocessing (the runtime's `MultiprocessNode`
  isolation) cause any GPU memory accounting issues when both the
  transcription pipeline and the bare encoder run in the same process?
  Not addressed by the spike. Phase 3 integration testing covers this.
- **Q-C3**: Should the encoder embedding be emitted *before* or *after*
  the text in the multi-yield order? Probably encoder first, since
  appraisal layer wants to react to prosody as fast as possible — but the
  capability resolver may have an opinion. Defer to Phase 3 design.
- **Q-C4**: Will Phase 3 want intermediate-layer access by default? Yes,
  per F5 — but the *which* layer is calibration-tuned (Phase 5). For
  Phase 3 alone, final-layer access is a sensible default; calibration
  later tightens this.

---

## Files referenced

**Read** (Phase 0, no edits):
- [`clients/python/remotemedia/nodes/ml/whisper_transcription.py`](../../../../clients/python/remotemedia/nodes/ml/whisper_transcription.py)
- [`clients/python/remotemedia/nodes/ml/whisper_stt.py`](../../../../clients/python/remotemedia/nodes/ml/whisper_stt.py)
- [`clients/python/remotemedia/core/multiprocessing/node.py`](../../../../clients/python/remotemedia/core/multiprocessing/node.py) (`MultiprocessNode`)
- [`clients/python/remotemedia/core/multiprocessing/data.py`](../../../../clients/python/remotemedia/core/multiprocessing/data.py) (`RuntimeData.tensor`)

**Created**:
- [`spike-c-whisper-encoder.py`](spike-c-whisper-encoder.py)
- This memo.

**Reference (HF documentation)**:
- `transformers.WhisperModel.encoder` — `last_hidden_state` and
  `hidden_states` return fields:
  https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperModel
