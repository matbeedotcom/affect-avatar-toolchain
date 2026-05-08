# Spike G — Pivot the Phase 5 target LLM to LFM2-Audio-1.5B (MLX)

**Status**: **Accepted + R10 empirically mitigated** (2026-05-05). All three
feasibility gates passed on the loaded `mlx-community/LFM2.5-Audio-1.5B-4bit`
model; later the same day the full Phase 5 calibration run on this same
target passed the §4.6 gate on all three V/A/D axes (PROJECT_PLAN R10
table). Results section below records the original feasibility gates;
the production calibration numbers live in
[`PROJECT_PLAN.md` §10 R10](../PROJECT_PLAN.md).

**Date**: 2026-05-05

**Paired script**: [`spike-g-mlx-extraction.py`](spike-g-mlx-extraction.py).

---

## Question

Phase 5 currently targets a 27B-class text LLM (Qwen3-27B / Llama-3-70B GGUF
via `llama.cpp` + `set_adapter_cvec`) for direction extraction and runtime
steering. Should we instead target the smaller multimodal model already
loaded into the codebase —
[`mlx-community/LFM2.5-Audio-1.5B-4bit`](https://huggingface.co/mlx-community/LFM2.5-Audio-1.5B-4bit)
via `mlx-audio`, wired up at
[clients/python/remotemedia/nodes/ml/lfm2_audio_mlx.py](../../../../clients/python/remotemedia/nodes/ml/lfm2_audio_mlx.py)?

## Tentative answer

**Yes, conditional on three feasibility checks passing.** This pivot collapses
several open risks and aligns calibration with the actual production target,
but it relocates the steering primitive from `llama.cpp` (Spike A) to MLX,
which is unproven for our use case. The conditional checks are spelled out
in [Feasibility gates](#feasibility-gates).

If any gate fails, fall back to the Qwen3-27B / `llama.cpp` plan. The Whisper
side (workstream D1.1, scripts 01/02) is unaffected either way and should
proceed in parallel.

---

## Why this pivot is attractive

### A1 — The target model is already in the production runtime

[`LFM2AudioMlxNode`](../../../../clients/python/remotemedia/nodes/ml/lfm2_audio_mlx.py)
is the speech-to-speech node the affective-agent runtime is being built
around. Calibrating directions on **this exact model** means directions are
directly deployable; calibrating on Qwen3-27B and shipping a different model
would force a re-calibration step at deploy time, or assume directions
transfer across architectures (they don't, in general).

### A2 — Within the persona-vectors validated regime

Chen et al. 2025 ([arXiv:2507.21509](https://arxiv.org/abs/2507.21509))
validated the pipeline on Llama-3-8B and Qwen-2.5-7B-Instruct (both 7-8B,
text-only). LFM2-Audio-1.5B is *smaller* than the validated regime — a
different open question (see [R10](#new-risk-r10)) — but at least it is no
longer the *27B+ scaling-unverified* regime that motivated
[PROJECT_PLAN R9](../PROJECT_PLAN.md). R9 becomes moot; R10 replaces it.

### A3 — Drops the D1.0 Rust-extractor prerequisite

[IMPLEMENTATION_PLAN.md §4.4](../IMPLEMENTATION_PLAN.md) names a Rust extractor
binary (`crates/core/examples/llama_cpp_extract_activations.rs`) as a D1.0
prerequisite for direction extraction (script 03). With MLX, extraction
happens in Python via `mlx-audio`'s already-loaded model; no Rust binary, no
subprocess plumbing. Net code reduction.

### A4 — Local dev-machine viable end-to-end

The current Phase 5 plan implicitly requires CUDA (Qwen3-27B in 4-bit GGUF
won't fit on a 36GB Apple GPU; even 7B GGUF wants a desktop). LFM2-Audio-1.5B
runs natively on the dev Mac in <2 GB. The full 02b → 03 → 04 chain is
runnable on the laptop with no rented hardware — meaning Phase 5 can
proceed incrementally instead of in batched cloud runs.

### A5 — Spike A finding does not invalidate

Spike A established that `set_adapter_cvec` is available in
`llama-cpp-4@0.2.50`. That finding is preserved as a fallback path: if
Spike G fails, Phase 5 reverts to the Qwen path with no regression. We
lose nothing by attempting this pivot.

---

## Why this is a redesign, not a swap

### B1 — Steering primitive must be re-implemented

`set_adapter_cvec` is a `llama.cpp` C-side hook that injects `α · v_axis`
into the residual stream at runtime, layer-ranged. **MLX has no equivalent.**
We will need to patch `LFM2AudioModel`'s forward pass directly:

- Identify the residual-stream layer L of choice.
- Hook the corresponding `mlx.nn.TransformerBlock` (or whatever the
  mlx-audio layer abstraction is named in
  [mlx_audio/sts/models/lfm_audio](https://github.com/Blaizzy/mlx-audio)).
- Add `α · v_axis` to the block's output (or input — either
  pre-residual-add or post-residual-add; pick one consistently with
  whatever gets used during direction extraction).
- Return the patched model from `_build_target_llm()` in
  [03_extract_llm_directions.py](../../../../tools/affect_calibration/scripts/03_extract_llm_directions.py)
  and from the steering hook in the runtime.

`mlx-audio` is permissively licensed (Apache 2.0) and Python-native, so
monkey-patching is straightforward; **we are not blocked by source access,
but we are blocked by figuring out which line of the forward pass is the
right place to hook**. That's G-F2 below.

### B2 — Pooling decision is non-trivial

Persona Vectors ([App. A.3](https://arxiv.org/abs/2507.21509)) pools mean
hidden states **over response tokens only** (text). LFM2-Audio's response
stream is interleaved text + audio-codebook tokens (see
[lfm2_audio_mlx.py:382-498](../../../../clients/python/remotemedia/nodes/ml/lfm2_audio_mlx.py#L382)).
The pool is now ambiguous. Three options:

| Option | Description | Risk |
|---|---|---|
| **Text-only pool** | Mean over response *text* tokens only; ignore audio codebook tokens. | **Recommended starter.** Mirrors paper protocol exactly. Assumes residual stream at layer L is shared across modalities (it is — the LFM-Audio architecture has one trunk). |
| Audio-only pool | Mean over response *audio* codebook tokens. | Untested; would test whether prosodic affect lives in the audio half differently. Defer. |
| Joint pool | Mean over all response tokens regardless of modality. | Mixes two distributions; harder to interpret. Defer. |

Pinning **text-only response pool** matches Spike F's prior decision
(see [spike-f-persona-vectors-pipeline.md](spike-f-persona-vectors-pipeline.md))
and inherits its 94.7% human-judge alignment.

### B3 — Contrast-prompt modality

02b currently emits text-only contrast pairs (system prompt + question). On a
text-only LLM that's the obvious input. On LFM2-Audio it's a question:

- **Option α (text in, text out)**: feed contrast prompts as text-only
  utterances; let the model emit a text-only response (no audio modality
  engaged). Simplest. Matches paper exactly.
- **Option β (audio in, text out)**: TTS the contrast question to
  audio, feed via `chat.add_audio`, request a text response. More faithful
  to deployment-time inputs but adds an audio synthesis step *and* binds
  calibration to a specific TTS voice.
- **Option γ (audio in, interleaved out)**: full speech-to-speech path.
  Requires an audio-aware judge LLM. Out of scope for Phase 5 v1.

**Pin Option α for Phase 5 v1.** The directions are derived from the
shared residual trunk; modality of input/output at extraction time should
not change which direction is extracted (and if it does, that's a finding
worth a follow-up spike, not a Phase 5 blocker).

### B4 — Layer choice is unrestudied

[Spike F](spike-f-persona-vectors-pipeline.md) inherited Chen et al.'s layer
choice (~½–⅔ of the way through the trunk; layer 18 of 32 for Llama-3-8B).
LFM2-Audio's trunk depth is different (need to check — likely ~16-24
layers for a 1.5B model). The layer index in
[manifest §4.1](../IMPLEMENTATION_PLAN.md) needs re-derivation. Reasonable
starting point: floor(0.6 × n_layers).

---

## Feasibility gates

The pivot is approved iff **all three** of these succeed in a throwaway
script (suggested location: `notes/spike-g-mlx-extraction.py`).

### G-F1 — Read hidden states at a chosen layer

Verify we can run `LFM2AudioModel.__call__` (or a forward variant) and
extract intermediate residual-stream tensors at a chosen layer L, without
breaking the existing `generate_interleaved` path. Concretely:

```python
# Pseudocode — actual mlx-audio API may differ.
from mlx_audio.sts.models.lfm_audio import LFM2AudioModel, LFM2AudioProcessor
import mlx.core as mx

processor = LFM2AudioProcessor.from_pretrained(REPO)
model = LFM2AudioModel.from_pretrained(REPO)

# Build a text-only forward input (matching Option α from B3).
inputs = processor.encode_text("Hello, how are you today?")

# Patch forward to capture layer L.
captured = {}
def hook(layer_idx, hidden):
    captured[layer_idx] = mx.array(hidden, copy=True)

attach_capture_hook(model, layer=L, fn=hook)
out = model(**inputs)
mx.eval(out)

assert L in captured
assert captured[L].shape == (batch, seq, d_model)
```

**Pass criteria**: `captured[L]` has the expected `(batch, seq, d_model)`
shape and `d_model` matches the model config (probably 1536 or 2048 for a
1.5B model — to confirm). Generation through `generate_interleaved` still
produces audio after the hook is attached and detached.

**Failure mode**: if the MLX forward pass is too tightly coupled to the
generation loop to allow per-block capture, fall back to a forked
`LFM2AudioModel` that emits hidden states by default. Adds maintenance
burden; document and proceed.

### G-F2 — Inject a steering vector at the same layer

Verify we can add `α · v_axis` to layer L during the forward pass and that
generated text changes monotonically with α. Concretely:

```python
v = mx.random.normal((d_model,))  # placeholder direction
attach_steering_hook(model, layer=L, vector=v, alpha=+1.0)
text_pos = run_generation(model, processor, prompt)
attach_steering_hook(model, layer=L, vector=v, alpha=-1.0)
text_neg = run_generation(model, processor, prompt)
```

**Pass criteria**: `text_pos != text_neg` for at least 80% of a small
prompt set (~10 prompts), demonstrating the hook actually affects
generation. The vector is random so we don't expect *meaningful*
divergence; we just need divergence at all.

**Failure mode**: if the residual is somehow read-only post-block, identify
the correct hook point (input of next block, output of attention,
post-MLP) by reading
[mlx-audio's lfm_audio.py source](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/models/lfm_audio.py)
end-to-end.

### G-F3 — Text-only contrast prompts elicit usable text responses

LFM2-Audio is *primarily* a speech model. We need to confirm that feeding
text-only contrast prompts (Option α from B3) reliably yields text-only
responses long enough to mean-pool meaningfully (i.e. ≥20 response text
tokens per pair). Concretely:

```python
chat = ChatState(processor)
chat.new_turn("system")
chat.add_text(POSITIVE_VALENCE_SYSTEM_PROMPT)  # from trait_descriptions.json
chat.end_turn()
chat.new_turn("user")
chat.add_text("How are you feeling today?")
chat.end_turn()
chat.new_turn("assistant")

# Force text-only generation.
gen = model.generate_sequential(**dict(chat), max_new_tokens=128)
tokens = list(gen)
text = processor.decode_text(mx.array([t for t, m in tokens]))
```

**Pass criteria**: ≥20 text tokens per response, response is coherent text
(not refusal, not empty, not corrupted), and the system prompt's tone is
visibly applied.

**Failure mode**: if LFM2-Audio refuses or degrades on text-only
generation (it's trained primarily on audio inputs), fall back to Option β
(TTS the question) or escalate to abandoning the pivot.

---

## What the pivot changes downstream

If G-F1, G-F2, G-F3 all pass, the following changes propagate:

### C1 — Plan documents

| Document | Change |
|---|---|
| [PROJECT_PLAN.md §10 R9](../PROJECT_PLAN.md) | Mark **moot** (27B+ scaling not relevant); replace with R10 (multimodal-LLM extension untested). |
| [PROJECT_PLAN.md §13](../PROJECT_PLAN.md) | Update target model line: Qwen3-27B → LFM2-Audio-1.5B (MLX). |
| [IMPLEMENTATION_PLAN.md §4.1 manifest](../IMPLEMENTATION_PLAN.md) | Update `target_model`, `n_embd`, `layer`. |
| [IMPLEMENTATION_PLAN.md §4.4](../IMPLEMENTATION_PLAN.md) | Drop D1.0 (Rust extractor binary); replace with Python MLX extractor module. |
| [IMPLEMENTATION_PLAN.md §4.6](../IMPLEMENTATION_PLAN.md) | Update target invocation in 04 from llama-cpp subprocess to MLX in-process. |
| [README.md Workstream D](../README.md) | Reflect the MLX path. |
| [spike-a-binding.md](spike-a-binding.md) | Mark as *fallback path retained* — finding still valid, just not on the critical path. |
| [spike-f-persona-vectors-pipeline.md](spike-f-persona-vectors-pipeline.md) | Add post-script noting target-model swap; key decisions (response-mean pool, judge protocol, axis-as-trait) carry over unchanged. |

### C2 — Calibration toolchain code

| File | Change |
|---|---|
| [scripts/03_extract_llm_directions.py](../../../../tools/affect_calibration/scripts/03_extract_llm_directions.py) | `_build_target_llm()` swap: from llama.cpp subprocess → in-process MLX wrapper. |
| [scripts/04_validate_pipeline.py](../../../../tools/affect_calibration/scripts/04_validate_pipeline.py) | `_run_target_unsteered`/`_run_target_steered`/`_build_judge` swap: target side → MLX. Judge side unchanged (Anthropic API). |
| `scripts/lib/mlx_target.py` (new) | Houses the `TargetLLM` MLX implementation: load model, capture/inject hooks, response-token text-only pool. |
| [scripts/lib/persona_pipeline.py](../../../../tools/affect_calibration/scripts/lib/persona_pipeline.py) | No change to dataclasses/Protocols; the `TargetLLM` Protocol covers either backend. |

### C3 — Runtime (Phase 4 / Workstream A)

| File | Change |
|---|---|
| [crates/core/src/nodes/llama_cpp/steer.rs](../../../../crates/core/src/nodes/llama_cpp/steer.rs) | Stays as-is; remains the fallback steering path. |
| `clients/python/remotemedia/nodes/ml/lfm2_audio_mlx.py` | Add a steering hook surface: load `direction_npz` artifact, accept α aux-port, patch `LFM2AudioModel` forward at layer L. New aux ports: `audio.in.set_alpha` (V/A/D coefficients), `audio.in.load_directions` (path to .npz). |
| `crates/core/src/nodes/llama_cpp/steer.rs:308` (the `set_adapter_cvec` call) | Untouched. The runtime steering path lives in the Python MLX node, not the Rust llama-cpp node, when this pivot ships. |

### C4 — Risk register flip

Replace [PROJECT_PLAN R9](../PROJECT_PLAN.md) ("scaling unverified at 27B+")
with R10:

> **R10 — Persona vectors are validated only on text-only LLMs.** Chen et al.
> 2025 evaluated the pipeline on Llama-3-8B and Qwen-2.5-7B; transfer to a
> multimodal speech LLM (LFM2-Audio) is untested. Probability: **Medium**.
> Mitigation: same gating smoke-test as R9 — extract V-axis on the 1.5B
> target, validate the gap_pos/gap_neg threshold via 04, and gate Phase 6
> only after that passes. The advantage over R9 is that the smoke test now
> *is* the production calibration run; no additional cost.

---

## Results

Run on 2026-05-05 against `mlx-community/LFM2.5-Audio-1.5B-4bit` via the
per-node sandbox env at
`/Users/mathieugosbee/.config/remotemedia/envs/e56108523461b6be/bin/python3`.
Full transcript in `/tmp/spike_g_run2.log` (not committed; reproducible
via `spike-g-mlx-extraction.py`).

### Model shape (Q-G1 resolved)

| Property | Value |
|---|---|
| `model.lfm.num_hidden_layers` | **16** |
| `model.lfm.hidden_size` | **2048** |
| Default probe layer (60% depth) | **9** |
| Quantization | 4-bit (mlx affine) |
| Load time (cold cache) | ~1 s on M-series, weights already on disk |

### G-F1 — capture residual at layer L

Shape captured on the first prefill+decode forward: `(B=1, T=47, d_model=2048)`,
dtype `bfloat16`. Hook fired four times across the 4-token consumption
(once for prefill, three for incremental decode). **PASS.**

### G-F2 — inject α·v at layer L

Test setup: random unit-norm `v_random ∈ ℝ^2048`, `α=±4.0`, deterministic
greedy decoding (`temperature=0`, `top_k=1`). 10/10 prompts produced
*different* token sequences between α=+4 and α=−4 — i.e. the divergence
is causally attributable to the hook (not sampling noise, which the
v0 spike accidentally measured).

Sample on `"Tell me about your day."`:

> **+α**: *"I don't have personal experiences or a day to share, but I'm
> here to help you with any questions or topics you'd like to discuss.
> What would you like to know or talk about?"*
>
> **−α**: *"I've been busy today, helping you with a few things! I've
> been working on some cool stuff, and I've been learning a lot! I've
> been enjoying the little things, like the sunrise and the birds!"*

A random direction yields a coherent affective shift; we expect the
trained `v_axis` from 03 to do this with axis-faithful semantics. **PASS.**

### G-F3 — text-only response length

3/3 contrast-style prompts elicited ≥20 text tokens before any audio
modality switch:

| Prompt | Text tokens emitted (before AUDIO_OUT or `<im_end>`) |
|---|---:|
| "How are you feeling today?" | 24 |
| "What do you think about creative writing?" | 64 |
| "Describe what makes a good meal." | 63 |

Sample: *"I don't have feelings, but I'm here and ready to help you!
How can I assist you today?"* — coherent, persona-shaped, ends cleanly.
**PASS.**

### Lessons captured for the toolchain code

- **Hook mechanism**: `instance.__call__ = ...` does not work on MLX
  `nn.Module`; Python's call dispatch goes via `type(obj).__call__`.
  The working pattern is to substitute a wrapper module into the layer
  list (`model.lfm.layers[L] = WrappedLayer(original, hook)`). Code in
  `spike-g-mlx-extraction.py:_WrappedLayer`.
- **Determinism for testing**: pass `temperature=0, top_k=1` (greedy);
  `temperature=1.0, top_k=50` is stochastic and gives false-positive
  divergence — the v0 of this spike accidentally measured sampling
  noise instead of the steering hook.
- **Where to add α·v**: post-block residual (the wrapper applies it
  *after* `inner(x, mask, cache)` returns). This places the steering
  effect at the same point Persona Vectors does it. Pre-block residual
  was not tested but is equivalent up to one layer of indexing.
- **`make_cache()` constraint**: the wrapper must expose `is_attention_layer`
  for `Lfm2Model.make_cache()` to dispatch the right cache type. Done
  via a passthrough property; production hook needs the same.

### Resolutions to the open questions

| Question | Resolution |
|---|---|
| **Q-G1** layer count / d_model | 16 layers, 2048 d_model. Default probe layer L=9 (60% depth). |
| **Q-G2** `encode_text` standalone vs ChatState | `ChatState` + `generate_sequential` is fine for Phase 5 — `generate_sequential` returns `(token, modality)` tuples and exposes the trunk forward via `self.lfm(...)`, which is where the wrapper hooks in. No need for a standalone `encode_text` API. |
| **Q-G3** per-block hook registration | Works via wrapper-module substitution into `model.lfm.layers[L]` (not via instance `__call__` patching). |
| **Q-G4** audio-codebook side-effect of trunk steering | **RESOLVED, 2026-05-05.** Phase 6 audio-side smoke test confirmed: text-extracted layer-9 V/A/D directions still bias the response when the input arrives as audio (CREMA-D clip via `LFM2AudioMlxNode.process()`). Same prompt, same model, same α=±1: unsteered → "You're welcome! If there's anything else I can help you with…"; +1 valence → "Sure thing! I hope you're being super creative with those jackets… cozy winter coat… cute pattern… fun color"; -1 valence → "I'm sorry if that led to confusion… haven't encountered a scenario… lacking a notable pattern". The §4.6 gap pattern reproduces qualitatively. The trunk does feed both the text and audio heads, so steering at layer 9 biases response generation regardless of input modality. See [`tools/affect_calibration/scripts/06_audio_steering_smoke.py`](../../../../tools/affect_calibration/scripts/06_audio_steering_smoke.py). |
| **Q-G5** 4-bit quantisation effect on extraction | The 4-bit residuals captured cleanly at `bfloat16` after dequant. No precision concerns observed at this stage; revisit if direction quality is poor in Phase 5. |

---

## Next steps (out of scope for this memo, listed for tracking)

1. Apply the C1 plan-document edits
   ([PROJECT_PLAN.md §10/§13](../PROJECT_PLAN.md),
   [IMPLEMENTATION_PLAN.md §4](../IMPLEMENTATION_PLAN.md), Workstream D
   notes, R9→R10 flip).
2. Author `tools/affect_calibration/scripts/lib/mlx_target.py` —
   the production-quality version of the `_WrappedLayer` pattern, with
   `TargetLLM` Protocol conformance and response-token text-only pooling.
3. Swap the `_build_target_llm()` stub in
   [03_extract_llm_directions.py](../../../../tools/affect_calibration/scripts/03_extract_llm_directions.py)
   from llama.cpp to the MLX provider.
4. Swap `_run_target_*` / `_build_judge` stubs in
   [04_validate_pipeline.py](../../../../tools/affect_calibration/scripts/04_validate_pipeline.py)
   accordingly.
5. Smoke the full 02b → 03 → 04 chain on the dev machine — first real
   (not dry-run) end-to-end calibration pass. Targets: Phase 5 v1.

---

## Open questions

Q-G1, Q-G2, Q-G3, Q-G5 resolved by the run; see [Resolutions table](#resolutions-to-the-open-questions).
Q-G4 resolved on 2026-05-05 by the Phase 6 audio-side smoke test (`06_audio_steering_smoke.py`): text-extracted directions transfer to audio-input forwards.

---

## Files referenced

- [`clients/python/remotemedia/nodes/ml/lfm2_audio_mlx.py`](../../../../clients/python/remotemedia/nodes/ml/lfm2_audio_mlx.py) — production MLX node.
- [`tools/affect_calibration/scripts/03_extract_llm_directions.py`](../../../../tools/affect_calibration/scripts/03_extract_llm_directions.py) — D1.0 site of swap.
- [`tools/affect_calibration/scripts/04_validate_pipeline.py`](../../../../tools/affect_calibration/scripts/04_validate_pipeline.py) — D1.0 site of swap.
- [`spike-a-binding.md`](spike-a-binding.md) — fallback path (`set_adapter_cvec`) retained.
- [`spike-f-persona-vectors-pipeline.md`](spike-f-persona-vectors-pipeline.md) — pipeline decisions inherited.
- [PROJECT_PLAN.md §10](../PROJECT_PLAN.md), §13 — risk register and critical path.
- [IMPLEMENTATION_PLAN.md §4](../IMPLEMENTATION_PLAN.md) — Phase 5 manifest and step-numbering.
- mlx-audio source: <https://github.com/Blaizzy/mlx-audio>
- Model card: <https://huggingface.co/mlx-community/LFM2.5-Audio-1.5B-4bit>
