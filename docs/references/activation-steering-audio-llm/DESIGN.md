# Audio-Driven LLM Persona Steering — Feasibility & Design

> Status: **Feasibility study, pre-implementation.** Captures the architectural
> investigation, current-vs-target gap analysis, workstream decomposition,
> research grounding, and decision log for the proposed feature: using user
> speech prosody, captured via Whisper encoder embeddings and projected
> through a low-dimensional affect bottleneck, to steer a frozen LLM's hidden
> states during inference for real-time persona modulation.
>
> Companion materials in this directory:
> - [`README.md`](README.md) — annotated bibliography
> - [`bibliography.bib`](bibliography.bib) — BibTeX
> - [`fetch.sh`](fetch.sh) — fetch arXiv PDFs into `./pdfs/`

---

## 1. TL;DR

The **vision**: a streaming audio→LLM→audio pipeline where the user's emotional
state — measured directly from their voice prosody, not just from transcribed
words — modulates how the LLM perceives the user's message and how the
assistant responds. The LLM stays **frozen** (no retraining); persona is
controlled via runtime activation steering at chosen transformer layers.

The **mechanism**: a learned linear (or near-linear) projection
`Whisper_encoder_embedding → (valence, arousal, dominance) → LLM_hidden_direction`,
applied per turn through llama.cpp's `llama_apply_adapter_cvec` API. User-side
and assistant-side vectors are scoped to prefill vs. decode phases of inference,
so the model "perceives" the user's affect in its KV cache and "speaks" with a
configurable assistant affect.

The **gap from current state**: the runtime has the steering control plane
([`LlamaCppSteerNode`](../../../crates/core/src/nodes/llama_cpp/steer.rs)) but
not the actual injection (it is currently a metadata-only stub). Whisper nodes
emit text only, not encoder embeddings. No prosody→VAD regressor exists, no
VAD→hidden-direction projection exists, no persona-as-VAD configuration
surface exists.

The **estimate**: ~1 quarter to a full demo. A is ~1 week (engineering, blocking
everything else), C is ~1 week (engineering), B is ~3 days (engineering),
D1 calibration is the dominant cost at ~2-3 weeks (offline ML training and data
wrangling), D2+E together are ~1 week (engineering).

The **research grounding**: foundational steering technique is well-established
([Panickssery et al. 2024](https://arxiv.org/abs/2312.06681),
[Turner et al. 2023](https://arxiv.org/abs/2308.10248)). Cross-modal steering
of audio-LLMs has one close precedent ([2603.14636](https://arxiv.org/abs/2603.14636)).
Whisper as a speech-emotion feature space is well-validated
([2602.06000](https://arxiv.org/abs/2602.06000)). The **specific architecture**
proposed here — audio prosody → VAD bottleneck → frozen-LLM activation steering —
appears unpublished as of April 2026.

---

## 2. The Vision

A user speaks into a microphone. The system simultaneously:

1. Transcribes the speech to text (Whisper decoder) — feeds the LLM's prompt.
2. Captures the user's prosodic affect from the encoder side of the same
   Whisper pass — projects it into a 3D valence/arousal/dominance vector.
3. Maps that VAD vector into the LLM's hidden-state geometry as a per-turn
   steering direction, applied during prefill of the user's message.
4. Applies a separate persona-derived steering direction during the assistant's
   token generation, optionally modulated by the user's VAD (mirror, dampen,
   counter-regulate).
5. Synthesizes the assistant's text via TTS (potentially affect-aware).

The user-visible result: an assistant that *hears* whether the user is calm,
agitated, sleepy, angry, excited — and adjusts not just *what* it responds with
(via the prompt) but *how* the model attends to the user's message and *how*
the assistant's voice presents.

### Why this is interesting

- **Frozen LLM**: the entire pipeline works with any GGUF model the runtime
  already supports. No retraining, no model-specific adapters, no LoRA.
- **Decoupled**: Whisper, projection, LLM, and TTS swap independently.
- **Operable**: the VAD bottleneck is human-readable. Operators tune
  `target_VAD = (0.6, 0.2, 0.3)` ("warm advisor"), not opaque 4096-D vectors.
- **Real-time**: every component already fits the streaming pipeline; the
  added cost per turn is one matmul plus one `llama_apply_adapter_cvec` call.

### What this is *not*

- Not end-to-end-trained speech-LLM (USDM, Qwen-Audio, SALMONN do that). Those
  systems are more expressive but require model-specific training.
- Not voice-cloning or affect-conditioned TTS (ParaStyleTTS does that). Those
  control the *output* speech style; we control the *LLM's response generation*.
- Not emotion classification as a system-prompt injection ("the user sounds
  angry, respond carefully"). That works but is brittle, lossy, and discrete.
  We want the affect to *condition the model's representations* directly.

---

## 3. Current State of the Codebase

### 3.1 Existing components (verified in the tree)

| Component | Path | State |
|---|---|---|
| LLM generation | [`crates/core/src/nodes/llama_cpp/generation.rs`](../../../crates/core/src/nodes/llama_cpp/generation.rs) | Working end-to-end (smoke test passes) |
| LLM activation extraction | [`crates/core/src/nodes/llama_cpp/activation.rs`](../../../crates/core/src/nodes/llama_cpp/activation.rs) | Working for **text** input only |
| LLM activation steering — control plane | [`crates/core/src/nodes/llama_cpp/steer.rs`](../../../crates/core/src/nodes/llama_cpp/steer.rs) | Vector registration, per-session coefficients, runtime updates via control messages, delta math — all functional |
| LLM activation steering — actual injection | [`crates/core/src/nodes/llama_cpp/steer.rs:308-313`](../../../crates/core/src/nodes/llama_cpp/steer.rs#L308) | **Stubbed.** Warns "metadata mode" — text passes through unmodified |
| llama.cpp `llama_apply_adapter_cvec` binding | (not present) | Not referenced anywhere in `crates/`; binding situation TBD |
| LLM smoke test | [`crates/core/examples/llama_cpp_chat_smoke.rs`](../../../crates/core/examples/llama_cpp_chat_smoke.rs) | Working; loads model on GPU, two turns, asserts no `<think>` leak, sentinel present |
| Whisper transcription | [`clients/python/remotemedia/nodes/ml/whisper_transcription.py`](../../../clients/python/remotemedia/nodes/ml/whisper_transcription.py) | Working; emits text only |
| Whisper STT (alternate) | [`clients/python/remotemedia/nodes/ml/whisper_stt.py`](../../../clients/python/remotemedia/nodes/ml/whisper_stt.py) | Working; emits text only |
| Audio→prosody affect estimation | (not present) | — |
| VAD-space ↔ LLM-hidden-space projection | (not present) | — |
| Persona-as-VAD config surface | (not present) | — |

### 3.2 Steering control plane that already works

The steering node already supports the runtime controls a real-time
persona-swap system needs:

- `update_coefficients` accepts per-session coefficient updates and clamps to
  `max_coefficient` (`steer.rs:147-158`).
- `process_control_message` accepts mid-stream `{"coefficients": {...}}`
  payloads (`steer.rs:471-504`), enabling persona changes between turns.
- `register_vector` accepts `RuntimeData::Tensor` from upstream nodes with
  `metadata.emotion`/`metadata.label` (`steer.rs:126-145`).
- `compute_steering_delta` correctly computes `Σ(coef_i × layer_norm × vec_i)`
  (`steer.rs:199-230`) — verified by `test_steering_delta`.

**What's missing is only the last hop**: handing this delta to llama.cpp's
control-vector API before each `llama_decode`. Everything upstream is built.

### 3.3 Runtime architecture properties relevant to this feature

From [`CLAUDE.md`](../../../CLAUDE.md), confirmed by reading the code:

- **Session router** (`runtime/src/grpc_service/session_router.rs`) routes
  data between nodes for the duration of a streaming session. Each node runs
  in its own tokio task; outputs flow back through a shared channel.
- **Multiprocess Python isolation** is supported via iceoryx2 IPC. Trained
  ML models (e.g., the prosody→VAD regressor) can run in Python without
  blocking the Rust audio path.
- **Pipeline graph executor (spec 021)** supports topological ordering,
  fan-out, and fan-in. The proposed multi-output Whisper node (text +
  embedding) and the dual-branch steering pipeline are within the executor's
  current capabilities.
- **Capability resolution (spec 023)** enforces format compatibility between
  nodes. The new tensor-shape contract (Whisper embedding `[d_whisper]`,
  VAD `[3]`, LLM steering vector `[d_llm]`) needs to be expressed as
  declared capabilities.

---

## 4. Target Architecture

### 4.1 Data flow

```
                ┌────────── text branch ──────────┐
                │                                 │
mic ──→ whisper_node ──→ tokenize ──→ llm_node ──→ tts ──→ speaker
                │                          ▲
                └─ encoder_embedding ──┐   │
                                       ▼   │
                              affect_projection_node
                              (Whisper → VAD)
                                       │
                                       ▼
                              persona_policy_node
                              (user VAD + persona target → assistant target)
                                       │
                              ┌────────┴────────┐
                              ▼                 ▼
                  affect_to_steering_node   affect_to_steering_node
                  (VAD_user → LLM dir,      (VAD_target → LLM dir,
                   role=user)                role=assistant)
                              │                 │
                              └────────┬────────┘
                                       ▼
                              llama_cpp_steer_node
                              (registers vectors,
                               applies per phase)
                                       │
                                       └─→ feeds llm_node's
                                           cvec buffer per token
```

### 4.2 New nodes (proposed)

| Node | Language | Responsibility |
|---|---|---|
| `WhisperEmbeddingExtractorNode` | Python (extends existing) | Emit Whisper encoder hidden states alongside text. Tap configurable layer index. Optionally pre-pool. |
| `MeanPoolNode` (or `AttentionPoolNode`) | Rust | Collapse `[T_frames × d_whisper]` → `[d_whisper]`. Trivial. Attention-pool variant carries learned query weights. |
| `AffectProjectionNode` | Python (multiprocess) | Loads trained `g: ℝ^d_whisper → ℝ³`. Optional EMA smoothing. Tags output with `role=user`. |
| `PersonaPolicyNode` | Rust | Pure config. Computes `assistant_target_VAD` from `(user_VAD, persona_VAD, mirror_strength, clamps)`. |
| `AffectToSteeringNode` | Rust | Pure linalg. Loads `(3 × d_llm × n_layers)` direction tensor. Emits `[d_llm × n_layers]` steering vector with role+layer metadata. |

### 4.3 Modified nodes

| Node | Change |
|---|---|
| `LlamaCppActivationNode` | Add `role` field to emitted tensor metadata. |
| `LlamaCppSteerNode` config | Add `role: SteerRole = User \| Assistant \| Both` to `LlamaCppSteerVector`. |
| `LlamaCppSteerNode` inference path | Connect computed delta to llama.cpp's `llama_apply_adapter_cvec` (currently stubbed). Apply user-role cvec during user-token decode; swap to assistant-role cvec before first sampled assistant token. |
| `inference.rs::run_generation_with_ctx` | Hold a `Option<ControlVectorState>` on the persistent context. Provide phase-aware cvec swap API. |

### 4.4 New manifest surface

```yaml
nodes:
  - id: whisper
    node_type: WhisperTranscriptionNode
    config:
      model: large-v3
      emit_encoder_embedding: true
      encoder_layer: -2     # intermediate layers > final for SER (2602.06000)

  - id: pool
    node_type: MeanPoolNode

  - id: affect
    node_type: AffectProjectionNode
    config:
      regressor_path: models/whisper_to_vad.onnx
      ema_alpha: 0.7

  - id: persona
    node_type: PersonaPolicyNode
    config:
      target_VAD: [0.6, 0.2, 0.3]   # warm advisor
      mirror_strength: 0.3
      max_arousal: 0.7

  - id: user_steer
    node_type: AffectToSteeringNode
    config:
      directions: models/llm_vad_directions.npz
      layer: 18
      role: user

  - id: assistant_steer
    node_type: AffectToSteeringNode
    config:
      directions: models/llm_vad_directions.npz
      layer: 18
      role: assistant

  - id: llm
    node_type: LlamaCppSteerNode
    config:
      model_path: models/qwen3-27b.gguf
      layer: 18
      max_coefficient: 2.0
      generation: { ... }

connections:
  - { from: mic,             to: whisper }
  - { from: whisper.text,    to: llm }                     # text branch
  - { from: whisper.encoder, to: pool }                    # prosody branch
  - { from: pool,            to: affect }
  - { from: affect,          to: persona }
  - { from: affect,          to: user_steer }              # user-side steer
  - { from: persona,         to: assistant_steer }         # assistant-side steer
  - { from: user_steer,      to: llm }
  - { from: assistant_steer, to: llm }
  - { from: llm,             to: tts }
```

---

## 5. Core Technical Concepts

### 5.1 Activation steering (control vectors)

A learned direction `v ∈ ℝ^d_model` added to the residual stream at chosen
transformer layer(s) during forward passes. Implemented in llama.cpp via
`llama_apply_adapter_cvec(ctx, data, len, n_embd, il_start, il_end)`. Equivalent
to ActAdd / Contrastive Activation Addition.

Key properties:

- **Linear**: scales by a coefficient `α ∈ ℝ`. Sign flips reverse the effect.
- **Composable**: multiple vectors sum linearly.
- **Layer-localized**: applied only at specified layer range. Middle layers
  (~40-60% depth) typically carry the most semantic/persona-relevant signal.
- **Cheap at inference**: no weight changes; one vector add per forward pass.

### 5.2 User-side vs assistant-side scoping

Standard activation steering applies during all decode steps. The proposed
extension splits by phase:

- **Prefill phase** (decoding the user's input tokens to populate KV cache):
  apply only `User`/`Both` vectors. The KV-cache entries for the user's
  tokens are computed with the user-emotion direction baked in.
- **Decode phase** (generating assistant tokens): apply only `Assistant`/`Both`
  vectors.
- **Transition**: between the last user-token forward pass and the first
  sampled assistant token, call `llama_apply_adapter_cvec` again with the
  assistant-role buffer.

The user-side scoping is the more interesting half: it doesn't change *what*
the model says, it changes *how it represents the user* internally. When the
model later attends back to the user's KV entries during generation, it
"remembers" the user as having spoken with the conditioned affect — even
though the literal tokens are identical.

Calibration coefficients differ between sides: input-side typically needs
~0.3-0.5× the magnitude of output-side, since over-steering the KV cache
produces hallucinated user intent.

### 5.3 Cross-modal projection

We need a function `f: ℝ^d_whisper → ℝ^d_llm` such that `f(whisper_emb)` lands
near the same direction the LLM's text-derived emotion contrast vectors point
in. Three increasingly capable approaches:

1. **Linear probe via paired contrast data**. ~50 lines of numpy. Solve
   least-squares `W` such that `W · Δw ≈ Δh` over emotion contrast pairs.
2. **Contrastive alignment**. Small MLP trained to pull
   `f(whisper(audio_i))` toward `h_llm(text_i, emotion_i)` and push from
   mismatched emotions. Generalizes to continuous affect.
3. **End-to-end fine-tuning** with backprop through the frozen LLM into `f`.
   Cleanest signal, requires labeled (audio, ideal-response) pairs.

(1) is the recommended starting point.

The LLaVA/Flamingo literature establishes precedent that a small projector
between a frozen sensory encoder and a frozen LLM is sufficient for cross-modal
conditioning — though those systems project into the LLM's *input* space, not
its *steering subspace*. The LLM-steering version of this is the architectural
novelty of the proposed work.

### 5.4 VAD dimensional bottleneck

Replacing per-emotion-label vector lookup with a continuous 3D
valence/arousal/dominance space, then mapping VAD → LLM hidden direction:

```
g: ℝ^d_whisper → ℝ³           (Whisper → VAD coordinates)
h: ℝ³           → ℝ^d_llm     (VAD → LLM steering direction)
```

The bottleneck is the engineering win:

- **Interpretable**: `(v, a, d)` triples are human-readable.
- **Controllable**: clip, smooth, or bias per-axis programmatically.
- **Composable**: persona = a target VAD point; mirroring = `lerp(persona, user, α)`.
- **Cheap**: smoothing 3 floats with EMA dampens jitter; smoothing the
  high-dim hidden vector directly is harder to reason about.

`h: ℝ³ → ℝ^d_llm` factors as three direction vectors `dir_v, dir_a, dir_d`
in LLM hidden space, with `h(v, a, d) = v·dir_v + a·dir_a + d·dir_d`. If
linearity proves insufficient in practice, replace `h` with a small MLP —
the upstream interface is unchanged.

#### Why VAD specifically

The valence/arousal/dominance model is the standard psychological account of
affect (Russell 1980 circumplex; Mehrabian 1996 PAD). NRC-VAD provides a
20k-word lexicon for text→VAD lookup, which lets us bootstrap the LLM-side
direction extraction without manual annotation. MSP-Podcast and IEMOCAP
provide audio→VAD ground-truth training data. The 3D framing is mature and
well-resourced.

If 3 dimensions prove insufficient (likely for sarcasm, hesitation, certainty,
formality), raise `k` to 6-8 with learned dimensions on top of VAD: keep first
3 as VAD for interpretability, let the rest float. The same `h` linear-map
plumbing extends trivially.

### 5.5 The dual-vector composition

At inference, the steering vector applied during assistant generation is:

```
v_assistant = h(target_VAD)
v_user      = h(user_VAD)

# Applied per phase:
prefill_cvec = α_user · v_user
decode_cvec  = α_asst · v_assistant
```

`target_VAD` is computed by `PersonaPolicyNode`:

```
target_VAD = clip(
    persona_VAD + mirror_strength · (user_VAD - persona_VAD),
    [-1, 1]^3
)
```

This expresses: "anchor on the persona, but drift toward the user's affect by
`mirror_strength`." Setting `mirror_strength=0` is pure persona-anchored.
Setting `mirror_strength=1` is pure user-mirroring. Setting it negative
counter-regulates (calm an agitated user). All knobs are 3-D and human-readable.

---

## 6. Workstream Decomposition (the delta)

Five workstreams, with explicit dependencies. **A blocks everything; D1 is the
single biggest time sink and is offline ML work** — staff differently from
the Rust pipeline work.

```
A (steering injection) ─┬─→ B (role separation) ─┐
                        │                          ├─→ E (persona policy)
C (whisper encoder)  ───┴─→ D1 (calibration) ───→ D2 (runtime nodes) ──┘
```

### Workstream A — Make existing steering actually steer

**Goal**: connect `LlamaCppSteerNode`'s already-computed delta into
`llama_apply_adapter_cvec` in the inference path.

1. Verify whether the binding (`llama-cpp-2` or equivalent in
   [`inference.rs`](../../../crates/core/src/nodes/llama_cpp/inference.rs))
   already wraps `llama_apply_adapter_cvec`. If not: binding bump, fork, or
   `unsafe extern "C"` shim. **This determines whether A is 1 week or 3 weeks.**
2. Extend the persistent generation context (held across turns by
   `run_generation_with_ctx`) with `Option<ControlVectorState { data, n_embd, il_start, il_end }>`.
3. Wire `LlamaCppSteerNode::generate_steered`
   ([`steer.rs:233-263`](../../../crates/core/src/nodes/llama_cpp/steer.rs#L233))
   to call into the generation context with the freshly-computed delta
   *before* `llama_decode`.
4. Add a helper `pad_to_layer_buffer(layer_idx, vec) → padded_buffer` for the
   `n_layers × n_embd` flat format `llama_apply_adapter_cvec` expects.
5. Smoke test: extend
   [`llama_cpp_chat_smoke.rs`](../../../crates/core/examples/llama_cpp_chat_smoke.rs)
   into `llama_cpp_steer_smoke.rs`. Generate the same prompt with `coef=0` and
   `coef=2.0` of a contrast-pair vector; assert outputs differ.
6. Flip `applied: false` → `applied: true` in metadata; remove the warning at
   [`steer.rs:308-313`](../../../crates/core/src/nodes/llama_cpp/steer.rs#L308).

**Deliverable**: `LlamaCppActivationNode → LlamaCppSteerNode → text out` pipeline
where coefficient changes visibly change tone. Independent of audio entirely.

**Estimate**: ~1 week.

### Workstream B — User/assistant role separation

**Goal**: scope steering vectors to prefill vs. decode phases.

1. Add `role: SteerRole` to `LlamaCppSteerVector`
   ([`steer.rs:19`](../../../crates/core/src/nodes/llama_cpp/steer.rs#L19))
   and to the metadata `LlamaCppActivationNode` stamps on its tensors
   ([`activation.rs:158-169`](../../../crates/core/src/nodes/llama_cpp/activation.rs#L158)).
2. In `run_generation_with_ctx`, split the decode into two phases: apply
   user-role cvec, decode user tokens, swap to assistant-role cvec, decode
   assistant tokens.
3. Smoke test: same input text, three runs (user-only / assistant-only / both).
   Verify assistant-only produces emotional output but neutral perception of
   user; user-only produces emotional perception but neutral voice.

**Deliverable**: role-tagged steering working on text-derived vectors.

**Estimate**: ~3 days after A lands.

### Workstream C — Whisper encoder embedding exposure

**Goal**: make Whisper a multi-output node emitting both text and encoder
hidden states.

1. Decide where Whisper runs:
   - **Easiest path**: extend
     [`whisper_transcription.py`](../../../clients/python/remotemedia/nodes/ml/whisper_transcription.py)
     to also emit encoder output. `faster-whisper` exposes it; raw `whisper`
     exposes via `model.encoder(mel)`.
   - **Best perf**: port to `whisper.cpp` Rust binding and call
     `whisper_full_get_encoder_state`.
2. Make the node multi-output: `RuntimeData::Text` + `RuntimeData::Tensor[T_frames × d_whisper]`.
   Per spec 021, the executor supports fan-out.
3. Add a `MeanPoolNode` (or `AttentionPoolNode` later) — pure Rust, ~30 lines.
4. Smoke test: 5-second audio clip → both outputs land, embedding tensor
   has the expected shape.

**Deliverable**: working `audio → whisper → (text + embedding)` fan-out.

**Estimate**: ~3-5 days, dominated by Whisper binding work.

### Workstream D1 — Offline calibration (the ML cost center)

**Goal**: train the projection `Whisper → VAD → LLM-hidden-direction`. Run once
per `(LLM, layer, Whisper)` tuple; offline; Python; whatever ML framework.

1. **Dataset acquisition**: MSP-Podcast (best for conversational), IEMOCAP,
   ESD. License-check each.
2. **Train `g: ℝ^d_whisper → ℝ³`** on (audio, VAD-label) pairs. MLP
   `(d_whisper → 64 → 3)` with MSE. Validate per-axis RMSE on held-out data.
   Export to ONNX.
3. **Extract LLM directions `h: ℝ³ → ℝ^d_llm`**:
   - Curate emotion contrast sentences; label each with NRC-VAD lexicon-averaged
     sentence-level VAD.
   - Run sentences through `LlamaCppActivationNode` at the chosen steering layer.
   - Solve `min ‖H − V·D‖²` where `H ∈ ℝ^{N × d_llm}` are activations, `V ∈ ℝ^{N × 3}` are VAD vectors, `D ∈ ℝ^{3 × d_llm}` are direction rows.
   - Save `D` keyed by `(model_path_hash, layer)`.
4. **Validation**: generate text with `h(VAD_target)` injected at varying
   coefficients. Either rubric-score outputs by hand or use a judge LLM. Tune
   coefficient scale.

**Deliverable**: trained `g.onnx` and `D.npz` artifacts; per-axis RMSE numbers;
qualitative samples showing affect modulation.

**Estimate**: ~2-3 weeks. Most time is dataset wrangling and validation,
not modeling.

### Workstream D2 — Runtime nodes for the projection

**Goal**: make D1's artifacts run in the streaming pipeline.

1. `AffectProjectionNode` (Python multiprocess): loads `g.onnx`, accepts
   `RuntimeData::Tensor[d_whisper]`, emits `RuntimeData::Tensor[3]` with
   `metadata = {role: "user", axes: ["v", "a", "d"]}`.
2. `AffectToSteeringNode` (Rust, pure linalg): loads `D.npz`, accepts
   `RuntimeData::Tensor[3]`, emits `RuntimeData::Tensor[d_llm]` with
   `metadata = {role, layer}`.
3. Optional: EMA smoothing on the 3-D tensor inside `AffectProjectionNode`.

**Deliverable**: end-to-end pipeline. Speaking angrily into the mic visibly
shifts LLM output.

**Estimate**: ~1 week.

### Workstream E — Persona config & control surface

**Goal**: operator-facing knobs.

1. Define `PersonaConfig { target_VAD: [f32; 3], mirror_strength: f32, max_arousal: f32, ... }`.
2. `PersonaPolicyNode`: accepts user VAD, emits assistant target VAD via
   `lerp(persona_VAD, user_VAD, mirror_strength)` with axis clamps.
3. Per-turn structured logging:
   `{user_VAD, target_VAD, applied_coef_user, applied_coef_assistant, delta_norm_user, delta_norm_assistant}`.

**Deliverable**: tunable, observable persona.

**Estimate**: ~3 days.

---

## 7. Critical Path & Calendar

| Week | Workstream | Focus |
|------|-----------|-------|
| 1 | A spike | Determine binding situation for `llama_apply_adapter_cvec`. |
| 1-2 | A | Land steering injection. Smoke test passing. |
| 2 | B + C in parallel | Role separation; Whisper embedding extraction. |
| 2-4 | D1 | Dataset acquisition, train `g`, extract `D`, validate. |
| 4 | D2 | Runtime nodes. |
| 4-5 | E | Persona policy + logging. |
| 5 | End-to-end demo | Same audio, two persona configs, audibly different responses. |

This is **best-case staffing**: one Rust engineer on A/B/C/D2/E, one ML
engineer on D1, both starting Week 1. Sequential single-developer staffing
is ~8-10 weeks.

---

## 8. Risks & Mitigations

Risk-ordered by impact on schedule and feasibility.

### R1: `llama_apply_adapter_cvec` not exposed by Rust binding
**Probability**: Medium. **Impact**: High (A goes from 1 week → 3 weeks).
**Mitigation**: Day-1 spike. If unexposed, options in order of preference:
upstream PR to the binding crate; vendored fork; `unsafe extern "C"` shim
through `llama-cpp-sys`. The shim is ~30 lines but harder to maintain.

### R2: VAD calibration overfits to acted speech
**Probability**: High if RAVDESS-only; Medium with MSP-Podcast.
**Impact**: Medium (real users underperform vs. text-derived vectors).
**Mitigation**: MSP-Podcast as primary dataset (spontaneous conversational
speech). Cross-validate on out-of-distribution speakers.

### R3: Linear `h: ℝ³ → ℝ^d_llm` too crude
**Probability**: Medium. High-arousal-positive (excited) and
high-arousal-negative (angry) may need genuinely non-linear handling.
**Impact**: Medium (replace `h` with small MLP, retrain — same node interface).
**Mitigation**: Test linear first; characterize where it fails by ablating
diagonal vs. interaction effects; upgrade only if needed.

### R4: Whisper encoder mean-pool discards salient frames
**Probability**: High for long utterances with one emotional word.
**Impact**: Medium (mean-pool reads as neutral; signal is there but smoothed
out).
**Mitigation**: Attention-weighted pool with a learned query for "emotionally
salient frames." Add as a second pooling option after mean-pool baseline.
Literature: [2602.06000](https://arxiv.org/abs/2602.06000) explicitly explores
this and finds attention pooling outperforms.

### R5: Speaker identity leakage in projection
**Probability**: High. Whisper embeddings encode speaker, not just affect.
**Impact**: Medium (projection learns "this is Bob" instead of "this is angry
Bob").
**Mitigation**: (a) Hold speaker constant in contrast pairs during D1
training; (b) per-speaker mean-subtraction at inference (running mean of recent
user VAD, subtract before injection); (c) adversarial speaker-classifier head
during D1 training.

### R6: KV-cache incoherence on mid-utterance persona flip
**Probability**: Low if persona changes only at turn boundaries.
**Impact**: Medium (sharp flips mid-token can read as incoherent).
**Mitigation**: Restrict persona changes to turn boundaries. If mid-turn
modulation is desired, ramp coefficients gradually rather than step-change.

### R7: The "Nudging Hidden States" precedent already covers our novelty claim
**Probability**: Low (different direction of cross-modal transfer; they apply
text-derived steering to audio-LLM inputs, we propose audio-derived steering
of text-LLM outputs). **Impact**: Low (engineering value unchanged; only
publication framing affected).
**Mitigation**: Read [2603.14636](https://arxiv.org/abs/2603.14636) carefully
before any external framing. Position relative to it explicitly.

### R8: License/redistribution constraints on training data
**Probability**: Medium (MSP-Podcast and IEMOCAP have specific terms).
**Impact**: Low for internal use, Medium if commercializing.
**Mitigation**: Audit licenses pre-acquisition. ESD is more permissively
licensed if commercial use is in scope.

---

## 9. Research Grounding

Full annotated bibliography in [`README.md`](README.md). BibTeX in
[`bibliography.bib`](bibliography.bib). Run [`./fetch.sh`](fetch.sh) to
download all 17 PDFs into `./pdfs/` (gitignored).

The four-paper priority reading list:

1. **Panickssery et al. 2024 — CAA on Llama 2** ([2312.06681](https://arxiv.org/abs/2312.06681)).
   The technique your `LlamaCppSteerNode` already implements; concrete Llama
   demonstration with behavioral benchmarks.
2. **Nudging Hidden States 2026 — Cross-modal steering of audio-LLMs** ([2603.14636](https://arxiv.org/abs/2603.14636)).
   Closest published precedent for the cross-modal-steering architecture.
   Read carefully before any novelty claim.
3. **Whisper SER with Attentive Pooling 2026** ([2602.06000](https://arxiv.org/abs/2602.06000)).
   Validates Whisper-encoder-based emotion features. Notes intermediate
   layers > final for SER — drives `WhisperEmbeddingExtractorNode`'s layer
   parameter.
4. **Speech LLMs Survey 2024** ([2410.18908](https://arxiv.org/abs/2410.18908)).
   Field landscape; positions this work relative to input-layer-conditioning
   speech-LLMs (USDM, PALLM, ParalinGPT).

### What the literature establishes

- Activation steering on frozen LLMs is well-validated (Workstream A).
- Multimodal activation steering is an emerging subfield (cross-modal extension
  of A — VLM and audio-LLM precedents exist).
- Whisper encoder embeddings are a strong feature space for speech emotion
  recognition (Workstream C+D1).
- Linear projections between frozen sensory encoders and frozen LLMs work
  (LLaVA, Flamingo lineage) — supports Workstream D1's `f` projection
  by analogy, though those project to LLM *input* tokens not steering vectors.

### What the literature does *not* cover

No paper found that does specifically: project audio-derived prosodic
embeddings through a low-dimensional VAD bottleneck into the steering subspace
of a frozen LLM, with user/assistant role separation. The closest precedent
([2603.14636](https://arxiv.org/abs/2603.14636)) does *cross-modal* steering
but in the inverse direction (text vectors steering audio-LLM behavior). The
specific architecture proposed here appears unpublished as of April 2026.

This is mild novelty — useful if writing this up; not the bar for shipping
internally.

---

## 10. Open Questions

These remain unresolved at the time of this writing and should be revisited
as workstreams complete.

### Q1: Which Whisper encoder layer?
[2602.06000](https://arxiv.org/abs/2602.06000) reports intermediate layers
beat final layers for SER, but the optimum varies by model size. Treat
`encoder_layer` as a hyperparameter; sweep during D1.

### Q2: Which LLM steering layer?
The activation-steering literature broadly converges on middle layers
(40-60% depth) for behavioral steering. But the steering layer must match
between extraction (D1's contrast-sentence step) and injection (Workstream A).
The choice is fixed at calibration time and frozen at inference. Worth
sweeping during D1 across 3-5 candidate layers.

### Q3: How many VAD axes?
Start with k=3 (canonical VAD). If sarcasm, hesitation, certainty, formality
prove important, raise to k=6-8 with learned dimensions on top. The decision
gates on D1 validation results.

### Q4: Per-speaker normalization at inference?
Speaker normalization (subtract running-mean VAD per speaker) costs almost
nothing and likely reduces R5 (identity leakage). Worth implementing in
`AffectProjectionNode` from day 1 with a session-scoped EMA.

### Q5: Is end-to-end fine-tuning of `f` ever worth it?
D1's linear/MLP probe is the cheap baseline. If ablations show the projection
is the bottleneck (vs. the LLM's capacity to be steered), end-to-end
backprop through frozen LLM into `f` becomes worth it. Punt this question
until baseline performance is measured.

### Q6: TTS affect-matching?
The pipeline ends at TTS. If the LLM's output is steered toward
`target_VAD = (warm, calm, mild_dominance)` but the TTS speaks it flatly, the
audible result understates the steering. Workstream beyond E: feed
`target_VAD` to an affect-aware TTS (ParaStyleTTS-like). Out of scope for
the minimum demo; obvious follow-up.

### Q7: How does this interact with the existing `EmotionExtractorNode`?
[`activation.rs`](../../../crates/core/src/nodes/llama_cpp/activation.rs)
schema mentions compatibility with `EmotionExtractorNode`. Read that node
before D1 starts to avoid duplicating work or producing incompatible artifacts.

### Q8: Does `whisper.cpp`'s Rust binding expose encoder hidden states?
Open question for Workstream C. If yes: cleaner Rust path. If no: stay in
Python via `faster-whisper`. Day-1 spike for C, parallel to A's binding spike.

---

## 11. Decision Log

Decisions made during this feasibility investigation, with reasoning.

| Decision | Rationale | Alternatives considered |
|---|---|---|
| Steering at *layer*, not at *input tokens* | Frozen LLM; no retraining cost; decoupled from any model-specific tokenizer/audio-encoder pairing. | Input-layer conditioning (USDM/PALLM/ParalinGPT). Rejected: requires per-LLM training, fragile, contradicts the "any GGUF model" runtime goal. |
| User-side and assistant-side as separate vectors | Lets us bias LLM perception of user (input-side) independently from assistant voice (output-side). Different psychological mechanism. | Single combined vector. Rejected: conflates "model thinks user is angry" with "model speaks angrily" — these should be independently controllable. |
| VAD as the bottleneck dimension | 3D, interpretable, well-resourced (NRC-VAD lexicon, MSP-Podcast labels). Operator knobs are human-readable. | 1D scalar (too lossy); discrete labels (ad hoc blends, can only express trained categories); learned `k=8` latent (more expressive but uninterpretable). Recommend: VAD now, raise k later if needed. |
| Linear `h: ℝ³ → ℝ^d_llm` first | Simplest possible thing that could work. Cheap to fit, cheap to debug, sets a baseline. | MLP `h`. Defer until linear baseline is measured. |
| Calibration is offline, runtime is in-pipeline | Two-stage architecture isolates the ML cost. Trained artifacts (`g.onnx`, `D.npz`) ship as files; runtime is a thin matmul chain. | Online calibration. Rejected: training during a streaming session is infeasible. |
| MSP-Podcast as primary D1 dataset | Spontaneous conversational speech matches deployment distribution. RAVDESS-style acted speech overfits. | IEMOCAP (acted, large), RAVDESS (acted, small). Use as supplementary. |
| `LlamaCppSteerNode` accepts `Tensor` from upstream | Already implemented; reusable as the plumbing for VAD-derived vectors with no node changes. | Custom node API. Unnecessary; existing tensor+metadata contract suffices. |
| Treat the references repo as public | Conservative default; metadata + fetch script committed, PDFs gitignored. Can flip later via one-line `.gitignore` edit. | Commit PDFs. Rejected for now per redistribution-license caution. |
| 17-paper bibliography curated, not exhaustive | Covers the four argumentative claims (steering works, cross-modal steering exists, Whisper carries emotion, paralinguistic speech-LLMs exist) with priority reading marked. | Exhaustive arXiv crawl. Premature; revisit when writing up. |

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Activation steering** | Adding a learned direction `v` to the residual stream at chosen transformer layers during forward passes. Synonyms: control vector, ActAdd, CAA. |
| **Control vector** | The vector `v` itself. In llama.cpp: passed via `llama_apply_adapter_cvec`. |
| **CAA** | Contrastive Activation Addition. Vectors derived from the difference between activations on positive vs. negative example pairs. |
| **VAD** | Valence/Arousal/Dominance — the canonical 3D continuous affect space (Russell 1980, Mehrabian 1996). |
| **PAD** | Pleasure/Arousal/Dominance — the same model under a different name. Use VAD throughout this work. |
| **Prefill** | The forward passes that populate the KV cache from the user's prompt tokens. No tokens sampled. |
| **Decode** | The forward passes that generate assistant tokens, one per pass. |
| **KV cache** | Cached key/value attention states from earlier tokens, reused on subsequent forward passes. |
| **Hidden state / residual stream** | The token-position-indexed activation vectors at each transformer layer. |
| **Whisper encoder** | The audio→hidden-state half of OpenAI's Whisper. The decoder converts these to text; the encoder output also carries prosodic information. |
| **MSP-Podcast** | Spontaneous conversational speech dataset with VAD annotations. Recommended for D1. |
| **NRC-VAD** | A 20k-word English lexicon mapping each word to (v, a, d) coordinates. |
| **LLaVA / Flamingo lineage** | Multimodal LLMs joining a frozen vision encoder to a frozen LLM via a small trained projector. Architectural precedent for our `f`. |
| **Persona** | A configured `target_VAD` plus `mirror_strength` and clamps. The operator-facing surface. |
