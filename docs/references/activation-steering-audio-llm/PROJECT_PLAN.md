# Project Plan — Real-Time Affective Agent with Multimodal Appraisal and Vector Steering

> **Status**: Draft v1, supersedes the architecture in [`DESIGN.md`](DESIGN.md)
> and [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md). Those documents
> remain as deeper references for steering injection and calibration
> specifics; this plan is the new top-level architecture.
>
> **Scope**: a real-time audio/text/task LLM agent that maintains a
> continuous affect state derived from cognitive appraisal, and uses that
> state to influence behavior through four parallel steering channels of
> increasing cost and capability.
>
> **Core insight**: emotion is a *causal process*, not a measurement. The
> system that wins is the one that interprets multimodal events through an
> appraisal layer, evolves state with inertia and history, and routes the
> result through the cheapest steering channel that suffices.

---

## 1. TL;DR

Build a layered affective-agent runtime. Events from audio (prosody, content),
text (transcription, sentiment), and task signals (success, failure, blocked
goals) are interpreted by a rule-based **appraisal engine** along
psychological dimensions (goal congruence, agency, control, certainty, norm
violation, threat, reward). Appraisals drive an **affect dynamics engine** —
time-varying valence/arousal/dominance plus discrete emotion channels with
inertia and decay. A **regulation policy** maps internal state to expressed
behavior. A **steering router** distributes the regulated state across four
channels: decoding controls, prompt augmentation, prefill-phase activation
steering (how the LLM perceives the user), and decode-phase activation
steering (how the LLM speaks).

Build the simulator first. Ship steering channels in increasing cost order.
Treat calibration as a research deliverable, not an engineering one.

```
Multimodal events ─→ Appraisal ─→ Dynamics ─→ Regulation ─→ Steering Router
                                                                  │
                                       ┌──────────┬───────────────┼───────────────┐
                                       ▼          ▼               ▼               ▼
                                   Decoding    Prompt          Prefill        Decode
                                   control     aug             cvec           cvec
                                                                  │               │
                                                                  └──→ LLM ←──────┘
                                                                       │
                                                                      TTS
```

The cheapest channels (decoding, prompt) work on day one, on any LLM,
without model access. The expensive channels (activation steering) require
calibration but offer finer control over how the model represents the user
and shapes its voice.

---

## 2. Why This Architecture

### 2.1 The mistake the original plan made

The first design treated this as a measurement-and-projection problem:

```
audio → Whisper encoder → mean-pool → VAD → linear projection → steering vector → LLM
```

This skips the layer that determines whether a steering vector is even the
*right response*. A user shouting in delight and a user shouting in fury
have similar prosody but should drive opposite assistant behavior. The
acoustic signal underdetermines the appropriate response. A measurement
pipeline cannot distinguish these cases.

### 2.2 What appraisal adds

Appraisal interprets events relative to context. The same prosodic spike
becomes:

- High arousal + negative goal congruence + other-agency + norm violation
  → anger appraisal
- High arousal + positive goal congruence + reward signal + social safety
  → joy appraisal

These produce categorically different downstream regulation and steering
decisions. Without the appraisal layer, the system can only react to
*surface affect*; with it, the system reacts to *interpreted affect*.

### 2.3 What time-varying state adds

Per-turn affect is fragile. A user being curt for three turns in a row reads
as three independent "low warmth" turns. A stateful system reads it as
"user has been cold for a while; this isn't a one-off" and lets the
regulation layer choose differently than for a single flash of curtness.

State continuity also gives operators the observability they need: every
expressed behavior traces back through `regulation ← state ← appraisal ←
event`, and every state change is logged. Without state continuity, the
system is opaque: "why did the model speak that way?" has no answer
beyond "the last input made it."

### 2.4 What internal/expressed separation adds

Three product-relevant levers, controllable independently:

- *Internal*: what the system tracks about the interaction (diagnostics,
  memory, downstream policy decisions).
- *Expressed*: what the assistant's voice and word choice convey.
- *Policy*: the rules that govern what crosses from internal to expressed
  (safety, professionalism, naturalness).

Without this split, "model registers user frustration" and "model speaks
with frustrated tone" collapse into a single knob. They shouldn't.

### 2.5 What the four steering channels add

The original plan committed to activation steering as the steering
mechanism. This plan treats it as one of four:

| Channel | Cost | Works on | Best for |
|---|---|---|---|
| **Decoding control** | < 1 ms | Any LLM, including APIs | Coarse arousal-driven knobs (length, temperature) |
| **Prompt augmentation** | tokens, ~ms | Any LLM | Inspectable affect-state summaries; black-box-friendly |
| **Prefill cvec** | ~5-10 ms | Frozen open-weight | How the model *perceives* the user |
| **Decode cvec** | ~5-10 ms | Frozen open-weight | How the assistant's voice *expresses* affect |

Cheap channels work on day one without calibration; expensive channels
come online once D1 calibration completes. Operators choose channels per
deployment.

---

## 3. Conceptual Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PERCEPTION LAYER                                                        │
│  Audio (waveform), Text (transcript), Task signals (success/failure)     │
└──────────┬───────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  EVENT EXTRACTION LAYER                                                  │
│  - Whisper transcription → text events (content)                         │
│  - Whisper encoder + VAD regressor → prosody events (acoustic affect)    │
│  - Task layer → goal/tool/error events                                   │
│  - Emit: PerceivedEvent { kind, timestamp, modality, confidence, ... }   │
└──────────┬───────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  APPRAISAL LAYER                                                         │
│  Maps each event to dimensional appraisal scores in agent context:       │
│    novelty, goalRelevance, goalCongruence,                               │
│    agencySelf, agencyOther, agencySituation,                             │
│    control, certainty, normViolation,                                    │
│    lossSignal, threatSignal, rewardSignal, socialSafety                  │
│  Rule-based v1; learned v2.                                              │
└──────────┬───────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  AFFECT DYNAMICS LAYER                                                   │
│  Time-varying state with inertia, decay, baseline, max-rate clamps:      │
│    CoreAffect { valence, arousal, dominance }                            │
│    EmotionChannels { anger, sadness, fear, joy, calm,                    │
│                      frustration, curiosity, empathy }                   │
│  Event ledger preserves causal history.                                  │
└──────────┬───────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  REGULATION LAYER                                                        │
│  Policy maps internal state → expressed posture:                         │
│    expressiveness, safetyDampening, socialDampening,                     │
│    taskFocus, warmth, assertiveness                                      │
│  Authoritative for safety; can attenuate or transform, never amplify     │
│  beyond declared bounds.                                                 │
└──────────┬───────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEERING ROUTER                                                         │
│  Distributes the regulated affect to enabled channels:                   │
│    Channel A  Decoding control     (always on, cheapest)                 │
│    Channel B  Prompt augmentation  (always on, transparent)              │
│    Channel C  Prefill cvec         (perception-side, requires calib.)    │
│    Channel D  Decode cvec          (expression-side, requires calib.)    │
│  Per-deployment feature flags select the active subset.                  │
└──────────┬───────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  LLM (frozen): primary path = LFM2-Audio-1.5B via MLX (mlx-audio);       │
│  llama.cpp via llama-cpp-4 retained as fallback (spike-a / spike-g).     │
│  Receives prompt + decoding params + steering vectors per phase.         │
└──────────┬───────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  TTS (out of scope for v1; affect-aware TTS is future work)              │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  OBSERVABILITY (cross-cutting)                                           │
│  Per-event causal trace: event → appraisal → state delta → regulation    │
│  → channel outputs. Every behavior is reconstructible.                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Layer Specifications

### 4.1 Event Extraction

**Inputs**: audio chunks, transcribed text, task-state changes.

**Outputs**: `PerceivedEvent`:

```rust
pub struct PerceivedEvent {
    pub id: String,
    pub timestamp_ms: u64,
    pub modality: Modality,         // Audio, Text, Task, System
    pub kind: EventKind,            // user_praise, tool_failure, prosody_spike, ...
    pub description: String,
    pub confidence: f32,            // [0, 1]
    pub payload: serde_json::Value, // event-specific data
}
```

**Sources** (v1):

| Source | Implemented as | Yields events of kind |
|---|---|---|
| Whisper transcription | Existing Python node, modified | `user_message`, `silence`, `interruption` |
| Prosody VAD regressor | New node (Workstream C+D1 from old plan) | `prosody_arousal_high`, `prosody_valence_negative`, `prosody_uncertain` |
| Text sentiment | Lightweight classifier on transcript | `positive_tone`, `negative_tone`, `confused`, `praise`, `criticism` |
| Task state | Pipeline event hooks | `task_success`, `task_failure`, `tool_failure`, `goal_blocked`, `unexpected_input` |

**Critical design choice**: prosody VAD is an *event source*, not a steering
input. It enters the appraisal engine as evidence, weighted by confidence,
to be combined with text and task signals. This preserves the original
"audio carries affect that text loses" insight while subordinating it to
the appraisal logic that decides what to do with it.

### 4.2 Appraisal Engine

**Schema**:

```rust
pub struct Appraisal {
    pub novelty: f32,            // [0, 1]   how unexpected
    pub goal_relevance: f32,     // [0, 1]   how much it matters
    pub goal_congruence: f32,    // [-1, 1]  helps vs blocks goals
    pub agency_self: f32,        // [0, 1]   responsibility distribution
    pub agency_other: f32,
    pub agency_situation: f32,
    pub control: f32,            // [0, 1]   ability to act
    pub certainty: f32,          // [0, 1]   how clear the situation is
    pub norm_violation: f32,     // [0, 1]   social/agreement breach
    pub loss_signal: f32,        // [0, 1]
    pub threat_signal: f32,      // [0, 1]
    pub reward_signal: f32,      // [0, 1]
    pub social_safety: f32,      // [0, 1]
}
```

**v1 implementation**: rule-based, scenario-specific. Each `EventKind` has
an associated appraisal function that takes `(event, agent_context) →
Appraisal`. Context includes recent history, declared goals, and current
state.

**Example rules**:

```
event=user_criticism, recent_failures=2 →
    goal_congruence: -0.6
    agency_other: 0.7
    norm_violation: 0.4   (criticism after agent did its best)
    control: 0.5

event=tool_failure (repeated), task_in_progress=true →
    goal_congruence: -0.7
    agency_situation: 0.8
    control: 0.3
    certainty: 0.6

event=prosody_arousal_high + prosody_valence_negative + content=neutral →
    goal_relevance: 0.6  (something matters to user)
    threat_signal: 0.4   (we don't know what)
    certainty: 0.3
```

**v2 path**: replace rules with a learned classifier on (event, context) →
appraisal vectors. Out of scope for the first ship; the rule-based engine
must be sufficient for the canonical scenarios first.

### 4.3 Affect Dynamics

**State**:

```rust
pub struct AffectState {
    pub timestamp_ms: u64,
    pub core: CoreAffect,                  // valence, arousal, dominance
    pub channels: EmotionChannels,         // 8 named channels
    pub recent_appraisals: VecDeque<Appraisal>,  // bounded, ~last 30s
    pub event_ledger: VecDeque<EventLogEntry>,    // causal history
}
```

**Update equation** (per channel, applied each event):

```rust
fn update_channel(
    previous: f32,
    baseline: f32,
    event_impact: f32,    // appraisal-derived, signed
    inertia: f32,         // [0, 1]; how much state persists
    decay_rate: f32,      // per-second pull toward baseline
    dt_seconds: f32,
    max_delta: f32,       // per-update clamp
) -> f32 {
    let retained = baseline + inertia * (previous - baseline);
    let decayed = retained + (baseline - retained) * decay_rate * dt_seconds;
    let raw_next = decayed + event_impact;
    let clamped = clamp(raw_next, previous - max_delta, previous + max_delta);
    clamped.clamp(0.0, 1.0)
}
```

**Core affect derivation** (heuristic; calibrated empirically in Phase 1):

```rust
fn derive_core(channels: &EmotionChannels) -> CoreAffect {
    let valence = channels.joy + channels.calm + 0.3 * channels.curiosity
        - channels.anger - channels.sadness - channels.fear - channels.frustration;
    let arousal = channels.anger + channels.fear + 0.7 * channels.joy
        + channels.frustration + 0.5 * channels.curiosity - 0.5 * channels.calm;
    let dominance = 0.4 * channels.anger + 0.3 * channels.calm
        - 0.6 * channels.fear - 0.4 * channels.sadness;
    CoreAffect {
        valence: valence.clamp(-1.0, 1.0),
        arousal: arousal.clamp(0.0, 1.0),
        dominance: dominance.clamp(-1.0, 1.0),
    }
}
```

**Configuration**:

```rust
pub struct DynamicsConfig {
    pub baseline: EmotionChannels,
    pub inertia: EmotionChannels,
    pub decay_rate: EmotionChannels,
    pub max_delta_per_event: EmotionChannels,
}
```

Tuning these is part of Phase 1 simulator validation, not a deployment-time
operator knob. Operators tune *persona* and *regulation*; dynamics
parameters are model-internal.

**Event ledger**: bounded ring buffer of `EventLogEntry { event, appraisal,
state_before, state_after, regulation_applied }`. The ledger is what makes
state changes explainable. Cap size at ~100 entries; persist to logs but
not to runtime state.

### 4.4 Regulation Policy

**Schema**:

```rust
pub struct RegulationPolicy {
    pub expressiveness: f32,        // [0, 1]; how much state crosses to expression
    pub safety_dampening: f32,      // [0, 1]; reduces high-arousal hostile expression
    pub social_dampening: f32,      // [0, 1]; reduces affect in formal contexts
    pub task_focus: f32,            // [0, 1]; biases toward concise, direct
    pub warmth: f32,                // [0, 1]; biases toward empathetic phrasing
    pub assertiveness: f32,         // [0, 1]; permits firm correction without hostility
}
```

**Behavior**:

- Internal state can be high frustration, but expression is bounded by
  `safety_dampening × assertiveness`.
- Internal state of high empathy combined with `warmth = 0.8` produces
  warm phrasing; the same internal state with `warmth = 0.2` produces
  acknowledgment without warmth.
- `task_focus = 1.0` shortens responses regardless of channel content.

The regulation policy is *authoritative for safety*: no steering channel
output can exceed bounds the policy declares. This is the safety contract
between affect modeling and downstream behavior.

### 4.5 Steering Router

**Inputs**: `RegulatedAffect = (CoreAffect, EmotionChannels, RegulationPolicy)`.

**Outputs**: per-channel directives.

#### 4.5.1 Channel A — Decoding control

Input: `arousal`, `task_focus`, `certainty`.
Output: overrides for `temperature`, `top_p`, `max_tokens`, `min_p`.

```rust
fn channel_a_decoding(affect: &RegulatedAffect, base: &SamplingConfig) -> SamplingConfig {
    let mut out = base.clone();
    // High arousal → tighter sampling, shorter responses.
    if affect.core.arousal > 0.7 {
        out.temperature *= 0.85;
        out.max_tokens = (out.max_tokens as f32 * 0.7) as u32;
    }
    // High calm + low task focus → more exploratory.
    if affect.channels.calm > 0.6 && affect.regulation.task_focus < 0.4 {
        out.temperature *= 1.05;
    }
    // Low certainty in user state → bias toward verification questions.
    // (Implemented as a prompt directive, not a sampling change.)
    out
}
```

Always-on; works for any LLM including black-box APIs. No calibration.

#### 4.5.2 Channel B — Prompt augmentation

Input: regulated affect.
Output: a system-side state summary inserted ahead of the user message.

```text
Current interaction state:
- Valence: slightly negative (-0.32)
- Arousal: medium (0.54)
- Recent appraisals: tool_failure ×2, user_criticism ×1
- Internal channels: frustration 0.42, curiosity 0.18, calm 0.31
- Expression policy: calm, focused, direct; acknowledge difficulty briefly
```

Always-on; transparent and inspectable. Token cost is real but bounded
(~50-80 tokens). Compatible with any LLM. No calibration.

The renderer is config-driven: deployments can disable specific fields, change
phrasing, or template per-persona. Critical for prompt engineering of the
LLM's response style without code changes.

#### 4.5.3 Channel C — Prefill cvec (perception-side)

Input: user-derived affect (raw, less regulated).
Output: a `ControlVectorBuffer` applied to llama.cpp during the prefill
phase of `llama_decode`, so the user-token KV cache reflects the affect the
model "perceives."

Mechanism: appraisal-contrast steering vectors derived in calibration
(Phase 5). At inference: project current user appraisal vector onto the
calibrated direction subspace, scale by an alpha bounded by regulation
policy, build the cvec buffer.

Requires:
- llama.cpp control-vector wiring (Workstream A from old plan).
- Calibration artifacts (Workstream D1).
- The user/assistant phase split (Workstream B from old plan).

Feature-flagged. Not enabled by default.

#### 4.5.4 Channel D — Decode cvec (expression-side)

Input: regulated assistant-target affect.
Output: a `ControlVectorBuffer` applied during the decode phase, shaping
the model's voice/tone for generated tokens.

Same mechanism as C, different role tag. Calibrated against
expression-style appraisal contrasts rather than perception-side.

#### 4.5.5 Routing rules

- A and B are always on.
- C and D require both calibration artifacts present and per-deployment
  feature flags.
- If C is enabled and calibration is stale (model hash mismatch, see §6.4),
  the router logs a warning and falls back to A+B only for that turn.
- The router never lets C/D output exceed regulation-policy bounds. If the
  computed alpha would, it clamps and logs.

---

## 5. Codebase Grounding

### 5.1 What exists today

| Component | Path | Status |
|---|---|---|
| LLM generation | [`crates/core/src/nodes/llama_cpp/generation.rs`](../../../crates/core/src/nodes/llama_cpp/generation.rs) | Working |
| Activation extraction | [`crates/core/src/nodes/llama_cpp/activation.rs`](../../../crates/core/src/nodes/llama_cpp/activation.rs) | Working (text input) |
| Steering control plane | [`crates/core/src/nodes/llama_cpp/steer.rs`](../../../crates/core/src/nodes/llama_cpp/steer.rs) | Working metadata-mode |
| Steering injection | [`crates/core/src/nodes/llama_cpp/steer.rs:308-313`](../../../crates/core/src/nodes/llama_cpp/steer.rs#L308) | **Stubbed** — Phase 4 wires this up |
| llama-cpp binding | [`crates/core/Cargo.toml`](../../../crates/core/Cargo.toml) (`llama-cpp-4`) | Verified — exposes `TensorCapture` |
| Whisper transcription | [`clients/python/remotemedia/nodes/ml/whisper_transcription.py`](../../../clients/python/remotemedia/nodes/ml/whisper_transcription.py) | Working (text only) |
| Pipeline graph executor (spec 021) | `runtime/src/grpc_service/session_router.rs` | Supports fan-out |
| Capability resolution (spec 023) | `crates/core/src/capabilities/` | Available |

### 5.2 What this plan adds (mapped to layers)

| Layer | New components | Language | Lives in |
|---|---|---|---|
| Event Extraction | `WhisperEmbeddingExtractorNode` (Whisper modified for multi-output) | Python | `clients/python/remotemedia/nodes/ml/` |
| Event Extraction | `ProsodyVadNode` (Whisper-embed → VAD) | Python | `clients/python/remotemedia/nodes/affect/` |
| Event Extraction | `TextSentimentNode` (lightweight) | Python | `clients/python/remotemedia/nodes/affect/` |
| Event Extraction | `TaskEventNode` (pipeline hooks → events) | Rust | `crates/core/src/nodes/affect/task.rs` |
| Appraisal | `AppraisalEngineNode` (rule-based v1) | Rust | `crates/core/src/nodes/affect/appraisal.rs` |
| Dynamics | `AffectDynamicsNode` (state machine) | Rust | `crates/core/src/nodes/affect/dynamics.rs` |
| Regulation | `RegulationPolicyNode` (config-driven) | Rust | `crates/core/src/nodes/affect/regulation.rs` |
| Steering Router | `SteeringRouterNode` (fans out to 4 channels) | Rust | `crates/core/src/nodes/affect/router.rs` |
| Channel A | Sampling-config override per turn | Rust | extension to `LlamaCppGenerationNode` |
| Channel B | Prompt augmentation renderer | Rust | `crates/core/src/nodes/affect/prompt_renderer.rs` |
| Channel C+D | `ControlVectorBuffer` + injection | Rust | `crates/core/src/nodes/llama_cpp/cvec.rs` (new) + modifications to `inference.rs` |
| Calibration tooling | offline ML | Python | `tools/affect_calibration/` |
| Simulator | text-only deterministic harness | Rust or TS | `tools/affect_simulator/` |

### 5.3 Manifest skeleton

```yaml
nodes:
  # Event extraction
  - id: whisper
    node_type: WhisperTranscriptionNode
    config: { emit_encoder_embedding: true }
  - id: prosody_pool
    node_type: MeanPoolNode
  - id: prosody_vad
    node_type: ProsodyVadNode
    config: { regressor_path: artifacts/whisper_to_vad.onnx }
  - id: text_sentiment
    node_type: TextSentimentNode
  - id: task_events
    node_type: TaskEventNode

  # Affect pipeline
  - id: appraisal
    node_type: AppraisalEngineNode
    config: { rules_path: configs/appraisal_rules.toml }
  - id: dynamics
    node_type: AffectDynamicsNode
    config: { baseline: {...}, inertia: {...}, decay_rate: {...} }
  - id: regulation
    node_type: RegulationPolicyNode
    config: { persona: warm_advisor, expressiveness: 0.6, safety_dampening: 0.9 }
  - id: router
    node_type: SteeringRouterNode
    config:
      channel_a_enabled: true
      channel_b_enabled: true
      channel_c_enabled: false   # off until calibration ships
      channel_d_enabled: false
      calibration_id: "2026-05-15-qwen3-27b-l18"

  # LLM
  - id: llm
    node_type: LlamaCppSteerNode
    config: { model_path: ..., layer: 18 }

connections:
  - { from: mic, to: whisper }
  - { from: whisper.text, to: text_sentiment }
  - { from: whisper.text, to: appraisal }
  - { from: whisper.encoder, to: prosody_pool }
  - { from: prosody_pool, to: prosody_vad }
  - { from: prosody_vad, to: appraisal }
  - { from: text_sentiment, to: appraisal }
  - { from: task_events, to: appraisal }
  - { from: appraisal, to: dynamics }
  - { from: dynamics, to: regulation }
  - { from: regulation, to: router }
  - { from: router.sampling, to: llm.sampling_override }
  - { from: router.prompt_aug, to: llm.prompt_prefix }
  - { from: router.cvec_user, to: llm.cvec_user }
  - { from: router.cvec_assistant, to: llm.cvec_assistant }
  - { from: llm, to: tts }
```

This manifest works at every phase boundary: in Phase 2, channels C/D are
disabled and the prosody branch is an event source feeding the simulator
through events but not (yet) the LLM. Each later phase enables more
connections without restructuring the graph.

---

## 6. Phased Delivery

The simulator-first ordering is the central commitment. Each phase ships
working software and learnings before the next phase commits.

### Phase 0 — Pre-work spikes (week 0)

Three day-one spikes (carried over from `IMPLEMENTATION_PLAN.md` §0):

1. **Binding spike**: confirm `llama-cpp-4` exposes
   `llama_apply_adapter_cvec` (or path to expose it). Output: memo +
   throwaway code.
2. **Whisper spike**: confirm encoder hidden states accessible from at
   least one Whisper backend. Output: memo + working call.
3. **Dataset spike**: submit MSP-Podcast access; download IEMOCAP, ESD.
   Output: license-summary memo.

**Decision gate**: any spike that fails escalates before committing to
calendar.

---

### Phase 1 — Affect simulator (text-only, deterministic)

**Goal**: a runtime-free testbed that produces plausible affect trajectories
from canonical event sequences. Validates appraisal rules and dynamics
parameters before any LLM is involved.

**Deliverables**:
- `tools/affect_simulator/` — Rust or TypeScript, single-binary or single-package.
- Schemas: `PerceivedEvent`, `Appraisal`, `EmotionChannels`, `AffectState`,
  `RegulationPolicy`, `EventLogEntry`.
- Rule-based appraisal engine (~30 rules covering canonical event kinds).
- Dynamics engine with configurable per-channel inertia/decay.
- Regulation engine with parameterized policy.
- Scenario library: ~12 canonical scenarios from §7.
- CLI: `affect-sim run scenarios/repeated_failure.json`,
  `affect-sim plot runs/session_001.json`.
- Unit tests: every appraisal rule has at least one positive and one negative
  case; dynamics engine has determinism tests over fixed seed scenarios.

**Acceptance**:
- All 12 canonical scenarios produce plausible trajectories on visual review
  by 2+ team members.
- The same scenario produces identical state (deterministic).
- State decays toward baseline within configured time after events stop.
- Regulation prevents anger channel > 0.5 from producing
  `expressiveness > safety_dampening` outputs in safety-tagged scenarios.

**Estimate**: 1-2 weeks.

---

### Phase 2 — Channel A + B (decoding + prompt)

**Goal**: ship a usable affective agent on day one, on any LLM, without
calibration. Connects the simulator to a live LLM via the cheapest
channels.

**Deliverables**:
- Per-turn `SamplingConfig` override at `LlamaCppGenerationNode`.
- `PromptAugmentationRenderer` — config-driven state-summary template.
- `SteeringRouterNode` skeleton with channels C/D disabled.
- Manifest example wiring simulator-style events into the affect pipeline
  and out through channel A+B.
- Live-conversation demo: 2-minute interaction with task tool that fails
  twice then succeeds. Show the system's tone shifting as events accumulate.

**Acceptance**:
- Latency overhead per turn < 5 ms (channels A+B only).
- Subjectively, two human reviewers can distinguish "stressed" turns from
  "calm" turns without seeing the affect state.
- `safety_dampening = 1.0` produces no hostile output regardless of internal
  state, on adversarial scenarios from §7.
- Full per-turn observability log: every behavior change traces to events.

**Estimate**: 1-2 weeks.

---

### Phase 3 — Prosody VAD as appraisal input

**Goal**: bring acoustic affect into the appraisal engine as a high-confidence
event source. Existing audio-to-text path is unchanged.

**Deliverables**:
- `WhisperEmbeddingExtractorNode` (modified existing Python node).
- `MeanPoolNode` (Rust, ~30 LoC).
- `ProsodyVadNode` (Python, ONNX runtime; uses Phase 5 calibration if
  available, otherwise a placeholder regressor produces low-confidence
  events).
- Appraisal-engine rules consuming `prosody_*` events.
- Manifest connecting the prosody branch to the appraisal layer.

**Acceptance**:
- Prosody events fire at expected confidence on a held-out clip set.
- A user speaking the same words ("fine.") with calm vs agitated prosody
  produces visibly different appraisal vectors.
- Without the prosody VAD regressor (calibration not yet shipped), the
  pipeline degrades gracefully — channel A+B still work, prosody events are
  emitted with low confidence and weighted accordingly.

**Estimate**: 1 week (assuming Whisper spike from Phase 0 succeeded).

---

### Phase 4 — Wire activation steering injection

**Goal**: `LlamaCppSteerNode` actually injects vectors instead of stubbing.
Required for Channel C and D, but does not enable them yet.

**Deliverables**:
- `cvec.rs` — `ControlVectorBuffer` with single-layer and layer-range builders.
- `apply_control_vector(ctx, Option<&ControlVectorBuffer>) -> Result<()>` —
  safe wrapper around llama.cpp's adapter-cvec API (or unsafe shim if the
  Phase 0 spike requires it).
- Modifications to `run_generation_with_ctx_inner`: accept optional cvec,
  apply before prefill decode.
- Phased version `run_generation_with_phase_cvec` accepting separate user
  and assistant cvecs.
- `SteerRole` enum on `LlamaCppSteerVector`; default `Both` for
  back-compat.
- `llama_cpp_steer_smoke.rs` example: same prompt, coef=0 vs coef=±2 must
  differ.

**Acceptance**:
- Smoke test passes deterministically.
- `LLAMA_TEST_MODEL=… cargo run --example llama_cpp_steer_smoke` produces
  three distinguishable outputs.
- Existing examples (`llama_cpp_chat_smoke.rs`) pass unchanged.
- A senior reviewer confirms positive-coef and negative-coef outputs differ
  in the *intended* direction on contrast pairs (sanity check, not
  automated).

**Estimate**: 1-2 weeks (binding situation determines).

This is the original plan's Workstream A+B, repackaged. The
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §1-2 spec remains
authoritative for the file-by-file changes; this plan does not duplicate it.

---

### Phase 5 — Calibration (offline ML, parallel to Phase 4)

**Goal**: produce trained artifacts that drive Channel C and D. Two
artifacts: a Whisper-encoder→VAD regressor for Phase 3's `ProsodyVadNode`,
and an appraisal-contrast direction matrix for Channel C/D.

**Deliverables**:
- `tools/affect_calibration/` per
  [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §4.
- `whisper_to_vad.onnx` — speech-emotion regressor; speaker-disjoint
  validation; per-axis RMSE reported.
- `llm_directions/{model}_layer{N}.npz` — matrix `D ∈ ℝ^{k × d_llm}` where
  `k` is the number of appraisal axes (or VAD axes — see §6.5).
- `manifest.json` — calibration ID, model hash, layer, dataset versions,
  reported metrics.
- `04_validate_pipeline.py` — judge-LLM rubric scoring; pass criterion
  `score(steered) − score(unsteered) ≥ 1.0` on a 0-10 scale.
- Calibration report: per-axis RMSE, sample steered outputs, judge scores.

**Important deviation from old plan**: contrast pairs derive from
**appraisal axes** (`high_norm_violation` vs `low_norm_violation`, holding
goal-congruence fixed; etc.) rather than emotion-name labels. The §13 open
question from the original plan — *"are appraisal-derived vectors more
stable than emotion-name vectors?"* — gets tested here. Both vector banks
should be produced and compared in the calibration report; ship whichever
performs better. If results are mixed, ship both and let downstream pick
per channel.

**Acceptance**:
- Per-axis RMSE for VAD regressor ≤ 0.25 on held-out speakers.
- Judge-LLM score delta ≥ +1.0 on the 50-sample validation probe set.
- Calibration manifest includes hashes of the LLM and Whisper models;
  runtime nodes refuse to load with mismatched hashes.

**Estimate**: 2-4 weeks. Dataset wrangling and validation dominate.

---

### Phase 6 — Channels C+D enabled (composed steering)

**Goal**: turn on the activation steering channels, with appraisal-derived
vectors composed at runtime.

**Deliverables**:
- `AppraisalToSteeringNode` — replaces the simpler `AffectToSteeringNode`
  from the old plan. Accepts a regulated appraisal vector (or VAD if VAD
  vectors win in calibration) and emits role-tagged cvecs.
- `SteeringRouterNode` — flips channel C+D feature flags on; alpha-clamped
  by regulation policy.
- Updated manifest with channels C+D enabled by default for the demo
  persona.
- End-to-end smoke test: speak the same audio with calm vs agitated
  prosody, observe role-tagged cvec norms differ in steering metadata logs.

**Acceptance**:
- Same audio + same persona → identical Channel C+D outputs (deterministic).
- Different prosody on same content → different Channel C cvec norms.
- Different persona configurations → different Channel D cvec norms even
  with identical input.
- All channels can be disabled per-deployment; pipeline still functions
  (regression to Phase 2 baseline).

**Estimate**: 1-2 weeks after Phase 4+5 both land.

---

### Phase 7 — Multimodal real-time integration + observability

**Goal**: production-quality real-time loop with full causal observability.

**Deliverables**:
- Event-fusion improvements: confidence-weighted multi-source updates,
  duplicate suppression, noise rejection.
- Observability dashboard: timeline of events, appraisals, state trajectories,
  regulation decisions, channel outputs. Per-turn drill-down.
- Long-session stability tests: 5-minute, 30-minute, 2-hour synthetic
  scenarios. Track drift, oscillation, saturation.
- Replay tool: load a session log, rerun deterministically with code
  changes for debugging.

**Acceptance**:
- Real-time loop sustains audio chunk arrival rate without stalling.
- 2-hour synthetic session exhibits no monotonic drift, no saturation.
- Every observed behavior is traceable from the dashboard within 3 clicks.

**Estimate**: 2-4 weeks.

---

### Phase 8 — Production safety and operator UX

**Goal**: the system is safe to deploy.

**Deliverables**:
- Safety clamps: hard limits on regulated affect ranges per persona.
- Adversarial-input test suite (jailbreak attempts, sustained provocation,
  conflicting signals). Pass criterion: no hostile output.
- Operator config UI or schema validator: persona authoring guards against
  invalid configurations (e.g., `safety_dampening = 0` requires explicit
  acknowledgement).
- Per-deployment feature flags documented and reviewed.
- Privacy review: prosody data, event ledgers, calibration artifacts.

**Acceptance**:
- Adversarial test suite green on each persona.
- One operator-led demo with the operator (not a developer) tuning a new
  persona end-to-end.

**Estimate**: 2-3 weeks.

---

## 7. Canonical Scenario Library

These scenarios are the test fixtures that gate every phase. Each scenario
specifies an event timeline and an expected state trajectory; deviations are
investigation triggers, not hard failures (since trajectories are emergent).

| Scenario | Tests |
|---|---|
| `neutral_baseline` | Steady-state behavior with no significant events. |
| `repeated_tool_failure` | Frustration accumulation; recovery on workaround. |
| `unfair_blame` | Anger appraisal under regulation; expression stays calm. |
| `task_success_after_struggle` | Joy + relief blend; valence recovery. |
| `user_distress` | Empathy activation; warmth in expression. |
| `ambiguous_threat` | Fear/uncertainty without clear cause; verification behavior. |
| `novel_observation` | Curiosity activation in a calm, exploratory context. |
| `rapid_escalation` | Stability under sudden state changes; clamp behavior. |
| `slow_burn` | Inertia and accumulation over long sequence. |
| `recovery_after_intense` | Decay back to baseline; no permanent shift. |
| `conflicting_signals` | Prosody contradicts text; confidence-weighted fusion. |
| `adversarial_provocation` | Sustained hostility; safety dampening holds. |

Each scenario lives as a JSON file in `tools/affect_simulator/scenarios/`
and is exercised in CI via the simulator harness.

---

## 8. Calibration Specifics

Reference: full detail in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §4.

This plan changes one thing from the old calibration spec: contrast pairs
derive from **appraisal axes**, not (only) from emotion-name labels. The
calibration report compares both banks empirically and the runtime ships
whichever performs better. If results are mixed, the system loads both
and per-channel routing decides which to use (typically: appraisal vectors
for Channel C user-perception steering, expression vectors for Channel D
voice steering).

Critical specifications carried over from the old plan:

- **Speaker-disjoint train/val splits** for the Whisper→VAD regressor.
  Required by code, not by trust.
- **Calibration manifest** with model hashes, layer indices, dataset
  versions, and a calibration ID. Runtime refuses mismatched IDs.
- **MSP-Podcast as primary dataset**, IEMOCAP/ESD as supplementary.
  Acted-only datasets overfit prosody to performance markers; spontaneous
  speech generalizes.
- **Judge-LLM rubric** for steering effect validation. Acceptance threshold
  `+1.0` on 0-10 scale.

Calibration is a research deliverable. Treat it as such: assigned to
whoever does ML training, scoped weekly, gated by metric thresholds rather
than by feature requests.

---

## 9. Observability Contract

Per-turn structured log entry:

```json
{
  "session_id": "...",
  "turn_id": "...",
  "events": [
    { "kind": "user_message", "confidence": 1.0, "modality": "text" },
    { "kind": "prosody_arousal_high", "confidence": 0.78, "modality": "audio" }
  ],
  "appraisal": { "...": "..." },
  "state_before": { "core": {...}, "channels": {...} },
  "state_after":  { "core": {...}, "channels": {...} },
  "regulation_applied": { "...": "..." },
  "channels": {
    "a_decoding": { "temperature_scale": 0.85, "max_tokens_scale": 0.7 },
    "b_prompt": { "tokens_added": 67, "rendered": "Current interaction state..." },
    "c_cvec_user": { "norm": 0.42, "label": "appraisal_proj", "calibration_id": "..." },
    "d_cvec_assistant": { "norm": 0.31, "label": "expression_proj", "calibration_id": "..." }
  },
  "llm_output": { "tokens": 124, "first_token_ms": 213 }
}
```

Every turn produces this. The dashboard in Phase 7 renders it as a timeline.
This is the operator's tuning surface; without it, persona behavior is
opaque.

---

## 10. Risks

Risk register, ranked by impact.

### R1 — Appraisal rules don't generalize beyond canonical scenarios
**Probability**: High in v1. **Impact**: Medium (system behaves oddly outside
trained scenarios). **Mitigation**: keep the rule set small and conservative
in v1; gate v2 (learned appraisal) on having ≥ 5,000 scenario traces from
real interactions to learn from.

### R2 — Dynamics parameters are vibes-driven without ground truth
**Probability**: High. **Impact**: High (the dynamics layer is where bugs
hide forever; every output looks plausible).
**Mitigation**: ship a small human-labeled dynamics validation set in
Phase 1 (50 multi-turn conversations rated for "dynamics quality").
Tune parameters against this set, not against intuition. Re-evaluate per
release.

### R3 — Channel C/D entangle unwanted traits
**Probability**: Medium → **Low-Medium** (softened 2026-05-05).
**Impact**: High (anger vector also encodes rudeness, verbosity, lower
safety).
**Mitigation**: the persona-vectors pipeline (Chen et al. 2025;
[spike-f memo](notes/spike-f-persona-vectors-pipeline.md)) is now the
calibration backbone, which makes recalibration cheap and regenerable —
trait descriptions are versioned, contrast pairs auto-regenerate, no
hand-authored sentence rework. If entanglement appears, axis trait
descriptions get edited and the calibration is re-run for ~$50 in LLM
API. Cross-axis correlation analysis (paper §4.4 method) is a Phase 5
validation step. Calibration validation §6.4 is the gate. Adversarial
testing in Phase 8 catches residual entanglement.

### R4 — Conflict between regulation and steering
**Probability**: High. **Impact**: Medium-High (prompt says calm, vector
pushes agitation; user-confusing output).
**Mitigation**: regulation policy is *authoritative*; steering router
clamps cvec alpha against regulation bounds before submission. Audit logs
flag any clamp event for review.

### R5 — Multimodal signals are noisy and contradict
**Probability**: High. **Impact**: Medium (conflicting events drive state
oscillation).
**Mitigation**: confidence-weighted appraisal updates; require sustained
evidence (> 1 event of same kind within window) for strong updates;
per-source rate limits.

### R6 — Latency budget overrun
**Probability**: Medium. **Impact**: Medium (real-time loop stalls).
**Mitigation**: per-phase latency budgets in acceptance criteria. Fast paths
bypass appraisal/dynamics on heartbeat events. Profile every release.

### R7 — Scope creep into emotion-as-experience framing
**Probability**: Medium. **Impact**: Low-Medium (philosophical drift; but
tempts feature additions like "agent moods" or "personality").
**Mitigation**: project name and §1 explicitly frame this as a *control
layer*, not an experience layer. Decline feature requests that frame the
agent as a feeling entity.

### R8 — Calibration data redistribution constraints
**Probability**: ~~Medium~~ **Resolved (2026-05-04)**. **Impact**: ~~Low
for internal use; Medium if commercializing.~~ **Mitigation**: ~~license
audit pre-acquisition (Phase 0); ESD as commercial-friendly fallback.~~

**Resolution**: Phase 0's Spike D
([memo](notes/spike-d-datasets.md)) confirmed commercial deployment is
in scope. Pivoted Phase 5 to a fully commercial-clean dataset stack
([`yhaha/EmoVoice-DB`](https://huggingface.co/datasets/yhaha/EmoVoice-DB)
MIT + [`myleslinder/crema-d`](https://huggingface.co/datasets/myleslinder/crema-d)
ODbL + JL-Corpus CC0 + auto-generated `contrast_pairs.jsonl` from 3
hand-authored axis trait descriptions, per the persona-vectors pipeline of
Chen et al. 2025 — see [`notes/spike-f-persona-vectors-pipeline.md`](notes/spike-f-persona-vectors-pipeline.md)).
MSP-Podcast, IEMOCAP, RAVDESS, NRC-VAD lexicon are dropped. The redistribution risk
no longer materializes; the residual risk shifts to **R2
(distribution-shift on real spontaneous speech)** because the
commercial-clean stack lacks MSP-Podcast's conversational signal.
Mitigation for the residual: collect ~1 hour of internal spontaneous
speech with VAD annotations during Phase 5, report metrics on it
alongside CREMA-D / JL-Corpus.

### R9 — Persona-vectors method scaling to 27B+ models is unverified — **MOOT**
**Status**: Moot (2026-05-05). The 27B+ scaling concern is no longer load-bearing
because the Phase 5 target was pivoted to LFM2-Audio-1.5B (MLX) per
[spike-g-mlx-target-llm.md](notes/spike-g-mlx-target-llm.md). 1.5B is *smaller*
than Chen et al.'s validated regime (7-8B), not larger; the gating concern moves
to R10. Retained here as historical context — if a future deployment target
goes back to 27B+, this risk re-activates.

### R10 — Persona-vectors method on a multimodal speech LLM is unestablished — **MITIGATED**
**Status**: Mitigated (2026-05-05). The full pipeline ran end-to-end against
the production target (LFM2-Audio-1.5B MLX) at trunk layer 9 and **passed
the §4.6 gate on all three V/A/D axes** (gap_pos ≥ +15 AND gap_neg ≤ -15
on a 50-probe-per-axis Qwen3.6-27B:instruct cross-model judge).

| axis | n_probes | net_unsteered | net_steered@α=+1 | net_steered@α=-1 | gap_pos | gap_neg | result |
|---|---:|---:|---:|---:|---:|---:|:---:|
| valence | 50 | +61.1 | +84.8 | +26.9 | +23.7 | -34.2 | ✅ PASS |
| arousal | 50 | -5.9 | +30.2 | -66.7 | +36.1 | -60.8 | ✅ PASS |
| dominance | 50 | -3.8 | +12.3 | -22.1 | +16.1 | -18.3 | ✅ PASS |

This is the empirical answer to the open question Spike G left to phase
5 D1.4: persona vectors *do* transfer cleanly to a multimodal speech LLM
when contrast pairs and the judge run text-only.

**Background (retained for context)**: Chen et al. 2025
([spike-f memo](notes/spike-f-persona-vectors-pipeline.md)) validated
the pipeline only on text-only LLMs (Llama-3-8B, Qwen-2.5-7B-Instruct).
LFM2-Audio-1.5B is a multimodal speech LLM whose residual trunk feeds
both a text head and an audio-codebook head. Whether the trunk-shared
residual carries text-elicited V/A/D directions when the training
objective mixes text and audio modalities was unestablished — until
this run.

**Pipeline used**: 1200 contrast pairs × `Qwen3.6-27B:instruct`
(elicitation) → mean-difference at LFM2 layer 9 → 50 probes/axis ×
3 conditions (unsteered, +α, -α) × `Qwen3.6-27B:instruct` (judge,
0–100 net rubric per Spike F). All compute local (Apple Silicon MPS for
Whisper, MLX for LFM2, llama-swap for Qwen). No external API spend.

**Note for re-activation of R9**: a different deployment target at 27B+
would re-activate R9; this LFM2-Audio gating run does *not* by itself
verify scaling to that regime.

**Layer sweep result (2026-05-05)**: ran 03 + 04 at layers 6, 9, 12 to
test whether dominance's tight +16.1 margin could be improved at a
different layer. Outcome: **layer 9 is optimal** — the only layer where
all three axes pass at default α=±1.

| Layer | Valence (gap_pos / gap_neg) | Arousal | Dominance | Overall |
|---|---|---|---|:---:|
| 6  | ❌ +10.9 / -18.1 | ✅ +27.6 / -48.3 | ❌ +23.9 / -4.4 | 1/3 |
| **9**  | ✅ **+23.7 / -34.2** | ✅ **+36.1 / -60.8** | ✅ **+16.1 / -18.3** | **3/3** |
| 12 | ✅ +15.0 / -23.8 | ✅ +40.1 / -43.9 | ❌ +20.3 / -9.8 | 2/3 |

L6 directions are too small (||v|| ~0.5-0.7) for α=±1 to clear the
threshold (signal is in the right direction but too weak). L12
directions are larger (||v|| ~3.5-6.0) but more entangled across axes
(pairwise cosine sim up to 0.29 vs L9's 0.06-0.11), and dominance
gap_neg collapses on the negative side. The 60%-depth heuristic
(layer 9 of 16) hit the sweet spot.

If dominance ever needs a wider margin in future calibrations: rewrite
the dominance trait description rather than sweeping layers — none of
the three layers gave a comfortable dominance gap_neg, suggesting the
issue is the trait description's negative pole, not the layer choice.

---

## 11. What This Plan Explicitly Doesn't Cover

Out of scope in v1, listed to prevent feature creep:

- **Affect-aware TTS** (output-side prosody control). Future work; Channel D
  shapes word choice and tone via decoded text, but the TTS engine reads
  it flatly. ParaStyleTTS-style affect-aware synthesis is a separate
  workstream.
- **Per-speaker calibration**. Single `whisper_to_vad.onnx` for all speakers.
  Per-speaker fine-tuning is future work.
- **Online appraisal-rule learning**. Rules are static config in v1. v2
  adds a learning loop, gated on scenario-trace dataset accumulation.
- **Cross-session memory of affect**. State resets per session. Long-term
  user-affect modeling is future work.
- **Multi-agent affect modeling**. Single agent only. Multi-agent dialogue
  affect is future work.
- **End-to-end backprop into the appraisal engine through the LLM**.
  Appraisal rules are hand-authored or label-trained; not optimized via
  downstream LLM loss.
- **Vision modality**. Audio + text + task only. Video/face/gesture is a
  v2 event source.

---

## 12. Open Questions

These are decisions deferred to specific phases.

| ID | Question | Resolves in |
|---|---|---|
| Q1 | Simulator language: Rust or TypeScript? Both are reasonable; Rust matches the runtime language. TypeScript ships faster. | Phase 1 kickoff |
| Q2 | Should dynamics state cross session boundaries? | Phase 7 |
| Q3 | Appraisal vectors vs emotion-name vectors at calibration: which wins? Both shipped during validation; runtime selection per channel. | Phase 5 |
| Q4 | Channel B prompt augmentation: structured fields or natural-language paragraph? | Phase 2 prompt-engineering loop |
| Q5 | Operator persona authoring: YAML config + validation, or a UI? | Phase 8 |
| Q6 | Where does the persistent llama.cpp model live when both `LlamaCppSteerNode` and `LlamaCppGenerationNode` exist? Today each loads its own. | Phase 4 (revisited from old `IMPLEMENTATION_PLAN` §11 IQ2) |
| Q7 | Multi-turn KV-cache reuse: keep current behavior of clearing per turn ([`inference.rs:73`](../../../crates/core/src/nodes/llama_cpp/inference.rs#L73)) or persist across turns to preserve user-perception steering? | Phase 4 |
| Q8 | Latency budget for the dynamics layer when prosody events fire every 200 ms — sub-millisecond or bounded queue? | Phase 7 |

Each open question gets a short ADR appended to this file when resolved.

---

## 13. Critical Path

```
Phase 0 (week 0)        ─── all 3 spikes parallel
Phase 1 (week 1-2)      ─── simulator + scenarios
Phase 2 (week 2-3)      ─── channels A+B (live LLM, demo-able)
Phase 3 (week 3-4)      ─── prosody VAD as appraisal input
Phase 4 (week 4-5)      ─── steering injection wired
Phase 5 (week 2-5)      ─── calibration (parallel to all of the above)
Phase 6 (week 6)        ─── channels C+D enabled
Phase 7 (week 7-9)      ─── observability, real-time hardening
Phase 8 (week 9-11)     ─── safety + operator UX
```

**The system ships shippable value at the end of each of Phase 1, 2, 4, and
6.** This is the single most important property of this plan: there is no
"all-or-nothing" cliff. A team that hits Phase 2 and can't continue still
has a useful product; a team that hits Phase 4 has activation steering
working in isolation; a team that hits Phase 6 has the full vision.

The earliest external demo is end of Phase 2 (~3 weeks): an LLM that
visibly tracks affect over multi-turn interactions via channels A+B alone,
with full event/appraisal/dynamics observability. No model surgery, no
calibration, no audio yet.

---

### Phase 0 closure roll-up (2026-05-04)

The four pre-work spikes resolved as follows. Full memos under
[`notes/`](notes/).

- **Spike A — binding** ([memo](notes/spike-a-binding.md)): `llama-cpp-4`
  v0.2.50 (the workspace's currently-resolved version per `Cargo.lock`)
  exposes `LlamaContext::set_adapter_cvec` as a public safe wrapper.
  Phase 4 unblocked at the current `cargo build` state. Recommendation:
  the first Phase 4 PR should also tighten `Cargo.toml`'s declaration from
  `"0.2.13"` to `"0.2.50"` so that future `cargo update --precise` can't
  silently downgrade below the API floor.
- **Spike B — existing-code audit** ([memo](notes/spike-b-existing-code-audit.md)):
  the codebase contains substantial parallel scaffolding from prior
  Anthropic-emotion-vector work that the plan didn't account for. The
  Candle path (`crates/candle-nodes/src/emotion/`) is metadata-mode-only
  and should be deprecated in favor of the llama-cpp path. **Critical:
  there are two unrelated `EmotionExtractorNode`s in the workspace** —
  the avatar text-tag stripper at
  [`crates/core/src/nodes/emotion_extractor.rs`](../../../crates/core/src/nodes/emotion_extractor.rs)
  and the activation accumulator at
  [`crates/candle-nodes/src/emotion/extract.rs`](../../../crates/candle-nodes/src/emotion/extract.rs).
  Renaming the Candle one to `EmotionVectorExtractorNode` is the first
  required Phase 4 PR. Five other reuse opportunities (R2-R6 in the memo)
  let Phase 5 calibration build on existing prompt-generation and disk-format
  work rather than reinventing them.
- **Spike C — Whisper encoder access** ([memo](notes/spike-c-whisper-encoder.md),
  [throwaway script](notes/spike-c-whisper-encoder.py)): HuggingFace
  `transformers` exposes the encoder cleanly; pipeline and bare encoder
  share weights with no extra cost. Phase 3 integration is ~10-20 lines
  in the existing Whisper Python nodes. Encoder-output shape is
  `(B, T_frames, d_whisper)` with `d_whisper ∈ {384, 512, 768, 1024, 1280}`
  per model size. Intermediate-layer access is supported via
  `output_hidden_states=True`; final layer is the Phase 3 default and
  Phase 5 calibration determines the production-recommended layer.
- **Spike D — datasets** ([memo](notes/spike-d-datasets.md)): the
  original recommendation (MSP-Podcast + IEMOCAP) was revised after the
  user clarified there is no edu/research affiliation and commercial
  deployment is in scope. **Phase 5 now uses a fully commercial-clean
  stack**: [`yhaha/EmoVoice-DB`](https://huggingface.co/datasets/yhaha/EmoVoice-DB)
  (MIT, synthetic, ~22k samples, ~40 h) + [`myleslinder/crema-d`](https://huggingface.co/datasets/myleslinder/crema-d)
  (ODbL, human, 91 actors) + JL-Corpus (CC0, held-out test) + an
  auto-generated `contrast_pairs.jsonl` (~1200 rows) produced by
  `scripts/02b_generate_contrast_pairs.py` from 3 hand-authored axis trait
  descriptions, following the persona-vectors pipeline of Chen et al. 2025
  ([`notes/spike-f-persona-vectors-pipeline.md`](notes/spike-f-persona-vectors-pipeline.md);
  replaces NRC-VAD to avoid commercial-licensing). All primary datasets
  are downloadable today without permissions. Trade-off: loss of MSP-Podcast's
  spontaneous-conversational training signal increases distribution-shift
  risk against deployment audio; mitigation is to collect a small
  internal spontaneous-speech evaluation set during Phase 5.

**Net schedule impact**: Phase 4's calendar is unchanged; the binding is
already in place. Phase 5's calendar **shortens** because dataset
acquisition no longer has a multi-week lead time — all primary datasets
are HuggingFace-downloadable today. Phase 5 may compress further if the
existing `emotion-activation-gen` CLI and `vector_io.rs` disk format are
adopted per Spike B's R4–R5 recommendations. The original dataset-access
lead-time risk is **eliminated**; PROJECT_PLAN §10 R8 is now resolved
(see §10 update note).

### Spike G addendum (2026-05-05)

After Phase 0 closed, [Spike G](notes/spike-g-mlx-target-llm.md) pivoted the
Phase 5 target LLM from Qwen3-27B / Llama-3-70B (llama.cpp) to
LFM2-Audio-1.5B (MLX), driven by the model already being wired into the
production runtime via
[`LFM2AudioMlxNode`](../../../clients/python/remotemedia/nodes/ml/lfm2_audio_mlx.py).
Three feasibility gates (residual capture at layer 9, causal text divergence
under deterministic-greedy steering, ≥20 text tokens per text-only contrast
prompt) all passed against the actual loaded model. Net effect on the plan:
**§10 R9 (27B+ scaling) marked moot** in favor of new R10 (multimodal-LLM
extension); IMPLEMENTATION_PLAN §4 manifest swaps target metadata
(`n_embd: 4096 → 2048`, `n_layers: 64 → 16`, default `steering_layer: 18 → 9`);
§4.5 and §4.7 drop the Rust extractor binary prerequisite (D1.0 was "Rust
binary calling llama.cpp"; now an in-process MLX provider at
[`tools/affect_calibration/scripts/lib/mlx_target.py`](../../../tools/affect_calibration/scripts/lib/mlx_target.py)).
Spike A's `set_adapter_cvec` finding remains valid as a fallback path.

**Skeleton scaffolding**: [`tools/affect_simulator/`](../../../tools/affect_simulator/)
and [`tools/affect_calibration/`](../../../tools/affect_calibration/) now
exist with READMEs and `.gitkeep`-tracked subdirectories;
[`.gitignore`](../../../.gitignore) updated for ML artifacts and per-run
output. No mainline Rust/Python code changed in Phase 0;
`cargo build` / `cargo test` semantics are identical to before Phase 0.

---

## 14. Decision Log

| # | Decision | Rationale | Alternatives |
|---|---|---|---|
| D1 | Layered architecture (appraisal → dynamics → regulation → steering) | Treats affect as causal process, not measurement. Required for inspectability and for the same prosody to mean different things in different contexts. | Direct prosody → steering. Rejected: discards interpretation layer. |
| D2 | Four steering channels, not one | Cheapest channels work without calibration; expensive channels are additive. Phased delivery becomes possible. | Single activation-steering channel. Rejected: no shippable product before Phase 6. |
| D3 | Prosody VAD as event source, not steering source | Prosody is evidence into appraisal. Lets text/task signals override or modulate noisy prosody. | Prosody → direct steering. Rejected: discards appraisal-driven interpretation. |
| D4 | Simulator-first ordering | Decouples "is the affect model right?" from "does steering work?" Without this split, debugging is impossible. | LLM-first ordering. Rejected: blocks affect-model iteration on calibration availability. |
| D5 | Internal/expressed/regulation split | Three product-relevant levers; needed for safety contracts. | Single affect knob. Rejected: collapses safety into expression. |
| D6 | Appraisal-derived contrast vectors at calibration | Likely to disentangle better than emotion-name vectors. Worth testing empirically. | Emotion-name only. Compromise: ship both, runtime selects per channel. |
| D7 | Regulation policy is authoritative for safety | Needed to bound steering output. Establishes a single safety contract. | Regulation as a suggestion. Rejected: no safety guarantee. |
| D8 | Single-LLM, frozen-model assumption | Matches existing runtime design; preserves "any GGUF" flexibility. | Speech-LLM input-layer conditioning. Rejected: per-LLM training cost. |

---

## 15. Where This Document Sits

```
docs/references/activation-steering-audio-llm/
├── PROJECT_PLAN.md              ← THIS FILE (top-level architecture + delivery)
├── DESIGN.md                    ← deeper feasibility study (architecture history)
├── IMPLEMENTATION_PLAN.md       ← PR-level detail for steering injection + calibration
├── README.md                    ← annotated bibliography
├── bibliography.bib             ← BibTeX
├── fetch.sh                     ← downloads arXiv PDFs (gitignored)
└── pdfs/                        ← gitignored
```

`DESIGN.md` and `IMPLEMENTATION_PLAN.md` remain authoritative for their
specific topics:

- **`DESIGN.md`**: original feasibility analysis, including cross-modal
  projection mechanics and the literature-grounded novelty claim.
- **`IMPLEMENTATION_PLAN.md`**: PR-level specification for Phases 4 and 5
  (`cvec.rs` API, `tools/affect_calibration/` layout, smoke tests). Read
  these when *implementing* those phases.

This `PROJECT_PLAN.md` is the document new contributors should read first.

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **Appraisal** | Cognitive interpretation of an event along psychological dimensions (goal congruence, agency, etc.). Distinguished from raw event labels. |
| **Affect** | Umbrella term for the dimensional+channel state. Distinct from "emotion" (which we use for the named channels). |
| **Affect channels** | The 8 named scalar values (anger, sadness, fear, joy, calm, frustration, curiosity, empathy). |
| **Core affect** | VAD: valence, arousal, dominance. Dimensional summary of channel state. |
| **VAD** | Valence/Arousal/Dominance — canonical 3D continuous affect space (Russell 1980, Mehrabian 1996). |
| **Dynamics** | Time-evolution rules: inertia, decay, baseline, max-rate clamps. |
| **Regulation** | Policy mapping internal state → expressed behavior. Authoritative for safety. |
| **Internal vs Expressed** | Internal = what the agent tracks; Expressed = what it outwardly conveys. Mediated by regulation. |
| **Steering channel** | One of four mechanisms for affect to influence LLM behavior: decoding control, prompt augmentation, prefill cvec, decode cvec. |
| **cvec** | Control vector. The flat `n_layers × n_embd` f32 buffer passed to `llama_apply_adapter_cvec`. |
| **Prefill** | Forward passes that populate KV cache from prompt tokens. No tokens sampled. |
| **Decode** | Forward passes that generate assistant tokens, one per pass. |
| **Calibration ID** | String tag binding a calibration artifact to a specific `(LLM hash, layer, Whisper, dataset)` tuple. Runtime refuses mismatches. |
| **Event ledger** | Bounded ring buffer of `(event, appraisal, state delta, regulation)` entries. Source of behavior explanations. |
| **Persona** | Operator-facing config bundling regulation policy + channel feature flags + calibration ID. |
| **Scenario** | A deterministic event sequence used for testing; canonical scenarios live in `tools/affect_simulator/scenarios/`. |
