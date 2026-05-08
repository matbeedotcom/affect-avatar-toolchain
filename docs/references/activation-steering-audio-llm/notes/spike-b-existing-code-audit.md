# Spike B — Existing Emotion/Steering Code Audit

**Status**: Complete. Substantial pre-existing scaffolding identified.
Recommendations issued for consolidation, naming, and reuse.

**Date**: 2026-05-04

---

## Question

What pre-existing scaffolding in the codebase already addresses parts of
[`PROJECT_PLAN.md`](../PROJECT_PLAN.md)? Where is overlap, where is reusable,
where would consolidation help? What names are colliding?

## TL;DR

The codebase contains a near-complete sister implementation of the steering
half of this project, modeled on Anthropic's 2026 emotion-vector
research. Specifically:

- **`crates/candle-nodes/src/emotion/`** — extract-and-steer pipeline,
  Tensor-based vector exchange, per-session coefficient state, vector I/O,
  pipeline manifest example. Currently in metadata-mode-only because the
  Candle framework lacks forward hooks (the same limitation that
  `llama_cpp/steer.rs` has, but for a different reason).
- **`examples/cli/emotion-activation-gen`** — emotion-labeled prompt
  generator, intended to feed an external LLM that captures residual-stream
  activations. Implements the dataset-generation half of D1 calibration in
  a usable form.
- **`examples/cli/pipelines/emotion-vector.yaml`** — canonical end-to-end
  pipeline manifest connecting extraction → steering with documented
  Python integration patterns.

Two parallel and **mutually unaware** implementations of activation steering
now exist in the workspace:

```
                        ┌─────────────┐                  ┌─────────────┐
   Activations    ───▶  │ Candle path │     vs.    ───▶  │ llama-cpp   │
                        │ (emotion/)  │                  │ (llama_cpp/)│
                        └─────────────┘                  └─────────────┘
                              │                                │
                       metadata-only                    metadata-only
                       (no Candle hooks)                (set_adapter_cvec
                                                         not yet wired)
```

The plan should consolidate around the **llama-cpp path** (the binding
already exposes the runtime injection API per
[`spike-a-binding.md`](spike-a-binding.md)) but **adopt the
Candle path's serialization format, pipeline manifest patterns, and prompt
generator** wholesale.

There is one **critical naming collision** that must be resolved before
either path is extended (see R1).

---

## Findings

### F1 — Naming collision: two unrelated `EmotionExtractorNode`s

There are two different node types named `EmotionExtractorNode` in the same
workspace. They share no code and have completely different purposes.

| Name | Path | Purpose |
|---|---|---|
| `EmotionExtractorNode` | [`crates/core/src/nodes/emotion_extractor.rs`](../../../../crates/core/src/nodes/emotion_extractor.rs) | **Text-tag stripper.** Strips `[EMOTION:emoji]` markers from a text stream and emits structured emotion events for avatar lip-sync / emoji rendering. Per spec [`docs/superpowers/specs/2026-04-27-live2d-audio2face-rvc-avatar-design.md`]. Has nothing to do with steering. |
| `EmotionExtractorNode` | [`crates/candle-nodes/src/emotion/extract.rs`](../../../../crates/candle-nodes/src/emotion/extract.rs) | **Activation accumulator.** Receives labeled `RuntimeData::Tensor` activations from upstream, accumulates per-emotion sums, computes mean-subtraction direction vectors on `"compute"` trigger. This is what our calibration plan (Phase 5) needs. |

This is a runtime hazard: both register against the same `node_type()`
string `"EmotionExtractorNode"`, and any pipeline manifest using that name
is ambiguous. Today they don't collide because they live in separate
factories that aren't both registered into the same registry by default —
but a future contributor wiring both could create a runtime conflict.

**Resolution required before Phase 4**: rename the Candle one. Suggested:
`EmotionVectorExtractorNode` (the avatar one keeps the shorter name; it's
the user-facing one).

### F2 — Anthropic emotion-vector lineage

Both [`crates/candle-nodes/src/emotion/mod.rs:3-10`](../../../../crates/candle-nodes/src/emotion/mod.rs#L3)
and [`examples/cli/emotion-activation-gen/README.md:94`](../../../../examples/cli/emotion-activation-gen/README.md#L94)
explicitly cite:

- **Anthropic's 2026 emotion-vector research**:
  https://transformer-circuits.pub/2026/emotions/index.html
- **Reference implementation**: https://github.com/mezoistvan/llm_feeling_weather

This is *exactly* the methodological foundation of our
[`PROJECT_PLAN.md`](../PROJECT_PLAN.md). The Candle path was built to
replicate this paper's technique. **The team has prior work on this exact
problem** that the planning docs didn't account for.

Implication: Phase 5 calibration should not start from scratch. The
contrast-pair pipeline (prompt generation → activation capture → mean-
subtraction → L2-normalize) is already partly built and tested.

### F3 — File-by-file classification

| File | Purpose | Classification |
|---|---|---|
| [`crates/candle-nodes/src/emotion/mod.rs`](../../../../crates/candle-nodes/src/emotion/mod.rs) | Module root; doc comment laying out the two-phase architecture | **Reusable as-is** for Candle path. Doc-only file. |
| [`crates/candle-nodes/src/emotion/config.rs`](../../../../crates/candle-nodes/src/emotion/config.rs) | `EmotionExtractConfig`, `EmotionSteerConfig`, `EmotionVectorMetadata`, `PoolingMode`, `SteeringVectorConfig` | **Reusable with adaptation.** `EmotionVectorMetadata` (lines 19-47) is a partial calibration manifest already (model, layer, hidden_size, emotion, pooling, n_positive, n_neutral, raw_norm, dataset_hash, system_prompt, normalized). Phase 5's manifest extends this with VAD/appraisal axes. |
| [`crates/candle-nodes/src/emotion/extract.rs`](../../../../crates/candle-nodes/src/emotion/extract.rs) | `EmotionExtractorNode` — accumulator + mean-subtraction computer | **Reusable with renaming.** Per F1, must rename. The `ActivationAccumulator` (lines 47-92) and `compute_vectors` (lines 207-288) logic is exactly D1 step 3 ("extract LLM directions") in [`IMPLEMENTATION_PLAN.md` §4.4](../IMPLEMENTATION_PLAN.md). |
| [`crates/candle-nodes/src/emotion/steer.rs`](../../../../crates/candle-nodes/src/emotion/steer.rs) | `EmotionSteeringNode` — vector registry, coefficient state, delta computation | **Parallel/divergent.** Same shape as [`llama_cpp/steer.rs`](../../../../crates/core/src/nodes/llama_cpp/steer.rs); the two should consolidate. Recommendation: keep llama-cpp version (has runtime path via `set_adapter_cvec`); deprecate the Candle one until Candle gains hooks. |
| [`crates/candle-nodes/src/emotion/vector_io.rs`](../../../../crates/candle-nodes/src/emotion/vector_io.rs) | Disk format (4-byte length prefix + JSON metadata + f32 LE vector data); math kernels (mean, subtract, l2_normalize, cosine_similarity, scale_for_steering, compute_steering_delta) | **Reusable as-is.** The disk format is a clean choice — adopt for calibration artifacts. The math kernels are pure-Rust, Candle-free, and should move to a shared crate (e.g. `crates/affect-math` or directly into `crates/core`). |
| [`crates/core/src/nodes/emotion_extractor.rs`](../../../../crates/core/src/nodes/emotion_extractor.rs) | Avatar text-tag stripper. Unrelated to steering. | **Out of scope.** No reuse for the affect pipeline; flagged only for the naming collision in F1. |
| [`examples/cli/emotion-activation-gen/`](../../../../examples/cli/emotion-activation-gen/) (Cargo.toml + src/main.rs + README.md) | Generates emotion-labeled prompts in 5 modes (Natural, Deflection, UnexpressedNeutral, UnexpressedStory, UnexpressedOthers); CLI with clap; outputs `prompts.jsonl` + `metadata.yaml` | **Reusable with adaptation.** Implements D1 step 3a (contrast prompt generation) in working form. Adaptation: extend `--emotions` CLI to support appraisal contrast labels (`high_norm_violation`/`low_norm_violation`) per [`PROJECT_PLAN.md` §6 Phase 5](../PROJECT_PLAN.md). The existing modes — especially Deflection (real vs displayed emotion) and Unexpressed — are richer than the typical contrast-pair scheme and may produce cleaner direction vectors. |
| [`emotion_prompts.md`](../../../../emotion_prompts.md) | System-prompt templates for each generation mode | **Reusable as-is.** ~55 KB of curated prompt-engineering work. Phase 5 should not redo this. |
| [`examples/cli/pipelines/emotion-vector.yaml`](../../../../examples/cli/pipelines/emotion-vector.yaml) | End-to-end YAML manifest for extract → steer with Python integration example | **Reusable as template.** Phase 6's manifest in [`PROJECT_PLAN.md` §5.3](../PROJECT_PLAN.md) maps closely; adopt this as a starting point and extend with appraisal nodes. |
| [`crates/core/src/nodes/llama_cpp/activation.rs`](../../../../crates/core/src/nodes/llama_cpp/activation.rs) | text → hidden state at chosen layers via `TensorCapture::for_layers` | **Already what we need for D1 step 2.** This is the "External: Feed prompts to LLM, capture activations" step from `emotion-vector.yaml:67`. The Candle pipeline assumes this is external; the llama-cpp path provides it natively. |
| [`crates/core/src/nodes/llama_cpp/steer.rs`](../../../../crates/core/src/nodes/llama_cpp/steer.rs) | Same shape as Candle's steering node, but on the llama.cpp path. Metadata-only stub at lines 308-313. | **Parallel/divergent.** See F4 below. |

### F4 — Architectural redundancy: Candle vs llama-cpp

Side-by-side comparison of the two implementations:

| Aspect | Candle (`emotion/steer.rs`) | llama-cpp (`llama_cpp/steer.rs`) |
|---|---|---|
| Receives vectors as `RuntimeData::Tensor` from pipeline | ✅ `register_vector` | ✅ `register_vector` |
| Per-session coefficient state | ✅ `SteeringState` | ✅ `SteeringState` |
| Runtime coefficient updates via JSON | ✅ `update_coefficients` | ✅ `update_coefficients` |
| Computes steering delta `Σ(coef × layer_norm × vec)` | ✅ via `compute_steering_delta` | ✅ inline in `compute_delta` |
| `max_coefficient` safety clamp | ✅ | ✅ |
| **Actual injection into model forward pass** | ❌ Blocked on Candle hooks | ⏸ Blocked on Phase 4 wiring (API exists per [Spike A](spike-a-binding.md)) |
| Underlying inference framework | Candle (Rust-native) | llama.cpp via `llama-cpp-4` |
| Currently runs an LLM | ❌ No (no model in the code path) | ⏸ Generation node exists, steer node calls into it (per `IMPLEMENTATION_PLAN.md` Phase 4) |

The Candle steer node is essentially a stub — it never invokes a Candle
model. It accepts tensors, computes deltas, emits metadata, but the
"steering" half of "extract and steer" is purely conceptual on the Candle
side until Candle gains forward-hook APIs.

The llama-cpp steer node has the same shape *and* sits next to a working
generation node *and* the binding has the runtime injection API now.
**The llama-cpp path is closer to functional steering by a wide margin.**

### F5 — Reuse opportunities mapped against PROJECT_PLAN §5.2

For each new component listed in [`PROJECT_PLAN.md` §5.2](../PROJECT_PLAN.md), what already exists?

| Planned new component | Existing thing | Reuse strategy |
|---|---|---|
| `WhisperEmbeddingExtractorNode` | None | Build new (Phase 3). |
| `ProsodyVadNode` | None | Build new (Phase 3). |
| `TextSentimentNode` | None | Build new (Phase 3). |
| `TaskEventNode` | None | Build new (Phase 3). |
| `AppraisalEngineNode` | None | Build new (Phase 1). |
| `AffectDynamicsNode` | None | Build new (Phase 1). |
| `RegulationPolicyNode` | None | Build new (Phase 1). |
| `SteeringRouterNode` | None | Build new (Phase 2). |
| Channel A (decoding control) | None | Phase 2. |
| Channel B (prompt aug) | None | Phase 2. |
| `ControlVectorBuffer` (Channel C/D) | None | Phase 4. |
| Calibration tooling (`tools/affect_calibration/`) | **`emotion-activation-gen` (CLI)** + **Candle `EmotionExtractorNode`** + **`vector_io.rs`** + **`emotion_prompts.md`** | **Adopt and extend.** See F6. |
| Simulator (`tools/affect_simulator/`) | None | Phase 1. |

So the **affect-pipeline layers (events/appraisal/dynamics/regulation/router)
are entirely greenfield**, but the **calibration toolchain has substantial
preexisting work** that the plan should explicitly absorb.

### F6 — Calibration pipeline reuse plan

Mapping our `IMPLEMENTATION_PLAN.md` §4 calibration steps to existing code:

| Calibration step (from IMPLEMENTATION_PLAN §4) | Existing code | Adapt? |
|---|---|---|
| §4.2 step 1: extract Whisper embeddings | None | New (Phase 3 + Phase 5). |
| §4.3 step 2: train VAD regressor | None | New (Phase 5). |
| §4.4 step 3a: generate contrast sentences | [`emotion-activation-gen`](../../../../examples/cli/emotion-activation-gen/) | **Adopt with extension.** Add `--axis valence/arousal/dominance/norm_violation/...` flags to generate appraisal-axis contrast pairs alongside existing emotion-name contrasts. |
| §4.4 step 3b: feed contrast sentences to LLM, capture hidden states | [`crates/core/src/nodes/llama_cpp/activation.rs`](../../../../crates/core/src/nodes/llama_cpp/activation.rs) (production runtime) — but for offline calibration we need a CLI variant. | Build a small `examples/cli/llm-activation-capture/` binary that takes a prompt JSONL and writes activations to a `.bin` file. Reuses `run_activation` from `inference.rs:305`. |
| §4.4 step 3c: compute mean-subtraction direction `D` | [`crates/candle-nodes/src/emotion/extract.rs`](../../../../crates/candle-nodes/src/emotion/extract.rs) (`ActivationAccumulator` + `compute_vectors`) and [`crates/candle-nodes/src/emotion/vector_io.rs`](../../../../crates/candle-nodes/src/emotion/vector_io.rs) | **Adopt as-is** for emotion-name vectors. **Extend** to support multi-axis VAD vectors: the same accumulator pattern with three labels (high_v, neutral, low_v) per axis, then linear least-squares to fit `D ∈ ℝ^{3 × d_llm}`. |
| §4.4 disk format for `D.npz` | [`vector_io.rs`](../../../../crates/candle-nodes/src/emotion/vector_io.rs) (4-byte length + JSON meta + f32 LE) | **Adopt the format**, optionally rename `.bin` → `.npz`-compatible if numpy interop matters; keep the binary format Phase 5 produces uniform with what the runtime nodes load. |
| §4.5 step 4: validate (judge LLM rubric) | None | New (Phase 5). |

### F7 — Math kernels worth promoting to a shared location

[`crates/candle-nodes/src/emotion/vector_io.rs`](../../../../crates/candle-nodes/src/emotion/vector_io.rs)
contains pure-Rust functions that are **not Candle-specific** and would
benefit both the Candle and llama-cpp paths if moved:

- `mean_vectors` (lines 168-205)
- `subtract_vectors` (lines 207-221)
- `l2_normalize` (lines 224-232)
- `cosine_similarity` (lines 235-252)
- `scale_for_steering` (lines 255-258)
- `compute_steering_delta` (lines 263-300)

The llama-cpp `steer.rs` has its own inlined `compute_steering_delta`
(lines 199-230) that's structurally identical. Consolidating these
prevents drift.

Recommendation: in Phase 4, lift the math kernels into either
`crates/core/src/nodes/affect/math.rs` (preferred, since the new affect
nodes will need them anyway) or a small `crates/affect-math` crate. Both
emotion paths re-export from there.

### F8 — Dependency graph (what would break if we consolidate)

Searched the workspace for imports of the Candle emotion module:

- `crates/candle-nodes/src/lib.rs` re-exports `pub mod emotion`.
- No other crate imports `candle_nodes::emotion::*` directly that I found
  in the audit.
- The `emotion-vector.yaml` example references node types `EmotionExtractorNode`
  and `EmotionSteeringNode` by name; the manifest loader resolves these at
  runtime from the registry. Renaming the Candle ones requires updating the
  YAML.
- `examples/cli/emotion-activation-gen` does not import the Candle code; it
  produces JSONL output that the Candle pipeline consumes via separate
  pipeline orchestration.

So the consolidation blast radius is small and well-bounded:

- `crates/candle-nodes/src/lib.rs` (re-exports)
- `crates/candle-nodes/src/emotion/mod.rs` (the renames)
- `examples/cli/pipelines/emotion-vector.yaml` (node_type strings)
- (No external Python/runtime consumer found)

### F9 — Tests (existing)

- [`crates/core/tests/emotion_extractor_test.rs`](../../../../crates/core/tests/emotion_extractor_test.rs)
  — tests the avatar text-tag node only (per F1).
- [`crates/core/tests/avatar_synthetic_emotion_e2e.rs`](../../../../crates/core/tests/avatar_synthetic_emotion_e2e.rs)
  — avatar pipeline e2e; uses the avatar `EmotionExtractorNode`.
- Unit tests inside `crates/candle-nodes/src/emotion/{extract,steer,vector_io,config}.rs`
  cover the activation pipeline; all functional today.

No e2e test exercising the full Candle extract→steer pipeline against a
real model was found. This is consistent with the "metadata-only" status.

---

## Architectural redundancy table

| Concern | Candle path | llama-cpp path | Plan recommendation |
|---|---|---|---|
| Activation extraction | External (assumed) | `activation.rs` (working) | **Use llama-cpp.** |
| Vector accumulation + mean-subtraction | `extract.rs::EmotionExtractorNode` (Candle) | None | **Adopt Candle's** (with rename per F1). It's framework-agnostic; nothing about it requires Candle. |
| Vector disk format | `vector_io.rs` | None | **Adopt Candle's.** |
| Math kernels (delta, normalize, cosine_sim) | `vector_io.rs` | inline in `steer.rs` | **Consolidate into `crates/core/src/nodes/affect/math.rs`** (Phase 4). |
| Steering control plane | Candle `steer.rs` | llama-cpp `steer.rs` | **Use llama-cpp.** Deprecate Candle's until Candle hooks land. |
| Runtime injection | Blocked on Candle hooks | `set_adapter_cvec` available | **Use llama-cpp.** |
| Prompt generation | `emotion-activation-gen` (CLI) | None | **Adopt and extend** for appraisal-axis contrasts. |
| Pipeline manifest example | `emotion-vector.yaml` | None | **Adopt as template** for Phase 6 manifest. |

---

## Reuse opportunities (PROJECT_PLAN §5.2 line items)

This is the explicit cross-check the plan called for. For every "new
component" in [`PROJECT_PLAN.md` §5.2](../PROJECT_PLAN.md), is there a
prior implementation?

- **Event extraction layer**: no existing scaffolding. Greenfield.
- **Appraisal layer**: no existing scaffolding. Greenfield.
- **Dynamics layer**: no existing scaffolding. Greenfield.
- **Regulation layer**: no existing scaffolding. Greenfield.
- **Steering router**: no existing scaffolding. Greenfield.
- **Channel A/B**: no existing scaffolding. Greenfield.
- **Channel C/D**: foundation exists in llama-cpp `steer.rs`; needs Phase 4
  wiring.
- **Calibration**:
  - Prompt generation → **adopt `emotion-activation-gen`**.
  - Activation capture → **use `LlamaCppActivationNode`** (already works).
  - Vector accumulation → **adopt Candle `EmotionExtractorNode`** (rename).
  - Disk format → **adopt Candle `vector_io.rs`**.
  - Math kernels → **promote to shared module**.
  - Pipeline manifest → **adopt `emotion-vector.yaml`** as template.
  - Judge-LLM validation → greenfield.

---

## Consolidation recommendations

### R1 — Rename the Candle `EmotionExtractorNode` (resolve naming collision)

**Before any new Phase 4 PR.** Rename:

- `crates/candle-nodes/src/emotion/extract.rs::EmotionExtractorNode`
  → `EmotionVectorExtractorNode`
- Update factory's `node_type()` string from `"EmotionExtractorNode"` to
  `"EmotionVectorExtractorNode"`.
- Update [`emotion-vector.yaml:45`](../../../../examples/cli/pipelines/emotion-vector.yaml#L45)
  accordingly.
- Update doc comments in [`mod.rs`](../../../../crates/candle-nodes/src/emotion/mod.rs)
  and [`steer.rs`](../../../../crates/candle-nodes/src/emotion/steer.rs)
  that reference the old name.

The avatar `EmotionExtractorNode` ([`crates/core/src/nodes/emotion_extractor.rs`](../../../../crates/core/src/nodes/emotion_extractor.rs))
keeps its name — it's the more user-facing of the two and was named first
per the avatar spec.

Estimated diff: ~20 lines across 4 files. No behavior change.

**Decision rationale**: the names collide on `node_type()` registration,
which is a runtime hazard. Resolving now (a small chore) prevents subtle
bugs when a future contributor wires both into a single registry. The
Candle name is the one to change because (a) it's less documented in
external specs, (b) the avatar one is referenced from
`docs/superpowers/specs/2026-04-27-live2d-audio2face-rvc-avatar-design.md`
which is harder to update.

### R2 — Steering: consolidate around llama-cpp; deprecate Candle's

Phase 4 wires `set_adapter_cvec` into [`crates/core/src/nodes/llama_cpp/steer.rs`](../../../../crates/core/src/nodes/llama_cpp/steer.rs).
At that point the llama-cpp path becomes functional; the Candle path
remains stub-only.

Recommendation:

- Mark `EmotionSteeringNode` (Candle) with a `#[deprecated]` attribute
  pointing to the llama-cpp version, with a deprecation message:
  *"Steering injection requires Candle forward hooks (not available);
  use LlamaCppSteerNode for runtime steering."*
- Keep the Candle code in tree — it's working metadata-mode logic and
  may become usable when Candle gains hooks. But discourage new pipelines
  from using it.
- Update [`emotion-vector.yaml`](../../../../examples/cli/pipelines/emotion-vector.yaml)
  to optionally route through `LlamaCppSteerNode` when the user wants
  runtime steering.

This is a **Phase 4** task, not Phase 0.

### R3 — Promote math kernels to a shared module

Phase 4 deliverable: create [`crates/core/src/nodes/affect/math.rs`](../../../../crates/core/src/nodes/affect/)
(or similar) and move:

- `mean_vectors`, `subtract_vectors`, `l2_normalize`, `cosine_similarity`,
  `scale_for_steering`, `compute_steering_delta`

from [`crates/candle-nodes/src/emotion/vector_io.rs`](../../../../crates/candle-nodes/src/emotion/vector_io.rs).
The Candle module re-exports from the new location for backwards compat
within the workspace; the llama-cpp `steer.rs` drops its inlined
`compute_steering_delta` and uses the shared one.

Vector I/O (`load_emotion_vector` / `save_emotion_vector`) can stay where
it is or move depending on whether it's a calibration-time tool (move to
`tools/affect_calibration/lib`) or a runtime concern (keep in core).

### R4 — Adopt `emotion-activation-gen` as Phase 5's contrast-prompt generator

Don't build a new prompt generator. Phase 5 extends the existing one:

- Add `--axis` CLI flag accepting `valence | arousal | dominance |
  norm_violation | goal_congruence | ...`. When set, instead of generating
  emotion-named pairs, generate appraisal-axis contrast pairs.
- Extend [`emotion_prompts.md`](../../../../emotion_prompts.md) with system
  prompts for each appraisal axis.
- Output remains `prompts.jsonl` + `metadata.yaml`; downstream consumers
  unchanged.

This recovers the prompt-engineering work already invested
(`emotion_prompts.md` is ~55 KB of curated templates).

### R5 — Adopt `vector_io.rs` disk format for calibration manifests

The 4-byte-prefix-plus-JSON-plus-f32 format is well-designed and already
implemented. Phase 5 calibration artifacts (`whisper_to_vad.onnx` is ONNX,
but `D ∈ ℝ^{3 × d_llm}` direction matrices) should use this format.

The `EmotionVectorMetadata` struct (config.rs:19-47) covers most of what
the Phase 5 calibration manifest in
[`IMPLEMENTATION_PLAN.md` §4.1](../IMPLEMENTATION_PLAN.md) calls for.
Extend with `axes: ["valence", "arousal", "dominance"]`, `calibration_id`,
and `model_hash` fields.

### R6 — Adopt `emotion-vector.yaml` as the Phase 6 manifest template

The structure is exactly right. Phase 6 extends with appraisal/dynamics/
regulation nodes upstream of the steering node, and replaces
`EmotionSteeringNode` with `LlamaCppSteerNode`.

---

## Open questions for owner discussion

These are issues the audit surfaced that should be confirmed with whoever
authored the existing emotion code before consolidation lands.

- **Q-B1**: Why was Candle chosen for the original emotion-vector work
  rather than llama-cpp, given that Candle lacks forward hooks? (Best
  guess: original author wanted Rust-native model inference; llama-cpp
  steering was perhaps not a near-term option in workspace at that time.)
- **Q-B2**: Is there an in-flight effort on Candle forward hooks
  (referenced in the doc comment as `TODO: candle-forward-hooks`)? If so,
  the Candle path could become viable later and the deprecation note in R2
  should be soft.
- **Q-B3**: Was `examples/cli/llm-activation-capture/` (referenced in §F6
  as a needed Phase 5 binary) ever discussed? It would replace the
  "External: Feed prompts to LLM, capture activations" step in
  `emotion-vector.yaml`.
- **Q-B4**: Is the `llm_feeling_weather` reference repo a hard upstream
  dependency, or just a methodological reference? If hard, Phase 5 should
  align its output formats with whatever `llm_feeling_weather` ingests.

---

## Decision gate

The plan declared Spike B's gate as "produces recommendations, not
consolidation work." Holding to that.

**Recommended follow-up tasks** (not Phase 0):

| When | Task |
|---|---|
| Phase 4 (first PR) | R1: rename Candle `EmotionExtractorNode` → `EmotionVectorExtractorNode`. |
| Phase 4 (during steering wiring) | R2: deprecate Candle steering node; R3: promote math kernels to shared module. |
| Phase 5 kickoff | R4: extend `emotion-activation-gen` with `--axis` flag; R5: adopt vector_io disk format for calibration manifests. |
| Phase 6 kickoff | R6: adopt `emotion-vector.yaml` as manifest template. |

Each becomes a sub-task line in the relevant phase's PR plan.

---

## Files referenced

**Pre-existing scaffolding (read-only in Phase 0)**:
- [`crates/candle-nodes/src/emotion/mod.rs`](../../../../crates/candle-nodes/src/emotion/mod.rs)
- [`crates/candle-nodes/src/emotion/config.rs`](../../../../crates/candle-nodes/src/emotion/config.rs)
- [`crates/candle-nodes/src/emotion/extract.rs`](../../../../crates/candle-nodes/src/emotion/extract.rs)
- [`crates/candle-nodes/src/emotion/steer.rs`](../../../../crates/candle-nodes/src/emotion/steer.rs)
- [`crates/candle-nodes/src/emotion/vector_io.rs`](../../../../crates/candle-nodes/src/emotion/vector_io.rs)
- [`crates/core/src/nodes/emotion_extractor.rs`](../../../../crates/core/src/nodes/emotion_extractor.rs) (avatar; namespace collision only)
- [`crates/core/src/nodes/llama_cpp/steer.rs`](../../../../crates/core/src/nodes/llama_cpp/steer.rs)
- [`crates/core/src/nodes/llama_cpp/activation.rs`](../../../../crates/core/src/nodes/llama_cpp/activation.rs)
- [`examples/cli/emotion-activation-gen/`](../../../../examples/cli/emotion-activation-gen/)
- [`examples/cli/pipelines/emotion-vector.yaml`](../../../../examples/cli/pipelines/emotion-vector.yaml)
- [`emotion_prompts.md`](../../../../emotion_prompts.md)

**Reference works cited by existing code**:
- Anthropic Transformer Circuits: https://transformer-circuits.pub/2026/emotions/index.html
- llm_feeling_weather: https://github.com/mezoistvan/llm_feeling_weather
