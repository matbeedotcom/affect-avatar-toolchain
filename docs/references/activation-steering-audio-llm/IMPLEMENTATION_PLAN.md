# Implementation Plan — Audio-Driven LLM Persona Steering

> Companion to [`DESIGN.md`](DESIGN.md). This document is the **how**, written
> at PR-level granularity: concrete function signatures, file-by-file changes,
> test names, error types, observability hooks, and review gates. Where
> `DESIGN.md` argues *that* a workstream is feasible, this document specifies
> exactly *what to write*.
>
> **Current binding**: `llama-cpp-4` (verified at
> [`inference.rs:27`](../../../crates/core/src/nodes/llama_cpp/inference.rs#L27)).
> Plan assumes this crate; substitute names if upstream changes.

---

## 0. Pre-work (week 0, before any PRs)

Three day-one spikes. Each is a single-developer task; all should be done
before opening Workstream-A PRs because they determine major architectural
choices.

### 0.1 Binding spike — does `llama-cpp-4` expose `llama_apply_adapter_cvec`?

**Goal**: produce a one-page memo answering: *"Can we call
`llama_apply_adapter_cvec` from Rust today, and if so what's the safe API
surface?"*

Steps:

1. `cargo doc --open -p llama-cpp-4` and grep the rustdoc for `cvec`,
   `adapter`, `control_vector`. Look for methods on `LlamaContext`.
2. If absent: check the crate's source on crates.io / GitHub for unsafe
   bindings — `llama-cpp-sys-4` likely re-exports the C symbol even if the
   safe wrapper isn't there.
3. If still absent: write a 30-line `unsafe extern "C"` shim referencing the
   symbol from `llama.h`. Document the buffer ownership rules
   (caller-owned, lifetime ≥ inference call).
4. Smoke-spike: a 50-line throwaway binary that loads any GGUF, applies a
   random `cvec`, generates 20 tokens, prints them. Confirms the call
   compiles, links, runs without crashing.

**Output**: `docs/references/activation-steering-audio-llm/notes/binding-spike.md`
documenting findings + the throwaway code.

**Decision gate**: If the spike works, Workstream A proceeds as planned. If
not, escalate before committing to the calendar in `DESIGN.md`.

### 0.2 Whisper encoder spike — can we extract encoder hidden states?

**Goal**: confirm we can get `encoder_output ∈ ℝ^{T × d_whisper}` from at
least one Whisper backend the runtime supports.

Steps:

1. Read [`whisper_transcription.py`](../../../clients/python/remotemedia/nodes/ml/whisper_transcription.py) —
   identify which Whisper library it imports (`faster-whisper`, `openai-whisper`, `transformers`).
2. For that library, find the API to access the encoder forward output. For
   `faster-whisper`: `WhisperModel.encode()` returns `Features`. For
   `openai-whisper`: `model.embed_audio(mel)` or `model.encoder(mel)`. For
   HuggingFace: `WhisperModel.encoder(input_features).last_hidden_state`.
3. Throwaway script: load a 5-second clip, run encoder, print
   `(T_frames, d_whisper)`.

**Output**: `notes/whisper-spike.md` with the working call and the shape.

**Decision gate**: If `faster-whisper` exposes it cleanly, Workstream C stays
in Python. If only `openai-whisper`/HuggingFace do, decide whether to switch
backends or run two Whisper instances (cheap; encoder-only forward is fast).

### 0.3 Dataset access — can we get MSP-Podcast?

**Goal**: lock down the D1 dataset before writing training code.

Steps:

1. Submit MSP-Podcast access request to UTD. Lead time is days to weeks.
2. While waiting: download IEMOCAP (signed agreement), CREMA-D (open),
   RAVDESS (open).
3. Write a license-summary table: `notes/datasets.md` listing each dataset's
   redistribution terms, label format, and speaker-disjoint train/test split
   recipe.

**Decision gate**: If MSP-Podcast access is blocked, fall back to IEMOCAP +
CREMA-D as primary; document expected distribution-shift risk.

---

## 1. Workstream A — Wire Steering Into Inference

This is the technical crux. Detailed because (a) it blocks everything, and
(b) it's the part most likely to surface unknowns.

### 1.1 Files affected

| Path | Change |
|---|---|
| [`crates/core/src/nodes/llama_cpp/inference.rs`](../../../crates/core/src/nodes/llama_cpp/inference.rs) | Add `ControlVectorState` and `apply_control_vector` helper. Modify `run_generation_with_ctx_inner` to accept optional cvec. |
| [`crates/core/src/nodes/llama_cpp/cvec.rs`](../../../crates/core/src/nodes/llama_cpp/cvec.rs) | **New.** Buffer formatting helpers. |
| [`crates/core/src/nodes/llama_cpp/mod.rs`](../../../crates/core/src/nodes/llama_cpp/mod.rs) | Export new module. |
| [`crates/core/src/nodes/llama_cpp/steer.rs`](../../../crates/core/src/nodes/llama_cpp/steer.rs) | Replace metadata-mode stub at lines 308-313, 233-263 with real injection. |
| [`crates/core/src/nodes/llama_cpp/generation.rs`](../../../crates/core/src/nodes/llama_cpp/generation.rs) | Accept optional cvec from caller (passthrough to inference). |
| [`crates/core/examples/llama_cpp_steer_smoke.rs`](../../../crates/core/examples/) | **New.** Coef=0 vs coef=2.0 differential test. |
| [`crates/core/Cargo.toml`](../../../crates/core/Cargo.toml) | If shim required: add `llama-cpp-sys-4` dep. |

### 1.2 New module: `cvec.rs`

```rust
//! Control-vector buffer formatting for llama.cpp activation steering.
//!
//! `llama_apply_adapter_cvec` expects a flat `n_layers × n_embd` f32 buffer
//! laid out as `buffer[layer_idx * n_embd + embd_idx]`. Layers outside the
//! steering range are zero-padded.

use crate::error::Error;

/// A control vector ready to hand to llama.cpp.
///
/// Owns the f32 buffer; lifetime must extend through the inference call.
#[derive(Debug, Clone)]
pub struct ControlVectorBuffer {
    /// Flat buffer: `data[layer * n_embd + d]`.
    pub data: Vec<f32>,
    /// Embedding dimension (per layer).
    pub n_embd: usize,
    /// Total layers in the model.
    pub n_layers: usize,
    /// First layer to apply (inclusive).
    pub il_start: usize,
    /// Last layer to apply (exclusive).
    pub il_end: usize,
}

impl ControlVectorBuffer {
    /// Build a buffer that steers a single layer.
    ///
    /// `vec` must have length exactly `n_embd`. All other layers are zeroed.
    pub fn single_layer(
        layer: usize,
        vec: &[f32],
        n_embd: usize,
        n_layers: usize,
    ) -> Result<Self, Error> {
        if vec.len() != n_embd {
            return Err(Error::Execution(format!(
                "control vector length {} != n_embd {}",
                vec.len(),
                n_embd
            )));
        }
        if layer >= n_layers {
            return Err(Error::Execution(format!(
                "layer {} >= n_layers {}",
                layer, n_layers
            )));
        }
        let mut data = vec![0.0f32; n_embd * n_layers];
        let start = layer * n_embd;
        data[start..start + n_embd].copy_from_slice(vec);
        Ok(Self {
            data,
            n_embd,
            n_layers,
            il_start: layer,
            il_end: layer + 1,
        })
    }

    /// Build a buffer that steers a contiguous layer range with the same vector.
    pub fn layer_range(
        il_start: usize,
        il_end: usize,
        vec: &[f32],
        n_embd: usize,
        n_layers: usize,
    ) -> Result<Self, Error> { /* ... */ }

    /// Zero buffer — equivalent to "no steering."
    pub fn zero(n_embd: usize, n_layers: usize) -> Self {
        Self {
            data: vec![0.0; n_embd * n_layers],
            n_embd,
            n_layers,
            il_start: 0,
            il_end: 0,
        }
    }
}
```

Tests in the same file:

- `test_single_layer_buffer_layout` — assert byte layout matches expectation.
- `test_single_layer_zero_pads_others` — only requested layer is non-zero.
- `test_length_mismatch_returns_err`
- `test_layer_out_of_range_returns_err`

### 1.3 Inference-side wiring

Modify [`inference.rs`](../../../crates/core/src/nodes/llama_cpp/inference.rs):

```rust
/// Apply a control vector to a llama context.
///
/// Safe wrapper around `llama_apply_adapter_cvec`. Pass `None` to clear.
/// Buffer must outlive the next `decode` call.
pub fn apply_control_vector(
    ctx: &mut llama_cpp_4::context::LlamaContext,
    cvec: Option<&ControlVectorBuffer>,
) -> Result<(), Error> {
    // Implementation depends on §0.1 spike outcome:
    //   (a) Safe wrapper exposed by llama-cpp-4: just call it.
    //   (b) Only sys crate exposes it: small unsafe block, here.
    todo!("filled in by Workstream A1 PR")
}

fn run_generation_with_ctx_inner(
    model: &llama_cpp_4::model::LlamaModel,
    ctx: &mut llama_cpp_4::context::LlamaContext,
    config: &LlamaCppGenerationConfig,
    prompt: &str,
    cvec: Option<&ControlVectorBuffer>,   // NEW
) -> Result<GenerationResult, Error> {
    // existing tokenization, batch setup ...

    // NEW: apply cvec before prefill.
    apply_control_vector(ctx, cvec)?;

    ctx.decode(&mut batch)
        .map_err(|e| Error::Execution(format!("Prefill decode failed: {}", e)))?;

    // existing decode loop ...
}
```

Public-API change: `run_generation_with_ctx` gains a `cvec` parameter.
Default callers pass `None`; backwards compatible.

### 1.4 Steer-node-side wiring

In [`steer.rs`](../../../crates/core/src/nodes/llama_cpp/steer.rs):

```rust
async fn generate_steered(
    &self,
    prompt: &str,
    session_id: &str,
) -> Result<(Vec<String>, f32, HashMap<String, f32>), Error> {
    let (delta, coefficients) = self.compute_delta(session_id).await?;
    let delta_norm: f32 = delta.iter().map(|x| x * x).sum::<f32>().sqrt();

    {
        let mut sessions = self.sessions.write().await;
        if let Some(state) = sessions.get_mut(session_id) {
            state.last_delta_norm = Some(delta_norm);
        }
    }

    // NEW: build cvec buffer.
    let cvec = if delta_norm > 1e-6 {
        Some(ControlVectorBuffer::single_layer(
            self.config.layer,
            &delta,
            self.hidden_size,    // cached at initialize() — see §1.5
            self.n_layers,       // ditto
        )?)
    } else {
        None
    };

    let gen_config = self.config.generation.clone();
    let prompt = prompt.to_string();

    let chunks = tokio::task::spawn_blocking(move || {
        super::inference::run_generation_with_cvec(&gen_config, &prompt, cvec.as_ref())
    })
    .await
    .map_err(|e| Error::Execution(format!("Task join failed: {}", e)))??;

    Ok((chunks, delta_norm, coefficients))
}
```

Update the metadata emission (lines 336-346, 375-385): set
`"applied": true` and `"mode": "kv_inject"` when `cvec.is_some()`.

Remove the warning at lines 308-313. Replace with:

```rust
info!(
    node = "llama-cpp-steer",
    layer = self.config.layer,
    n_vectors = self.config.vectors.len(),
    "LlamaCppSteerNode active — control vector injection enabled"
);
```

### 1.5 Initialize-time metadata caching

`LlamaCppSteerNode` needs `hidden_size` (a.k.a. `n_embd`) and `n_layers` to
build cvec buffers. Today these aren't available; they live on the model.

Options, ranked:

1. **Load model in `initialize()`, cache** `hidden_size: usize, n_layers: usize`
   on the node. Requires holding the model on a worker thread (matches the
   existing pattern for `LlamaCppGenerationNode`).
2. **Probe by reading the GGUF metadata** without a full load. `gguf-rs`
   crate. Lighter but adds a dep and a code path.

Choose (1). It also sets up Workstream A2 (persistent context across turns)
naturally.

### 1.6 Smoke test: `llama_cpp_steer_smoke.rs`

Modeled after [`llama_cpp_chat_smoke.rs`](../../../crates/core/examples/llama_cpp_chat_smoke.rs):

```rust
//! End-to-end smoke for steering: same prompt with coef=0 and coef=2.0
//! must produce different outputs.

#[cfg(feature = "llama-cpp")]
fn main() {
    // 1. Build two contrast sentences known to encode opposite affect:
    //    "I love this so much, you're amazing!"  vs.
    //    "I hate this. Leave me alone."
    // 2. Run both through LlamaCppActivationNode → get h_pos, h_neg.
    // 3. Build steering vector v = h_pos - h_neg, normalize.
    // 4. Run LlamaCppSteerNode with the same prompt three times:
    //      a. coef = 0.0    (must match unsteered baseline)
    //      b. coef = +2.0   (output should lean positive)
    //      c. coef = -2.0   (output should lean negative)
    // 5. Assertions:
    //      - All three produce non-empty text.
    //      - Outputs (a) and (b) differ (Levenshtein > threshold).
    //      - Outputs (a) and (c) differ.
    //      - Token-level avg log-prob doesn't collapse (sanity: no gibberish).
    todo!()
}
```

This test is the gate for declaring Workstream A done. Without it nothing
downstream is provable.

### 1.7 PR shape for Workstream A

Three PRs:

- **A1**: cvec module + binding wiring (`cvec.rs`, `apply_control_vector`,
  `run_generation_with_ctx_inner` accepts `Option<&ControlVectorBuffer>`).
  No behavior change for existing callers; passes existing smoke test.
- **A2**: persistent model holding in `LlamaCppSteerNode::initialize`,
  `hidden_size`/`n_layers` cached. No behavior change.
- **A3**: wire `compute_delta` → `apply_control_vector`, flip the warning,
  add the smoke test. Behavior change visible.

Each PR < 500 lines diff including tests. A1 is the riskiest review gate.

---

## 2. Workstream B — User/Assistant Phase Separation

### 2.1 Phase boundary in the existing inference loop

From [`inference.rs:117-195`](../../../crates/core/src/nodes/llama_cpp/inference.rs#L117):

- **Prefill phase**: lines 125-134. All prompt tokens batched, single
  `ctx.decode(&mut batch)` call.
- **Decode phase**: lines 167-195. Loop, one token per `ctx.decode`.

The transition is between line 134 and line 167. That's the call site for
swapping cvecs.

### 2.2 Schema changes

[`config.rs`](../../../crates/core/src/nodes/llama_cpp/config.rs) — add:

```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SteerRole {
    #[default]
    Both,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppSteerVector {
    pub label: String,
    pub coefficient: f32,
    #[serde(default)]
    pub role: SteerRole,
}
```

Backwards compat: `role` defaults to `Both`, so existing manifests behave
identically to today.

### 2.3 Activation node tags its outputs

[`activation.rs:158-169`](../../../crates/core/src/nodes/llama_cpp/activation.rs#L158)
already adds an `emotion` field; add a `role` field driven by config:

```rust
// In LlamaCppActivationConfig:
#[serde(default)]
pub default_role: SteerRole,

// In activation.rs metadata emission:
let mut metadata = serde_json::json!({
    "model": self.config.model_path,
    "layer": capture.layer,
    "hidden_size": capture.hidden_size,
    "pooling": format!("{:?}", self.config.pooling),
    "normalized": self.config.normalize,
    "raw_norm": capture.raw_norm,
    "role": self.config.default_role,   // NEW
});
```

### 2.4 Steer node groups vectors by role

In [`steer.rs`](../../../crates/core/src/nodes/llama_cpp/steer.rs), replace
`compute_delta` with two:

```rust
async fn compute_delta_for_role(
    &self,
    session_id: &str,
    role: SteerRole,
) -> Result<(Vec<f32>, HashMap<String, f32>), Error>;
```

Filter: include vectors where `loaded.role == role || loaded.role == Both`.

### 2.5 Phased inference

New helper in `inference.rs`:

```rust
pub fn run_generation_with_phase_cvec(
    config: &LlamaCppGenerationConfig,
    prompt: &str,
    cvec_user: Option<&ControlVectorBuffer>,       // applied during prefill
    cvec_assistant: Option<&ControlVectorBuffer>,  // applied during decode
) -> Result<GenerationResult, Error> {
    // ... boilerplate ...

    apply_control_vector(ctx, cvec_user)?;
    ctx.decode(&mut batch)?;          // prefill — user-role active

    apply_control_vector(ctx, cvec_assistant)?;
    // decode loop — assistant-role active
    for _ in 0..config.max_tokens { /* ... */ }
}
```

Subtle point: the chat template ([`inference.rs:106`](../../../crates/core/src/nodes/llama_cpp/inference.rs#L106))
puts the *system prompt* and *user message* both inside the prefill batch. The
user-role cvec applies to both. This is fine — system prompts are typically
neutral, and the user-emotion vector lightly biases the system tokens too,
which is acceptable. If finer-grained scoping is needed later, split the
batch into a system-only first decode + user-only second decode.

### 2.6 Smoke test extension

Extend `llama_cpp_steer_smoke.rs`:

- Run three configurations on the same prompt:
  - User-only vector (coef positive)
  - Assistant-only vector (coef positive)
  - Both vectors
- Inspect outputs qualitatively. Hard to assert programmatically without a
  judge LLM; log to stderr for human review and assert "all three differ
  pairwise."

### 2.7 PR shape for Workstream B

Single PR after A3 lands. ~300 lines.

---

## 3. Workstream C — Whisper Encoder Embedding Extraction

### 3.1 Decision tree from §0.2 spike

| Spike outcome | Path |
|---|---|
| `faster-whisper` exposes encoder cleanly | Extend Python node. ~1 day. |
| Only HuggingFace `transformers` exposes it | Add new Python node using transformers; existing transcription node unchanged. ~3 days. |
| Want Rust path for perf | Add `whisper-cpp-rs` binding work; ~5 days. |

Default to (1).

### 3.2 Multi-output Whisper node

Modify [`whisper_transcription.py`](../../../clients/python/remotemedia/nodes/ml/whisper_transcription.py)
to emit a second output channel:

```python
class WhisperTranscriptionNode(Node):
    def __init__(self, ..., emit_encoder_embedding: bool = False, encoder_layer: int = -1):
        ...
        self.emit_encoder_embedding = emit_encoder_embedding
        self.encoder_layer = encoder_layer  # -1 means final layer

    async def process(self, audio_chunk):
        # existing transcription path
        text = await self._transcribe(audio_chunk)
        yield RuntimeData.text(text, channel="text")

        if self.emit_encoder_embedding:
            embed = self._encode(audio_chunk, layer=self.encoder_layer)
            # shape: (T_frames, d_whisper)
            yield RuntimeData.tensor(
                embed.tobytes(),
                shape=list(embed.shape),
                dtype=DTYPE_F32,
                metadata={"role": "user", "source": "whisper_encoder"},
                channel="encoder_embedding",
            )
```

Per CLAUDE.md spec 021, multi-output streaming requires the factory's
`is_multi_output_streaming() -> true` and the connection schema to support
named output ports.

### 3.3 New Rust node: `MeanPoolNode`

`crates/core/src/nodes/audio/mean_pool.rs` (or under a new `tensor_ops/`
namespace):

```rust
pub struct MeanPoolNode { node_id: String, axis: usize }

#[async_trait::async_trait]
impl AsyncStreamingNode for MeanPoolNode {
    async fn process(&self, data: RuntimeData) -> Result<RuntimeData, Error> {
        match data {
            RuntimeData::Tensor { data, shape, dtype, metadata } => {
                if dtype != 0 { return Err(...); }
                let pooled = mean_pool_axis(&data, &shape, self.axis)?;
                let new_shape = remove_axis(&shape, self.axis);
                Ok(RuntimeData::Tensor { data: pooled, shape: new_shape, dtype, metadata })
            }
            other => Err(Error::Execution(format!("MeanPoolNode expects Tensor, got {}", other.data_type()))),
        }
    }
    // ... boilerplate
}
```

Tests: `[2, 4]` shape pooled along axis 0 → `[4]`; preserves metadata.

### 3.4 Capability declarations

Per CLAUDE.md spec 023, declare on the new/modified factories:

```rust
fn capability_behavior(&self) -> CapabilityBehavior { CapabilityBehavior::Static }
fn media_capabilities(&self, _: &Value) -> Option<MediaCapabilities> {
    Some(/* tensor shape constraints */)
}
```

### 3.5 PR shape for Workstream C

Two PRs (run in parallel with A):

- **C1**: extend Whisper Python node with optional encoder output.
- **C2**: add `MeanPoolNode` in Rust + capability declarations.

---

## 4. Workstream D1 — Calibration (offline ML)

This workstream produces *artifacts*, not runtime code. Lives in a new
directory: `tools/affect_calibration/`.

### 4.1 Directory structure

```
tools/affect_calibration/
├── README.md                       # how to reproduce a calibration run
├── pyproject.toml                  # uv/poetry; pins torch, onnx, faster-whisper
├── data/                           # gitignored
│   ├── emovoice_db/
│   ├── crema_d/
│   ├── jl_corpus/
│   ├── trait_descriptions.json     # 3 axis descriptions (V/A/D), hand-authored
│   └── contrast_pairs.jsonl        # auto-generated by 02b script
├── scripts/
│   ├── 01_extract_whisper_embeddings.py
│   ├── 02_train_vad_regressor.py
│   ├── 02b_generate_contrast_pairs.py
│   ├── 03_extract_llm_directions.py
│   ├── 04_validate_pipeline.py
│   └── lib/
│       ├── datasets.py             # VAD-labeled audio loaders
│       ├── emotion_to_vad.py       # categorical → VAD mapping helpers
│       ├── persona_pipeline.py     # paper-aligned generator + extractor helpers
│       └── eval.py                 # rubric / judge-LLM scoring
└── artifacts/                      # output; checked-in metadata, .gitignored binaries
    ├── whisper_to_vad.onnx
    ├── llm_directions/
    │   └── lfm2-audio-1.5b_layer9.npz
    ├── manifest.json               # which model/layer/hashes
    └── calibration_report.md       # per-axis RMSE, sample outputs
```

`artifacts/manifest.json` schema (target-LLM metadata reflects Spike G pivot
to LFM2-Audio-1.5B MLX; see [`notes/spike-g-mlx-target-llm.md`](notes/spike-g-mlx-target-llm.md)):

```json
{
  "calibration_id": "2026-05-15-lfm2-audio-1.5b-l9",
  "whisper": { "model": "large-v3", "encoder_layer": -2, "d_whisper": 1280 },
  "llm": {
    "model_id": "mlx-community/LFM2.5-Audio-1.5B-4bit",
    "backend": "mlx",
    "n_embd": 2048,
    "n_layers": 16,
    "steering_layer": 9
  },
  "datasets": [{ "name": "msp_podcast_v1.10", "rows_train": 12345, "rows_val": 1234 }],
  "regressor": {
    "rmse_v": 0.18, "rmse_a": 0.21, "rmse_d": 0.24,
    "params_path": "whisper_to_vad.onnx"
  },
  "directions": {
    "source": "persona_vectors_pipeline",
    "elicitation_model": "claude-3-5-sonnet@20250620",
    "n_pos_prompts_per_axis": 5,
    "n_neg_prompts_per_axis": 5,
    "n_questions_per_axis": 40,
    "n_contrast_pairs_total": 1200,
    "pooling": "response_mean",
    "params_path": "llm_directions/lfm2-audio-1.5b_layer9.npz"
  },
  "validation": {
    "judge_model": "claude-3-5-sonnet@20250620",
    "judge_rubric_scale": "0-100 net (positive pole − negative pole)",
    "n_probe_pairs_per_axis": 50,
    "judge_net_score_unsteered": { "v": 0, "a": 0, "d": 0 },
    "judge_net_score_steered_pos": { "v": 18, "a": 22, "d": 16 },
    "judge_net_score_steered_neg": { "v": -19, "a": -20, "d": -17 },
    "human_agreement_spotcheck_n": 30
  }
}
```

The runtime nodes load `whisper_to_vad.onnx` and `llm_directions/*.npz` and
reject mismatched calibration IDs at `initialize()`.

### 4.2 Step 1: extract Whisper embeddings (`01_extract_whisper_embeddings.py`)

```python
# Inputs:  data/{dataset}/clips/*.wav, data/{dataset}/labels.csv (with VAD)
# Outputs: data/{dataset}/whisper_embeds.npz  (key: clip_id → mean-pooled [d_whisper])

for clip in dataset:
    audio = load_audio(clip.path, sr=16000)
    mel = log_mel(audio)
    encoder_states = whisper_model.encoder(mel, output_hidden_states=True)
    layer_output = encoder_states.hidden_states[args.encoder_layer]
    pooled = layer_output.mean(dim=1)  # [d_whisper]
    save(clip.id, pooled)
```

Scope: ~100 LoC. Dataset-loader logic dominates.

Validation: spot-check 5 clips manually; assert non-NaN, non-zero.

### 4.3 Step 2: train VAD regressor (`02_train_vad_regressor.py`)

```python
# Architecture: MLP(d_whisper → 64 → 3), tanh on output to bound [-1, 1].
# Loss: MSE per axis, summed.
# Optimizer: AdamW, lr=1e-3, weight_decay=1e-4.
# Scheduler: cosine over 50 epochs.
# Speaker-disjoint train/val split: hold out 20% of speakers, not 20% of clips.

class VADHead(nn.Module):
    def __init__(self, d_whisper: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_whisper, 64),
            nn.GELU(),
            nn.Linear(64, 3),
            nn.Tanh(),
        )
    def forward(self, x): return self.net(x)

# Training: standard.
# Export: torch → ONNX with dynamic batch axis.
torch.onnx.export(model, dummy_input, "whisper_to_vad.onnx",
                  input_names=["whisper_embed"], output_names=["vad"],
                  dynamic_axes={"whisper_embed": {0: "batch"}, "vad": {0: "batch"}})
```

Reporting: per-axis RMSE on held-out speakers; speaker-disjoint must be
enforced by code, not by trust.

### 4.4 Step 3a: generate contrast pairs (`02b_generate_contrast_pairs.py`)

Per the persona-vectors pipeline (Chen et al. 2025,
[arXiv:2507.21509](https://arxiv.org/abs/2507.21509); adoption recorded in
[`notes/spike-f-persona-vectors-pipeline.md`](notes/spike-f-persona-vectors-pipeline.md)),
the contrast set is generated automatically from three trait descriptions
— one per VAD axis — rather than hand-authored.

**Trait modeling: axis-as-trait (pinned).** Each VAD axis is treated as a
single bipolar trait. "Positive system prompts" induce the high pole of
the axis (high V = pleasant, high A = aroused/energized, high D =
dominant); "negative system prompts" induce the low pole. Each direction
`v_axis = mean(high) − mean(low)` projects bipolarly. We produce **3
directions, one per axis** — not per-quadrant ("happy", "angry",
"sad", …) and not via NRC-VAD-style category-to-VAD projection.

`data/trait_descriptions.json` schema:

```json
{
  "valence":   { "positive_pole": "...", "negative_pole": "...", "context": "..." },
  "arousal":   { "positive_pole": "...", "negative_pole": "...", "context": "..." },
  "dominance": { "positive_pole": "...", "negative_pole": "...", "context": "..." }
}
```

Each `*_pole` field is a 2–4 sentence behavioral description that the
elicitation LLM uses to draft system prompts and questions. The `context`
field is shared boilerplate ("the LLM is a conversational assistant",
deployment constraints, etc.) that anchors prompts to our domain.

```python
# Inputs:  data/trait_descriptions.json — 3 paragraphs (one per V/A/D axis),
#          each describing the positive/negative pole behaviorally.
#          $ELICITATION_LLM — large model (Claude/GPT/Qwen) used once, offline.
# Outputs: data/contrast_pairs.jsonl — one row per (axis, pole, system_prompt,
#          question). For each axis, the full cartesian product per pole:
#            5 positive system prompts × 40 questions = 200 positive rows
#            5 negative system prompts × 40 questions = 200 negative rows
#            ⇒ 400 rows per axis. Three axes ⇒ 1200 rows total.

for axis in ["valence", "arousal", "dominance"]:
    desc = trait_descriptions[axis]
    pos_prompts = elicit("generate 5 system prompts that induce the POSITIVE pole", desc)
    neg_prompts = elicit("generate 5 system prompts that induce the NEGATIVE pole", desc)
    questions   = elicit("generate 40 questions designed to surface this trait", desc)
    for sp in pos_prompts:
        for q in questions:
            emit({"axis": axis, "pole": "positive", "system_prompt": sp, "question": q})
    for sp in neg_prompts:
        for q in questions:
            emit({"axis": axis, "pole": "negative", "system_prompt": sp, "question": q})
```

Scope: ~150 LoC including JSON I/O and provider abstraction. Output is
deterministic given a seeded elicitation LLM; archive both the trait
descriptions and generated pairs so a calibration is fully reproducible
from `data/trait_descriptions.json` alone.

### 4.5 Step 3b: extract LLM directions (`03_extract_llm_directions.py`)

```python
# Inputs:  data/contrast_pairs.jsonl produced by 02b above.
#          MLX-loaded LFM2-Audio target via lib/mlx_target.MLXTargetLLM.
# Outputs: artifacts/llm_directions/{model}_layer{N}.npz with array `D` of
#          shape (3, d_llm).

target_llm = MLXTargetLLM("mlx-community/LFM2.5-Audio-1.5B-4bit")  # n_embd=2048
D = np.zeros((3, target_llm.n_embd), dtype=np.float32)
for axis_idx, axis in enumerate([Axis.VALENCE, Axis.AROUSAL, Axis.DOMINANCE]):
    D[axis_idx] = extract_axis_direction(
        contrast_pairs=load_contrast_pairs("data/contrast_pairs.jsonl"),
        target_llm=target_llm,
        layer=L,
        axis=axis,
        pooling="response_mean",
    )

np.savez(f"lfm2-audio-1.5b_layer{L}.npz", D=D, n_embd=target_llm.n_embd, layer=L)
```

`MLXTargetLLM.generate_and_pool_response(system, question, layer, "response_mean")`
returns the mean of the residual-stream hidden states at layer L over the
*response* tokens only (excluding system+user prompt context). Mean-difference
per axis replaces the previous least-squares `V @ D = H` formulation; the
new contrast-pair file format makes mean-difference the natural choice
(the file no longer carries continuous sentence-level VAD scalars).

**Pooling strategy (pinned):** response-token mean. The MLX provider
generates a response under each (system_prompt, question) pair via
`generate_sequential` (text-only path; `temperature=0, top_k=1` for
determinism) and pools residual hiddens at layer L over the response-token
positions only. Prompt-token pooling and last-token pooling are not used.
Rationale: Chen et al. 2025 App. A.3 finds response-token averaging
produces stronger directions across all tested traits.

**Multi-layer incremental steering** (`v_ℓ − v_{ℓ−1}`, paper App. J.3) —
deferred to v2.

Calibration-time choice: which `layer L`? LFM2-Audio-1.5B has 16 layers
and `d_model=2048` (verified by Spike G). Default `L=9` (~60% depth);
sweep `[6, 9, 12]` if the default layer's directions fail the §4.6
validation gate.

The MLX provider lives at
[`tools/affect_calibration/scripts/lib/mlx_target.py`](../../../tools/affect_calibration/scripts/lib/mlx_target.py).
It splices a wrapper module into `model.lfm.layers[L]` to capture
post-block residual at layer L (the verified pattern from
[`notes/spike-g-mlx-extraction.py`](notes/spike-g-mlx-extraction.py);
patching `instance.__call__` is silently broken on MLX `nn.Module`).
~150 LoC including the lazy import surface that keeps `--dry-run`
free of the mlx dependency. The legacy llama.cpp / GGUF activation
extraction path is retained as fallback at
[`crates/core/src/nodes/llama_cpp/activation.rs`](../../../crates/core/src/nodes/llama_cpp/activation.rs)
but is not on the Phase 5 critical path.

### 4.6 Step 4: validation (`04_validate_pipeline.py`)

Adopts the LLM-judge evaluator pattern of Chen et al. 2025 App. B
(94.7% agreement with human raters in their study). For each VAD axis,
the judge scores a response on a 0–100 scale measuring expression of the
positive pole.

```python
# For 50 (audio, target_VAD) probe pairs per axis:
#   1. audio → whisper_embeds → g → predicted user_VAD
#   2. user_VAD → mirror policy → target_VAD
#   3. target_VAD → D · target_VAD → steering vector
#   4. Run frozen LLM with vs without steering on a fixed prompt set.
#   5. Judge LLM rates each response on 0–100 per axis using a rubric of
#      the form: "Rate how strongly this response expresses {axis-positive
#      pole description from data/trait_descriptions.json}. 0 = not at all,
#      100 = strongly. Output an integer."
#      A separate call rates the negative pole; net score = pos - neg.
#   6. Aggregate per axis: mean(steered) − mean(unsteered).
#
# Pass criterion (per axis):
#   - mean(steered@α=+1) − mean(unsteered) ≥ 15 points (0–100 net scale)
#   - mean(steered@α=−1) − mean(unsteered) ≤ −15 points
#   - sign of net-score change matches sign of α (monotonicity sanity check)
```

The judge LLM should be a different model than the target LLM (avoid
self-grading bias). Recommended pairing: target = Qwen3-27B / Llama-3-70B,
judge = Claude-3.5-Sonnet or GPT-4o. The judge rubric and the trait
descriptions used for scoring are pinned in `manifest.json` so a
re-validation run is fully reproducible.

Cheap fallback for internal go/no-go: human-rate 30 random samples per
axis on the same 0–100 rubric. Useful as a one-time sanity check that
the LLM-judge agrees with humans at our distribution; not a replacement
for the automated metric.

### 4.7 PR shape for Workstream D1

D1 doesn't need to land as a single PR; it's a tool, not a code change.
Per Spike G (target-LLM pivot to LFM2-Audio-1.5B MLX), D1.0's original
"Qwen3-7B + Rust extractor" gate has been retired — the gating concern
moves to R10 (multimodal-LLM extension), and the gating run *is* the
production calibration on the actual deployment target. Recommended:

- **D1.0** (skeleton + dry-run): scaffold `tools/affect_calibration/`,
  land `02b → 03 → 04` driver skeletons with `--dry-run`, plus
  `lib/mlx_target.py` and `lib/emotion_to_vad.py` helpers. Verifies
  the I/O contract end-to-end without compute. **Done** (commits
  `94fe72a`, `ed06ac2`).
- **D1.1**: step 1 (Whisper extractor real path) + dataset loaders
  for EmoVoice-DB / CREMA-D / JL-Corpus.
- **D1.2**: step 2 (VAD regressor real training) with hyperparameter
  sweep results + ONNX export.
- **D1.3**: step 3a real path (elicitation LLM provider for 02b) +
  step 3b real-mode invocation on the full 1200-row contrast set
  via the MLX provider. **Layer sweep here**: extract D at layers
  [6, 9, 12]; pick the one that passes §4.6 with the largest gap.
- **D1.4**: step 4 with a reproducible report. The cross-model judge
  (Claude-3.5-Sonnet target ≠ judge) implementation lands here.
  **Pass on this PR is the equivalent of R10 mitigation passing**;
  if any axis fails the ≥15-point gap with all swept layers,
  pause Phase 5 and revisit trait descriptions or extraction modality.

All four PRs land calibration *capability*; the trained artifacts themselves
ship out-of-band (cloud bucket, Hugging Face Hub, internal artifact store).
Don't commit `.onnx` / `.npz` to the repo.

---

## 5. Workstream D2 — Runtime Projection Nodes

### 5.1 New nodes

| Node | Path | Language |
|---|---|---|
| `AffectProjectionNode` | `clients/python/remotemedia/nodes/affect/projection.py` | Python (multiprocess) |
| `AffectToSteeringNode` | `crates/core/src/nodes/affect/to_steering.rs` | Rust |

### 5.2 `AffectProjectionNode`

```python
class AffectProjectionNode(MultiprocessNode):
    """Projects Whisper embeddings to VAD coordinates.

    Input:  RuntimeData.tensor (shape=[d_whisper], dtype=f32)
    Output: RuntimeData.tensor (shape=[3], dtype=f32, metadata={role, axes, calibration_id})
    """

    def __init__(self, regressor_path: str, ema_alpha: float = 0.0,
                 calibration_id: str = ""):
        self.session = onnxruntime.InferenceSession(regressor_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.ema_alpha = ema_alpha
        self.ema_state: Optional[np.ndarray] = None
        self.calibration_id = calibration_id

    async def process(self, data: RuntimeData) -> RuntimeData:
        embed = np.frombuffer(data.tensor_bytes, dtype=np.float32)
        embed = embed.reshape(1, -1)
        vad = self.session.run(["vad"], {"whisper_embed": embed})[0][0]  # (3,)

        if self.ema_alpha > 0:
            if self.ema_state is None:
                self.ema_state = vad
            else:
                self.ema_state = self.ema_alpha * self.ema_state + (1 - self.ema_alpha) * vad
            vad_out = self.ema_state
        else:
            vad_out = vad

        return RuntimeData.tensor(
            vad_out.tobytes(),
            shape=[3],
            dtype=DTYPE_F32,
            metadata={
                "role": "user",
                "axes": ["valence", "arousal", "dominance"],
                "calibration_id": self.calibration_id,
                "ema_alpha": self.ema_alpha,
            },
        )
```

Tests:

- `test_known_input_produces_known_output` — golden test against committed
  fixture.
- `test_ema_smooths_step_change` — feed step input, assert output ramps.
- `test_calibration_id_propagates_to_metadata`.

### 5.3 `AffectToSteeringNode`

```rust
pub struct AffectToSteeringNode {
    node_id: String,
    directions: Array2<f32>,   // shape [3, d_llm], from .npz
    layer: usize,
    role: SteerRole,
    calibration_id: String,
    n_embd: usize,
}

#[async_trait::async_trait]
impl AsyncStreamingNode for AffectToSteeringNode {
    async fn process(&self, data: RuntimeData) -> Result<RuntimeData, Error> {
        let vad = parse_vad_tensor(&data)?;              // [f32; 3]

        // direction_vector = V · D  where V is (1, 3), D is (3, d_llm)
        let mut steering = vec![0.0f32; self.n_embd];
        for dim in 0..self.n_embd {
            for axis in 0..3 {
                steering[dim] += vad[axis] * self.directions[[axis, dim]];
            }
        }

        let bytes: Vec<u8> = steering.iter().flat_map(|&x| x.to_le_bytes()).collect();

        Ok(RuntimeData::Tensor {
            data: bytes,
            shape: vec![self.n_embd as i32],
            dtype: 0,
            metadata: Some(serde_json::json!({
                "role": format!("{:?}", self.role).to_lowercase(),
                "layer": self.layer,
                "label": "vad_projection",
                "calibration_id": self.calibration_id,
                "vad": vad,
            })),
        })
    }
}
```

The output tensor is a drop-in for `LlamaCppSteerNode` — it goes through the
existing `register_vector` path with no node-side changes.

### 5.4 Calibration-ID guard

When `LlamaCppSteerNode` registers a vector, it should warn (not error) on
calibration-ID mismatch:

```rust
async fn register_vector(&self, tensor: &RuntimeData) -> Result<(), Error> {
    let incoming_cal = tensor_metadata_string(tensor, "calibration_id");
    if let Some(expected) = &self.config.expected_calibration_id {
        if incoming_cal.as_deref() != Some(expected) {
            warn!(
                expected = expected,
                got = ?incoming_cal,
                "Calibration ID mismatch — vector may steer in unexpected direction"
            );
        }
    }
    // ... existing path
}
```

### 5.5 PR shape for Workstream D2

- **D2.1**: `AffectToSteeringNode` (Rust) + tests with synthetic D matrix.
- **D2.2**: `AffectProjectionNode` (Python) + ONNX runtime integration.
- **D2.3**: end-to-end manifest test that wires C → D2 → A. Uses fixture
  artifacts so it doesn't require D1 to be complete.

---

## 6. Workstream E — Persona Policy & Observability

### 6.1 `PersonaPolicyNode`

`crates/core/src/nodes/affect/persona.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaConfig {
    /// Anchor target VAD point: where the persona "wants to be."
    pub target_vad: [f32; 3],
    /// 0.0 = pure persona; 1.0 = pure user mirror; negative = counter-regulate.
    #[serde(default)]
    pub mirror_strength: f32,
    /// Per-axis output clamps applied after mirroring.
    #[serde(default = "default_clamp")]
    pub clamp: [(f32, f32); 3],
}

fn default_clamp() -> [(f32, f32); 3] { [(-1.0, 1.0); 3] }

pub struct PersonaPolicyNode { config: PersonaConfig }

impl PersonaPolicyNode {
    fn compute_assistant_target(&self, user_vad: [f32; 3]) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        for i in 0..3 {
            let raw = self.config.target_vad[i]
                + self.config.mirror_strength * (user_vad[i] - self.config.target_vad[i]);
            out[i] = raw.max(self.config.clamp[i].0).min(self.config.clamp[i].1);
        }
        out
    }
}
```

The node accepts `RuntimeData::Tensor[3]` (user VAD), emits
`RuntimeData::Tensor[3]` (assistant target VAD), tagged with
`metadata.role = "assistant"`.

### 6.2 Observability

Per-turn structured logging at the steering node:

```rust
info!(
    target: "affect.steering",
    session_id = %session_id,
    user_vad = ?user_vad,
    target_vad = ?target_vad,
    coef_user = %coef_user,
    coef_assistant = %coef_assistant,
    delta_norm_user = %delta_norm_user,
    delta_norm_assistant = %delta_norm_assistant,
    layer = self.config.layer,
    "Affect steering applied"
);
```

This becomes the operator's tuning surface. Without it, persona behavior is
opaque.

### 6.3 PR shape for Workstream E

Single PR. ~250 lines.

---

## 7. Cross-Cutting Concerns

### 7.1 Error taxonomy

Add one variant to `crate::Error`:

```rust
pub enum Error {
    // ... existing ...
    SteeringConfig(String),  // calibration mismatch, invalid VAD, invalid layer
}
```

Use cases:

- `LlamaCppSteerNode::initialize` — model too small for declared layer.
- `AffectToSteeringNode::initialize` — D matrix dimension mismatch.
- `AffectProjectionNode::initialize` — ONNX session load failure.

### 7.2 Capability declarations (CLAUDE.md spec 023)

Each new/modified node declares:

| Node | Inputs | Outputs |
|---|---|---|
| `WhisperTranscriptionNode` (modified) | `audio(sample_rate=16000, channels=1, format=F32)` | `text` + `tensor[T, d_whisper]` |
| `MeanPoolNode` | `tensor` (any shape) | `tensor` (axis removed) |
| `AffectProjectionNode` | `tensor[d_whisper]` | `tensor[3]` |
| `PersonaPolicyNode` | `tensor[3]` | `tensor[3]` |
| `AffectToSteeringNode` | `tensor[3]` | `tensor[d_llm]` |
| `LlamaCppSteerNode` (modified) | `tensor[d_llm]` + text | text |

`d_whisper` and `d_llm` are static-known after `initialize()`. The capability
resolver (spec 023) needs these threaded through; static integers if known
ahead, otherwise `Configured`.

### 7.3 Backwards compatibility

Every change must be opt-in:

- `LlamaCppSteerVector::role` defaults to `Both` → existing manifests unchanged.
- `WhisperTranscriptionNode::emit_encoder_embedding` defaults to `false` →
  existing pipelines unchanged.
- `LlamaCppSteerNode` with no registered vectors and `coef≈0` → behaves as
  generation node.

Existing examples (`llama_cpp_chat_smoke.rs`) must continue passing
unmodified through every PR.

### 7.4 Testing strategy

| Test type | Where | Triggered by |
|---|---|---|
| Unit | `crates/core/src/nodes/llama_cpp/{cvec,steer,activation}.rs` | `cargo test -p remotemedia-core` |
| Smoke (no model) | Existing test modules | Fast CI |
| Smoke (with model) | `crates/core/examples/llama_cpp_*_smoke.rs` | Manual / nightly with `LLAMA_TEST_MODEL=...` |
| Integration | `crates/ui/e2e/` Playwright | UI changes only |
| Calibration validation | `tools/affect_calibration/scripts/04_validate_pipeline.py` | Per calibration release |

Smoke tests with models gate Workstream A and B. Without them, "it compiles"
isn't enough.

### 7.5 Performance budget

Per-turn added cost vs. unsteered baseline:

| Operation | Cost |
|---|---|
| Whisper encoder embedding (already computed; tap is free) | 0 ms |
| Mean pool over T frames | < 1 ms |
| ONNX forward (`whisper → VAD`) on CPU | 1-5 ms |
| Linear `D @ vad` (3 × d_llm matmul) | < 1 ms |
| `llama_apply_adapter_cvec` | model-dependent; budget 5-10 ms |
| EMA smoothing | < 0.1 ms |

Total budget: **< 20 ms per turn**, dwarfed by LLM generation latency.

If the ONNX forward is hot-path: move regressor to GPU via ORT CUDA provider,
or port to Rust via `tract`/`candle` for fully Rust pipeline.

### 7.6 Rollback strategy

Each workstream lands behind a feature flag-equivalent: an opt-in config field
or a node that only activates when wired into the manifest. Rolling back a
buggy steering implementation requires only a manifest change (drop the
steering node from `connections`); no code revert needed.

---

## 8. PR Sequence & Review Gates

```
Pre-work (week 0)
    ├── A.0  Binding spike memo                       [docs only, fast review]
    ├── C.0  Whisper spike memo                       [docs only, fast review]
    └── D1.0 Dataset access memo                      [docs only, fast review]

Phase 1 — Steering works (weeks 1-2)
    ├── A.1  cvec module + binding wiring             [must pass existing tests]
    ├── A.2  Persistent model in steer node           [must pass existing tests]
    ├── A.3  End-to-end steering smoke test           [GATE: steering verifiable]
    └── B    Role separation                          [GATE: phase scoping verifiable]

Phase 2 — Audio plumbing (week 2-3, parallel to Phase 1)
    ├── C.1  Whisper encoder output                   [GATE: embedding tensor flows]
    └── C.2  MeanPoolNode                             [GATE: pooled tensor flows]

Phase 3 — Calibration capability (weeks 2-4)
    ├── D1.1 Tools skeleton + extract embeddings
    ├── D1.2 Train VAD regressor + report
    ├── D1.3 Extract LLM directions + tool
    └── D1.4 Validation pipeline                      [GATE: judge score delta ≥ +1.0]

Phase 4 — Runtime projection (week 4-5)
    ├── D2.1 AffectToSteeringNode (Rust)
    ├── D2.2 AffectProjectionNode (Python)
    └── D2.3 End-to-end manifest test                 [GATE: fixture round-trip]

Phase 5 — Operator surface (week 5)
    └── E    Persona policy + observability           [GATE: tunable demo]

Demo (end of week 5)
    └── Same audio, two persona configs, audibly different responses.
```

Review gates (in addition to passing tests):

- After A.3: a senior reviewer must confirm coef-positive vs coef-negative
  outputs differ qualitatively, not just numerically.
- After D1.4: the calibration report (per-axis RMSE + judge score delta)
  must be reviewed.
- After E: the persona-config knobs must be exercised live by someone other
  than the implementer.

---

## 9. Validation Milestones

You know it works when:

1. **A.3 passes**: `cargo run --example llama_cpp_steer_smoke --features llama-cpp-cuda`
   prints three different responses for `coef ∈ {0, +2, -2}` on the same
   prompt, all coherent English.

2. **B passes**: same example with `--user-only` vs `--assistant-only` flags
   produces three different responses; `--both` differs from each.

3. **C passes**: a manifest with
   `mic → whisper(emit_encoder=true) → mean_pool → log` prints
   `tensor[1280]` to stderr per audio chunk.

4. **D1.4 passes**: `tools/affect_calibration/scripts/04_validate_pipeline.py
   --probe-set probe_v1.jsonl --judge claude-haiku-4-5`
   reports `avg score(steered) − avg score(unsteered) ≥ 1.0`.

5. **D2 passes**: end-to-end manifest pipeline runs with fixture artifacts;
   per-turn log shows non-zero `delta_norm_user` and `delta_norm_assistant`.

6. **E passes**: switching `persona.target_vad` from `(0.6, 0.2, 0.3)` to
   `(-0.4, 0.7, -0.2)` audibly changes the assistant's tone on the same
   user audio.

7. **Demo**: a single prompt audio file ("can you help me with this?") spoken
   in two emotional registers (calm vs agitated), feeding two parallel pipeline
   instances configured with the same persona, produces two responses
   distinguishable by a human listener as "the model heard the user
   differently."

Milestone 7 is the executive demo; it's the only one stakeholders see, and
the only one any of the others exists to enable.

---

## 10. Out-of-Scope (explicitly)

These are noted to prevent scope creep:

- **Affect-aware TTS**: out of scope for this plan. The LLM's output is text;
  the existing TTS speaks it. Future work, see DESIGN.md §10 Q6.
- **Multi-user / per-speaker calibration**: a single `g.onnx` for all
  speakers. Per-speaker fine-tuning is future work.
- **Online adaptation**: `g.onnx` is frozen at deployment. Updating it from
  user feedback during a session is future work.
- **>3 affect dimensions**: VAD only. Sarcasm/hesitation/certainty axes are
  future work (DESIGN.md §10 Q3).
- **End-to-end backprop into `g`**: only the linear-probe and contrastive
  approaches are in this plan. Backprop-through-frozen-LLM is future work
  (DESIGN.md §10 Q5).

---

## 11. Open Implementation Questions

These are flagged as decisions to be made *during* implementation, not
upfront:

- **IQ1**: Does `llama-cpp-4` need a version bump for cvec support? Resolve
  in §0.1 spike.
- **IQ2**: Where does the persistent model live when `LlamaCppSteerNode` and
  `LlamaCppGenerationNode` are both in the same pipeline? Today they each
  load their own; for steering they should share. Defer until A.2 reveals
  whether the existing pattern is too costly.
- **IQ3**: How are calibration artifacts distributed? S3? HF Hub? Internal
  artifact store? Affects D2's path resolution. Defer to D1.4.
- **IQ4**: Should `MeanPoolNode` be `mean_pool` or general `tensor_op` family
  (`mean_pool`, `attention_pool`, `last_token`)? Defer until C.2; build
  whichever is needed first.
- **IQ5**: Per-session vs per-turn EMA reset for VAD smoothing? If session
  outlives many turns, smoothing across turns is the more useful behavior;
  but session boundary detection happens at the runtime level. Defer to D2.2.

Each open question gets a short ADR (architecture decision record) appended
to this file when resolved.
