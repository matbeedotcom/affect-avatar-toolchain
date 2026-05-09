# Affect-avatar — remotemedia-sdk integration

How the listener-face pipeline composes into a remotemedia-sdk
runtime. Companion to:
- [`STAGE1_VAE_PLAN.md`](STAGE1_VAE_PLAN.md) — Stage-1 AE/VAE history and dashboard.
- [`LISTENER_MODE_PLAN.md`](LISTENER_MODE_PLAN.md) — channel-split, masking, smoothing rationale.

---

## 1. Two listener paths, picking one

The toolchain produces two viable inference architectures for a
listener face. They differ in cost, determinism, and naturalism:

| path | conditioning | model | output character |
|---|---|---|---|
| **A. Masked DiT** | user audio + user-VAD | AE v3 + DiT v3 (~37 M params) | richer micro-motion, may show residual lip-sync correlation in upper face |
| **B. VAD anchor lookup** *(default)* | user-VAD only | 21-anchor RBF over MEAD aggregates (~5 KB lookup) | deterministic, no audio path, pure affect-driven |

**For the SDK integration we recommend B as the default** because:
- Inference is a 21-row matrix multiply per frame — sub-millisecond on
  any device, zero allocation per call.
- No audio path → categorically impossible to lip-sync to user speech.
- Anchors are a ~5 KB asset — bundleable into the binary or shipped as
  a sidecar `.npz`.
- Failure modes are obvious and debuggable (each anchor is a known
  emotion×intensity blend).

Path A remains useful for "expressive speaker" mode (the assistant is
talking) where DiT's audio-driven micro-motion is desirable. Both
paths share the same `.glb` and ARKit map.

The rest of this doc assumes Path B unless noted.

---

## 2. End-to-end pipeline

```
┌──────────────┐   16 kHz mono f32   ┌──────────────────┐
│ MicInputNode │────────────────────▶│ WhisperHidden-   │
└──────────────┘                     │  States (turbo)  │
                                     └────────┬─────────┘
                                              │ (T_audio, 1280) fp16
                                              ▼
                                     ┌──────────────────┐
                                     │ WhisperToVAD-    │
                                     │  Regressor       │   tools/affect_calibration
                                     │  (.onnx)         │   already produces this
                                     └────────┬─────────┘
                                              │ per-frame (V, A, D, intensity)
                                              ▼
                                     ┌──────────────────┐
                                     │  VadToBlendshape │   ← THIS DOC's main artefact
                                     │  (RBF lookup)    │
                                     └────────┬─────────┘
                                              │ (T, 54) MEAD blendshape
                                              ▼
                                     ┌──────────────────┐
                                     │ ListenerSmoother │   ← σ=5 frames on mouth
                                     │ + ChannelMask    │   ← LISTENER_SPEECH_ONLY zero
                                     └────────┬─────────┘
                                              │ (T, 52) ARKit, listener-shaped
                                              ▼
                                     ┌──────────────────┐
                                     │ AvatarRenderNode │   crates/core/src/nodes/cc_render
                                     │  (existing Bevy) │
                                     └──────────────────┘
```

The four new components introduced by this toolchain are:
- **WhisperToVADRegressor** — already exists in `tools/affect_calibration`.
  Needs one onnxruntime-Python node wrapper (multiprocess Python node
  per [CLAUDE.md `multiprocess` pattern](../../CLAUDE.md)).
- **VadToBlendshape** — implemented today in
  [`scripts/lib/vad_to_blendshape.py`](scripts/lib/vad_to_blendshape.py).
  Pure-numpy, ~150 lines. Trivial to port to native Rust if hot path
  warrants it.
- **ListenerSmoother** — fold into `VadToBlendshape` or a thin wrapper.
  EMA on VAD before lookup + σ=5-frame Gaussian on the surviving
  affect-mouth channels (already implemented in
  [`05_sample_smoke.py`](scripts/05_sample_smoke.py)).
- **ChannelMask** — applies `LISTENER_SPEECH_ONLY_CHANNELS`
  (19 indices) defined in
  [`scripts/lib/vae.py`](scripts/lib/vae.py).

---

## 3. Asset bundle

What the runtime needs at start-up:

| file | size | purpose |
|---|---:|---|
| `vad_anchors_delta.npz` | ~5 KB | 21 emotion×intensity deltas + neutral baseline. Read once, cached in RAM. |
| `whisper_to_vad.onnx` | ~10–50 MB | The `affect_calibration` regressor. Loaded by onnxruntime. |
| `beautiful_assistant.smpl_aligned.glb` | 130 MB | Avatar mesh — already shipped via git-lfs in `avatars/processed/`. |
| `beautiful_assistant.arkit_map.resolved.json` | ~30 KB | ARKit-name → CC5 morph-name map. Used by the existing render node. |

Only the first two are new; the GLB + map already exist in the SDK
tree (see commit `bf76171`).

---

## 4. Node spec — `VadToFaceNode` (proposed)

Single-input, single-output Python multiprocess node per the
[CLAUDE.md multiprocess pattern](../../CLAUDE.md).

### Inputs

```yaml
input:
  type: tensor
  shape: (B, 4)          # batched (V, A, D, intensity)
  dtype: f32
  rate: 30 Hz            # matches blendshape output frame rate
```

### Outputs

```yaml
output:
  type: blendshapes
  vocabulary: arkit_52
  shape: (B, 52)
  dtype: f32
  rate: 30 Hz
```

### Constructor params

```yaml
params:
  anchors_path: "artifacts/vad_anchors_delta.npz"
  tau: 0.25                          # RBF bandwidth
  intensity_floor: 0.20              # min intensity multiplier
  add_baseline: false                # emit pure delta (recommended for listener)
  listener_mask: true                # zero out 19 speech channels
  mouth_smooth_sigma: 5.0            # frames; 0 disables
  vad_smooth_alpha: 0.2              # EMA on VAD trajectory; 1.0 disables
```

### Capabilities

```rust
// Per spec 023 capability declaration
input_caps:  tensor(shape=(_, 4), dtype=f32, rate=30Hz)
output_caps: blendshapes(vocabulary=arkit_52, rate=30Hz, channels=52)
behavior:    Static
```

### State (per session)

```python
class VadToFaceNodeState:
    interp: VadToBlendshape           # loaded once at init
    vad_ema: np.ndarray | None = None # (4,) running EMA, None = first frame
    mouth_history: deque              # ring buffer for σ=5 Gaussian (~15 frames)
```

### Per-frame logic

```python
# pseudo-code
def process(self, vad: np.ndarray) -> np.ndarray:
    # 1. Smooth incoming VAD with EMA (suppresses regressor jitter
    #    cheaper than smoothing on output)
    if self.vad_ema is None:
        self.vad_ema = vad
    else:
        a = self.cfg.vad_smooth_alpha
        self.vad_ema = a * vad + (1 - a) * self.vad_ema

    # 2. RBF lookup: VAD → (54,) MEAD blendshape (deltas only)
    bs54 = self.interp.lookup(self.vad_ema[:3], intensity=self.vad_ema[3])

    # 3. Listener channel mask: zero speech-coupled indices
    if self.cfg.listener_mask:
        bs54[LISTENER_SPEECH_ONLY_CHANNELS] = 0.0

    # 4. Map MEAD-54 → ARKit-52
    bs52 = mead_to_arkit(bs54)

    # 5. Mouth smoothing: σ=5 Gaussian over a ring buffer
    if self.cfg.mouth_smooth_sigma > 0:
        bs52 = self._gaussian_smooth_mouth(bs52)

    return bs52
```

Cost per call: ~10 µs RBF + ~5 µs map + ~5 µs smooth = **~20 µs/frame**
on M1 Pro. At 30 Hz that's <0.1 % CPU.

---

## 5. Manifest example

```yaml
nodes:
  - id: mic
    node_type: MicInputNode
    params: { sample_rate: 16000, channels: 1, format: F32 }

  - id: whisper_hidden
    node_type: WhisperHiddenStatesNode    # exists; emits encoder layer −2
    params: { model: "openai/whisper-large-v3-turbo" }

  - id: vad_regressor
    node_type: WhisperToVADNode           # new; wraps affect_calibration .onnx
    executor: multiprocess
    params: { onnx_path: "tools/affect_calibration/artifacts/whisper_to_vad.onnx" }

  - id: face
    node_type: VadToFaceNode              # new; this doc
    executor: multiprocess
    params:
      anchors_path: "tools/affect_avatar/artifacts/vad_anchors_delta.npz"
      tau: 0.25
      listener_mask: true
      mouth_smooth_sigma: 5.0

  - id: render
    node_type: BevyAvatarRenderNode       # exists; consumes ARKit blendshapes
    params:
      glb: "avatars/processed/beautiful_assistant.smpl_aligned.glb"
      arkit_map: "avatars/beautiful_assistant.arkit_map.resolved.json"
      focus: face
      fit_frame: true

connections:
  - { from: mic, to: whisper_hidden }
  - { from: whisper_hidden, to: vad_regressor }
  - { from: vad_regressor, to: face }
  - { from: face, to: render }
```

The pipeline graph executes in topological order. End-to-end latency
budget on M1 Pro:

| stage | budget | notes |
|---|---:|---|
| MicInputNode | <1 ms | hardware buffer |
| WhisperHiddenStatesNode | ~80 ms | dominated by Whisper encoder; 1.5-second stream window typical |
| WhisperToVADNode | <5 ms | small MLP onnx |
| VadToFaceNode | <0.1 ms | RBF lookup |
| BevyAvatarRenderNode | ~10 ms/frame | shared with the rest of the rendering graph |

**Total path latency** ~100 ms — dominated by Whisper. The face
itself adds <15 ms; the perception of "robotic delay" is set by the
Whisper window, not this toolchain.

---

## 6. What's built today vs what's needed

### ✅ Built

- AE v3 + DiT v3 (Path A speaker mode) — ckpts in
  `tools/affect_avatar/artifacts/`.
- VAD anchor table — `artifacts/vad_anchors_delta.npz`,
  21 anchors, peak-aggregated, neutral-baseline-subtracted.
- VAD→blendshape interpolator —
  [`scripts/lib/vad_to_blendshape.py`](scripts/lib/vad_to_blendshape.py).
- Listener channel-mask + mouth-smoothing logic —
  [`scripts/lib/vae.py`](scripts/lib/vae.py) (`LISTENER_SPEECH_ONLY_CHANNELS`,
  `LISTENER_AFFECT_MOUTH_CHANNELS`) + applied in
  [`scripts/05_sample_smoke.py`](scripts/05_sample_smoke.py).
- Re-baked GLB with mouth morphs preserved
  (`avatars/processed/beautiful_assistant.smpl_aligned.glb`).
- ARKit map (`avatars/beautiful_assistant.arkit_map.resolved.json`).
- Bevy renderer with `--focus=face` framing.
- Sanity animatic of all 21 anchors —
  `tools/affect_avatar/artifacts/anchors_animatic_v3.mp4`.

### ⏳ To build for SDK integration

1. **`WhisperToVADNode`** — Python multiprocess node that wraps the
   `affect_calibration` .onnx. ~80 lines. Per the CLAUDE.md multiprocess
   pattern. Estimated: half a day including manifest tests.
2. **`VadToFaceNode`** — Python multiprocess node wrapping
   `VadToBlendshape`. ~120 lines. Estimated: half a day.
3. **`BevyAvatarRenderNode` updates** — already accepts ARKit blendshape
   input (per
   `crates/core/src/nodes/cc_render/`); needs the new defaults
   (`focus=face`, GLB path, map path) wired through `params`. Estimated:
   half a day.
4. **Manifest plumbing** — write a sample manifest exercising the four
   nodes end-to-end; add to the SDK example set. Estimated: half a day.
5. **Integration tests** — record a short user-speech sample, run it
   through the manifest, snapshot the resulting blendshape JSONL +
   compare against an offline-computed reference. Estimated: 1 day.

**Total: 2.5–3 days** to ship a working listener face in the SDK
runtime.

### Optional / later

- **Native Rust port** of `VadToBlendshape`. Would remove a Python
  process from the hot path and shave ~5 ms IPC overhead. Not on the
  critical path.
- **`ExpressiveSpeakerNode`** (Path A): wraps AE v3 + DiT v3 for the
  *assistant talking* mode. Heavier (~37 M params, MPS/CUDA dependency).
  Belongs in a follow-up scope.
- **L1 retrain**: if listener-mode users still see micro-leakage in the
  upper face, retrain the AE+DiT with the channel mask applied at
  training time (per
  [`LISTENER_MODE_PLAN.md` §4.L1](LISTENER_MODE_PLAN.md)). Defer until
  L0 production telemetry shows it's needed.
- **L2 reactive dynamics**: rule-based brow-flash on phrase emphasis,
  empathic mirror-smile ramp on sustained user-positive-valence. Per
  [`LISTENER_MODE_PLAN.md` §4.L2](LISTENER_MODE_PLAN.md).

---

## 7. Open questions for runtime owners

- **Mode switching**: how does the runtime decide listener-mode vs
  expressive-speaker-mode? Suggest a single bus input from the
  conversation orchestrator (`is_user_speaking: bool`). When true,
  route VAD→Face; when false, route audio→DiT→Face.
- **VAD source during silence**: the regressor will produce noisy VAD
  on silent / non-speech audio. Either gate via the existing
  `silero_vad` node (use it to suppress audio→VAD when no voice is
  detected) or add a deadband in `VadToFaceNode` (return rest pose
  when |VAD| < threshold).
- **Per-user calibration**: should the system learn a per-user offset
  on the VAD regressor (some speakers are systemically "more aroused"
  than the regressor's training distribution suggests)? This is a
  product decision — likely future scope.
- **Cross-platform deployment**: numpy + onnxruntime cover Linux/Mac.
  Windows + WASM deployments need either a Rust port of both nodes or
  a WASM build of the Python multiprocess executor. Out of scope for
  the initial integration.

---

## 8. Recipe — produce all assets from a fresh clone

```bash
cd tools/affect_avatar

# 1. Train Stage-1 AE (~50 min)  — only needed for Path A.
.venv/bin/python scripts/09_train_ae.py --data-dir data --output artifacts/blendshape_ae_v3.pt

# 2. Train Stage-2 DiT (~2.5 hr) — only needed for Path A.
.venv/bin/python scripts/04_train_dit.py \
  --vae artifacts/blendshape_ae_v3.pt --data-dir data \
  --whisper-cache artifacts/whisper_cache \
  --epochs 30 --batch-size 32 --crops-per-clip 2 --no-zscore-latents \
  --output artifacts/blendshape_dit_v3.pt

# 3. Build VAD anchor table for Path B (~1 min) — sole asset for Path B.
.venv/bin/python scripts/10_build_vad_anchors.py \
  --data-dir data --output artifacts/vad_anchors_delta.npz

# 4. Re-bake avatar GLB if it doesn't already have mouth morphs (~5 min).
git lfs pull --include="avatars/beautiful_assistant.Fbx"
scripts/avatars/onboard_avatar.sh \
  ../../avatars/beautiful_assistant.Fbx \
  --arkit-map ../../avatars/beautiful_assistant.arkit_map.resolved.json \
  --out beautiful_assistant --no-render --no-diagnose

# 5. Sanity render — all 21 anchors as an animatic (~30 sec render).
target/release/examples/affect_face_smoke \
  --glb avatars/processed/beautiful_assistant.smpl_aligned.glb \
  --map avatars/beautiful_assistant.arkit_map.resolved.json \
  --face artifacts/anchors_animatic_v3.jsonl \
  --out artifacts/anchors_animatic_v3.mp4 --fps 30
```

Step 3 is reproducible from the source via
[`scripts/10_build_vad_anchors.py`](scripts/10_build_vad_anchors.py)
(introduced in this commit). Step 5 expects the JSONL produced by
running 10_build_vad_anchors's animatic-helper sibling — see the
inline animatic build that produced
`artifacts/anchors_animatic_v3.jsonl` (codify if needed in a
follow-up).
