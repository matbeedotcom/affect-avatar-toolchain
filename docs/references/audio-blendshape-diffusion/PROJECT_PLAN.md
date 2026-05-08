# Audio → Blendshape Latent-Diffusion Model — PROJECT PLAN

**Working title**: `affect_avatar` (sibling to `affect_calibration`).

**Status**: B0 executed (RAVDESS pilot, 2026-05-06) — accepted with
caveats; B1 unblocked. See
[`notes/spike-b0-blendshape-extraction.md`](notes/spike-b0-blendshape-extraction.md).

**Owner**: @mathieugosbee.

**One-line pitch**: Train a small diffusion model that maps speech audio +
optional V/A/D affect coordinates to ARKit-52 blendshape trajectories, so
the affective-agent runtime can drive a rigged avatar with both lip-sync
and emotional facial expression — locally, in real-time, without
NVIDIA Audio2Face.

---

## 0. Goal & non-goals

### Goal

A drop-in replacement / complement for Audio2Face that:

1. Takes 16-kHz mono speech audio as input.
2. Optionally takes V/A/D affect coordinates (`(v, a, d) ∈ [-1, +1]^3`)
   as conditioning, so the same audio can be rendered neutral, joyful,
   tense, or somber on demand.
3. Outputs a per-frame ARKit-flavored blendshape trajectory (51 floats,
   ~30 fps; ARKit-52 minus `tongueOut` per Spike B0 finding M-B0-1)
   suitable for driving Live2D, MetaHuman, or any rig with the standard
   ARKit vocabulary. The runtime exposer remaps to 52-D by writing
   `tongueOut=0` so downstream rigs see the full ARKit shape.
4. Runs in <real-time on a single Apple-Silicon GPU (target: <30 ms per
   1-second window on M-series; chunked-streamable for live use).
5. Exports cleanly to ONNX for deployment as a `BlendshapeDiffusionNode`
   inside the existing RemoteMedia pipeline.

### Non-goals

- **Full-face video synthesis.** Output is blendshapes only; rendering is
  someone else's job (Live2D engine, Unreal, three.js, etc.).
- **Body / hand motion.** Out of scope. Body motion is the
  `affect_motion` follow-on (separate plan, not this one).
- **End-to-end training of the audio frontend.** We re-use the Whisper
  encoder from `affect_calibration` (Spike C); no audio-feature
  pretraining.
- **Sub-frame precision lip-sync metrics on par with Audio2Face.** Audio2Face
  is the result of NVIDIA's research budget. We aim for *credible*, not
  *state-of-the-art*. See §6 evaluation thresholds.

---

## 1. Architectural overview

```
                              affect coords (v,a,d) ∈ [-1, +1]^3
                                          │
                                          ▼
                                    ┌──────────┐
                                    │  MLP     │   small (16→128) projection
                                    │  embed   │   for cross-attention conditioning
                                    └────┬─────┘
                                         │
   audio (16 kHz mono)                   │
        │                                ▼
        ▼                          ┌───────────┐
  ┌─────────────┐  hidden states   │           │
  │  Whisper    │ ──(B, T, 1280)──▶│  DiT-1D   │── denoising U-Net over latents
  │  encoder    │                  │  denoiser │   8 blocks × d_model=384
  │ (large-v3-  │                  │           │   ~30-50M params
  │  turbo)     │                  └─────┬─────┘
  └─────────────┘                        │
                                         ▼
                                   z₀ ∈ ℝ^{T_lat × d_lat}
                                         │
                                         ▼
                                  ┌────────────┐
                                  │  VAE       │
                                  │  decoder   │   small Conv1d stack
                                  │ (frozen at │   ~500k params
                                  │  inference)│
                                  └─────┬──────┘
                                        │
                                        ▼
                              blendshapes ∈ ℝ^{T_frames × 52}
                              (ARKit-52, ~30 fps)
```

Three components, two training stages:

| Component | Params | Trained when |
|---|---:|---|
| Whisper encoder | frozen | never (re-used from `affect_calibration` Spike C) |
| VAE encoder + decoder | ~500k | Stage 1 (recon-only, blendshapes ↔ latents) |
| DiT-1D denoiser | ~30-50M | Stage 2 (audio → latent diffusion, with V/A/D CFG) |

Latent dimension target: `d_lat = 16`, `T_lat = T_frames / 4` (4× temporal
downsample inside VAE). Picked so the diffusion target is small enough to
train on consumer GPUs but rich enough to retain phoneme timing.

---

## 2. Data plan

### Datasets — primary path: pre-extracted MEAD_3D

The audio + blendshape ground-truth pipeline pivots away from
"raw video → MediaPipe extraction" to a HuggingFace mirror of MEAD that
**already ships extracted blendshape trajectories paired with audio**:

[`BarryFutureman/MEAD_3D`](https://huggingface.co/datasets/BarryFutureman/MEAD_3D).

| Field | Value |
|---|---|
| Rows | 1,312 audio + blendshape pairs |
| Schema | `(audio @ 16 kHz, actions: list<list<float32>>, path: string)` |
| Trajectory shape | `(T, 54)` per clip; T ≈ 100-150 frames at ~30 fps |
| Range | [0.0, ~1.0] — normalized blendshape activations |
| Actors (this parquet) | `W009` (654 clips) + `video28` (658) |
| Per-actor companions | `BarryFutureman/MEAD_3D_W040` / `_W026` / `_W035` / `_M003` |
| Emotions | 8 — angry, contempt, disgusted, fear, happy, neutral, sad, surprised |
| Intensities | level_1 / level_2 / level_3 |
| Path encoding | `{actor}_video/video/front/{emotion}/level_{intensity}/{utt}.mp4` |
| Compressed size | 237 MB main parquet, plus ~50-150 MB per per-actor split |

This collapses the original "Phase B0 spike + Phase B1 extraction" arc
into a single `huggingface-cli download` for the MEAD branch. Spike B0
(MediaPipe extraction validation) remains useful and is retained as the
**fallback path** for datasets that *don't* ship pre-extracted (CREMA-D,
RAVDESS, in-the-wild data); see §RB6.

### Blendshape vocabulary — K = 54 caveat

MEAD_3D uses K = 54, not the ARKit-52 / 51 we targeted in the Spike B0
memo. The dataset card does not document the channel order. We treat
this as a **black-box latent vocabulary** during training:

- Training operates natively in K=54.
- Runtime exports two paths:
  1. K=54 native (consumed by an upcoming `mead3d_to_arkit.onnx`
     remap layer that is fitted post-hoc once the ARKit mapping is
     known — either from author contact or by aligning the same MEAD
     videos under MediaPipe extraction).
  2. K=52 ARKit-padded for legacy rigs that assume the standard
     vocabulary (drop or zero-pad after the remap is fitted).

This keeps the architecture decoupled from any one vocabulary choice.

### Splits — caveat under MEAD_3D

The main parquet has only 2 speakers, so the strict speaker-disjoint
split protocol from `affect_calibration/02_train_vad_regressor.py` would
produce a 1-train / 1-val / 0-test partitioning. To preserve the
speaker-disjoint contract, we **fan out** to the per-actor companion
parquets (W040 / W026 / W035 / M003) before splitting. Final split
target: 4 train speakers / 1 val / 1 test. If the per-actor splits
turn out to share a single speaker after deduplication, fall back to
clip-disjoint within speaker and document the bias.

### Datasets — fallback / augmentation

Used only if (a) MEAD_3D coverage proves insufficient (e.g. emotion
class imbalance, or speakers all sound similar) or (b) we discover the
54-D vocabulary is unusable for downstream rigs. These reuse the
MediaPipe extraction path validated in Spike B0:

| Dataset | License | Hours | Speakers | Use |
|---|---|---:|---:|---|
| **CREMA-D** | ODbL | ~5 h | 91 | speaker pool diversity |
| **RAVDESS** | CC BY-NC-SA | ~3 h | 24 | small held-out (B0 already extracted) |

Original-MEAD (CUHK research access) is **no longer on the critical
path**; if the access form ever clears we may retroactively run our
own MediaPipe extraction on the source videos to reverse-engineer the
54-D channel mapping.

### V/A/D labels

Per-clip categorical → V/A/D mapping reuses the table from
`affect_calibration/scripts/lib/emotion_to_vad.py`. No new lexicon.

---

## 3. Audio features

**Re-use the Whisper encoder from Spike C** (`affect_calibration/scripts/lib/whisper_encoder.py`).

- Extract `last_hidden_state` from `whisper-large-v3-turbo` at layer −2.
- Shape: `(B, T_frames, 1280)` where `T_frames = audio_seconds × 50`
  (Whisper's 20-ms frame rate).
- No further pooling; the diffusion model attends over the full
  `(T_frames, 1280)` sequence via cross-attention.

This is the single biggest reuse from the existing project: the same
audio-frontend artifact already validated for V/A/D regression provides
phoneme-timing-aware embeddings for blendshape generation.

---

## 4. Model architecture

### 4.1 VAE (Stage 1)

- **Input dimension**: K = 54 (MEAD_3D native vocabulary). Note this
  differs from the ARKit-52 figure quoted in §0; runtime export
  remaps K=54 → K=52 via a separately-fitted linear adapter, see RB6.
- **Encoder**: 4 × Conv1d blocks; 2 stride-2 (temporal /2 each) +
  2 stride-1, giving 4× temporal downsample net.
  `(T_frames, 54) → (T_frames/4, d_lat=16)`.
- **Decoder**: 4 × ConvTranspose1d mirror; sigmoid output (data is
  in [0, 1]).
- **Loss**: `MSE(reconstruction) + 0.001 · KL(N(μ, σ) || N(0, I))`.
  Light KL weight; we want a near-deterministic latent space, not full
  generative VAE behavior.
- **Param count**: ~500k.
- **Training**: 50 epochs, batch=64, AdamW(1e-4), cosine schedule.
  Variable-length clips → random 64-frame crops per training step.

### 4.2 DiT-1D denoiser (Stage 2)

- **Backbone**: 8 × DiT-1D transformer blocks, `d_model=384`, 6 heads.
  Sequence length = `T_lat`. Per-block: self-attention over latents +
  cross-attention into Whisper hidden states.
- **Noise schedule**: cosine (Nichol & Dhariwal 2021), 1000 steps train,
  50-step DDIM at inference.
- **Conditioning**: V/A/D coords passed through a 16→128 MLP and added
  to the timestep embedding (AdaLN, modulates per-block scale/shift).
- **Classifier-free guidance**: drop V/A/D embedding 10 % of training
  steps. Inference guidance scale = 2.5 (V/A/D) and 4.0 (audio).
- **Param count**: ~30-50M (target).

### 4.3 Loss

Standard diffusion `ε`-prediction MSE on Stage-1 latents. No auxiliary
losses in v1; lip-sync and affect are evaluated post-hoc in §6, not
trained against directly.

---

## 5. Training

### Stage 1 — VAE (~1-2 GPU-hours)

```
python scripts/01_extract_blendshapes.py --datasets mead,crema_d,ravdess
python scripts/02_train_vae.py --epochs 50 --output artifacts/blendshape_vae.pt
```

### Stage 2 — DiT-1D diffusion (~20-40 GPU-hours)

```
python scripts/03_train_dit.py \
    --vae artifacts/blendshape_vae.pt \
    --whisper-cache artifacts/whisper_embeddings/ \
    --epochs 200 --batch 16 \
    --output artifacts/blendshape_dit_ema.pt
```

- AdamW, lr=1e-4, cosine schedule.
- EMA weights, decay=0.9999. Eval / export uses EMA only.
- Gradient checkpointing on the cross-attention blocks (Whisper sequences
  can be long).

### Compute estimate

- Stage 1: trivial, M2 Mac in ~2 hours.
- Stage 2: 30-50 GPU-hours on a single A100. Rentable for ~$100 total.

---

## 6. Evaluation

Three independent metrics, evaluated on the held-out test split:

### 6.1 Lip-sync — LSE-D / LSE-C

Run **SyncNet** (Chung & Zisserman 2016) on (rendered face, audio)
pairs. LSE-D = sync error in frames; LSE-C = sync confidence.

| Pass threshold | LSE-D ≤ 8.5 | LSE-C ≥ 6.5 |
|---|---|---|

For reference, Audio2Face publishes ~6.0 / ~9.0. We aim within 30 % of
that on commodity infra.

### 6.2 Affect controllability

Run the audio-side V/A/D regressor (`whisper_to_vad.onnx` from
`affect_calibration`; not to be confused with the LLM-side
persona-vector directions used by Channel D steering) on synthesized
blendshape trajectories rendered to a generic face. For a fixed audio
clip, sweep V/A/D conditioning across the corners of the cube and
measure:

```
gap_pos[axis] = mean(predicted[axis] | conditioned=+1) - mean(predicted[axis] | conditioned=0)
gap_neg[axis] = mean(predicted[axis] | conditioned=-1) - mean(predicted[axis] | conditioned=0)
```

| Pass threshold | gap_pos ≥ +0.30 | gap_neg ≤ -0.30 | per axis |
|---|---|---|---|

(Same shape of test as `04_validate_pipeline.py` for the LLM directions,
adapted from token-net-score to V/A/D-regressor-output.)

### 6.3 Cross-axis interference

Steering valence to +1 should not bleed into arousal or dominance by
more than 0.15 (Spike F's same-cube-corner test, applied to the avatar
output rather than the LLM output).

---

## 7. Deployment

### 7.1 ONNX export

```
python scripts/05_export_onnx.py \
    --vae artifacts/blendshape_vae.pt \
    --dit artifacts/blendshape_dit_ema.pt \
    --output artifacts/blendshape_diffusion.onnx
```

Two-graph ONNX bundle: `vae_decoder.onnx` + `dit_denoiser.onnx`. The DDIM
sampler runs in Python (or as a tiny Rust scheduler) and calls both at
each step.

### 7.2 RemoteMedia node

A new `BlendshapeDiffusionNode` lands in
`clients/python/remotemedia/nodes/avatar/`, with the same multiprocess
shape as `WhisperNode`:

- Input: `RuntimeData::Audio` + optional `RuntimeData::AffectVad`.
- Output: `RuntimeData::Blendshapes` (new variant; 52 floats per frame).
- Backend: ONNX runtime (CPU or CoreML EP on Mac).

### 7.3 Live2D / Unreal binding

Out of scope for this project; `crates/avatar/` already has Live2D
plumbing. The new `Blendshapes` `RuntimeData` variant slots into that
existing pipeline.

---

## 8. Phased delivery

Mirrors the `affect_calibration` cadence:

| Phase | Deliverable | Duration | Status |
|---|---|---|---|
| **B0** | Spike memo: MediaPipe extraction validated on RAVDESS pilot. Path retained as fallback. | 1 day | ✅ done (2026-05-06) |
| **B1** | `01_load_mead_3d.py`. Validate parquet + emit manifest + per-channel stats. (Replaces full MediaPipe extraction; pre-extracted dataset.) | 0.5 day | next |
| **B2** | `02_train_vae.py`. Stage-1 trained on MEAD_3D K=54, recon MSE < 0.005. | 2 days | |
| **B3** | `03_train_dit.py`. Stage-2 trained, sample quality eyeballed. | 1 week | |
| **B4** | `04_validate.py`. LSE-D, V/A/D controllability, cross-axis. | 3-4 days | |
| **B5** | `05_export_onnx.py` + `BlendshapeDiffusionNode` + K=54→ARKit-52 remap layer. | 3-4 days | |

Total: ~4-5 weeks elapsed (compressed by ~1 week thanks to pre-extracted
MEAD_3D), ~30-50 GPU-hours.

---

## 9. Risks

| ID | Risk | Mitigation |
|---|---|---|
| **RB1** | MediaPipe blendshape extraction quality is noisy on emotional faces. | Spike B0 measures it before B1 commits. Fallback: ARKit's iPhone-front-cam ground truth on a hand-recorded set. |
| **RB2** | Whisper hidden states aren't fine-grained enough for phoneme-precise lip-sync. Audio2Face uses a custom audio frontend (Mel + raw waveform). | If LSE-D ≥ 12, augment Whisper features with a parallel Mel branch (small Conv1d stack, learned with the DiT). Spike B3-A. |
| **RB3** | V/A/D conditioning gets ignored — DiT relies on audio alone and CFG drop doesn't bite. | Aggressive CFG drop (20 %), guidance scale 5.0 at inference. If gap_pos < 0.15, retrain with stronger AdaLN modulation. |
| **RB4** | Compute exceeds budget. | Halve `d_model` to 256 and blocks to 6, retrain. |
| **RB5** | Quality clearly lags Audio2Face. | Acceptable v1; this is a *control* primitive, not a quality replacement. Document the gap, ship anyway. |
| **RB6** | MEAD_3D K=54 channel order is undocumented; we may train into a vocabulary that no rig understands. | (a) Treat K=54 as opaque latent and fit a remap layer post-hoc by aligning MediaPipe-extracted MEAD videos to MEAD_3D trajectories. (b) Contact dataset author. (c) Worst case: fall back to MediaPipe extraction on RAVDESS+CREMA-D as the original plan, accepting the smaller speaker pool and weaker affect range that Spike B0 observed. |
| **RB7** | MEAD_3D's 2-speaker main parquet is too narrow for a speaker-disjoint split. | Download the per-actor companion parquets (W040 / W026 / W035 / M003) before training. If they share underlying actors with W009 / video28, fall back to clip-disjoint within speaker and document the bias. |

---

## 10. Repository structure

```
tools/affect_avatar/
├── README.md
├── pyproject.toml
├── data/                              # gitignored
│   ├── mead_3d/                       # primary — HF parquet
│   ├── ravdess/                       # B0 pilot + augmentation fallback
│   └── crema_d/                       # augmentation fallback
├── scripts/
│   ├── 01_load_mead_3d.py             # primary path (replaces 01_extract_blendshapes.py)
│   ├── 01_extract_blendshapes.py      # fallback path (MediaPipe; runs on RAVDESS/CREMA-D)
│   ├── 02_train_vae.py
│   ├── 03_train_dit.py
│   ├── 04_validate.py
│   ├── 05_export_onnx.py
│   ├── spike_b0_extract_smoke.py      # B0 paired script
│   ├── build_ravdess_labels.py        # B0 paired script
│   └── lib/
│       ├── blendshape_extractor.py    # MediaPipe wrapper (B0 fallback path)
│       ├── mead_3d_loader.py          # HF parquet wrapper (primary path)
│       ├── vae.py
│       ├── dit.py
│       ├── diffusion.py               # noise schedule + DDIM
│       └── eval.py                    # LSE-D / SyncNet wrapper
├── artifacts/                         # gitignored
│   ├── mead_3d_manifest.json          # B1 output
│   ├── blendshape_vae.pt
│   ├── blendshape_dit_ema.pt
│   ├── blendshape_diffusion.onnx
│   └── eval_report.md
└── notes/
```

---

## 11. Integration with existing project

This project **reuses three artifacts** from `affect_calibration`:

1. **Whisper encoder** (`whisper_encoder.py`) — same model, same layer.
2. **V/A/D regressor** (`whisper_to_vad.onnx`) — used in §6.2 evaluation
   as an *external auditor* of the avatar's affect controllability.
3. **`emotion_to_vad.py`** — categorical → V/A/D map for label
   construction.

This project **extends the runtime** by adding:

- `RuntimeData::Blendshapes` variant.
- `BlendshapeDiffusionNode` (multiprocess Python).
- `clients/python/remotemedia/nodes/avatar/` directory.

This project **does not modify** any existing affect-calibration script,
the activation-steering pipeline, or any LLM-side code. Strictly
additive.

---

## 12. What we'd do first

Spike B0: 1-day feasibility memo. Sample 100 MEAD clips × 3 emotions
each, run MediaPipe blendshape extraction, sanity-check that:

1. Blendshape trajectories are smooth (no jitter that needs filtering).
2. Phoneme-aligned mouth shapes look right (visual eyeball, not metric).
3. Affect-correlated shapes (e.g. `mouthSmileLeft/Right` for happy,
   `browInnerUp` for sad) move in the expected direction across emotions.

If B0 passes, B1 (full extraction) commits. If B0 fails on (1) or (3),
fall back to ARKit-iPhone ground truth or DECA/EMOCA — at higher
implementation cost, before any DiT training.

See [`notes/spike-b0-blendshape-extraction.md`](notes/spike-b0-blendshape-extraction.md)
for the spike protocol.
