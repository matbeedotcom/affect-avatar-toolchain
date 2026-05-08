# affect-avatar-toolchain

Audio → ARKit-flavored blendshape **latent-diffusion** model. Maps
speech audio (Whisper hidden states) + optional V/A/D (valence /
arousal / dominance) coordinates to per-frame 54-channel facial
animation, generic across rigs that consume ARKit-style blendshapes
(Live2D, MetaHuman, Unreal, three.js).

**Specs**: [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).
**Status**: B3 (DiT denoiser) on 26-actor MEAD_3D corpus, val ε-MSE 0.193.

---

## Architecture

```
audio (16 kHz)                               V/A/D coords (3-d, optional)
    │                                                │
    ▼                                                ▼
┌────────────────┐                            ┌───────────┐
│  Whisper       │  hidden states             │  MLP      │
│  large-v3-turbo│ ─(B, T_audio, 1280)──┐    │  embed    │
│  (frozen)      │                       │    └─────┬─────┘
└────────────────┘                       │          │
                                         ▼          ▼
                                ┌───────────────────────┐
                                │  DiT-1D denoiser      │   8 blocks × d_model=384
                                │  cross-attn → Whisper │   AdaLN-Zero, CFG-trained
                                │  AdaLN ← time + V/A/D │   ~37M params
                                └───────────┬───────────┘
                                            │ ε prediction
                                            ▼
                                    z₀ ∈ ℝ^{T/4 × 16}        ← VAE latent
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │  VAE decoder (frozen) │   ~300k params
                                └───────────┬───────────┘
                                            │
                                            ▼
                            blendshapes ∈ ℝ^{T × 54}  (ARKit-flavored)
```

**Two stages, training in order**:
1. **Stage 1 — VAE** (`02_train_vae.py`): compresses `(T, 54)` blendshapes
   to `(T/4, 16)` latents. ~5 min on M1 Pro / MPS.
2. **Stage 2 — DiT** (`04_train_dit.py`): cosine-schedule diffusion on
   the VAE latent space, conditioned on Whisper + V/A/D, with 10 %
   classifier-free-guidance dropout. ~2 hr / 30 epochs on MPS.

---

## Layout

```
.
├── README.md                           # this file
├── pyproject.toml
├── docs/
│   ├── PROJECT_PLAN.md                 # specs, phased delivery, risks
│   ├── INDEX.md
│   └── notes/
│       ├── README.md
│       └── spike-b0-blendshape-extraction.md   # MediaPipe fallback path memo
├── data/                               # gitignored — datasets per developer
├── artifacts/                          # tracked: VAE.pt + manifests + reports
│   ├── blendshape_vae.pt               # trained Stage-1 VAE
│   ├── blendshape_vae.csv              # per-epoch loss curve
│   ├── blendshape_vae.report.md
│   ├── blendshape_dit.csv              # per-epoch loss curve
│   ├── blendshape_dit.report.md
│   ├── mead_3d_manifest.json           # main parquet metadata
│   └── mead_3d_multi_manifest.json     # 26-actor union metadata
└── scripts/
    ├── 01_load_mead_3d.py              # validate parquet(s) + emit manifest
    ├── 02_train_vae.py                 # Stage-1 VAE training
    ├── 03_cache_whisper_embeddings.py  # pre-compute Whisper hidden states
    ├── 04_train_dit.py                 # Stage-2 diffusion training
    ├── build_ravdess_labels.py         # RAVDESS label generator (B0 fallback)
    ├── spike_b0_extract_smoke.py       # MediaPipe extraction feasibility
    └── lib/
        ├── audio_decoder.py            # MEAD_3D audio bytes → 16 kHz np.float32
        ├── blendshape_extractor.py     # MediaPipe FaceLandmarker wrapper
        ├── diffusion.py                # cosine schedule + DDIM sampler
        ├── dit.py                      # DiT-1D model
        ├── emotion_to_vad.py           # MEAD emotion → V/A/D coords
        ├── mead_3d_loader.py           # multi-parquet HF reader
        └── vae.py                      # Stage-1 Conv1d VAE
```

---

## Reproduction

### 1. Set up the environment

```bash
# Python 3.11 venv via uv
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
    torch transformers soundfile librosa pyarrow datasets \
    "mediapipe>=0.10" "opencv-python>=4.9" "numpy<2"
```

### 2. Download datasets

The model trains on
[`BarryFutureman/MEAD_3D`](https://huggingface.co/datasets/BarryFutureman/MEAD_3D)
plus the per-actor companion parquets and the 8 sharded parquets —
together a ~26-actor MEAD subset that ships **pre-extracted blendshape
trajectories paired with audio** (no MediaPipe step needed).

```bash
# main parquet (W009 + video28)
mkdir -p data/mead_3d && \
  huggingface-cli download BarryFutureman/MEAD_3D --repo-type dataset \
    --local-dir data/mead_3d

# per-actor parquets (9 actors)
for actor in M003 W024 W026 W029 W035 W036 W037 W038 W040; do
  mkdir -p data/mead_3d_${actor} && \
    huggingface-cli download BarryFutureman/MEAD_3D_${actor} --repo-type dataset \
      --local-dir data/mead_3d_${actor}
done

# shards (16 files across 8 shards — multi-actor each)
for n in 00 01 02 03 04 05 06 07; do
  mkdir -p data/mead_3d_shard_${n} && \
    huggingface-cli download BarryFutureman/MEAD_3D_shard_${n} --repo-type dataset \
      --local-dir data/mead_3d_shard_${n}
done
```

### 3. Validate the union (1 min)

```bash
.venv/bin/python scripts/01_load_mead_3d.py --data-dir data \
    --output artifacts/mead_3d_multi_manifest.json
# → 26 actors, ~17k unique clips, all 8 emotions × 3 intensities
```

### 4. Cache Whisper hidden states (~4-5 hr on M1 Pro / MPS)

```bash
.venv/bin/python scripts/03_cache_whisper_embeddings.py \
    --data-dir data --output-dir artifacts/whisper_cache
# → ~8 GB of per-clip .npz files
```

### 5. Train Stage-1 VAE (~5 min)

```bash
.venv/bin/python scripts/02_train_vae.py \
    --data-dir data \
    --val-actors video28,video26 --test-actors M003,W040 \
    --epochs 30 --crops-per-clip 4 \
    --output artifacts/blendshape_vae.pt
# → val recon ~0.003, test ~0.002
```

### 6. Train Stage-2 DiT (~2.2 hr)

```bash
.venv/bin/python scripts/04_train_dit.py \
    --data-dir data \
    --val-actors video28,video26 --test-actors M003,W040 \
    --epochs 30 --crops-per-clip 2 --batch-size 32 \
    --output artifacts/blendshape_dit.pt
# → best val ε-MSE ~0.19, test ~0.21
```

---

## Status

| Phase | Goal | State |
|---|---|---|
| B0 | MediaPipe extraction feasibility (fallback path for non-MEAD_3D data) | done — see [`spike-b0-blendshape-extraction.md`](docs/notes/spike-b0-blendshape-extraction.md) |
| B1 | Multi-source data loader + manifest | done |
| B2 | Stage-1 VAE | done — val recon 0.003 |
| B3 | Stage-2 DiT diffusion | done (baseline) — val ε-MSE 0.193 |
| B4 | Eval (LSE-D / V/A/D controllability / cross-axis) | not started |
| B5 | ONNX export + runtime node | not started |

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for full deliverable
list, risks (RB1-RB7), and architectural rationale.

---

## License

MIT (TBD).
