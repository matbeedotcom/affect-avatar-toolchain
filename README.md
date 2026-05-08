# affect-avatar-toolchain

Toolchains for building **affect-aware speech-driven avatars**. Four
related projects in one repo:

| Tool | Purpose | Stack |
|---|---|---|
| [`tools/affect_avatar`](tools/affect_avatar/) | Audio → ARKit-flavored blendshape latent diffusion. *The namesake.* | Python · PyTorch · MPS |
| [`tools/affect_calibration`](tools/affect_calibration/) | Whisper → V/A/D regressor + Persona-Vectors LLM steering directions. | Python · PyTorch · MPS |
| [`tools/affect_simulator`](tools/affect_simulator/) | Text-only deterministic affect dynamics simulator (appraisal → state → channels). | Rust |
| [`tools/affect_coach`](tools/affect_coach/) | Lightweight runtime helper for affect-driven face baking / coaching. | Python |

The pieces compose at runtime: the **simulator** maps events to V/A/D
trajectories, the **calibration** artifacts give the LLM and audio
sides their direction vectors and regressor, and the **avatar**
diffusion model renders the resulting V/A/D + speech audio into
per-frame blendshapes.

---

## Specs

Two reference-document families:

- [`docs/references/audio-blendshape-diffusion`](docs/references/audio-blendshape-diffusion/) — specs for `affect_avatar` (PROJECT_PLAN, B0 spike memo).
- [`docs/references/activation-steering-audio-llm`](docs/references/activation-steering-audio-llm/) — specs for `affect_calibration` + simulator (PROJECT_PLAN, DESIGN, IMPLEMENTATION_PLAN, A-G spike memos).

---

## Status (snapshot)

| Project | Phase | State |
|---|---|---|
| `affect_avatar` | B3 — DiT diffusion | val ε-MSE 0.193 on 26-actor MEAD_3D; B4 eval pending |
| `affect_calibration` | D1.x | full LLM-side pipeline + V/A/D regressor; paired-eval harness |
| `affect_simulator` | Phase 1 | scenarios authored, dynamics validated; trace-and-render path runs |
| `affect_coach` | early | runtime baking + face-coaching utility |

---

## Layout

```
.
├── README.md                                 # this file
├── docs/
│   └── references/
│       ├── audio-blendshape-diffusion/       # affect_avatar specs
│       └── activation-steering-audio-llm/    # affect_calibration + simulator specs
└── tools/
    ├── affect_avatar/                        # Python · diffusion · audio→blendshape
    │   ├── README.md                         # detailed reproduction recipe
    │   ├── pyproject.toml
    │   ├── scripts/
    │   │   ├── 01_load_mead_3d.py
    │   │   ├── 02_train_vae.py
    │   │   ├── 03_cache_whisper_embeddings.py
    │   │   ├── 04_train_dit.py
    │   │   ├── build_ravdess_labels.py
    │   │   ├── spike_b0_extract_smoke.py
    │   │   └── lib/
    │   └── artifacts/                        # tracked: VAE.pt + manifests + reports
    │
    ├── affect_calibration/                   # Python · Whisper-VAD + LLM directions
    │   ├── README.md
    │   ├── pyproject.toml
    │   ├── data/                             # hand-authored inputs (trait_descriptions, contrast_pairs, probe_prompts)
    │   └── scripts/
    │       ├── 01_extract_whisper_embeddings.py
    │       ├── 02_train_vad_regressor.py
    │       ├── 02b_generate_contrast_pairs.py
    │       ├── 03_extract_llm_directions.py
    │       ├── 04_validate_pipeline.py
    │       ├── 05_runtime_steering_smoke.py
    │       ├── 06_audio_steering_smoke.py
    │       ├── 07_channel_ab_smoke.py
    │       ├── 08_prosody_vad_smoke.py
    │       ├── 09_paired_demo.py
    │       ├── 10_fg_eval.py
    │       └── lib/
    │
    ├── affect_simulator/                     # Rust · text-only affect dynamics
    │   ├── Cargo.toml
    │   ├── README.md
    │   ├── scenarios/                        # 14 hand-authored test scenarios
    │   └── src/
    │       ├── lib.rs / main.rs
    │       ├── appraisal.rs
    │       ├── dynamics.rs
    │       ├── regulation.rs
    │       ├── channel_a.rs / channel_d.rs
    │       ├── event.rs
    │       └── renderer.rs
    │
    └── affect_coach/                         # Python · runtime affect helpers
        ├── README.md
        ├── coach.py
        └── bake_affect_face.py
```

---

## Per-tool quick-start

Each tool has its own README with the full reproduction recipe; this is
the one-screen overview.

### `affect_avatar`

```bash
cd tools/affect_avatar
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch transformers soundfile librosa pyarrow datasets

# 1) download MEAD_3D parquets (~1.2 GB main+per-actor, +5 GB shards)
# 2) cache Whisper hidden states (~4-5 hr on M1 Pro / MPS)
.venv/bin/python scripts/03_cache_whisper_embeddings.py --data-dir data --output-dir artifacts/whisper_cache

# 3) train Stage-1 VAE (~5 min)
.venv/bin/python scripts/02_train_vae.py --data-dir data \
    --val-actors video28,video26 --test-actors M003,W040 \
    --epochs 30 --output artifacts/blendshape_vae.pt

# 4) train Stage-2 DiT (~2.2 hr)
.venv/bin/python scripts/04_train_dit.py --data-dir data \
    --val-actors video28,video26 --test-actors M003,W040 \
    --epochs 30 --output artifacts/blendshape_dit.pt
```

### `affect_calibration`

See [`tools/affect_calibration/README.md`](tools/affect_calibration/README.md). Produces:
- `whisper_to_vad.onnx` — speech-emotion regressor mapping Whisper
  encoder embeddings to (valence, arousal, dominance).
- `llm_directions/{model}_layer{N}.npz` — Persona-Vectors-style
  direction matrix for LLM steering.

### `affect_simulator`

```bash
cd tools/affect_simulator
cargo run --release -- --scenario scenarios/warm_admiration.json
```

### `affect_coach`

Lightweight runtime baking. See [`tools/affect_coach/README.md`](tools/affect_coach/README.md).

---

## Composition

```
                                        ┌────────────────────┐
                                        │ affect_simulator   │  text events → V/A/D
                                        │ (Rust, real-time)  │  trajectory
                                        └─────────┬──────────┘
                                                  │
                                                  ▼
events                          ┌────────────────────────────────────┐
   │                            │     V/A/D coords                   │
   ▼                            │  (valence, arousal, dominance)     │
audio (16 kHz mono)             └────────────┬───────────────────────┘
   │                                         │
   ├──────────► whisper_to_vad.onnx ─────────┤   (calibration's regressor;
   │            (affect_calibration)         │    auditor at eval time)
   │                                         │
   │                                         ▼
   │           ┌───────────────────────────────────────┐
   ├──────────►│         affect_avatar (DiT)           │──► blendshapes (54 ARKit)
   │           │  Whisper hidden states → cross-attn   │
   │           │  + V/A/D coords → AdaLN modulation    │
   │           └───────────────────────────────────────┘
   │
   └──────────► llm_directions.npz → Persona-Vectors steering hook
                (affect_calibration; runtime-injected α·v_axis)
```

---

## License

MIT (TBD).
