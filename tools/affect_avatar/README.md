# Affect Avatar Toolchain

Audio → ARKit-52 blendshape latent-diffusion model. Sibling to
[`tools/affect_calibration/`](../affect_calibration/).

**Specs**: [`docs/references/audio-blendshape-diffusion/`](../../docs/references/audio-blendshape-diffusion/).

**Status**: Skeleton only. Spike B0 (MediaPipe blendshape extraction
feasibility) is the next concrete deliverable; no training scripts
land before B0 passes.

---

## What this produces

1. `blendshape_vae.pt` — VAE encoder/decoder mapping ARKit-52 trajectories
   ↔ a compact latent space (`d_lat=16`, 4× temporal downsample).
2. `blendshape_dit_ema.pt` — DiT-1D denoiser conditioned on Whisper hidden
   states + V/A/D affect coords, classifier-free-guidance trained.
3. `blendshape_diffusion.onnx` — exported bundle (vae_decoder + dit) for
   the `BlendshapeDiffusionNode` runtime.
4. `eval_report.md` — LSE-D / LSE-C lip-sync, V/A/D controllability,
   cross-axis interference.

---

## Layout (target)

```
tools/affect_avatar/
├── README.md                          # this file
├── pyproject.toml
├── data/                              # gitignored
│   ├── mead/
│   ├── crema_d/
│   └── ravdess/
├── scripts/                           # filled in B1+
│   ├── 01_extract_blendshapes.py
│   ├── 02_train_vae.py
│   ├── 03_train_dit.py
│   ├── 04_validate.py
│   ├── 05_export_onnx.py
│   └── lib/
│       ├── blendshape_extractor.py
│       ├── vae.py
│       ├── dit.py
│       ├── diffusion.py
│       └── eval.py
├── artifacts/                         # gitignored
└── notes/                             # per-run ADRs
```

---

## Phased delivery (mirrors [PROJECT_PLAN §8](../../docs/references/audio-blendshape-diffusion/PROJECT_PLAN.md))

| Phase | Deliverable | Effort |
|---|---|---|
| **B0** | Spike: MediaPipe blendshape extraction feasibility on 100 MEAD clips. | 1 day |
| **B1** | `01_extract_blendshapes.py`. MEAD + CREMA-D + RAVDESS extracted. | 2-3 days |
| **B2** | `02_train_vae.py`. Recon MSE < 0.005. | 2 days |
| **B3** | `03_train_dit.py`. CFG + EMA. ~30-50 GPU-hours. | 1 week |
| **B4** | `04_validate.py`. LSE-D ≤ 8.5, gap_pos ≥ +0.30 per axis. | 3-4 days |
| **B5** | `05_export_onnx.py` + `BlendshapeDiffusionNode`. | 3-4 days |

---

## Reuse from `affect_calibration`

- **Whisper encoder** — `affect_calibration/scripts/lib/whisper_encoder.py`
  is imported as-is; same model (`whisper-large-v3-turbo`), same layer (-2).
- **V/A/D regressor** — `whisper_to_vad.onnx` is used at evaluation time
  as an *external auditor* of the avatar's affect controllability
  (independent of training).
- **Speaker-disjoint splitter** — `_speaker_disjoint_split` from
  `02_train_vad_regressor.py`.
- **Categorical → V/A/D map** — `lib/emotion_to_vad.py`.

This toolchain does **not** depend on the activation-steering or LLM
calibration paths; the affect-avatar project trains in parallel and
ships independently.

---

## Reproduction (target — B1+)

```bash
cd tools/affect_avatar
uv sync
huggingface-cli download <MEAD bucket TBD> --local-dir data/mead
huggingface-cli download myleslinder/crema-d --repo-type dataset --local-dir data/crema_d

./scripts/01_extract_blendshapes.py --datasets mead,crema_d,ravdess
./scripts/02_train_vae.py --epochs 50 --output artifacts/blendshape_vae.pt
./scripts/03_train_dit.py \
    --vae artifacts/blendshape_vae.pt \
    --epochs 200 --batch 16 \
    --output artifacts/blendshape_dit_ema.pt
./scripts/04_validate.py --report artifacts/eval_report.md
./scripts/05_export_onnx.py --output artifacts/blendshape_diffusion.onnx
```

(Commands above are illustrative; real flag surfaces are decided per phase.)
