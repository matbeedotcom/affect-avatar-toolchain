# Stage-2 DiT-1D training report

- Device: `mps`
- Epochs trained: 30
- Params: 61,752,080
- d_lat: 16  d_model: 512  n_blocks: 8  n_heads: 8
- Diffusion timesteps: 1000  d_whisper: 1280

## Splits

- train: 17556 clips
- val: 1238 clips
- test: 1683 clips

## Final ε-loss snapshot (saved checkpoint = best-EMA from epoch 3)

| split | ε-MSE | weights | source |
|---|---:|---|---|
| train | 0.21257 | online | last epoch |
| val   | 0.24173 | EMA | best (epoch 3) |
| test  | 0.25808 | EMA | evaluated on best-EMA |

- Last-epoch val_ema: 0.26853 (diverged from best by +0.027; clear overfit signal — see Notes).

## Notes

- ε-MSE around `1.0` indicates no learning (zero is a competitive baseline). Below `0.5` is real denoising signal. Below `0.3` is decent for a first cycle.
- The val curve typically bottoms out then climbs as the model overfits the training-speaker crops. We export the best-EMA snapshot — the saved checkpoint is the model from the val-minimum epoch, not the final epoch.
- Cross-speaker generalization gap is the dominant cost per RB7 (only 2 speakers in the main parquet). Mitigation: fan out to per-actor companion parquets (W040 / W026 / W035 / M003) before the next training cycle.
