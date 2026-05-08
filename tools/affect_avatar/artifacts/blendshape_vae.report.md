# Stage-1 VAE training report

- Device: `mps`
- Epochs: 30
- Params: 297,878
- Latent dim: 16  hidden: 128
- KL weight: 0.001

## Splits

- train: 12947 clips
- val: 1226 clips
- test: 1084 clips

## Final-epoch metrics

| split | recon_mse | kl |
|---|---:|---:|
| train | 0.00274 | 0.8500 |
| val | 0.00281 | 0.8310 |
| test | 0.00233 | 0.8473 |

## Per-channel MSE (test, sampled 50 clips)

- mean: 0.00248  min: 0.00000  max: 0.01155
- worst 5 channels (by MSE):
  - k=24  mse=0.01155
  - k= 1  mse=0.01144
  - k= 0  mse=0.01090
  - k=19  mse=0.00929
  - k=43  mse=0.00750

## Pass criterion (PROJECT_PLAN §8 row B2)

- recon MSE < 0.005 target.  Achieved (test): **0.00233** — ✅ PASS
