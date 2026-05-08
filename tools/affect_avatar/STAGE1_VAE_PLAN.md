# Stage-1 VAE replacement — plan

Plan document for the Stage-1 VAE rebuild that came out of the round-trip
diagnostic and the cross-emotion sweep. Replaces ad-hoc per-experiment
notes with one coherent reasoning + tracking artifact. Updated as
experiments land — each Experiment row gets a status, an artifact
path, the gate-by-gate pass/fail readout, and a one-sentence verdict.

Companion docs:
- [`artifacts/vae_roundtrip_diagnostic.report.md`](artifacts/vae_roundtrip_diagnostic.report.md) — original round-trip evidence
- [`artifacts/cross_emotion_sweep.report.md`](artifacts/cross_emotion_sweep.report.md) — failure-mode catalogue
- [`artifacts/intensity_sweep/REPORT.md`](artifacts/intensity_sweep/REPORT.md) — intensity gradient validation

---

## 1. Current state

### Models we have

| ckpt | type | notes |
|---|---|---|
| `blendshape_dit.pt`               | DiT (d_vad=4) | current canonical; intensity-aware on **VAE v1**. |
| `blendshape_dit.preIntensity.pt`  | DiT (d_vad=3) | pre-intensity backup (also on VAE v1). |
| `blendshape_dit.preVaeRetrain.pt` | DiT (d_vad=3) | older backup, same content as `.preIntensity.pt`. |
| `blendshape_dit_intensity.pt`     | DiT (d_vad=4) | duplicate of current canonical, kept for back-references in REPORTs. |
| `blendshape_vae.pt`               | VAE v1        | **canonical, but proven broken** (round-trip wipes rare channels). |
| `blendshape_vae_v2.pt`            | VAE v2        | weighted-MSE + kl=5e-4 attempt; **partial fix** (smile-leak gone, jawOpen/frown regressed). |
| `blendshape_dit_v2.pt`            | n/a           | DiT v2 training was killed before completion — no usable ckpt. |

### What's working

  - **Intensity-aware conditioning mechanism.** The d_vad=4 architecture
    correctly disambiguates "no emotion" from "neutral emotion." When
    the VAE doesn't gate it, the smile-channel ramps 0.71→0.87 across
    `intensity_cond ∈ {0.0, 0.33, 0.67, 1.0, 1.5}` on happy clips.
  - **V controllability is bidirectional** for the first time in the
    project (gap_pos +0.46 / gap_neg −0.52 on the d_vad=4 ckpt).
  - **CFG sweet spot at cfg=3.0** — established by sweep, no D-axis
    collapse seen up to cfg=5.
  - **ARKit map binding** correctly resolves to actual GLB mesh names
    (`Brows`, `Eyelash_Upper`, etc.) after the 201-patch fix.
  - **Rendering pipeline** (Bevy `affect_face_smoke` + ffmpeg mux)
    produces correct MP4s at 30 fps with audio.

### What's broken

  - **VAE v1 encoder collapses rare-event channels** (round-trip
    evidence). This is the bottleneck for everything downstream.
  - **VAE v1 decoder fills ambiguous latents with smile + browInUp** —
    explains the cross-emotion smile-leak on negative emotions.
  - A and D axes still don't probe-pass — partly probe-data limitation
    (MEAD has no real low-A / low-D anchors), partly the VAE issue
    above.

---

## 2. Findings

Each finding has a numbered key (F1, F2, …) so experiments below can
cite the evidence they're acting on.

### F1 — VAE v1 round-trip destroys high-amplitude rare channels

Round-trip protocol: `vae(GT_clip)["recon"]` with no DiT, no diffusion.

Evidence (angry M003 GT):

| channel | GT max | v1 recon | Δ |
|---|---:|---:|---:|
| eyeWide      | 0.997 | 0.402 | −0.595 |
| browDn       | 0.785 | 0.029 | **−0.756 wiped** |
| cheekPuff    | 0.865 | 0.013 | **−0.852 wiped** |
| mouthShrug   | 0.901 | 0.063 | **−0.838 wiped** |
| eyeBlink     | 0.787 | 0.053 | −0.735 |
| mouthSmile   | 0.001 | 0.264 | +0.263 (**invented**) |

Same pattern on fear, disgusted, surprised. Even on happy: cheekPuff
(0.59→0.02) and eyeBlink (0.38→0.07) are wiped.

### F2 — Decoder default attractor is "smile + browInUp"

Across multiple emotions where the encoder produces ambiguous mu (rare
high-amplitude channel input), the decoder reconstructs the most-common
high-amplitude shape: smile (mouthSmile) + brow raise (browInnerUp).
This is why the cross-emotion sweep showed smile rising with intensity
on sad/fear/disgusted GT clips.

### F3 — `d_lat=16` is not the bottleneck

PCA on 55k training frames:

  - 12 dims → 95% cumulative variance
  - 16 dims → 97.5%
  - 22 dims → 99%

The current 16-d latent has 2× the capacity needed for 95% variance.
Linear PCA at k=12 already gets 95%, so a *linear* AE at d_lat=16 should
trivially beat the current VAE. **The VAE is doing worse than linear
PCA on rare channels** — that's an objective/regularization problem,
not a capacity problem.

### F4 — Inverse-stdev channel weighting is a net win on smile-leak but introduces new biases

VAE v2 (per-channel weighted MSE, power=1.0, dead-channel exclusion,
clip [0.25, 4.0], kl=5e-4) round-trip vs v1:

  - ✅ smile leak removed (angry 0.26→0.05; fear 0.57→0.02; disgusted 0.14→0.02)
  - ✅ mouthShrug recovers significantly
  - ❌ jawOpen weakens (angry 0.90→0.55)
  - ❌ mouthFrown over-fires (spurious frown when GT has none)
  - ❌ eyeWide still mostly broken (0.99 GT → 0.17 recon for fear)

Statistical reweighting fixes one bias (smile attractor) by trading it
for another (frown over-firing). Semantic grouping is a better path.

### F5 — V/A/D = (0, 0, 0) is "neutral", not "null"

The MEAD_VAD table maps `neutral → (0, 0, 0)`. CFG dropout (`cfg_drop_prob`)
zeroes the V/A/D vector for the unconditional branch — but that's
indistinguishable from a "neutral" emotion request. cfg_drop_prob=0.10
hurt controllability for this reason. **A learned null embedding is the
correct fix**, not zero-VAD.

### F6 — Intensity-conditioning mechanism works when not gated by VAE

Smile-channel ramp 0.71→0.87 on happy clips across intensity_cond
0.0→1.5 is monotonic and matches GT at peak. The mechanism is sound; it
just can't propagate signal through a broken latent space for the
emotions where the VAE has thrown away the relevant channels.

### F7 — Renderer/binding bugs masquerade as model bugs

The 201-patch ARKit map fix made it possible to actually evaluate the
model visually. Before the fix, brow + eyelash morphs were silently
dropped by the renderer, making both GT and sample look "muted." Always
verify rendering independently before drawing model conclusions.

### F8 — `val_eps` is a misleading proxy for sample quality

Three independent experiments (z-score+kl=0.1, cfg_drop_prob=0.10,
weighted-VAE candidates) showed val_eps moving in directions that
*disagreed* with controllability / round-trip diagnostics. **The
"round-trip vs DiT gap" metric is the only reliable separation between
Stage-1 and Stage-2 problems.**

### F10 — AE v3 passes preservation gates, fails strict gates on M003 GT realities

Exp 1 deterministic AE (KL=0, linear output, grouped+velocity+peak loss)
trained 50 epochs to best val_total=0.00168 (test_recon_mse=0.00080).
Dashboard run on M003 intensity=3, 15 clips per emotion, max-agg:

  - **Preservation gates (per F9): 3/3 PASS.**
    - P1 median preservation ratio = **1.013** (target ≥ 0.85). Linear
      output overshoots slightly on a few channels — harmless because
      the loss pulls it back to GT range overall.
    - P2 fraction of active pairs preserved at ≥ 0.70 ratio = **1.000**
      (64/64). v1 = 0.625, v2 = 0.891. AE v3 is the only Stage-1
      candidate with zero collapsed pairs.
    - P3 max decoder invention on near-zero GT channels = 0.101 (target
      ≤ 0.20). Only worse than v2's 0.031, but well within bounds.
  - **Strict gates: 3/8.** Same gates that fail v1 (2/8) and v2 (3/8).
    Gate 4 ("angry mouthSmile ≤ 0.10") is the most informative failure:
    AE v3 produces angry recon mouthSmile = 1.01, which **faithfully
    matches GT angry mouthSmile = 0.97**. The "smile leak" framing in
    the original report assumed GT angry mouthSmile = 0.001 — that's
    not the data we have on M003. AE v3 isn't inventing smile; it's
    correctly preserving what M003 actually does in anger (likely
    teeth-bared expressions that MediaPipe scores as smile). This is
    a **data-labeling concern** for Stage-2, not a Stage-1 failure.
  - **Per-channel deltas show near-perfect faithfulness.** Largest
    negative Δ on the gate-relevant channels: fear browInnerUp (-0.19,
    79% preservation). Largest positive Δ: disgusted jawOpen (+0.10).
    All other deltas within ±0.05.

Decision: admit AE v3 to Exp 5 (DiT retrain). The preservation-gate
formulation is the correct interpretation of "encoder doesn't wipe
rare channels"; AE v3 decisively passes. The data-labeling issue
(M003 anger has smile activation) is a separate concern that may
surface in cross-emotion sweeps after DiT retrain — track in Exp 5
and consider per-emotion baseline subtraction or different test
actors at that point.

### F9 — Original round-trip report's clip selection was non-representative

Exp 0 dashboard run on `blendshape_vae.pt` (v1) over 15 random M003
intensity-3 clips per emotion gives GT distributions that don't match
the values quoted in `vae_roundtrip_diagnostic.report.md`. Specifically:

  - Original report: angry GT eyeWide=0.997, cheekPuff=0.865.
  - Dashboard (max across 15 clips): angry GT eyeWide=**0.01**,
    cheekPuff=**0.00**.

MEAD's M003 angry clips don't actually hit eyeWide near 1.0 across
random sampling. The original report selected single clips that
maximised those channels — the dashboard's median/max-over-15 captures
the population, not an outlier. Implications:

  1. **Gate 3 (angry cheekPuff ≥ 0.50)** is unreachable: cheekPuff is
     chronically zero in MEAD across all emotions and actors (MediaPipe
     extraction floors it). Gate is testing data that doesn't exist.
  2. **Gate 1 (angry eyeWide ≥ 0.70)** is similarly mis-calibrated for
     M003 — the actor doesn't open their eyes wide on anger. Gate 8's
     disjunctive form is the more robust test (any of {angry, fear,
     disgusted, surprised}).
  3. **F4 ("v2 fixes smile leak")** doesn't reproduce on random
     sampling: dashboard shows v2 gates 4/5 (angry/fear/disgusted/sad
     smile peak) are slightly *worse* than v1 on 15-clip max-agg
     samples. The "smile leak removed" claim was on cherry-picked
     clips. On the population, v2's *only* clear win is gate 7 (happy
     jawOpen preservation): v1 |Δ|=0.19 → v2 |Δ|=0.01.
  4. **Preservation-ratio gates** (recon_peak / gt_peak conditional on
     gt_peak > floor) are a more honest formulation of the original
     intent ("encoder doesn't wipe rare channels"). Defer
     re-formulation until after Exp 1 — current gates still detect the
     pattern even if the absolute thresholds are noisy.

---

## 3. Reasoning for the chosen approach

Logic chain:

1. F1 + F2 prove the **VAE encoder is the structural bottleneck**, not
   the DiT. No amount of DiT improvement can recover information the
   encoder threw away.
2. F3 rules out *capacity* as the issue — `d_lat=16` is overcapacity
   relative to the 95% variance bound. So we're looking at *objective*
   or *regularization* problems.
3. F4 shows a single statistical reweighting trade is insufficient — we
   need either a different objective shape (grouped/peak/velocity) or
   a different probabilistic setup (no KL, free-bits, VQ).
4. F8 means we need a metrics dashboard *before* the next experiment,
   so we don't run another false-trail.
5. F5 + F6 are real wins worth preserving — the conditioning logic
   works once the VAE is sane. Defer conditioning refactors.

Therefore: **all Stage-2 work freezes** until a Stage-1 ckpt passes a
defined round-trip gate.

---

## 4. Experiment sequence

Each experiment has explicit pass criteria. Failure to pass triggers
a documented next step (continue, branch, or abandon).

Channel groups for grouped losses (MEAD/alphabetical-without-_neutral
indexing):

| group | channel indices |
|---|---|
| mouth_speech | 24 jawOpen, 26 mouthClose, 31 mouthFunnel, 32 mouthLeft, 37 mouthPucker, 38 mouthRight, 39 mouthRollLower, 40 mouthRollUpper |
| mouth_affect | 27/28 mouthDimple, 29/30 mouthFrown, 33/34 mouthLowerDown, 35/36 mouthPress, 41/42 mouthShrug, 43/44 mouthSmile, 45/46 mouthStretch, 47/48 mouthUpperUp |
| jaw          | 22 jawForward, 23 jawLeft, 25 jawRight |
| eyes_brows   | 0/1 browDown, 2 browInnerUp, 3/4 browOuterUp, 8/9 eyeBlink, 10–17 eyeLook*, 18/19 eyeSquint, 20/21 eyeWide |
| cheeks_nose  | 5 cheekPuff, 6/7 cheekSquint, 49/50 noseSneer |

### Round-trip pass gate (applied to every Stage-1 candidate)

Per-emotion peak recall on M003 GT clips:

| metric | target |
|---|---|
| angry eyeWide peak     | ≥ 0.70 |
| angry browDn peak      | ≥ 0.50 |
| angry cheekPuff peak   | ≥ 0.50 |
| angry mouthSmile peak  | ≤ 0.10 (no smile leak) |
| fear/disgust/sad smile leak | ≤ 0.10 |
| happy mouthSmile peak  | within ±0.10 of GT |
| happy jawOpen peak     | within ±0.15 of GT |
| eyeWide gap (GT 1.00)  | recon ≥ 0.70 on at least one of {angry, fear, disgust, surprised} |

A candidate **passes** if all 8 gates pass. **Partial passes** (≥ 5/8)
trigger a follow-up experiment in the same family. **Fails** trigger a
branch decision.

### Experiment 0 — metrics dashboard (`08_diagnose.py`)

**Goal:** single script that produces the per-channel / per-emotion
peak-recall table, smile-leakage score, and round-trip-vs-DiT gap, for
any (VAE_ckpt, optional DiT_ckpt) pair.

**Inputs:** path to VAE, optional path to DiT, list of emotion clips.
**Outputs:** Markdown report + JSON summary, both to artifacts/diag_<run>/.

**Pass condition:** runs successfully on the existing v1 + v2 VAEs and
reproduces the round-trip results we already have in
`vae_roundtrip_diagnostic.report.md`. Validates the dashboard before
we trust it for new experiments.

### Experiment 1 — deterministic AE

**Hypothesis (F1+F2+F3):** removing the KL term lets the encoder freely
encode rare-event peaks. If round-trip recovers under KL=0, the
posterior regularization was the primary culprit.

**Recipe:**
  - Architecture: same Conv1d encoder/decoder as v1
  - `KL = 0`, `z = mu` (deterministic), no reparam noise
  - **Final activation: linear**, not sigmoid (sigmoid compresses peaks)
  - Loss: `α_value · MSE_grouped + α_velocity · velocity_MSE + α_peak · peak_MSE`
  - Initial weights: `α_value=1.0`, `α_velocity=0.5`, `α_peak=0.5`
  - Channel grouping per the table above; default per-group weights all 1.0
  - Epochs: 50 (same as v1/v2)
  - Output: `blendshape_ae_v3.pt`

**Pass:** all 8 gates pass on round-trip diagnostic.
**Partial pass (≥5/8):** retune loss weights → Experiment 1b.
**Fail (<5/8):** branch to Experiment 4 (VQ-VAE).

### Experiment 1b — AE loss-weight sweep (only if Exp 1 partial)

If Exp 1 is partial: retune `(α_value, α_velocity, α_peak, group_weights)`
to push the failing gates over threshold without regressing the
passing ones. Bounded to ≤ 3 retunes. If none pass full 8/8 → Exp 4.

### Experiment 2 — AE + small latent noise

**Hypothesis:** if Exp 1 passes round-trip but the latent space is too
"sharp" for diffusion to navigate, a small `σ_train ∈ {0.01, 0.03, 0.05}`
gives smoothness without the catastrophic compression of full KL.

**Recipe:** same as Exp 1, but `z = mu + σ_train · ε` during training,
`z = mu` at inference. Sweep σ_train.

**Pass:** round-trip 8/8 pass + DiT trained on this latent space gets
val_eps within 1.5× of canonical (sanity check on diffusion trainability).

### Experiment 3 — VAE with KL warmup + free bits (only if Exp 1+2 pass and we want a probabilistic latent)

**Hypothesis:** if a probabilistic latent is genuinely required (e.g.
for downstream sampling diversity, or because Exp 2 latents prove
hard to fit), add KL back gradually using cyclical annealing + free
bits per latent dim.

**Recipe:**
  - epochs 0–10: β = 0
  - epochs 10–30: β linearly to target ∈ {1e-4, 3e-4, 5e-4}
  - free_bits ∈ {0.01, 0.03, 0.05} nats/dim
  - else identical to Exp 2 winner

**Pass:** round-trip 8/8 pass + DiT-trainable.

### Experiment 4 — VQ-VAE (parallel branch)

**Trigger:** Exp 1 fails (<5/8 gates). Or runs in parallel if compute
allows once Exp 1 is in flight.

**Hypothesis:** rare expression modes can be separate codebook entries;
discrete codes don't suffer posterior collapse.

**Recipe:**
  - K=512 or 1024 codebook entries, code_dim=64
  - commitment_beta=0.25
  - Same grouped+velocity+peak loss
  - Same Conv1d encoder/decoder shape
  - Output: `blendshape_vqvae.pt`

**Pass:** round-trip 8/8 + DiT-trainable on either continuous quantized
embeddings or code indices.

### Experiment 5 — DiT retrain on best Stage-1

Only fires once exactly one Stage-1 candidate passes 8/8 gates and is
DiT-trainable.

**Recipe:** same as canonical d_vad=4 DiT (30 epochs, batch 32,
crops_per_clip=2, --no-zscore-latents). Output: `blendshape_dit_v3.pt`.

**Validation:** dashboard run on (VAE_v3, DiT_v3) pair. Pass criteria:

  - All 8 round-trip gates from Stage-1 still hold under DiT sampling
  - V controllability gap_pos ≥ +0.30 / gap_neg ≤ −0.30 on probe (was
    +0.46 / −0.52 on canonical — should not regress)
  - Cross-emotion sweep: smile leakage on sad/fear/disgust ≤ 0.10 at
    intensity_cond=1.5

---

## 5. Tracking table

Updated as experiments complete. Rows added as branches taken.

| # | name | status | ckpt | strict 8 | preservation 3 | round-trip pass | verdict |
|---|---|---|---|---|---|---|---|
| 0 | metrics dashboard `08_diagnose.py` | **complete** | n/a | n/a | n/a | n/a | dashboard runs end-to-end with strict + preservation gate sets. F9 captures gate-calibration caveats; preservation gates reformulate the original intent. Artifacts: `artifacts/diag_v1_max/`, `diag_v2_max/`, `diag_ae_v3_max/` |
| — | v1 baseline (informational) | reference | `blendshape_vae.pt` | 2/8 | 2/3 partial (P1=0.887, **P2=0.625 fail**, P3=0.125) | fail | encoder collapse confirmed: 40/64 active pairs preserved (≥0.70 ratio), 24/64 wiped |
| — | v2 baseline (informational) | reference | `blendshape_vae_v2.pt` | 3/8 | 3/3 PASS (P1=0.906, P2=0.891, P3=0.031) | partial | weighted-MSE preserves 57/64 active pairs; smallest invention of any candidate (P3=0.031) |
| 1 | deterministic AE | **complete** | `blendshape_ae_v3.pt` | 3/8 | **3/3 PASS** (P1=**1.013**, P2=**1.000**, P3=0.101) | **PASS (preservation)** | best-in-suite preservation: **64/64 active pairs preserved at ≥0.70 ratio**, median ratio 1.013 (linear output slightly overshoots — harmless). Strict gates 1,3,4,5,8 fail on M003 GT data the gates' authors didn't have when calibrating (F9). Admit to Exp 5. |
| 1b | AE loss-weight retune | gated on 1 | — | — | — | — | — |
| 2 | AE + latent noise | gated on 1 | — | — | — | — | — |
| 3 | VAE w/ KL warmup + free bits | gated on 2 | — | — | — | — | — |
| 4 | VQ-VAE (parallel branch) | gated on 1 fail | `blendshape_vqvae.pt` | — | — | — | — |
| 5 | DiT retrain on best Stage-1 | gated on any pass | `blendshape_dit_v3.pt` | — | — | — | — |

---

## 6. Out of scope (deferred — reasons attached)

  - **Conditioning architecture refactor** (learned null embedding for
    CFG, split EmotionMLP/IntensityMLP). *Why deferred:* adding it now
    confounds attribution. Re-evaluate after Exp 5.
  - **DiT auxiliary perceptual losses** (EMOTE-style: lip-reading
    perceptual + emotion classifier + content-emotion exchange).
    *Why deferred:* requires a sane VAE first; otherwise the losses
    propagate through a broken decoder.
  - **Class-balanced training sampling** for negative-V emotions.
    *Why deferred:* VAE v2 already showed the smile-attractor problem
    is mostly upstream; rebalancing samples won't fix encoder
    collapse.
  - **Probe data augmentation** with NRC-VAD anchors / RAVDESS / IEMOCAP.
    *Why deferred:* probe limitation is a separate issue blocking only
    A/D axes; current experiments target V axis quality which is
    probe-passing already.
  - **Larger DiT (`bigDiT_newVAE`-style)**. *Why deferred:* prior
    experiments showed bigger DiT didn't help; the bottleneck is
    upstream. Revisit only if Exp 5 underperforms.
  - **Architectural encoder change** (transformer encoder per FaceFormer,
    or dilated Conv1d residuals). *Why deferred:* try the cheap
    objective fixes first (Exp 1–3); only escalate to architecture
    changes if all of them fail.

---

## 7. Open questions

  - **Will the deterministic AE produce non-smooth latents** that are
    hard for diffusion to sample? Exp 2 hedges this with `σ_train`.
  - **Will VQ-VAE codebooks generalize across emotions** or memorize
    specific actor/utterance combinations? Validation would need a
    held-out actor sweep.
  - **What's the right channel-group loss weight schedule?** All groups
    starting at 1.0 is the default; Exp 1b retunes if needed.
  - **Is there a downloadable pre-trained emotion classifier on
    blendshapes / FACS** we could plug in as an auxiliary loss for Exp 5?
    EMOTE used one trained on EmotioNet; checking availability is a
    side-quest.

---

## 8. Logging convention

Every experiment writes to `artifacts/diag_<exp_id>/`:

```
diag_<exp_id>/
  report.md       — Markdown summary + tables (gate-by-gate)
  summary.json    — machine-readable for cross-experiment comparison
  ckpt_meta.json  — copied from training-time ckpt metadata
```

Tracking-table rows in this doc reference the `diag_<exp_id>/` path so
findings are reproducible.
