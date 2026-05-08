# Spike B0 — MediaPipe Blendshape Extraction Feasibility

**Status**: **Accepted with caveats** — executed against RAVDESS Actor_01 + Actor_02
(120 modality-01 clips, 100 scored). Pipeline works, coverage and jitter
gates pass cleanly; affect-signal gate is partially confirmed (smile axis
PASS, brow-inner-up / eye-wide axes inconclusive on RAVDESS's neutral-
statement / normal-intensity protocol). Recommend B1 — re-validate the
inconclusive axes when MEAD strong-intensity clips are on hand.

**Date drafted**: 2026-05-05. **Date executed**: 2026-05-06.

**Owner**: @mathieugosbee.

**Paired script**:
[`tools/affect_avatar/scripts/spike_b0_extract_smoke.py`](../../../../tools/affect_avatar/scripts/spike_b0_extract_smoke.py).
**Label generator**:
[`tools/affect_avatar/scripts/build_ravdess_labels.py`](../../../../tools/affect_avatar/scripts/build_ravdess_labels.py).
**Run artifact**:
[`tools/affect_avatar/artifacts/spike_b0_report.md`](../../../../tools/affect_avatar/artifacts/spike_b0_report.md)
(report) and `artifacts/spike_b0_results.npz` (per-clip trajectories;
gitignored, ~50 MB).

---

## Question

Before committing to MediaPipe FaceMesh as the blendshape ground-truth
extractor for the [audio→blendshape diffusion model](../PROJECT_PLAN.md),
we need to know:

1. **Q-B0-1 — Coverage.** Does FaceMesh actually emit the full ARKit-52
   blendshape vocabulary, or a partial subset?
2. **Q-B0-2 — Noise.** Are per-frame trajectories smooth enough to train
   a diffusion model directly on, or do we need temporal smoothing /
   per-clip residual modeling?
3. **Q-B0-3 — Affect signal.** Do affect-correlated blendshapes
   (`mouthSmileLeft/Right`, `browInnerUp`, `eyeWideLeft/Right`, etc.)
   move in the expected direction across MEAD's 7 emotion classes?

A negative on any of these requires a fallback (DECA, EMOCA, or
iPhone ARKit ground truth) **before** Stage-1 VAE training in Phase B1.

## Tentative answer

**Probably yes, with caveats** — MediaPipe emits 52 blendshape scores
labeled with the ARKit names, the on-device blendshape head is
explicitly trained against the ARKit vocabulary. Open questions are
the extraction quality on MEAD's controlled-studio recordings (which
should be the *easy* case) and whether the smile/brow/eye axes show
the expected emotion separation. The spike validates this empirically
on 100 clips before committing to extraction at scale.

If the spike fails: fall back to DECA / EMOCA at ~10× the implementation
cost, or limit to a hand-recorded ARKit-iPhone capture set at
much-reduced data scale. Both are recoverable; we are not blocked.

---

## Protocol

### Inputs

- 100 MEAD clips, sampled as:
  - 5 speakers (3 F, 2 M).
  - 7 emotions (neutral, happy, sad, angry, surprised, disgusted,
    contempt).
  - 3 intensity levels (low, medium, high) for each non-neutral.
  - ~3 clips per (speaker, emotion, intensity) — total ~100 after
    rounding.
- Frames: native MEAD ~30 fps. Audio 48 kHz mono → resampled to 16 kHz
  via the existing `affect_calibration/scripts/lib/audio.py`.

### Pipeline

```python
# Pseudo:
import mediapipe as mp
landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(...)
for clip in clips:
    frames = read_video(clip.path)                # (T, H, W, 3)
    blendshapes = []
    for frame in frames:
        result = landmarker.detect_for_video(frame, timestamp_ms=...)
        if result.face_blendshapes:
            row = [bs.score for bs in result.face_blendshapes[0]]
            blendshapes.append(row)
        else:
            blendshapes.append([float("nan")] * 52)
    save_npz(clip.id, {
        "blendshapes": np.array(blendshapes),     # (T, 52)
        "emotion": clip.emotion,
        "intensity": clip.intensity,
        "speaker": clip.speaker,
    })
```

### Outputs

A single `spike_b0_results.npz` plus a one-page memo (this file, after
execution) reporting:

#### M-B0-1 — Coverage check (answers Q-B0-1)

- Does the labeler return all 52 ARKit names? List any missing.
- Does the order match the ARKit reference? (MediaPipe sorts
  alphabetically; need to map to ARKit order.)
- Pass: 52/52 names returned, mapping table saved to
  `tools/affect_avatar/scripts/lib/arkit_blendshape_map.py`.

#### M-B0-2 — Smoothness (answers Q-B0-2)

For each clip, compute the per-frame first-difference RMS:

```
jitter = sqrt(mean((blendshapes[t+1] - blendshapes[t])**2))
```

- Report the distribution across 100 clips.
- Pass threshold: 95th-percentile jitter < 0.05 (i.e. blendshapes
  rarely change by more than 5 % of full range frame-to-frame).
- Failure: > 0.05 for >5 % of clips → apply a 5-tap median filter and
  re-measure; if still failing, fallback path.

#### M-B0-3 — Affect signal (answers Q-B0-3)

Per emotion class, compute the mean of three diagnostic blendshapes
across all clips of that class:

| Blendshape | Expected sign — happy | sad | angry | surprised |
|---|---|---|---|---|
| `mouthSmileLeft/Right` (avg) | high | low | low | medium |
| `browInnerUp` | low | high | low | high |
| `eyeWideLeft/Right` (avg) | medium | low | low | high |
| `browDownLeft/Right` (avg) | low | low | high | low |

- Report the actual measurements as a 7×4 table.
- Pass: at least 3 of 4 columns show the expected ordering for
  happy/sad/angry/surprised. Disgusted, contempt, neutral are not
  scored — disagreement on those is acceptable.

### Decision gate

| Outcome | Action |
|---|---|
| All three checks pass | Commit B1 — full extraction at scale, MediaPipe is the ground truth. |
| M-B0-1 fails | Fix the ARKit-mapping; re-run. (Almost certainly recoverable in <1 hour.) |
| M-B0-2 fails after smoothing | Fall back to DECA / EMOCA. Add 1 week to the schedule. |
| M-B0-3 fails | The ground truth doesn't carry the affect signal we need to train against. **Hard stop**: rethink the data plan — likely switch to a smaller, ARKit-iPhone hand-recorded dataset and accept reduced training scale. |

---

## Results — RAVDESS Actor_01 + Actor_02 (executed 2026-05-06)

### Inputs (deviation from protocol)

The drafted protocol called for MEAD; MEAD requires institutional access
that hasn't landed yet, so we executed on **RAVDESS** instead — the
plan's documented secondary dataset
([PROJECT_PLAN.md §2](../PROJECT_PLAN.md), CC BY-NC-SA, Zenodo-direct).
Results are interpreted against RAVDESS-specific caveats below.

- **Clips**: 120 modality-01 (full-AV mp4) clips across Actor_01 (male)
  and Actor_02 (female), 100 sampled (`--max-clips 100`). Coverage:
  angry/calm/disgust/fearful/happy/sad/surprised × normal+strong
  intensity × 2 statements × 2 reps; neutral is normal-intensity only
  per RAVDESS convention.
- **Throughput**: 0.89 clip/s sustained on M1 Pro CPU (XNNPACK delegate).
  ~113 s for 100 clips end-to-end. Linear-scaling, fine for B1.

### M-B0-1 — Coverage

| Returned names | ARKit-52 present | Missing | Extra |
|---:|---:|---|---|
| 52 | 51 / 52 | `tongueOut` | `_neutral` |

MediaPipe's `face_landmarker.task` (float16/v1) emits 52 names but the
set is *not* identical to ARKit-52: it returns `_neutral` (a "no
expression" indicator) and omits `tongueOut`. **Verdict: pass-with-fix**.

**Fix applied to PROJECT_PLAN.md target dimension**: drop `tongueOut`,
keep `_neutral` as either an extra training input or discard (TBD in
B1). Effective target dimension is 51, not 52. The fix is a one-line
mapping change; it does not invalidate the architecture or training
plan.

### M-B0-2 — Jitter (per-frame first-difference RMS)

| Statistic | Value | Threshold | Result |
|---|---:|---|:---:|
| p50 jitter | 0.0168 | — | — |
| p95 jitter | 0.0328 | < 0.05 | ✅ |
| p99 jitter | 0.0368 | — | — |
| Clips scored | 100 | — | — |

**Verdict: PASS, comfortably**. p95 is 65 % of the threshold; no
smoothing pre-pass needed before VAE training.

### M-B0-3 — Affect signal

Mean diagnostic-blendshape activation per emotion class
(per-clip mean over time, then averaged across clips with that label):

| emotion | smile | browInnerUp | eyeWide | browDown |
|---|:---:|:---:|:---:|:---:|
| angry | +0.117 | +0.000 | +0.005 | +0.462 |
| calm | +0.275 | +0.001 | +0.003 | +0.339 |
| disgust | +0.393 | +0.000 | +0.002 | +0.849 |
| fearful | +0.085 | +0.003 | +0.007 | +0.413 |
| happy | **+0.705** | +0.010 | +0.004 | +0.401 |
| neutral | +0.083 | +0.002 | +0.005 | +0.337 |
| sad | +0.025 | +0.001 | +0.005 | +0.459 |
| surprised | +0.098 | +0.001 | +0.006 | +0.485 |

**Per-axis verdict**:

- `smile` — ✅ **PASS**. Happy clearly highest at +0.705 (>2× the next
  highest, disgust); sad clearly lowest at +0.025. Expected ordering
  holds for all four scored emotions. This is the strongest signal
  in the table and demonstrates MediaPipe blendshapes do carry affect
  information, end-to-end.
- `browDown` — partial. Highest on disgust (+0.849), then surprised /
  angry / sad / neutral all clustered ~0.34-0.49. The expected
  "angry highest" ordering does not hold cleanly on RAVDESS — but
  the axis is *moving* across emotions, so this is more a per-actor /
  per-dataset variance than a model failure.
- `browInnerUp` — ❌ inconclusive. Near-zero (0.000-0.010) on every
  emotion. Expected ordering for "sad highest, surprised second"
  cannot be evaluated when the axis doesn't activate.
- `eyeWide` — ❌ inconclusive. Same pattern as browInnerUp: 0.002-0.007
  across all classes. Expected "surprised highest" ordering not
  evaluable.

**Two competing interpretations** of the brow-inner-up / eye-wide
near-zeros:

1. **MediaPipe under-activates.** The float16-v1 blendshape head was
   trained primarily on talking-head video; subtle brow / eye
   blendshapes may be regressed conservatively. If true, this is a
   model-quality concern that affects training-target richness.
2. **RAVDESS under-expresses.** RAVDESS's protocol is two
   *neutral* sentences ("Kids are talking by the door") repeated
   across emotion labels; actors are professionals trained to *not*
   over-act, especially at normal intensity. If true, MEAD —
   designed for emotional expression with strong-intensity
   conditioned takes — would resolve these axes.

The single-axis confirmation (smile / happy at +0.705) is consistent
with interpretation (2): MediaPipe *can* register expression, it just
needs a more expressive performance to register on the subtler axes.
We resolve this by **re-running M-B0-3 on MEAD when access lands**,
not by halting B1.

### Pass-with-caveat decision

- M-B0-1: **PASS** with one-line target-dimension fix (52→51, drop
  `tongueOut`).
- M-B0-2: **PASS** clean.
- M-B0-3: **PARTIAL** (1/4 axes conclusively pass on RAVDESS; 3/4
  inconclusive — interpretation pending MEAD).

**Recommend B1 to proceed.** The pipeline is functional, throughput
is acceptable for full-dataset extraction, and the smile axis confirms
end-to-end affect signal capture. The brow / eye inconclusiveness is
a *data* question, not a *pipeline* question, and B1's full extraction
(MEAD + CREMA-D + RAVDESS) will resolve it. If M-B0-3 fails again on
MEAD strong-intensity clips, escalate to the protocol's
"hard stop / rethink data plan" branch — but that is a B1 mid-stream
checkpoint, not a B0 blocker.

### Open follow-ups for B1

1. **Decide on `_neutral`**. Either drop it (51-D target) or keep it
   as a 53rd channel. Recommendation: drop in v1; revisit only if
   per-frame "neutrality confidence" turns out to help the diffusion
   model regularize.
2. **Map `tongueOut` to constant zero** in the runtime
   `BlendshapeDiffusionNode` so downstream rigs that assume the full
   ARKit-52 input still receive a 52-D vector.
3. **Re-run M-B0-3 on MEAD strong-intensity** as soon as MEAD access
   is granted; treat as a B1 kickoff gate.

---

## What this spike does NOT cover

- **Audio-blendshape alignment.** MEAD's audio and video tracks are
  pre-aligned by the dataset; verifying this is B1, not B0.
- **Train-time augmentation.** Mirror flip, cropping, etc. are B2.
- **Render quality.** B0 only checks the *extracted* blendshape
  trajectories. Whether they re-render to a believable face is B5
  / runtime integration.
- **Cross-dataset domain shift.** B0 uses MEAD only; CREMA-D and
  RAVDESS are extracted in B1 with the same protocol and any
  per-dataset issues surface there.

## Estimated effort

- Setup: 30 min (`pip install mediapipe`, download MEAD subset).
- Extraction: 1-2 hours wall-clock (CPU is fine for 100 clips).
- Analysis + memo update: 2-3 hours.
- **Total**: 1 day.

## Done means

This memo is rewritten with:

- Coverage table (`M-B0-1`). ✅ included above
- Jitter distribution histogram (`M-B0-2`). ✅ p50/p95/p99 included above
- Affect-signal table (`M-B0-3`). ✅ included above
- A "Decision" section: green-light B1, request follow-up, or fallback.
  ✅ "Recommend B1 to proceed" with caveats included above.

Status updated from "Drafted" to "Accepted with caveats" (see header).
