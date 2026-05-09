# Listener-mode plan — assistant face reacts to user audio

Sketch for retargeting the existing speaker-conditioned AE+DiT into a
listener that reacts to incoming user speech instead of generating its
own. Companion to [`STAGE1_VAE_PLAN.md`](STAGE1_VAE_PLAN.md).

## 1. Why this is non-trivial

MEAD is a speaker dataset:
  - **Audio** = the trained subject's own voice.
  - **Face** = that same subject's lip-sync + emotion expression.
  - **Emotion label** = the subject's *intended* emotion, not a reaction.

Our current pipeline learned `(user_audio_via_Whisper, V/A/D) → speaker_face`.
Naively reusing it as `(user_audio, user_VAD) → assistant_face` produces:
  - **Wrong**: assistant lip-syncs to the user's words.
  - **Wrong**: assistant mouth opens because the user's audio has high RMS.
  - **Right**: brow / eye / cheek conditioning still tracks emotion VAD —
    those channels generalise across speaker-vs-listener.

So the model isn't broken, but ~half its output is mis-targeted. Fix
the conditioning + output channels, leave the architecture.

## 2. Channel split (MEAD_3D 54-ch indexing)

Listener output should differ from speaker output along *speech-coupled*
channels but match along *affect* channels. Three groups, in increasing
permissiveness for a listener:

| group | indices | listener policy |
|---|---|---|
| **speech-only** (lip-sync, jaw) | 22 jawForward, 23 jawLeft, 24 jawOpen, 25 jawRight, 26 mouthClose, 31 mouthFunnel, 32 mouthLeft, 37 mouthPucker, 38 mouthRight, 39 mouthRollLower, 40 mouthRollUpper | **zero out** at inference. These should not move while listening. |
| **affect-mouth** (smile / frown / dimple / press) | 27/28 mouthDimple, 29/30 mouthFrown, 33/34 mouthLowerDown, 35/36 mouthPress, 41/42 mouthShrug, 43/44 mouthSmile, 45/46 mouthStretch, 47/48 mouthUpperUp | **keep** — these encode empathic mirroring (smile back, sympathetic frown). |
| **upper face** (brow / eye / cheek / nose) | 0–21, 5, 6/7, 49/50 | **keep** — emotion signal lives here. |

Pure "kill all mouth" reads as creepy / dead-lower-face on the renderer.
The split above keeps the affect cues live and only suppresses the
speech-driven lip motion.

## 3. Conditioning swap

Today (speaker mode):
```
whisper_features ← speaker_audio
vad ← MEAD_VAD[speaker_emotion_label]   (training)
vad ← user knob                         (inference)
```

Listener mode:
```
whisper_features ← user_audio  (still the same Whisper hidden states,
                                but it's the user's voice now)
vad ← affect_calibration regressor(user_audio)  ← KEY CHANGE
```

The `affect_calibration` toolchain (sibling directory in this repo)
already trains a `whisper_to_vad.onnx` from labelled audio. That
becomes the listener-mode conditioning source. The DiT itself sees
the same `(whisper, vad)` shape it was trained on.

Open question: should listener still get the cross-attention to user
audio, or should it be silenced? Two options:

  - **A (mirror)**: keep Whisper cross-attn live → assistant face has
    micro-motion timed to user prosody (eyebrow flicks at user emphasis,
    cheek twitch at silences). More alive, slight lip-leak risk.
  - **B (still)**: replace user Whisper with a silence Whisper embedding
    (or a learned null) → assistant face purely tracks slow VAD with no
    audio-rate jitter. Calmer, possibly under-reactive.

Default to A. If the speech-only mask + smoothing don't suppress
audio-rate lip motion, fall back to B.

## 4. Three-phase rollout

### Phase L0 — zero-training inference patch

Goal: working listener prototype this afternoon. No retrain.

  - Add `--listener-mode` flag to [`05_sample_smoke.py`](scripts/05_sample_smoke.py).
  - Inside the sampling loop, after `vae.decode`, zero out the
    speech-only channel indices (per §2 table) on the decoded output.
  - VAD source: temporarily reuse `--vad` CLI argument; later wire to
    the `affect_calibration` regressor.
  - Audio source: feed user audio through the existing Whisper cache /
    encoder path. Same shape, same downstream pipeline.

  *Touchpoints*:
  - `tools/affect_avatar/scripts/05_sample_smoke.py` — add flag + 5-line
    channel-mask post-processor.
  - No model file changes.

  *Validation*: render the same clip as the v3 happy smoke, but with
  `--listener-mode`. Mouth should freeze; brow + cheek + eye should
  still respond to the V/A/D conditioning. Visually verify on the
  rebaked GLB.

### Phase L1 — training-time mouth-channel mask

Goal: AE + DiT internalise that lip-sync isn't their job.

  - Add `--mouth-mask {none, speech_only, all_lower}` to
    [`09_train_ae.py`](scripts/09_train_ae.py) and 04_train_dit.py.
  - In `BlendshapeVAE.ae_loss` and the DiT ε-loss, multiply the
    masked-channel residual by 0 before averaging. Encoder/decoder
    still compute the values; the loss simply doesn't penalise them.
  - Output: `blendshape_ae_v3_listener.pt` and `blendshape_dit_v3_listener.pt`.

  *Touchpoints*:
  - `tools/affect_avatar/scripts/lib/vae.py:ae_loss` — accept `channel_mask`.
  - `tools/affect_avatar/scripts/04_train_dit.py` — same.

  *Why bother over L0*: at inference, L0 just zeros the channels post-hoc;
  the latent space still encodes the lip-sync signal and may bleed into
  neighbouring channels via the decoder. L1 lets those latent dimensions
  reallocate to whatever's still being supervised (upper face + affect
  mouth), producing a cleaner listener latent.

  *Cost*: ~3 hr AE retrain + ~2.6 hr DiT retrain. Same as Exp 5.

### Phase L2 — listener dynamics (rule-based or learned)

Goal: assistant feels engaged, not staring. Add micro-reactions that
MEAD doesn't supply.

Rule-based candidates, all driven from the live audio + VAD trajectory
(no model changes):

  - **Brow flash** on user phrase emphasis (audio-RMS spike past
    threshold) — brief `browInnerUp` pulse, ~150 ms.
  - **Empathic mirror smile** — when user VAD valence > +0.5 sustained
    > 1 s, ramp `mouthSmile` to 0.3–0.5 with 500 ms attack.
  - **Concern frown** — symmetric: user valence < −0.5 sustained → mild
    `browDown` + `mouthFrown` ramp.
  - **Listening nod** — handled at avatar pose level (head pitch
    oscillation), not blendshape. Out of scope for this plan; flag it
    for the rendering layer.

  *Touchpoints*:
  - New file `tools/affect_avatar/scripts/lib/listener_dynamics.py`.
  - Called from `05_sample_smoke.py` after the DiT decode + mask, before
    JSONL write. Adds reactive deltas onto the masked output.

  *Open question*: rules vs. a trained reaction model. Rules are
  immediate and inspectable. A trained model needs listener data
  (LIRIS-CSE / IEMOCAP listener segments) and is L3+.

## 5. Conditioning data wiring

The user-side VAD comes from `tools/affect_calibration/`. Specifically
the `whisper_to_vad` regressor takes Whisper-large-v3-turbo hidden
states and emits `(V, A, D)` per audio frame. Two integration points:

  - **Online (real-time chat)**: regressor runs on user-mic Whisper
    hidden states; emits VAD at ~33 Hz; smooth + feed to the DiT
    sampler at the same 30-fps cadence as the face output.
  - **Offline (batch render)**: regressor runs once over the user
    utterance; output is a `(T_audio, 3)` tensor. The DiT sampler
    accepts a 4-d `(V, A, D, intensity)` per-clip vector today; needs a
    minor extension to accept a per-frame VAD if we want the listener
    face to track shifting user affect within an utterance.

Per-frame VAD conditioning is a small DiT-side change: add a temporal
projection that broadcasts (or interpolates) the VAD trajectory across
the latent timeline, in addition to the existing global VAD broadcast.

## 6. Risks

  - **Dead-lower-face syndrome**. L0 alone may produce a frozen mouth
    that looks uncanny. Mitigation: L2 mirror-smile rule, or relax the
    speech-only mask to leave `mouthSmile` and `mouthFrown` unzeroed.
  - **VAD regressor latency**. If the regressor adds > ~100 ms, the face
    reaction lags speech and looks robotic. Mitigation: run regressor
    on a streaming Whisper window, accept the inherent ~20-frame
    Whisper hop.
  - **Whisper-conditioned lip leak**. The DiT may still drive lip
    motion via cross-attention even with channels masked at decode (the
    latent is still encoding "this audio sounds like /b/"). Mitigation:
    L1 retraining; or option B above (silence Whisper for inference).
  - **Speaker-vs-listener affect mismatch**. MEAD subjects expressing
    "happy" = wide smile because they're delivering a happy line. A
    listener feeling "happy at the user's good news" probably does
    something subtler — a soft smile, not full grin. The current
    intensity scalar (d_vad=4) lets the caller dial it down at
    inference; default `intensity_cond ≈ 0.4` for listener mode.

## 7. What this plan does NOT cover

  - Avatar pose / head motion (nods, lean-in). Handled by the renderer
    layer, not the blendshape pipeline.
  - Listener-side audio response (TTS). Different stack.
  - Multi-turn affect dynamics (listener emotion drift over a long
    conversation). The `affect_simulator` toolchain handles this; it
    would feed VAD into the listener pipeline as a drop-in upstream.
  - Eye gaze. The `eyeLook*` channels exist in MEAD but listener gaze
    follows the user / camera, which MEAD doesn't capture. Out of scope.

## 8. Decision gates

  - **L0 ships if** the rendered listener clip on a known-emotion user
    utterance shows mouth frozen (no jaw-open) AND brow/cheek/eye
    activation matching the user's VAD label. ~1 day.
  - **L1 ships if** L0 has visible audio-rate lip leak that the
    speech-only mask doesn't fully suppress. ~1 week incl. retrain.
  - **L2 ships if** L0/L1 looks correct but feels under-reactive in
    user testing. Rule additions are incremental.
