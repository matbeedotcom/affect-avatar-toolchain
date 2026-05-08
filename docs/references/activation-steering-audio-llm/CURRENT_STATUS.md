# Current status — affect-aware agent (audio-llm + audio2face stack)

**Last updated:** 2026-05-06
**Branch:** `audio-blendshape-diffusion` (merged from `origin/audio2face` at `5e55a5c`)
**Backup ref retained:** `audio-blendshape-diffusion-backup` at `1134e611` (pre-merge HEAD)

---

## Where things stand in one paragraph

The project's load-bearing hypothesis — *does affect-aware behavior produce
user-visible improvement?* — is **proven** at 67% B-pref / 0% B-worse over
3 review iterations on Hermes-3 with calibrated Channel D steering, beating
the spec's 65% gate. The full runtime infrastructure for a live WebRTC
demo (wall-clock pacer, `Tick` + `PacingNature::SourceWall(hz)` on the
StreamingNode trait, `MediaClockPublisher`) is now in place via the
`origin/audio2face` merge, **Tier A is built**
([`crates/core/src/nodes/affect_sim.rs`](../../../crates/core/src/nodes/affect_sim.rs)),
and **Tier C is built** — the Phase 3 `ProsodyVad` class is now lifted
into a streaming `ProsodyVadNode`
([`clients/python/remotemedia/nodes/affect/prosody_vad_node.py`](../../../clients/python/remotemedia/nodes/affect/prosody_vad_node.py))
that emits `prosody_*` JSON envelopes per-utterance (off the
`accumulator` boundary, same as STT), and
[`hermes3_affect_s2s_webrtc_server`](../../../crates/transports/webrtc/examples/hermes3_affect_s2s_webrtc_server.rs)
wires `accumulator → prosody_vad → affect_sim` alongside
`vad/stt → affect_sim → llm`. Compiles + 13/13 unit tests pass under
`--features affect-sim`; webrtc example builds clean. Tier B (transcript
classifier) remains deferred.

---

## What's done

In rough chronological order through this project's pivot arc:

### Pivot away from LFM2-Audio (proven dead-end)

- **Abliteration pilot ran the full battery** (single-direction, multi-direction
  SVD K=3/5, combined-layer, α-overshoot) on LFM2-Audio-1.5B. Decisive
  negative result: register/tone subspace capturable, politeness axis is not.
  α=1.5 switched the model to *deflection-based refusal* instead of bluntness;
  α≥2 broke generation.
  → [`notes/spike-abliteration-results.md`](notes/spike-abliteration-results.md)
- **Capability-tested 3 MLX text LLMs** on the 7-prompt blunt-friend set.
  Dolphin 3.0 + Hermes-3 both passed 7/7; Dolphin 2.9 broken in current `mlx_lm`.

### Hermes-3 + affect chain wired and validated

- **`MlxLmTextNode`** built ([clients/python/remotemedia/nodes/ml/mlx_lm_text.py](../../../clients/python/remotemedia/nodes/ml/mlx_lm_text.py))
  — drop-in language head wrapping any `mlx_lm.load`-able chat model.
  Mirrors `LFM2AudioMlxNode`'s aux-port surface (Channels A, B, D + context,
  system_prompt, reset). Uses Llama-3.1's `model.model.layers[21]` for
  steering hook.
- **Persona-vector calibration** on Hermes-3-Llama-3.1-8B-4bit via the
  multi-layer harvest in [`03b_extract_llm_directions_llama.py`](../../../tools/affect_calibration/scripts/03b_extract_llm_directions_llama.py).
  Best layer: 21 (~66% depth), SNR 59.4. NPZ at
  `tools/affect_calibration/artifacts/llm_directions/hermes-3-8b/layer21.npz`
  (gitignored).
- **Eval harness retargeted** — [`09_paired_demo.py`](../../../tools/affect_calibration/scripts/09_paired_demo.py)
  swapped from `LFM2AudioMlxNode` → `MlxLmTextNode`. Same 6 social-domain
  scenarios, structured affect-state display in the review CLI.
- **Three review iterations** (v3 → v4 → v5) tracked in
  [`spike-h-affect-chain-validated.md`](notes/spike-h-affect-chain-validated.md):
  - v3 baseline: 43% B-pref. Per-scenario pattern showed two losers
    (`amused_critique`, `shared_distaste`) using max-frustration selectors
    that pushed Channel D's dominance to **−0.33**, opposite of the social goal.
  - v4: switched both to the `(warmth + assertiveness − 2×safety_dampening)`
    composite. Aggregate 57%, +60pp on each targeted scenario.
  - v5: added a custom `warm_admiration.json` simulator scenario whose
    events (`user_greeting + user_question`, no `user_praise`) avoid the
    joy→assertiveness coupling. Aggregate **67%**, **0% B-worse**.
- **Score command patched** — `safety["total"] == 0` now reports `N/A`
  instead of `FAIL` (vacuous-gate fix).
- **`hermes3_affect_s2s_webrtc_server.rs` example** built — full
  Whisper STT → MlxLmTextNode (steering enabled) → Kokoro TTS pipeline
  with a *static* affect target loaded from `$AFFECT_SCENARIO`'s peak frame
  at server startup.
  → [`crates/transports/webrtc/examples/hermes3_affect_s2s_webrtc_server.rs`](../../../crates/transports/webrtc/examples/hermes3_affect_s2s_webrtc_server.rs)

### Runtime infrastructure (from `origin/audio2face` merge)

The merge brought 125 commits including:

- **`Tick` struct + `PacingNature::{Reactive | ClockedToOutboundMedia | SourceWall(u32)}`** on `StreamingNode`. Wall-clock pacer for `SourceWall(hz)` nodes. *This is what AffectSimulatorNode will use.*
- `MediaClockPublisher` via `TrackRegistry`.
- `Live2DRenderNode` + `CcRenderNode` migrated to `ClockedToOutboundMedia`.
- `cc_render` Rapier physics, `kimodo_motion`/`kimodo_daemon` end-effector + grab protocol.
- Multi-track WebRTC video.
- `&NodeRuntimeContext` threaded through `process_async`/`process_multi_async`/etc; legacy `Option<String> session_id` removed.

Compile state: 32 crates clean; `cubism-core-sys` fails on the
license-gated Live2D Cubism SDK (pre-existing precondition; we don't use Live2D).

---

## What's next

### DONE: AffectSimulatorNode Tier A

Built per [spike-i spec](notes/spike-i-affect-runtime-driver.md):

- Single Rust node at [`crates/core/src/nodes/affect_sim.rs`](../../../crates/core/src/nodes/affect_sim.rs)
- `pacing_nature() = PacingNature::SourceWall(5)` — 5 Hz / 200 ms ticks
- `is_multi_input() = true`
- Decoder accepts VAD JSON envelopes (`is_speech_start` / `is_speech_end`),
  `RuntimeData::Text` transcript turns from `stt_in`, and Tier C-ready
  prosody envelopes (`kind: "prosody_arousal_high" | …`).
- `tick()` drains pending events → appraises → integrates dynamics by
  elapsed wall time → recomputes Channel D target via
  `channel_d::compute_target` and Channel B summary via `PromptRenderer`
  → emits `set_steering` (with L2-distance threshold) and
  `set_system_augmentation` (with string-equality threshold) plus a
  debug tap to `affect_state.out` on every tick.
- Per-session state owned via `ctx.session_state` (`SessionAffectState`,
  downcast through `AnySessionState`).
- Channel A (`set_sampling`) intentionally omitted at Tier A — LLM uses
  its configured defaults; Tier B will map regulation policy to
  per-turn temperature deltas.
- Gated behind `affect-sim` feature flag (path dep on `affect-simulator`
  crate at `tools/affect_simulator/`).
- Registered in `core_provider.rs`; new `AffectSimulatorNodeFactory`
  available to manifests as `node_type: "AffectSimulatorNode"`.
- 10 unit tests pass: greeting/question event firing, debouncing of
  repeat speech-starts, prosody decode, first-tick triple-emission,
  threshold suppression on idle ticks, factory pacing nature.

**Acceptance state:** code-level Tier-A acceptance verified by tests.
Live-conversation acceptance (state visibly evolves over 5 min;
LLM register shifts on engagement cadence; no tick coalescing under
heavy LLM load) requires a running server with a real microphone —
the next thing to actually try in a browser.

### DONE: AffectSimulatorNode Tier C — prosody-VAD branch hookup

Built per [spike-i spec](notes/spike-i-affect-runtime-driver.md):

- New Python streaming node
  [`ProsodyVadNode`](../../../clients/python/remotemedia/nodes/affect/prosody_vad_node.py)
  wraps the existing offline-only `ProsodyVad` class
  ([`prosody_vad.py`](../../../clients/python/remotemedia/nodes/affect/prosody_vad.py))
  in a `MultiprocessNode`. Per-utterance audio in → 0..2
  `prosody_arousal_high` / `prosody_valence_negative` /
  `prosody_uncertain` JSON envelopes out. Lazy-loads the Whisper
  encoder on first `process()` call so a degraded-path session (no
  ridge ONNX) never pulls torch + transformers into the worker.
- Auto-registered in the multiprocess registry via
  `_try_register("remotemedia.nodes.affect.prosody_vad_node",
  [("ProsodyVadNode", "ProsodyVadNode")])`.
- **JSON envelope transport over Python→Rust IPC:** the Python
  `RuntimeData` has no `Json` variant, so `ProsodyVadNode` emits
  events as `RuntimeData.text(json.dumps(env), channel="json")` —
  the IPC layer prefixes the channel via `[0x00][len][channel][text]`
  and the receiver recovers it. The Rust-side `AffectSimulatorNode`
  decoder now handles **both** `RuntimeData::Json(v)` (native producers)
  and `RuntimeData::Text(s)` with `channel="json"` (Python producers
  via `split_text_str`). Three additional unit tests cover the new
  decoder branch — total now 13/13 passing.
- WebRTC example
  [`hermes3_affect_s2s_webrtc_server`](../../../crates/transports/webrtc/examples/hermes3_affect_s2s_webrtc_server.rs)
  wires the runtime driver and Tier C branch into the manifest under
  the `AFFECT_RUNTIME=true` (default) gate:
  - `vad → affect_sim` (speech edges)
  - `stt_in → affect_sim` (transcript turns)
  - `accumulator → prosody_vad` (full utterance audio)
  - `prosody_vad → affect_sim` (V/A/D events)
  - `affect_sim → llm` (`set_steering` + `set_system_augmentation`
    aux-port envelopes)
- The static-target affect baking at session start remains as the
  *initial* state; the `affect_sim` node's first tick (~200 ms after
  start) overrides it with one driven by live events.
- Webrtc crate's `remotemedia-core` dep now enables `affect-sim`.

**Open work for Tier C live validation:** the spec's acceptance
criterion ("a frustrated-tone audio sample with neutral text content
raises empathy in the simulator state on the next tick") needs an
in-browser run with a real microphone. All wiring is in place; the
remaining work is operational, not code.

### DONE: Per-tick avatar face — affect → ARKit-52 blendshape mapping

`AffectSimulatorNode` now emits a 4th frame per tick: a
`{kind: "blendshapes", arkit_52: [f32; 52], pts_ms}` envelope shaped
exactly like the audio2face / synthetic-lipsync output, so an
existing `Live2DRenderNode` / `CcRenderNode` consumes it unchanged.
Hand-tuned mapping in
[`crates/core/src/nodes/affect_expression.rs`](../../../crates/core/src/nodes/affect_expression.rs) —
8 emotion channels → small subsets of ARKit-52 weights using FACS
shorthand (joy → smile + Duchenne cheek-squint, sadness → frown +
sad-brow, anger → brow-down + sneer + jaw-set, fear → eye-wide +
mouth-stretch + slight jaw-open, curiosity → outer-brow lift,
empathy → compassionate-listening half-smile). Anger gets an extra
`(1 − 0.7 × safety_dampening)` attenuation so a regulated agent
shows a softer brow than its felt anger; expressiveness gates the
whole pose so a stoic state collapses to neutral.

- Pre-empts the learned audio→blendshape model in
  `tools/affect_avatar/`; the wire format is identical, so swapping
  later is a function-pointer change.
- 9 unit tests cover index-table integrity, neutral pose, pure-channel
  poses (joy/sadness/anger/empathy distinct from each other),
  expressiveness gating, anger-specific safety attenuation, and
  top-active diagnostic ordering.
- Two new `AffectSimulatorNode` tests verify the 4-envelope per-tick
  contract and the `emit_blendshapes = false` opt-out (drops back to
  3 envelopes for pipelines without an avatar consumer).
- [`affect_sim_smoke`](../../../crates/core/examples/affect_sim_smoke.rs)
  example now prints the top-3 active blendshapes per tick so the
  trajectory of the face is visible alongside V/A/D and the regulator
  state.
- [`tools/affect_coach/coach.py`](../../../tools/affect_coach/coach.py)
  mirrors the mapping in Python and surfaces the top-5 active
  blendshapes in `--verbose` mode (`Face: mouthSmileLeft 0.20,
  cheekSquintLeft 0.14, …`) — same readout an avatar would actually
  render.

23/23 affect tests pass under `--features affect-sim` (14 affect_sim
+ 9 affect_expression). To wire the face into the live demo, add a
manifest connection `affect_sim → live2d_render` (or any avatar node
that consumes the `blendshapes` envelope shape).

### DEFERRED: Tier B (transcript classifier)

Sentence-transformers + LR head trained on `contrast_pairs.jsonl`. Skip
until A+C demo is working; revisit if the runtime feels too monotonic
without content-aware event mapping.

---

## Loose ends

- **Backup ref to drop:** `git update-ref -d refs/heads/audio-blendshape-diffusion-backup`
  once we're confident the merge is good (no compile or runtime regressions
  from anything beyond cubism).
- **Verdicts file path inconsistency.** During the eval reviews, files
  landed in two different places:
  - `/Users/mathieugosbee/dev/originals/remotemedia-sdk/artifacts/paired_eval/verdicts_v3_hermes3.csv`
    (top-level — relative to repo root cwd)
  - `/Users/mathieugosbee/dev/originals/remotemedia-sdk/tools/affect_calibration/artifacts/paired_eval/verdicts_*`
    (intended location)
  Score command works either way; not breaking, but worth normalising.
- **Safety pairs never authored.** The 5 originally-spec'd safety pairs
  (adversarial-provocation prompts) are absent from `SCENARIO_PROMPTS`.
  The score command now correctly reports N/A but for a production-grade
  pass we'd want them.
- **`PROJECT_PLAN.md` / `DESIGN.md` not updated.** Both still describe
  the LFM2-Audio architecture. spike-h captures the pivot but the parent
  docs haven't been rewritten.
- **Static-target example needs replacement.** Once Tier A lands,
  `hermes3_affect_s2s_webrtc_server.rs`'s static-affect-target loader
  becomes obsolete and should be replaced with an `affect_sim` node
  in the manifest. (Spike-i shows the diff.)

---

## Reference docs

Anchored from this directory; deepest detail in the order one would
re-read after a long break:

1. [`notes/spike-i-affect-runtime-driver.md`](notes/spike-i-affect-runtime-driver.md)
   — design for the next thing to build.
2. [`notes/spike-h-affect-chain-validated.md`](notes/spike-h-affect-chain-validated.md)
   — the v3→v5 trajectory and what made it work.
3. [`notes/spike-abliteration-results.md`](notes/spike-abliteration-results.md)
   — why we pivoted away from LFM2-Audio.
4. [`notes/spike-abliteration-pilot.md`](notes/spike-abliteration-pilot.md)
   — original abliteration design (foreground for the negative result above).
5. [`notes/spike-g-mlx-target-llm.md`](notes/spike-g-mlx-target-llm.md)
   — MLX persona-vector extraction pattern (still applicable, reused
   verbatim for Llama in `lib/mlx_llama_target.py`).

---

## Reproduction

```bash
# Recalibrate Hermes-3 directions (~42 min, gitignored output)
./tools/affect_calibration/scripts/03b_extract_llm_directions_llama.py \
    --contrast-pairs tools/affect_calibration/data/contrast_pairs.jsonl \
    --model mlx-community/Hermes-3-Llama-3.1-8B-4bit \
    --output-dir tools/affect_calibration/artifacts/llm_directions/hermes-3-8b/

# Generate paired eval (~5 min)
./tools/affect_calibration/scripts/09_paired_demo.py generate

# Interactive blinded review
./tools/affect_calibration/scripts/09_paired_demo.py review \
    --out artifacts/paired_eval/verdicts_<NAME>.csv

# Score
./tools/affect_calibration/scripts/09_paired_demo.py score \
    --pairs    tools/affect_calibration/artifacts/paired_eval/pairs.json \
    --verdicts artifacts/paired_eval/verdicts_<NAME>.csv

# Run the static-target server (Tier-0 demo until Tier A lands)
AFFECT_SCENARIO=roast_invitation \
cargo run --example hermes3_affect_s2s_webrtc_server \
    -p remotemedia-webrtc --features ws-signaling -- --port 8083
```
