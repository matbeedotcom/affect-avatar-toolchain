---
name: Affect runtime driver — AffectSimulatorNode design
type: spike-spec
status: design
last_updated: 2026-05-06
---

# AffectSimulatorNode — runtime affect-state evolution

## TL;DR

A new Rust `StreamingNode` that owns the simulator's runtime state during a
live session, advances it on a wall-clock tick, and emits Channel A/B/D
updates as control-bus envelopes to the language-head node. Replaces the
*static* affect target the
[`hermes3_affect_s2s_webrtc_server`](../../../crates/transports/webrtc/examples/hermes3_affect_s2s_webrtc_server.rs)
example loads at startup with **per-turn evolution driven by what's
actually happening in the conversation.**

Built on the pacing-domains infrastructure brought in by the
`origin/audio2face` merge (`Tick`, `PacingNature::SourceWall(hz)`,
`tick()` on `StreamingNode`). The 200 ms simulator clock becomes a
`SourceWall(5)` declaration; the runtime's pacer drives ticks
independent of LLM blocking time, eliminating the "ticks pile up
during heavy generation" problem flagged in
[spike-h §What made this work](spike-h-affect-chain-validated.md).

Builds in three tiers — start with clock-only, add transcript
classification, then prosody. Each tier ships independently and the
demo gets more "alive" at each step.

## Why this, why now

[spike-h](spike-h-affect-chain-validated.md) proved at 67 % B-pref that
the affect chain produces user-visible improvement *given the right
peak frame for the social register*. The eval ran offline on
hand-authored conversation histories with a single peak frame per
scenario. To deliver that effect in a live conversation, the right
peak frame has to evolve over the conversation — and the conversation's
events (VAD speech turns, transcript content, prosody) are what should
drive that evolution.

Until the merge, the wall-clock-driven dispatcher pattern wasn't first
class in the runtime; we'd have had to roll our own with `tokio::spawn`
inside `process_streaming` and accept the timing-jitter that the
existing `audio2face` work had already solved. The merge means we can
land this cleanly the first time.

## Component architecture

```
                                                 ┌──── audio.in.set_steering
                                                 │     audio.in.set_system_augmentation
                                                 │     audio.in.set_sampling
   vad ───────────► ┐                            │
   stt_in ────────► │   AffectSimulatorNode      │
   (optional)       ├──► (Rust, SourceWall(5))  ─┤
   prosody_vad ───► │   ─ ticks at 5 Hz          │
                    │   ─ owns simulator state   │
                    │   ─ publishes A/B/D        │
                    └──┬─►                       │
                       │                         └──── llm (MlxLmTextNode)
                       │
                       └─►  affect_state.out (control-bus tap for UI)
```

**Single node.** The simulator + the runtime driver are one node, not
two — no benefit to separating them, and the `tick()` callback is the
natural boundary.

**Language: Rust.** Justified:
- The `affect-simulator` crate is already Rust; we call into its public
  API directly instead of shelling out + JSON-serialising state.
- `Tick` and `PacingNature::SourceWall(5)` are Rust trait surface;
  Python nodes wouldn't get the wall-clock pacer wired up the same way.
- The control-bus is JSON-serialisable, so downstream Python nodes
  (`MlxLmTextNode`, future classifier) consume the envelopes
  language-agnostically.

**Pacing nature.** `PacingNature::SourceWall(5)` → 5 Hz / 200 ms
ticks. Matches the simulator's `tick_ms`. The runtime's wall pacer
fires `tick()` regardless of upstream input or downstream backpressure;
ticks are catch-up-coalesced in the simulator (the existing
`runner::SimulatorRun` already integrates by elapsed-since-last
rather than fixed 200 ms increments).

**Multi-input on the data side.** `is_multi_input() = true`. The node
receives:
- VAD speech-start / speech-end events (control-bus JSON)
- Transcript text on each user turn (`RuntimeData::Text` from `stt_in`)
- Optional: prosody-VAD events from a Whisper-encoder branch

These arrive *between ticks*; the node buffers them on internal
state, and the next `tick()` consumes them, advances simulator state,
and publishes outputs.

**Output side.** Three control-bus envelopes per emit:

```json
// To LLM (existing aux-port shape in MlxLmTextNode)
{ "__aux_port__": "set_steering",
  "payload": { "target_vad": [v, a, d], "alpha": 1.0 } }

{ "__aux_port__": "set_system_augmentation",
  "payload": { "text": "<channel B state summary>" } }

{ "__aux_port__": "set_sampling",
  "payload": { "temperature": 0.0, "max_new_tokens": 200 } }
```

Plus a debug tap so a UI panel can render the live state:
```json
// To `affect_state.out` (browser-subscribable)
{ "ts_ms": ..., "channels": {...}, "policy": {...},
  "channel_d_target_vad": [v, a, d], "channel_b": "..." }
```

**Connection topology in the manifest:**

```yaml
connections:
  - { from: vad,         to: affect_sim }
  - { from: stt_in,      to: affect_sim }
  # Optional Tier C:
  # - { from: prosody_vad, to: affect_sim }

  # Affect outputs route into the LLM node — its existing
  # `_extract_envelope` branch dispatches them to set_steering /
  # set_system_augmentation / set_sampling.
  - { from: affect_sim,  to: llm }
```

No change to `MlxLmTextNode`'s contract — it already handles
aux-port envelopes through `process()`'s existing branch.

## Tiered rollout

### Tier A — clock + VAD only (~1 day)

Minimum viable runtime. Drives the simulator from VAD events alone:
- `speech_start` (first turn of session) → emit a `UserGreeting` event
  into the simulator
- `speech_end` (every user turn) → emit a `UserQuestion` event
- Inter-turn silence → simulator's existing channel decay does the work

Channel A/B/D updated on each tick when state has changed *materially*
(threshold to suppress no-op spam — see Open Questions).

**What this proves:** the runtime loop works. State drifts visibly
between turns. Demo-able in a conversation that goes a couple of
minutes — the UI's affect-state tap shows channels rising on speech
events and decaying through silence.

**What this doesn't prove:** the agent reacts to *what was said*. All
turns look the same in EventKind space.

### Tier B — transcript classifier (~1-2 days)

Add a small classifier that maps each user transcript turn to one of
`{user_praise, user_criticism, user_distress, user_blame,
user_provocation, novel_observation, user_question, user_greeting}` —
the existing `EventKind` set. Inject the classified event into the
simulator instead of the generic `UserQuestion`.

**Classifier choice:** sentence-transformers + logistic-regression head
(or a tiny MLP). Trained on the existing
`tools/affect_calibration/data/contrast_pairs.jsonl` (1200 examples,
already labeled per-axis). Cross-axis labels could be derived from a
mapping (`valence/positive` → mostly `user_praise`/`user_greeting`,
`valence/negative` → `user_criticism`/`user_distress`,
`dominance/negative` → `user_blame`/`user_provocation`). Even crude
mapping should land >70 % on a held-out split — the simulator's
appraisal layer is already coarse-grained.

**Where it lives:** new Python node `TranscriptClassifierNode` between
`stt_in` and `affect_sim`. Subscribes to transcript text, publishes
`{ "kind": "user_distress", "confidence": 0.84 }` events.

**What this proves:** the agent reacts to content, not just engagement
cadence. This is where the affect chain starts feeling *alive*.

### Tier C — prosody-VAD branch (~half day if Phase 3 branch is in shape)

Hook up the existing Phase 3 prosody-VAD branch (`whisper_encoder`
`+ ridge regressor → V/A/D scores per chunk`). Emit
`prosody_arousal_high` / `prosody_valence_negative` events when scores
cross thresholds.

**What this proves:** the agent reacts to *how* the user is speaking,
not just what they said. A frustrated tone of voice raises empathy
even before the transcript classifier sees the words.

The Phase 3 branch was scaffolded but never end-to-end-validated in
the WebRTC pipeline. Need to verify the Whisper-encoder hookup still
works after the audio2face merge, and that the regressor's NPZ
produces calibrated VAD scores from real WebRTC-arrived audio (not
just the offline test fixtures).

## API surface (Rust)

```rust
// In a new crate or under crates/core/src/nodes/affect_sim.rs

use affect_simulator::{
    AffectState, EventKind, PerceivedEvent,
    RegulationEngine, SimulatorRun, channel_d,
};

pub struct AffectSimulatorNode {
    config: AffectConfig,
}

#[derive(Clone)]
pub struct AffectConfig {
    pub initial_scenario: Option<String>,    // e.g. "warm_admiration"
    pub steering_alpha: f32,                 // default 1.0
    pub emit_threshold_state_change: f32,    // suppress no-op publishes
    pub max_history_turns: usize,            // for Channel B summary
}

#[derive(Default)]
struct SessionAffectState {
    sim: AffectState,                        // owns the simulator state
    last_emitted_target_vad: [f32; 3],
    last_emitted_summary: String,
    pending_events: Vec<PerceivedEvent>,     // queued between ticks
    last_tick_at_ms: u64,
}

#[async_trait::async_trait]
impl AsyncStreamingNode for AffectSimulatorNode {
    fn pacing_nature(&self) -> PacingNature {
        PacingNature::SourceWall(5)          // 5 Hz = 200 ms ticks
    }

    fn is_multi_input(&self) -> bool { true }

    async fn process_multi_streaming_async(
        &self,
        inputs: HashMap<String, RuntimeData>,
        ctx: &NodeRuntimeContext,
        callback: ...,
    ) -> Result<(), Error> {
        // Buffer incoming VAD/transcript/prosody events into
        // ctx.session_state's pending_events. Don't emit here — the
        // tick() entry point owns emission so all envelopes are
        // strictly time-aligned.
    }

    async fn tick(
        &self,
        tick: Tick,
        session_id: Option<String>,
        callback: Box<dyn FnMut(RuntimeData) -> Result<(), Error> + Send>,
    ) -> Result<(), Error> {
        // 1. Look up session state (or initialize from config.initial_scenario)
        // 2. Drain pending_events; appraise each into the simulator
        // 3. Advance simulator dynamics by elapsed-since-last-tick
        // 4. Apply regulation policy to compute new policy[]
        // 5. Compute Channel D target VAD via channel_d::compute_target
        // 6. Render Channel B state summary (one-paragraph natural language)
        // 7. If state changed materially, emit:
        //    - aux-port envelope: set_steering
        //    - aux-port envelope: set_system_augmentation
        //    - aux-port envelope: set_sampling (Channel A; Tier B+)
        //    - debug tap: affect_state.out
        Ok(())
    }
}
```

**Per-session state via `ctx.session_state`.** Multi-session servers
(one runtime, many browser clients) need each session's simulator to
evolve independently. The `NodeRuntimeContext` already carries
`session_state: Arc<()>` for exactly this. Cast our `SessionAffectState`
into/out of the Arc on each call.

## Integration with `hermes3_affect_s2s_webrtc_server`

The existing example loads a *static* affect target at startup. The
runtime version replaces that with a live node:

```diff
-// Resolve the static affect target before we build the manifest so we
-// can bake Channel B (system augmentation) into the LLM's startup
-// system prompt and Channel D (target VAD) into the steering hook.
-let scenario = std::env::var("AFFECT_SCENARIO")...
-let target = load_affect_target(&scenario);
-let manifest = Arc::new(build_manifest(&target));
+// LLM starts with neutral steering (target_vad=[0,0,0]); the
+// AffectSimulatorNode in the pipeline overrides it on every tick.
+let manifest = Arc::new(build_manifest());
```

And the manifest gains:

```rust
nodes.push(NodeManifest {
    id: "affect_sim".to_string(),
    node_type: "AffectSimulatorNode".to_string(),
    params: serde_json::json!({
        "initial_scenario": std::env::var("AFFECT_SCENARIO").ok(),
        "steering_alpha": 1.0,
    }),
    ..Default::default()
});

connections.extend([
    Connection { from: "vad".into(),        to: "affect_sim".into() },
    Connection { from: "stt_in".into(),     to: "affect_sim".into() },
    Connection { from: "affect_sim".into(), to: "llm".into() },
]);
```

If `AFFECT_SCENARIO` env is set, the simulator pre-loads that scenario's
peak frame as the initial state — gives a starting register without
waiting for events to accumulate. If unset, starts at baseline.

## Open questions

1. **Emit threshold for state-change.** Without one, every 200 ms tick
   publishes near-identical envelopes (200 control-bus events/min per
   session). Proposed: emit `set_steering` only when `‖new_target_vad −
   last_emitted_target_vad‖ > 0.05`; emit `set_system_augmentation`
   only when the rendered summary string changes.
2. **Pre-LLM-turn rendering.** The current LLM node rebuilds the system
   prompt at session start; runtime aux-port updates only take effect
   on the *next* turn (the session's chat-state cache is invalidated).
   We should verify that's true under the merged code, and add a fast
   path if not (the cost of rebuilding the chat state per-tick is
   small but unnecessary).
3. **Tier B classifier latency.** The classifier runs after Whisper
   completes a turn; the simulator should consume the event within the
   200 ms window or the affect lags one full turn behind speech. If
   the classifier is too slow, fall back to an event-injection on
   transcript-arrival (rather than waiting for a tick boundary) to
   close that gap.
4. **Backpressure during LLM generation.** The simulator's tick is
   independent of LLM blocking time, but the LLM's aux-port handling
   is *not* — `set_steering` arriving mid-generation is queued for
   the next turn. This is correct (steering changes don't apply
   mid-token), but needs to be documented and verified.
5. **Initial-state hot-loading.** When the user reconnects mid-session,
   we currently start the simulator from baseline. Should we restore
   from the last published `affect_state.out` snapshot? Probably not for
   v1 — restart-clean is the simpler contract.

## Acceptance criteria

**Tier A complete when:**
- A 5-minute conversation with the demo server shows visibly evolving
  affect state in the `affect_state.out` debug tap (warmth and
  curiosity rise on engagement, decay through silence).
- The LLM's responses are observably steered by the live state (run
  the same prompt at simulator-state-A vs simulator-state-B and
  observe register difference).
- No tick coalescing under heavy LLM load (verify by logging
  `tick.frame_idx` continuity across a 30-second LLM blocking
  generation).

**Tier B complete when:**
- The transcript classifier achieves >70 % accuracy on a held-out
  split of `contrast_pairs.jsonl`.
- A turn matching `user_distress` (e.g. "I just lost my job")
  measurably raises empathy in the simulator state within one tick of
  the transcript arriving.
- A regression-style smoke test: 6-turn conversation with prepared
  inputs reproduces a deterministic affect-state trajectory.

**Tier C complete when:**
- The Phase 3 prosody-VAD branch publishes `prosody_*` events to
  `affect_sim` over the WebRTC path (not just from offline fixtures).
- A frustrated-tone audio sample (with neutral text content) raises
  empathy in the simulator state on the next tick — proving prosody
  is contributing independently of transcript classification.

## Calendar

- Tier A: 1 day (chassis + VAD-only event sources)
- Tier B: 1-2 days (classifier training + integration)
- Tier C: ½ day if Phase 3 branch works as-is; +½ day for
  WebRTC-pipeline integration if not

Total: ~3-4 working days for the full runtime affect chain, after
which the demo is the project's full thesis statement: *speak in,
observe affect-aware behavior evolving in response to what you said,
hear it back through Hermes-3 + Channel D + TTS.*

## Files to create

- [`crates/core/src/nodes/affect_sim.rs`](../../../crates/core/src/nodes/affect_sim.rs)
  — the `AffectSimulatorNode` implementation
- [`crates/core/src/nodes/affect_sim/render.rs`](../../../crates/core/src/nodes/affect_sim/render.rs)
  — Channel B summary renderer (state → natural-language paragraph)
- (Tier B) [`clients/python/remotemedia/nodes/ml/transcript_classifier.py`](../../../clients/python/remotemedia/nodes/ml/transcript_classifier.py)
  — sentence-transformers + LR head, publishes `EventKind` events
- New WebRTC example variant or extension of `hermes3_affect_s2s_webrtc_server`
  to wire the new node and connections.

## References

- [`spike-h-affect-chain-validated.md`](spike-h-affect-chain-validated.md)
  — offline eval that proved the affect chain produces user-visible
  improvement.
- `crates/core/src/nodes/streaming_node.rs` — `Tick`, `PacingNature`,
  `tick()` API.
- `tools/affect_simulator/src/` — the simulator crate; `AffectState`,
  `EventKind`, `RegulationEngine`, `channel_d::compute_target`.
- `crates/transports/webrtc/examples/hermes3_affect_s2s_webrtc_server.rs`
  — the static-target example this node replaces.
