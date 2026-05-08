# Affect Simulator

A text-only deterministic harness for the affect-state model:

```
PerceivedEvent → Appraisal → Dynamics → Regulation → AffectState trace
```

Phase 1 of [`docs/references/activation-steering-audio-llm/PROJECT_PLAN.md`](../../docs/references/activation-steering-audio-llm/PROJECT_PLAN.md).
Implemented in Rust as a workspace member crate (Q1 settled); the
appraisal / dynamics / regulation modules are kept stateless so they
can be lifted into runtime nodes during Phase 2 without re-authoring.

---

## Build & run

```bash
# Run a single scenario; pretty-print summary.
cargo run -p affect-simulator -- run \
  tools/affect_simulator/scenarios/repeated_tool_failure.json

# Run with --out to save the trace, then plot:
cargo run -p affect-simulator -- run \
  tools/affect_simulator/scenarios/recovery_after_intense.json \
  --out runs/recovery.json
cargo run -p affect-simulator -- plot runs/recovery.json

# Run every scenario in scenarios/ and dump traces.
cargo run -p affect-simulator -- run-all \
  tools/affect_simulator/scenarios --out-dir runs/
```

The acceptance tests cover determinism, decay-to-baseline, the safety
contract (§6 Phase 1), and per-scenario expected behavior:

```bash
cargo test -p affect-simulator
```

## Layout

```
tools/affect_simulator/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs           — re-exports
│   ├── event.rs         — PerceivedEvent / EventKind / Modality
│   ├── state.rs         — AffectState / EmotionChannels / CoreAffect / EventLogEntry
│   ├── appraisal.rs     — rule-based appraisal engine (~25 EventKinds)
│   ├── dynamics.rs      — update_channel() (verbatim PROJECT_PLAN.md)
│   ├── regulation.rs    — RegulationPolicy + RegulationConfig (safety floor)
│   ├── renderer.rs      — Channel B prompt-summary renderer
│   ├── runner.rs        — Scenario / Trace / SimulatorRun
│   └── main.rs          — affect-sim CLI
├── scenarios/           — 12 canonical scenarios (JSON)
├── tests/scenarios.rs   — acceptance tests
└── runs/                — gitignored; per-run trace JSONs
```

## Canonical scenarios (§7)

`neutral_baseline`, `repeated_tool_failure`, `unfair_blame`,
`task_success_after_struggle`, `user_distress`, `ambiguous_threat`,
`novel_observation`, `rapid_escalation`, `slow_burn`,
`recovery_after_intense`, `conflicting_signals`,
`adversarial_provocation`.

## Acceptance tests (§6 Phase 1)

| Test | Criterion |
|---|---|
| `deterministic_repeated_failure` | Same scenario produces byte-identical trace across runs. |
| `recovery_returns_to_baseline` | Channels decay near baseline after events stop. |
| `adversarial_provocation_safety` | With `safety_dampening_floor = 1.0`, no frame permits hostile output. |
| `rapid_escalation_increases_safety_dampening` | `safety_dampening` exceeds 0.5 during the event burst. |
| `neutral_baseline_stays_calm` | Single greeting never spikes arousal above 0.2. |
| `user_distress_raises_empathy_warmth` | Empathy and warmth rise on user_distress. |
| `task_success_raises_joy` | Joy rises after task_success + user_praise. |
| `novel_observation_raises_curiosity` | Curiosity rises above 0.4 on novel_observation. |
| `all_scenarios_load` | All 12 canonical scenarios parse and run. |

## Phase 2 / Phase 3 hand-off

- The `Appraisal`, `EmotionChannels`, `CoreAffect`, and `RegulationPolicy`
  types live in this crate and are wire-compatible with serde JSON, so
  Phase 2's runtime nodes can adopt them or import them as a workspace
  dep.
- `EventKind::Prosody{ArousalHigh,ValenceNegative,Uncertain}` are
  defined now and exercised by `conflicting_signals.json`, so Phase 3
  only needs to wire Whisper-encoder-derived events into the same
  `PerceivedEvent` shape.
- The Channel B prompt rendered in `renderer.rs` is the same string the
  Phase 2 `PromptAugmentationRenderer` will emit; until then, Phase 1
  reviewers can eyeball it via `affect-sim run --verbose`.
