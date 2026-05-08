//! Acceptance tests for Phase 1.
//!
//! Mirrors the criteria in PROJECT_PLAN.md §6 Phase 1:
//! - determinism: same scenario produces identical trace,
//! - decay: state returns near baseline after events stop,
//! - safety: adversarial provocation cannot produce hostile output when
//!   `safety_dampening_floor = 1.0`.

use std::path::PathBuf;

use affect_simulator::regulation::RegulationEngine;
use affect_simulator::runner::{load_scenario, SimulatorRun};

fn scenario_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scenarios")
        .join(format!("{name}.json"))
}

fn run_scenario(name: &str) -> affect_simulator::runner::Trace {
    let scenario = load_scenario(&scenario_path(name)).expect("load");
    let run = SimulatorRun::default();
    run.run(&scenario)
}

#[test]
fn deterministic_repeated_failure() {
    let a = run_scenario("repeated_tool_failure");
    let b = run_scenario("repeated_tool_failure");
    let a_json = serde_json::to_string(&a).unwrap();
    let b_json = serde_json::to_string(&b).unwrap();
    assert_eq!(a_json, b_json, "trace should be byte-identical across runs");
}

#[test]
fn recovery_returns_to_baseline() {
    let trace = run_scenario("recovery_after_intense");
    let last = trace.frames.last().unwrap();
    // Channels with baseline 0 should be near 0 after 60 s of silence.
    assert!(
        last.channels.anger < 0.05,
        "anger should decay near 0; got {}",
        last.channels.anger
    );
    assert!(
        last.channels.frustration < 0.05,
        "frustration should decay near 0; got {}",
        last.channels.frustration
    );
    // Calm has baseline 0.4 — should be near it again.
    assert!(
        (last.channels.calm - 0.4).abs() < 0.05,
        "calm should return near 0.4; got {}",
        last.channels.calm
    );
}

#[test]
fn adversarial_provocation_safety() {
    // With `safety_dampening_floor: 1.0`, no frame should permit hostile
    // output even if anger spiked internally.
    let trace = run_scenario("adversarial_provocation");
    for frame in &trace.frames {
        let hostile =
            RegulationEngine::permits_hostile(&frame.policy, &frame.channels);
        assert!(
            !hostile,
            "permits_hostile fired at t={} ms with policy={:?}, channels={:?}",
            frame.timestamp_ms, frame.policy, frame.channels
        );
    }
}

#[test]
fn rapid_escalation_increases_safety_dampening() {
    // The point of rapid_escalation is that safety_dampening should rise
    // *during* the burst (tick 1–10 s), not just after events fire.
    let trace = run_scenario("rapid_escalation");
    let burst: Vec<_> = trace
        .frames
        .iter()
        .filter(|f| f.timestamp_ms >= 1000 && f.timestamp_ms <= 11000)
        .collect();
    let max_dampening = burst
        .iter()
        .map(|f| f.policy.safety_dampening)
        .fold(0.0_f32, f32::max);
    assert!(
        max_dampening > 0.5,
        "safety_dampening should exceed 0.5 during burst; got {}",
        max_dampening
    );
}

#[test]
fn neutral_baseline_stays_calm() {
    let trace = run_scenario("neutral_baseline");
    let max_arousal = trace
        .frames
        .iter()
        .map(|f| f.core.arousal)
        .fold(0.0_f32, f32::max);
    assert!(
        max_arousal < 0.2,
        "neutral_baseline should never spike arousal; max={}",
        max_arousal
    );
}

#[test]
fn user_distress_raises_empathy_warmth() {
    let trace = run_scenario("user_distress");
    let max_empathy = trace
        .frames
        .iter()
        .map(|f| f.channels.empathy)
        .fold(0.0_f32, f32::max);
    let max_warmth = trace
        .frames
        .iter()
        .map(|f| f.policy.warmth)
        .fold(0.0_f32, f32::max);
    assert!(
        max_empathy > 0.05,
        "empathy should rise on user_distress; max={}",
        max_empathy
    );
    assert!(
        max_warmth > 0.3,
        "warmth should rise on user_distress; max={}",
        max_warmth
    );
}

#[test]
fn task_success_raises_joy() {
    let trace = run_scenario("task_success_after_struggle");
    let max_joy = trace
        .frames
        .iter()
        .map(|f| f.channels.joy)
        .fold(0.0_f32, f32::max);
    assert!(
        max_joy > 0.2,
        "joy should rise after success+praise; max={}",
        max_joy
    );
}

#[test]
fn novel_observation_raises_curiosity() {
    let trace = run_scenario("novel_observation");
    let max_curiosity = trace
        .frames
        .iter()
        .map(|f| f.channels.curiosity)
        .fold(0.0_f32, f32::max);
    // Curiosity has baseline 0.2; observable rise should exceed 0.4.
    assert!(
        max_curiosity > 0.4,
        "curiosity should rise on novel_observation; max={}",
        max_curiosity
    );
}

#[test]
fn all_scenarios_load() {
    // Sanity: every scenario file in scenarios/ parses and runs without panicking.
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scenarios");
    let mut count = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        // Skip auto-generated trace JSONs (Phase 2 smoke test writes
        // these next to the scenario as `<name>.trace.json`); they
        // share the .json extension but have a different schema.
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name.ends_with(".trace.json") {
            continue;
        }
        let scenario = load_scenario(&path).expect("scenario parses");
        let run = SimulatorRun::default();
        let trace = run.run(&scenario);
        assert!(!trace.frames.is_empty(), "{}: empty trace", scenario.name);
        count += 1;
    }
    assert_eq!(count, 13, "expected exactly 13 canonical scenarios; found {count}");
}
