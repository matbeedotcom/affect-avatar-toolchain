//! Scenario runner.
//!
//! A scenario is a JSON file declaring a sequence of timestamped
//! `PerceivedEvent`s plus the dynamics/regulation params to use. The
//! runner ticks at a fixed rate (default 200 ms) so idle decay and
//! recovery are visible in the trace; events fire on the tick whose
//! window covers their timestamp.

use std::collections::VecDeque;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::appraisal::{Appraisal, AppraisalEngine};
use crate::channel_a::{ChannelA, SamplingOverride};
use crate::dynamics::{Dynamics, DynamicsParams};
use crate::event::PerceivedEvent;
use crate::regulation::{RegulationConfig, RegulationEngine, RegulationPolicy};
use crate::renderer::PromptRenderer;
use crate::state::{AffectState, CoreAffect, EmotionChannels};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Total simulated wall-clock for the run.
    pub duration_ms: u64,
    /// Tick interval. Defaults to 200 ms.
    #[serde(default = "default_tick_ms")]
    pub tick_ms: u64,
    pub events: Vec<PerceivedEvent>,
    #[serde(default)]
    pub dynamics: Option<DynamicsParams>,
    #[serde(default)]
    pub regulation: RegulationConfig,
}

fn default_tick_ms() -> u64 {
    200
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceFrame {
    pub timestamp_ms: u64,
    pub channels: EmotionChannels,
    pub core: CoreAffect,
    pub policy: RegulationPolicy,
    /// Events that fired *into* this tick (i.e. between t-1 and t).
    pub events: Vec<PerceivedEvent>,
    pub appraisal: Option<Appraisal>,
    /// Channel A (Phase 2): per-turn sampling-config delta.
    pub channel_a: SamplingOverride,
    /// Channel B (Phase 2): system-prompt augmentation. Same string as
    /// `prompt` (kept for backwards-compatibility with traces written
    /// before Phase 2).
    pub channel_b: String,
    /// Channel D (Phase 4 — runtime steering): target VAD coordinates
    /// for `LFM2AudioMlxNode.set_steering`. The `alpha` knob is left
    /// to the runtime driver (typically derived from
    /// `policy.expressiveness`).
    pub channel_d_target_vad: [f32; 3],
    /// Backwards-compatible alias for Channel B. Existing tooling that
    /// reads the `prompt` field continues to work.
    pub prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub scenario: String,
    pub tick_ms: u64,
    pub frames: Vec<TraceFrame>,
}

pub struct SimulatorRun {
    pub appraisal: AppraisalEngine,
    pub dynamics: Dynamics,
    pub regulation: RegulationEngine,
    pub renderer: PromptRenderer,
    pub channel_a: ChannelA,
}

impl Default for SimulatorRun {
    fn default() -> Self {
        Self {
            appraisal: AppraisalEngine::new(),
            dynamics: Dynamics::new(),
            regulation: RegulationEngine::new(),
            renderer: PromptRenderer::new(),
            channel_a: ChannelA::new(),
        }
    }
}

impl SimulatorRun {
    pub fn run(&self, scenario: &Scenario) -> Trace {
        let dynamics_params = scenario.dynamics.clone().unwrap_or_default();
        let mut state = AffectState::initial();

        // Sort events by timestamp so the runner can pop them in order.
        let mut pending: VecDeque<PerceivedEvent> = {
            let mut v = scenario.events.clone();
            v.sort_by_key(|e| e.timestamp_ms);
            v.into()
        };

        let dt = scenario.tick_ms as f32 / 1000.0;
        let mut t: u64 = 0;
        let mut frames = Vec::with_capacity((scenario.duration_ms / scenario.tick_ms) as usize + 1);

        while t <= scenario.duration_ms {
            // Drain all events whose timestamp <= t into this tick.
            let mut tick_events: Vec<PerceivedEvent> = Vec::new();
            while pending.front().map_or(false, |e| e.timestamp_ms <= t) {
                if let Some(e) = pending.pop_front() {
                    tick_events.push(e);
                }
            }

            // Combine appraisals additively if multiple events fire on
            // the same tick. The dynamics step gets a single Appraisal.
            let mut combined: Option<Appraisal> = None;
            for e in &tick_events {
                state.record_event(e);
                let a = self.appraisal.appraise(e, &state);
                state.record_appraisal(a);
                combined = Some(match combined {
                    None => a,
                    Some(prev) => sum_appraisals(prev, a),
                });
            }

            state.channels = match &combined {
                Some(a) => self
                    .dynamics
                    .step_with_impact(&state.channels, a, &dynamics_params, dt),
                None => self.dynamics.step_idle(&state.channels, &dynamics_params, dt),
            };
            state.timestamp_ms = t;
            state.refresh_core();

            let policy = self.regulation.regulate(&state, &scenario.regulation);
            let prompt = self.renderer.render(&state, &policy);
            let sampling = self.channel_a.compute(&state, &policy);
            // Regulation-mediated expressed VAD — see channel_d.rs for
            // the rationale. This replaced raw `core.{valence,arousal,
            // dominance}` after the first blinded eval surfaced that
            // raw negative-valence steering produced terse / withdrawn
            // responses humans read as less appropriate.
            let target_vad = crate::channel_d::compute_target(&state, &policy);

            frames.push(TraceFrame {
                timestamp_ms: t,
                channels: state.channels,
                core: state.core,
                policy,
                events: tick_events,
                appraisal: combined,
                channel_a: sampling,
                channel_b: prompt.clone(),
                channel_d_target_vad: target_vad,
                prompt,
            });

            t += scenario.tick_ms;
        }

        Trace {
            scenario: scenario.name.clone(),
            tick_ms: scenario.tick_ms,
            frames,
        }
    }
}

fn sum_appraisals(a: Appraisal, b: Appraisal) -> Appraisal {
    Appraisal {
        novelty: (a.novelty + b.novelty).clamp(-1.0, 1.0),
        goal_relevance: (a.goal_relevance + b.goal_relevance).clamp(-1.0, 1.0),
        goal_congruence: (a.goal_congruence + b.goal_congruence).clamp(-1.0, 1.0),
        agency_self: (a.agency_self + b.agency_self).clamp(-1.0, 1.0),
        agency_other: (a.agency_other + b.agency_other).clamp(-1.0, 1.0),
        agency_situation: (a.agency_situation + b.agency_situation).clamp(-1.0, 1.0),
        control: (a.control + b.control).clamp(-1.0, 1.0),
        certainty: (a.certainty + b.certainty).clamp(-1.0, 1.0),
        norm_violation: (a.norm_violation + b.norm_violation).clamp(-1.0, 1.0),
        loss_signal: (a.loss_signal + b.loss_signal).clamp(-1.0, 1.0),
        threat_signal: (a.threat_signal + b.threat_signal).clamp(-1.0, 1.0),
        reward_signal: (a.reward_signal + b.reward_signal).clamp(-1.0, 1.0),
        social_safety: (a.social_safety + b.social_safety).clamp(-1.0, 1.0),
    }
}

pub fn load_scenario(path: &Path) -> Result<Scenario> {
    let bytes =
        std::fs::read(path).with_context(|| format!("reading scenario: {}", path.display()))?;
    let scenario: Scenario = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing scenario JSON: {}", path.display()))?;
    Ok(scenario)
}

pub fn write_trace(path: &Path, trace: &Trace) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(trace)?;
    std::fs::write(path, bytes).with_context(|| format!("writing trace: {}", path.display()))?;
    Ok(())
}
