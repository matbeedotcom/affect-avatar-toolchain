//! Channel A — decoding-config override.
//!
//! Translates an `AffectState` + `RegulationPolicy` into a per-turn
//! adjustment of the LLM's sampling parameters. Equations are verbatim
//! from PROJECT_PLAN.md §6 Phase 2.
//!
//! The output is a *delta* applied to the LLM's base sampling config:
//! a runtime driver multiplies the current `temperature` and
//! `max_tokens` by these factors. A factor of 1.0 means "no change".
//! This keeps the channel agnostic to the actual base values, which
//! differ per LLM (LFM2-Audio defaults `audio_temperature=1.0`,
//! llama-cpp-served Phi-3 defaults `temperature=0.7`).

use serde::{Deserialize, Serialize};

use crate::regulation::RegulationPolicy;
use crate::state::AffectState;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SamplingOverride {
    /// Multiplicative factor on temperature.
    pub temperature_scale: f32,
    /// Multiplicative factor on max_tokens. Driver should round to int.
    pub max_tokens_scale: f32,
}

impl Default for SamplingOverride {
    fn default() -> Self {
        Self {
            temperature_scale: 1.0,
            max_tokens_scale: 1.0,
        }
    }
}

#[derive(Debug, Default)]
pub struct ChannelA;

impl ChannelA {
    pub fn new() -> Self {
        Self
    }

    /// Verbatim from PROJECT_PLAN.md §6 Phase 2 (`channel_a_decoding`).
    /// High arousal narrows the distribution and shortens responses;
    /// high calm with low task_focus widens slightly. All other states
    /// keep base values.
    pub fn compute(&self, state: &AffectState, policy: &RegulationPolicy) -> SamplingOverride {
        let mut out = SamplingOverride::default();
        if state.core.arousal > 0.7 {
            out.temperature_scale *= 0.85;
            out.max_tokens_scale *= 0.7;
        }
        if state.channels.calm > 0.6 && policy.task_focus < 0.4 {
            out.temperature_scale *= 1.05;
        }
        out
    }
}
