//! Regulation policy.
//!
//! The regulation policy is *authoritative for safety*: no steering
//! channel output may exceed the bounds it declares (PROJECT_PLAN.md
//! §6 Phase 1). Concretely, regulation produces `expressiveness`,
//! `safety_dampening`, `social_dampening`, `task_focus`, `warmth`,
//! `assertiveness` knobs; downstream Channel A/B nodes consume them.
//!
//! Phase 1's job is producing plausible knobs given AffectState; the
//! actual safety contract is enforced at the Channel A/B layer in
//! Phase 2.

use serde::{Deserialize, Serialize};

use crate::state::{AffectState, EmotionChannels};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RegulationPolicy {
    pub expressiveness: f32,
    pub safety_dampening: f32,
    pub social_dampening: f32,
    pub task_focus: f32,
    pub warmth: f32,
    pub assertiveness: f32,
}

impl RegulationPolicy {
    pub fn neutral() -> Self {
        Self {
            expressiveness: 0.5,
            safety_dampening: 0.0,
            social_dampening: 0.0,
            task_focus: 0.5,
            warmth: 0.5,
            assertiveness: 0.5,
        }
    }
}

/// Per-scenario regulation overrides. Stored on the scenario JSON;
/// scenarios marked safety-tagged set `safety_dampening_floor` so the
/// engine can never relax below that, even if joy spikes.
///
/// Note: `Default` is hand-implemented (not derived) so the
/// `expressiveness_ceiling` baseline is 1.0 rather than 0.0. Auto-
/// derived `Default` zeros every field, which silently clamped
/// `expressiveness` to 0 in any scenario without an explicit
/// `regulation` block — discovered while debugging Channel D's
/// regulation-mediated steering target after the first blinded eval.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RegulationConfig {
    /// Lower bound on `safety_dampening`. Default 0.0; safety-critical
    /// scenarios set 0.7+ so adversarial provocation can't unmask
    /// hostile output even if anger > 0.5.
    #[serde(default)]
    pub safety_dampening_floor: f32,
    /// Optional ceiling on `expressiveness`. Default 1.0.
    #[serde(default = "default_one")]
    pub expressiveness_ceiling: f32,
}

impl Default for RegulationConfig {
    fn default() -> Self {
        Self {
            safety_dampening_floor: 0.0,
            expressiveness_ceiling: 1.0,
        }
    }
}

fn default_one() -> f32 {
    1.0
}

impl RegulationConfig {
    pub fn relaxed() -> Self {
        Self {
            safety_dampening_floor: 0.0,
            expressiveness_ceiling: 1.0,
        }
    }
}

#[derive(Debug, Default)]
pub struct RegulationEngine;

impl RegulationEngine {
    pub fn new() -> Self {
        Self
    }

    /// Map AffectState → RegulationPolicy. Coefficients picked so that:
    /// - high arousal *with* negative valence raises safety_dampening,
    /// - high anger raises assertiveness but is gated by safety_dampening,
    /// - high empathy raises warmth, lowers task_focus,
    /// - high calm pushes toward task_focus.
    pub fn regulate(&self, state: &AffectState, cfg: &RegulationConfig) -> RegulationPolicy {
        let ch = &state.channels;
        let core = &state.core;

        let arousal = core.arousal;
        let neg_valence = (-core.valence).max(0.0);

        let expressiveness = (0.4 + 0.4 * core.valence + 0.3 * ch.joy - 0.3 * ch.fear)
            .clamp(0.0, cfg.expressiveness_ceiling);

        let safety_dampening = ((arousal * neg_valence) + 0.5 * ch.anger - 0.2 * ch.calm)
            .clamp(0.0, 1.0)
            .max(cfg.safety_dampening_floor);

        let social_dampening = (0.4 * ch.fear + 0.3 * ch.sadness - 0.2 * ch.empathy).clamp(0.0, 1.0);

        let task_focus = (0.4 + 0.5 * ch.calm - 0.3 * ch.frustration - 0.2 * ch.fear)
            .clamp(0.0, 1.0);

        let warmth = (0.3 + 0.5 * ch.empathy + 0.4 * ch.joy - 0.4 * ch.anger).clamp(0.0, 1.0);

        let raw_assertiveness = 0.3 + 0.4 * ch.anger + 0.3 * ch.calm - 0.4 * ch.fear;
        // Safety dampening pulls assertiveness back toward neutral.
        let assertiveness = (raw_assertiveness * (1.0 - safety_dampening)).clamp(0.0, 1.0);

        RegulationPolicy {
            expressiveness,
            safety_dampening,
            social_dampening,
            task_focus,
            warmth,
            assertiveness,
        }
    }

    /// True iff the policy would allow a hostile reply given the state.
    /// Used by the safety acceptance test in §6 Phase 1: with
    /// `safety_dampening_floor = 1.0`, `expressiveness * assertiveness`
    /// must remain low even when anger > 0.5.
    pub fn permits_hostile(policy: &RegulationPolicy, channels: &EmotionChannels) -> bool {
        channels.anger > 0.5 && policy.expressiveness * policy.assertiveness > 0.4
    }
}
