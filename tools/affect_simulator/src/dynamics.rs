//! Dynamics: appraisal → per-channel impact → next channels.
//!
//! `update_channel` is verbatim from PROJECT_PLAN.md — inertia preserves
//! state, decay pulls toward baseline, event impact adds, and a per-step
//! `max_delta` clamp prevents single events from saturating a channel.

use serde::{Deserialize, Serialize};

use crate::appraisal::Appraisal;
use crate::state::EmotionChannels;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ChannelParams {
    pub baseline: f32,
    /// [0, 1]; how much state persists between ticks before decay.
    pub inertia: f32,
    /// Per-second pull toward baseline.
    pub decay_rate: f32,
    /// Per-update absolute clamp on Δ — prevents one event from ramping
    /// 0 → 1 in a single tick. Tuned per channel because anger and joy
    /// rise faster than calm.
    pub max_delta: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsParams {
    pub anger: ChannelParams,
    pub sadness: ChannelParams,
    pub fear: ChannelParams,
    pub joy: ChannelParams,
    pub calm: ChannelParams,
    pub frustration: ChannelParams,
    pub curiosity: ChannelParams,
    pub empathy: ChannelParams,
}

impl Default for DynamicsParams {
    fn default() -> Self {
        // Defaults tuned so the 12 canonical scenarios produce visibly
        // distinct trajectories — slow channels (sadness, calm) decay
        // gently; fast channels (anger, fear, joy) decay quicker.
        let fast = |baseline: f32| ChannelParams {
            baseline,
            inertia: 0.85,
            decay_rate: 0.12,
            max_delta: 0.30,
        };
        let medium = |baseline: f32| ChannelParams {
            baseline,
            inertia: 0.90,
            decay_rate: 0.07,
            max_delta: 0.25,
        };
        let slow = |baseline: f32| ChannelParams {
            baseline,
            inertia: 0.93,
            decay_rate: 0.04,
            max_delta: 0.20,
        };
        Self {
            anger: fast(0.0),
            sadness: slow(0.0),
            fear: fast(0.0),
            joy: fast(0.0),
            calm: medium(0.4),
            frustration: medium(0.0),
            curiosity: medium(0.2),
            empathy: medium(0.0),
        }
    }
}

/// PROJECT_PLAN.md §6 Phase 1 verbatim.
fn update_channel(
    previous: f32,
    baseline: f32,
    event_impact: f32,
    inertia: f32,
    decay_rate: f32,
    dt_seconds: f32,
    max_delta: f32,
) -> f32 {
    let retained = baseline + inertia * (previous - baseline);
    let decayed = retained + (baseline - retained) * decay_rate * dt_seconds;
    let raw_next = decayed + event_impact;
    let clamped = raw_next.clamp(previous - max_delta, previous + max_delta);
    clamped.clamp(0.0, 1.0)
}

/// Project an appraisal onto per-channel signed impact magnitudes.
/// Coefficients are deliberately small (≤ 0.5 each) — the dynamics step
/// further clamps with `max_delta` and the appraisal itself was already
/// confidence-weighted upstream.
fn impacts_from_appraisal(a: &Appraisal) -> ChannelImpacts {
    let pos_outcome = a.reward_signal + a.goal_congruence.max(0.0);
    let neg_outcome = a.loss_signal + (-a.goal_congruence).max(0.0);
    let lost_control = (-a.control).max(0.0);
    let threat = a.threat_signal;
    let blame_other = a.agency_other * (-a.goal_congruence).max(0.0);
    let novelty = a.novelty;
    let unsafe_social = (-a.social_safety).max(0.0);
    let safe_social = a.social_safety.max(0.0);

    ChannelImpacts {
        anger: 0.5 * blame_other + 0.3 * a.norm_violation - 0.1 * safe_social,
        sadness: 0.5 * neg_outcome + 0.2 * unsafe_social,
        fear: 0.5 * threat + 0.3 * lost_control,
        joy: 0.5 * pos_outcome + 0.2 * safe_social,
        calm: -0.3 * (threat + neg_outcome + lost_control)
            + 0.2 * pos_outcome
            + 0.1 * a.certainty.max(0.0),
        frustration: 0.4 * neg_outcome + 0.3 * lost_control,
        curiosity: 0.5 * novelty + 0.2 * a.certainty.min(0.0).abs(),
        // Empathy fires on user_distress / user_blame against someone
        // else; here it's driven by goal_relevance with a negative goal
        // (someone else's loss visible to the agent).
        empathy: 0.4 * (a.goal_relevance * neg_outcome).max(0.0)
            + 0.2 * unsafe_social,
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ChannelImpacts {
    anger: f32,
    sadness: f32,
    fear: f32,
    joy: f32,
    calm: f32,
    frustration: f32,
    curiosity: f32,
    empathy: f32,
}

#[derive(Debug, Default)]
pub struct Dynamics;

impl Dynamics {
    pub fn new() -> Self {
        Self
    }

    /// Apply the appraisal's per-channel impact in a single tick of
    /// duration `dt_seconds`. Decay/inertia/baseline are read from
    /// `params`.
    pub fn step_with_impact(
        &self,
        prev: &EmotionChannels,
        appraisal: &Appraisal,
        params: &DynamicsParams,
        dt_seconds: f32,
    ) -> EmotionChannels {
        let imp = impacts_from_appraisal(appraisal);
        EmotionChannels {
            anger: update_channel(
                prev.anger,
                params.anger.baseline,
                imp.anger,
                params.anger.inertia,
                params.anger.decay_rate,
                dt_seconds,
                params.anger.max_delta,
            ),
            sadness: update_channel(
                prev.sadness,
                params.sadness.baseline,
                imp.sadness,
                params.sadness.inertia,
                params.sadness.decay_rate,
                dt_seconds,
                params.sadness.max_delta,
            ),
            fear: update_channel(
                prev.fear,
                params.fear.baseline,
                imp.fear,
                params.fear.inertia,
                params.fear.decay_rate,
                dt_seconds,
                params.fear.max_delta,
            ),
            joy: update_channel(
                prev.joy,
                params.joy.baseline,
                imp.joy,
                params.joy.inertia,
                params.joy.decay_rate,
                dt_seconds,
                params.joy.max_delta,
            ),
            calm: update_channel(
                prev.calm,
                params.calm.baseline,
                imp.calm,
                params.calm.inertia,
                params.calm.decay_rate,
                dt_seconds,
                params.calm.max_delta,
            ),
            frustration: update_channel(
                prev.frustration,
                params.frustration.baseline,
                imp.frustration,
                params.frustration.inertia,
                params.frustration.decay_rate,
                dt_seconds,
                params.frustration.max_delta,
            ),
            curiosity: update_channel(
                prev.curiosity,
                params.curiosity.baseline,
                imp.curiosity,
                params.curiosity.inertia,
                params.curiosity.decay_rate,
                dt_seconds,
                params.curiosity.max_delta,
            ),
            empathy: update_channel(
                prev.empathy,
                params.empathy.baseline,
                imp.empathy,
                params.empathy.inertia,
                params.empathy.decay_rate,
                dt_seconds,
                params.empathy.max_delta,
            ),
        }
    }

    /// "Empty" tick — no event impact, just decay-toward-baseline. Used
    /// between events so trajectories show recovery.
    pub fn step_idle(
        &self,
        prev: &EmotionChannels,
        params: &DynamicsParams,
        dt_seconds: f32,
    ) -> EmotionChannels {
        let zero = Appraisal::default();
        self.step_with_impact(prev, &zero, params, dt_seconds)
    }
}
