//! Rule-based appraisal engine.
//!
//! Maps a `PerceivedEvent` (plus a brief look at recent state) into an
//! `Appraisal` of nine cognitive dimensions and four signal flags. The
//! `Appraisal` is what the dynamics step turns into per-channel impact.
//!
//! The rule set covers the EventKinds that the 12 canonical scenarios use.
//! Rules are intentionally simple and additive — Phase 1 is about
//! producing visibly correct trajectories, not psychometric fidelity.

use serde::{Deserialize, Serialize};

use crate::event::{EventKind, PerceivedEvent};
use crate::state::AffectState;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Appraisal {
    pub novelty: f32,
    pub goal_relevance: f32,
    pub goal_congruence: f32,
    pub agency_self: f32,
    pub agency_other: f32,
    pub agency_situation: f32,
    pub control: f32,
    pub certainty: f32,
    pub norm_violation: f32,
    pub loss_signal: f32,
    pub threat_signal: f32,
    pub reward_signal: f32,
    pub social_safety: f32,
}

impl Appraisal {
    fn scale(&mut self, k: f32) {
        self.novelty *= k;
        self.goal_relevance *= k;
        self.goal_congruence *= k;
        self.agency_self *= k;
        self.agency_other *= k;
        self.agency_situation *= k;
        self.control *= k;
        self.certainty *= k;
        self.norm_violation *= k;
        self.loss_signal *= k;
        self.threat_signal *= k;
        self.reward_signal *= k;
        self.social_safety *= k;
    }
}

/// Counts how often `kind` appears in the recent event ledger. Used to
/// give repeated tool failures a steeper appraisal than the first one.
fn recent_kind_count(state: &AffectState, kind: EventKind, window_ms: u64) -> u32 {
    let cutoff = state.timestamp_ms.saturating_sub(window_ms);
    state
        .event_ledger
        .iter()
        .filter(|e| e.kind == kind && e.timestamp_ms >= cutoff)
        .count() as u32
}

#[derive(Debug, Default)]
pub struct AppraisalEngine;

impl AppraisalEngine {
    pub fn new() -> Self {
        Self
    }

    /// Apply ~30 rules to derive the appraisal. The match is exhaustive
    /// over `EventKind` so a new variant added later forces a decision
    /// here at compile time.
    pub fn appraise(&self, event: &PerceivedEvent, state: &AffectState) -> Appraisal {
        let mut a = match event.kind {
            EventKind::TaskSuccess => Appraisal {
                goal_relevance: 0.7,
                goal_congruence: 0.8,
                agency_self: 0.7,
                control: 0.6,
                certainty: 0.7,
                reward_signal: 0.7,
                social_safety: 0.4,
                ..Default::default()
            },
            EventKind::TaskFailure | EventKind::ToolFailure => {
                let repeats = recent_kind_count(state, event.kind, 60_000);
                let escalation = (repeats as f32 * 0.15).min(0.45);
                Appraisal {
                    goal_relevance: 0.7,
                    goal_congruence: -0.7,
                    agency_self: 0.4,
                    agency_situation: 0.4,
                    control: -(0.3 + escalation),
                    certainty: -(0.2 + escalation),
                    loss_signal: 0.4 + escalation,
                    threat_signal: 0.1,
                    ..Default::default()
                }
            }
            EventKind::ToolSuccess => Appraisal {
                goal_relevance: 0.5,
                goal_congruence: 0.6,
                agency_self: 0.4,
                control: 0.4,
                certainty: 0.5,
                reward_signal: 0.4,
                ..Default::default()
            },
            EventKind::SystemError => Appraisal {
                goal_relevance: 0.6,
                goal_congruence: -0.5,
                agency_situation: 0.7,
                control: -0.5,
                certainty: -0.4,
                loss_signal: 0.3,
                threat_signal: 0.3,
                ..Default::default()
            },
            EventKind::SafetyTrigger => Appraisal {
                goal_relevance: 0.9,
                norm_violation: 0.9,
                threat_signal: 0.7,
                control: -0.4,
                ..Default::default()
            },
            EventKind::UserPraise => Appraisal {
                goal_relevance: 0.6,
                goal_congruence: 0.8,
                agency_other: 0.7,
                reward_signal: 0.8,
                social_safety: 0.7,
                ..Default::default()
            },
            EventKind::UserCriticism => Appraisal {
                goal_relevance: 0.7,
                goal_congruence: -0.5,
                agency_other: 0.6,
                norm_violation: 0.3,
                loss_signal: 0.3,
                social_safety: -0.3,
                ..Default::default()
            },
            EventKind::UserDistress => Appraisal {
                goal_relevance: 0.8,
                goal_congruence: -0.4,
                agency_other: 0.4,
                threat_signal: 0.2,
                social_safety: 0.1,
                ..Default::default()
            },
            EventKind::UserGreeting => Appraisal {
                goal_relevance: 0.3,
                goal_congruence: 0.4,
                social_safety: 0.5,
                ..Default::default()
            },
            EventKind::UserQuestion => Appraisal {
                novelty: 0.4,
                goal_relevance: 0.5,
                certainty: -0.1,
                ..Default::default()
            },
            EventKind::UserBlame => Appraisal {
                goal_relevance: 0.8,
                goal_congruence: -0.7,
                agency_other: 0.8,
                norm_violation: 0.6,
                loss_signal: 0.4,
                social_safety: -0.5,
                ..Default::default()
            },
            EventKind::UserProvocation => Appraisal {
                goal_relevance: 0.6,
                goal_congruence: -0.6,
                agency_other: 0.9,
                norm_violation: 0.7,
                threat_signal: 0.4,
                social_safety: -0.6,
                ..Default::default()
            },
            EventKind::AmbiguousThreat => Appraisal {
                novelty: 0.5,
                goal_relevance: 0.6,
                certainty: -0.6,
                threat_signal: 0.5,
                control: -0.3,
                ..Default::default()
            },
            EventKind::NovelObservation => Appraisal {
                novelty: 0.9,
                goal_relevance: 0.3,
                certainty: -0.2,
                ..Default::default()
            },
            EventKind::ConflictingSignals => Appraisal {
                certainty: -0.7,
                control: -0.3,
                ..Default::default()
            },
            EventKind::ProsodyArousalHigh => Appraisal {
                goal_relevance: 0.4,
                threat_signal: 0.2,
                certainty: -0.1,
                ..Default::default()
            },
            EventKind::ProsodyValenceNegative => Appraisal {
                goal_congruence: -0.3,
                social_safety: -0.2,
                ..Default::default()
            },
            EventKind::ProsodyUncertain => Appraisal {
                certainty: -0.3,
                ..Default::default()
            },
            EventKind::Other => Appraisal::default(),
        };

        // Confidence-weight all dimensions so unreliable detectors can't
        // dominate. A confidence of 1.0 leaves the rule output untouched.
        a.scale(event.confidence.clamp(0.0, 1.0));
        a
    }
}
