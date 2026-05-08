//! Perceived events — the simulator's only input.
//!
//! Phase 3 will add prosody-derived events (`ProsodyArousalHigh`,
//! `ProsodyValenceNegative`, `ProsodyUncertain`) sourced from the Whisper
//! encoder branch. Phase 1 stops at task / text / system events.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Audio,
    Text,
    Task,
    System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    // Task / system
    TaskSuccess,
    TaskFailure,
    ToolFailure,
    ToolSuccess,
    SystemError,
    SafetyTrigger,

    // Text / dialog
    UserPraise,
    UserCriticism,
    UserDistress,
    UserGreeting,
    UserQuestion,
    UserBlame,
    UserProvocation,
    AmbiguousThreat,

    // Observational
    NovelObservation,
    ConflictingSignals,

    // Prosody (Phase 3 — defined now so scenarios can stub them)
    ProsodyArousalHigh,
    ProsodyValenceNegative,
    ProsodyUncertain,

    // Catch-all for scenarios that need a custom kind without expanding
    // the enum mid-Phase-1. The string lives in `payload.tag`.
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceivedEvent {
    pub id: String,
    pub timestamp_ms: u64,
    pub modality: Modality,
    pub kind: EventKind,
    /// Detection confidence in [0, 1]. Prosody events arrive < 1.0; rule
    /// outputs scale by this so unreliable signals can't dominate.
    pub confidence: f32,
    #[serde(default)]
    pub payload: serde_json::Value,
}

impl PerceivedEvent {
    pub fn new(id: impl Into<String>, t_ms: u64, modality: Modality, kind: EventKind) -> Self {
        Self {
            id: id.into(),
            timestamp_ms: t_ms,
            modality,
            kind,
            confidence: 1.0,
            payload: serde_json::Value::Null,
        }
    }
}
