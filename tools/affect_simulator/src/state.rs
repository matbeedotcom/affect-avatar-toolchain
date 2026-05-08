//! `AffectState` — the simulator's running state vector.
//!
//! `EmotionChannels` is the primary substrate; `CoreAffect` (V/A/D) is
//! derived heuristically and is what Phase 4 steering ultimately reads.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::appraisal::Appraisal;
use crate::event::{EventKind, Modality, PerceivedEvent};

/// V/A/D coordinates derived from per-channel intensities. Dominance is
/// signed; the persona-vector pipeline reads V and D from [-1, 1] and
/// arousal from [0, 1].
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct CoreAffect {
    pub valence: f32,
    pub arousal: f32,
    pub dominance: f32,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct EmotionChannels {
    pub anger: f32,
    pub sadness: f32,
    pub fear: f32,
    pub joy: f32,
    pub calm: f32,
    pub frustration: f32,
    pub curiosity: f32,
    pub empathy: f32,
}

impl EmotionChannels {
    /// Per-channel baseline — what `decay_rate` pulls toward when no
    /// event impact is applied.
    pub fn baselines() -> Self {
        Self {
            calm: 0.4,
            curiosity: 0.2,
            ..Default::default()
        }
    }

    pub fn as_array(&self) -> [(&'static str, f32); 8] {
        [
            ("anger", self.anger),
            ("sadness", self.sadness),
            ("fear", self.fear),
            ("joy", self.joy),
            ("calm", self.calm),
            ("frustration", self.frustration),
            ("curiosity", self.curiosity),
            ("empathy", self.empathy),
        ]
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut f32> {
        match name {
            "anger" => Some(&mut self.anger),
            "sadness" => Some(&mut self.sadness),
            "fear" => Some(&mut self.fear),
            "joy" => Some(&mut self.joy),
            "calm" => Some(&mut self.calm),
            "frustration" => Some(&mut self.frustration),
            "curiosity" => Some(&mut self.curiosity),
            "empathy" => Some(&mut self.empathy),
            _ => None,
        }
    }
}

/// Verbatim from PROJECT_PLAN.md — heuristic mapping from channels to V/A/D.
pub fn derive_core(channels: &EmotionChannels) -> CoreAffect {
    let valence = channels.joy + channels.calm + 0.3 * channels.curiosity
        - channels.anger
        - channels.sadness
        - channels.fear
        - channels.frustration;
    let arousal = channels.anger
        + channels.fear
        + 0.7 * channels.joy
        + channels.frustration
        + 0.5 * channels.curiosity
        - 0.5 * channels.calm;
    let dominance = 0.4 * channels.anger + 0.3 * channels.calm
        - 0.6 * channels.fear
        - 0.4 * channels.sadness;
    CoreAffect {
        valence: valence.clamp(-1.0, 1.0),
        arousal: arousal.clamp(0.0, 1.0),
        dominance: dominance.clamp(-1.0, 1.0),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventLogEntry {
    pub timestamp_ms: u64,
    pub event_id: String,
    pub kind: EventKind,
    pub modality: Modality,
    pub confidence: f32,
}

impl From<&PerceivedEvent> for EventLogEntry {
    fn from(e: &PerceivedEvent) -> Self {
        Self {
            timestamp_ms: e.timestamp_ms,
            event_id: e.id.clone(),
            kind: e.kind,
            modality: e.modality,
            confidence: e.confidence,
        }
    }
}

/// `recent_appraisals` ring keeps ~30 s of context for prompt rendering.
const RECENT_APPRAISAL_CAPACITY: usize = 32;
const EVENT_LEDGER_CAPACITY: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffectState {
    pub timestamp_ms: u64,
    pub core: CoreAffect,
    pub channels: EmotionChannels,
    pub recent_appraisals: VecDeque<Appraisal>,
    pub event_ledger: VecDeque<EventLogEntry>,
}

impl AffectState {
    pub fn initial() -> Self {
        let channels = EmotionChannels::baselines();
        Self {
            timestamp_ms: 0,
            core: derive_core(&channels),
            channels,
            recent_appraisals: VecDeque::with_capacity(RECENT_APPRAISAL_CAPACITY),
            event_ledger: VecDeque::with_capacity(EVENT_LEDGER_CAPACITY),
        }
    }

    pub fn record_appraisal(&mut self, a: Appraisal) {
        if self.recent_appraisals.len() == RECENT_APPRAISAL_CAPACITY {
            self.recent_appraisals.pop_front();
        }
        self.recent_appraisals.push_back(a);
    }

    pub fn record_event(&mut self, e: &PerceivedEvent) {
        if self.event_ledger.len() == EVENT_LEDGER_CAPACITY {
            self.event_ledger.pop_front();
        }
        self.event_ledger.push_back(EventLogEntry::from(e));
    }

    pub fn refresh_core(&mut self) {
        self.core = derive_core(&self.channels);
    }
}
