//! Channel B prompt-summary renderer.
//!
//! Produces the structured state summary that Phase 2's
//! `PromptAugmentationRenderer` will eventually inject into the
//! system prompt. Lives in the simulator now so we can eyeball the
//! output during Phase 1 review.
//!
//! PROJECT_PLAN.md §12 Q4 (structured fields vs natural-language) is
//! still open. This implementation produces structured fields — they
//! are easier to validate deterministically.

use std::fmt::Write as _;

use crate::regulation::RegulationPolicy;
use crate::state::AffectState;

#[derive(Debug, Default)]
pub struct PromptRenderer;

impl PromptRenderer {
    pub fn new() -> Self {
        Self
    }

    pub fn render(&self, state: &AffectState, policy: &RegulationPolicy) -> String {
        let mut out = String::with_capacity(512);
        let _ = writeln!(out, "Current interaction state:");

        let val_label = bucket_signed(state.core.valence);
        let aro_label = bucket_unsigned(state.core.arousal);
        let _ = writeln!(
            out,
            "- Valence: {} ({:+.2})",
            val_label, state.core.valence
        );
        let _ = writeln!(
            out,
            "- Arousal: {} ({:.2})",
            aro_label, state.core.arousal
        );

        let recent: Vec<String> = state
            .event_ledger
            .iter()
            .rev()
            .take(5)
            .map(|e| format!("{:?}", e.kind).to_lowercase())
            .collect();
        if !recent.is_empty() {
            let _ = writeln!(out, "- Recent events: {}", recent.join(", "));
        }

        let ch = &state.channels;
        let _ = writeln!(
            out,
            "- Internal channels: anger {:.2}, sadness {:.2}, fear {:.2}, joy {:.2}, calm {:.2}, frustration {:.2}, curiosity {:.2}, empathy {:.2}",
            ch.anger, ch.sadness, ch.fear, ch.joy, ch.calm, ch.frustration, ch.curiosity, ch.empathy
        );

        let style = describe_policy(policy);
        let _ = writeln!(out, "- Expression policy: {}", style);

        out
    }
}

fn bucket_signed(x: f32) -> &'static str {
    match x {
        v if v >= 0.5 => "positive",
        v if v >= 0.15 => "slightly positive",
        v if v > -0.15 => "neutral",
        v if v > -0.5 => "slightly negative",
        _ => "negative",
    }
}

fn bucket_unsigned(x: f32) -> &'static str {
    match x {
        v if v >= 0.7 => "high",
        v if v >= 0.4 => "medium",
        v if v >= 0.15 => "low",
        _ => "minimal",
    }
}

fn describe_policy(p: &RegulationPolicy) -> String {
    let mut tags: Vec<&str> = Vec::new();
    if p.safety_dampening >= 0.6 {
        tags.push("careful, de-escalate");
    }
    if p.warmth >= 0.6 {
        tags.push("warm");
    }
    if p.task_focus >= 0.6 {
        tags.push("focused");
    }
    if p.assertiveness >= 0.6 {
        tags.push("direct");
    }
    if p.social_dampening >= 0.5 {
        tags.push("reserved");
    }
    if p.expressiveness < 0.3 {
        tags.push("muted");
    }
    if tags.is_empty() {
        "neutral".to_string()
    } else {
        tags.join(", ")
    }
}
