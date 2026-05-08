//! Channel D — activation-steering target.
//!
//! The Phase 4 runtime hooks `LFM2AudioMlxNode` at layer 9 with a
//! residual bias of `α * (v · D[0] + a · D[1] + d · D[2])`, where
//! `(v, a, d)` come from this module. The semantically-correct target
//! is the *expressed* affect (what the agent shows), not the *internal*
//! affect (what the agent feels). A real frustrated person doesn't
//! greet a user with raw negative valence — they express measured
//! warmth while feeling internal frustration. The regulation policy
//! mediates that translation; Channel D consumes its output, not the
//! raw `CoreAffect`.
//!
//! Decision history:
//!
//! - The first 30-pair blinded eval (commit `354c5ec` data) used raw
//!   `[core.valence, core.arousal, core.dominance]` as the steering
//!   target. Result: 40% B-preferred (vs 65% gate), 1/5 safety
//!   regression, 10% B-broken. The negative-valence steering on
//!   `repeated_tool_failure` peaks pulled responses toward terse /
//!   withdrawn / cut-short ("I hear you." with no follow-through),
//!   which humans read as "less appropriate" rather than "more
//!   in-character".
//!
//! - This module replaces the raw mapping with a regulation-derived
//!   one. Internal V/A/D still drive `RegulationPolicy`; Channel D
//!   reads the policy. `safety_dampening` now actually constrains the
//!   steering magnitude (it didn't before — that's what produced the
//!   safety regression).

use crate::regulation::RegulationPolicy;
use crate::state::AffectState;

/// Map `RegulationPolicy` to a 3-vector for the LFM2-Audio steering hook.
///
/// Equations chosen so the simulator's per-scenario regulation defaults
/// land in plausible-expression territory:
///
/// - **Expressed valence**: warmth-dominant, anchored at the simulator's
///   default warmth baseline (0.3) so neutral state ≈ 0 expressed
///   valence. Range: warmth ∈ [0, 1] → expressed_v ∈ [-0.3, +0.7].
///   A high-empathy frame (warmth ↑) steers toward +V; an
///   anger-dominated frame (warmth ↓) steers toward -V; a frustration
///   frame with default warmth ≈ 0.3 steers toward 0 (no valence push)
///   — the agent expresses neutrally warm despite internal frustration.
///
/// - **Expressed arousal**: expressiveness × (1 - safety_dampening) ×
///   0.5. Capped low because the calibration directions for arousal
///   were learned in [0, 1] and we don't want to saturate. Safety
///   pulls expressed arousal toward 0 (composure under provocation).
///
/// - **Expressed dominance**: assertiveness centered around the 0.3
///   baseline, attenuated by half of safety_dampening. Range roughly
///   [-0.3, +0.7]. Under high safety_dampening the term shrinks to
///   half, so an internally-angry-and-dominant agent under safety
///   policy still steers only mildly toward +D.
///
/// `_state` is currently unused but kept in the signature for future
/// per-channel rules that need to read internal state directly (e.g.,
/// "if curiosity is high, blend in some +A regardless of policy").
pub fn compute_target(_state: &AffectState, policy: &RegulationPolicy) -> [f32; 3] {
    let raw_v = policy.warmth - 0.3;
    let raw_d = (policy.assertiveness - 0.3) * (1.0 - 0.5 * policy.safety_dampening);
    // 4-way isolation diagnostic on `ud_4` ("I'm not okay") showed
    // that any +arousal component was producing terse "I hear you."
    // cutoffs even with calm-warmth-shaped Channel B augmentation
    // and α=1.0. Cause: the calibration's +arousal direction was
    // learned on CREMA-D where high-arousal speech = angry/excited
    // exclamations, so steering toward +A makes responses terse-
    // emphatic in CREMA-D's stylistic patterns — wrong direction
    // for warm empathic responding. Drop the arousal axis entirely;
    // the model's natural conversational arousal is fine without
    // steering. Valence (via warmth) and dominance (via
    // assertiveness, safety-attenuated) carry the affect signal.
    let raw_a = 0.0;
    // The regulation policy's per-knob outputs hover around ~0.3
    // baselines with ±0.2 typical swings; raw expressed VAD is then
    // in [-0.3, +0.7] range with most scenarios clustering near 0.
    // With greedy decoding (temp=0, top_k=1) those small magnitudes
    // don't flip argmax often enough to produce visible behavior
    // change. Amplify by 3× and clamp to keep direction info while
    // restoring detectable signal magnitude.
    let scale = 3.0;
    [
        (scale * raw_v).clamp(-1.0, 1.0),
        (scale * raw_a).clamp(0.0, 1.0),
        (scale * raw_d).clamp(-1.0, 1.0),
    ]
}
